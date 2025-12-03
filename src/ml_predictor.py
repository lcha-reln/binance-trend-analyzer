"""
机器学习预测模块 - XGBoost 和 LSTM 预测模型
@author Reln Ding
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow 导入（可选，如果没有安装会跳过 LSTM）
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("警告: TensorFlow 未安装，LSTM 模型不可用")


class FeatureEngineer:
    """特征工程类"""

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        创建机器学习特征

        Args:
            df: 包含技术指标的K线数据

        Returns:
            包含特征的DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 价格变化特征
        features['price_change'] = df['close'].pct_change()
        features['price_change_2'] = df['close'].pct_change(2)
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)

        # 波动率特征
        features['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        features['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

        # 高低价特征
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_to_high'] = (df['high'] - df['close']) / df['close']
        features['close_to_low'] = (df['close'] - df['low']) / df['close']

        # 成交量特征
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df.get('Volume_MA', df['volume'].rolling(20).mean())

        # 技术指标特征
        if 'RSI' in df.columns:
            features['rsi'] = df['RSI']
            features['rsi_change'] = df['RSI'].diff()

        if 'MACD_Hist' in df.columns:
            features['macd_hist'] = df['MACD_Hist']
            features['macd_hist_change'] = df['MACD_Hist'].diff()

        if 'ADX' in df.columns:
            features['adx'] = df['ADX']
            features['plus_di'] = df.get('Plus_DI', 0)
            features['minus_di'] = df.get('Minus_DI', 0)
            features['di_diff'] = features['plus_di'] - features['minus_di']

        # MA 相对位置
        if 'MA7' in df.columns:
            features['price_ma7_ratio'] = df['close'] / df['MA7'] - 1
        if 'MA25' in df.columns:
            features['price_ma25_ratio'] = df['close'] / df['MA25'] - 1
        if 'MA99' in df.columns:
            features['price_ma99_ratio'] = df['close'] / df['MA99'] - 1

        # MA 排列特征
        if all(col in df.columns for col in ['MA7', 'MA25', 'MA99']):
            features['ma_bullish'] = ((df['MA7'] > df['MA25']) & (df['MA25'] > df['MA99'])).astype(int)
            features['ma_bearish'] = ((df['MA7'] < df['MA25']) & (df['MA25'] < df['MA99'])).astype(int)

        # 布林带特征
        if 'BOLL_Upper' in df.columns:
            boll_width = df['BOLL_Upper'] - df['BOLL_Lower']
            features['boll_position'] = (df['close'] - df['BOLL_Lower']) / boll_width
            features['boll_width'] = boll_width / df['BOLL_Middle']

        # EMA 特征
        if 'EMA12' in df.columns:
            features['ema12_26_diff'] = (df['EMA12'] - df['EMA26']) / df['close']
        if 'EMA50' in df.columns:
            features['price_ema50_ratio'] = df['close'] / df['EMA50'] - 1

        # 动量特征
        if 'Momentum' in df.columns:
            features['momentum'] = df['Momentum'] / df['close']
        if 'ROC' in df.columns:
            features['roc'] = df['ROC']

        # OBV 特征
        if 'OBV' in df.columns and 'OBV_MA' in df.columns:
            features['obv_ma_ratio'] = df['OBV'] / df['OBV_MA'].replace(0, np.nan)

        # Stochastic RSI 特征
        if 'StochRSI_K' in df.columns:
            features['stoch_rsi_k'] = df['StochRSI_K']
            features['stoch_rsi_d'] = df.get('StochRSI_D', df['StochRSI_K'])
            features['stoch_rsi_diff'] = features['stoch_rsi_k'] - features['stoch_rsi_d']

        # K线形态特征
        if 'pattern_type' in df.columns:
            features['pattern_bullish'] = (df['pattern_type'] == 'bullish').astype(int)
            features['pattern_bearish'] = (df['pattern_type'] == 'bearish').astype(int)

        if 'pattern_strength' in df.columns:
            features['pattern_strength'] = df['pattern_strength']

        return features

    @staticmethod
    def create_labels(df: pd.DataFrame, periods: int = 5, threshold: float = 0.3) -> pd.Series:
        """
        创建标签（未来价格变化方向）

        Args:
            df: K线数据
            periods: 预测的K线周期数
            threshold: 涨跌判断阈值（百分比）

        Returns:
            标签序列 (1: 上涨, 0: 横盘, -1: 下跌)
        """
        future_return = df['close'].shift(-periods) / df['close'] - 1
        future_return = future_return * 100  # 转为百分比

        labels = pd.Series(0, index=df.index)
        labels[future_return > threshold] = 1
        labels[future_return < -threshold] = -1

        return labels


class XGBoostPredictor:
    """XGBoost 预测模型"""

    def __init__(self, model_path: str = None):
        """
        初始化 XGBoost 预测器

        Args:
            model_path: 模型保存/加载路径
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_path = model_path or 'models/xgboost_model.pkl'

    def prepare_data(self, df: pd.DataFrame, prediction_periods: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据

        Args:
            df: 包含技术指标的K线数据
            prediction_periods: 预测周期数

        Returns:
            (特征矩阵, 标签, 特征列名)
        """
        # 创建特征
        features = FeatureEngineer.create_features(df)

        # 创建标签
        labels = FeatureEngineer.create_labels(df, periods=prediction_periods)
        # XGBoost 需要标签从 0 开始: -1,0,1 -> 0,1,2
        labels = labels + 1

        # 合并并删除空值
        data = features.copy()
        data['label'] = labels

        # 删除未来数据泄露的行
        data = data.iloc[:-prediction_periods]
        data = data.dropna()

        X = data.drop('label', axis=1)
        y = data['label']

        self.feature_columns = X.columns.tolist()

        return X.values, y.values, self.feature_columns

    def train(self, df: pd.DataFrame, prediction_periods: int = 5, test_size: float = 0.2) -> Dict:
        """
        训练模型

        Args:
            df: 包含技术指标的K线数据
            prediction_periods: 预测周期数
            test_size: 测试集比例

        Returns:
            训练结果统计
        """
        X, y, feature_cols = self.prepare_data(df, prediction_periods)

        # 分割数据集（保持时间顺序）
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练 XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        # 评估
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # 特征重要性
        feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'top_features': top_features
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        预测

        Args:
            df: 包含技术指标的K线数据

        Returns:
            预测结果
        """
        if self.model is None:
            return {'error': '模型未训练'}

        # 创建特征
        features = FeatureEngineer.create_features(df)
        features = features.dropna()

        if len(features) == 0:
            return {'error': '特征数据不足'}

        # 确保特征列一致
        missing_cols = set(self.feature_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0

        features = features[self.feature_columns]

        # 取最新一行
        X = features.iloc[-1:].values
        X_scaled = self.scaler.transform(X)

        # 预测
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # 转换回原始标签: 0,1,2 -> -1,0,1
        prediction = int(prediction) - 1

        # 类别映射 (模型输出 0,1,2 对应 下跌,横盘,上涨)
        prob_dict = {-1: float(probabilities[0]), 0: float(probabilities[1]), 1: float(probabilities[2])}

        direction_map = {1: '上涨', 0: '横盘', -1: '下跌'}
        direction = direction_map.get(prediction, '未知')

        # 置信度
        confidence = max(probabilities)
        if confidence > 0.6:
            confidence_level = '高'
        elif confidence > 0.4:
            confidence_level = '中'
        else:
            confidence_level = '低'

        return {
            'direction': direction,
            'prediction': int(prediction),
            'probabilities': prob_dict,
            'confidence': confidence,
            'confidence_level': confidence_level
        }

    def save(self, path: str = None):
        """保存模型"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)

    def load(self, path: str = None) -> bool:
        """加载模型"""
        path = path or self.model_path
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']

        return True


class LSTMPredictor:
    """LSTM 预测模型"""

    def __init__(self, model_path: str = None, sequence_length: int = 20):
        """
        初始化 LSTM 预测器

        Args:
            model_path: 模型保存/加载路径
            sequence_length: 输入序列长度
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow 未安装，无法使用 LSTM 模型")

        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.feature_columns = None
        self.model_path = model_path or 'models/lstm_model.h5'

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建 LSTM 输入序列

        Args:
            X: 特征矩阵
            y: 标签

        Returns:
            (序列化特征, 对应标签)
        """
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape: Tuple[int, int], num_classes: int = 3):
        """
        构建 LSTM 模型

        Args:
            input_shape: 输入形状 (sequence_length, features)
            num_classes: 分类数量
        """
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, df: pd.DataFrame, prediction_periods: int = 5,
              test_size: float = 0.2, epochs: int = 50) -> Dict:
        """
        训练模型

        Args:
            df: 包含技术指标的K线数据
            prediction_periods: 预测周期数
            test_size: 测试集比例
            epochs: 训练轮数

        Returns:
            训练结果统计
        """
        # 创建特征
        features = FeatureEngineer.create_features(df)

        # 创建标签 (转换为 0, 1, 2)
        labels = FeatureEngineer.create_labels(df, periods=prediction_periods)
        labels = labels + 1  # -1,0,1 -> 0,1,2

        # 合并并删除空值
        data = features.copy()
        data['label'] = labels
        data = data.iloc[:-prediction_periods]
        data = data.dropna()

        X = data.drop('label', axis=1).values
        y = data['label'].values

        self.feature_columns = features.columns.tolist()

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 创建序列
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # 分割数据
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # 构建模型
        self.build_model(input_shape=(self.sequence_length, X_scaled.shape[1]))

        # 训练
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 评估
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)

        return {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss'])
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        预测

        Args:
            df: 包含技术指标的K线数据

        Returns:
            预测结果
        """
        if self.model is None:
            return {'error': '模型未训练'}

        # 创建特征
        features = FeatureEngineer.create_features(df)
        features = features.dropna()

        if len(features) < self.sequence_length:
            return {'error': f'数据不足，需要至少 {self.sequence_length} 条记录'}

        # 确保特征列一致
        missing_cols = set(self.feature_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0

        # 标准化
        X = features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # 取最新序列
        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)

        # 预测
        probabilities = self.model.predict(X_seq, verbose=0)[0]
        prediction = np.argmax(probabilities) - 1  # 0,1,2 -> -1,0,1

        direction_map = {1: '上涨', 0: '横盘', -1: '下跌'}
        direction = direction_map.get(prediction, '未知')

        # 置信度
        confidence = max(probabilities)
        if confidence > 0.6:
            confidence_level = '高'
        elif confidence > 0.4:
            confidence_level = '中'
        else:
            confidence_level = '低'

        return {
            'direction': direction,
            'prediction': int(prediction),
            'probabilities': {-1: float(probabilities[0]), 0: float(probabilities[1]), 1: float(probabilities[2])},
            'confidence': float(confidence),
            'confidence_level': confidence_level
        }

    def save(self, path: str = None):
        """保存模型"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)

        # 保存 scaler 和特征列
        with open(path.replace('.h5', '_meta.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length
            }, f)

    def load(self, path: str = None) -> bool:
        """加载模型"""
        path = path or self.model_path
        if not os.path.exists(path):
            return False

        self.model = load_model(path)

        meta_path = path.replace('.h5', '_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.scaler = meta['scaler']
                self.feature_columns = meta['feature_columns']
                self.sequence_length = meta['sequence_length']

        return True


class MLPredictor:
    """统一的机器学习预测接口"""

    def __init__(self, model_type: str = 'xgboost'):
        """
        初始化预测器

        Args:
            model_type: 模型类型 ('xgboost' 或 'lstm')
        """
        self.model_type = model_type
        if model_type == 'xgboost':
            self.predictor = XGBoostPredictor()
        elif model_type == 'lstm':
            if not HAS_TENSORFLOW:
                print("警告: TensorFlow 未安装，回退到 XGBoost")
                self.model_type = 'xgboost'
                self.predictor = XGBoostPredictor()
            else:
                self.predictor = LSTMPredictor()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def train(self, df: pd.DataFrame, **kwargs) -> Dict:
        """训练模型"""
        return self.predictor.train(df, **kwargs)

    def predict(self, df: pd.DataFrame) -> Dict:
        """预测"""
        return self.predictor.predict(df)

    def save(self, path: str = None):
        """保存模型"""
        self.predictor.save(path)

    def load(self, path: str = None) -> bool:
        """加载模型"""
        return self.predictor.load(path)


def train_and_evaluate(symbol: str = 'BTCUSDT', interval: str = '1h'):
    """
    训练并评估模型

    Args:
        symbol: 交易对
        interval: K线周期
    """
    from collector import get_klines
    from indicators import calculate_all_indicators

    print(f"正在获取 {symbol} {interval} 数据...")
    df = get_klines(symbol, interval, limit=1000)
    if df is None:
        print("获取数据失败")
        return

    print("计算技术指标...")
    df = calculate_all_indicators(df)

    # 训练 XGBoost
    print("\n=== 训练 XGBoost 模型 ===")
    xgb_predictor = MLPredictor('xgboost')
    xgb_result = xgb_predictor.train(df)
    print(f"准确率: {xgb_result['accuracy']:.2%}")
    print(f"训练样本: {xgb_result['train_samples']}")
    print(f"测试样本: {xgb_result['test_samples']}")
    print("特征重要性 Top 10:")
    for feat, imp in xgb_result['top_features']:
        print(f"  {feat}: {imp:.4f}")

    # 预测
    print("\n=== 当前预测 ===")
    prediction = xgb_predictor.predict(df)
    print(f"预测方向: {prediction['direction']}")
    print(f"置信度: {prediction['confidence']:.2%} ({prediction['confidence_level']})")
    print(f"概率分布: 下跌={prediction['probabilities'].get(-1, 0):.2%}, "
          f"横盘={prediction['probabilities'].get(0, 0):.2%}, "
          f"上涨={prediction['probabilities'].get(1, 0):.2%}")

    # 保存模型
    xgb_predictor.save()
    print("\n模型已保存")

    # 训练 LSTM（如果可用）
    if HAS_TENSORFLOW:
        print("\n=== 训练 LSTM 模型 ===")
        lstm_predictor = MLPredictor('lstm')
        lstm_result = lstm_predictor.train(df, epochs=30)
        print(f"准确率: {lstm_result['accuracy']:.2%}")
        print(f"训练轮数: {lstm_result['epochs_trained']}")

        prediction = lstm_predictor.predict(df)
        print(f"\nLSTM 预测方向: {prediction['direction']}")
        print(f"置信度: {prediction['confidence']:.2%} ({prediction['confidence_level']})")

        lstm_predictor.save()


if __name__ == '__main__':
    train_and_evaluate('BTCUSDT', '1h')
