"""
反馈学习模块 - SQLite存储 + 在线学习 + 增量训练
@author Reln Ding
"""

import sqlite3
import json
import time
import threading
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# 数据库路径
DATA_DIR = Path(__file__).parent.parent / 'data'
DB_PATH = DATA_DIR / 'predictions.db'

# 训练配置
MIN_SAMPLES_FOR_TRAINING = 50      # 最少需要多少样本才能训练
RETRAIN_THRESHOLD = 20             # 每新增多少已验证样本触发重训练
MODEL_DIR = Path(__file__).parent.parent / 'models'


class FeedbackDatabase:
    """反馈学习数据库管理"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        """确保目录存在"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 预测记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    direction_normalized TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    score REAL NOT NULL,
                    price_at_prediction REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    verified INTEGER DEFAULT 0,
                    actual_direction TEXT,
                    price_at_verify REAL,
                    price_change_percent REAL,
                    is_correct INTEGER,
                    verify_datetime TEXT,
                    features_json TEXT,
                    used_for_training INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_interval ON predictions(symbol, interval)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_verified ON predictions(verified)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_used_for_training ON predictions(used_for_training)')

            # 模型版本表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    accuracy REAL,
                    train_samples INTEGER,
                    test_samples INTEGER,
                    model_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    metadata_json TEXT
                )
            ''')

            # 训练日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    interval TEXT,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def insert_prediction(self, prediction: Dict, features: Dict = None):
        """
        插入预测记录

        Args:
            prediction: 预测数据
            features: 特征快照（技术指标等）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 检查是否已存在相似预测（同一symbol+interval，时间间隔小于验证周期的80%）
            verify_periods = {'5m': 5, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
            interval = prediction.get('interval', '1h')
            min_gap = verify_periods.get(interval, 60) * 60 * 0.8

            cursor.execute('''
                SELECT id FROM predictions
                WHERE symbol = ? AND interval = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (prediction['symbol'], interval, time.time() - min_gap))

            if cursor.fetchone():
                return  # 已有近期预测

            features_json = json.dumps(features) if features else None

            cursor.execute('''
                INSERT OR IGNORE INTO predictions
                (id, symbol, interval, direction, direction_normalized, confidence,
                 score, price_at_prediction, timestamp, datetime, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction['id'],
                prediction['symbol'],
                prediction['interval'],
                prediction['direction'],
                prediction['direction_normalized'],
                prediction['confidence'],
                prediction['score'],
                prediction['price_at_prediction'],
                prediction['timestamp'],
                prediction['datetime'],
                features_json
            ))

    def update_verification(self, pred_id: str, actual_direction: str,
                           price_at_verify: float, price_change_percent: float,
                           is_correct: bool):
        """更新验证结果"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions
                SET verified = 1, actual_direction = ?, price_at_verify = ?,
                    price_change_percent = ?, is_correct = ?,
                    verify_datetime = ?
                WHERE id = ?
            ''', (
                actual_direction, price_at_verify, price_change_percent,
                1 if is_correct else 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                pred_id
            ))

    def get_unverified_predictions(self) -> List[Dict]:
        """获取待验证的预测"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions WHERE verified = 0
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_training_samples(self, symbol: str = None, interval: str = None,
                            only_unused: bool = False) -> List[Dict]:
        """
        获取训练样本

        Args:
            symbol: 过滤交易对
            interval: 过滤周期
            only_unused: 仅获取未用于训练的样本
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = 'SELECT * FROM predictions WHERE verified = 1 AND features_json IS NOT NULL'
            params = []

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            if interval:
                query += ' AND interval = ?'
                params.append(interval)
            if only_unused:
                query += ' AND used_for_training = 0'

            query += ' ORDER BY timestamp ASC'

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def mark_samples_as_used(self, sample_ids: List[str]):
        """标记样本已用于训练"""
        if not sample_ids:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in sample_ids])
            cursor.execute(f'''
                UPDATE predictions SET used_for_training = 1
                WHERE id IN ({placeholders})
            ''', sample_ids)

    def get_accuracy_stats(self) -> Dict:
        """获取准确率统计"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 总体统计
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified_count,
                    SUM(CASE WHEN verified = 1 AND is_correct = 1 THEN 1 ELSE 0 END) as correct_count
                FROM predictions
            ''')
            total_stats = dict(cursor.fetchone())

            # 按周期统计
            cursor.execute('''
                SELECT interval,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions WHERE verified = 1
                GROUP BY interval
            ''')
            by_interval = {row['interval']: {
                'total': row['total'],
                'correct': row['correct'],
                'accuracy': round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0
            } for row in cursor.fetchall()}

            # 按置信度统计
            cursor.execute('''
                SELECT confidence,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions WHERE verified = 1
                GROUP BY confidence
            ''')
            by_confidence = {row['confidence']: {
                'total': row['total'],
                'correct': row['correct'],
                'accuracy': round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0
            } for row in cursor.fetchall()}

            # 按交易对统计
            cursor.execute('''
                SELECT symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions WHERE verified = 1
                GROUP BY symbol
            ''')
            by_symbol = {row['symbol']: {
                'total': row['total'],
                'correct': row['correct'],
                'accuracy': round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0
            } for row in cursor.fetchall()}

            # 最近预测
            cursor.execute('''
                SELECT symbol, interval, direction, confidence, datetime,
                       price_at_prediction, price_at_verify, price_change_percent, is_correct
                FROM predictions WHERE verified = 1
                ORDER BY timestamp DESC LIMIT 20
            ''')
            recent = [dict(row) for row in cursor.fetchall()]

            verified = total_stats['verified_count'] or 0
            correct = total_stats['correct_count'] or 0

            return {
                'total_predictions': total_stats['total'],
                'verified_count': verified,
                'pending_count': total_stats['total'] - verified,
                'correct_count': correct,
                'overall_accuracy': round(correct / verified * 100, 1) if verified > 0 else None,
                'by_interval': by_interval,
                'by_confidence': by_confidence,
                'by_symbol': by_symbol,
                'recent_predictions': recent
            }

    def get_new_sample_count(self) -> int:
        """获取新的未训练样本数"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as count FROM predictions
                WHERE verified = 1 AND used_for_training = 0 AND features_json IS NOT NULL
            ''')
            return cursor.fetchone()['count']

    def save_model_version(self, symbol: str, interval: str, version: int,
                          accuracy: float, train_samples: int, test_samples: int,
                          model_path: str, metadata: Dict = None):
        """保存模型版本记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 将旧版本标记为非活跃
            cursor.execute('''
                UPDATE model_versions SET is_active = 0
                WHERE symbol = ? AND interval = ?
            ''', (symbol, interval))

            # 插入新版本
            cursor.execute('''
                INSERT INTO model_versions
                (symbol, interval, version, accuracy, train_samples, test_samples,
                 model_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, interval, version, accuracy, train_samples, test_samples,
                model_path, json.dumps(metadata) if metadata else None
            ))

    def log_training_event(self, symbol: str, interval: str, event_type: str, message: str):
        """记录训练日志"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_logs (symbol, interval, event_type, message)
                VALUES (?, ?, ?, ?)
            ''', (symbol, interval, event_type, message))

    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cutoff = time.time() - (days * 24 * 60 * 60)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE timestamp < ?', (cutoff,))
            cursor.execute('DELETE FROM training_logs WHERE created_at < datetime("now", "-30 days")')


class OnlineLearner:
    """在线学习器 - 支持增量训练"""

    def __init__(self, db: FeedbackDatabase = None):
        self.db = db or FeedbackDatabase()
        self._lock = threading.RLock()
        self._models: Dict[str, xgb.XGBClassifier] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._feature_columns: Dict[str, List[str]] = {}

    def _get_model_key(self, symbol: str, interval: str) -> str:
        return f"{symbol}_{interval}"

    def _get_model_path(self, symbol: str, interval: str, version: int) -> str:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return str(MODEL_DIR / f"feedback_{symbol}_{interval}_v{version}.pkl")

    def should_retrain(self, symbol: str = None, interval: str = None) -> bool:
        """检查是否应该重新训练"""
        new_samples = self.db.get_new_sample_count()
        return new_samples >= RETRAIN_THRESHOLD

    def prepare_training_data(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据

        Args:
            samples: 数据库中的样本记录

        Returns:
            (特征矩阵, 标签, 特征列名)
        """
        X_list = []
        y_list = []
        feature_cols = None

        for sample in samples:
            if not sample.get('features_json'):
                continue

            features = json.loads(sample['features_json'])
            if not features:
                continue

            if feature_cols is None:
                feature_cols = sorted(features.keys())

            # 构建特征向量
            X_row = [features.get(col, 0) for col in feature_cols]
            X_list.append(X_row)

            # 标签：actual_direction -> 0(bearish), 1(neutral), 2(bullish)
            direction_map = {'bearish': 0, 'neutral': 1, 'bullish': 2}
            y_list.append(direction_map.get(sample['actual_direction'], 1))

        if not X_list:
            return None, None, None

        return np.array(X_list), np.array(y_list), feature_cols

    def train(self, symbol: str = None, interval: str = None,
              force: bool = False) -> Optional[Dict]:
        """
        训练模型

        Args:
            symbol: 交易对（None表示全部）
            interval: 周期（None表示全部）
            force: 是否强制训练

        Returns:
            训练结果
        """
        with self._lock:
            # 获取训练样本
            samples = self.db.get_training_samples(symbol, interval)

            if len(samples) < MIN_SAMPLES_FOR_TRAINING:
                self.db.log_training_event(
                    symbol or 'ALL', interval or 'ALL', 'SKIP',
                    f'样本不足: {len(samples)} < {MIN_SAMPLES_FOR_TRAINING}'
                )
                return None

            # 准备数据
            X, y, feature_cols = self.prepare_training_data(samples)
            if X is None or len(X) < MIN_SAMPLES_FOR_TRAINING:
                return None

            # 分割数据（保持时间顺序）
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练 XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )

            # 评估
            y_pred = model.predict(X_test_scaled)
            accuracy = (y_pred == y_test).mean()

            # 保存模型
            model_key = self._get_model_key(symbol or 'ALL', interval or 'ALL')

            # 获取版本号
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(version) as max_v FROM model_versions
                    WHERE symbol = ? AND interval = ?
                ''', (symbol or 'ALL', interval or 'ALL'))
                result = cursor.fetchone()
                version = (result['max_v'] or 0) + 1

            model_path = self._get_model_path(symbol or 'ALL', interval or 'ALL', version)

            # 保存到文件
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_columns': feature_cols
                }, f)

            # 更新内存中的模型
            self._models[model_key] = model
            self._scalers[model_key] = scaler
            self._feature_columns[model_key] = feature_cols

            # 标记样本已使用
            sample_ids = [s['id'] for s in samples]
            self.db.mark_samples_as_used(sample_ids)

            # 保存模型版本记录
            self.db.save_model_version(
                symbol or 'ALL', interval or 'ALL', version,
                accuracy, len(X_train), len(X_test), model_path,
                {'feature_columns': feature_cols}
            )

            # 记录日志
            self.db.log_training_event(
                symbol or 'ALL', interval or 'ALL', 'TRAIN',
                f'训练完成: accuracy={accuracy:.2%}, samples={len(samples)}, version={version}'
            )

            return {
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'version': version,
                'model_path': model_path
            }

    def predict(self, symbol: str, interval: str, features: Dict) -> Optional[Dict]:
        """
        使用反馈学习模型预测

        Args:
            symbol: 交易对
            interval: 周期
            features: 当前特征

        Returns:
            预测结果
        """
        model_key = self._get_model_key(symbol, interval)

        # 尝试加载模型
        if model_key not in self._models:
            # 先尝试加载通用模型
            general_key = self._get_model_key('ALL', 'ALL')
            if general_key in self._models:
                model_key = general_key
            else:
                # 尝试从文件加载
                if not self._load_latest_model(symbol, interval):
                    if not self._load_latest_model('ALL', 'ALL'):
                        return None
                    model_key = general_key

        model = self._models.get(model_key)
        scaler = self._scalers.get(model_key)
        feature_cols = self._feature_columns.get(model_key)

        if not all([model, scaler, feature_cols]):
            return None

        # 构建特征向量
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = scaler.transform(X)

        # 预测
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        direction_map = {0: '下跌', 1: '横盘', 2: '上涨'}
        direction = direction_map.get(prediction, '未知')

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
            'probabilities': {
                'bearish': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'bullish': float(probabilities[2])
            },
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'model_key': model_key
        }

    def _load_latest_model(self, symbol: str, interval: str) -> bool:
        """加载最新模型"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_path, metadata_json FROM model_versions
                WHERE symbol = ? AND interval = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
            ''', (symbol, interval))
            result = cursor.fetchone()

            if not result or not result['model_path']:
                return False

            model_path = result['model_path']
            if not Path(model_path).exists():
                return False

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            model_key = self._get_model_key(symbol, interval)
            self._models[model_key] = data['model']
            self._scalers[model_key] = data['scaler']
            self._feature_columns[model_key] = data['feature_columns']

            return True


# 全局实例
_db_instance: Optional[FeedbackDatabase] = None
_learner_instance: Optional[OnlineLearner] = None


def get_feedback_db() -> FeedbackDatabase:
    """获取数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = FeedbackDatabase()
    return _db_instance


def get_online_learner() -> OnlineLearner:
    """获取学习器实例"""
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = OnlineLearner()
    return _learner_instance
