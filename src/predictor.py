"""
预测模型模块 - 使用多种方法进行价格预测
@author Reln Ding
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from config import PREDICTION_PERIODS


class TrendPredictor:
    """趋势预测器"""

    def __init__(self, df: pd.DataFrame):
        """
        初始化预测器

        Args:
            df: 包含技术指标的K线数据
        """
        self.df = df.copy()
        self.scaler = StandardScaler()

    def predict_linear_trend(self, periods: int = PREDICTION_PERIODS) -> Tuple[float, str]:
        """
        使用线性回归预测趋势

        Args:
            periods: 预测周期数

        Returns:
            (预测价格变化百分比, 趋势方向)
        """
        close_prices = self.df['close'].values[-50:]  # 使用最近50根K线
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        model = LinearRegression()
        model.fit(X, y)

        # 预测未来价格
        future_X = np.arange(len(close_prices), len(close_prices) + periods).reshape(-1, 1)
        predicted_prices = model.predict(future_X)

        current_price = close_prices[-1]
        predicted_price = predicted_prices[-1]
        change_percent = ((predicted_price - current_price) / current_price) * 100

        if change_percent > 0.5:
            direction = "上涨"
        elif change_percent < -0.5:
            direction = "下跌"
        else:
            direction = "横盘"

        return change_percent, direction

    def predict_with_indicators(self) -> Dict:
        """
        基于技术指标的综合预测

        Returns:
            包含各指标信号的字典
        """
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        signals = {}

        # RSI信号
        rsi = latest.get('RSI', 50)
        if rsi > 70:
            signals['RSI'] = {'signal': '超买', 'direction': '看跌', 'value': rsi}
        elif rsi < 30:
            signals['RSI'] = {'signal': '超卖', 'direction': '看涨', 'value': rsi}
        else:
            signals['RSI'] = {'signal': '中性', 'direction': '观望', 'value': rsi}

        # MACD信号
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_hist = latest.get('MACD_Hist', 0)
        prev_macd_hist = prev.get('MACD_Hist', 0)

        if macd > macd_signal and prev_macd_hist < 0 and macd_hist > 0:
            signals['MACD'] = {'signal': '金叉', 'direction': '看涨', 'value': macd_hist}
        elif macd < macd_signal and prev_macd_hist > 0 and macd_hist < 0:
            signals['MACD'] = {'signal': '死叉', 'direction': '看跌', 'value': macd_hist}
        elif macd > macd_signal:
            signals['MACD'] = {'signal': '多头', 'direction': '看涨', 'value': macd_hist}
        else:
            signals['MACD'] = {'signal': '空头', 'direction': '看跌', 'value': macd_hist}

        # 布林带信号
        close = latest['close']
        boll_upper = latest.get('BOLL_Upper', close)
        boll_lower = latest.get('BOLL_Lower', close)
        boll_middle = latest.get('BOLL_Middle', close)

        boll_position = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper != boll_lower else 0.5

        if close >= boll_upper:
            signals['BOLL'] = {'signal': '触及上轨', 'direction': '看跌', 'value': boll_position}
        elif close <= boll_lower:
            signals['BOLL'] = {'signal': '触及下轨', 'direction': '看涨', 'value': boll_position}
        elif close > boll_middle:
            signals['BOLL'] = {'signal': '上半区', 'direction': '偏多', 'value': boll_position}
        else:
            signals['BOLL'] = {'signal': '下半区', 'direction': '偏空', 'value': boll_position}

        # MA趋势信号
        ma7 = latest.get('MA7', close)
        ma25 = latest.get('MA25', close)
        ma99 = latest.get('MA99', close)

        if ma7 > ma25 > ma99:
            signals['MA'] = {'signal': '多头排列', 'direction': '看涨', 'value': 1}
        elif ma7 < ma25 < ma99:
            signals['MA'] = {'signal': '空头排列', 'direction': '看跌', 'value': -1}
        else:
            signals['MA'] = {'signal': '震荡', 'direction': '观望', 'value': 0}

        # 成交量信号
        volume = latest['volume']
        volume_ma = latest.get('Volume_MA', volume)
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1

        if volume_ratio > 1.5:
            signals['Volume'] = {'signal': '放量', 'direction': '关注', 'value': volume_ratio}
        elif volume_ratio < 0.5:
            signals['Volume'] = {'signal': '缩量', 'direction': '观望', 'value': volume_ratio}
        else:
            signals['Volume'] = {'signal': '正常', 'direction': '中性', 'value': volume_ratio}

        return signals

    def get_comprehensive_prediction(self) -> Dict:
        """
        综合预测结果（成交量加权版本）

        Returns:
            综合预测信息
        """
        # 线性趋势预测
        change_percent, linear_direction = self.predict_linear_trend()

        # 指标信号
        signals = self.predict_with_indicators()

        # 计算成交量权重
        volume_data = signals.get('Volume', {})
        volume_ratio = volume_data.get('value', 1.0)

        # 成交量加权系数：放量时信号更可信
        if volume_ratio > 2.0:
            volume_weight = 1.5  # 大幅放量，信号权重提升50%
        elif volume_ratio > 1.5:
            volume_weight = 1.3  # 明显放量，信号权重提升30%
        elif volume_ratio > 1.2:
            volume_weight = 1.1  # 轻微放量
        elif volume_ratio < 0.5:
            volume_weight = 0.7  # 缩量，信号可信度降低
        else:
            volume_weight = 1.0  # 正常成交量

        # 计算综合评分（成交量加权）
        score = 0.0
        bullish_count = 0
        bearish_count = 0

        for indicator, data in signals.items():
            if indicator == 'Volume':  # 成交量本身不参与评分
                continue

            direction = data['direction']
            if '涨' in direction or '多' in direction:
                bullish_count += 1
                score += 1 * volume_weight
            elif '跌' in direction or '空' in direction:
                bearish_count += 1
                score -= 1 * volume_weight

        # 综合判断（调整阈值以适应加权评分）
        if score >= 3.5:
            overall_direction = "强烈看涨"
            confidence = "高"
        elif score >= 2.0:
            overall_direction = "看涨"
            confidence = "中高"
        elif score >= 1.0:
            overall_direction = "偏多"
            confidence = "中"
        elif score <= -3.5:
            overall_direction = "强烈看跌"
            confidence = "高"
        elif score <= -2.0:
            overall_direction = "看跌"
            confidence = "中高"
        elif score <= -1.0:
            overall_direction = "偏空"
            confidence = "中"
        else:
            overall_direction = "震荡观望"
            confidence = "低"

        return {
            'linear_change_percent': change_percent,
            'linear_direction': linear_direction,
            'signals': signals,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'score': round(score, 2),
            'volume_weight': volume_weight,
            'overall_direction': overall_direction,
            'confidence': confidence
        }
