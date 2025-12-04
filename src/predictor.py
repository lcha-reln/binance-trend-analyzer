"""
预测模型模块 - 使用多种方法进行价格预测（深度优化版）
@author Reln Ding
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from config import PREDICTION_PERIODS


# 指标权重配置（权重越高，对最终评分影响越大）
INDICATOR_WEIGHTS = {
    'ADX_TREND': 2.0,      # ADX趋势强度 - 最重要
    'MA_ALIGNMENT': 1.8,   # MA排列
    'MACD': 1.5,           # MACD
    'EMA_TREND': 1.3,      # EMA趋势
    'RSI': 1.0,            # RSI
    'STOCH_RSI': 1.0,      # Stochastic RSI
    'BOLL': 0.8,           # 布林带
    'MOMENTUM': 1.2,       # 动量
    'OBV': 1.0,            # OBV量价
}


class TrendPredictor:
    """趋势预测器（深度优化版）"""

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
        使用线性回归预测趋势（优化版：使用加权最近数据）

        Args:
            periods: 预测周期数

        Returns:
            (预测价格变化百分比, 趋势方向)
        """
        close_prices = self.df['close'].values[-50:]
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        # 使用加权，最近的数据权重更高
        weights = np.exp(np.linspace(0, 1, len(close_prices)))

        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)

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

    def analyze_trend_strength(self) -> Dict:
        """
        使用ADX分析趋势强度

        Returns:
            趋势强度信息
        """
        latest = self.df.iloc[-1]
        adx = latest.get('ADX', 0)
        plus_di = latest.get('Plus_DI', 0)
        minus_di = latest.get('Minus_DI', 0)

        # ADX判断趋势强度
        if adx >= 50:
            strength = "极强趋势"
            strength_score = 2.0
        elif adx >= 25:
            strength = "强趋势"
            strength_score = 1.5
        elif adx >= 20:
            strength = "趋势形成"
            strength_score = 1.0
        else:
            strength = "震荡/无趋势"
            strength_score = 0.5

        # DI判断方向
        if plus_di > minus_di:
            di_direction = "多头"
            di_score = 1
        else:
            di_direction = "空头"
            di_score = -1

        return {
            'adx': adx,
            'strength': strength,
            'strength_score': strength_score,
            'di_direction': di_direction,
            'di_score': di_score,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    def analyze_momentum(self) -> Dict:
        """
        分析动量指标

        Returns:
            动量分析结果
        """
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        momentum = latest.get('Momentum', 0)
        roc = latest.get('ROC', 0)
        prev_momentum = prev.get('Momentum', 0)

        # 动量方向
        if momentum > 0 and momentum > prev_momentum:
            signal = "加速上涨"
            direction = "看涨"
            score = 1.5
        elif momentum > 0:
            signal = "上涨动能减弱"
            direction = "偏多"
            score = 0.5
        elif momentum < 0 and momentum < prev_momentum:
            signal = "加速下跌"
            direction = "看跌"
            score = -1.5
        elif momentum < 0:
            signal = "下跌动能减弱"
            direction = "偏空"
            score = -0.5
        else:
            signal = "动能平稳"
            direction = "观望"
            score = 0

        return {
            'momentum': momentum,
            'roc': roc,
            'signal': signal,
            'direction': direction,
            'score': score
        }

    def analyze_obv_divergence(self) -> Dict:
        """
        分析OBV量价背离

        Returns:
            OBV背离分析结果
        """
        # 取最近20根K线分析
        recent = self.df.tail(20)

        price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
        obv_trend = recent['OBV'].iloc[-1] - recent['OBV'].iloc[0]

        latest = self.df.iloc[-1]
        obv = latest.get('OBV', 0)
        obv_ma = latest.get('OBV_MA', 0)

        # 判断背离
        if price_trend > 0 and obv_trend < 0:
            signal = "顶背离"
            direction = "看跌"
            score = -1.5
        elif price_trend < 0 and obv_trend > 0:
            signal = "底背离"
            direction = "看涨"
            score = 1.5
        elif obv > obv_ma:
            signal = "量能积累"
            direction = "偏多"
            score = 0.5
        elif obv < obv_ma:
            signal = "量能衰减"
            direction = "偏空"
            score = -0.5
        else:
            signal = "量价同步"
            direction = "中性"
            score = 0

        return {
            'obv': obv,
            'obv_ma': obv_ma,
            'signal': signal,
            'direction': direction,
            'score': score
        }

    def analyze_trend_confirmation(self) -> Dict:
        """
        趋势确认分析 - 检查连续K线是否同向

        Returns:
            趋势确认结果
        """
        # 检查最近5根K线
        recent = self.df.tail(5)

        up_count = 0
        down_count = 0

        for i in range(len(recent)):
            if recent['close'].iloc[i] > recent['open'].iloc[i]:
                up_count += 1
            else:
                down_count += 1

        if up_count >= 4:
            signal = "强势上涨确认"
            direction = "看涨"
            confirmation_score = 1.5
        elif up_count >= 3:
            signal = "上涨趋势"
            direction = "偏多"
            confirmation_score = 0.8
        elif down_count >= 4:
            signal = "强势下跌确认"
            direction = "看跌"
            confirmation_score = -1.5
        elif down_count >= 3:
            signal = "下跌趋势"
            direction = "偏空"
            confirmation_score = -0.8
        else:
            signal = "趋势不明"
            direction = "观望"
            confirmation_score = 0

        return {
            'up_count': up_count,
            'down_count': down_count,
            'signal': signal,
            'direction': direction,
            'score': confirmation_score
        }

    def predict_with_indicators(self) -> Dict:
        """
        基于技术指标的综合预测（深度优化版）

        Returns:
            包含各指标信号的字典
        """
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        signals = {}

        # 1. RSI信号
        rsi = latest.get('RSI', 50)
        if rsi > 80:
            signals['RSI'] = {'signal': '严重超买', 'direction': '看跌', 'value': rsi, 'score': -1.5}
        elif rsi > 70:
            signals['RSI'] = {'signal': '超买', 'direction': '看跌', 'value': rsi, 'score': -1.0}
        elif rsi < 20:
            signals['RSI'] = {'signal': '严重超卖', 'direction': '看涨', 'value': rsi, 'score': 1.5}
        elif rsi < 30:
            signals['RSI'] = {'signal': '超卖', 'direction': '看涨', 'value': rsi, 'score': 1.0}
        elif rsi > 50:
            signals['RSI'] = {'signal': '偏强', 'direction': '偏多', 'value': rsi, 'score': 0.3}
        else:
            signals['RSI'] = {'signal': '偏弱', 'direction': '偏空', 'value': rsi, 'score': -0.3}

        # 2. Stochastic RSI信号（更敏感）
        stoch_k = latest.get('StochRSI_K', 50)
        stoch_d = latest.get('StochRSI_D', 50)
        prev_stoch_k = prev.get('StochRSI_K', 50)

        if stoch_k > 80 and stoch_k < prev_stoch_k:
            signals['StochRSI'] = {'signal': '超买回落', 'direction': '看跌', 'value': stoch_k, 'score': -1.2}
        elif stoch_k < 20 and stoch_k > prev_stoch_k:
            signals['StochRSI'] = {'signal': '超卖反弹', 'direction': '看涨', 'value': stoch_k, 'score': 1.2}
        elif stoch_k > stoch_d and prev_stoch_k <= prev.get('StochRSI_D', 50):
            signals['StochRSI'] = {'signal': '金叉', 'direction': '看涨', 'value': stoch_k, 'score': 1.0}
        elif stoch_k < stoch_d and prev_stoch_k >= prev.get('StochRSI_D', 50):
            signals['StochRSI'] = {'signal': '死叉', 'direction': '看跌', 'value': stoch_k, 'score': -1.0}
        else:
            signals['StochRSI'] = {'signal': '中性', 'direction': '观望', 'value': stoch_k, 'score': 0}

        # 3. MACD信号
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_hist = latest.get('MACD_Hist', 0)
        prev_macd_hist = prev.get('MACD_Hist', 0)

        if macd > macd_signal and prev_macd_hist < 0 and macd_hist > 0:
            signals['MACD'] = {'signal': '金叉', 'direction': '看涨', 'value': macd_hist, 'score': 1.5}
        elif macd < macd_signal and prev_macd_hist > 0 and macd_hist < 0:
            signals['MACD'] = {'signal': '死叉', 'direction': '看跌', 'value': macd_hist, 'score': -1.5}
        elif macd_hist > 0 and macd_hist > prev_macd_hist:
            signals['MACD'] = {'signal': '多头加强', 'direction': '看涨', 'value': macd_hist, 'score': 1.0}
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            signals['MACD'] = {'signal': '空头加强', 'direction': '看跌', 'value': macd_hist, 'score': -1.0}
        elif macd_hist > 0:
            signals['MACD'] = {'signal': '多头', 'direction': '偏多', 'value': macd_hist, 'score': 0.5}
        else:
            signals['MACD'] = {'signal': '空头', 'direction': '偏空', 'value': macd_hist, 'score': -0.5}

        # 4. 布林带信号
        close = latest['close']
        boll_upper = latest.get('BOLL_Upper', close)
        boll_lower = latest.get('BOLL_Lower', close)
        boll_middle = latest.get('BOLL_Middle', close)

        boll_position = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper != boll_lower else 0.5

        if close >= boll_upper:
            signals['BOLL'] = {'signal': '触及上轨', 'direction': '看跌', 'value': boll_position, 'score': -0.8}
        elif close <= boll_lower:
            signals['BOLL'] = {'signal': '触及下轨', 'direction': '看涨', 'value': boll_position, 'score': 0.8}
        elif close > boll_middle:
            signals['BOLL'] = {'signal': '上半区', 'direction': '偏多', 'value': boll_position, 'score': 0.3}
        else:
            signals['BOLL'] = {'signal': '下半区', 'direction': '偏空', 'value': boll_position, 'score': -0.3}

        # 5. MA趋势信号（权重较高）
        ma7 = latest.get('MA7', close)
        ma25 = latest.get('MA25', close)
        ma99 = latest.get('MA99', close)

        if ma7 > ma25 > ma99:
            signals['MA'] = {'signal': '多头排列', 'direction': '看涨', 'value': 1, 'score': 1.5}
        elif ma7 < ma25 < ma99:
            signals['MA'] = {'signal': '空头排列', 'direction': '看跌', 'value': -1, 'score': -1.5}
        elif ma7 > ma25:
            signals['MA'] = {'signal': '短期偏多', 'direction': '偏多', 'value': 0.5, 'score': 0.5}
        elif ma7 < ma25:
            signals['MA'] = {'signal': '短期偏空', 'direction': '偏空', 'value': -0.5, 'score': -0.5}
        else:
            signals['MA'] = {'signal': '震荡', 'direction': '观望', 'value': 0, 'score': 0}

        # 6. EMA趋势信号
        ema12 = latest.get('EMA12', close)
        ema26 = latest.get('EMA26', close)
        ema50 = latest.get('EMA50', close)

        if close > ema12 > ema26 > ema50:
            signals['EMA'] = {'signal': 'EMA多头排列', 'direction': '看涨', 'value': 1, 'score': 1.2}
        elif close < ema12 < ema26 < ema50:
            signals['EMA'] = {'signal': 'EMA空头排列', 'direction': '看跌', 'value': -1, 'score': -1.2}
        elif close > ema26:
            signals['EMA'] = {'signal': 'EMA偏多', 'direction': '偏多', 'value': 0.5, 'score': 0.4}
        else:
            signals['EMA'] = {'signal': 'EMA偏空', 'direction': '偏空', 'value': -0.5, 'score': -0.4}

        # 7. 成交量信号
        volume = latest['volume']
        volume_ma = latest.get('Volume_MA', volume)
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1

        if volume_ratio > 2.0:
            signals['Volume'] = {'signal': '大幅放量', 'direction': '关注', 'value': volume_ratio, 'score': 0}
        elif volume_ratio > 1.5:
            signals['Volume'] = {'signal': '明显放量', 'direction': '关注', 'value': volume_ratio, 'score': 0}
        elif volume_ratio < 0.5:
            signals['Volume'] = {'signal': '缩量', 'direction': '观望', 'value': volume_ratio, 'score': 0}
        else:
            signals['Volume'] = {'signal': '正常', 'direction': '中性', 'value': volume_ratio, 'score': 0}

        return signals

    def get_comprehensive_prediction(self) -> Dict:
        """
        综合预测结果（深度优化版 - 多维度加权）

        Returns:
            综合预测信息
        """
        # 线性趋势预测
        change_percent, linear_direction = self.predict_linear_trend()

        # 指标信号
        signals = self.predict_with_indicators()

        # 趋势强度分析（ADX）
        trend_strength = self.analyze_trend_strength()

        # 动量分析
        momentum_analysis = self.analyze_momentum()

        # OBV量价背离分析
        obv_analysis = self.analyze_obv_divergence()

        # 趋势确认分析
        trend_confirmation = self.analyze_trend_confirmation()

        # 计算成交量权重
        volume_data = signals.get('Volume', {})
        volume_ratio = volume_data.get('value', 1.0)

        if volume_ratio > 2.0:
            volume_weight = 1.5
        elif volume_ratio > 1.5:
            volume_weight = 1.3
        elif volume_ratio > 1.2:
            volume_weight = 1.1
        elif volume_ratio < 0.5:
            volume_weight = 0.7
        else:
            volume_weight = 1.0

        # 计算综合评分（加权版本）
        weighted_score = 0.0
        total_weight = 0.0
        bullish_count = 0
        bearish_count = 0

        # 指标权重映射
        indicator_weight_map = {
            'MA': INDICATOR_WEIGHTS['MA_ALIGNMENT'],
            'EMA': INDICATOR_WEIGHTS['EMA_TREND'],
            'MACD': INDICATOR_WEIGHTS['MACD'],
            'RSI': INDICATOR_WEIGHTS['RSI'],
            'StochRSI': INDICATOR_WEIGHTS['STOCH_RSI'],
            'BOLL': INDICATOR_WEIGHTS['BOLL'],
        }

        for indicator, data in signals.items():
            if indicator == 'Volume':
                continue

            weight = indicator_weight_map.get(indicator, 1.0)
            score = data.get('score', 0)
            direction = data['direction']

            weighted_score += score * weight * volume_weight
            total_weight += weight

            if '涨' in direction or '多' in direction:
                bullish_count += 1
            elif '跌' in direction or '空' in direction:
                bearish_count += 1

        # 添加ADX趋势分数（如果趋势明确）
        if trend_strength['strength_score'] >= 1.0:
            adx_score = trend_strength['di_score'] * trend_strength['strength_score']
            weighted_score += adx_score * INDICATOR_WEIGHTS['ADX_TREND'] * volume_weight
            total_weight += INDICATOR_WEIGHTS['ADX_TREND']

        # 添加动量分数
        momentum_score = momentum_analysis['score']
        weighted_score += momentum_score * INDICATOR_WEIGHTS['MOMENTUM'] * volume_weight
        total_weight += INDICATOR_WEIGHTS['MOMENTUM']

        # 添加OBV分数
        obv_score = obv_analysis['score']
        weighted_score += obv_score * INDICATOR_WEIGHTS['OBV'] * volume_weight
        total_weight += INDICATOR_WEIGHTS['OBV']

        # 添加趋势确认分数
        confirmation_score = trend_confirmation['score']
        weighted_score += confirmation_score * volume_weight

        # 标准化评分
        normalized_score = weighted_score / total_weight if total_weight > 0 else 0

        # 根据ADX调整置信度
        adx_multiplier = trend_strength['strength_score']

        # 综合判断
        final_score = normalized_score * adx_multiplier

        if final_score >= 1.5:
            overall_direction = "强烈看涨"
            confidence = "极高" if trend_strength['strength_score'] >= 1.5 else "高"
        elif final_score >= 0.8:
            overall_direction = "看涨"
            confidence = "高" if trend_strength['strength_score'] >= 1.0 else "中高"
        elif final_score >= 0.3:
            overall_direction = "偏多"
            confidence = "中"
        elif final_score <= -1.5:
            overall_direction = "强烈看跌"
            confidence = "极高" if trend_strength['strength_score'] >= 1.5 else "高"
        elif final_score <= -0.8:
            overall_direction = "看跌"
            confidence = "高" if trend_strength['strength_score'] >= 1.0 else "中高"
        elif final_score <= -0.3:
            overall_direction = "偏空"
            confidence = "中"
        else:
            overall_direction = "震荡观望"
            confidence = "低"

        # 如果ADX显示无趋势，降低置信度
        if trend_strength['strength_score'] < 1.0 and confidence in ["极高", "高"]:
            confidence = "中"
            overall_direction = overall_direction.replace("强烈", "")

        return {
            'linear_change_percent': change_percent,
            'linear_direction': linear_direction,
            'signals': signals,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'score': round(final_score, 2),
            'raw_score': round(normalized_score, 2),
            'volume_weight': volume_weight,
            'overall_direction': overall_direction,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'momentum': momentum_analysis,
            'obv_divergence': obv_analysis,
            'trend_confirmation': trend_confirmation
        }

    def generate_trading_advice(self, support_levels: list = None, resistance_levels: list = None) -> Dict:
        """
        生成交易建议（是否开仓、止盈止损）

        Args:
            support_levels: 支撑位列表
            resistance_levels: 阻力位列表

        Returns:
            交易建议字典
        """
        prediction = self.get_comprehensive_prediction()
        latest = self.df.iloc[-1]
        current_price = latest['close']

        # ATR用于计算止损
        atr = latest.get('ATR', current_price * 0.02)  # 默认2%

        # 获取趋势强度和方向
        score = prediction['score']
        confidence = prediction['confidence']
        direction = prediction['overall_direction']
        trend_strength = prediction['trend_strength']

        # 计算波动率（用于杠杆建议）
        atr_percent = (atr / current_price) * 100  # ATR占价格的百分比

        # 初始化建议
        advice = {
            'action': '观望',           # 观望/做多/做空
            'action_en': 'wait',
            'reason': '',               # 原因说明
            'entry_price': current_price,
            'stop_loss': None,
            'stop_loss_percent': None,
            'take_profit_1': None,      # 第一止盈位
            'take_profit_2': None,      # 第二止盈位
            'take_profit_3': None,      # 第三止盈位
            'risk_reward_ratio': None,  # 风险收益比
            'position_suggestion': '',  # 仓位建议
            'leverage_suggestion': '',  # 杠杆建议
            'max_leverage': None,       # 建议最大杠杆
            'safe_leverage': None,      # 安全杠杆
            'risk_level': '中',         # 风险等级
            'key_points': [],           # 关键提示
        }

        # 判断是否建议开仓
        should_open = False
        is_long = False

        # 强烈信号且高置信度
        if confidence in ['极高', '高'] and trend_strength['strength_score'] >= 1.0:
            if '看涨' in direction or direction == '偏多':
                should_open = True
                is_long = True
                advice['action'] = '建议做多'
                advice['action_en'] = 'long'
            elif '看跌' in direction or direction == '偏空':
                should_open = True
                is_long = False
                advice['action'] = '建议做空'
                advice['action_en'] = 'short'

        # 中等信号
        elif confidence == '中高':
            if '看涨' in direction:
                should_open = True
                is_long = True
                advice['action'] = '可考虑做多'
                advice['action_en'] = 'long'
            elif '看跌' in direction:
                should_open = True
                is_long = False
                advice['action'] = '可考虑做空'
                advice['action_en'] = 'short'

        # 不建议开仓的情况
        if not should_open:
            # 分析原因
            reasons = []
            if trend_strength['strength_score'] < 1.0:
                reasons.append('ADX显示趋势不明确')
            if confidence in ['低', '中']:
                reasons.append('信号置信度不足')
            if '震荡' in direction:
                reasons.append('市场处于震荡区间')

            advice['reason'] = '，'.join(reasons) if reasons else '无明确交易信号'
            advice['key_points'] = [
                '当前市场方向不明确，建议等待更清晰的信号',
                '可关注关键支撑阻力位的突破情况',
                '保持耐心，避免频繁交易'
            ]
            advice['risk_level'] = '高' if trend_strength['strength_score'] < 0.8 else '中'
            return advice

        # 计算止损位
        if is_long:
            # 做多止损：使用ATR或支撑位
            atr_stop = current_price - (atr * 1.5)
            if support_levels and len(support_levels) > 0:
                nearest_support = max([s for s in support_levels if s < current_price], default=atr_stop)
                # 止损设在支撑位下方一点
                support_stop = nearest_support * 0.995
                advice['stop_loss'] = max(atr_stop, support_stop)
            else:
                advice['stop_loss'] = atr_stop

            advice['stop_loss_percent'] = ((current_price - advice['stop_loss']) / current_price) * 100

            # 计算止盈位
            risk = current_price - advice['stop_loss']
            advice['take_profit_1'] = current_price + (risk * 1.5)  # 1.5倍风险
            advice['take_profit_2'] = current_price + (risk * 2.5)  # 2.5倍风险
            advice['take_profit_3'] = current_price + (risk * 4.0)  # 4倍风险

            # 如果有阻力位，调整止盈
            if resistance_levels and len(resistance_levels) > 0:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                if nearest_resistance:
                    # 第一止盈不超过最近阻力位
                    advice['take_profit_1'] = min(advice['take_profit_1'], nearest_resistance * 0.998)

            advice['reason'] = f"多头信号明确，ADX={trend_strength['adx']:.1f}显示{trend_strength['strength']}"

        else:
            # 做空止损：使用ATR或阻力位
            atr_stop = current_price + (atr * 1.5)
            if resistance_levels and len(resistance_levels) > 0:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=atr_stop)
                # 止损设在阻力位上方一点
                resistance_stop = nearest_resistance * 1.005
                advice['stop_loss'] = min(atr_stop, resistance_stop)
            else:
                advice['stop_loss'] = atr_stop

            advice['stop_loss_percent'] = ((advice['stop_loss'] - current_price) / current_price) * 100

            # 计算止盈位
            risk = advice['stop_loss'] - current_price
            advice['take_profit_1'] = current_price - (risk * 1.5)
            advice['take_profit_2'] = current_price - (risk * 2.5)
            advice['take_profit_3'] = current_price - (risk * 4.0)

            # 如果有支撑位，调整止盈
            if support_levels and len(support_levels) > 0:
                nearest_support = max([s for s in support_levels if s < current_price], default=None)
                if nearest_support:
                    advice['take_profit_1'] = max(advice['take_profit_1'], nearest_support * 1.002)

            advice['reason'] = f"空头信号明确，ADX={trend_strength['adx']:.1f}显示{trend_strength['strength']}"

        # 计算风险收益比
        risk_amount = abs(current_price - advice['stop_loss'])
        reward_amount = abs(advice['take_profit_1'] - current_price)
        advice['risk_reward_ratio'] = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0

        # 仓位建议
        if confidence == '极高':
            advice['position_suggestion'] = '可用30-50%仓位'
            advice['risk_level'] = '低'
        elif confidence == '高':
            advice['position_suggestion'] = '建议20-30%仓位'
            advice['risk_level'] = '中低'
        elif confidence == '中高':
            advice['position_suggestion'] = '建议10-20%仓位'
            advice['risk_level'] = '中'
        else:
            advice['position_suggestion'] = '建议轻仓试探(5-10%)'
            advice['risk_level'] = '中高'

        # 杠杆倍数建议
        # 基于止损百分比计算安全杠杆（爆仓风险控制在止损点）
        # 公式：最大杠杆 = 100 / 止损百分比 * 安全系数
        stop_loss_pct = advice['stop_loss_percent'] or 3.0  # 默认3%止损

        # 基础杠杆计算（保证止损不爆仓）
        base_leverage = 100 / stop_loss_pct * 0.8  # 0.8是安全系数

        # 根据置信度调整杠杆
        if confidence == '极高':
            confidence_multiplier = 1.0
            max_allowed = 20
        elif confidence == '高':
            confidence_multiplier = 0.8
            max_allowed = 15
        elif confidence == '中高':
            confidence_multiplier = 0.6
            max_allowed = 10
        else:
            confidence_multiplier = 0.4
            max_allowed = 5

        # 根据趋势强度调整
        if trend_strength['strength_score'] >= 1.5:
            trend_multiplier = 1.2  # 强趋势可适当提高
        elif trend_strength['strength_score'] >= 1.0:
            trend_multiplier = 1.0
        else:
            trend_multiplier = 0.7  # 弱趋势降低杠杆

        # 根据波动率调整（波动大则降低杠杆）
        if atr_percent > 5:
            volatility_multiplier = 0.5  # 高波动
        elif atr_percent > 3:
            volatility_multiplier = 0.7
        elif atr_percent > 2:
            volatility_multiplier = 0.85
        else:
            volatility_multiplier = 1.0  # 低波动

        # 计算最终杠杆建议
        calculated_leverage = base_leverage * confidence_multiplier * trend_multiplier * volatility_multiplier
        max_leverage = min(int(calculated_leverage), max_allowed)
        safe_leverage = max(2, int(max_leverage * 0.6))  # 安全杠杆为最大的60%

        # 确保杠杆在合理范围内
        max_leverage = max(2, min(max_leverage, 50))
        safe_leverage = max(2, min(safe_leverage, max_leverage))

        advice['max_leverage'] = max_leverage
        advice['safe_leverage'] = safe_leverage

        # 生成杠杆建议文字
        if max_leverage <= 3:
            advice['leverage_suggestion'] = f'建议低杠杆 {safe_leverage}-{max_leverage}x（高风险市况）'
        elif max_leverage <= 5:
            advice['leverage_suggestion'] = f'建议谨慎 {safe_leverage}-{max_leverage}x'
        elif max_leverage <= 10:
            advice['leverage_suggestion'] = f'可用 {safe_leverage}-{max_leverage}x'
        elif max_leverage <= 15:
            advice['leverage_suggestion'] = f'可用 {safe_leverage}-{max_leverage}x（趋势明确）'
        else:
            advice['leverage_suggestion'] = f'可用 {safe_leverage}-{max_leverage}x（强趋势）'

        # 关键提示
        advice['key_points'] = []

        # 动量提示
        momentum = prediction['momentum']
        if '加速' in momentum['signal']:
            advice['key_points'].append(f"动量{momentum['signal']}，趋势可能延续")
        elif '减弱' in momentum['signal']:
            advice['key_points'].append(f"注意：{momentum['signal']}，可能出现回调")

        # OBV提示
        obv = prediction['obv_divergence']
        if '背离' in obv['signal']:
            advice['key_points'].append(f"警告：出现{obv['signal']}，谨慎操作")

        # 止损提示
        advice['key_points'].append(f"严格止损，最大亏损控制在{advice['stop_loss_percent']:.1f}%")

        # 杠杆提示
        if max_leverage <= 5:
            advice['key_points'].append(f"当前波动较大，建议使用{safe_leverage}x以下杠杆")
        else:
            advice['key_points'].append(f"杠杆建议：稳健{safe_leverage}x，激进不超过{max_leverage}x")

        # 分批止盈提示
        advice['key_points'].append("建议分批止盈：TP1平30%，TP2平40%，TP3平30%")

        return advice
