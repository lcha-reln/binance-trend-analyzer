"""
趋势分析模块 - 整合数据采集、指标计算和预测
@author Reln Ding
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from collector import get_klines, get_ticker_price, get_24h_stats
from indicators import calculate_all_indicators
from predictor import TrendPredictor
from config import SYMBOLS, INTERVALS


class TrendAnalyzer:
    """趋势分析器"""

    def __init__(self):
        self.results = {}

    def analyze_symbol(self, symbol: str, interval: str) -> Optional[Dict]:
        """
        分析单个交易对

        Args:
            symbol: 交易对
            interval: K线周期

        Returns:
            分析结果字典
        """
        # 获取K线数据
        df = get_klines(symbol, interval)
        if df is None or len(df) < 100:
            return None

        # 计算技术指标
        df = calculate_all_indicators(df)

        # 获取当前价格
        current_price = get_ticker_price(symbol)

        # 获取24小时统计
        stats_24h = get_24h_stats(symbol)

        # 预测
        predictor = TrendPredictor(df)
        prediction = predictor.get_comprehensive_prediction()

        # 构建结果
        latest = df.iloc[-1]
        result = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'open_time': latest['open_time'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume'],
            'stats_24h': stats_24h,
            'indicators': {
                'MA7': latest.get('MA7'),
                'MA25': latest.get('MA25'),
                'MA99': latest.get('MA99'),
                'RSI': latest.get('RSI'),
                'MACD': latest.get('MACD'),
                'MACD_Signal': latest.get('MACD_Signal'),
                'MACD_Hist': latest.get('MACD_Hist'),
                'BOLL_Upper': latest.get('BOLL_Upper'),
                'BOLL_Middle': latest.get('BOLL_Middle'),
                'BOLL_Lower': latest.get('BOLL_Lower'),
                'ATR': latest.get('ATR'),
            },
            'prediction': prediction
        }

        return result

    def analyze_all(self) -> Dict:
        """
        分析所有配置的交易对和周期

        Returns:
            所有分析结果
        """
        results = {}
        for symbol in SYMBOLS:
            results[symbol] = {}
            for interval in INTERVALS:
                result = self.analyze_symbol(symbol, interval)
                if result:
                    results[symbol][interval] = result

            # 添加多周期共振分析
            results[symbol]['resonance'] = self.analyze_resonance(results[symbol])

        return results

    def analyze_resonance(self, symbol_data: Dict) -> Dict:
        """
        分析多周期共振

        Args:
            symbol_data: 单个交易对的所有周期数据

        Returns:
            共振分析结果
        """
        # 收集各周期的方向信号
        directions = {}
        scores = {}
        volume_weights = {}

        for interval, data in symbol_data.items():
            if interval == 'resonance' or data is None:
                continue
            pred = data.get('prediction', {})
            directions[interval] = pred.get('overall_direction', '观望')
            scores[interval] = pred.get('score', 0)
            volume_weights[interval] = pred.get('volume_weight', 1.0)

        if not directions:
            return {'resonance_level': '无数据', 'signal': '观望', 'confidence': '无'}

        # 统计多空方向
        bullish_intervals = []
        bearish_intervals = []

        for interval, direction in directions.items():
            if '涨' in direction or '多' in direction:
                bullish_intervals.append(interval)
            elif '跌' in direction or '空' in direction:
                bearish_intervals.append(interval)

        total_intervals = len(directions)
        bullish_count = len(bullish_intervals)
        bearish_count = len(bearish_intervals)

        # 计算加权总分（大周期权重更高）
        interval_weights = {'5m': 0.5, '15m': 0.8, '30m': 1.0, '1h': 1.5, '4h': 2.0, '1d': 3.0}
        weighted_score = 0
        total_weight = 0

        for interval, score in scores.items():
            weight = interval_weights.get(interval, 1.0)
            vol_weight = volume_weights.get(interval, 1.0)
            weighted_score += score * weight * vol_weight
            total_weight += weight

        avg_weighted_score = weighted_score / total_weight if total_weight > 0 else 0

        # 判断共振级别
        if bullish_count == total_intervals:
            resonance_level = "完全共振"
            signal = "强烈看涨"
            confidence = "极高"
        elif bearish_count == total_intervals:
            resonance_level = "完全共振"
            signal = "强烈看跌"
            confidence = "极高"
        elif bullish_count >= total_intervals * 0.75:
            resonance_level = "高度共振"
            signal = "看涨"
            confidence = "高"
        elif bearish_count >= total_intervals * 0.75:
            resonance_level = "高度共振"
            signal = "看跌"
            confidence = "高"
        elif bullish_count >= total_intervals * 0.5:
            resonance_level = "部分共振"
            signal = "偏多"
            confidence = "中"
        elif bearish_count >= total_intervals * 0.5:
            resonance_level = "部分共振"
            signal = "偏空"
            confidence = "中"
        else:
            resonance_level = "无共振"
            signal = "观望"
            confidence = "低"

        return {
            'resonance_level': resonance_level,
            'signal': signal,
            'confidence': confidence,
            'bullish_intervals': bullish_intervals,
            'bearish_intervals': bearish_intervals,
            'weighted_score': round(avg_weighted_score, 2),
            'directions': directions
        }

    def format_output(self, results: Dict) -> str:
        """
        格式化输出结果

        Args:
            results: 分析结果

        Returns:
            格式化的字符串
        """
        output = []
        output.append("\n" + "=" * 80)
        output.append(f"  币安趋势分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)

        for symbol, intervals in results.items():
            output.append(f"\n{'─' * 80}")
            output.append(f"  {symbol}")
            output.append(f"{'─' * 80}")

            for interval, data in intervals.items():
                if interval == 'resonance':
                    continue
                output.append(f"\n  ▶ 周期: {interval}")
                output.append(f"  ├─ 当前价格: ${data['current_price']:,.2f}")

                # 24小时统计
                if data['stats_24h']:
                    stats = data['stats_24h']
                    change_symbol = "+" if stats['price_change_percent'] >= 0 else ""
                    output.append(f"  ├─ 24h涨跌: {change_symbol}{stats['price_change_percent']:.2f}%")
                    output.append(f"  ├─ 24h最高: ${stats['high_price']:,.2f}")
                    output.append(f"  ├─ 24h最低: ${stats['low_price']:,.2f}")

                # K线数据
                output.append(f"  ├─ 开盘: ${data['open']:,.2f} | 最高: ${data['high']:,.2f} | 最低: ${data['low']:,.2f} | 收盘: ${data['close']:,.2f}")

                # 技术指标
                ind = data['indicators']
                output.append(f"  │")
                output.append(f"  ├─ 技术指标:")
                if ind['MA7'] and ind['MA25'] and ind['MA99']:
                    output.append(f"  │  ├─ MA: 7={ind['MA7']:,.2f} | 25={ind['MA25']:,.2f} | 99={ind['MA99']:,.2f}")
                if ind['RSI']:
                    rsi_status = "超买" if ind['RSI'] > 70 else ("超卖" if ind['RSI'] < 30 else "中性")
                    output.append(f"  │  ├─ RSI: {ind['RSI']:.2f} ({rsi_status})")
                if ind['MACD'] is not None:
                    output.append(f"  │  ├─ MACD: {ind['MACD']:.4f} | Signal: {ind['MACD_Signal']:.4f} | Hist: {ind['MACD_Hist']:.4f}")
                if ind['BOLL_Upper']:
                    output.append(f"  │  └─ BOLL: 上轨={ind['BOLL_Upper']:,.2f} | 中轨={ind['BOLL_Middle']:,.2f} | 下轨={ind['BOLL_Lower']:,.2f}")

                # 预测结果
                pred = data['prediction']
                output.append(f"  │")
                output.append(f"  ├─ 信号分析:")
                for indicator, signal in pred['signals'].items():
                    output.append(f"  │  ├─ {indicator}: {signal['signal']} → {signal['direction']}")

                output.append(f"  │")
                output.append(f"  └─ 综合预测:")
                output.append(f"     ├─ 趋势预测: {pred['linear_direction']} (预计变化: {pred['linear_change_percent']:+.2f}%)")
                output.append(f"     ├─ 多空比: 看涨{pred['bullish_count']} / 看跌{pred['bearish_count']}")
                output.append(f"     ├─ 综合评分: {pred['score']} (成交量权重: {pred.get('volume_weight', 1.0):.1f}x)")
                output.append(f"     └─ 建议: {pred['overall_direction']} (置信度: {pred['confidence']})")

            # 多周期共振分析
            resonance = intervals.get('resonance', {})
            if resonance:
                output.append(f"\n  {'═' * 76}")
                output.append(f"  ★ 多周期共振分析")
                output.append(f"  {'═' * 76}")
                output.append(f"  ├─ 共振级别: {resonance.get('resonance_level', 'N/A')}")
                output.append(f"  ├─ 综合信号: {resonance.get('signal', 'N/A')}")
                output.append(f"  ├─ 置信度: {resonance.get('confidence', 'N/A')}")
                output.append(f"  ├─ 加权评分: {resonance.get('weighted_score', 0)}")

                # 显示各周期方向
                directions = resonance.get('directions', {})
                if directions:
                    dir_str = " | ".join([f"{k}: {v}" for k, v in directions.items()])
                    output.append(f"  ├─ 各周期: {dir_str}")

                bullish = resonance.get('bullish_intervals', [])
                bearish = resonance.get('bearish_intervals', [])
                if bullish:
                    output.append(f"  ├─ 看涨周期: {', '.join(bullish)}")
                if bearish:
                    output.append(f"  ├─ 看跌周期: {', '.join(bearish)}")

                # 最终建议
                conf = resonance.get('confidence', '')
                if conf in ['极高', '高']:
                    output.append(f"  └─ >>> 建议参考: {resonance.get('signal')} (多周期{resonance.get('resonance_level')}) <<<")
                else:
                    output.append(f"  └─ >>> 建议观望，等待更明确信号 <<<")

        output.append("\n" + "=" * 80)
        output.append("  风险提示: 以上分析仅供参考，不构成投资建议，投资有风险，入市需谨慎！")
        output.append("=" * 80 + "\n")

        return "\n".join(output)
