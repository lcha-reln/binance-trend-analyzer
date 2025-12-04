"""
预测准确率追踪模块 - 记录历史预测并验证准确率
@author Reln Ding
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


# 预测记录存储路径
DATA_DIR = Path(__file__).parent.parent / 'data'
PREDICTIONS_FILE = DATA_DIR / 'predictions.json'

# 各周期的验证时间（分钟）
VERIFY_PERIODS = {
    '5m': 5,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}

# 预测方向映射
DIRECTION_MAP = {
    '强烈看涨': 'bullish',
    '看涨': 'bullish',
    '偏多': 'bullish',
    '强烈看跌': 'bearish',
    '看跌': 'bearish',
    '偏空': 'bearish',
    '震荡观望': 'neutral',
}


class PredictionTracker:
    """预测准确率追踪器"""

    def __init__(self):
        self._predictions: List[Dict] = []
        self._lock = threading.RLock()
        self._load_predictions()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_predictions(self):
        """从文件加载历史预测"""
        self._ensure_data_dir()
        try:
            if PREDICTIONS_FILE.exists():
                with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                    self._predictions = json.load(f)
                # 清理超过7天的旧数据
                self._cleanup_old_predictions()
        except Exception as e:
            print(f"[WARN] 加载预测记录失败: {e}")
            self._predictions = []

    def _save_predictions(self):
        """保存预测到文件"""
        self._ensure_data_dir()
        try:
            with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._predictions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 保存预测记录失败: {e}")

    def _cleanup_old_predictions(self):
        """清理超过7天的旧预测"""
        cutoff_time = time.time() - (7 * 24 * 60 * 60)
        with self._lock:
            self._predictions = [
                p for p in self._predictions
                if p.get('timestamp', 0) > cutoff_time
            ]

    def record_prediction(self, symbol: str, interval: str,
                          direction: str, confidence: str,
                          score: float, price: float):
        """
        记录一条预测

        Args:
            symbol: 交易对
            interval: 周期
            direction: 预测方向
            confidence: 置信度
            score: 评分
            price: 当前价格
        """
        # 检查是否已有相同的近期预测（避免重复记录）
        now = time.time()
        verify_minutes = VERIFY_PERIODS.get(interval, 60)
        min_gap = verify_minutes * 60 * 0.8  # 80%的验证周期作为最小间隔

        with self._lock:
            # 查找最近的同类预测
            for p in reversed(self._predictions):
                if (p['symbol'] == symbol and
                    p['interval'] == interval and
                    now - p['timestamp'] < min_gap):
                    # 已有近期预测，不重复记录
                    return

            # 标准化方向
            direction_normalized = DIRECTION_MAP.get(direction, 'neutral')

            prediction = {
                'id': f"{symbol}_{interval}_{int(now)}",
                'symbol': symbol,
                'interval': interval,
                'direction': direction,
                'direction_normalized': direction_normalized,
                'confidence': confidence,
                'score': score,
                'price_at_prediction': price,
                'timestamp': now,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'verified': False,
                'actual_direction': None,
                'price_at_verify': None,
                'price_change_percent': None,
                'is_correct': None,
            }

            self._predictions.append(prediction)
            self._save_predictions()

    def verify_predictions(self, current_prices: Dict[str, float]):
        """
        验证历史预测

        Args:
            current_prices: {symbol: current_price}
        """
        now = time.time()
        updated = False

        with self._lock:
            for pred in self._predictions:
                if pred['verified']:
                    continue

                symbol = pred['symbol']
                interval = pred['interval']
                timestamp = pred['timestamp']

                # 检查是否到了验证时间
                verify_minutes = VERIFY_PERIODS.get(interval, 60)
                verify_time = timestamp + (verify_minutes * 60)

                if now < verify_time:
                    continue

                # 获取当前价格
                current_price = current_prices.get(symbol)
                if current_price is None:
                    continue

                # 计算价格变化
                price_at_prediction = pred['price_at_prediction']
                price_change = current_price - price_at_prediction
                price_change_percent = (price_change / price_at_prediction) * 100

                # 判断实际方向
                if price_change_percent > 0.1:
                    actual_direction = 'bullish'
                elif price_change_percent < -0.1:
                    actual_direction = 'bearish'
                else:
                    actual_direction = 'neutral'

                # 判断预测是否正确
                predicted_direction = pred['direction_normalized']
                is_correct = self._check_prediction_correct(
                    predicted_direction, actual_direction, price_change_percent
                )

                # 更新预测记录
                pred['verified'] = True
                pred['actual_direction'] = actual_direction
                pred['price_at_verify'] = current_price
                pred['price_change_percent'] = round(price_change_percent, 4)
                pred['is_correct'] = is_correct
                pred['verify_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                updated = True

            if updated:
                self._save_predictions()

    def _check_prediction_correct(self, predicted: str, actual: str,
                                   change_percent: float) -> bool:
        """
        判断预测是否正确

        规则：
        - 预测看涨/看跌，实际方向一致即为正确
        - 预测中性/观望，实际变化在±0.5%内即为正确
        """
        if predicted == 'neutral':
            return abs(change_percent) < 0.5

        return predicted == actual

    def get_accuracy_stats(self) -> Dict:
        """
        获取准确率统计

        Returns:
            统计数据字典
        """
        with self._lock:
            verified = [p for p in self._predictions if p['verified']]

            if not verified:
                return {
                    'total_predictions': len(self._predictions),
                    'verified_count': 0,
                    'pending_count': len(self._predictions),
                    'overall_accuracy': None,
                    'by_interval': {},
                    'by_symbol': {},
                    'by_confidence': {},
                    'recent_predictions': [],
                }

            # 总体准确率
            correct_count = sum(1 for p in verified if p['is_correct'])
            overall_accuracy = (correct_count / len(verified)) * 100

            # 按周期统计
            by_interval = {}
            for interval in VERIFY_PERIODS.keys():
                interval_preds = [p for p in verified if p['interval'] == interval]
                if interval_preds:
                    correct = sum(1 for p in interval_preds if p['is_correct'])
                    by_interval[interval] = {
                        'total': len(interval_preds),
                        'correct': correct,
                        'accuracy': round((correct / len(interval_preds)) * 100, 1),
                    }

            # 按交易对统计
            by_symbol = {}
            symbols = set(p['symbol'] for p in verified)
            for symbol in symbols:
                symbol_preds = [p for p in verified if p['symbol'] == symbol]
                correct = sum(1 for p in symbol_preds if p['is_correct'])
                by_symbol[symbol] = {
                    'total': len(symbol_preds),
                    'correct': correct,
                    'accuracy': round((correct / len(symbol_preds)) * 100, 1),
                }

            # 按置信度统计
            by_confidence = {}
            confidences = set(p['confidence'] for p in verified)
            for conf in confidences:
                conf_preds = [p for p in verified if p['confidence'] == conf]
                correct = sum(1 for p in conf_preds if p['is_correct'])
                by_confidence[conf] = {
                    'total': len(conf_preds),
                    'correct': correct,
                    'accuracy': round((correct / len(conf_preds)) * 100, 1),
                }

            # 最近的预测记录（最近20条已验证的）
            recent = sorted(verified, key=lambda x: x['timestamp'], reverse=True)[:20]
            recent_predictions = [{
                'symbol': p['symbol'],
                'interval': p['interval'],
                'direction': p['direction'],
                'confidence': p['confidence'],
                'datetime': p['datetime'],
                'price_at_prediction': p['price_at_prediction'],
                'price_at_verify': p['price_at_verify'],
                'price_change_percent': p['price_change_percent'],
                'is_correct': p['is_correct'],
            } for p in recent]

            return {
                'total_predictions': len(self._predictions),
                'verified_count': len(verified),
                'pending_count': len(self._predictions) - len(verified),
                'correct_count': correct_count,
                'overall_accuracy': round(overall_accuracy, 1),
                'by_interval': by_interval,
                'by_symbol': by_symbol,
                'by_confidence': by_confidence,
                'recent_predictions': recent_predictions,
            }

    def get_interval_accuracy(self, symbol: str, interval: str) -> Optional[Dict]:
        """
        获取特定交易对和周期的准确率

        Args:
            symbol: 交易对
            interval: 周期

        Returns:
            准确率信息或None
        """
        with self._lock:
            verified = [
                p for p in self._predictions
                if p['verified'] and p['symbol'] == symbol and p['interval'] == interval
            ]

            if not verified:
                return None

            correct = sum(1 for p in verified if p['is_correct'])
            return {
                'total': len(verified),
                'correct': correct,
                'accuracy': round((correct / len(verified)) * 100, 1),
            }


# 全局实例
_tracker_instance: Optional[PredictionTracker] = None


def get_tracker() -> PredictionTracker:
    """获取全局追踪器实例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance
