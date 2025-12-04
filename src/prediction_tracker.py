"""
预测准确率追踪模块 - 使用 SQLite 存储，支持反馈学习
@author Reln Ding
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

from feedback_learning import get_feedback_db, get_online_learner, FeedbackDatabase

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
    """预测准确率追踪器（SQLite版本）"""

    def __init__(self):
        self.db: FeedbackDatabase = get_feedback_db()
        self._lock = threading.RLock()
        self._training_check_counter = 0

    def record_prediction(self, symbol: str, interval: str,
                          direction: str, confidence: str,
                          score: float, price: float,
                          features: Dict = None):
        """
        记录一条预测

        Args:
            symbol: 交易对
            interval: 周期
            direction: 预测方向
            confidence: 置信度
            score: 评分
            price: 当前价格
            features: 特征快照（技术指标等）
        """
        now = time.time()

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
        }

        self.db.insert_prediction(prediction, features)

    def verify_predictions(self, current_prices: Dict[str, float]):
        """
        验证历史预测

        Args:
            current_prices: {symbol: current_price}
        """
        now = time.time()
        unverified = self.db.get_unverified_predictions()

        for pred in unverified:
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

            # 更新数据库
            self.db.update_verification(
                pred['id'], actual_direction, current_price,
                round(price_change_percent, 4), is_correct
            )

        # 检查是否需要触发训练
        self._check_and_trigger_training()

    def _check_prediction_correct(self, predicted: str, actual: str,
                                   change_percent: float) -> bool:
        """判断预测是否正确"""
        if predicted == 'neutral':
            return abs(change_percent) < 0.5
        return predicted == actual

    def _check_and_trigger_training(self):
        """检查并触发增量训练"""
        self._training_check_counter += 1

        # 每10次验证检查一次是否需要训练
        if self._training_check_counter % 10 != 0:
            return

        learner = get_online_learner()
        if learner.should_retrain():
            # 异步触发训练
            thread = threading.Thread(target=self._do_training, daemon=True)
            thread.start()

    def _do_training(self):
        """执行训练（在后台线程中）"""
        try:
            learner = get_online_learner()
            result = learner.train()
            if result:
                print(f"[INFO] 反馈学习训练完成: accuracy={result['accuracy']:.2%}, "
                      f"samples={result['train_samples']}")
        except Exception as e:
            print(f"[ERROR] 反馈学习训练失败: {e}")

    def get_accuracy_stats(self) -> Dict:
        """获取准确率统计"""
        return self.db.get_accuracy_stats()

    def get_interval_accuracy(self, symbol: str, interval: str) -> Optional[Dict]:
        """获取特定交易对和周期的准确率"""
        stats = self.db.get_accuracy_stats()
        by_interval = stats.get('by_interval', {})
        return by_interval.get(interval)


# 全局实例
_tracker_instance: Optional[PredictionTracker] = None


def get_tracker() -> PredictionTracker:
    """获取全局追踪器实例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance
