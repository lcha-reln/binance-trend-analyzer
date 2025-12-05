"""
恐惧贪婪指数模块
@author Reln Ding

数据来源: alternative.me Fear & Greed Index API
指数范围: 0-100
- 0-24: 极度恐惧 (Extreme Fear)
- 25-44: 恐惧 (Fear)
- 45-55: 中性 (Neutral)
- 56-75: 贪婪 (Greed)
- 76-100: 极度贪婪 (Extreme Greed)
"""

import requests
import time
from typing import Optional, Dict, Any

# 缓存
_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0
}
CACHE_TTL = 300  # 缓存5分钟


def get_fear_greed_index() -> Optional[Dict[str, Any]]:
    """
    获取恐惧贪婪指数

    返回:
        {
            'value': int,           # 指数值 0-100
            'classification': str,  # 分类（中文）
            'classification_en': str,  # 分类（英文）
            'sentiment': str,       # 情绪描述
            'signal': str,          # 交易信号建议
            'update_time': str      # 更新时间
        }
    """
    global _cache

    # 检查缓存
    now = time.time()
    if _cache['data'] and (now - _cache['timestamp']) < CACHE_TTL:
        return _cache['data']

    try:
        response = requests.get(
            'https://api.alternative.me/fng/?limit=1',
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get('data') and len(data['data']) > 0:
            fng = data['data'][0]
            value = int(fng['value'])
            classification_en = fng['value_classification']

            # 中文分类
            classification_map = {
                'Extreme Fear': '极度恐惧',
                'Fear': '恐惧',
                'Neutral': '中性',
                'Greed': '贪婪',
                'Extreme Greed': '极度贪婪'
            }
            classification = classification_map.get(classification_en, classification_en)

            # 情绪描述和交易信号
            sentiment, signal = _analyze_sentiment(value)

            # 更新时间
            timestamp = int(fng['timestamp'])
            update_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))

            result = {
                'value': value,
                'classification': classification,
                'classification_en': classification_en,
                'sentiment': sentiment,
                'signal': signal,
                'update_time': update_time
            }

            # 更新缓存
            _cache['data'] = result
            _cache['timestamp'] = now

            return result

    except Exception as e:
        print(f'[ERROR] 获取恐惧贪婪指数失败: {e}')
        # 返回缓存数据（如果有）
        if _cache['data']:
            return _cache['data']
        return None


def _analyze_sentiment(value: int) -> tuple:
    """
    分析情绪并给出交易信号

    返回: (情绪描述, 交易信号)
    """
    if value <= 10:
        return (
            '市场处于极度恐慌状态，恐惧情绪达到极端',
            '历史上极度恐惧往往是抄底良机，可考虑分批建仓'
        )
    elif value <= 24:
        return (
            '市场恐惧情绪浓厚，投资者普遍悲观',
            '逆向思维：极度恐惧时可能接近底部区域'
        )
    elif value <= 44:
        return (
            '市场情绪偏向恐惧，观望者居多',
            '市场情绪偏空，注意风险控制，可小仓位试探'
        )
    elif value <= 55:
        return (
            '市场情绪中性，多空力量相对平衡',
            '情绪中性，可根据技术面和基本面综合判断'
        )
    elif value <= 75:
        return (
            '市场情绪偏向贪婪，乐观情绪上升',
            '情绪偏乐观，注意追高风险，可持有但谨慎加仓'
        )
    elif value <= 89:
        return (
            '市场贪婪情绪浓厚，FOMO情绪明显',
            '市场过热信号，注意止盈，不宜重仓追涨'
        )
    else:
        return (
            '市场处于极度贪婪状态，狂热情绪蔓延',
            '历史上极度贪婪往往预示顶部区域，建议逐步减仓'
        )


def get_fng_adjustment(value: int) -> Dict[str, Any]:
    """
    根据恐惧贪婪指数获取交易建议调整参数

    用于调整交易建议的激进程度
    """
    if value <= 20:
        # 极度恐惧 - 适合抄底
        return {
            'bias': 'bullish',  # 偏向做多
            'confidence_boost': 0.1,  # 做多置信度提升
            'leverage_adjust': 0.8,  # 杠杆降低（风险控制）
            'note': '极度恐惧，可逆向考虑做多'
        }
    elif value <= 35:
        # 恐惧
        return {
            'bias': 'slightly_bullish',
            'confidence_boost': 0.05,
            'leverage_adjust': 0.9,
            'note': '恐惧情绪，技术面做多信号可适当加权'
        }
    elif value <= 65:
        # 中性
        return {
            'bias': 'neutral',
            'confidence_boost': 0,
            'leverage_adjust': 1.0,
            'note': '情绪中性，按技术面判断'
        }
    elif value <= 80:
        # 贪婪
        return {
            'bias': 'slightly_bearish',
            'confidence_boost': -0.05,
            'leverage_adjust': 0.9,
            'note': '贪婪情绪，技术面做空信号可适当加权'
        }
    else:
        # 极度贪婪 - 警惕顶部
        return {
            'bias': 'bearish',
            'confidence_boost': -0.1,
            'leverage_adjust': 0.7,
            'note': '极度贪婪，警惕回调风险'
        }


if __name__ == '__main__':
    # 测试
    result = get_fear_greed_index()
    if result:
        print(f"恐惧贪婪指数: {result['value']}")
        print(f"分类: {result['classification']}")
        print(f"情绪: {result['sentiment']}")
        print(f"信号: {result['signal']}")
        print(f"更新时间: {result['update_time']}")

        adj = get_fng_adjustment(result['value'])
        print(f"\n交易调整: {adj}")
