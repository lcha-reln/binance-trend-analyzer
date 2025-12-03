"""
数据采集模块 - 从币安API获取K线数据
@author Reln Ding
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from config import BINANCE_BASE_URL, KLINE_LIMIT


def get_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> Optional[pd.DataFrame]:
    """
    获取K线数据

    Args:
        symbol: 交易对，如 BTCUSDT
        interval: K线周期，如 30m, 1h
        limit: 获取K线数量

    Returns:
        DataFrame包含: open_time, open, high, low, close, volume, close_time
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 转换数据类型
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)

        # 只保留需要的列
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]

        return df

    except requests.RequestException as e:
        print(f"[ERROR] 获取K线数据失败 {symbol} {interval}: {e}")
        return None


def get_ticker_price(symbol: str) -> Optional[float]:
    """
    获取当前价格

    Args:
        symbol: 交易对

    Returns:
        当前价格
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.RequestException as e:
        print(f"[ERROR] 获取价格失败 {symbol}: {e}")
        return None


def get_24h_stats(symbol: str) -> Optional[dict]:
    """
    获取24小时统计数据

    Args:
        symbol: 交易对

    Returns:
        24小时统计信息
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    params = {"symbol": symbol}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'high_price': float(data['highPrice']),
            'low_price': float(data['lowPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume'])
        }
    except requests.RequestException as e:
        print(f"[ERROR] 获取24h统计失败 {symbol}: {e}")
        return None
