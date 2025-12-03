"""
技术指标计算模块
@author Reln Ding
"""

import pandas as pd
import numpy as np
from config import MA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BOLL_PERIOD, BOLL_STD


def calculate_ma(df: pd.DataFrame, periods: list = MA_PERIODS) -> pd.DataFrame:
    """
    计算移动平均线

    Args:
        df: K线数据
        periods: MA周期列表

    Returns:
        添加MA列后的DataFrame
    """
    for period in periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    计算指数移动平均线

    Args:
        df: K线数据
        period: EMA周期

    Returns:
        EMA序列
    """
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    计算RSI指标

    Args:
        df: K线数据
        period: RSI周期

    Returns:
        添加RSI列后的DataFrame
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW,
                   signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """
    计算MACD指标

    Args:
        df: K线数据
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        添加MACD相关列后的DataFrame
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_bollinger(df: pd.DataFrame, period: int = BOLL_PERIOD,
                        std_dev: int = BOLL_STD) -> pd.DataFrame:
    """
    计算布林带

    Args:
        df: K线数据
        period: 布林带周期
        std_dev: 标准差倍数

    Returns:
        添加布林带列后的DataFrame
    """
    df['BOLL_Middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['BOLL_Upper'] = df['BOLL_Middle'] + (rolling_std * std_dev)
    df['BOLL_Lower'] = df['BOLL_Middle'] - (rolling_std * std_dev)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算ATR（平均真实波幅）

    Args:
        df: K线数据
        period: ATR周期

    Returns:
        添加ATR列后的DataFrame
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    return df


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算成交量移动平均

    Args:
        df: K线数据
        period: 周期

    Returns:
        添加成交量MA后的DataFrame
    """
    df['Volume_MA'] = df['volume'].rolling(window=period).mean()
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标

    Args:
        df: K线数据

    Returns:
        添加所有指标后的DataFrame
    """
    df = calculate_ma(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger(df)
    df = calculate_atr(df)
    df = calculate_volume_ma(df)
    return df
