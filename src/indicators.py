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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算ADX（平均趋向指数）- 用于判断趋势强度

    Args:
        df: K线数据
        period: ADX周期

    Returns:
        添加ADX、+DI、-DI列后的DataFrame
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 计算+DM和-DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1

    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

    # 重新计算
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    # 当+DM > -DM时，-DM = 0；反之亦然
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    # 计算TR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑TR、+DM、-DM
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

    # 计算+DI和-DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # 计算DX和ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = adx

    return df


def calculate_stoch_rsi(df: pd.DataFrame, rsi_period: int = 14,
                        stoch_period: int = 14, k_period: int = 3,
                        d_period: int = 3) -> pd.DataFrame:
    """
    计算Stochastic RSI - 比RSI更敏感的超买超卖指标

    Args:
        df: K线数据
        rsi_period: RSI周期
        stoch_period: Stochastic周期
        k_period: K线平滑周期
        d_period: D线平滑周期

    Returns:
        添加StochRSI_K和StochRSI_D后的DataFrame
    """
    # 确保RSI已计算
    if 'RSI' not in df.columns:
        df = calculate_rsi(df, rsi_period)

    rsi = df['RSI']

    # 计算Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()

    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

    # K线和D线
    df['StochRSI_K'] = stoch_rsi.rolling(window=k_period).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(window=d_period).mean()

    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算OBV（能量潮）- 量价关系分析

    Args:
        df: K线数据

    Returns:
        添加OBV和OBV_MA后的DataFrame
    """
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    return df


def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算动量指标

    Args:
        df: K线数据
        period: 动量周期

    Returns:
        添加Momentum和ROC后的DataFrame
    """
    # 动量 = 当前价格 - N周期前价格
    df['Momentum'] = df['close'] - df['close'].shift(period)

    # ROC (变化率) = (当前价格 - N周期前价格) / N周期前价格 * 100
    df['ROC'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    return df


def calculate_ema_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算EMA趋势指标

    Args:
        df: K线数据

    Returns:
        添加EMA指标后的DataFrame
    """
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

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
    df = calculate_ema_trend(df)
    df = calculate_rsi(df)
    df = calculate_stoch_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_obv(df)
    df = calculate_momentum(df)
    df = calculate_volume_ma(df)
    return df
