"""
配置模块
@author Reln Ding
"""

# 币安API配置（公开接口无需API Key）
BINANCE_BASE_URL = "https://api.binance.com"

# 交易对配置
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# K线周期配置
INTERVALS = ["5m", "30m", "1h", "4h", "1d"]

# 数据获取配置
KLINE_LIMIT = 1000  # 获取最近1000根K线用于分析

# 技术指标参数
MA_PERIODS = [7, 25, 99]  # 移动平均线周期
RSI_PERIOD = 14  # RSI周期
MACD_FAST = 12  # MACD快线
MACD_SLOW = 26  # MACD慢线
MACD_SIGNAL = 9  # MACD信号线
BOLL_PERIOD = 20  # 布林带周期
BOLL_STD = 2  # 布林带标准差倍数

# 预测模型配置
PREDICTION_PERIODS = 5  # 预测未来5根K线

# 刷新间隔（秒）
REFRESH_INTERVAL = 60  # 每60秒刷新一次
