# Binance Trend Analyzer

币安交易对趋势分析与预测系统，支持多周期共振分析和成交量加权评分。

## 功能特性

- **多周期分析**: 5分钟、30分钟、1小时、4小时、日线
- **技术指标**: MA、RSI、MACD、布林带、ATR
- **成交量加权**: 放量信号权重提升，缩量信号权重降低
- **多周期共振**: 判断各周期方向一致性，给出综合置信度
- **Web界面**: 实时刷新的可视化面板

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/binance-trend-analyzer.git
cd binance-trend-analyzer
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行

**Web界面模式（推荐）:**

```bash
python3 src/app.py
```

访问 http://127.0.0.1:5000

**命令行模式:**

```bash
python3 src/main.py
```

## 配置说明

编辑 `src/config.py` 自定义配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| SYMBOLS | BTCUSDT, ETHUSDT | 监控的交易对 |
| INTERVALS | 5m, 30m, 1h, 4h, 1d | K线周期 |
| KLINE_LIMIT | 1000 | 获取K线数量 |
| REFRESH_INTERVAL | 60 | 刷新间隔(秒) |

## 信号解读

### 置信度等级

| 置信度 | 说明 |
|--------|------|
| 极高 | 所有周期完全共振，可参考 |
| 高 | 75%以上周期同向，可参考 |
| 中 | 50%周期同向，谨慎参考 |
| 低 | 信号分歧，建议观望 |

### 成交量权重

| 成交量比率 | 权重 | 说明 |
|------------|------|------|
| > 2.0 | 1.5x | 大幅放量 |
| > 1.5 | 1.3x | 明显放量 |
| < 0.5 | 0.7x | 缩量 |

## 项目结构

```
binance-trend-analyzer/
├── requirements.txt      # 依赖
├── README.md
└── src/
    ├── config.py         # 配置
    ├── collector.py      # 数据采集
    ├── indicators.py     # 技术指标
    ├── predictor.py      # 预测模型
    ├── analyzer.py       # 趋势分析
    ├── main.py           # 命令行入口
    ├── app.py            # Web服务入口
    └── templates/
        └── index.html    # Web页面
```

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。加密货币市场波动剧烈，投资有风险，入市需谨慎。

## License

MIT
