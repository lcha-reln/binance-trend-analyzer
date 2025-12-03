#!/usr/bin/env python3
"""
币安交易对趋势分析与预测系统
@author Reln Ding

功能:
- 获取BTC-USDT, ETH-USDT的K线数据
- 计算多种技术指标 (MA, RSI, MACD, BOLL等)
- 预测价格走势和涨跌方向
- 持续运行，定时刷新

使用方法:
    python main.py
"""

import sys
import time
import signal
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import TrendAnalyzer
from config import REFRESH_INTERVAL, SYMBOLS, INTERVALS


class BinanceTrendMonitor:
    """币安趋势监控器"""

    def __init__(self):
        self.analyzer = TrendAnalyzer()
        self.running = True

    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n\n正在退出...")
        self.running = False

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """打印头部信息"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        币安趋势分析与预测系统                                 ║
║                     Binance Trend Analyzer & Predictor                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  交易对: {symbols:<66} ║
║  周  期: {intervals:<66} ║
║  刷新间隔: {refresh}秒                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """.format(
            symbols=", ".join(SYMBOLS),
            intervals=", ".join(INTERVALS),
            refresh=REFRESH_INTERVAL
        ))

    def run(self):
        """运行监控"""
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print("正在启动币安趋势分析系统...")
        print(f"监控交易对: {', '.join(SYMBOLS)}")
        print(f"K线周期: {', '.join(INTERVALS)}")
        print(f"刷新间隔: {REFRESH_INTERVAL}秒")
        print("\n按 Ctrl+C 退出\n")

        while self.running:
            try:
                # 清屏
                self.clear_screen()

                # 打印头部
                self.print_header()

                # 执行分析
                print("正在获取数据并分析...\n")
                results = self.analyzer.analyze_all()

                # 格式化输出
                output = self.analyzer.format_output(results)
                print(output)

                # 倒计时
                print(f"\n下次刷新倒计时: ", end="", flush=True)
                for i in range(REFRESH_INTERVAL, 0, -1):
                    if not self.running:
                        break
                    print(f"\r下次刷新倒计时: {i}秒  ", end="", flush=True)
                    time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[ERROR] 发生错误: {e}")
                print("5秒后重试...")
                time.sleep(5)

        print("\n系统已退出。")


def main():
    """主函数"""
    monitor = BinanceTrendMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
