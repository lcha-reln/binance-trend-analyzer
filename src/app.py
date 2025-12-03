#!/usr/bin/env python3
"""
Web服务入口
@author Reln Ding
"""

import os
import sys
from flask import Flask, render_template, jsonify
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import TrendAnalyzer
from config import REFRESH_INTERVAL

app = Flask(__name__, template_folder='templates', static_folder='static')
analyzer = TrendAnalyzer()


@app.route('/')
def index():
    """首页"""
    return render_template('index.html', refresh_interval=REFRESH_INTERVAL)


@app.route('/api/analysis')
def get_analysis():
    """获取分析数据API"""
    try:
        results = analyzer.analyze_all()

        # 转换数据格式
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols': {}
        }

        for symbol, intervals in results.items():
            data['symbols'][symbol] = {
                'intervals': {},
                'resonance': intervals.get('resonance', {})
            }

            for interval, interval_data in intervals.items():
                if interval == 'resonance':
                    continue

                # 简化数据
                data['symbols'][symbol]['intervals'][interval] = {
                    'current_price': interval_data['current_price'],
                    'stats_24h': interval_data['stats_24h'],
                    'prediction': {
                        'direction': interval_data['prediction']['overall_direction'],
                        'confidence': interval_data['prediction']['confidence'],
                        'score': interval_data['prediction']['score'],
                        'volume_weight': interval_data['prediction'].get('volume_weight', 1.0),
                        'change_percent': interval_data['prediction']['linear_change_percent']
                    },
                    'indicators': {
                        'rsi': interval_data['indicators'].get('RSI'),
                        'macd_hist': interval_data['indicators'].get('MACD_Hist')
                    }
                }

        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("启动Web服务: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
