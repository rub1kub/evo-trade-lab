#!/usr/bin/env python3
"""
Ichimoku Cloud strategy (simplified for 1m/5m scalping).

BUY  — price crosses above the cloud (Senkou Span A > Senkou Span B, price > A).
SELL — price drops below the cloud.

Uses Tenkan/Kijun cross as confidence booster.
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class IchimokuStrategy:
    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
                 position_size_pct: float = 25):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.position_size_pct = position_size_pct
        self.name = "Ichimoku Cloud"
        self.description = f"Ichimoku({tenkan}/{kijun}/{senkou_b})"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        need = max(self.senkou_b, 52) + 5
        if len(klines) < need:
            return {'signal': 'HOLD', 'reason': 'Not enough data', 'confidence': 0}

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        for col in ['close', 'high', 'low']:
            df[col] = pd.to_numeric(df[col])

        ichi = ta.trend.IchimokuIndicator(
            df['high'], df['low'],
            window1=self.tenkan, window2=self.kijun, window3=self.senkou_b
        )

        tenkan_val = ichi.ichimoku_conversion_line().iloc[-1]
        kijun_val = ichi.ichimoku_base_line().iloc[-1]
        span_a = ichi.ichimoku_a().iloc[-1]
        span_b = ichi.ichimoku_b().iloc[-1]
        price = df['close'].iloc[-1]

        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)

        signal = 'HOLD'
        reason = f'Price={price:.4f} Cloud=[{cloud_bot:.4f}..{cloud_top:.4f}]'
        confidence = 0

        above_cloud = price > cloud_top
        below_cloud = price < cloud_bot
        tk_cross_up = tenkan_val > kijun_val

        if not current_position and above_cloud:
            signal = 'BUY'
            dist_pct = ((price - cloud_top) / cloud_top) * 100 if cloud_top else 0
            confidence = min(90, 30 + dist_pct * 15)
            if tk_cross_up:
                confidence = min(95, confidence + 15)
            reason = f'Above cloud +{dist_pct:.2f}%, TK {"bull" if tk_cross_up else "bear"}'

        elif current_position and below_cloud:
            signal = 'SELL'
            dist_pct = ((cloud_bot - price) / cloud_bot) * 100 if cloud_bot else 0
            confidence = min(90, 30 + dist_pct * 15)
            reason = f'Below cloud -{dist_pct:.2f}%'

        return {'signal': signal, 'reason': reason, 'confidence': float(confidence)}

    def get_config(self) -> Dict:
        return {
            'name': self.name, 'tenkan': self.tenkan,
            'kijun': self.kijun, 'senkou_b': self.senkou_b,
        }
