#!/usr/bin/env python3
"""
News Flow Proxy strategy
- Real news API is optional; this strategy uses "news shock" proxy from abnormal return + volume spike.
- Useful for fast reaction to market-moving events in demo mode.
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class NewsFlowProxyStrategy:
    def __init__(
        self,
        shock_z_threshold: float = 2.2,
        reverse_z_threshold: float = 1.4,
        position_size_pct: float = 25,
    ):
        self.shock_z_threshold = shock_z_threshold
        self.reverse_z_threshold = reverse_z_threshold
        self.position_size_pct = position_size_pct

        self.name = "News Flow Proxy"
        self.description = "Abnormal return+volume news proxy with trend confirmation"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        if len(klines) < 90:
            return {
                'signal': 'HOLD',
                'reason': 'Недостаточно данных',
                'confidence': 0,
            }

        df = pd.DataFrame(
            klines,
            columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume']
        )
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        close = df['close']
        volume = df['volume']

        ret = close.pct_change().fillna(0)
        ret_now = float(ret.iloc[-1] * 100)

        # z-scores on recent window
        r_window = ret.iloc[-61:-1]
        v_window = volume.iloc[-61:-1]

        ret_mean = float(r_window.mean())
        ret_std = float(r_window.std()) if float(r_window.std()) > 1e-9 else 1e-9
        vol_mean = float(v_window.mean())
        vol_std = float(v_window.std()) if float(v_window.std()) > 1e-9 else 1e-9

        ret_z = float((ret.iloc[-1] - ret_mean) / ret_std)
        vol_z = float((volume.iloc[-1] - vol_mean) / vol_std)

        news_score = abs(ret_z) * 0.65 + max(0.0, vol_z) * 0.35
        direction = 1 if ret_z > 0 else -1 if ret_z < 0 else 0

        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

        signal = 'HOLD'
        confidence = 0.0
        reason = f"news_proxy={news_score:.2f}, ret_z={ret_z:.2f}, vol_z={vol_z:.2f}"

        if not current_position:
            if news_score >= self.shock_z_threshold and direction > 0 and close.iloc[-1] > ema21:
                signal = 'BUY'
                confidence = min(95.0, 40 + news_score * 18)
                reason = f"Позитивный news-shock proxy: score={news_score:.2f}, ret={ret_now:.2f}%"

            # capitulation reversal as "bad news already priced"
            elif news_score >= (self.shock_z_threshold + 0.3) and direction < 0 and rsi < 30:
                signal = 'BUY'
                confidence = min(90.0, 35 + news_score * 15)
                reason = f"Капитуляция/отскок: score={news_score:.2f}, rsi={rsi:.1f}"

        else:
            if direction < 0 and news_score >= self.reverse_z_threshold:
                signal = 'SELL'
                confidence = min(95.0, 35 + news_score * 20)
                reason = f"Негативный news-shock proxy: score={news_score:.2f}, ret={ret_now:.2f}%"
            elif rsi > 79 and ret_z < 0:
                signal = 'SELL'
                confidence = min(88.0, 45 + (rsi - 70) * 2)
                reason = f"Перегрев после news-impulse: rsi={rsi:.1f}"

        return {
            'signal': signal,
            'news_score': float(news_score),
            'ret_z': float(ret_z),
            'vol_z': float(vol_z),
            'rsi': float(rsi),
            'reason': reason,
            'confidence': float(confidence),
        }

    def calculate_position_size(self, balance_usdt: float, current_price: float) -> float:
        amount_usdt = balance_usdt * (self.position_size_pct / 100)
        return amount_usdt / current_price

    def get_config(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'shock_z_threshold': self.shock_z_threshold,
            'reverse_z_threshold': self.reverse_z_threshold,
            'position_size_pct': self.position_size_pct,
        }
