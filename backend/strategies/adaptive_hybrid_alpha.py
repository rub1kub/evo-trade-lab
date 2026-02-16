#!/usr/bin/env python3
"""
Adaptive Hybrid Alpha strategy
- Technical trend + momentum + crowd psychology + news proxy in one weighted score.
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class AdaptiveHybridAlphaStrategy:
    def __init__(self, position_size_pct: float = 22):
        self.position_size_pct = position_size_pct
        self.name = "Adaptive Hybrid Alpha"
        self.description = "Tech + psychology + news-proxy weighted alpha"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        if len(klines) < 100:
            return {
                'signal': 'HOLD',
                'reason': 'Недостаточно данных',
                'confidence': 0,
            }

        df = pd.DataFrame(
            klines,
            columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume']
        )
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['volume']

        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_hist = macd.macd_diff().iloc[-1]

        ret = close.pct_change().fillna(0)
        ret1 = float(ret.iloc[-1] * 100)
        ret5 = float(close.pct_change(5).iloc[-1] * 100)

        vol_avg = float(vol.iloc[-31:-1].mean()) if len(vol) > 31 else float(vol.mean())
        vol_ratio = float(vol.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

        # news proxy
        rwin = ret.iloc[-61:-1]
        ret_std = float(rwin.std()) if float(rwin.std()) > 1e-9 else 1e-9
        ret_z = float((ret.iloc[-1] - float(rwin.mean())) / ret_std)
        news_score = abs(ret_z) * 0.7 + max(0.0, vol_ratio - 1) * 0.3

        # crowd psychology proxy from wick imbalance
        o = float(df['open'].iloc[-1])
        c = float(close.iloc[-1])
        h = float(high.iloc[-1])
        l = float(low.iloc[-1])
        body = abs(c - o)
        upper = max(0.0, h - max(o, c))
        lower = max(0.0, min(o, c) - l)

        alpha = 0.0
        # technical trend
        alpha += 1.0 if c > ema20 > ema50 else -1.0 if c < ema20 < ema50 else 0.0
        alpha += 0.9 if macd_hist > 0 else -0.9

        # momentum + RSI balance
        if ret5 > 0.5:
            alpha += 0.6
        elif ret5 < -0.5:
            alpha -= 0.6

        if rsi < 33:
            alpha += 0.7
        elif rsi > 72:
            alpha -= 0.7

        # crowd candle behaviour
        if lower > body * 0.8 and ret1 < 0:
            alpha += 0.5
        if upper > body * 0.8 and ret1 > 0:
            alpha -= 0.5

        # news impact direction
        if news_score > 1.8:
            alpha += 0.8 if ret_z > 0 else -0.8

        signal = 'HOLD'
        conf = min(96.0, max(0.0, abs(alpha) * 26 + max(0.0, news_score - 1.2) * 14))
        reason = f"alpha={alpha:.2f}, rsi={rsi:.1f}, macd_hist={float(macd_hist):.4f}, news={news_score:.2f}"

        if not current_position:
            if alpha >= 1.8:
                signal = 'BUY'
                reason = f"Hybrid alpha LONG: {reason}"
            elif alpha <= -2.1:
                # long-only engine: skip short, no SELL without position
                signal = 'HOLD'

        else:
            if alpha <= -1.2:
                signal = 'SELL'
                reason = f"Hybrid alpha exit: {reason}"
            elif rsi > 80 and ret1 < 0:
                signal = 'SELL'
                reason = f"Exhaustion exit: rsi={rsi:.1f}, ret1={ret1:.2f}%"

        return {
            'signal': signal,
            'alpha': float(alpha),
            'rsi': float(rsi),
            'ret1': float(ret1),
            'ret5': float(ret5),
            'news_score': float(news_score),
            'volume_ratio': float(vol_ratio),
            'reason': reason,
            'confidence': float(conf),
        }

    def calculate_position_size(self, balance_usdt: float, current_price: float) -> float:
        amount_usdt = balance_usdt * (self.position_size_pct / 100)
        return amount_usdt / current_price

    def get_config(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'position_size_pct': self.position_size_pct,
        }
