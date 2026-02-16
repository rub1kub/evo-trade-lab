#!/usr/bin/env python3
"""
Momentum / Rate of Change (ROC) strategy.

Buy when ROC crosses above threshold (momentum picking up).
Sell when ROC crosses below negative threshold (momentum fading).
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class MomentumROCStrategy:
    def __init__(self, roc_period: int = 12, buy_threshold: float = 1.5,
                 sell_threshold: float = -1.5, position_size_pct: float = 25):
        self.roc_period = roc_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_size_pct = position_size_pct
        self.name = "Momentum ROC"
        self.description = f"ROC({roc_period}): Buy>{buy_threshold}%, Sell<{sell_threshold}%"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        if len(klines) < self.roc_period + 2:
            return {'signal': 'HOLD', 'roc': None, 'reason': 'Not enough data', 'confidence': 0}

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        df['close'] = pd.to_numeric(df['close'])

        roc = ta.momentum.ROCIndicator(df['close'], window=self.roc_period)
        roc_val = roc.roc().iloc[-1]
        roc_prev = roc.roc().iloc[-2]

        signal = 'HOLD'
        reason = f'ROC={roc_val:.2f}%'
        confidence = 0

        if not current_position and roc_val > self.buy_threshold and roc_prev <= self.buy_threshold:
            signal = 'BUY'
            reason = f'ROC crossed up {roc_val:.2f}% > {self.buy_threshold}%'
            confidence = min(95, abs(roc_val) * 8)
        elif current_position and roc_val < self.sell_threshold and roc_prev >= self.sell_threshold:
            signal = 'SELL'
            reason = f'ROC crossed down {roc_val:.2f}% < {self.sell_threshold}%'
            confidence = min(95, abs(roc_val) * 8)

        return {'signal': signal, 'roc': float(roc_val), 'reason': reason, 'confidence': float(confidence)}

    def get_config(self) -> Dict:
        return {
            'name': self.name, 'roc_period': self.roc_period,
            'buy_threshold': self.buy_threshold, 'sell_threshold': self.sell_threshold,
        }
