#!/usr/bin/env python3
"""
Williams %R strategy.

Williams %R oscillates between 0 and -100.
  - Below -80 → oversold → BUY
  - Above -20 → overbought → SELL
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class WilliamsRStrategy:
    def __init__(self, period: int = 14, oversold: float = -80,
                 overbought: float = -20, position_size_pct: float = 25):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size_pct = position_size_pct
        self.name = "Williams %R"
        self.description = f"W%R({period}): Buy<{oversold}, Sell>{overbought}"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        if len(klines) < self.period + 2:
            return {'signal': 'HOLD', 'williams_r': None, 'reason': 'Not enough data', 'confidence': 0}

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        for col in ['close', 'high', 'low']:
            df[col] = pd.to_numeric(df[col])

        wr = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=self.period)
        wr_val = wr.williams_r().iloc[-1]

        signal = 'HOLD'
        reason = f'W%R={wr_val:.1f}'
        confidence = 0

        if not current_position and wr_val < self.oversold:
            signal = 'BUY'
            reason = f'W%R={wr_val:.1f} < {self.oversold} (oversold)'
            confidence = min(95, abs(wr_val - self.oversold) * 2.5)
        elif current_position and wr_val > self.overbought:
            signal = 'SELL'
            reason = f'W%R={wr_val:.1f} > {self.overbought} (overbought)'
            confidence = min(95, abs(wr_val - self.overbought) * 2.5)

        return {'signal': signal, 'williams_r': float(wr_val), 'reason': reason, 'confidence': float(confidence)}

    def get_config(self) -> Dict:
        return {
            'name': self.name, 'period': self.period,
            'oversold': self.oversold, 'overbought': self.overbought,
        }
