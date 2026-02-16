#!/usr/bin/env python3
"""
Crowd Psychology strategy
- Uses technical indicators + crowd behaviour proxies (panic/euphoria, wick imbalance, volume impulse).
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class CrowdPsychologyStrategy:
    def __init__(
        self,
        rsi_period: int = 14,
        panic_drop_pct: float = -0.9,
        euphoria_pump_pct: float = 1.0,
        volume_spike: float = 1.8,
        position_size_pct: float = 20,
    ):
        self.rsi_period = rsi_period
        self.panic_drop_pct = panic_drop_pct
        self.euphoria_pump_pct = euphoria_pump_pct
        self.volume_spike = volume_spike
        self.position_size_pct = position_size_pct

        self.name = "Crowd Psychology"
        self.description = "Panic/euphoria crowd behavior + technical confirmation"

    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        if len(klines) < 80:
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

        rsi = ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi().iloc[-1]
        ema_fast = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
        ema_slow = ta.trend.EMAIndicator(close, window=55).ema_indicator().iloc[-1]

        ret1 = close.pct_change().iloc[-1] * 100
        ret5 = close.pct_change(5).iloc[-1] * 100

        avg_vol = vol.iloc[-21:-1].mean() if len(vol) > 21 else vol.mean()
        vol_ratio = (vol.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else 1.0

        # Candle psychology (wick dominance)
        c_open = float(df['open'].iloc[-1])
        c_close = float(close.iloc[-1])
        c_high = float(high.iloc[-1])
        c_low = float(low.iloc[-1])
        body = abs(c_close - c_open)
        upper_wick = max(0.0, c_high - max(c_open, c_close))
        lower_wick = max(0.0, min(c_open, c_close) - c_low)

        trend_up = (c_close > ema_fast > ema_slow)
        trend_down = (c_close < ema_fast < ema_slow)

        panic = ret1 <= self.panic_drop_pct and vol_ratio >= self.volume_spike and rsi < 38
        euphoria = ret1 >= self.euphoria_pump_pct and vol_ratio >= self.volume_spike and rsi > 66

        signal = 'HOLD'
        confidence = 0.0
        reason = f"rsi={rsi:.1f}, ret1={ret1:.2f}%, volx={vol_ratio:.2f}"

        if not current_position:
            # Panic rebound: crowd fear capitulation + long lower wick (buyers absorb)
            if panic and lower_wick > body * 0.8 and not trend_down:
                signal = 'BUY'
                confidence = min(95.0, 55 + abs(ret1) * 10 + (vol_ratio - 1) * 12)
                reason = f"Паника толпы: спад {ret1:.2f}% + объём x{vol_ratio:.2f}, выкуп снизу"

            # Controlled trend continuation (greed but not full euphoria)
            elif trend_up and ret5 > 0.6 and 45 <= rsi <= 72 and vol_ratio > 1.1:
                signal = 'BUY'
                confidence = min(90.0, 35 + ret5 * 6 + (vol_ratio - 1) * 15)
                reason = f"Тренд+психология: ret5={ret5:.2f}%, rsi={rsi:.1f}, vol x{vol_ratio:.2f}"

        else:
            # Euphoria exhaustion: long upper wick + overheated RSI
            if euphoria and (upper_wick > body * 1.0 or rsi > 78):
                signal = 'SELL'
                confidence = min(95.0, 45 + (rsi - 65) * 1.5 + (vol_ratio - 1) * 10)
                reason = f"Эйфория толпы/фиксация: rsi={rsi:.1f}, wick↑, vol x{vol_ratio:.2f}"
            elif trend_down and ret1 < -0.3:
                signal = 'SELL'
                confidence = min(85.0, 35 + abs(ret1) * 10)
                reason = f"Слом тренда: ret1={ret1:.2f}%"

        return {
            'signal': signal,
            'rsi': float(rsi),
            'ret1': float(ret1),
            'ret5': float(ret5),
            'volume_ratio': float(vol_ratio),
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
            'rsi_period': self.rsi_period,
            'panic_drop_pct': self.panic_drop_pct,
            'euphoria_pump_pct': self.euphoria_pump_pct,
            'volume_spike': self.volume_spike,
            'position_size_pct': self.position_size_pct,
        }
