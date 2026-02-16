#!/usr/bin/env python3
"""
MACD (Moving Average Convergence Divergence) стратегия
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class MACDStrategy:
    """MACD торговая стратегия"""
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 position_size_pct: float = 10):
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_size_pct = position_size_pct
        
        self.name = "MACD Strategy"
        self.description = f"MACD({fast_period},{slow_period},{signal_period})"
    
    def calculate_macd(self, klines: List) -> tuple:
        """Рассчитать MACD"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        
        macd_indicator = ta.trend.MACD(
            df['close'],
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period
        )
        
        macd = macd_indicator.macd()
        signal = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        return macd, signal, histogram
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < self.slow_period + self.signal_period:
            return {
                'signal': 'HOLD',
                'macd': None,
                'reason': 'Недостаточно данных для MACD',
                'confidence': 0
            }
        
        macd, signal, histogram = self.calculate_macd(klines)
        
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'MACD={current_macd:.2f}'
        confidence = 0
        
        # Пересечение снизу вверх (бычий сигнал)
        if prev_histogram < 0 and current_histogram > 0 and not current_position:
            signal_type = 'BUY'
            reason = f'MACD пересекла сигнальную линию снизу вверх (бычий сигнал)'
            confidence = min(100, abs(current_histogram) * 50)
        
        # Пересечение сверху вниз (медвежий сигнал)
        elif prev_histogram > 0 and current_histogram < 0 and current_position:
            signal_type = 'SELL'
            reason = f'MACD пересекла сигнальную линию сверху вниз (медвежий сигнал)'
            confidence = min(100, abs(current_histogram) * 50)
        
        return {
            'signal': signal_type,
            'macd': float(current_macd),
            'signal_line': float(current_signal),
            'histogram': float(current_histogram),
            'reason': reason,
            'confidence': float(confidence)
        }
    
    def calculate_position_size(self, balance_usdt: float, current_price: float) -> float:
        """Рассчитать размер позиции"""
        amount_usdt = balance_usdt * (self.position_size_pct / 100)
        quantity = amount_usdt / current_price
        return quantity
    
    def get_config(self) -> Dict:
        """Получить конфигурацию стратегии"""
        return {
            'name': self.name,
            'description': self.description,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'position_size_pct': self.position_size_pct,
        }
