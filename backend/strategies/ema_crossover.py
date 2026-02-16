#!/usr/bin/env python3
"""
EMA Crossover стратегия
Быстрая EMA пересекает медленную снизу вверх → BUY
Быстрая EMA пересекает медленную сверху вниз → SELL
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class EMACrossoverStrategy:
    """EMA Crossover торговая стратегия"""
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 position_size_pct: float = 10):
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size_pct = position_size_pct
        
        self.name = "EMA Crossover"
        self.description = f"EMA({fast_period}/{slow_period})"
    
    def calculate_ema(self, klines: List) -> tuple:
        """Рассчитать EMA"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        
        fast_ema = ta.trend.EMAIndicator(df['close'], window=self.fast_period).ema_indicator()
        slow_ema = ta.trend.EMAIndicator(df['close'], window=self.slow_period).ema_indicator()
        
        return fast_ema, slow_ema
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < self.slow_period:
            return {
                'signal': 'HOLD',
                'fast_ema': None,
                'reason': 'Недостаточно данных для EMA',
                'confidence': 0
            }
        
        fast_ema, slow_ema = self.calculate_ema(klines)
        
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'Fast EMA={current_fast:.2f}, Slow EMA={current_slow:.2f}'
        confidence = 0
        
        # Пересечение снизу вверх (золотой крест)
        if prev_fast <= prev_slow and current_fast > current_slow and not current_position:
            signal_type = 'BUY'
            reason = f'Золотой крест: Fast EMA пересекла Slow EMA снизу вверх'
            distance = abs(current_fast - current_slow)
            confidence = min(100, (distance / current_slow) * 1000)
        
        # Пересечение сверху вниз (мёртвый крест)
        elif prev_fast >= prev_slow and current_fast < current_slow and current_position:
            signal_type = 'SELL'
            reason = f'Мёртвый крест: Fast EMA пересекла Slow EMA сверху вниз'
            distance = abs(current_fast - current_slow)
            confidence = min(100, (distance / current_slow) * 1000)
        
        return {
            'signal': signal_type,
            'fast_ema': float(current_fast),
            'slow_ema': float(current_slow),
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
            'position_size_pct': self.position_size_pct,
        }
