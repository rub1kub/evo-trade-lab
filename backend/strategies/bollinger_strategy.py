#!/usr/bin/env python3
"""
Bollinger Bands стратегия
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class BollingerStrategy:
    """Bollinger Bands торговая стратегия"""
    
    def __init__(self, 
                 period: int = 20,
                 std_dev: float = 2.0,
                 position_size_pct: float = 10):
        
        self.period = period
        self.std_dev = std_dev
        self.position_size_pct = position_size_pct
        
        self.name = "Bollinger Bands"
        self.description = f"BB({period}, {std_dev}σ)"
    
    def calculate_bollinger(self, klines: List) -> tuple:
        """Рассчитать Bollinger Bands"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        
        bb_indicator = ta.volatility.BollingerBands(
            df['close'],
            window=self.period,
            window_dev=self.std_dev
        )
        
        upper = bb_indicator.bollinger_hband()
        middle = bb_indicator.bollinger_mavg()
        lower = bb_indicator.bollinger_lband()
        
        return upper, middle, lower
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < self.period:
            return {
                'signal': 'HOLD',
                'bb_upper': None,
                'reason': 'Недостаточно данных для Bollinger Bands',
                'confidence': 0
            }
        
        upper, middle, lower = self.calculate_bollinger(klines)
        
        current_price = float(klines[-1][4])  # close price
        
        current_upper = upper.iloc[-1]
        current_middle = middle.iloc[-1]
        current_lower = lower.iloc[-1]
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'Цена={current_price:.2f}, BB={current_lower:.2f}-{current_upper:.2f}'
        confidence = 0
        
        # Цена касается нижней границы (перепродан)
        if current_price <= current_lower and not current_position:
            signal_type = 'BUY'
            reason = f'Цена у нижней границы BB (перепродан)'
            distance = abs(current_price - current_lower)
            confidence = min(100, (distance / current_lower) * 1000)
        
        # Цена касается верхней границы (перекуплен)
        elif current_price >= current_upper and current_position:
            signal_type = 'SELL'
            reason = f'Цена у верхней границы BB (перекуплен)'
            distance = abs(current_price - current_upper)
            confidence = min(100, (distance / current_upper) * 1000)
        
        return {
            'signal': signal_type,
            'bb_upper': float(current_upper),
            'bb_middle': float(current_middle),
            'bb_lower': float(current_lower),
            'price': current_price,
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
            'period': self.period,
            'std_dev': self.std_dev,
            'position_size_pct': self.position_size_pct,
        }
