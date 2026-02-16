#!/usr/bin/env python3
"""
Grid Trading (Сеточная торговля)
Размещает сетку лимитных ордеров на покупку/продажу с фиксированным шагом
Прибыльно в боковике (sideways market)
"""
import pandas as pd
from typing import Dict, List, Optional


class GridTradingStrategy:
    """Grid Trading стратегия"""
    
    def __init__(self, 
                 grid_levels: int = 10,
                 grid_range_pct: float = 5.0,
                 position_size_pct: float = 10):
        
        self.grid_levels = grid_levels
        self.grid_range_pct = grid_range_pct  # Диапазон сетки в % от цены
        self.position_size_pct = position_size_pct
        
        self.name = "Grid Trading"
        self.description = f"Сетка {grid_levels} уровней, {grid_range_pct}% диапазон"
        
        # Текущая сетка
        self.grid_orders = []
        self.base_price = None
    
    def generate_grid(self, current_price: float) -> List[Dict]:
        """
        Генерация сетки ордеров
        
        Returns:
            [{'side': 'BUY/SELL', 'price': float, 'quantity': float}]
        """
        if self.base_price is None:
            self.base_price = current_price
        
        grid = []
        
        # Диапазон цен
        lower_bound = current_price * (1 - self.grid_range_pct / 100)
        upper_bound = current_price * (1 + self.grid_range_pct / 100)
        
        # Шаг сетки
        step = (upper_bound - lower_bound) / self.grid_levels
        
        # Генерируем уровни
        for i in range(self.grid_levels):
            price = lower_bound + (i * step)
            
            # Ордера на покупку ниже текущей цены
            if price < current_price:
                grid.append({
                    'side': 'BUY',
                    'price': price,
                    'level': i,
                })
            # Ордера на продажу выше текущей цены
            elif price > current_price:
                grid.append({
                    'side': 'SELL',
                    'price': price,
                    'level': i,
                })
        
        return grid
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """
        Анализ и генерация сигнала
        
        Grid trading не использует классический анализ - вместо этого
        генерирует сетку ордеров
        """
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        df['close'] = pd.to_numeric(df['close'])
        
        current_price = df['close'].iloc[-1]
        
        # Проверяем волатильность - Grid лучше работает в боковике
        volatility = df['close'].pct_change().std() * 100
        
        # Генерируем сетку
        grid = self.generate_grid(current_price)
        
        return {
            'signal': 'GRID',  # Специальный сигнал для grid trading
            'current_price': float(current_price),
            'grid': grid,
            'volatility': float(volatility),
            'reason': f'Grid {len(grid)} ордеров в диапазоне ±{self.grid_range_pct}%',
            'confidence': 70 if volatility < 2 else 40,  # Выше уверенность в низкой волатильности
        }
    
    def calculate_position_size(self, balance_usdt: float, current_price: float) -> float:
        """Рассчитать размер позиции"""
        # Для grid trading делим баланс на количество уровней
        amount_usdt = balance_usdt * (self.position_size_pct / 100) / self.grid_levels
        quantity = amount_usdt / current_price
        return quantity
    
    def get_config(self) -> Dict:
        """Получить конфигурацию стратегии"""
        return {
            'name': self.name,
            'description': self.description,
            'grid_levels': self.grid_levels,
            'grid_range_pct': self.grid_range_pct,
            'position_size_pct': self.position_size_pct,
        }
