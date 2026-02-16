#!/usr/bin/env python3
"""
DCA (Dollar Cost Averaging) стратегия
Усреднение позиции при просадке - докупаем дешевле, снижая среднюю цену входа
"""
import pandas as pd
from typing import Dict, List, Optional


class DCAStrategy:
    """DCA стратегия с усреднением"""
    
    def __init__(self, 
                 initial_buy_threshold: float = 30,  # RSI для первой покупки
                 dca_levels: int = 3,  # Количество докупок
                 dca_step_pct: float = 2.0,  # Шаг докупки (при -2%, -4%, -6%)
                 position_size_pct: float = 10):
        
        self.initial_buy_threshold = initial_buy_threshold
        self.dca_levels = dca_levels
        self.dca_step_pct = dca_step_pct
        self.position_size_pct = position_size_pct
        
        self.name = "DCA Strategy"
        self.description = f"Усреднение {dca_levels} уровней по {dca_step_pct}%"
        
        # Состояние
        self.dca_orders = []  # История докупок
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < 20:
            return {
                'signal': 'HOLD',
                'reason': 'Недостаточно данных',
                'confidence': 0
            }
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        df['close'] = pd.to_numeric(df['close'])
        
        current_price = df['close'].iloc[-1]
        
        # RSI для определения зон перепроданности
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'RSI={current_rsi:.1f}'
        confidence = 0
        
        # Первая покупка (нет позиции)
        if not current_position and current_rsi < self.initial_buy_threshold:
            signal_type = 'BUY'
            reason = f'Первая покупка: RSI={current_rsi:.1f} < {self.initial_buy_threshold}'
            confidence = 70
        
        # Докупка (есть позиция и просадка)
        elif current_position:
            entry_price = current_position.get('entry_price', current_price)
            drawdown_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Определяем уровень докупки
            dca_count = len(self.dca_orders)
            next_dca_level = -(dca_count + 1) * self.dca_step_pct
            
            if drawdown_pct <= next_dca_level and dca_count < self.dca_levels:
                signal_type = 'DCA_BUY'
                reason = f'Докупка {dca_count + 1}: просадка {drawdown_pct:.2f}%'
                confidence = 80
            
            # Выход (RSI > 70 = перекуплен)
            elif current_rsi > 70:
                signal_type = 'SELL'
                reason = f'Выход: RSI={current_rsi:.1f} > 70 (перекуплен)'
                confidence = 70
        
        return {
            'signal': signal_type,
            'rsi': float(current_rsi),
            'price': float(current_price),
            'reason': reason,
            'confidence': float(confidence),
            'dca_count': len(self.dca_orders),
        }
    
    def record_dca(self, price: float, quantity: float):
        """Записать докупку"""
        self.dca_orders.append({
            'price': price,
            'quantity': quantity,
            'timestamp': pd.Timestamp.now(),
        })
    
    def reset_dca(self):
        """Сбросить историю докупок после продажи"""
        self.dca_orders = []
    
    def calculate_average_entry(self) -> float:
        """Рассчитать среднюю цену входа с учётом докупок"""
        if not self.dca_orders:
            return 0
        
        total_cost = sum(o['price'] * o['quantity'] for o in self.dca_orders)
        total_quantity = sum(o['quantity'] for o in self.dca_orders)
        
        return total_cost / total_quantity if total_quantity > 0 else 0
    
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
            'initial_buy_threshold': self.initial_buy_threshold,
            'dca_levels': self.dca_levels,
            'dca_step_pct': self.dca_step_pct,
            'position_size_pct': self.position_size_pct,
        }
