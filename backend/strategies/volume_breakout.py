#!/usr/bin/env python3
"""
Volume Breakout стратегия - торговля на прорывах с высоким объёмом
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class VolumeBreakoutStrategy:
    """Стратегия прорыва с объёмом"""
    
    def __init__(self, 
                 volume_multiplier: float = 2.0,
                 price_threshold_pct: float = 1.0,
                 position_size_pct: float = 10):
        
        self.volume_multiplier = volume_multiplier
        self.price_threshold_pct = price_threshold_pct
        self.position_size_pct = position_size_pct
        
        self.name = "Volume Breakout"
        self.description = f"Прорыв x{volume_multiplier} объём, {price_threshold_pct}% цена"
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < 50:
            return {
                'signal': 'HOLD',
                'volume_ratio': None,
                'reason': 'Недостаточно данных',
                'confidence': 0
            }
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Средний объём за последние 20 свечей
        avg_volume = df['volume'].iloc[-21:-1].mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Изменение цены
        prev_close = df['close'].iloc[-2]
        current_close = df['close'].iloc[-1]
        price_change_pct = ((current_close - prev_close) / prev_close) * 100
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'Volume ratio={volume_ratio:.2f}, Price change={price_change_pct:.2f}%'
        confidence = 0
        
        # Бычий прорыв: высокий объём + рост цены
        if volume_ratio >= self.volume_multiplier and price_change_pct >= self.price_threshold_pct and not current_position:
            signal_type = 'BUY'
            reason = f'Бычий прорыв: объём x{volume_ratio:.1f}, цена +{price_change_pct:.2f}%'
            confidence = min(100, volume_ratio * 20)
        
        # Медвежий прорыв: высокий объём + падение цены (закрываем позицию)
        elif volume_ratio >= self.volume_multiplier and price_change_pct <= -self.price_threshold_pct and current_position:
            signal_type = 'SELL'
            reason = f'Медвежий прорыв: объём x{volume_ratio:.1f}, цена {price_change_pct:.2f}%'
            confidence = min(100, volume_ratio * 20)
        
        return {
            'signal': signal_type,
            'volume_ratio': float(volume_ratio),
            'price_change_pct': float(price_change_pct),
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
            'volume_multiplier': self.volume_multiplier,
            'price_threshold_pct': self.price_threshold_pct,
            'position_size_pct': self.position_size_pct,
        }
