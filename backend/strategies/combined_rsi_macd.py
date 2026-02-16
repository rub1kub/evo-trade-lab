#!/usr/bin/env python3
"""
Комбинированная стратегия: RSI + MACD
Входим когда оба индикатора дают сигнал
"""
import pandas as pd
import ta
from typing import Dict, List, Optional


class RSI_MACD_Strategy:
    """RSI + MACD комбинированная стратегия"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 position_size_pct: float = 10):
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.position_size_pct = position_size_pct
        
        self.name = "RSI+MACD Combo"
        self.description = f"RSI({rsi_period}) + MACD({macd_fast},{macd_slow},{macd_signal})"
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """Анализ и генерация сигнала"""
        if len(klines) < max(self.macd_slow, self.rsi_period) + 10:
            return {
                'signal': 'HOLD',
                'rsi': None,
                'reason': 'Недостаточно данных',
                'confidence': 0
            }
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        df['close'] = pd.to_numeric(df['close'])
        
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period)
        rsi = rsi_indicator.rsi()
        current_rsi = rsi.iloc[-1]
        
        # MACD
        macd_indicator = ta.trend.MACD(
            df['close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        macd = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]
        
        # Определяем сигнал
        signal_type = 'HOLD'
        reason = f'RSI={current_rsi:.1f}, MACD histogram={current_histogram:.2f}'
        confidence = 0
        
        # ОБА индикатора дают BUY
        rsi_buy = current_rsi < self.rsi_oversold
        macd_buy = prev_histogram < 0 and current_histogram > 0
        
        if rsi_buy and macd_buy and not current_position:
            signal_type = 'BUY'
            reason = f'Двойной сигнал: RSI={current_rsi:.1f} (перепродан) + MACD пересечение вверх'
            confidence = 80  # Высокая уверенность при двойном подтверждении
        
        # ОБА индикатора дают SELL
        rsi_sell = current_rsi > self.rsi_overbought
        macd_sell = prev_histogram > 0 and current_histogram < 0
        
        if rsi_sell and macd_sell and current_position:
            signal_type = 'SELL'
            reason = f'Двойной сигнал: RSI={current_rsi:.1f} (перекуплен) + MACD пересечение вниз'
            confidence = 80
        
        # Один из индикаторов (меньшая уверенность)
        elif (rsi_buy or macd_buy) and not current_position:
            signal_type = 'BUY'
            reason = f'Одиночный сигнал: RSI={current_rsi:.1f}, MACD hist={current_histogram:.2f}'
            confidence = 40
        
        elif (rsi_sell or macd_sell) and current_position:
            signal_type = 'SELL'
            reason = f'Одиночный сигнал: RSI={current_rsi:.1f}, MACD hist={current_histogram:.2f}'
            confidence = 40
        
        return {
            'signal': signal_type,
            'rsi': float(current_rsi),
            'macd_histogram': float(current_histogram),
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
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'position_size_pct': self.position_size_pct,
        }
