#!/usr/bin/env python3
"""
RSI (Relative Strength Index) стратегия
- RSI < 30 → перепродан → сигнал на покупку
- RSI > 70 → перекуплен → сигнал на продажу
"""
import pandas as pd
import ta
from typing import Dict, List, Optional
import time


class RSIStrategy:
    """RSI торговая стратегия"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 position_size_pct: float = 10):  # % от баланса на сделку
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size_pct = position_size_pct
        
        self.name = "RSI Strategy"
        self.description = f"RSI({rsi_period}): Buy<{rsi_oversold}, Sell>{rsi_overbought}"
    
    def calculate_rsi(self, klines: List) -> pd.Series:
        """Рассчитать RSI из свечей MEXC"""
        # MEXC klines format: [open_time, open, high, low, close, volume, close_time, quote_volume]
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        
        # Используем библиотеку ta для расчёта RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period)
        
        return rsi.rsi()
    
    def analyze(self, klines: List, current_position: Optional[Dict] = None) -> Dict:
        """
        Анализ и генерация сигнала
        
        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'rsi': float,
                'reason': str,
                'confidence': float  # 0-100
            }
        """
        if len(klines) < self.rsi_period + 1:
            return {
                'signal': 'HOLD',
                'rsi': None,
                'reason': 'Недостаточно данных для RSI',
                'confidence': 0
            }
        
        rsi_series = self.calculate_rsi(klines)
        current_rsi = rsi_series.iloc[-1]
        
        # Определяем сигнал
        signal = 'HOLD'
        reason = f'RSI={current_rsi:.1f}'
        confidence = 0
        
        if current_rsi < self.rsi_oversold and not current_position:
            signal = 'BUY'
            reason = f'RSI={current_rsi:.1f} < {self.rsi_oversold} (перепродан)'
            # Чем ниже RSI, тем выше уверенность
            confidence = min(100, (self.rsi_oversold - current_rsi) * 3)
        
        elif current_rsi > self.rsi_overbought and current_position:
            signal = 'SELL'
            reason = f'RSI={current_rsi:.1f} > {self.rsi_overbought} (перекуплен)'
            # Чем выше RSI, тем выше уверенность
            confidence = min(100, (current_rsi - self.rsi_overbought) * 3)
        
        return {
            'signal': signal,
            'rsi': float(current_rsi),
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
            'position_size_pct': self.position_size_pct,
        }


def test_strategy():
    """Тест стратегии"""
    import sys
    sys.path.append('..')
    from mexc_client import MEXCClient
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = MEXCClient(
        api_key=os.getenv('MEXC_API_KEY'),
        secret_key=os.getenv('MEXC_SECRET_KEY'),
        demo_mode=True
    )
    
    strategy = RSIStrategy()
    
    print("\n=== Тест RSI Стратегии ===\n")
    print(f"Стратегия: {strategy.name}")
    print(f"Описание: {strategy.description}\n")
    
    # Получаем исторические данные
    symbol = 'BTCUSDT'
    print(f"Получаю данные {symbol}...")
    klines = client.get_klines(symbol, interval='5m', limit=100)
    
    # Анализируем
    analysis = strategy.analyze(klines)
    
    print(f"\nАнализ:")
    print(f"  Сигнал: {analysis['signal']}")
    print(f"  RSI: {analysis['rsi']:.2f}")
    print(f"  Причина: {analysis['reason']}")
    print(f"  Уверенность: {analysis['confidence']:.1f}%")
    
    # Если есть сигнал на покупку
    if analysis['signal'] == 'BUY':
        current_price = client.get_ticker_price(symbol)
        balance = client.get_balance()
        
        quantity = strategy.calculate_position_size(balance['USDT'], current_price)
        
        print(f"\n💡 Рекомендация:")
        print(f"  Купить {quantity:.6f} BTC по ~${current_price:,.2f}")
        print(f"  Стоимость: ~${quantity * current_price:.2f}")


if __name__ == '__main__':
    test_strategy()
