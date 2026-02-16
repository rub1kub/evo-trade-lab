#!/usr/bin/env python3
"""
Режимы торговли: Conservative, Balanced, Aggressive, Degen
"""
from typing import Dict


class TradingMode:
    """Базовый класс режима торговли"""
    
    def __init__(self, name: str, description: str, 
                 position_size_pct: float, 
                 min_confidence: float,
                 take_profit_pct: float,
                 stop_loss_pct: float):
        self.name = name
        self.description = description
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence  # Минимальная уверенность для входа
        self.take_profit_pct = take_profit_pct  # % прибыли для фиксации
        self.stop_loss_pct = stop_loss_pct  # % убытка для выхода
    
    def should_enter(self, signal: str, confidence: float) -> bool:
        """Проверить, стоит ли входить в позицию"""
        if signal == 'HOLD':
            return False
        return confidence >= self.min_confidence
    
    def should_exit(self, entry_price: float, current_price: float, signal: str) -> tuple:
        """Проверить, стоит ли выходить из позиции"""
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Take profit
        if profit_pct >= self.take_profit_pct:
            return True, f'Take profit: +{profit_pct:.2f}%'
        
        # Stop loss
        if profit_pct <= -self.stop_loss_pct:
            return True, f'Stop loss: {profit_pct:.2f}%'
        
        # Сигнал на продажу
        if signal == 'SELL':
            return True, f'Сигнал на продажу (прибыль {profit_pct:.2f}%)'
        
        return False, None
    
    def get_config(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'position_size_pct': self.position_size_pct,
            'min_confidence': self.min_confidence,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
        }


class ConservativeMode(TradingMode):
    """Консервативный режим: повышенная активность"""
    def __init__(self):
        super().__init__(
            name="Conservative",
            description="Умеренный риск, но частые входы",
            position_size_pct=12.0,
            min_confidence=35.0,
            take_profit_pct=1.0,
            stop_loss_pct=1.2
        )


class BalancedMode(TradingMode):
    """Сбалансированный режим: агрессивнее"""
    def __init__(self):
        super().__init__(
            name="Balanced",
            description="Умеренно-агрессивный риск и высокая частота",
            position_size_pct=25.0,
            min_confidence=20.0,
            take_profit_pct=1.5,
            stop_loss_pct=1.8
        )


class AggressiveMode(TradingMode):
    """Агрессивный режим: много входов и высокий риск"""
    def __init__(self):
        super().__init__(
            name="Aggressive",
            description="Крупные позиции, много сделок",
            position_size_pct=60.0,
            min_confidence=6.0,
            take_profit_pct=2.0,
            stop_loss_pct=2.4
        )


class BalancedPlusMode(TradingMode):
    """Усиленный balanced: умеренный буст сайза и активности."""
    def __init__(self):
        super().__init__(
            name="BalancedPlus",
            description="Balanced+ для ускоренного роста без полного degen",
            position_size_pct=38.0,
            min_confidence=12.0,
            take_profit_pct=1.6,
            stop_loss_pct=1.9
        )


class DegenMode(TradingMode):
    """Деген режим: максимум риска и частоты"""
    def __init__(self):
        super().__init__(
            name="Degen",
            description="💎🙌 Максимальный риск, почти без фильтра",
            position_size_pct=95.0,
            min_confidence=0.0,
            take_profit_pct=2.5,
            stop_loss_pct=3.5
        )


class ScalpMode(TradingMode):
    """Скальп режим: быстрые сделки на 1m.

    Поднят TP относительно комиссии, чтобы не пилить баланс в ноль.
    """
    def __init__(self):
        super().__init__(
            name="Scalp",
            description="⚡ Скальп: быстрые сделки на 1m",
            position_size_pct=65.0,
            min_confidence=0.0,
            take_profit_pct=0.90,
            stop_loss_pct=0.70
        )


MODES = {
    'conservative': ConservativeMode,
    'balanced': BalancedMode,
    'balanced_plus': BalancedPlusMode,
    'aggressive': AggressiveMode,
    'degen': DegenMode,
    'scalp': ScalpMode,
}


def get_mode(mode_name: str) -> TradingMode:
    """Получить режим торговли по имени"""
    mode_class = MODES.get(mode_name.lower(), BalancedMode)
    return mode_class()
