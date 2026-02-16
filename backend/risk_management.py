#!/usr/bin/env python3
"""
Управление рисками и умные стоп-лоссы
"""
from typing import Dict, Optional
from datetime import datetime


class RiskManager:
    """Менеджер рисков для бота"""
    
    def __init__(self, bot_id: int, config: Dict = None):
        self.bot_id = bot_id
        self.config = config or {}
        
        # Параметры trailing stop
        self.trailing_stop_enabled = self.config.get('trailing_stop_enabled', True)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 1.5)  # %
        self.trailing_activation_pct = self.config.get('trailing_activation_pct', 1.0)  # Активация при +1%
        
        # Параметры break-even stop
        self.breakeven_enabled = self.config.get('breakeven_enabled', True)
        self.breakeven_activation_pct = self.config.get('breakeven_activation_pct', 2.0)  # При +2% переносим в безубыток
        self.breakeven_buffer_pct = self.config.get('breakeven_buffer_pct', 0.5)  # +0.5% от точки входа
        
        # Состояние
        self.highest_price = None  # Максимальная цена после входа (для trailing)
        self.breakeven_activated = False
        
    def update_position(self, position: Dict, current_price: float) -> tuple:
        """
        Обновить позицию и проверить условия выхода
        
        Returns:
            (should_exit: bool, reason: str)
        """
        if not position:
            return False, None
        
        entry_price = position['entry_price']
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Обновляем максимальную цену
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        
        # 1. Trailing Stop
        if self.trailing_stop_enabled and profit_pct >= self.trailing_activation_pct:
            # Trailing stop активирован
            drawdown_from_peak = ((self.highest_price - current_price) / self.highest_price) * 100
            
            if drawdown_from_peak >= self.trailing_stop_pct:
                return True, f'Trailing Stop: откат {drawdown_from_peak:.2f}% от пика ${self.highest_price:.2f}'
        
        # 2. Break-even Stop
        if self.breakeven_enabled and not self.breakeven_activated and profit_pct >= self.breakeven_activation_pct:
            self.breakeven_activated = True
        
        if self.breakeven_activated:
            breakeven_price = entry_price * (1 + self.breakeven_buffer_pct / 100)
            
            if current_price < breakeven_price:
                return True, f'Break-even Stop: защита прибыли при ${breakeven_price:.2f}'
        
        return False, None
    
    def reset(self):
        """Сбросить состояние после закрытия позиции"""
        self.highest_price = None
        self.breakeven_activated = False
    
    def get_config(self) -> Dict:
        """Получить конфигурацию"""
        return {
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_pct': self.trailing_stop_pct,
            'trailing_activation_pct': self.trailing_activation_pct,
            'breakeven_enabled': self.breakeven_enabled,
            'breakeven_activation_pct': self.breakeven_activation_pct,
            'breakeven_buffer_pct': self.breakeven_buffer_pct,
        }


class PortfolioRiskManager:
    """Управление рисками портфеля (все боты)"""
    
    def __init__(self):
        self.max_total_exposure_pct = 70  # Максимум 70% баланса в позициях
        self.daily_loss_limit_pct = 5  # Останов при -5% за день
        self.max_correlated_positions = 2  # Не более 2 коррелированных монет
        
        # Корреляция монет (упрощённо)
        self.correlation_groups = {
            'BTC': ['BTC', 'ETH', 'BNB', 'LTC'],  # Топ-монеты движутся вместе
            'ALT': ['SOL', 'ADA', 'DOT', 'AVAX', 'ATOM', 'LINK'],  # Альты
            'MEME': ['DOGE', 'SHIB', 'PEPE'],  # Мемные
        }
    
    def check_can_open_position(self, all_bots: list, symbol: str) -> tuple:
        """
        Проверить, можно ли открыть позицию
        
        Returns:
            (can_open: bool, reason: str)
        """
        # 1. Проверка общей экспозиции
        total_balance = sum(b.bot_manager.get_bot_balance(b.bot_id).get('USDT', 0) for b in all_bots)
        total_in_positions = sum(
            b.current_position['quantity'] * b.market_client.get_ticker_price(b.symbol)
            if b.current_position else 0
            for b in all_bots
        )
        
        exposure_pct = (total_in_positions / total_balance * 100) if total_balance > 0 else 0
        
        if exposure_pct >= self.max_total_exposure_pct:
            return False, f'Превышен лимит экспозиции: {exposure_pct:.1f}% (макс {self.max_total_exposure_pct}%)'
        
        # 2. Проверка корреляции
        symbol_base = symbol.replace('USDT', '')
        open_symbols = [b.symbol.replace('USDT', '') for b in all_bots if b.current_position]
        
        for group_name, group_symbols in self.correlation_groups.items():
            if symbol_base in group_symbols:
                correlated_count = sum(1 for s in open_symbols if s in group_symbols)
                
                if correlated_count >= self.max_correlated_positions:
                    return False, f'Превышен лимит коррелированных позиций ({group_name}): {correlated_count}/{self.max_correlated_positions}'
        
        return True, None
    
    def check_daily_loss_limit(self, bot) -> tuple:
        """
        Проверить дневной лимит убытков
        
        Returns:
            (should_stop: bool, reason: str)
        """
        # Считаем прибыль за сегодня
        today = datetime.utcnow().date()
        today_trades = [
            t for t in bot.trades_history
            if datetime.fromisoformat(t['exit_time']).date() == today
        ]
        
        today_profit = sum(t['profit_usdt'] for t in today_trades)
        initial_balance = 100  # Начальный баланс бота
        
        loss_pct = (today_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        if loss_pct <= -self.daily_loss_limit_pct:
            return True, f'Достигнут дневной лимит убытков: {loss_pct:.2f}% (лимит -{self.daily_loss_limit_pct}%)'
        
        return False, None
