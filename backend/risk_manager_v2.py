"""
Продвинутый риск-менеджмент: Kelly Criterion, Daily Limits, Correlation
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Kelly Criterion для оптимального размера позиции"""
    
    @staticmethod
    def calculate(win_rate: float, avg_win: float, avg_loss: float, 
                  fraction: float = 0.5) -> float:
        """
        Рассчитывает оптимальный размер позиции по Kelly
        
        win_rate: вероятность выигрыша (0-1)
        avg_win: средний выигрыш в %
        avg_loss: средний проигрыш в % (положительное число)
        fraction: фракция Kelly (0.5 = Half Kelly, безопаснее)
        
        Returns: % от капитала для позиции
        """
        if avg_loss == 0:
            return 0
        
        # Kelly Formula: f* = (bp - q) / b
        # b = отношение выигрыша к проигрышу
        # p = вероятность выигрыша
        # q = вероятность проигрыша
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Ограничиваем kelly
        kelly = max(0, min(kelly, 0.25))  # Макс 25% от капитала
        
        # Применяем фракцию (Half Kelly)
        return kelly * fraction * 100  # В процентах


class DailyLossLimit:
    """Дневной лимит потерь"""
    
    def __init__(self, limit_percent: float = 5.0):
        self.limit_percent = limit_percent  # Макс потери в день %
        self.daily_pnl: Dict[int, float] = defaultdict(float)  # bot_id -> daily pnl
        self.last_reset: Dict[int, datetime] = {}
        self.locked_bots: set = set()
    
    def record_trade(self, bot_id: int, pnl: float, balance: float):
        """Записать результат сделки"""
        self._check_reset(bot_id)
        
        self.daily_pnl[bot_id] += pnl
        loss_percent = abs(self.daily_pnl[bot_id]) / balance * 100
        
        if self.daily_pnl[bot_id] < 0 and loss_percent >= self.limit_percent:
            self.locked_bots.add(bot_id)
            logger.warning(f"Bot {bot_id}: ЗАБЛОКИРОВАН - дневной лимит потерь {loss_percent:.2f}%")
            return False
        
        return True
    
    def _check_reset(self, bot_id: int):
        """Сброс в начале нового дня"""
        now = datetime.utcnow()
        last = self.last_reset.get(bot_id)
        
        if last is None or last.date() < now.date():
            self.daily_pnl[bot_id] = 0
            self.last_reset[bot_id] = now
            self.locked_bots.discard(bot_id)
    
    def is_locked(self, bot_id: int) -> bool:
        """Проверка заблокирован ли бот"""
        self._check_reset(bot_id)
        return bot_id in self.locked_bots
    
    def get_daily_stats(self, bot_id: int) -> dict:
        self._check_reset(bot_id)
        return {
            'daily_pnl': self.daily_pnl.get(bot_id, 0),
            'limit_percent': self.limit_percent,
            'is_locked': bot_id in self.locked_bots
        }


class CorrelationMatrix:
    """Корреляционная матрица портфеля"""
    
    # Примерные корреляции крипто-активов (можно обновлять динамически)
    CORRELATIONS = {
        ('BTC', 'ETH'): 0.85,
        ('BTC', 'SOL'): 0.75,
        ('BTC', 'BNB'): 0.70,
        ('BTC', 'ADA'): 0.72,
        ('BTC', 'XRP'): 0.65,
        ('BTC', 'DOT'): 0.78,
        ('BTC', 'LINK'): 0.73,
        ('BTC', 'AVAX'): 0.76,
        ('BTC', 'ATOM'): 0.71,
        ('BTC', 'LTC'): 0.80,
        ('BTC', 'DOGE'): 0.60,
        ('BTC', 'SHIB'): 0.55,
        ('BTC', 'PEPE'): 0.50,
        ('ETH', 'SOL'): 0.80,
        ('ETH', 'BNB'): 0.75,
        ('DOGE', 'SHIB'): 0.85,
        ('DOGE', 'PEPE'): 0.82,
        ('SHIB', 'PEPE'): 0.90,
    }
    
    @classmethod
    def get_correlation(cls, symbol1: str, symbol2: str) -> float:
        """Получить корреляцию между двумя активами"""
        asset1 = symbol1.replace('USDT', '')
        asset2 = symbol2.replace('USDT', '')
        
        if asset1 == asset2:
            return 1.0
        
        # Проверяем оба направления
        key1 = (asset1, asset2)
        key2 = (asset2, asset1)
        
        return cls.CORRELATIONS.get(key1, cls.CORRELATIONS.get(key2, 0.6))  # Default 0.6
    
    @classmethod
    def calculate_portfolio_risk(cls, positions: List[dict]) -> dict:
        """
        Рассчитать риск портфеля с учётом корреляций
        positions: [{'symbol': 'BTCUSDT', 'value': 100, 'weight': 0.3}, ...]
        """
        if not positions:
            return {'diversification_score': 0, 'effective_positions': 0, 'risk_adjustment': 1.0}
        
        total_value = sum(p['value'] for p in positions)
        if total_value == 0:
            return {'diversification_score': 0, 'effective_positions': 0, 'risk_adjustment': 1.0}
        
        # Добавляем веса
        for p in positions:
            p['weight'] = p['value'] / total_value
        
        # Рассчитываем эффективное количество позиций (HHI inverse)
        hhi = sum(p['weight'] ** 2 for p in positions)
        effective_n = 1 / hhi if hhi > 0 else len(positions)
        
        # Средняя корреляция
        total_corr = 0
        pairs = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                corr = cls.get_correlation(p1['symbol'], p2['symbol'])
                weight = p1['weight'] * p2['weight']
                total_corr += corr * weight
                pairs += weight
        
        avg_correlation = total_corr / pairs if pairs > 0 else 0
        
        # Diversification score (0-100)
        # Чем больше позиций и меньше корреляция - тем лучше
        div_score = min(100, effective_n * 20 * (1 - avg_correlation))
        
        # Risk adjustment (множитель для размера позиции)
        # Высокая корреляция = снижаем размер
        risk_adj = 1 - (avg_correlation * 0.3)
        
        return {
            'diversification_score': round(div_score, 1),
            'effective_positions': round(effective_n, 2),
            'average_correlation': round(avg_correlation, 3),
            'risk_adjustment': round(risk_adj, 3),
            'recommendation': 'Хорошая диверсификация' if div_score > 60 else 
                            'Средняя диверсификация' if div_score > 30 else 
                            'Низкая диверсификация - высокий риск'
        }


class AdvancedRiskManager:
    """Комплексный риск-менеджер"""
    
    def __init__(self, max_portfolio_risk: float = 20.0, 
                 max_position_size: float = 10.0,
                 daily_loss_limit: float = 5.0):
        self.max_portfolio_risk = max_portfolio_risk  # Макс % портфеля под риском
        self.max_position_size = max_position_size  # Макс % на одну позицию
        self.daily_limit = DailyLossLimit(daily_loss_limit)
        self.kelly = KellyCriterion()
        self.correlation = CorrelationMatrix()
    
    def calculate_position_size(self, bot_id: int, balance: float, 
                                win_rate: float, avg_win: float, avg_loss: float,
                                current_positions: List[dict],
                                symbol: str) -> dict:
        """
        Рассчитать оптимальный размер позиции
        """
        # Проверяем дневной лимит
        if self.daily_limit.is_locked(bot_id):
            return {
                'size_percent': 0,
                'size_usdt': 0,
                'reason': 'Бот заблокирован - дневной лимит потерь',
                'allowed': False
            }
        
        # Kelly Criterion
        kelly_size = self.kelly.calculate(win_rate, avg_win, avg_loss)
        
        # Корреляционная корректировка
        if current_positions:
            portfolio_risk = self.correlation.calculate_portfolio_risk(current_positions)
            kelly_size *= portfolio_risk['risk_adjustment']
        
        # Ограничения
        final_size = min(kelly_size, self.max_position_size)
        
        return {
            'kelly_optimal': round(kelly_size, 2),
            'final_size_percent': round(final_size, 2),
            'size_usdt': round(balance * final_size / 100, 2),
            'allowed': True,
            'win_rate': win_rate,
            'max_position': self.max_position_size
        }
    
    def get_portfolio_status(self, bots: List[dict]) -> dict:
        """Общий статус портфеля"""
        positions = []
        total_value = 0
        total_at_risk = 0
        
        for bot in bots:
            if bot.get('current_position'):
                pos = bot['current_position']
                value = pos.get('entry_price', 0) * pos.get('quantity', 0)
                positions.append({
                    'symbol': bot['symbol'],
                    'value': value,
                    'bot_id': bot['bot_id']
                })
                total_value += value
                
                # Риск = размер позиции * stop loss %
                sl_pct = bot.get('mode', {}).get('stop_loss_pct', 2)
                total_at_risk += value * sl_pct / 100
        
        correlation_analysis = self.correlation.calculate_portfolio_risk(positions)
        
        return {
            'total_positions': len(positions),
            'total_value': round(total_value, 2),
            'total_at_risk': round(total_at_risk, 2),
            'risk_percent': round(total_at_risk / total_value * 100, 2) if total_value > 0 else 0,
            'correlation_analysis': correlation_analysis,
            'recommendation': self._get_recommendation(len(positions), correlation_analysis)
        }
    
    def _get_recommendation(self, n_positions: int, corr_analysis: dict) -> str:
        if n_positions == 0:
            return "Нет открытых позиций"
        
        div_score = corr_analysis.get('diversification_score', 0)
        
        if div_score > 70:
            return "✅ Отличная диверсификация"
        elif div_score > 50:
            return "👍 Хорошая диверсификация"
        elif div_score > 30:
            return "⚠️ Добавьте некоррелированные активы"
        else:
            return "🚨 Высокий риск концентрации!"


# Глобальный экземпляр
risk_manager = AdvancedRiskManager()
