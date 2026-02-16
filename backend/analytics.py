#!/usr/bin/env python3
"""
Аналитика и метрики для ботов
"""
from typing import List, Dict
from datetime import datetime, timedelta
import json


class BotAnalytics:
    """Аналитика для бота"""
    
    def __init__(self, bot):
        self.bot = bot
    
    def get_equity_curve(self) -> List[Dict]:
        """
        Построить equity curve (график баланса во времени)
        
        Returns:
            [{'timestamp': str, 'balance': float, 'profit': float}]
        """
        initial_balance = 100  # Начальный баланс
        current_balance = initial_balance
        
        curve = [{
            'timestamp': datetime.utcnow().isoformat(),
            'balance': initial_balance,
            'profit': 0,
        }]
        
        # Проходим по всем сделкам
        for trade in self.bot.trades_history:
            current_balance += trade['profit_usdt']
            
            curve.append({
                'timestamp': trade['exit_time'],
                'balance': round(current_balance, 2),
                'profit': round(trade['profit_usdt'], 2),
                'cumulative_profit': round(current_balance - initial_balance, 2),
            })
        
        return curve
    
    def get_performance_metrics(self) -> Dict:
        """
        Расширенные метрики производительности
        """
        if not self.bot.trades_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_profit_per_trade': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
            }
        
        profits = [t['profit_usdt'] for t in self.bot.trades_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        # Profit Factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max Drawdown
        equity_curve = self.get_equity_curve()
        balances = [e['balance'] for e in equity_curve]
        
        max_drawdown = 0
        peak = balances[0]
        
        for balance in balances:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (упрощённый)
        import numpy as np
        if len(profits) > 1:
            avg_profit = np.mean(profits)
            std_profit = np.std(profits)
            sharpe_ratio = avg_profit / std_profit if std_profit > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': len(self.bot.trades_history),
            'win_rate': self.bot.stats['win_rate'],
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_profit_per_trade': round(sum(profits) / len(profits), 2) if profits else 0,
            'avg_win': round(sum(wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0,
            'largest_win': round(max(wins), 2) if wins else 0,
            'largest_loss': round(min(losses), 2) if losses else 0,
        }
    
    def get_hourly_heatmap(self) -> Dict:
        """
        Heat map по часам (когда прибыльнее торговать)
        
        Returns:
            {hour: profit_sum}
        """
        hourly_profits = {}
        
        for trade in self.bot.trades_history:
            exit_time = datetime.fromisoformat(trade['exit_time'])
            hour = exit_time.hour
            
            if hour not in hourly_profits:
                hourly_profits[hour] = 0
            
            hourly_profits[hour] += trade['profit_usdt']
        
        return hourly_profits
    
    def get_daily_performance(self, days: int = 7) -> List[Dict]:
        """
        Производительность по дням
        
        Returns:
            [{'date': str, 'trades': int, 'profit': float, 'win_rate': float}]
        """
        daily_stats = {}
        
        for trade in self.bot.trades_history:
            exit_time = datetime.fromisoformat(trade['exit_time'])
            date_str = exit_time.date().isoformat()
            
            if date_str not in daily_stats:
                daily_stats[date_str] = {
                    'date': date_str,
                    'trades': 0,
                    'profit': 0,
                    'wins': 0,
                    'losses': 0,
                }
            
            daily_stats[date_str]['trades'] += 1
            daily_stats[date_str]['profit'] += trade['profit_usdt']
            
            if trade['profit_usdt'] > 0:
                daily_stats[date_str]['wins'] += 1
            else:
                daily_stats[date_str]['losses'] += 1
        
        # Рассчитываем win rate
        for stats in daily_stats.values():
            total = stats['wins'] + stats['losses']
            stats['win_rate'] = (stats['wins'] / total * 100) if total > 0 else 0
        
        # Сортируем по дате
        return sorted(daily_stats.values(), key=lambda x: x['date'], reverse=True)[:days]


class PortfolioAnalytics:
    """Аналитика всего портфеля"""
    
    def __init__(self, all_bots: List):
        self.all_bots = all_bots
    
    def get_total_equity_curve(self) -> List[Dict]:
        """Общий equity curve всех ботов"""
        all_events = []
        
        for bot in self.all_bots:
            for trade in bot.trades_history:
                all_events.append({
                    'timestamp': trade['exit_time'],
                    'profit': trade['profit_usdt'],
                    'bot_name': bot.name,
                })
        
        # Сортируем по времени
        all_events.sort(key=lambda x: x['timestamp'])
        
        # Строим кумулятивную кривую
        cumulative = 0
        curve = []
        
        for event in all_events:
            cumulative += event['profit']
            curve.append({
                'timestamp': event['timestamp'],
                'cumulative_profit': round(cumulative, 2),
                'bot_name': event['bot_name'],
            })
        
        return curve
    
    def get_top_performers(self, by: str = 'profit', top_n: int = 5) -> List[Dict]:
        """
        Топ ботов по метрике
        
        Args:
            by: 'profit', 'winrate', 'trades'
        """
        bots_data = []
        
        for bot in self.all_bots:
            bots_data.append({
                'bot_id': bot.bot_id,
                'name': bot.name,
                'profit': bot.stats['total_profit_usdt'],
                'winrate': bot.stats['win_rate'],
                'trades': bot.stats['total_trades'],
            })
        
        # Сортируем
        if by == 'profit':
            bots_data.sort(key=lambda x: x['profit'], reverse=True)
        elif by == 'winrate':
            bots_data.sort(key=lambda x: x['winrate'], reverse=True)
        elif by == 'trades':
            bots_data.sort(key=lambda x: x['trades'], reverse=True)
        
        return bots_data[:top_n]
    
    def get_correlation_matrix(self) -> Dict:
        """
        Матрица корреляции между ботами (упрощённо)
        """
        # TODO: Реализовать через numpy correlation
        return {}
