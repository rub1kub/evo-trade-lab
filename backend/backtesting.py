"""
Backtesting Engine - тестирование стратегий на исторических данных
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Сделка в бэктесте"""
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'TP', 'SL', 'SIGNAL', 'END'


@dataclass
class BacktestResult:
    """Результат бэктеста"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_time_hours: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """Движок для бэктестинга стратегий"""
    
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
    
    async def fetch_historical_data(self, symbol: str, interval: str = '1h', 
                                   days: int = 30) -> List[List]:
        """Получение исторических данных с MEXC"""
        try:
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            url = f"https://api.mexc.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
                    else:
                        logger.error(f"Failed to fetch klines: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def run_backtest(self, strategy, klines: List[List], 
                    take_profit_pct: float = 3.0,
                    stop_loss_pct: float = 1.5,
                    position_size_pct: float = 10.0) -> BacktestResult:
        """
        Запуск бэктеста на исторических данных
        
        strategy: объект стратегии с методом analyze()
        klines: исторические данные [[timestamp, open, high, low, close, volume, ...], ...]
        """
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        # Минимум данных для анализа
        min_candles = getattr(strategy, 'min_candles', 50)
        
        for i in range(min_candles, len(klines)):
            # Срез данных для анализа
            historical = klines[:i+1]
            current_candle = klines[i]
            
            timestamp = current_candle[0]
            open_price = float(current_candle[1])
            high = float(current_candle[2])
            low = float(current_candle[3])
            close = float(current_candle[4])
            volume = float(current_candle[5])
            
            current_time = datetime.fromtimestamp(timestamp / 1000).isoformat()
            
            # Если есть позиция - проверяем TP/SL
            if self.position:
                entry = self.position['entry_price']
                qty = self.position['quantity']
                
                tp_price = entry * (1 + take_profit_pct / 100)
                sl_price = entry * (1 - stop_loss_pct / 100)
                
                # Проверка TP (по high)
                if high >= tp_price:
                    self._close_position(tp_price, current_time, 'TP')
                # Проверка SL (по low)
                elif low <= sl_price:
                    self._close_position(sl_price, current_time, 'SL')
            
            # Если нет позиции - анализируем сигнал
            if not self.position:
                try:
                    signal = strategy.analyze(historical)
                    
                    if signal and signal.get('action') == 'BUY' and signal.get('confidence', 0) >= 60:
                        # Открываем позицию
                        size = self.balance * position_size_pct / 100
                        qty = size / close
                        
                        self.position = {
                            'entry_price': close,
                            'quantity': qty,
                            'entry_time': current_time,
                            'size': size
                        }
                except Exception as e:
                    pass  # Стратегия может падать на некоторых данных
            
            # Обновляем equity curve
            if self.position:
                unrealized = (close - self.position['entry_price']) * self.position['quantity']
                self.equity_curve.append(self.balance + unrealized)
            else:
                self.equity_curve.append(self.balance)
        
        # Закрываем позицию в конце если открыта
        if self.position and klines:
            last_candle = klines[-1]
            close = float(last_candle[4])
            last_time = datetime.fromtimestamp(last_candle[0] / 1000).isoformat()
            self._close_position(close, last_time, 'END')
        
        return self._calculate_results(strategy, klines)
    
    def _close_position(self, price: float, time: str, reason: str):
        """Закрытие позиции"""
        if not self.position:
            return
        
        entry = self.position['entry_price']
        qty = self.position['quantity']
        pnl = (price - entry) * qty
        pnl_pct = (price - entry) / entry * 100
        
        self.balance += pnl
        
        trade = BacktestTrade(
            entry_time=self.position['entry_time'],
            exit_time=time,
            entry_price=entry,
            exit_price=price,
            quantity=qty,
            side='LONG',
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason
        )
        self.trades.append(trade)
        self.position = None
    
    def _calculate_results(self, strategy, klines: List[List]) -> BacktestResult:
        """Расчёт итоговых метрик"""
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        
        # Win rate
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Average trade
        avg_trade = total_pnl / len(self.trades) if self.trades else 0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Best/worst
        best_trade = max((t.pnl for t in self.trades), default=0)
        worst_trade = min((t.pnl for t in self.trades), default=0)
        
        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        max_dd_pct = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Sharpe ratio (упрощённый)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1] if len(self.equity_curve) > 1 else []
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average holding time
        avg_holding = 0
        if self.trades:
            holding_times = []
            for t in self.trades:
                try:
                    entry = datetime.fromisoformat(t.entry_time)
                    exit = datetime.fromisoformat(t.exit_time)
                    holding_times.append((exit - entry).total_seconds() / 3600)
                except:
                    pass
            avg_holding = np.mean(holding_times) if holding_times else 0
        
        # Date range
        start_date = datetime.fromtimestamp(klines[0][0] / 1000).strftime('%Y-%m-%d') if klines else ''
        end_date = datetime.fromtimestamp(klines[-1][0] / 1000).strftime('%Y-%m-%d') if klines else ''
        
        return BacktestResult(
            strategy_name=getattr(strategy, 'name', 'Unknown'),
            symbol=getattr(strategy, 'symbol', 'Unknown'),
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=round(self.balance, 2),
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl / self.initial_balance * 100, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_percent=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            profit_factor=round(profit_factor, 3) if profit_factor != float('inf') else 999,
            avg_trade_pnl=round(avg_trade, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            avg_holding_time_hours=round(avg_holding, 2),
            trades=self.trades,
            equity_curve=self.equity_curve
        )


class MonteCarloSimulation:
    """Monte Carlo симуляция для оценки риска"""
    
    @staticmethod
    def run(trades: List[BacktestTrade], iterations: int = 1000, 
           initial_balance: float = 1000) -> dict:
        """
        Запуск Monte Carlo симуляции
        Перемешивает порядок сделок для оценки разброса результатов
        """
        if not trades:
            return {'error': 'No trades to simulate'}
        
        pnls = [t.pnl for t in trades]
        final_balances = []
        max_drawdowns = []
        
        for _ in range(iterations):
            # Перемешиваем порядок сделок
            shuffled = np.random.permutation(pnls)
            
            # Считаем equity curve
            equity = [initial_balance]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl)
            
            final_balances.append(equity[-1])
            
            # Max drawdown
            peak = initial_balance
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)
        
        return {
            'iterations': iterations,
            'final_balance': {
                'mean': round(np.mean(final_balances), 2),
                'std': round(np.std(final_balances), 2),
                'min': round(np.min(final_balances), 2),
                'max': round(np.max(final_balances), 2),
                'percentile_5': round(np.percentile(final_balances, 5), 2),
                'percentile_95': round(np.percentile(final_balances, 95), 2),
            },
            'max_drawdown': {
                'mean': round(np.mean(max_drawdowns), 2),
                'std': round(np.std(max_drawdowns), 2),
                'worst_case': round(np.percentile(max_drawdowns, 95), 2),
            },
            'risk_of_ruin': round(sum(1 for b in final_balances if b < initial_balance * 0.5) / iterations * 100, 2)
        }


# Глобальный экземпляр
backtest_engine = BacktestEngine()
