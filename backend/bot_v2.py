#!/usr/bin/env python3
"""
Улучшенный торговый бот с режимами торговли и отдельными балансами
"""
import time
import os
import json
from typing import Dict, Optional
from datetime import datetime
from mexc_client import MEXCClient
from bot_manager import get_bot_manager
from trading_modes import get_mode
from risk_management import RiskManager
from notifications import TelegramNotifier

try:
    from ml_data_collector import ml_collector
except Exception:
    ml_collector = None


class TradingBotV2:
    """Улучшенный торговый бот"""
    
    def __init__(self, bot_id: int, name: str, symbol: str, strategy, 
                 mode_name: str = 'balanced', initial_balance: float = 1000, 
                 leverage: int = 2, interval: str = '5m'):
        self.bot_id = bot_id
        self.name = name
        self.symbol = symbol
        self.strategy = strategy
        self.mode = get_mode(mode_name)
        self.leverage = leverage  # Плечо
        self.use_futures = os.getenv('USE_FUTURES', 'true').lower() == 'true'
        self.interval = interval  # Интервал анализа: 1m, 5m, 15m, 1h
        
        # Получаем менеджера ботов
        self.bot_manager = get_bot_manager()
        
        # Создаём отдельный счёт для бота
        self.bot_manager.create_bot_account(bot_id, initial_balance)
        
        # Клиент для получения рыночных данных
        self.market_client = self.bot_manager.market_client
        
        self.is_running = False
        self.current_position = None
        self.pending_orders = []  # Лимитные ордера в ожидании
        
        # Risk management
        self.risk_manager = RiskManager(bot_id)
        
        # Notifications
        self.notifier = TelegramNotifier()
        
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_usdt': 0,
            'profit_percentage': 0,
            'max_drawdown': 0,
            'win_rate': 0,
        }
        self.logs = []
        self.trades_history = []  # Подробная история
        
    def log(self, level: str, message: str, data: Optional[Dict] = None):
        """Логирование"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'data': data or {}
        }
        self.logs.append(log_entry)
        
        # Ограничиваем размер логов
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
        
        # Печатаем в консоль
        timestamp_str = datetime.utcnow().strftime('%H:%M:%S')
        print(f"[{timestamp_str}] [{level}] {self.name}: {message}")
        
    def start(self):
        """Запустить бота"""
        self.is_running = True
        self.log('INFO', f'Бот запущен. Стратегия: {self.strategy.name}, Режим: {self.mode.name}')
        
    def stop(self):
        """Остановить бота"""
        self.is_running = False
        self.log('INFO', 'Бот остановлен')
    
    def place_limit_order(self, side: str, quantity: float, price: float):
        """Разместить лимитный ордер"""
        order = {
            'order_id': f'LIMIT_{len(self.pending_orders) + 1}',
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'PENDING',
            'created_at': datetime.utcnow().isoformat(),
        }
        self.pending_orders.append(order)
        self.log('INFO', f'Лимитный ордер размещён: {side} {quantity:.6f} @ ${price:.2f}', order)
        return order
    
    def cancel_limit_order(self, order_id: str):
        """Отменить лимитный ордер"""
        for order in self.pending_orders:
            if order['order_id'] == order_id and order['status'] == 'PENDING':
                order['status'] = 'CANCELLED'
                self.log('INFO', f'Лимитный ордер отменён: {order_id}')
                return True
        return False
    
    def check_pending_orders(self, current_price: float):
        """Проверить лимитные ордера на исполнение"""
        for order in self.pending_orders:
            if order['status'] != 'PENDING':
                continue
            
            # Проверяем условия исполнения
            should_execute = False
            
            if order['side'] == 'BUY' and current_price <= order['price']:
                should_execute = True
            elif order['side'] == 'SELL' and current_price >= order['price']:
                should_execute = True
            
            if should_execute:
                order['status'] = 'FILLED'
                order['filled_at'] = datetime.utcnow().isoformat()
                order['filled_price'] = current_price
                
                self.log('SUCCESS', f'✅ Лимитный ордер исполнен: {order["side"]} {order["quantity"]:.6f} @ ${current_price:.2f}', order)
                
                # Исполняем как обычный ордер
                if order['side'] == 'BUY':
                    self._execute_limit_buy(order, current_price)
                elif order['side'] == 'SELL':
                    self._execute_limit_sell(order, current_price)
    
    def _execute_limit_buy(self, order: Dict, current_price: float):
        """Исполнить лимитную покупку"""
        quantity = order['quantity']
        
        # Размещаем ордер через менеджера
        filled_order = self.bot_manager.place_bot_order(
            bot_id=self.bot_id,
            symbol=self.symbol,
            side='BUY',
            order_type='LIMIT',
            quantity=quantity,
            price=current_price
        )
        
        if filled_order['status'] == 'FILLED':
            self.current_position = {
                'symbol': self.symbol,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': datetime.utcnow().isoformat(),
            }
            self.stats['total_trades'] += 1
    
    def _execute_limit_sell(self, order: Dict, current_price: float):
        """Исполнить лимитную продажу"""
        if not self.current_position:
            return
        
        quantity = order['quantity']
        entry_price = self.current_position['entry_price']
        
        # Размещаем ордер через менеджера
        filled_order = self.bot_manager.place_bot_order(
            bot_id=self.bot_id,
            symbol=self.symbol,
            side='SELL',
            order_type='LIMIT',
            quantity=quantity,
            price=current_price
        )
        
        if filled_order['status'] == 'FILLED':
            profit_usdt = (current_price - entry_price) * quantity
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            self.stats['total_profit_usdt'] += profit_usdt
            
            if profit_usdt > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            
            total = self.stats['winning_trades'] + self.stats['losing_trades']
            if total > 0:
                self.stats['win_rate'] = (self.stats['winning_trades'] / total) * 100
            
            trade_detail = {
                'entry_time': self.current_position['entry_time'],
                'exit_time': datetime.utcnow().isoformat(),
                'entry_price': entry_price,
                'exit_price': current_price,
                'quantity': quantity,
                'profit_usdt': profit_usdt,
                'profit_pct': profit_pct,
                'reason': 'Лимитный ордер исполнен'
            }
            self.trades_history.append(trade_detail)
            
            if len(self.trades_history) > 50:
                self.trades_history = self.trades_history[-50:]
            
            self.current_position = None
    
    def tick(self):
        """Один цикл работы бота"""
        if not self.is_running:
            return
        
        try:
            # Получаем свечи
            klines = self.market_client.get_klines(self.symbol, interval=self.interval, limit=100)
            current_price = self.market_client.get_ticker_price(self.symbol)
            
            # Проверяем лимитные ордера
            self.check_pending_orders(current_price)
            
            # Анализируем
            analysis = self.strategy.analyze(klines, self.current_position)
            
            self.log('DEBUG', f"Анализ: {analysis['signal']}", {
                'confidence': analysis.get('confidence'),
                'reason': analysis.get('reason')
            })

            # Логируем торговые сигналы для ML
            if ml_collector is not None and analysis.get('signal') in ['BUY', 'SELL']:
                try:
                    ml_collector.record_signal(
                        symbol=self.symbol,
                        strategy=self.strategy.name,
                        signal_type=analysis.get('signal'),
                        confidence=float(analysis.get('confidence', 0) or 0),
                        price=current_price,
                        context={
                            'rsi': analysis.get('rsi') or analysis.get('rsi_value'),
                            'macd': analysis.get('macd') or analysis.get('macd_value'),
                            'bb_position': analysis.get('bb_position'),
                            'volume_ratio': analysis.get('volume_ratio')
                        }
                    )
                except Exception:
                    pass
            
            # Проверяем позицию на выход
            if self.current_position:
                # 1. Risk management (trailing stop, break-even)
                should_exit_risk, risk_reason = self.risk_manager.update_position(
                    self.current_position,
                    current_price
                )
                
                if should_exit_risk:
                    self._execute_sell(analysis, risk_reason)
                    self.risk_manager.reset()
                    return
                
                # 2. Режим торговли (take profit, stop loss)
                should_exit, exit_reason = self.mode.should_exit(
                    self.current_position['entry_price'],
                    current_price,
                    analysis['signal']
                )
                
                if should_exit:
                    self._execute_sell(analysis, exit_reason)
                    self.risk_manager.reset()
                    return
            
            # Проверяем вход в позицию
            if analysis['signal'] == 'BUY' and not self.current_position:
                if self.mode.should_enter(analysis['signal'], analysis.get('confidence', 0)):
                    self._execute_buy(analysis)
        
        except Exception as e:
            self.log('ERROR', f'Ошибка в цикле: {str(e)}')
    
    def _execute_buy(self, analysis: Dict):
        """Исполнить вход в позицию (futures/spot)"""
        try:
            current_price = self.market_client.get_ticker_price(self.symbol)
            balance = self.bot_manager.get_bot_balance(self.bot_id)

            # Маржа зависит от режима
            position_size_pct = self.mode.position_size_pct
            margin_usdt = balance.get('USDT', 0) * (position_size_pct / 100)

            if margin_usdt < 5:  # Минимальный размер сделки
                self.log('WARN', 'Недостаточно средств для входа')
                return

            lev = max(1, int(getattr(self, 'leverage', 1) or 1))

            # ===== Futures emulation =====
            if self.use_futures and lev > 1:
                account = self.bot_manager.get_bot_account(self.bot_id)
                notional = margin_usdt * lev
                quantity = notional / current_price

                # комиссия как taker от номинала
                fee_open = notional * account.TAKER_FEE
                required = margin_usdt + fee_open

                if account.balance.get('USDT', 0) < required:
                    self.log('WARN', f'Недостаточно маржи для futures входа (нужно {required:.2f} USDT)')
                    return

                account.balance['USDT'] -= required
                account.total_fees_paid += fee_open

                entry_ts_ms = int(datetime.utcnow().timestamp() * 1000)
                self.current_position = {
                    'symbol': self.symbol,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.utcnow().isoformat(),
                    'entry_ts_ms': entry_ts_ms,
                    'leverage': lev,
                    'margin': margin_usdt,
                    'notional': notional,
                    'open_fee': fee_open,
                    'market_type': 'futures'
                }

                # compute planned TP/SL levels for chart markers
                try:
                    tp_pct = float(getattr(self.mode, 'take_profit_pct', 0) or 0)
                    sl_pct = float(getattr(self.mode, 'stop_loss_pct', 0) or 0)
                    self.current_position['tp_price'] = current_price * (1 + tp_pct / 100) if tp_pct else None
                    self.current_position['sl_price'] = current_price * (1 - sl_pct / 100) if sl_pct else None
                except Exception:
                    self.current_position['tp_price'] = None
                    self.current_position['sl_price'] = None

                self.log('SUCCESS', f"✅ ВХОД FUTURES {lev}x: {quantity:.6f} по ${current_price:,.2f}", {
                    'action': 'BUY',
                    'reason': analysis['reason'],
                    'margin': margin_usdt,
                    'notional': notional,
                    'open_fee': fee_open,
                    'confidence': analysis.get('confidence'),
                    'entry_price': current_price,
                    'quantity': quantity,
                    'mode': self.mode.name,
                    'strategy': self.strategy.name,
                })
                # write event for chart markers (entry fill)
                if ml_collector is not None:
                    try:
                        # entry fill
                        ml_collector.record_trade(
                            bot_id=self.bot_id,
                            bot_name=self.name,
                            trade={
                                'symbol': self.symbol,
                                'strategy': self.strategy.name,
                                'mode': self.mode.name,
                                'entry_time': self.current_position['entry_time'],
                                'exit_time': None,
                                'entry_price': current_price,
                                'exit_price': None,
                                'quantity': quantity,
                                'side': 'LONG',
                                'pnl': 0.0,
                                'pnl_percent': 0.0,
                                'exit_reason': 'ENTRY',
                                'market_type': 'futures',
                                'leverage': lev,
                                'margin': margin_usdt,
                                'notional': notional,
                                'open_fee': fee_open,
                                'close_fee': 0.0,
                                'entry_ts_ms': entry_ts_ms,
                            },
                            context={
                                'rsi': analysis.get('rsi'),
                                'macd': analysis.get('macd'),
                                'volume': analysis.get('volume'),
                                'btc_price': self.market_client.get_ticker_price('BTCUSDT'),
                                'btc_change_24h': 0,
                                'market_sentiment': 'neutral'
                            }
                        )

                        # planned TP/SL markers
                        if self.current_position.get('tp_price'):
                            conn = ml_collector._get_conn(); cur = conn.cursor()
                            # sqlite placeholders (DB_BACKEND isn't wired in this module)
                            cur.execute('''
                                INSERT INTO trade_events
                                (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, reason)
                                VALUES (?,?,?,?,?,?,?,?,?,?)
                            ''', (self.bot_id, self.name, self.symbol.upper(), 'TP_SET', 'LONG', entry_ts_ms, entry_ts_ms, float(self.current_position['tp_price']), float(quantity), 'MODE'))
                            conn.commit(); conn.close()

                        if self.current_position.get('sl_price'):
                            conn = ml_collector._get_conn(); cur = conn.cursor()
                            cur.execute('''
                                INSERT INTO trade_events
                                (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, reason)
                                VALUES (?,?,?,?,?,?,?,?,?,?)
                            ''', (self.bot_id, self.name, self.symbol.upper(), 'SL_SET', 'LONG', entry_ts_ms, entry_ts_ms, float(self.current_position['sl_price']), float(quantity), 'MODE'))
                            conn.commit(); conn.close()

                    except Exception:
                        pass

                self.stats['total_trades'] += 1
                return

            # ===== Spot fallback =====
            quantity = margin_usdt / current_price

            order = self.bot_manager.place_bot_order(
                bot_id=self.bot_id,
                symbol=self.symbol,
                side='BUY',
                order_type='MARKET',
                quantity=quantity,
                price=current_price
            )

            if order['status'] == 'FILLED':
                self.current_position = {
                    'symbol': self.symbol,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.utcnow().isoformat(),
                    'leverage': 1,
                    'margin': margin_usdt,
                    'market_type': 'spot'
                }

                self.log('SUCCESS', f"✅ ВХОД SPOT: Куплено {quantity:.6f} по ${current_price:,.2f}", {
                    'action': 'BUY',
                    'reason': analysis['reason'],
                    'cost': margin_usdt,
                    'confidence': analysis.get('confidence'),
                    'entry_price': current_price,
                    'quantity': quantity,
                    'mode': self.mode.name,
                    'strategy': self.strategy.name,
                })
                self.stats['total_trades'] += 1

        except Exception as e:
            self.log('ERROR', f'Ошибка покупки: {str(e)}')
    
    def _execute_sell(self, analysis: Dict, reason: str = None):
        """Исполнить выход из позиции (futures/spot)"""
        try:
            current_price = self.market_client.get_ticker_price(self.symbol)
            quantity = self.current_position['quantity']
            entry_price = self.current_position['entry_price']
            market_type = self.current_position.get('market_type', 'spot')

            # ===== Futures emulation =====
            if market_type == 'futures':
                account = self.bot_manager.get_bot_account(self.bot_id)
                margin = float(self.current_position.get('margin', 0) or 0)
                lev = max(1, int(self.current_position.get('leverage', 1) or 1))
                notional_exit = quantity * current_price

                gross_pnl = (current_price - entry_price) * quantity
                fee_close = notional_exit * account.TAKER_FEE
                profit_usdt = gross_pnl - fee_close
                profit_pct = (profit_usdt / margin * 100) if margin > 0 else 0

                account.balance['USDT'] = account.balance.get('USDT', 0) + margin + profit_usdt
                account.total_fees_paid += fee_close

                self.stats['total_profit_usdt'] += profit_usdt

                if profit_usdt > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1

                total = self.stats['winning_trades'] + self.stats['losing_trades']
                if total > 0:
                    self.stats['win_rate'] = (self.stats['winning_trades'] / total) * 100

                trade_detail = {
                    'entry_time': self.current_position['entry_time'],
                    'exit_time': datetime.utcnow().isoformat(),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'profit_usdt': profit_usdt,
                    'profit_pct': profit_pct,
                    'reason': reason or analysis.get('reason', 'Manual exit'),
                    'market_type': 'futures',
                    'leverage': lev,
                    'margin': margin,
                    'notional': float(self.current_position.get('notional', 0) or 0),
                    'open_fee': float(self.current_position.get('open_fee', 0) or 0),
                    'close_fee': fee_close,
                }
                self.trades_history.append(trade_detail)
                if len(self.trades_history) > 50:
                    self.trades_history = self.trades_history[-50:]

                exit_type = 'UNKNOWN'
                if reason and 'Take profit' in reason:
                    exit_type = 'TAKE_PROFIT'
                elif reason and 'Stop loss' in reason:
                    exit_type = 'STOP_LOSS'
                elif reason and 'Сигнал' in reason:
                    exit_type = 'SIGNAL'

                self.log('SUCCESS', f"✅ ВЫХОД FUTURES {lev}x: {quantity:.6f} по ${current_price:,.2f} | Прибыль: {profit_usdt:+.2f} USDT ({profit_pct:+.2f}%)", {
                    'action': 'SELL',
                    'exit_type': exit_type,
                    'reason': reason or analysis.get('reason'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'profit_usdt': round(profit_usdt, 2),
                    'profit_pct': round(profit_pct, 2),
                    'mode': self.mode.name,
                    'strategy': self.strategy.name,
                    'market_type': 'futures',
                })

                self.notifier.send_trade_alert(self.name, trade_detail)

                if ml_collector is not None:
                    try:
                        ml_collector.record_trade(
                            bot_id=self.bot_id,
                            bot_name=self.name,
                            trade={
                                'symbol': self.symbol,
                                'strategy': self.strategy.name,
                                'mode': self.mode.name,
                                'entry_time': trade_detail['entry_time'],
                                'exit_time': trade_detail['exit_time'],
                                'entry_price': trade_detail['entry_price'],
                                'exit_price': trade_detail['exit_price'],
                                'quantity': trade_detail['quantity'],
                                'side': 'LONG',
                                'pnl': trade_detail['profit_usdt'],
                                'pnl_percent': trade_detail['profit_pct'],
                                'exit_reason': trade_detail['reason'],
                                'market_type': trade_detail.get('market_type'),
                                'leverage': trade_detail.get('leverage'),
                                'margin': trade_detail.get('margin'),
                                'notional': trade_detail.get('notional'),
                                'open_fee': trade_detail.get('open_fee'),
                                'close_fee': trade_detail.get('close_fee')
                            },
                            context={
                                'rsi': analysis.get('rsi'),
                                'macd': analysis.get('macd'),
                                'volume': analysis.get('volume'),
                                'btc_price': self.market_client.get_ticker_price('BTCUSDT'),
                                'btc_change_24h': 0,
                                'market_sentiment': 'neutral'
                            }
                        )
                    except Exception as ml_err:
                        self.log('WARN', f'ML log failed: {ml_err}')

                self.current_position = None
                return

            # ===== Spot fallback =====
            order = self.bot_manager.place_bot_order(
                bot_id=self.bot_id,
                symbol=self.symbol,
                side='SELL',
                order_type='MARKET',
                quantity=quantity,
                price=current_price
            )

            if order['status'] == 'FILLED':
                profit_usdt = (current_price - entry_price) * quantity
                profit_pct = ((current_price - entry_price) / entry_price) * 100

                self.stats['total_profit_usdt'] += profit_usdt

                if profit_usdt > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1

                total = self.stats['winning_trades'] + self.stats['losing_trades']
                if total > 0:
                    self.stats['win_rate'] = (self.stats['winning_trades'] / total) * 100

                trade_detail = {
                    'entry_time': self.current_position['entry_time'],
                    'exit_time': datetime.utcnow().isoformat(),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'profit_usdt': profit_usdt,
                    'profit_pct': profit_pct,
                    'reason': reason or analysis.get('reason', 'Manual exit'),
                    'market_type': 'spot',
                    'leverage': 1,
                    'margin': float(self.current_position.get('margin', 0) or 0),
                    'notional': float(self.current_position.get('margin', 0) or 0),
                    'open_fee': 0.0,
                    'close_fee': 0.0,
                }
                self.trades_history.append(trade_detail)

                if len(self.trades_history) > 50:
                    self.trades_history = self.trades_history[-50:]

                exit_type = 'UNKNOWN'
                if reason and 'Take profit' in reason:
                    exit_type = 'TAKE_PROFIT'
                elif reason and 'Stop loss' in reason:
                    exit_type = 'STOP_LOSS'
                elif reason and 'Сигнал' in reason:
                    exit_type = 'SIGNAL'

                self.log('SUCCESS', f"✅ ВЫХОД SPOT: Продано {quantity:.6f} по ${current_price:,.2f} | Прибыль: {profit_usdt:+.2f} USDT ({profit_pct:+.2f}%)", {
                    'action': 'SELL',
                    'exit_type': exit_type,
                    'reason': reason or analysis.get('reason'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'profit_usdt': round(profit_usdt, 2),
                    'profit_pct': round(profit_pct, 2),
                    'hold_time': trade_detail['exit_time'],
                    'mode': self.mode.name,
                    'strategy': self.strategy.name,
                    'market_type': 'spot',
                })

                self.notifier.send_trade_alert(self.name, trade_detail)

                if ml_collector is not None:
                    try:
                        ml_collector.record_trade(
                            bot_id=self.bot_id,
                            bot_name=self.name,
                            trade={
                                'symbol': self.symbol,
                                'strategy': self.strategy.name,
                                'mode': self.mode.name,
                                'entry_time': trade_detail['entry_time'],
                                'exit_time': trade_detail['exit_time'],
                                'entry_price': trade_detail['entry_price'],
                                'exit_price': trade_detail['exit_price'],
                                'quantity': trade_detail['quantity'],
                                'side': 'LONG',
                                'pnl': trade_detail['profit_usdt'],
                                'pnl_percent': trade_detail['profit_pct'],
                                'exit_reason': trade_detail['reason'],
                                'market_type': trade_detail.get('market_type'),
                                'leverage': trade_detail.get('leverage'),
                                'margin': trade_detail.get('margin'),
                                'notional': trade_detail.get('notional'),
                                'open_fee': trade_detail.get('open_fee'),
                                'close_fee': trade_detail.get('close_fee')
                            },
                            context={
                                'rsi': analysis.get('rsi'),
                                'macd': analysis.get('macd'),
                                'volume': analysis.get('volume'),
                                'btc_price': self.market_client.get_ticker_price('BTCUSDT'),
                                'btc_change_24h': 0,
                                'market_sentiment': 'neutral'
                            }
                        )
                    except Exception as ml_err:
                        self.log('WARN', f'ML log failed: {ml_err}')

                self.current_position = None

        except Exception as e:
            self.log('ERROR', f'Ошибка продажи: {str(e)}')
    
    def get_status(self) -> Dict:
        """Получить статус бота"""
        balance = self.bot_manager.get_bot_balance(self.bot_id)
        
        # Рассчитываем общую стоимость портфеля
        total_value_usdt = balance.get('USDT', 0)
        
        if self.current_position:
            current_price = self.market_client.get_ticker_price(self.symbol)
            if self.current_position.get('market_type') == 'futures':
                margin = float(self.current_position.get('margin', 0) or 0)
                entry = float(self.current_position.get('entry_price', 0) or 0)
                qty = float(self.current_position.get('quantity', 0) or 0)
                unrealized = (current_price - entry) * qty
                total_value_usdt += (margin + unrealized)
            else:
                position_value = self.current_position['quantity'] * current_price
                total_value_usdt += position_value
        
        # Получаем информацию о комиссиях
        account = self.bot_manager.get_bot_account(self.bot_id)
        total_fees = getattr(account, 'total_fees_paid', 0)
        
        return {
            'bot_id': self.bot_id,
            'name': self.name,
            'symbol': self.symbol,
            'strategy': self.strategy.get_config(),
            'mode': self.mode.get_config(),
            'is_running': self.is_running,
            'current_position': self.current_position,
            'pending_orders': [o for o in self.pending_orders if o['status'] == 'PENDING'],
            'balance': balance,
            'total_value_usdt': round(total_value_usdt, 2),
            'total_fees_paid': round(total_fees, 2),
            'stats': self.stats,
            'recent_logs': self.logs[-20:],
            'trades_history': self.trades_history[-10:],
        }
