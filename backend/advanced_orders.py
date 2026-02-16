"""
Продвинутые ордера: Multi-TP, OCO, Trailing Stop
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TakeProfitLevel:
    """Уровень Take Profit"""
    level: int  # 1, 2, 3...
    price: float
    percent_to_close: float  # % позиции для закрытия
    triggered: bool = False
    triggered_at: Optional[str] = None


@dataclass
class TrailingStop:
    """Trailing Stop Loss"""
    activation_price: float  # Цена активации
    callback_rate: float  # % отката от максимума
    is_active: bool = False
    highest_price: float = 0
    current_stop: float = 0


@dataclass
class OCOOrder:
    """One-Cancels-Other: TP + SL связаны"""
    take_profit_price: float
    stop_loss_price: float
    status: str = 'PENDING'  # PENDING, TP_HIT, SL_HIT, CANCELLED


class AdvancedOrderManager:
    """Менеджер продвинутых ордеров"""
    
    def __init__(self):
        self.multi_tp: Dict[int, List[TakeProfitLevel]] = {}  # bot_id -> TPs
        self.trailing_stops: Dict[int, TrailingStop] = {}  # bot_id -> trailing
        self.oco_orders: Dict[int, OCOOrder] = {}  # bot_id -> OCO
        self.on_event = None  # optional callback(bot_id:int, event:dict)
    
    def setup_multi_tp(self, bot_id: int, entry_price: float, levels: List[dict]) -> List[TakeProfitLevel]:
        """
        Настройка нескольких уровней TP
        levels: [{'percent': 2.0, 'close_percent': 33}, ...]
        """
        tps = []
        for i, level in enumerate(levels, 1):
            tp = TakeProfitLevel(
                level=i,
                price=entry_price * (1 + level['percent'] / 100),
                percent_to_close=level['close_percent']
            )
            tps.append(tp)
        
        self.multi_tp[bot_id] = tps
        logger.info(f"Bot {bot_id}: Multi-TP настроен: {len(tps)} уровней")

        if self.on_event:
            now_ms = int(datetime.utcnow().timestamp() * 1000)
            for tp in tps:
                self.on_event(bot_id, {
                    'event_type': 'TP_SET',
                    'level': tp.level,
                    'price': tp.price,
                    'ts_server_ms': now_ms,
                })

        return tps
    
    def setup_trailing_stop(self, bot_id: int, entry_price: float, 
                           activation_percent: float, callback_rate: float) -> TrailingStop:
        """
        Настройка Trailing Stop
        activation_percent: % роста для активации
        callback_rate: % отката для срабатывания
        """
        ts = TrailingStop(
            activation_price=entry_price * (1 + activation_percent / 100),
            callback_rate=callback_rate,
            is_active=False,
            highest_price=entry_price,
            current_stop=entry_price * (1 - callback_rate / 100)
        )
        self.trailing_stops[bot_id] = ts
        logger.info(f"Bot {bot_id}: Trailing Stop настроен: активация ${ts.activation_price:.2f}, откат {callback_rate}%")
        return ts
    
    def setup_oco(self, bot_id: int, entry_price: float, tp_percent: float, sl_percent: float) -> OCOOrder:
        """Настройка OCO ордера"""
        oco = OCOOrder(
            take_profit_price=entry_price * (1 + tp_percent / 100),
            stop_loss_price=entry_price * (1 - sl_percent / 100)
        )
        self.oco_orders[bot_id] = oco
        logger.info(f"Bot {bot_id}: OCO настроен: TP ${oco.take_profit_price:.2f}, SL ${oco.stop_loss_price:.2f}")

        if self.on_event:
            now_ms = int(datetime.utcnow().timestamp() * 1000)
            self.on_event(bot_id, {
                'event_type': 'TP_SET',
                'price': oco.take_profit_price,
                'ts_server_ms': now_ms,
            })
            self.on_event(bot_id, {
                'event_type': 'SL_SET',
                'price': oco.stop_loss_price,
                'ts_server_ms': now_ms,
            })

        return oco
    
    def check_multi_tp(self, bot_id: int, current_price: float) -> Optional[TakeProfitLevel]:
        """Проверка срабатывания Multi-TP"""
        if bot_id not in self.multi_tp:
            return None
        
        for tp in self.multi_tp[bot_id]:
            if not tp.triggered and current_price >= tp.price:
                tp.triggered = True
                tp.triggered_at = datetime.utcnow().isoformat()
                logger.info(f"Bot {bot_id}: TP{tp.level} сработал @ ${current_price:.2f}")

                if self.on_event:
                    self.on_event(bot_id, {
                        'event_type': 'TP_HIT',
                        'level': tp.level,
                        'price': current_price,
                        'ts_server_ms': int(datetime.utcnow().timestamp() * 1000),
                    })

                return tp
        
        return None
    
    def check_trailing_stop(self, bot_id: int, current_price: float) -> Optional[float]:
        """
        Обновление и проверка Trailing Stop
        Возвращает цену стопа если сработал, иначе None
        """
        if bot_id not in self.trailing_stops:
            return None
        
        ts = self.trailing_stops[bot_id]
        
        # Активация
        if not ts.is_active and current_price >= ts.activation_price:
            ts.is_active = True
            ts.highest_price = current_price
            ts.current_stop = current_price * (1 - ts.callback_rate / 100)
            logger.info(f"Bot {bot_id}: Trailing Stop АКТИВИРОВАН @ ${current_price:.2f}")
        
        # Обновление если активен
        if ts.is_active:
            if current_price > ts.highest_price:
                ts.highest_price = current_price
                ts.current_stop = current_price * (1 - ts.callback_rate / 100)
            
            # Проверка срабатывания
            if current_price <= ts.current_stop:
                logger.info(f"Bot {bot_id}: Trailing Stop СРАБОТАЛ @ ${current_price:.2f}")

                if self.on_event:
                    self.on_event(bot_id, {
                        'event_type': 'SL_HIT',
                        'price': current_price,
                        'ts_server_ms': int(datetime.utcnow().timestamp() * 1000),
                        'reason': 'TRAILING_STOP',
                    })

                return ts.current_stop
        
        return None
    
    def check_oco(self, bot_id: int, current_price: float) -> Optional[str]:
        """
        Проверка OCO ордера
        Возвращает 'TP' или 'SL' если сработал
        """
        if bot_id not in self.oco_orders:
            return None
        
        oco = self.oco_orders[bot_id]
        
        if oco.status != 'PENDING':
            return None
        
        if current_price >= oco.take_profit_price:
            oco.status = 'TP_HIT'
            logger.info(f"Bot {bot_id}: OCO Take Profit @ ${current_price:.2f}")
            if self.on_event:
                self.on_event(bot_id, {
                    'event_type': 'TP_HIT',
                    'price': current_price,
                    'ts_server_ms': int(datetime.utcnow().timestamp() * 1000),
                    'reason': 'OCO',
                })
            return 'TP'
        
        if current_price <= oco.stop_loss_price:
            oco.status = 'SL_HIT'
            logger.info(f"Bot {bot_id}: OCO Stop Loss @ ${current_price:.2f}")
            if self.on_event:
                self.on_event(bot_id, {
                    'event_type': 'SL_HIT',
                    'price': current_price,
                    'ts_server_ms': int(datetime.utcnow().timestamp() * 1000),
                    'reason': 'OCO',
                })
            return 'SL'
        
        return None
    
    def clear_bot_orders(self, bot_id: int):
        """Очистка всех ордеров бота"""
        self.multi_tp.pop(bot_id, None)
        self.trailing_stops.pop(bot_id, None)
        self.oco_orders.pop(bot_id, None)
    
    def get_bot_orders_status(self, bot_id: int) -> dict:
        """Получение статуса всех продвинутых ордеров бота"""
        result = {
            'multi_tp': None,
            'trailing_stop': None,
            'oco': None
        }
        
        if bot_id in self.multi_tp:
            result['multi_tp'] = [
                {
                    'level': tp.level,
                    'price': tp.price,
                    'percent_to_close': tp.percent_to_close,
                    'triggered': tp.triggered,
                    'triggered_at': tp.triggered_at
                }
                for tp in self.multi_tp[bot_id]
            ]
        
        if bot_id in self.trailing_stops:
            ts = self.trailing_stops[bot_id]
            result['trailing_stop'] = {
                'activation_price': ts.activation_price,
                'callback_rate': ts.callback_rate,
                'is_active': ts.is_active,
                'highest_price': ts.highest_price,
                'current_stop': ts.current_stop
            }
        
        if bot_id in self.oco_orders:
            oco = self.oco_orders[bot_id]
            result['oco'] = {
                'take_profit_price': oco.take_profit_price,
                'stop_loss_price': oco.stop_loss_price,
                'status': oco.status
            }
        
        return result


# Глобальный экземпляр
advanced_order_manager = AdvancedOrderManager()
