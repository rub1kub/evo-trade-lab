"""
WebSocket сервер для real-time обновлений
"""
import asyncio
import json
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}  # ws -> {bot_id, ...}
    
    async def connect(self, websocket: WebSocket):
        """Подключение клиента"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WS: Клиент подключен. Всего: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Отключение клиента"""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"WS: Клиент отключен. Всего: {len(self.active_connections)}")
    
    def subscribe(self, websocket: WebSocket, bot_id: str):
        """Подписка на бота"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(bot_id)
            logger.debug(f"WS: Подписка на бота {bot_id}")
    
    def unsubscribe(self, websocket: WebSocket, bot_id: str):
        """Отписка от бота"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(bot_id)
    
    async def broadcast(self, message: dict):
        """Отправка всем клиентам"""
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        
        for ws in dead:
            self.disconnect(ws)
    
    async def send_to_subscribers(self, bot_id: str, message: dict):
        """Отправка подписчикам конкретного бота"""
        dead = []
        for ws, subs in self.subscriptions.items():
            if bot_id in subs or 'all' in subs:
                try:
                    await ws.send_json(message)
                except:
                    dead.append(ws)
        
        for ws in dead:
            self.disconnect(ws)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Личное сообщение"""
        try:
            await websocket.send_json(message)
        except:
            self.disconnect(websocket)


# Глобальный менеджер
ws_manager = ConnectionManager()


class EventEmitter:
    """Эмиттер событий для ботов"""
    
    @staticmethod
    async def emit_trade(bot_id: int, trade: dict):
        """Событие новой сделки"""
        await ws_manager.send_to_subscribers(str(bot_id), {
            'type': 'trade',
            'bot_id': bot_id,
            'trade': trade,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_position_open(bot_id: int, position: dict):
        """Событие открытия позиции"""
        await ws_manager.send_to_subscribers(str(bot_id), {
            'type': 'position_open',
            'bot_id': bot_id,
            'position': position,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_position_close(bot_id: int, trade: dict):
        """Событие закрытия позиции"""
        await ws_manager.send_to_subscribers(str(bot_id), {
            'type': 'position_close',
            'bot_id': bot_id,
            'trade': trade,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_tp_hit(bot_id: int, level: int, price: float, pnl: float):
        """Событие срабатывания TP"""
        await ws_manager.broadcast({
            'type': 'tp_hit',
            'bot_id': bot_id,
            'level': level,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_sl_hit(bot_id: int, price: float, pnl: float):
        """Событие срабатывания SL"""
        await ws_manager.broadcast({
            'type': 'sl_hit',
            'bot_id': bot_id,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_price_update(symbol: str, price: float):
        """Обновление цены (broadcast)"""
        await ws_manager.broadcast({
            'type': 'price',
            'symbol': symbol,
            'price': price,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_bot_status(bot_id: int, status: dict):
        """Обновление статуса бота"""
        await ws_manager.send_to_subscribers(str(bot_id), {
            'type': 'bot_status',
            'bot_id': bot_id,
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @staticmethod
    async def emit_alert(bot_id: int, message: str, level: str = 'info'):
        """Алерт от бота"""
        await ws_manager.broadcast({
            'type': 'alert',
            'bot_id': bot_id,
            'message': message,
            'level': level,  # info, warning, error, success
            'timestamp': datetime.utcnow().isoformat()
        })


# Глобальный эмиттер
event_emitter = EventEmitter()


async def websocket_handler(websocket: WebSocket):
    """Обработчик WebSocket соединений"""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            action = data.get('action')
            
            if action == 'subscribe':
                bot_id = data.get('bot_id', 'all')
                ws_manager.subscribe(websocket, str(bot_id))
                await ws_manager.send_personal(websocket, {
                    'type': 'subscribed',
                    'bot_id': bot_id
                })
            
            elif action == 'unsubscribe':
                bot_id = data.get('bot_id')
                if bot_id:
                    ws_manager.unsubscribe(websocket, str(bot_id))
            
            elif action == 'ping':
                await ws_manager.send_personal(websocket, {
                    'type': 'pong',
                    'timestamp': datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WS Error: {e}")
        ws_manager.disconnect(websocket)
