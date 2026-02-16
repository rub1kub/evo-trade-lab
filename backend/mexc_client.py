#!/usr/bin/env python3
"""
MEXC API Client с демо-режимом для безопасного тестирования
"""
import os
import time
import hmac
import hashlib
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class MEXCDemoAccount:
    """Виртуальный демо-счёт для тестирования без риска"""
    
    MAKER_FEE = 0.0002  # 0.02% maker fee (MEXC)
    TAKER_FEE = 0.0006  # 0.06% taker fee (MEXC)
    
    def __init__(self, initial_balance_usdt: float = 10000):
        self.balance = {
            'USDT': initial_balance_usdt,
        }
        self.positions = {}  # {symbol: {amount, avg_price, side}}
        self.orders = []  # История ордеров
        self.trades = []  # История сделок
        self.order_id_counter = 1
        self.total_fees_paid = 0  # Отслеживаем комиссии
        
    def get_balance(self) -> Dict:
        return self.balance.copy()
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, price: Optional[float] = None) -> Dict:
        """Эмуляция размещения ордера"""
        order = {
            'orderId': f'DEMO_{self.order_id_counter}',
            'symbol': symbol,
            'side': side,  # BUY / SELL
            'type': order_type,  # LIMIT / MARKET
            'quantity': quantity,
            'price': price,
            'status': 'FILLED',  # Для демо сразу исполняем
            'timestamp': int(time.time() * 1000),
        }
        self.order_id_counter += 1
        self.orders.append(order)
        
        # Эмулируем исполнение
        self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Dict):
        """Исполнение ордера — обновление баланса и позиций"""
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order['price']
        order_type = order.get('type', 'MARKET')
        
        base_asset = symbol.replace('USDT', '')
        
        if side == 'BUY':
            cost = quantity * price
            fee = cost * (self.TAKER_FEE if order_type == 'MARKET' else self.MAKER_FEE)
            total_cost = cost + fee
            
            if self.balance.get('USDT', 0) >= total_cost:
                self.balance['USDT'] -= total_cost
                self.balance[base_asset] = self.balance.get(base_asset, 0) + quantity
                self.total_fees_paid += fee
                
                trade = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'cost': cost,
                    'fee': fee,
                    'timestamp': order['timestamp'],
                }
                self.trades.append(trade)
            else:
                order['status'] = 'REJECTED'
                order['error'] = 'Insufficient balance'
        
        elif side == 'SELL':
            if self.balance.get(base_asset, 0) >= quantity:
                self.balance[base_asset] -= quantity
                proceeds = quantity * price
                fee = proceeds * (self.TAKER_FEE if order_type == 'MARKET' else self.MAKER_FEE)
                net_proceeds = proceeds - fee
                self.balance['USDT'] = self.balance.get('USDT', 0) + net_proceeds
                self.total_fees_paid += fee
                
                trade = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'proceeds': proceeds,
                    'fee': fee,
                    'net_proceeds': net_proceeds,
                    'timestamp': order['timestamp'],
                }
                self.trades.append(trade)
            else:
                order['status'] = 'REJECTED'
                order['error'] = 'Insufficient asset balance'
    
    def get_trades(self) -> List[Dict]:
        return self.trades.copy()


class MEXCClient:
    """MEXC API Client с поддержкой демо-режима"""
    
    BASE_URL = 'https://api.mexc.com'
    FUTURES_URL = 'https://futures.mexc.com/api/v1'
    
    def __init__(self, api_key: str, secret_key: str, demo_mode: bool = True, use_futures: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.demo_mode = demo_mode
        self.use_futures = use_futures
        
        if demo_mode:
            initial_balance = float(os.getenv('DEMO_BALANCE_USDT', 10000))
            self.demo_account = MEXCDemoAccount(initial_balance)
            print(f"🎮 DEMO MODE активен. Виртуальный баланс: {initial_balance} USDT")
        else:
            self.demo_account = None
            print("⚠️  REAL MODE активен!")
    
    def _sign(self, params: Dict) -> str:
        """Подпись запроса"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def get_ticker_price(self, symbol: str) -> float:
        """Получить текущую цену (реальная, не демо)"""
        url = f"{self.BASE_URL}/api/v3/ticker/price"
        params = {'symbol': symbol}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return float(data['price'])
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> List:
        """Получить свечи (реальные данные)"""
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_balance(self) -> Dict:
        """Получить баланс"""
        if self.demo_mode:
            return self.demo_account.get_balance()
        
        # TODO: Реальный API запрос (требует подписи)
        # Пока не реализовано для безопасности
        raise NotImplementedError("Real balance API not implemented yet")
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, price: Optional[float] = None) -> Dict:
        """Разместить ордер"""
        if self.demo_mode:
            # В демо-режиме берём текущую рыночную цену
            if price is None:
                price = self.get_ticker_price(symbol)
            
            return self.demo_account.place_order(symbol, side, order_type, quantity, price)
        
        # TODO: Реальный API запрос
        raise NotImplementedError("Real order API not implemented yet — safety first")
    
    def get_trades(self) -> List[Dict]:
        """Получить историю сделок"""
        if self.demo_mode:
            return self.demo_account.get_trades()
        
        raise NotImplementedError("Real trades API not implemented yet")


def main():
    """Тест клиента"""
    client = MEXCClient(
        api_key=os.getenv('MEXC_API_KEY'),
        secret_key=os.getenv('MEXC_SECRET_KEY'),
        demo_mode=True
    )
    
    print("\n=== Тест MEXC Client ===\n")
    
    # Проверка получения цены
    print("Получаю текущую цену BTC...")
    btc_price = client.get_ticker_price('BTCUSDT')
    print(f"BTC/USDT: ${btc_price:,.2f}")
    
    # Баланс
    print("\nБаланс:")
    balance = client.get_balance()
    print(json.dumps(balance, indent=2))
    
    # Тестовая покупка
    print(f"\nПокупаю 0.001 BTC по рынку (~${btc_price * 0.001:.2f})...")
    order = client.place_order('BTCUSDT', 'BUY', 'MARKET', 0.001)
    print(f"Ордер: {order['orderId']} — {order['status']}")
    
    # Баланс после покупки
    print("\nБаланс после покупки:")
    balance = client.get_balance()
    print(json.dumps(balance, indent=2))
    
    # История сделок
    print("\nИстория сделок:")
    trades = client.get_trades()
    for trade in trades:
        print(f"  {trade['side']} {trade['quantity']} @ ${trade['price']:.2f}")

if __name__ == '__main__':
    main()
