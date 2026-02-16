#!/usr/bin/env python3
"""
Менеджер ботов - управление отдельными балансами и автоматическое создание
"""
import os
from typing import Dict
from mexc_client import MEXCClient, MEXCDemoAccount
from dotenv import load_dotenv

load_dotenv()


class BotManager:
    """Управление ботами с отдельными демо-счетами"""
    
    def __init__(self, api_key: str, secret_key: str, demo_mode: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.demo_mode = demo_mode
        
        # Отдельные демо-счета для каждого бота
        self.bot_accounts: Dict[int, MEXCDemoAccount] = {}
        
        # Общий клиент для получения рыночных данных
        self.market_client = MEXCClient(api_key, secret_key, demo_mode=True)
    
    def create_bot_account(self, bot_id: int, initial_balance: float = 1000) -> MEXCDemoAccount:
        """Создать отдельный демо-счёт для бота"""
        if bot_id in self.bot_accounts:
            return self.bot_accounts[bot_id]
        
        account = MEXCDemoAccount(initial_balance_usdt=initial_balance)
        self.bot_accounts[bot_id] = account
        
        return account
    
    def get_bot_account(self, bot_id: int) -> MEXCDemoAccount:
        """Получить демо-счёт бота"""
        if bot_id not in self.bot_accounts:
            raise ValueError(f"Bot {bot_id} account not found")
        return self.bot_accounts[bot_id]
    
    def get_bot_balance(self, bot_id: int) -> Dict:
        """Получить баланс бота"""
        account = self.get_bot_account(bot_id)
        return account.get_balance()
    
    def place_bot_order(self, bot_id: int, symbol: str, side: str, 
                        order_type: str, quantity: float, price: float = None) -> Dict:
        """Разместить ордер от имени бота"""
        account = self.get_bot_account(bot_id)
        
        # В демо-режиме берём текущую рыночную цену если не указана
        if price is None:
            price = self.market_client.get_ticker_price(symbol)
        
        return account.place_order(symbol, side, order_type, quantity, price)
    
    def get_bot_trades(self, bot_id: int):
        """Получить историю сделок бота"""
        account = self.get_bot_account(bot_id)
        return account.get_trades()


# Глобальный менеджер
_bot_manager = None


def get_bot_manager() -> BotManager:
    """Получить глобальный менеджер ботов"""
    global _bot_manager
    if _bot_manager is None:
        _bot_manager = BotManager(
            api_key=os.getenv('MEXC_API_KEY'),
            secret_key=os.getenv('MEXC_SECRET_KEY'),
            demo_mode=os.getenv('DEMO_MODE', 'true').lower() == 'true'
        )
    return _bot_manager
