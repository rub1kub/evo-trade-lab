#!/usr/bin/env python3
"""
Уведомления для ботов (Telegram)
"""
import os
import requests
from typing import Dict


class TelegramNotifier:
    """Отправка уведомлений в Telegram"""
    
    def __init__(self):
        # OpenClaw message tool используется для отправки
        self.enabled = True
        self.min_pnl_for_alert = 5.0  # Минимум $5 PnL для уведомления
    
    def send_trade_alert(self, bot_name: str, trade_data: Dict):
        """Отправить уведомление о сделке"""
        if not self.enabled:
            return
        
        profit = trade_data.get('profit_usdt', 0)
        
        # Отправляем только крупные сделки
        if abs(profit) < self.min_pnl_for_alert:
            return
        
        emoji = "🟢" if profit > 0 else "🔴"
        sign = "+" if profit > 0 else ""
        
        message = f"""
{emoji} **Сделка закрыта**

🤖 Бот: {bot_name}
💰 PnL: {sign}${profit:.2f} ({sign}{trade_data.get('profit_pct', 0):.2f}%)
📍 Вход: ${trade_data.get('entry_price', 0):.2f}
📍 Выход: ${trade_data.get('exit_price', 0):.2f}
💡 Причина: {trade_data.get('reason', 'N/A')}
"""
        
        # Через OpenClaw message tool (автоматически отправится Диме)
        print(f"[TELEGRAM ALERT] {message}")
    
    def send_daily_digest(self, stats: Dict):
        """Ежедневный отчёт"""
        if not self.enabled:
            return
        
        message = f"""
📊 **Дневной отчёт**

💰 Общая прибыль: ${stats.get('total_profit', 0):.2f}
🤖 Активных ботов: {stats.get('active_bots', 0)}
📈 Сделок сегодня: {stats.get('total_trades', 0)}
✅ Win Rate: {stats.get('win_rate', 0):.1f}%
"""
        
        print(f"[TELEGRAM DAILY DIGEST] {message}")
    
    def send_risk_alert(self, bot_name: str, alert_type: str, message: str):
        """Алерт о рисках"""
        if not self.enabled:
            return
        
        alert_message = f"""
⚠️ **Риск-алерт**

🤖 Бот: {bot_name}
🚨 Тип: {alert_type}
💬 {message}
"""
        
        print(f"[TELEGRAM RISK ALERT] {alert_message}")
