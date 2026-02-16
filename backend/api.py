#!/usr/bin/env python3
"""
FastAPI backend для торговой платформы (v2 с улучшениями)
"""
import os
import copy
import random
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from mexc_client import MEXCClient
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_strategy import BollingerStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from bot_v2 import TradingBotV2
from bot_manager import get_bot_manager
from trading_modes import get_mode
from auto_create_bots_v3 import BOT_CONFIGS_V3 as BOT_CONFIGS
from analytics import BotAnalytics, PortfolioAnalytics

load_dotenv()

app = FastAPI(title="Trading Platform API v2")

# --- schema ensure (for existing sqlite dbs) ---

def ensure_ml_schema():
    """Ensure required tables exist in ML DB (sqlite/postgres)."""
    try:
        # postgres handled by migrations elsewhere; still attempt create-if-not-exists
        if DB_BACKEND == 'postgres':
            conn = psycopg2.connect(**PG_CONFIG)
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS trade_events (
                    id BIGSERIAL PRIMARY KEY,
                    bot_id INTEGER,
                    bot_name TEXT,
                    symbol TEXT NOT NULL,
                    market_type TEXT DEFAULT 'futures',
                    event_type TEXT NOT NULL,
                    side TEXT,
                    ts_exchange_ms BIGINT,
                    ts_server_ms BIGINT,
                    price DOUBLE PRECISION,
                    qty DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    fee DOUBLE PRECISION,
                    order_id TEXT,
                    client_order_id TEXT,
                    trade_id TEXT,
                    position_id TEXT,
                    reduce_only BOOLEAN,
                    is_tp BOOLEAN,
                    is_sl BOOLEAN,
                    reason TEXT,
                    raw_json JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_bot_symbol_ts ON trade_events(bot_id, symbol, ts_exchange_ms)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_type_ts ON trade_events(event_type, ts_exchange_ms)')
            conn.commit(); conn.close()
            return

        # sqlite
        os.makedirs(os.path.dirname(ML_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(ML_DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER,
                bot_name TEXT,
                symbol TEXT NOT NULL,
                market_type TEXT DEFAULT 'futures',
                event_type TEXT NOT NULL,
                side TEXT,
                ts_exchange_ms INTEGER,
                ts_server_ms INTEGER,
                price REAL,
                qty REAL,
                pnl REAL,
                fee REAL,
                order_id TEXT,
                client_order_id TEXT,
                trade_id TEXT,
                position_id TEXT,
                reduce_only INTEGER,
                is_tp INTEGER,
                is_sl INTEGER,
                reason TEXT,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_bot_symbol_ts ON trade_events(bot_id, symbol, ts_exchange_ms)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_type_ts ON trade_events(event_type, ts_exchange_ms)')
        conn.commit(); conn.close()
    except Exception:
        pass

ensure_ml_schema()

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты
bot_manager = get_bot_manager()
market_client = bot_manager.market_client

bots: Dict[int, TradingBotV2] = {}
bot_threads: Dict[int, threading.Thread] = {}
next_bot_id = 1

ML_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_data.db')
DB_BACKEND = os.getenv('DB_BACKEND', 'sqlite').lower()
PG_CONFIG = {
    'host': os.getenv('PGHOST', '127.0.0.1'),
    'port': int(os.getenv('PGPORT', '5432')),
    'dbname': os.getenv('PGDATABASE', 'trading_ml'),
    'user': os.getenv('PGUSER', 'trading'),
    'password': os.getenv('PGPASSWORD', ''),
}

PERSISTED_STATS_CACHE: Dict[str, Dict] = {}
PERSISTED_STATS_CACHE_TS = 0.0
PERSISTED_STATS_CACHE_TTL_SEC = 30.0


def load_persisted_trade_stats_by_bot_name() -> Dict[str, Dict]:
    """Загружает агрегированную историю сделок (PostgreSQL/SQLite)."""
    conn = None
    try:
        if DB_BACKEND == 'postgres':
            conn = psycopg2.connect(**PG_CONFIG)
        else:
            if not os.path.exists(ML_DB_PATH):
                return {}
            # read-only подключение + короткий timeout, чтобы не зависать на lock
            conn = sqlite3.connect(f"file:{ML_DB_PATH}?mode=ro", uri=True, timeout=0.2)

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                bot_name,
                COUNT(*) as total_trades,
                SUM(CASE WHEN COALESCE(pnl, 0) > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN COALESCE(pnl, 0) <= 0 THEN 1 ELSE 0 END) as losing_trades,
                COALESCE(SUM(COALESCE(pnl, 0)), 0) as total_profit_usdt
            FROM bot_trades
            GROUP BY bot_name
            """
        )

        out: Dict[str, Dict] = {}
        for bot_name, total_trades, winning_trades, losing_trades, total_profit_usdt in cursor.fetchall():
            total_trades = int(total_trades or 0)
            winning_trades = int(winning_trades or 0)
            losing_trades = int(losing_trades or 0)
            total_profit_usdt = float(total_profit_usdt or 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            out[bot_name] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_profit_usdt': total_profit_usdt,
                'win_rate': win_rate,
            }
        return out
    except Exception:
        return {}
    finally:
        if conn:
            conn.close()


def get_persisted_trade_stats_cached(force: bool = False) -> Dict[str, Dict]:
    """Кеш persisted-статистики, чтобы не бить SQLite на каждый API-запрос."""
    global PERSISTED_STATS_CACHE, PERSISTED_STATS_CACHE_TS

    now = time.time()
    if not force and PERSISTED_STATS_CACHE and (now - PERSISTED_STATS_CACHE_TS) < PERSISTED_STATS_CACHE_TTL_SEC:
        return PERSISTED_STATS_CACHE

    fresh = load_persisted_trade_stats_by_bot_name()
    if fresh:
        PERSISTED_STATS_CACHE = fresh
        PERSISTED_STATS_CACHE_TS = now

    # если БД временно залочена — отдаем предыдущий кеш, а не пустоту
    return PERSISTED_STATS_CACHE


def merge_runtime_and_persisted_stats(bot_name: str, runtime_stats: Dict, persisted_map: Dict[str, Dict]) -> Dict:
    """Для UI берём persisted-статы (история), чтобы после рестарта не было нулей."""
    merged = dict(runtime_stats or {})
    persisted = persisted_map.get(bot_name)
    if persisted:
        merged['total_trades'] = int(persisted.get('total_trades', merged.get('total_trades', 0) or 0))
        merged['winning_trades'] = int(persisted.get('winning_trades', merged.get('winning_trades', 0) or 0))
        merged['losing_trades'] = int(persisted.get('losing_trades', merged.get('losing_trades', 0) or 0))
        merged['total_profit_usdt'] = float(persisted.get('total_profit_usdt', merged.get('total_profit_usdt', 0) or 0))
        merged['win_rate'] = float(persisted.get('win_rate', merged.get('win_rate', 0) or 0))
    return merged


def load_recent_trades_from_db(bot_name: str, limit: int = 10) -> List[Dict]:
    """Последние сделки бота из БД (для UI после рестарта)."""
    if not bot_name:
        return []

    conn = None
    try:
        if DB_BACKEND == 'postgres':
            conn = psycopg2.connect(**PG_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT entry_time, exit_time, entry_price, exit_price, quantity, pnl, pnl_percent, exit_reason,
                       market_type, leverage, margin, notional, open_fee, close_fee
                FROM bot_trades
                WHERE bot_name = %s
                ORDER BY COALESCE(exit_time, entry_time) DESC
                LIMIT %s
                """,
                (bot_name, int(limit))
            )
        else:
            if not os.path.exists(ML_DB_PATH):
                return []
            conn = sqlite3.connect(f"file:{ML_DB_PATH}?mode=ro", uri=True, timeout=0.2)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT entry_time, exit_time, entry_price, exit_price, quantity, pnl, pnl_percent, exit_reason,
                       NULL as market_type, NULL as leverage, NULL as margin, NULL as notional, NULL as open_fee, NULL as close_fee
                FROM bot_trades
                WHERE bot_name = ?
                ORDER BY COALESCE(exit_time, entry_time) DESC
                LIMIT ?
                """,
                (bot_name, int(limit))
            )

        rows = []
        for entry_time, exit_time, entry_price, exit_price, quantity, pnl, pnl_percent, exit_reason, market_type, leverage, margin, notional, open_fee, close_fee in cursor.fetchall():
            rows.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': float(entry_price or 0),
                'exit_price': float(exit_price or 0),
                'quantity': float(quantity or 0),
                'pnl': float(pnl or 0),
                'profit_usdt': float(pnl or 0),
                'profit_pct': float(pnl_percent or 0),
                'reason': exit_reason or 'DB history',
                'market_type': market_type or 'spot',
                'leverage': int(leverage or 1),
                'margin': float(margin or 0),
                'notional': float(notional or 0),
                'open_fee': float(open_fee or 0),
                'close_fee': float(close_fee or 0),
            })

        return rows
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


# Models
class BotCreate(BaseModel):
    name: str
    symbol: str
    strategy_type: str = 'rsi'
    mode: str = 'balanced'
    initial_balance: float = 1000
    # RSI
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger
    bb_period: int = 20
    bb_std_dev: float = 2.0
    # EMA
    ema_fast: int = 12
    ema_slow: int = 26
    leverage: int = 2


class BotTune(BaseModel):
    mode: str | None = None
    leverage: int | None = None
    trailing_activation_pct: float | None = None
    trailing_stop_pct: float | None = None
    breakeven_activation_pct: float | None = None
    breakeven_buffer_pct: float | None = None


# API Endpoints

@app.get("/")
async def root():
    return {"status": "ok", "message": "Trading Platform API v2"}


@app.get("/balance")
async def get_balance():
    """Получить общий баланс всех ботов"""
    try:
        total_usdt = 0
        balances = {}
        
        for bot_id, bot in bots.items():
            bot_balance = bot_manager.get_bot_balance(bot_id)
            balances[f'Bot_{bot_id}'] = bot_balance
            total_usdt += bot_balance.get('USDT', 0)
        
        return {
            "success": True,
            "total_usdt": round(total_usdt, 2),
            "bots": balances
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/price/{symbol}")
async def get_price(symbol: str):
    """Получить текущую цену"""
    try:
        price = market_client.get_ticker_price(symbol)
        return {"success": True, "symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/klines/{symbol}")
async def get_klines(symbol: str, interval: str = '5m', limit: int = 100):
    """Получить свечи"""
    try:
        klines = market_client.get_klines(symbol, interval, limit)
        return {"success": True, "symbol": symbol, "klines": klines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bots")
async def list_bots():
    """Список всех ботов (краткая версия)"""
    bot_list = []
    persisted_map = get_persisted_trade_stats_cached()

    for bot_id, bot in bots.items():
        status = bot.get_status()
        merged_stats = merge_runtime_and_persisted_stats(status['name'], status.get('stats', {}), persisted_map)

        # Убираем тяжёлые данные для списка
        bot_list.append({
            'bot_id': status['bot_id'],
            'name': status['name'],
            'symbol': status['symbol'],
            'strategy': {'name': status['strategy']['name'], 'description': status['strategy']['description']},
            'mode': status['mode'],  # Полный mode для TP/SL
            'is_running': status['is_running'],
            'current_position': status['current_position'],
            'total_value_usdt': status['total_value_usdt'],
            'total_fees_paid': status.get('total_fees_paid', 0),
            'stats': merged_stats,
        })

    return {"success": True, "bots": bot_list}


@app.get("/research/bots")
async def list_bots_research():
    """Лёгкий research endpoint: только данные для рейтинга/мониторинга.
    ВАЖНО: без get_status() (он тяжёлый и может ходить в price API).
    """
    t0 = time.perf_counter()
    bot_list = []
    strategy_stats = {}
    persisted_map = get_persisted_trade_stats_cached()

    for bot_id, bot in bots.items():
        # Берём лёгкие поля + подмешиваем persisted trade-статы из БД
        bot_name = getattr(bot, 'name', f'Bot {bot_id}')
        runtime_stats = getattr(bot, 'stats', {}) or {}
        stats = merge_runtime_and_persisted_stats(bot_name, runtime_stats, persisted_map)
        strategy_name = getattr(getattr(bot, 'strategy', None), 'name', 'Unknown')
        mode_name = getattr(getattr(bot, 'mode', None), 'name', 'Unknown')

        # Research score: прибыль + winrate + статистическая значимость
        score = (float(stats.get('total_profit_usdt', 0) or 0) * 8) + \
                (float(stats.get('win_rate', 0) or 0) * 0.6) + \
                (min(int(stats.get('total_trades', 0) or 0), 20) * 0.5)

        bot_item = {
            'bot_id': bot_id,
            'name': getattr(bot, 'name', f'Bot {bot_id}'),
            'symbol': getattr(bot, 'symbol', ''),
            'strategy': {'name': strategy_name},
            'mode': {'name': mode_name},
            'is_running': bool(getattr(bot, 'is_running', False)),
            'has_position': getattr(bot, 'current_position', None) is not None,
            'stats': {
                'total_profit_usdt': float(stats.get('total_profit_usdt', 0) or 0),
                'win_rate': float(stats.get('win_rate', 0) or 0),
                'total_trades': int(stats.get('total_trades', 0) or 0)
            },
            'score': round(score, 2)
        }
        bot_list.append(bot_item)

        # Агрегация по стратегиям
        if strategy_name not in strategy_stats:
            strategy_stats[strategy_name] = {
                'name': strategy_name,
                'pnl': 0.0,
                'trades': 0,
                'bots': 0,
                'win_rate_sum': 0.0
            }

        strategy_stats[strategy_name]['pnl'] += bot_item['stats']['total_profit_usdt']
        strategy_stats[strategy_name]['trades'] += bot_item['stats']['total_trades']
        strategy_stats[strategy_name]['bots'] += 1
        strategy_stats[strategy_name]['win_rate_sum'] += bot_item['stats']['win_rate']

    # Сортировка ботов по score
    bot_list.sort(key=lambda x: x['score'], reverse=True)

    # Топ стратегий
    leaders = []
    for s in strategy_stats.values():
        avg_wr = s['win_rate_sum'] / s['bots'] if s['bots'] > 0 else 0
        strategy_score = (s['pnl'] * 10) + (s['trades'] * 0.4) + (avg_wr * 0.2)
        leaders.append({
            'name': s['name'],
            'pnl': round(s['pnl'], 4),
            'trades': s['trades'],
            'avg_win_rate': round(avg_wr, 2),
            'score': round(strategy_score, 2)
        })

    leaders.sort(key=lambda x: x['score'], reverse=True)

    server_ms = (time.perf_counter() - t0) * 1000

    return {
        "success": True,
        "bots": bot_list,
        "strategy_leaders": leaders[:5],
        "meta": {
            "server_ms": max(1.0, round(server_ms, 1))
        }
    }


@app.post("/bots")
async def create_bot(bot_data: BotCreate):
    """Создать нового бота"""
    global next_bot_id
    
    try:
        # Создаём стратегию
        if bot_data.strategy_type == 'rsi':
            strategy = RSIStrategy(
                rsi_oversold=bot_data.rsi_oversold,
                rsi_overbought=bot_data.rsi_overbought
            )
        elif bot_data.strategy_type == 'macd':
            strategy = MACDStrategy(
                fast_period=bot_data.macd_fast,
                slow_period=bot_data.macd_slow,
                signal_period=bot_data.macd_signal
            )
        elif bot_data.strategy_type == 'bollinger':
            strategy = BollingerStrategy(
                period=bot_data.bb_period,
                std_dev=bot_data.bb_std_dev
            )
        elif bot_data.strategy_type == 'ema':
            strategy = EMACrossoverStrategy(
                fast_period=bot_data.ema_fast,
                slow_period=bot_data.ema_slow
            )
        else:
            raise HTTPException(status_code=400, detail="Unknown strategy type")
        
        # Создаём бота
        bot = TradingBotV2(
            bot_id=next_bot_id,
            name=bot_data.name,
            symbol=bot_data.symbol,
            strategy=strategy,
            mode_name=bot_data.mode,
            initial_balance=bot_data.initial_balance,
            leverage=int(getattr(bot_data, 'leverage', None) or os.getenv('DEFAULT_LEVERAGE', '2'))
        )
        
        bots[next_bot_id] = bot
        next_bot_id += 1
        
        return {"success": True, "bot": bot.get_status()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bots/auto-create")
async def auto_create_bots():
    """Автоматически создать предустановленных ботов"""
    global next_bot_id
    
    try:
        created = []
        
        for config in BOT_CONFIGS:
            bot = TradingBotV2(
                bot_id=next_bot_id,
                name=config['name'],
                symbol=config['symbol'],
                strategy=config['strategy'],
                mode_name=config['mode'],
                initial_balance=config['balance'],
                leverage=int(config.get('leverage', os.getenv('DEFAULT_LEVERAGE', '2'))),
                interval=config.get('interval', '5m')  # Интервал анализа
            )
            
            bots[next_bot_id] = bot
            created.append(bot.get_status())
            next_bot_id += 1
        
        return {"success": True, "created": len(created), "bots": created}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _mode_key_from_mode_name(mode_name: str) -> str:
    n = (mode_name or '').strip().lower().replace(' ', '')
    mapping = {
        'conservative': 'conservative',
        'balanced': 'balanced',
        'balancedplus': 'balanced_plus',
        'aggressive': 'aggressive',
        'degen': 'degen',
        'scalp': 'scalp',
    }
    return mapping.get(n, 'balanced')


def _start_bot_worker(bot_id: int, bot: TradingBotV2):
    if bot.is_running:
        return

    bot.start()

    def bot_loop(b=bot):
        while b.is_running:
            b.tick()
            time.sleep(20)

    thread = threading.Thread(target=bot_loop, daemon=True)
    thread.start()
    bot_threads[bot_id] = thread


def _mutate_strategy(strategy):
    s = copy.deepcopy(strategy)

    def _jitter(attr: str, step: float, lo: float | None = None, hi: float | None = None):
        if not hasattr(s, attr):
            return
        try:
            base = float(getattr(s, attr))
            val = base + random.uniform(-step, step)
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            # ints stay ints
            if isinstance(getattr(s, attr), int):
                val = int(round(val))
            setattr(s, attr, val)
        except Exception:
            pass

    # universal-ish mutations for known strategy params
    _jitter('rsi_oversold', step=3, lo=20, hi=48)
    _jitter('rsi_overbought', step=3, lo=52, hi=85)
    _jitter('rsi_period', step=2, lo=4, hi=30)

    _jitter('fast_period', step=2, lo=4, hi=20)
    _jitter('slow_period', step=3, lo=10, hi=60)
    _jitter('signal_period', step=1, lo=3, hi=20)

    _jitter('shock_z_threshold', step=0.35, lo=1.2, hi=3.6)
    _jitter('reverse_z_threshold', step=0.3, lo=0.8, hi=2.8)

    _jitter('position_size_pct', step=6, lo=10, hi=95)

    return s


@app.post('/bots/evolve')
async def evolve_bots(clones_per_winner: int = 2, top_n: int = 5, min_trades: int = 4, max_new: int = 24):
    """Клонировать лучших ботов с мутациями параметров (эволюционный режим)."""
    global next_bot_id

    clones_per_winner = max(1, min(int(clones_per_winner), 6))
    top_n = max(1, min(int(top_n), 20))
    min_trades = max(0, min(int(min_trades), 1000))
    max_new = max(1, min(int(max_new), 80))

    # кандидаты: уже торгуют и в плюсе
    persisted_map = get_persisted_trade_stats_cached()
    candidates = []
    for bot_id, bot in list(bots.items()):
        if not getattr(bot, 'is_running', False):
            continue
        raw_stats = getattr(bot, 'stats', {}) or {}
        stats = merge_runtime_and_persisted_stats(getattr(bot, 'name', ''), raw_stats, persisted_map)

        pnl = float(stats.get('total_profit_usdt', 0) or 0)
        trades = int(stats.get('total_trades', 0) or 0)
        wr = float(stats.get('win_rate', 0) or 0)
        if trades >= min_trades and pnl > 0:
            score = (pnl * 12.0) + (wr * 0.25) + (trades * 0.1)
            candidates.append((score, bot_id, bot, pnl, trades, wr))

    if not candidates:
        return {'success': False, 'message': 'No winners to clone yet'}

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_n]

    created = []
    active_now = sum(1 for b in bots.values() if getattr(b, 'is_running', False))

    for _, base_id, base_bot, _, _, _ in top:
        if len(created) >= max_new:
            break
        for _i in range(clones_per_winner):
            if len(created) >= max_new:
                break

            # ограничим общий рост, чтобы не убить API
            if active_now + len(created) >= 120:
                break

            strategy = _mutate_strategy(base_bot.strategy)

            mode_key = _mode_key_from_mode_name(getattr(base_bot.mode, 'name', 'balanced'))
            mode_pool = [mode_key, 'scalp', 'aggressive', 'degen']
            mode_name = random.choice(mode_pool)

            leverage = max(2, min(20, int(getattr(base_bot, 'leverage', 5) + random.choice([-1, 0, 1, 2]))))
            seed_balance = 120
            try:
                bal = bot_manager.get_bot_balance(base_id)
                seed_balance = max(80, min(350, float((bal or {}).get('USDT', 120))))
            except Exception:
                pass

            clone_id = next_bot_id
            next_bot_id += 1
            clone_name = f"EVO-{base_bot.name}#{clone_id}"

            clone = TradingBotV2(
                bot_id=clone_id,
                name=clone_name,
                symbol=base_bot.symbol,
                strategy=strategy,
                mode_name=mode_name,
                initial_balance=seed_balance,
                leverage=leverage,
                interval=getattr(base_bot, 'interval', '1m')
            )
            bots[clone_id] = clone
            _start_bot_worker(clone_id, clone)

            created.append({
                'bot_id': clone_id,
                'name': clone_name,
                'symbol': clone.symbol,
                'mode': mode_name,
                'leverage': leverage,
                'parent_bot_id': base_id,
            })

    return {'success': True, 'created': len(created), 'clones': created}


@app.post("/bots/{bot_id}/start")
async def start_bot(bot_id: int):
    """Запустить бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    
    if bot.is_running:
        return {"success": False, "message": "Bot already running"}
    
    bot.start()
    
    # Запускаем в отдельном потоке
    def bot_loop():
        while bot.is_running:
            bot.tick()
            time.sleep(20)  # Проверка каждые 20 сек для более частых входов
    
    thread = threading.Thread(target=bot_loop, daemon=True)
    thread.start()
    bot_threads[bot_id] = thread
    
    return {"success": True, "bot": bot.get_status()}


@app.post("/bots/start-all")
async def start_all_bots():
    """Запустить всех ботов"""
    started = []
    
    for bot_id, bot in bots.items():
        if not bot.is_running:
            bot.start()
            
            def bot_loop(b=bot):
                while b.is_running:
                    b.tick()
                    time.sleep(20)
            
            thread = threading.Thread(target=bot_loop, daemon=True)
            thread.start()
            bot_threads[bot_id] = thread
            
            started.append(bot_id)
    
    return {"success": True, "started": started}


@app.post("/bots/{bot_id}/stop")
async def stop_bot(bot_id: int):
    """Остановить бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    bot.stop()
    
    return {"success": True, "bot": bot.get_status()}


@app.get("/bots/{bot_id}")
async def get_bot(bot_id: int):
    """Получить статус бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")

    bot = bots[bot_id]
    status = bot.get_status()

    persisted_map = get_persisted_trade_stats_cached()
    status['stats'] = merge_runtime_and_persisted_stats(status['name'], status.get('stats', {}), persisted_map)

    if not status.get('trades_history'):
        status['trades_history'] = load_recent_trades_from_db(status['name'], limit=10)

    return {"success": True, "bot": status}


@app.post('/bots/{bot_id}/tune')
async def tune_bot(bot_id: int, tune: BotTune):
    """Live tuning for mode/leverage/risk params (profit-first ops)."""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail='Bot not found')

    bot = bots[bot_id]

    if tune.mode:
        bot.mode = get_mode(tune.mode)

    if tune.leverage is not None:
        bot.leverage = max(1, min(int(tune.leverage), 25))

    # risk manager knobs
    rm = getattr(bot, 'risk_manager', None)
    if rm is not None:
        if tune.trailing_activation_pct is not None:
            rm.trailing_activation_pct = float(tune.trailing_activation_pct)
        if tune.trailing_stop_pct is not None:
            rm.trailing_stop_pct = float(tune.trailing_stop_pct)
        if tune.breakeven_activation_pct is not None:
            rm.breakeven_activation_pct = float(tune.breakeven_activation_pct)
        if tune.breakeven_buffer_pct is not None:
            rm.breakeven_buffer_pct = float(tune.breakeven_buffer_pct)

    return {
        'success': True,
        'bot_id': bot_id,
        'mode': bot.mode.get_config() if hasattr(bot.mode, 'get_config') else {'name': getattr(bot.mode, 'name', 'unknown')},
        'leverage': bot.leverage,
        'risk': rm.get_config() if rm and hasattr(rm, 'get_config') else {},
    }


@app.get("/bots/{bot_id}/trades")
async def get_bot_trades(bot_id: int):
    """Получить сделки бота (legacy)."""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")

    trades = bot_manager.get_bot_trades(bot_id)
    return {"success": True, "trades": trades}


@app.get('/events')
async def get_trade_events(bot_id: int = None, symbol: str = None, limit: int = 500):
    """Trade events for accurate chart markers/debug.

    Args:
      bot_id: optional
      symbol: optional
      limit: max rows
    """
    limit = max(1, min(int(limit or 500), 5000))

    conn = None
    try:
        # same backend selection as persisted stats
        if DB_BACKEND == 'postgres':
            conn = psycopg2.connect(**PG_CONFIG)
        else:
            if not os.path.exists(ML_DB_PATH):
                return {"success": True, "events": []}
            conn = sqlite3.connect(ML_DB_PATH, timeout=1)

        cursor = conn.cursor()

        where = []
        params = []
        if bot_id is not None:
            where.append('bot_id = %s' if DB_BACKEND == 'postgres' else 'bot_id = ?')
            params.append(int(bot_id))
        if symbol:
            where.append('symbol = %s' if DB_BACKEND == 'postgres' else 'symbol = ?')
            params.append(symbol.upper())

        where_sql = ('WHERE ' + ' AND '.join(where)) if where else ''
        lim_sql = '%s' if DB_BACKEND == 'postgres' else '?'

        q = f"""
            SELECT bot_id, bot_name, symbol, event_type, side,
                   ts_exchange_ms, ts_server_ms, price, qty, pnl, fee,
                   order_id, client_order_id, trade_id, position_id,
                   reduce_only, is_tp, is_sl, reason
            FROM trade_events
            {where_sql}
            ORDER BY COALESCE(ts_exchange_ms, ts_server_ms) DESC
            LIMIT {lim_sql}
        """
        params.append(limit)
        cursor.execute(q, params)

        cols = [d[0] for d in cursor.description]
        events = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return {"success": True, "events": events}
    except Exception:
        return {"success": True, "events": []}
    finally:
        if conn:
            conn.close()


@app.delete("/bots/{bot_id}")
async def delete_bot(bot_id: int):
    """Удалить бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    
    if bot.is_running:
        bot.stop()
    
    del bots[bot_id]
    
    return {"success": True, "message": "Bot deleted"}


@app.post("/bots/{bot_id}/orders/limit")
async def place_limit_order(bot_id: int, side: str, quantity: float, price: float):
    """Разместить лимитный ордер"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    order = bot.place_limit_order(side.upper(), quantity, price)
    
    return {"success": True, "order": order}


@app.delete("/bots/{bot_id}/orders/{order_id}")
async def cancel_limit_order(bot_id: int, order_id: str):
    """Отменить лимитный ордер"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    success = bot.cancel_limit_order(order_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Order not found or already filled/cancelled")
    
    return {"success": True, "message": "Order cancelled"}


@app.get("/analytics/portfolio")
async def get_portfolio_analytics():
    """Аналитика портфеля"""
    all_bots_list = list(bots.values())
    
    if not all_bots_list:
        return {"success": True, "data": {}}
    
    analytics = PortfolioAnalytics(all_bots_list)
    
    return {
        "success": True,
        "data": {
            "equity_curve": analytics.get_total_equity_curve(),
            "top_by_profit": analytics.get_top_performers(by='profit', top_n=5),
            "top_by_winrate": analytics.get_top_performers(by='winrate', top_n=5),
        }
    }


@app.get("/analytics/bot/{bot_id}")
async def get_bot_analytics(bot_id: int):
    """Аналитика конкретного бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    analytics = BotAnalytics(bot)
    
    return {
        "success": True,
        "data": {
            "equity_curve": analytics.get_equity_curve(),
            "performance_metrics": analytics.get_performance_metrics(),
            "hourly_heatmap": analytics.get_hourly_heatmap(),
            "daily_performance": analytics.get_daily_performance(days=7),
        }
    }


# ==== WEBSOCKET ====
from fastapi import WebSocket
from websocket_server import ws_manager, websocket_handler, event_emitter

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    await websocket_handler(websocket)


# ==== ADVANCED ORDERS ====
from advanced_orders import advanced_order_manager

# Wire advanced orders -> event stream (DB + websocket)
try:
    from ml_data_collector import ml_collector
except Exception:
    ml_collector = None

def _emit_trade_event(bot_id: int, event: Dict):
    """Persist to trade_events and broadcast."""
    try:
        bot = bots.get(bot_id)
        symbol = (bot.symbol if bot else event.get('symbol') or '').upper()
        if not symbol:
            return

        ts_server_ms = int(event.get('ts_server_ms') or (time.time() * 1000))
        ts_ex_ms = int(event.get('ts_exchange_ms') or ts_server_ms)
        ev_type = str(event.get('event_type') or '').upper()
        price = float(event.get('price') or 0)
        reason = event.get('reason')

        side = None
        qty = None
        bot_name = None
        if bot:
            bot_name = getattr(bot, 'name', None)
            if getattr(bot, 'current_position', None):
                side = bot.current_position.get('side')
                qty = bot.current_position.get('quantity')

        # persist
        if ml_collector is not None and ev_type and price:
            conn = ml_collector._get_conn()
            cur = conn.cursor()
            if DB_BACKEND == 'postgres':
                cur.execute('''
                    INSERT INTO trade_events
                    (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ''', (bot_id, bot_name, symbol, ev_type, side, ts_ex_ms, ts_server_ms, price, float(qty or 0), reason))
            else:
                cur.execute('''
                    INSERT INTO trade_events
                    (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, reason)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                ''', (bot_id, bot_name, symbol, ev_type, side, ts_ex_ms, ts_server_ms, price, float(qty or 0), reason))
            conn.commit(); conn.close()

        # broadcast a lightweight WS alert (optional)
        try:
            asyncio.create_task(event_emitter.emit_alert(bot_id, f"{ev_type} @ {price}", 'info'))
        except Exception:
            pass

    except Exception:
        pass

advanced_order_manager.on_event = _emit_trade_event

class MultiTPSetup(BaseModel):
    bot_id: int
    levels: List[dict]  # [{'percent': 2.0, 'close_percent': 33}, ...]

class TrailingStopSetup(BaseModel):
    bot_id: int
    activation_percent: float = 1.5
    callback_rate: float = 0.5

@app.post("/orders/multi-tp")
async def setup_multi_tp(setup: MultiTPSetup):
    """Настройка Multi Take Profit"""
    if setup.bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[setup.bot_id]
    if not bot.current_position:
        raise HTTPException(status_code=400, detail="No open position")
    
    tps = advanced_order_manager.setup_multi_tp(
        setup.bot_id, 
        bot.current_position['entry_price'],
        setup.levels
    )
    
    return {"success": True, "take_profit_levels": [
        {'level': tp.level, 'price': tp.price, 'close_percent': tp.percent_to_close}
        for tp in tps
    ]}

@app.post("/orders/trailing-stop")
async def setup_trailing_stop(setup: TrailingStopSetup):
    """Настройка Trailing Stop"""
    if setup.bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[setup.bot_id]
    if not bot.current_position:
        raise HTTPException(status_code=400, detail="No open position")
    
    ts = advanced_order_manager.setup_trailing_stop(
        setup.bot_id,
        bot.current_position['entry_price'],
        setup.activation_percent,
        setup.callback_rate
    )
    
    return {"success": True, "trailing_stop": {
        'activation_price': ts.activation_price,
        'callback_rate': ts.callback_rate
    }}

@app.get("/orders/bot/{bot_id}")
async def get_advanced_orders(bot_id: int):
    """Получить продвинутые ордера бота"""
    return {"success": True, "orders": advanced_order_manager.get_bot_orders_status(bot_id)}


# ==== RISK MANAGEMENT ====
from risk_manager_v2 import risk_manager

@app.get("/risk/portfolio")
async def get_portfolio_risk():
    """Анализ риска портфеля"""
    all_bots = [bots[bid].get_status() for bid in bots]
    return {"success": True, "risk": risk_manager.get_portfolio_status(all_bots)}

@app.get("/risk/bot/{bot_id}/position-size")
async def calculate_position_size(bot_id: int):
    """Расчёт оптимального размера позиции (Kelly)"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    status = bot.get_status()
    stats = status['stats']
    
    # Текущие позиции для корреляции
    current_positions = []
    for bid, b in bots.items():
        if b.current_position:
            pos = b.current_position
            current_positions.append({
                'symbol': b.symbol,
                'value': pos['entry_price'] * pos['quantity']
            })
    
    result = risk_manager.calculate_position_size(
        bot_id=bot_id,
        balance=status['total_value_usdt'],
        win_rate=stats['win_rate'] / 100,
        avg_win=stats.get('avg_win_pct', 2),
        avg_loss=stats.get('avg_loss_pct', 1),
        current_positions=current_positions,
        symbol=bot.symbol
    )
    
    return {"success": True, "position_sizing": result}

@app.get("/risk/daily/{bot_id}")
async def get_daily_stats(bot_id: int):
    """Дневная статистика и лимиты"""
    return {"success": True, "daily": risk_manager.daily_limit.get_daily_stats(bot_id)}


# ==== BACKTESTING ====
from backtesting import backtest_engine, MonteCarloSimulation

class BacktestRequest(BaseModel):
    symbol: str
    strategy_type: str = 'rsi'
    days: int = 30
    take_profit_pct: float = 3.0
    stop_loss_pct: float = 1.5
    position_size_pct: float = 10.0

@app.post("/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Запуск бэктеста стратегии"""
    # Создаём стратегию
    strategies = {
        'rsi': RSIStrategy,
        'macd': MACDStrategy,
        'bollinger': BollingerStrategy,
        'ema': EMACrossoverStrategy,
    }
    
    if request.strategy_type not in strategies:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy_type}")
    
    strategy = strategies[request.strategy_type]()
    strategy.name = request.strategy_type.upper()
    strategy.symbol = request.symbol
    
    # Получаем исторические данные
    klines = await backtest_engine.fetch_historical_data(request.symbol, '1h', request.days)
    
    if not klines:
        raise HTTPException(status_code=400, detail="Failed to fetch historical data")
    
    # Запускаем бэктест
    result = backtest_engine.run_backtest(
        strategy, klines,
        take_profit_pct=request.take_profit_pct,
        stop_loss_pct=request.stop_loss_pct,
        position_size_pct=request.position_size_pct
    )
    
    return {
        "success": True,
        "result": {
            "strategy": result.strategy_name,
            "symbol": result.symbol,
            "period": f"{result.start_date} - {result.end_date}",
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_pnl": result.total_pnl,
            "total_pnl_percent": result.total_pnl_percent,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_percent": result.max_drawdown_percent,
            "avg_trade_pnl": result.avg_trade_pnl,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_holding_time_hours": result.avg_holding_time_hours,
            "equity_curve": result.equity_curve[-50:],  # Последние 50 точек
        }
    }

@app.post("/backtest/monte-carlo/{bot_id}")
async def run_monte_carlo(bot_id: int, iterations: int = 1000):
    """Monte Carlo симуляция на основе сделок бота"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    status = bot.get_status()
    
    if not status['trades_history']:
        raise HTTPException(status_code=400, detail="Not enough trades for simulation")
    
    # Конвертируем сделки
    from backtesting import BacktestTrade
    trades = [
        BacktestTrade(
            entry_time=t.get('entry_time', ''),
            exit_time=t.get('exit_time', ''),
            entry_price=t.get('entry_price', 0),
            exit_price=t.get('exit_price', 0),
            quantity=t.get('quantity', 0),
            side='LONG',
            pnl=t.get('pnl', 0),
            pnl_percent=t.get('pnl_percent', 0),
            exit_reason=t.get('reason', 'SIGNAL')
        )
        for t in status['trades_history']
    ]
    
    result = MonteCarloSimulation.run(trades, iterations, status['total_value_usdt'])
    
    return {"success": True, "monte_carlo": result}


# ==== EXPORT ====
import csv
import io
from fastapi.responses import StreamingResponse

@app.get("/export/trades/{bot_id}")
async def export_trades_csv(bot_id: int):
    """Экспорт сделок в CSV"""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = bots[bot_id]
    status = bot.get_status()
    trades = status['trades_history']
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'Quantity', 'PnL', 'PnL %', 'Reason'])
    
    for t in trades:
        writer.writerow([
            t.get('entry_time', ''),
            t.get('exit_time', ''),
            t.get('entry_price', 0),
            t.get('exit_price', 0),
            t.get('quantity', 0),
            t.get('pnl', 0),
            t.get('pnl_percent', 0),
            t.get('reason', '')
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bot_{bot_id}_trades.csv"}
    )

@app.get("/export/portfolio")
async def export_portfolio_csv():
    """Экспорт всего портфеля в CSV"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Bot ID', 'Name', 'Symbol', 'Strategy', 'Mode', 'Balance', 'PnL', 'Trades', 'Win Rate', 'Status'])
    
    for bot_id, bot in bots.items():
        status = bot.get_status()
        writer.writerow([
            bot_id,
            status['name'],
            status['symbol'],
            status['strategy']['name'],
            status['mode']['name'],
            status['total_value_usdt'],
            status['stats']['total_profit_usdt'],
            status['stats']['total_trades'],
            status['stats']['win_rate'],
            'Running' if status['is_running'] else 'Stopped'
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio.csv"}
    )


# ==== LEADERBOARD ====

def _parse_period_to_cutoff_iso(period: str):
    """period examples: 24h, 7d, 30d, all"""
    period = (period or '24h').strip().lower()
    if period in ('all', '0', '0h', '0d'):
        return None

    mult = 1
    unit = period[-1]
    try:
        val = int(period[:-1])
    except Exception:
        val = 24
        unit = 'h'

    if unit == 'h':
        delta = timedelta(hours=val * mult)
    elif unit == 'd':
        delta = timedelta(days=val * mult)
    elif unit == 'w':
        delta = timedelta(days=7 * val * mult)
    else:
        delta = timedelta(hours=24)

    return (datetime.utcnow() - delta).isoformat()


def _safe_dt(x):
    if not x:
        return None
    if isinstance(x, datetime):
        return x
    try:
        return datetime.fromisoformat(str(x).replace('Z', '+00:00'))
    except Exception:
        return None


@app.get("/leaderboard")
async def get_leaderboard(period: str = '24h', sort: str = 'net_pnl', limit: int = 12, min_trades: int = 5, metric: str = None):
    """Лидерборд по persisted сделкам (bot_trades), пригодный для отбора стратегий.

    Query:
      period: 24h|7d|30d|all
      sort: net_pnl|profit_factor|expectancy|max_drawdown|win_rate|trades
      min_trades: отсечка от шума

    metric: legacy alias for sort.
    """
    sort = (metric or sort or 'net_pnl').strip().lower()
    limit = max(1, min(int(limit or 12), 100))
    min_trades = max(0, min(int(min_trades or 0), 1000))

    cutoff_iso = _parse_period_to_cutoff_iso(period)

    conn = None
    rows = []
    try:
        if DB_BACKEND == 'postgres':
            conn = psycopg2.connect(**PG_CONFIG)
            cur = conn.cursor()
            if cutoff_iso:
                cur.execute('''
                    SELECT bot_id, bot_name, symbol, strategy, mode,
                           entry_time, exit_time, pnl
                    FROM bot_trades
                    WHERE COALESCE(exit_time, entry_time) >= %s
                    ORDER BY COALESCE(exit_time, entry_time) ASC
                ''', (cutoff_iso,))
            else:
                cur.execute('''
                    SELECT bot_id, bot_name, symbol, strategy, mode,
                           entry_time, exit_time, pnl
                    FROM bot_trades
                    ORDER BY COALESCE(exit_time, entry_time) ASC
                ''')
            rows = cur.fetchall()
        else:
            if not os.path.exists(ML_DB_PATH):
                return {"success": True, "leaderboard": [], "meta": {"period": period, "cutoff": cutoff_iso}}
            conn = sqlite3.connect(ML_DB_PATH, timeout=3)
            cur = conn.cursor()
            if cutoff_iso:
                cur.execute('''
                    SELECT bot_id, bot_name, symbol, strategy, mode,
                           entry_time, exit_time, pnl
                    FROM bot_trades
                    WHERE COALESCE(exit_time, entry_time) >= ?
                    ORDER BY COALESCE(exit_time, entry_time) ASC
                ''', (cutoff_iso,))
            else:
                cur.execute('''
                    SELECT bot_id, bot_name, symbol, strategy, mode,
                           entry_time, exit_time, pnl
                    FROM bot_trades
                    ORDER BY COALESCE(exit_time, entry_time) ASC
                ''')
            rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # Aggregate
    by_bot: Dict[int, Dict] = {}
    for bot_id, bot_name, symbol, strategy, mode, entry_time, exit_time, pnl in rows:
        try:
            bot_id = int(bot_id or 0)
        except Exception:
            bot_id = 0
        if bot_id not in by_bot:
            by_bot[bot_id] = {
                'bot_id': bot_id,
                'name': bot_name or f'Bot {bot_id}',
                'symbol': (symbol or '').upper(),
                'strategy': strategy or 'Unknown',
                'mode': mode or 'Unknown',
                'pnls': [],
                'times': [],
            }
        p = float(pnl or 0)
        by_bot[bot_id]['pnls'].append(p)
        t = _safe_dt(exit_time) or _safe_dt(entry_time)
        by_bot[bot_id]['times'].append(t)

    out = []
    for bot_id, b in by_bot.items():
        pnls = b['pnls']
        trades = len(pnls)
        if trades < min_trades:
            continue

        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        win_rate = (len(wins) / trades * 100.0) if trades else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0  # negative
        expectancy = (win_rate/100.0) * avg_win + (1 - win_rate/100.0) * avg_loss

        # Max drawdown on cumulative pnl
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

        net_pnl = sum(pnls)

        out.append({
            'bot_id': bot_id,
            'name': b['name'],
            'symbol': b['symbol'],
            'strategy': b['strategy'],
            'mode': b['mode'],
            'net_pnl': net_pnl,
            'trades': trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
        })

    # sort
    key_map = {
        'net_pnl': 'net_pnl',
        'profit': 'net_pnl',
        'profit_factor': 'profit_factor',
        'pf': 'profit_factor',
        'expectancy': 'expectancy',
        'dd': 'max_drawdown',
        'max_drawdown': 'max_drawdown',
        'win_rate': 'win_rate',
        'winrate': 'win_rate',
        'trades': 'trades',
    }
    key = key_map.get(sort, 'net_pnl')
    # For drawdown: smaller is better
    reverse = False if key == 'max_drawdown' else True
    out.sort(key=lambda x: x.get(key, 0), reverse=reverse)

    out = out[:limit]
    for i, row in enumerate(out, 1):
        row['rank'] = i

    return {
        'success': True,
        'leaderboard': out,
        'meta': {
            'period': period,
            'cutoff': cutoff_iso,
            'sort': sort,
            'min_trades': min_trades,
        }
    }


# ==== ML DATA COLLECTION ====
from ml_data_collector import ml_collector, run_collection_job
import asyncio


def auto_cull_loop():
    """Profit-first auto-culling for running bots.

    Stops bots that are persistently losing, to reduce portfolio drag.
    """
    while True:
        try:
            if os.getenv('OPENCLAW_AUTO_CULL', '1') != '1':
                time.sleep(60)
                continue

            min_trades = int(os.getenv('OPENCLAW_AUTO_CULL_MIN_TRADES', '12'))
            max_loss = float(os.getenv('OPENCLAW_AUTO_CULL_MAX_LOSS_USDT', '-1.0'))  # negative
            min_wr = float(os.getenv('OPENCLAW_AUTO_CULL_MIN_WR', '18'))

            # hard kill-switch knobs
            kill_session_loss = float(os.getenv('OPENCLAW_KILL_SESSION_LOSS_USDT', '-0.8'))
            kill_loss_streak = int(os.getenv('OPENCLAW_KILL_LOSS_STREAK', '3'))

            # optional hard delete for persistent losers
            auto_delete = os.getenv('OPENCLAW_AUTO_DELETE_LOSERS', '1') == '1'
            delete_loss = float(os.getenv('OPENCLAW_AUTO_DELETE_MAX_LOSS_USDT', '-2.0'))

            persisted_map = get_persisted_trade_stats_cached()

            for bot_id, bot in list(bots.items()):
                s = merge_runtime_and_persisted_stats(
                    getattr(bot, 'name', ''),
                    getattr(bot, 'stats', {}) or {},
                    persisted_map
                )
                trades = int(s.get('total_trades', 0) or 0)
                pnl = float(s.get('total_profit_usdt', 0) or 0)
                wr = float(s.get('win_rate', 0) or 0)

                if not getattr(bot, 'is_running', False):
                    if auto_delete and trades >= min_trades and pnl <= delete_loss and not getattr(bot, 'current_position', None):
                        bots.pop(bot_id, None)
                        bot_threads.pop(bot_id, None)
                    continue

                # hard kill: session drawdown
                if trades >= 3 and pnl <= kill_session_loss:
                    bot.stop()
                    continue

                # hard kill: consecutive losses
                streak = 0
                try:
                    hist = list(getattr(bot, 'trades_history', []) or [])
                    for t in reversed(hist):
                        p = float(t.get('profit_usdt', t.get('pnl', 0)) or 0)
                        if p <= 0:
                            streak += 1
                        else:
                            break
                except Exception:
                    streak = 0

                if streak >= kill_loss_streak:
                    bot.stop()
                    continue

                # soft cull
                if trades >= min_trades and pnl <= max_loss:
                    bot.stop()
                elif trades >= max(20, min_trades) and pnl < 0 and wr < min_wr:
                    bot.stop()

        except Exception:
            pass

        time.sleep(180)


def auto_evolve_loop():
    """Периодически клонирует лучших ботов (эволюционный поиск)."""
    while True:
        try:
            if os.getenv('OPENCLAW_AUTO_EVOLVE', '1') != '1':
                time.sleep(120)
                continue

            clones_per_winner = int(os.getenv('OPENCLAW_EVOLVE_CLONES_PER_WINNER', '2'))
            top_n = int(os.getenv('OPENCLAW_EVOLVE_TOP_N', '5'))
            min_trades = int(os.getenv('OPENCLAW_EVOLVE_MIN_TRADES', '4'))
            max_new = int(os.getenv('OPENCLAW_EVOLVE_MAX_NEW', '12'))

            # endpoint-функция async, но без await внутри — запускаем цикл через asyncio.run
            try:
                asyncio.run(evolve_bots(
                    clones_per_winner=clones_per_winner,
                    top_n=top_n,
                    min_trades=min_trades,
                    max_new=max_new
                ))
            except Exception:
                pass

        except Exception:
            pass

        time.sleep(max(60, int(os.getenv('OPENCLAW_EVOLVE_INTERVAL_SEC', '300'))))


@app.get("/ml/stats")
async def get_ml_stats():
    """Статистика ML данных"""
    return {"success": True, "stats": ml_collector.get_stats()}

@app.post("/ml/collect")
async def trigger_ml_collection():
    """Запустить сбор данных вручную"""
    try:
        count = await ml_collector.collect_all_symbols()
        return {"success": True, "collected": count, "stats": ml_collector.get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/training-data/{symbol}")
async def get_training_data(symbol: str, limit: int = 1000):
    """Получить данные для обучения"""
    data = ml_collector.get_training_data(symbol.upper() + 'USDT', limit)
    return {"success": True, "count": len(data), "data": data[:100]}  # Первые 100 для примера

# Запуск ML collection в background (опционально)
@app.on_event("startup")
async def startup_event():
    """Запуск фоновых задач при старте"""
    global next_bot_id

    # Автовосстановление ботов после рестарта API
    if os.getenv("OPENCLAW_AUTO_BOOT_BOTS", "1") == "1" and not bots:
        keep_ids_env = (os.getenv("OPENCLAW_BOOT_KEEP_IDS", "") or "").strip()
        keep_ids = None
        if keep_ids_env:
            try:
                keep_ids = {int(x.strip()) for x in keep_ids_env.split(',') if x.strip()}
            except Exception:
                keep_ids = None

        for config in BOT_CONFIGS:
            bot_id = next_bot_id
            next_bot_id += 1

            if keep_ids is not None and bot_id not in keep_ids:
                continue

            bot = TradingBotV2(
                bot_id=bot_id,
                name=config['name'],
                symbol=config['symbol'],
                strategy=config['strategy'],
                mode_name=config['mode'],
                initial_balance=config['balance'],
                leverage=int(config.get('leverage', os.getenv('DEFAULT_LEVERAGE', '2'))),
                interval=config.get('interval', '5m')
            )
            bots[bot_id] = bot

        for bot_id, bot in bots.items():
            if not bot.is_running:
                bot.start()

                def bot_loop(b=bot):
                    while b.is_running:
                        b.tick()
                        time.sleep(20)

                thread = threading.Thread(target=bot_loop, daemon=True)
                thread.start()
                bot_threads[bot_id] = thread

    # По умолчанию выключено, чтобы не блокировать API/панель.
    # Включать только через env: OPENCLAW_ML_BG_COLLECTION=1
    if os.getenv("OPENCLAW_ML_BG_COLLECTION", "0") == "1":
        asyncio.create_task(run_collection_job())

    # Profit-first auto culling (stop/delete persistent losers)
    if os.getenv('OPENCLAW_AUTO_CULL', '1') == '1':
        t = threading.Thread(target=auto_cull_loop, daemon=True)
        t.start()

    # Evolution loop: clone winners with mutations
    if os.getenv('OPENCLAW_AUTO_EVOLVE', '1') == '1':
        t2 = threading.Thread(target=auto_evolve_loop, daemon=True)
        t2.start()


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Запуск Trading Platform API v3...")
    print("📊 Документация: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("🧠 ML Data Collection: enabled\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
