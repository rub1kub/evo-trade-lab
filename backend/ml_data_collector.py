"""
ML Data Collector - сбор данных для машинного обучения
Собирает: свечи, индикаторы, сигналы, результаты сделок
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import asyncio
import aiohttp
import logging
import psycopg2


logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_data.db')
DB_BACKEND = os.getenv('DB_BACKEND', 'sqlite').lower()

PG_CONFIG = {
    'host': os.getenv('PGHOST', '127.0.0.1'),
    'port': int(os.getenv('PGPORT', '5432')),
    'dbname': os.getenv('PGDATABASE', 'trading_ml'),
    'user': os.getenv('PGUSER', 'trading'),
    'password': os.getenv('PGPASSWORD', ''),
}


def init_db():
    """Инициализация базы данных для ML (SQLite/PostgreSQL)"""
    if DB_BACKEND == 'postgres':
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                rsi_7 DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION,
                rsi_21 DOUBLE PRECISION,
                macd DOUBLE PRECISION,
                macd_signal DOUBLE PRECISION,
                macd_hist DOUBLE PRECISION,
                bb_upper DOUBLE PRECISION,
                bb_middle DOUBLE PRECISION,
                bb_lower DOUBLE PRECISION,
                ema_9 DOUBLE PRECISION,
                ema_21 DOUBLE PRECISION,
                ema_50 DOUBLE PRECISION,
                sma_20 DOUBLE PRECISION,
                atr_14 DOUBLE PRECISION,
                volume_sma_20 DOUBLE PRECISION,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                strategy TEXT NOT NULL,
                signal_type TEXT,
                confidence DOUBLE PRECISION,
                price_at_signal DOUBLE PRECISION,
                rsi_value DOUBLE PRECISION,
                macd_value DOUBLE PRECISION,
                bb_position DOUBLE PRECISION,
                volume_ratio DOUBLE PRECISION,
                outcome TEXT,
                pnl_1h DOUBLE PRECISION,
                pnl_4h DOUBLE PRECISION,
                pnl_24h DOUBLE PRECISION,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, strategy)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_trades (
                id BIGSERIAL PRIMARY KEY,
                bot_id INTEGER,
                bot_name TEXT,
                symbol TEXT,
                strategy TEXT,
                mode TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION,
                quantity DOUBLE PRECISION,
                side TEXT,
                pnl DOUBLE PRECISION,
                pnl_percent DOUBLE PRECISION,
                exit_reason TEXT,
                rsi_at_entry DOUBLE PRECISION,
                macd_at_entry DOUBLE PRECISION,
                volume_at_entry DOUBLE PRECISION,
                btc_price_at_entry DOUBLE PRECISION,
                btc_change_24h DOUBLE PRECISION,
                market_sentiment TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Миграции колонок для расширенной futures-истории
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS market_type TEXT")
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS leverage INTEGER")
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS margin DOUBLE PRECISION")
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS notional DOUBLE PRECISION")
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS open_fee DOUBLE PRECISION")
        cursor.execute("ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS close_fee DOUBLE PRECISION")

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_bot ON bot_trades(bot_id, entry_time)')

        # Trade events (for accurate chart markers)
        cursor.execute('''
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_bot_symbol_ts ON trade_events(bot_id, symbol, ts_exchange_ms)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_type_ts ON trade_events(event_type, ts_exchange_ms)')

        conn.commit()
        conn.close()
        logger.info(f"ML Database initialized in PostgreSQL {PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица свечей с индикаторами
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            -- Индикаторы
            rsi_7 REAL,
            rsi_14 REAL,
            rsi_21 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            ema_9 REAL,
            ema_21 REAL,
            ema_50 REAL,
            sma_20 REAL,
            atr_14 REAL,
            volume_sma_20 REAL,
            -- Метаданные
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
    ''')

    # Таблица сигналов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT,  -- BUY, SELL, HOLD
            confidence REAL,
            price_at_signal REAL,
            -- Контекст
            rsi_value REAL,
            macd_value REAL,
            bb_position REAL,  -- % позиции между BB
            volume_ratio REAL,
            -- Результат (заполняется позже)
            outcome TEXT,  -- WIN, LOSS, PENDING
            pnl_1h REAL,  -- PnL через 1 час
            pnl_4h REAL,  -- PnL через 4 часа
            pnl_24h REAL,  -- PnL через 24 часа
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp, strategy)
        )
    ''')

    # Таблица сделок ботов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER,
            bot_name TEXT,
            symbol TEXT,
            strategy TEXT,
            mode TEXT,
            -- Сделка
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            side TEXT,
            -- Результат
            pnl REAL,
            pnl_percent REAL,
            exit_reason TEXT,
            -- Контекст на момент входа
            rsi_at_entry REAL,
            macd_at_entry REAL,
            volume_at_entry REAL,
            -- Рыночные условия
            btc_price_at_entry REAL,
            btc_change_24h REAL,
            market_sentiment TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Индексы
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_bot ON bot_trades(bot_id, entry_time)')

    conn.commit()
    conn.close()

    logger.info(f"ML Database initialized at {DB_PATH}")


class MLDataCollector:
    """Коллектор данных для ML"""
    
    SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'ATOMUSDT',
        'LTCUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT',
        # Новые монеты
        'NEARUSDT', 'FTMUSDT', 'INJUSDT', 'SUIUSDT', 'APTUSDT',
        'ARBUSDT', 'OPUSDT', 'MATICUSDT', 'FILUSDT', 'ICPUSDT'
    ]
    
    def __init__(self):
        init_db()

    def _get_conn(self):
        if DB_BACKEND == 'postgres':
            return psycopg2.connect(**PG_CONFIG)
        return sqlite3.connect(DB_PATH, timeout=30)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Расчёт RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9):
        """Расчёт MACD (корректный)"""
        if len(prices) < slow + signal_period:
            return 0.0, 0.0, 0.0

        arr = np.array(prices, dtype=float)
        ema_fast = self._ema_series(arr, fast)
        ema_slow = self._ema_series(arr, slow)

        macd_series = ema_fast - ema_slow
        signal_series = self._ema_series(macd_series, signal_period)

        macd_value = float(macd_series[-1])
        signal_value = float(signal_series[-1])
        hist_value = float(macd_value - signal_value)

        return macd_value, signal_value, hist_value

    def _ema_series(self, data: np.ndarray, period: int) -> np.ndarray:
        """EMA series"""
        if len(data) == 0:
            return np.array([])

        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Exponential Moving Average (последнее значение)"""
        series = self._ema_series(data, period)
        return float(series[-1]) if len(series) else 0.0
    
    def calculate_bollinger(self, prices: List[float], period: int = 20, std_dev: float = 2.0):
        """Расчёт Bollinger Bands"""
        if len(prices) < period:
            return 0, 0, 0
        
        prices = np.array(prices[-period:])
        middle = np.mean(prices)
        std = np.std(prices)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(highs) < period + 1:
            return 0
        
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        return np.mean(trs[-period:])
    
    async def fetch_klines(self, symbol: str, interval: str = '60m', limit: int = 100) -> List:
        """Получение свечей с MEXC"""
        url = f"https://api.mexc.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return []
    
    async def collect_candles_with_indicators(self, symbol: str):
        """Сбор свечей с индикаторами"""
        klines = await self.fetch_klines(symbol, '60m', 100)
        
        if not klines:
            return 0
        
        conn = self._get_conn()
        cursor = conn.cursor()
        inserted = 0
        
        # Подготовка данных
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        for i, k in enumerate(klines):
            if i < 26:  # Нужно минимум 26 свечей для MACD
                continue
            
            timestamp = int(k[0])
            
            # Срез данных до текущей свечи
            closes_slice = closes[:i+1]
            highs_slice = highs[:i+1]
            lows_slice = lows[:i+1]
            volumes_slice = volumes[:i+1]
            
            # Расчёт индикаторов
            rsi_7 = self.calculate_rsi(closes_slice, 7)
            rsi_14 = self.calculate_rsi(closes_slice, 14)
            rsi_21 = self.calculate_rsi(closes_slice, 21)
            
            macd, macd_signal, macd_hist = self.calculate_macd(closes_slice)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger(closes_slice)
            
            atr = self.calculate_atr(highs_slice, lows_slice, closes_slice)
            
            ema_9 = self._ema(np.array(closes_slice), 9)
            ema_21 = self._ema(np.array(closes_slice), 21)
            ema_50 = self._ema(np.array(closes_slice), 50) if len(closes_slice) >= 50 else 0
            
            sma_20 = np.mean(closes_slice[-20:]) if len(closes_slice) >= 20 else 0
            volume_sma = np.mean(volumes_slice[-20:]) if len(volumes_slice) >= 20 else 0
            
            vals = (
                symbol, timestamp,
                float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                rsi_7, rsi_14, rsi_21, macd, macd_signal, macd_hist,
                bb_upper, bb_middle, bb_lower, ema_9, ema_21, ema_50,
                sma_20, atr, volume_sma
            )

            if DB_BACKEND == 'postgres':
                cursor.execute('''
                    INSERT INTO candles
                    (symbol, timestamp, open, high, low, close, volume,
                     rsi_7, rsi_14, rsi_21, macd, macd_signal, macd_hist,
                     bb_upper, bb_middle, bb_lower, ema_9, ema_21, ema_50,
                     sma_20, atr_14, volume_sma_20)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        rsi_7 = EXCLUDED.rsi_7,
                        rsi_14 = EXCLUDED.rsi_14,
                        rsi_21 = EXCLUDED.rsi_21,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_hist = EXCLUDED.macd_hist,
                        bb_upper = EXCLUDED.bb_upper,
                        bb_middle = EXCLUDED.bb_middle,
                        bb_lower = EXCLUDED.bb_lower,
                        ema_9 = EXCLUDED.ema_9,
                        ema_21 = EXCLUDED.ema_21,
                        ema_50 = EXCLUDED.ema_50,
                        sma_20 = EXCLUDED.sma_20,
                        atr_14 = EXCLUDED.atr_14,
                        volume_sma_20 = EXCLUDED.volume_sma_20
                ''', vals)
            else:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO candles
                        (symbol, timestamp, open, high, low, close, volume,
                         rsi_7, rsi_14, rsi_21, macd, macd_signal, macd_hist,
                         bb_upper, bb_middle, bb_lower, ema_9, ema_21, ema_50,
                         sma_20, atr_14, volume_sma_20)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', vals)
                except sqlite3.IntegrityError:
                    pass  # Дубликат

            inserted += 1
        
        conn.commit()
        conn.close()
        return inserted
    
    def record_signal(self, symbol: str, strategy: str, signal_type: str,
                     confidence: float, price: float, context: Dict):
        """Запись сигнала"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        vals = (
            symbol, int(datetime.utcnow().timestamp() * 1000),
            strategy, signal_type, confidence, price,
            context.get('rsi'), context.get('macd'),
            context.get('bb_position'), context.get('volume_ratio')
        )

        if DB_BACKEND == 'postgres':
            cursor.execute('''
                INSERT INTO signals
                (symbol, timestamp, strategy, signal_type, confidence, price_at_signal,
                 rsi_value, macd_value, bb_position, volume_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp, strategy) DO UPDATE SET
                    signal_type = EXCLUDED.signal_type,
                    confidence = EXCLUDED.confidence,
                    price_at_signal = EXCLUDED.price_at_signal,
                    rsi_value = EXCLUDED.rsi_value,
                    macd_value = EXCLUDED.macd_value,
                    bb_position = EXCLUDED.bb_position,
                    volume_ratio = EXCLUDED.volume_ratio
            ''', vals)
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO signals
                (symbol, timestamp, strategy, signal_type, confidence, price_at_signal,
                 rsi_value, macd_value, bb_position, volume_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', vals)
        
        conn.commit()
        conn.close()
    
    def record_trade(self, bot_id: int, bot_name: str, trade: Dict, context: Dict):
        """Запись сделки бота + базовые trade_events для корректных меток."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # prefer numeric timestamps when provided
        entry_ts_ms = trade.get('entry_ts_ms')
        exit_ts_ms = trade.get('exit_ts_ms')
        ts_server_ms = int(datetime.utcnow().timestamp() * 1000)

        vals = (
            bot_id, bot_name,
            trade.get('symbol'), trade.get('strategy'), trade.get('mode'),
            trade.get('entry_time'), trade.get('exit_time'),
            trade.get('entry_price'), trade.get('exit_price'),
            trade.get('quantity'), trade.get('side', 'LONG'),
            trade.get('pnl'), trade.get('pnl_percent'), trade.get('exit_reason'),
            context.get('rsi'), context.get('macd'), context.get('volume'),
            context.get('btc_price'), context.get('btc_change_24h'),
            context.get('market_sentiment'),
            trade.get('market_type'), trade.get('leverage'), trade.get('margin'),
            trade.get('notional'), trade.get('open_fee'), trade.get('close_fee')
        )

        if DB_BACKEND == 'postgres':
            cursor.execute('''
                INSERT INTO bot_trades
                (bot_id, bot_name, symbol, strategy, mode,
                 entry_time, exit_time, entry_price, exit_price, quantity, side,
                 pnl, pnl_percent, exit_reason,
                 rsi_at_entry, macd_at_entry, volume_at_entry,
                 btc_price_at_entry, btc_change_24h, market_sentiment,
                 market_type, leverage, margin, notional, open_fee, close_fee)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', vals)
        else:
            cursor.execute('''
                INSERT INTO bot_trades
                (bot_id, bot_name, symbol, strategy, mode,
                 entry_time, exit_time, entry_price, exit_price, quantity, side,
                 pnl, pnl_percent, exit_reason,
                 rsi_at_entry, macd_at_entry, volume_at_entry,
                 btc_price_at_entry, btc_change_24h, market_sentiment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', vals[:20])

        # Also write minimal events for chart markers (works for demo now; later will be real fills)
        symbol = (trade.get('symbol') or '').upper()
        side = trade.get('side', 'LONG')
        entry_price = trade.get('entry_price')
        exit_price = trade.get('exit_price')
        exit_reason = trade.get('exit_reason') or trade.get('reason')

        def _ins_event(event_type: str, ts_ex_ms, price):
            if not symbol or ts_ex_ms is None or price is None:
                return
            if DB_BACKEND == 'postgres':
                cursor.execute('''
                    INSERT INTO trade_events
                    (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, pnl, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (bot_id, bot_name, symbol, event_type, side, int(ts_ex_ms), ts_server_ms, float(price), float(trade.get('quantity') or 0), float(trade.get('pnl') or 0), exit_reason))
            else:
                # sqlite
                cursor.execute('''
                    INSERT INTO trade_events
                    (bot_id, bot_name, symbol, event_type, side, ts_exchange_ms, ts_server_ms, price, qty, pnl, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (bot_id, bot_name, symbol, event_type, side, int(ts_ex_ms), ts_server_ms, float(price), float(trade.get('quantity') or 0), float(trade.get('pnl') or 0), exit_reason))

        # For now we use provided entry/exit ts_ms if present; otherwise fallback to server time.
        # Only emit events when data is present (supports "entry-only" records)
        if entry_price is not None:
            _ins_event('FILL_ENTRY', entry_ts_ms or ts_server_ms, entry_price)
        if exit_price is not None:
            _ins_event('FILL_EXIT', exit_ts_ms or ts_server_ms, exit_price)

        conn.commit()
        conn.close()
    
    async def collect_all_symbols(self):
        """Сбор данных по всем символам"""
        total = 0
        for symbol in self.SYMBOLS:
            try:
                count = await self.collect_candles_with_indicators(symbol)
                total += count
                logger.info(f"Collected {count} candles for {symbol}")
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
        
        return total
    
    def get_training_data(self, symbol: str = None, limit: int = 10000) -> List[Dict]:
        """Получение данных для обучения"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if symbol:
            if DB_BACKEND == 'postgres':
                cursor.execute('''
                    SELECT * FROM candles WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
                ''', (symbol, limit))
        else:
            if DB_BACKEND == 'postgres':
                cursor.execute('''
                    SELECT * FROM candles ORDER BY timestamp DESC LIMIT %s
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT * FROM candles ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_stats(self) -> Dict:
        """Статистика собранных данных"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM candles')
        candles_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT symbol) FROM candles')
        symbols_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM signals')
        signals_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM bot_trades')
        trades_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            'candles': candles_count,
            'symbols': symbols_count,
            'signals': signals_count,
            'trades': trades_count
        }
    
    def close(self):
        return


# Глобальный экземпляр
ml_collector = MLDataCollector()


async def run_collection_job():
    """Периодический сбор данных"""
    collector = MLDataCollector()
    while True:
        try:
            count = await collector.collect_all_symbols()
            logger.info(f"ML Data collection completed: {count} records")
        except Exception as e:
            logger.error(f"ML collection error: {e}")
        
        await asyncio.sleep(300)  # Каждые 5 минут


if __name__ == "__main__":
    # Тест
    async def test():
        collector = MLDataCollector()
        count = await collector.collect_all_symbols()
        print(f"Collected: {count}")
        print(f"Stats: {collector.get_stats()}")
    
    asyncio.run(test())
