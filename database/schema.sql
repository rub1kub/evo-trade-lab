-- Trading Platform Database Schema

-- Боты (стратегии)
CREATE TABLE IF NOT EXISTS bots (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'stopped',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ордера
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER REFERENCES bots(id),
    order_id VARCHAR(100) UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8),
    price DECIMAL(20, 8),
    status VARCHAR(20),
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Сделки (исполненные)
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER REFERENCES bots(id),
    order_id INTEGER REFERENCES orders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8),
    price DECIMAL(20, 8),
    cost DECIMAL(20, 8),
    profit DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Логи ботов
CREATE TABLE IF NOT EXISTS bot_logs (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER REFERENCES bots(id),
    level VARCHAR(20) NOT NULL,
    message TEXT,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trade events (источник истины для меток/агрегаций)
CREATE TABLE IF NOT EXISTS trade_events (
    id BIGSERIAL PRIMARY KEY,
    bot_id INTEGER,
    bot_name TEXT,
    symbol TEXT NOT NULL,
    market_type TEXT DEFAULT 'futures',
    event_type TEXT NOT NULL, -- FILL, ORDER_NEW, ORDER_FILLED, TP_SET, TP_HIT, SL_SET, SL_HIT, LIQUIDATION, FUNDING, FEE
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
);

-- Индексы для производительности
CREATE INDEX IF NOT EXISTS idx_orders_bot_id ON orders(bot_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_trades_bot_id ON trades(bot_id);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_bot_logs_bot_id ON bot_logs(bot_id);
CREATE INDEX IF NOT EXISTS idx_bot_logs_created_at ON bot_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_trade_events_bot_symbol_ts ON trade_events(bot_id, symbol, ts_exchange_ms);
CREATE INDEX IF NOT EXISTS idx_trade_events_type_ts ON trade_events(event_type, ts_exchange_ms);
