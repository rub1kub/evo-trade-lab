# 🧬 evo-trade-lab

**Evolutionary crypto trading bot platform with autonomous strategy discovery.**

A pet project exploring automated trading strategy optimization through evolutionary algorithms. The system spawns hundreds of demo trading bots, each running different strategy configurations, then automatically culls losers and clones/mutates winners — mimicking natural selection to find profitable setups.

## What it does

- **Mass bot spawning** — launches 100+ bots across multiple strategies (RSI, MACD, Bollinger, EMA, Grid, DCA, and custom combos)
- **Evolutionary loop** — top performers are automatically cloned with mutated parameters every N minutes
- **Auto-culling** — persistent losers get stopped and deleted based on configurable thresholds (PnL, win rate, loss streaks)
- **Real market data** — connects to MEXC exchange API for live price feeds and candle data
- **Demo mode** — all trading is simulated with virtual balances (no real money at risk)
- **Web dashboard** — real-time monitoring with custom charts (lightweight-charts), leaderboard, backtest panel, and analytics
- **Event-sourced markers** — entry/exit/TP/SL events stored and visualized on charts
- **Multiple strategy families**: RSI, MACD, Bollinger Bands, EMA Crossover, RSI+MACD combo, Volume Breakout, Grid, DCA, Crowd Psychology, News Flow Proxy, Adaptive Hybrid Alpha

## Architecture

```
frontend/          — Single-page dashboard (vanilla JS + lightweight-charts)
backend/
  api.py           — FastAPI server (REST + WebSocket)
  bot_v2.py        — Trading bot engine with risk management
  bot_manager.py   — Multi-bot account management
  trading_modes.py — Conservative / Balanced / Aggressive / Degen / Scalp
  strategies/      — Strategy implementations
  auto_create_bots_v3.py — Bot configuration presets (96+ configs)
  risk_management.py     — Trailing stop, breakeven, portfolio limits
  risk_manager_v2.py     — Kelly criterion, daily limits, correlation
  backtesting.py         — Historical backtesting + Monte Carlo
  analytics.py           — Equity curves, heatmaps, performance metrics
  ml_data_collector.py   — Market data collection for future ML
database/
  schema.sql       — PostgreSQL/SQLite schema
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rub1kub/evo-trade-lab.git
cd evo-trade-lab

# 2. Setup
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pandas ta python-dotenv psycopg2-binary requests websockets pydantic

# 3. Configure
cp .env.example .env
# Edit .env with your MEXC API keys (read-only keys are enough for demo mode)

# 4. Run
cd backend
python api.py

# 5. Open dashboard
# http://localhost:8000 or configure your reverse proxy
```

## Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /research/bots` | All bots with stats, strategy leaders |
| `GET /leaderboard` | Ranked bot performance (24h/7d/30d/all) |
| `POST /bots/evolve` | Clone top bots with mutated parameters |
| `POST /bots/auto-create` | Spawn all preset bot configs |
| `POST /bots/{id}/tune` | Live-tune mode, leverage, risk params |
| `GET /events` | Trade events for chart markers |
| `POST /backtest/run` | Run historical backtest |
| `GET /analytics/portfolio` | Portfolio-level analytics |
| `WS /ws` | Real-time updates |

## Evolution Parameters

The evolutionary loop is controlled via environment variables:

- `OPENCLAW_AUTO_EVOLVE=1` — enable/disable
- `OPENCLAW_EVOLVE_INTERVAL_SEC=300` — clone cycle interval
- `OPENCLAW_EVOLVE_CLONES_PER_WINNER=2` — clones per top bot
- `OPENCLAW_EVOLVE_TOP_N=5` — how many winners to clone
- `OPENCLAW_EVOLVE_MAX_NEW=10` — max new bots per cycle

Culling thresholds:

- `OPENCLAW_AUTO_CULL_MIN_TRADES=6` — min trades before evaluation
- `OPENCLAW_AUTO_CULL_MAX_LOSS_USDT=-1.50` — PnL threshold for soft cull
- `OPENCLAW_KILL_SESSION_LOSS_USDT=-3.00` — hard stop on deep drawdown
- `OPENCLAW_KILL_LOSS_STREAK=5` — stop after N consecutive losses

## Why I built this

I wanted to explore whether evolutionary parameter optimization could find profitable trading configurations faster than manual tuning. Instead of spending weeks backtesting individual setups, the system runs hundreds of variants simultaneously on live market data and lets natural selection do the work.

It's a learning project — not financial advice, not production-ready for real money. But the evolutionary approach to strategy discovery is genuinely interesting.

## Tech Stack

- **Backend**: Python, FastAPI, SQLite/PostgreSQL
- **Frontend**: Vanilla JS, [Lightweight Charts](https://github.com/nicholasgasior/lightweight-charts) by TradingView
- **Exchange**: MEXC (via REST API)
- **Strategies**: Technical analysis via [ta](https://github.com/bukosabino/ta) library

## License

MIT
