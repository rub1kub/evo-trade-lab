# 🧬 evo-trade-lab

> Evolutionary crypto trading bot platform with autonomous strategy discovery.

A research platform that spawns hundreds of trading bots, each with different strategy configurations, then evolves the best performers — survival of the fittest, applied to crypto trading.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Concept

Instead of manually optimizing trading strategies, evo-trade-lab:

1. **Spawns** hundreds of bots with random strategy configs (RSI/MACD/BB parameters, stop-loss levels, position sizing)
2. **Runs** them all on demo accounts simultaneously
3. **Evaluates** performance by Sharpe ratio, drawdown, win rate
4. **Breeds** top performers — combines their parameters
5. **Repeats** — each generation gets smarter

```
Generation 1:  500 bots → top 50 survive
Generation 2:  breed + mutate → 500 new bots
Generation 3:  ...
Generation N:  stable high-performing strategy
```

## Features

- 🤖 **Autonomous strategy discovery** — no manual parameter tuning
- 📊 **Multi-metric evaluation** — Sharpe ratio, max drawdown, profit factor
- 🔀 **Genetic operators** — crossover, mutation, selection
- 📈 **Real-time dashboard** — track all running bots and generations
- 💾 **Strategy persistence** — save and replay winning configurations
- 🔌 **Exchange integration** — via CCXT (Binance, Bybit, OKX)

## Architecture

```
Orchestrator
├── Strategy Generator (random params)
├── Bot Pool (concurrent execution)
├── Fitness Evaluator (metrics)
├── Genetic Engine (breed/mutate)
└── Dashboard (real-time monitoring)
```

## Status

🧪 **Active research project** — not for live trading. Demo/paper trading only.

Results from early experiments:
- Best strategies found in 10-20 generations (~2-4 hours)
- Consistent outperformance of random baseline on 30-day backtests
- Most fragile parameter: stop-loss placement

## Tech Stack

- Python + asyncio
- CCXT (exchange abstraction)
- SQLite (strategy/result storage)
- Streamlit (dashboard)

---

Built by [@rub1kub](https://github.com/rub1kub) · Part of ongoing crypto automation research
