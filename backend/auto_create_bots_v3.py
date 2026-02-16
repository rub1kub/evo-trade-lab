#!/usr/bin/env python3
"""
Улучшенная конфигурация ботов v3
- Больше монет (24 вместо 14)
- Оптимизированные RSI пороги
- 1-минутный интервал анализа
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_strategy import BollingerStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.combined_rsi_macd import RSI_MACD_Strategy
from strategies.volume_breakout import VolumeBreakoutStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.dca_strategy import DCAStrategy
from strategies.crowd_psychology import CrowdPsychologyStrategy
from strategies.news_flow_proxy import NewsFlowProxyStrategy
from strategies.adaptive_hybrid_alpha import AdaptiveHybridAlphaStrategy


# Расширенный список монет
COINS = {
    # Major
    'BTC': {'name': 'Bitcoin', 'volatility': 'low'},
    'ETH': {'name': 'Ethereum', 'volatility': 'low'},
    'BNB': {'name': 'BNB', 'volatility': 'medium'},
    'SOL': {'name': 'Solana', 'volatility': 'high'},
    'XRP': {'name': 'XRP', 'volatility': 'medium'},
    'ADA': {'name': 'Cardano', 'volatility': 'medium'},
    
    # DeFi
    'LINK': {'name': 'Chainlink', 'volatility': 'medium'},
    'AVAX': {'name': 'Avalanche', 'volatility': 'high'},
    'DOT': {'name': 'Polkadot', 'volatility': 'medium'},
    'ATOM': {'name': 'Cosmos', 'volatility': 'medium'},
    'LTC': {'name': 'Litecoin', 'volatility': 'low'},
    
    # New L1/L2
    'NEAR': {'name': 'NEAR Protocol', 'volatility': 'high'},
    'FTM': {'name': 'Fantom', 'volatility': 'high'},
    'INJ': {'name': 'Injective', 'volatility': 'high'},
    'SUI': {'name': 'Sui', 'volatility': 'very_high'},
    'APT': {'name': 'Aptos', 'volatility': 'high'},
    'ARB': {'name': 'Arbitrum', 'volatility': 'high'},
    'OP': {'name': 'Optimism', 'volatility': 'high'},
    'FIL': {'name': 'Filecoin', 'volatility': 'medium'},
    'ICP': {'name': 'Internet Computer', 'volatility': 'high'},
    
    # Meme (высокая волатильность)
    'DOGE': {'name': 'Dogecoin', 'volatility': 'high'},
    'SHIB': {'name': 'Shiba Inu', 'volatility': 'very_high'},
    'PEPE': {'name': 'Pepe', 'volatility': 'very_high'},
    'FLOKI': {'name': 'Floki', 'volatility': 'very_high'},
}


# Оптимизированные RSI пороги по волатильности
RSI_PARAMS = {
    'low': {'oversold': 25, 'overbought': 75, 'period': 14},      # BTC, ETH, LTC
    'medium': {'oversold': 30, 'overbought': 70, 'period': 14},   # BNB, XRP, ADA
    'high': {'oversold': 35, 'overbought': 65, 'period': 10},     # SOL, NEAR, ARB
    'very_high': {'oversold': 40, 'overbought': 60, 'period': 7}, # Meme coins
}


def get_optimal_rsi(coin: str) -> dict:
    """Получить оптимальные RSI параметры для монеты"""
    if coin not in COINS:
        return RSI_PARAMS['medium']
    volatility = COINS[coin]['volatility']
    return RSI_PARAMS.get(volatility, RSI_PARAMS['medium'])


# === КОНФИГУРАЦИИ БОТОВ v3 ===
BOT_CONFIGS_V3 = []

# 1. RSI боты для основных монет (оптимизированные пороги)
rsi_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'NEAR', 'ARB', 'SUI']
core_overrides = {
    # Profit-first core: slightly bigger size/leverage on proven winners
    'BNB': {'mode': 'balanced_plus', 'leverage': 7, 'balance': 120},
    'XRP': {'mode': 'balanced_plus', 'leverage': 7, 'balance': 130},
    'NEAR': {'mode': 'aggressive', 'leverage': 6, 'balance': 120},
}

for coin in rsi_coins:
    params = get_optimal_rsi(coin)
    volatility = COINS.get(coin, {}).get('volatility', 'medium')
    mode = 'degen' if volatility == 'very_high' else 'aggressive' if volatility == 'high' else 'balanced'

    ov = core_overrides.get(coin, {})

    cfg = {
        'name': f'RSI-OPT {coin}',
        'symbol': f'{coin}USDT',
        'strategy': RSIStrategy(
            rsi_oversold=params['oversold'],
            rsi_overbought=params['overbought'],
            rsi_period=params['period']
        ),
        'mode': ov.get('mode', mode),
        'balance': ov.get('balance', 100),
        'interval': '1m',  # 1-минутный анализ!
    }

    if 'leverage' in ov:
        cfg['leverage'] = ov['leverage']

    BOT_CONFIGS_V3.append(cfg)

# 2. MACD боты (быстрые параметры для скальпинга)
macd_coins = ['BTC', 'ETH', 'SOL', 'OP', 'INJ', 'APT']
for coin in macd_coins:
    BOT_CONFIGS_V3.append({
        'name': f'MACD-FAST {coin}',
        'symbol': f'{coin}USDT',
        'strategy': MACDStrategy(fast_period=8, slow_period=17, signal_period=5),
        'mode': 'scalp' if coin in ['BTC','ETH','SOL'] else 'balanced',
        'balance': 100,
        'interval': '1m',
    })

# 3. Bollinger боты (для боковиков)
bb_coins = ['BTC', 'ETH', 'LTC', 'ATOM', 'DOT']
for coin in bb_coins:
    BOT_CONFIGS_V3.append({
        'name': f'BB-BOUNCE {coin}',
        'symbol': f'{coin}USDT',
        'strategy': BollingerStrategy(period=20, std_dev=2.0),
        'mode': 'conservative',
        'balance': 100,
        'interval': '5m',  # BB лучше на 5м
    })

# 4. EMA Crossover (тренд-следование)
ema_coins = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
for coin in ema_coins:
    BOT_CONFIGS_V3.append({
        'name': f'EMA-TREND {coin}',
        'symbol': f'{coin}USDT',
        'strategy': EMACrossoverStrategy(fast_period=9, slow_period=21),
        'mode': 'balanced',
        'balance': 100,
        'interval': '5m',
    })

# 5. Комбо RSI+MACD (двойное подтверждение)
combo_coins = ['BTC', 'ETH', 'SOL', 'ARB']
for coin in combo_coins:
    params = get_optimal_rsi(coin)
    BOT_CONFIGS_V3.append({
        'name': f'COMBO {coin}',
        'symbol': f'{coin}USDT',
        'strategy': RSI_MACD_Strategy(
            rsi_oversold=params['oversold'],
            rsi_overbought=params['overbought']
        ),
        'mode': 'conservative',  # Комбо = консервативный
        'balance': 100,
        'interval': '1m',
    })

# 6. Volume Breakout (прорыв на объёме)
vol_coins = ['BTC', 'ETH', 'SOL', 'PEPE', 'SHIB']
for coin in vol_coins:
    vol_mult = 1.5 if COINS.get(coin, {}).get('volatility') == 'very_high' else 2.5
    BOT_CONFIGS_V3.append({
        'name': f'VOL-BREAK {coin}',
        'symbol': f'{coin}USDT',
        'strategy': VolumeBreakoutStrategy(volume_multiplier=vol_mult, price_threshold_pct=0.8),
        'mode': 'aggressive',
        'balance': 100,
        'interval': '1m',
    })

# 7. Grid боты (для боковиков, увеличенный баланс)
grid_coins = ['BTC', 'ETH', 'SOL']
for coin in grid_coins:
    BOT_CONFIGS_V3.append({
        'name': f'GRID {coin}',
        'symbol': f'{coin}USDT',
        'strategy': GridTradingStrategy(grid_levels=15, grid_range_pct=5.0),
        'mode': 'balanced',
        'balance': 200,
        'interval': '1m',
    })

# 8. DCA боты (усреднение)
dca_coins = ['BTC', 'ETH', 'SOL']
for coin in dca_coins:
    params = get_optimal_rsi(coin)
    BOT_CONFIGS_V3.append({
        'name': f'DCA {coin}',
        'symbol': f'{coin}USDT',
        'strategy': DCAStrategy(
            initial_buy_threshold=params['oversold'],
            dca_levels=4,
            dca_step_pct=2.5
        ),
        'mode': 'conservative',
        'balance': 150,
        'interval': '1m',
    })

# 9. Meme Degen боты (агрессивные настройки)
meme_coins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI']
for coin in meme_coins:
    if coin == 'FLOKI':
        symbol = 'FLOKIUSDT'
    else:
        symbol = f'{coin}USDT'

    # DOGE часто хорошо подходит под скальп из-за ликвидности
    mode = 'scalp' if coin == 'DOGE' else 'degen'

    BOT_CONFIGS_V3.append({
        'name': f'MEME-DEGEN {coin} 💎',
        'symbol': symbol,
        'strategy': RSIStrategy(rsi_oversold=45, rsi_overbought=55, rsi_period=5),  # Очень агрессивно
        'mode': mode,
        'balance': 50,  # Меньше на мемы
        'interval': '1m',
    })

# 10. Новые L2 боты
l2_coins = ['ARB', 'OP', 'NEAR', 'SUI', 'APT']
for coin in l2_coins:
    BOT_CONFIGS_V3.append({
        'name': f'L2-ALPHA {coin}',
        'symbol': f'{coin}USDT',
        'strategy': RSI_MACD_Strategy(rsi_oversold=35, rsi_overbought=65),
        'mode': 'aggressive',
        'balance': 100,
        'interval': '1m',
    })

# 11. Crowd Psychology боты (психология толпы + теханализ)
crowd_coins = ['BTC', 'ETH', 'SOL', 'DOGE', 'PEPE', 'ARB']
for coin in crowd_coins:
    symbol = f'{coin}USDT'
    is_meme = coin in ['DOGE', 'PEPE']
    BOT_CONFIGS_V3.append({
        'name': f'CROWD-{coin} ⚡',
        'symbol': symbol,
        'strategy': CrowdPsychologyStrategy(
            panic_drop_pct=-0.85 if not is_meme else -1.4,
            euphoria_pump_pct=0.95 if not is_meme else 1.8,
            volume_spike=1.6 if not is_meme else 2.0,
            position_size_pct=28 if not is_meme else 36,
        ),
        'mode': 'aggressive' if not is_meme else 'degen',
        'balance': 120 if not is_meme else 90,
        'interval': '1m',
        'leverage': 8 if not is_meme else 12,
    })

# 12. News Flow Proxy боты (новостной фон через market-shock proxy)
news_coins = ['BTC', 'ETH', 'SOL', 'OP', 'NEAR', 'ARB']
for coin in news_coins:
    BOT_CONFIGS_V3.append({
        'name': f'NEWS-{coin} 📰',
        'symbol': f'{coin}USDT',
        'strategy': NewsFlowProxyStrategy(
            shock_z_threshold=2.0,
            reverse_z_threshold=1.3,
            position_size_pct=30,
        ),
        'mode': 'scalp',
        'balance': 120,
        'interval': '1m',
        'leverage': 9,
    })

# 13. Adaptive Hybrid Alpha (тех+психология+news proxy)
hybrid_coins = ['BTC', 'ETH', 'SOL', 'NEAR', 'DOGE', 'PEPE']
for coin in hybrid_coins:
    is_meme = coin in ['DOGE', 'PEPE']
    BOT_CONFIGS_V3.append({
        'name': f'HYBRID-{coin} 🧠',
        'symbol': f'{coin}USDT',
        'strategy': AdaptiveHybridAlphaStrategy(position_size_pct=32 if not is_meme else 40),
        'mode': 'scalp' if not is_meme else 'degen',
        'balance': 130 if not is_meme else 100,
        'interval': '1m',
        'leverage': 10 if not is_meme else 14,
    })


# 14. Winner-clones (копии лучших с мутациями параметров)
winner_clone_setups = {
    'XRP': [
        {'tag': 'S1', 'strategy': RSIStrategy(rsi_period=8, rsi_oversold=33, rsi_overbought=67), 'mode': 'scalp', 'balance': 150, 'leverage': 10},
        {'tag': 'S2', 'strategy': RSI_MACD_Strategy(rsi_oversold=34, rsi_overbought=66), 'mode': 'aggressive', 'balance': 150, 'leverage': 10},
        {'tag': 'S3', 'strategy': MACDStrategy(fast_period=6, slow_period=15, signal_period=4), 'mode': 'scalp', 'balance': 140, 'leverage': 11},
    ],
    'BNB': [
        {'tag': 'S1', 'strategy': RSIStrategy(rsi_period=9, rsi_oversold=32, rsi_overbought=68), 'mode': 'scalp', 'balance': 145, 'leverage': 9},
        {'tag': 'S2', 'strategy': RSI_MACD_Strategy(rsi_oversold=33, rsi_overbought=67), 'mode': 'balanced_plus', 'balance': 145, 'leverage': 9},
        {'tag': 'S3', 'strategy': MACDStrategy(fast_period=7, slow_period=16, signal_period=4), 'mode': 'scalp', 'balance': 140, 'leverage': 10},
    ],
    'NEAR': [
        {'tag': 'S1', 'strategy': RSIStrategy(rsi_period=7, rsi_oversold=36, rsi_overbought=64), 'mode': 'aggressive', 'balance': 130, 'leverage': 8},
        {'tag': 'S2', 'strategy': AdaptiveHybridAlphaStrategy(position_size_pct=34), 'mode': 'scalp', 'balance': 130, 'leverage': 9},
    ],
}

for coin, variants in winner_clone_setups.items():
    for v in variants:
        BOT_CONFIGS_V3.append({
            'name': f'WIN-CLONE {coin}-{v["tag"]} 🧬',
            'symbol': f'{coin}USDT',
            'strategy': v['strategy'],
            'mode': v['mode'],
            'balance': v['balance'],
            'interval': '1m',
            'leverage': v['leverage'],
        })

# 15. Scalp swarm (массовые скальперы)
scalp_swarm_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ARB', 'OP']
for coin in scalp_swarm_coins:
    BOT_CONFIGS_V3.append({
        'name': f'SCALP-RSI {coin} ⚡',
        'symbol': f'{coin}USDT',
        'strategy': RSIStrategy(rsi_period=6, rsi_oversold=41, rsi_overbought=59),
        'mode': 'scalp',
        'balance': 110,
        'interval': '1m',
        'leverage': 10 if coin in ['DOGE', 'ARB', 'OP'] else 8,
    })
    BOT_CONFIGS_V3.append({
        'name': f'SCALP-COMBO {coin} ⚔️',
        'symbol': f'{coin}USDT',
        'strategy': RSI_MACD_Strategy(rsi_oversold=38, rsi_overbought=62),
        'mode': 'scalp',
        'balance': 110,
        'interval': '1m',
        'leverage': 9 if coin in ['BTC', 'ETH', 'BNB', 'XRP'] else 11,
    })

# 16. Hyper-combo (комбинированные стратегии с повышенным риском)
hyper_combo_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'NEAR']
for coin in hyper_combo_coins:
    BOT_CONFIGS_V3.append({
        'name': f'HYPER-{coin} 🧠⚡',
        'symbol': f'{coin}USDT',
        'strategy': AdaptiveHybridAlphaStrategy(position_size_pct=38),
        'mode': 'degen' if coin in ['SOL', 'NEAR'] else 'aggressive',
        'balance': 140,
        'interval': '1m',
        'leverage': 12 if coin in ['SOL', 'NEAR'] else 10,
    })


# Итого: expanded bot set with multiple alpha families
print(f"Total configs: {len(BOT_CONFIGS_V3)}")


if __name__ == "__main__":
    for i, cfg in enumerate(BOT_CONFIGS_V3, 1):
        interval = cfg.get('interval', '5m')
        print(f"{i}. {cfg['name']} | {cfg['symbol']} | {cfg['mode']} | {interval}")
