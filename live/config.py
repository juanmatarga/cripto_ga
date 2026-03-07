"""
Live trading configuration.

Loads from .env + live_config.yaml. Validates all required settings.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StrategyConfig:
    """Configuration for a single strategy to trade live."""
    key: str                    # e.g. "seed123_s2"
    genome: List[int]           # Integer genome for decode()
    direction: str              # "LONG" or "SHORT"
    tp_atr_mult: float          # Take profit in ATR multiples
    sl_atr_mult: float          # Stop loss in ATR multiples
    trail_atr_mult: float       # Trailing stop (0 = no trail)
    expression: str             # Human-readable expression
    weight: float = 1.0 / 3    # Portfolio weight (default equal)


@dataclass
class RiskConfig:
    """Risk management parameters."""
    leverage: int = 5                   # Leverage multiplier
    max_portfolio_dd_pct: float = 10.0  # Circuit breaker: halt if portfolio DD > X%
    max_daily_loss_pct: float = 3.0     # Halt trading for the day if loss > X%
    max_position_pct: float = 33.3      # Max capital per strategy (%)
    max_open_positions: int = 3         # Max simultaneous positions
    min_order_usdt: float = 10.0        # Minimum order size (Binance min)


@dataclass
class LiveConfig:
    """Complete live trading configuration."""
    # API
    api_key: str = ""
    api_secret: str = ""
    trading_mode: str = "testnet"       # "testnet" or "live"
    testnet_api_key: str = ""
    testnet_api_secret: str = ""

    # Trading
    symbol: str = "BTC/USDT:USDT"       # ccxt unified futures symbol
    timeframe: str = "15m"
    atr_period: int = 14
    lookback_bars: int = 200             # Bars of history to fetch for indicators
    poll_interval_seconds: int = 60      # Check every 60s (new candle every 900s)

    # Costs (for internal tracking, exchange applies real fees)
    fee_rate: float = 0.0004             # 0.04% taker fee on Binance Futures

    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Strategies
    strategies: List[StrategyConfig] = field(default_factory=list)

    @property
    def is_testnet(self) -> bool:
        return self.trading_mode == "testnet"

    @property
    def active_api_key(self) -> str:
        return self.testnet_api_key if self.is_testnet else self.api_key

    @property
    def active_api_secret(self) -> str:
        return self.testnet_api_secret if self.is_testnet else self.api_secret


def load_env():
    """Load .env file into environment variables."""
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        raise FileNotFoundError(
            f".env file not found at {env_path}. "
            f"Copy .env.example to .env and fill in your API credentials."
        )
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_strategies_from_results() -> List[StrategyConfig]:
    """Load the HIGH_RETURN portfolio strategies from experiment results."""
    import json

    # The 3 strategies in the HIGH_RETURN portfolio
    portfolio = [
        ('seed123', 2),   # SHORT — RSI + BBWIDTH + VOL_RATIO
        ('seed123', 0),   # LONG  — STOCH + RSI + PCT_B crossovers
        ('seed777', 4),   # SHORT — ROC + STOCH
    ]

    results_dir = Path(__file__).parent.parent / 'results'
    experiments = {}
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / 'ots_results.json').exists() and '_seed' in d.name:
            seed_part = d.name.split('_seed')[1].split('_')[0]
            experiments[f'seed{seed_part}'] = d

    strategies = []
    for seed_key, idx in portfolio:
        exp_dir = experiments[seed_key]

        with open(exp_dir / 'top_strategies.json') as f:
            top_strats = json.load(f)
        with open(exp_dir / 'ots_results.json') as f:
            ots_results = json.load(f)

        sd = top_strats[idx]

        # Get expression from OTS results
        expr = ""
        direction = ""
        for r in ots_results:
            if r.get('strategy_index') == idx:
                expr = r.get('expression', '')
                direction = r.get('direction', '')
                break

        strategies.append(StrategyConfig(
            key=f"{seed_key}_s{idx}",
            genome=sd['genome'],
            direction=direction or sd.get('direction', 'LONG'),
            tp_atr_mult=sd.get('tp_atr_mult', 0),
            sl_atr_mult=sd.get('sl_atr_mult', 1.0),
            trail_atr_mult=sd.get('trail_atr_mult', 0),
            expression=expr,
            weight=1.0 / len(portfolio),
        ))

    return strategies


def load_config() -> LiveConfig:
    """Load complete live trading configuration."""
    load_env()

    config = LiveConfig(
        api_key=os.environ.get('BINANCE_API_KEY', ''),
        api_secret=os.environ.get('BINANCE_API_SECRET', ''),
        trading_mode=os.environ.get('TRADING_MODE', 'testnet'),
        testnet_api_key=os.environ.get('BINANCE_TESTNET_API_KEY', ''),
        testnet_api_secret=os.environ.get('BINANCE_TESTNET_API_SECRET', ''),
    )

    # Load strategies from experiment results
    config.strategies = load_strategies_from_results()

    # Validate
    if not config.active_api_key or config.active_api_key.startswith('your_'):
        raise ValueError(
            f"API key not configured for mode '{config.trading_mode}'. "
            f"Edit .env file with your {'testnet' if config.is_testnet else 'live'} credentials."
        )

    if not config.strategies:
        raise ValueError("No strategies loaded. Check experiment results.")

    return config
