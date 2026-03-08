"""
Live trading configuration.

Loads from .env + experiment results. Validates all required settings.
Supports multi-asset portfolio (BTC, ETH, BNB).
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a single strategy to trade live."""
    key: str                    # e.g. "btc_seed123_s19"
    symbol: str                 # e.g. "BTC/USDT:USDT"
    genome: List[int]           # Integer genome for decode()
    direction: str              # "LONG" or "SHORT"
    tp_atr_mult: float          # Take profit in ATR multiples
    sl_atr_mult: float          # Stop loss in ATR multiples
    trail_atr_mult: float       # Trailing stop (0 = no trail)
    expression: str             # Human-readable expression
    weight: float = 1.0 / 7    # Portfolio weight (default equal across 7 strategies)


@dataclass
class RiskConfig:
    """Risk management parameters."""
    leverage: int = 10                  # Leverage multiplier
    max_portfolio_dd_pct: float = 10.0  # Circuit breaker: halt if portfolio DD > X%
    max_daily_loss_pct: float = 3.0     # Halt trading for the day if loss > X%
    max_position_pct: float = 100.0     # No per-strategy cap
    max_open_positions: int = 999       # No limit on simultaneous positions
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
    timeframe: str = "15m"
    atr_period: int = 14
    lookback_bars: int = 200             # Bars of history to fetch for indicators
    poll_interval_seconds: int = 60

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

    @property
    def symbols(self) -> List[str]:
        """All unique symbols in the portfolio."""
        return list(dict.fromkeys(s.symbol for s in self.strategies))

    def strategies_for_symbol(self, symbol: str) -> List[StrategyConfig]:
        """Get strategies for a specific symbol."""
        return [s for s in self.strategies if s.symbol == symbol]


# ============================================================================
# MULTI-ASSET PORTFOLIO DEFINITION
# ============================================================================

# Portfolio: 7 strategies across 3 assets (v5b grammar, multi-TF)
# All strategies passed CPCV validation + signal permutation + OTS positive
PORTFOLIO = [
    # --- BTC/USDT: 2 SHORT + 1 LONG ---
    {
        'symbol': 'BTC/USDT:USDT',
        'results_dir': 'experiment_seed123_*',
        'seed': 123,
        'strategy_index': 19,
        'label': 'btc_seed123_s19',
        # SHORT MACD_NORM(8,21,9,1h) < -0.5 → CAGR +36.0%, 33 trades, Sharpe 1.82
    },
    {
        'symbol': 'BTC/USDT:USDT',
        'results_dir': 'experiment_seed123_*',
        'seed': 123,
        'strategy_index': 6,
        'label': 'btc_seed123_s6',
        # SHORT MACD_NORM(16,26,9) < 0.5 & ROC(3) < -1.5 → CAGR +31.5%, Sharpe 2.35
    },
    {
        'symbol': 'BTC/USDT:USDT',
        'results_dir': 'experiment_seed42_*',
        'seed': 42,
        'strategy_index': 19,
        'label': 'btc_seed42_s19',
        # LONG RSI(7,4h) < RSI(14) & STOCH_K(9) cross STOCH_D(5) → CAGR +15.3%
    },
    # --- ETH/USDT: 1 LONG + 1 SHORT ---
    {
        'symbol': 'ETH/USDT:USDT',
        'results_dir': 'experiment_ETH_USDT_seed123_*',
        'seed': 123,
        'strategy_index': 7,
        'label': 'eth_seed123_s7',
        # LONG RSI(7)×STOCH_K(5,4h) & ROC(13)×0.5 & MFI(7,1h)>RSI(14)
        # CAGR +27.6%, Sortino 0.579, 32 trades, Sharpe 1.65, PF 1.82
    },
    {
        'symbol': 'ETH/USDT:USDT',
        'results_dir': 'experiment_ETH_USDT_seed777_*',
        'seed': 777,
        'strategy_index': 7,
        'label': 'eth_seed777_s7',
        # SHORT ROC(21)×1.5 & STOCH_K(9)×ADX(21)
        # CAGR +19.9%, Sortino 0.826, 47 trades, Sharpe 1.64, PF 1.53
    },
    # --- BNB/USDT: 1 LONG + 1 SHORT ---
    {
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed123_*',
        'seed': 123,
        'strategy_index': 18,
        'label': 'bnb_seed123_s18',
        # LONG RSI(7)×STOCH_D(14,1h) & ROC(21,1h)<-0.5 & STOCH_D(5)>ADX(7)
        # CAGR +33.6%, Sortino 0.316, 70 trades, Sharpe 1.18, PF 1.37
    },
    {
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed777_*',
        'seed': 777,
        'strategy_index': 4,
        'label': 'bnb_seed777_s4',
        # SHORT ROC(5,1h)×1.0 & MACD_NORM(8,26,9,1h)>ROC(3,1h) & RSI(high,7,4h)>10
        # CAGR +21.4%, Sortino 0.505, 37 trades, Sharpe 2.05, PF 1.88
    },
]


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


def _find_experiment_dir(results_dir: Path, pattern: str, seed: int) -> Path:
    """Find the experiment directory matching a pattern and seed."""
    import glob
    matches = sorted(results_dir.glob(pattern))
    # Filter for the specific seed
    for d in reversed(matches):  # Most recent first
        if d.is_dir() and (d / 'ots_results.json').exists():
            # Check metadata for seed match
            meta_path = d / 'metadata.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('seed') == seed:
                    return d
            # Fallback: check dir name for seed
            if f'seed{seed}' in d.name:
                return d
    raise FileNotFoundError(
        f"No experiment directory matching '{pattern}' with seed {seed}"
    )


def load_strategies_from_results() -> List[StrategyConfig]:
    """Load multi-asset portfolio strategies from experiment results."""
    results_dir = Path(__file__).parent.parent / 'results'
    n_strategies = len(PORTFOLIO)
    strategies = []

    for entry in PORTFOLIO:
        try:
            exp_dir = _find_experiment_dir(
                results_dir, entry['results_dir'], entry['seed']
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {entry['label']}: {e}")
            continue

        with open(exp_dir / 'top_strategies.json') as f:
            top_strats = json.load(f)
        with open(exp_dir / 'ots_results.json') as f:
            ots_results = json.load(f)

        idx = entry['strategy_index']
        sd = top_strats[idx]

        # Get expression and direction from OTS results
        expr = ""
        direction = ""
        for r in ots_results:
            if r.get('strategy_index') == idx:
                expr = r.get('expression', '')
                direction = r.get('direction', '')
                break

        strategies.append(StrategyConfig(
            key=entry['label'],
            symbol=entry['symbol'],
            genome=sd['genome'],
            direction=direction or sd.get('direction', 'LONG'),
            tp_atr_mult=sd.get('tp_atr_mult', 0),
            sl_atr_mult=sd.get('sl_atr_mult', 1.0),
            trail_atr_mult=sd.get('trail_atr_mult', 0),
            expression=expr,
            weight=1.0 / n_strategies,
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
