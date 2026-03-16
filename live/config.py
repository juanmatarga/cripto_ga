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

N_STRATEGIES = 10


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
    weight: float = 1.0 / N_STRATEGIES  # Portfolio weight (equal across strategies)
    # CMA-ES parameter overrides: {param_name: optimized_value}
    # Applied via rebuild_strategy() after genome decode. Empty = use original params.
    cmaes_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Risk management parameters."""
    leverage: int = 10                  # Leverage multiplier
    max_portfolio_dd_pct: float = 10.0  # Circuit breaker: halt if portfolio DD > X%
    max_daily_loss_pct: float = 20.0    # Halt trading for the day if loss > X%
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

# Portfolio v2: 10 strategies across 3 assets (v5b grammar + CMA-ES optimization)
# BTC: original GE strategies (CMA-ES doesn't help BTC — sharp fitness landscape)
# ETH/BNB: best version per strategy (original or CMA-ES optimized)
#
# ROLLBACK: To revert any CMA-ES strategy, remove its 'cmaes_params' dict.
# The genome is always the original GE genome, so decode() gives original params.
# Previous portfolio (v1, 7 strategies) is in PORTFOLIO_V1_ROLLBACK below.

PORTFOLIO = [
    # --- BTC/USDT: 2 SHORT + 1 LONG (all original, NO CMA-ES) ---
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
    # --- ETH/USDT: 1 LONG + 2 SHORT ---
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
        # SHORT ROC(21)×1.5 & STOCH_K(9)×ADX(21) — original (CMA-ES = SAME)
        # CAGR +19.9%, Sortino 0.826, 47 trades, Sharpe 1.64, PF 1.53
    },
    {
        # NEW — CMA-ES optimized (OTS +10.1% → +31.3%, PBO 0.060 → 0.008)
        'symbol': 'ETH/USDT:USDT',
        'results_dir': 'experiment_ETH_USDT_seed777_*',
        'seed': 777,
        'strategy_index': 26,
        'label': 'eth_seed777_s26_cmaes',
        # SHORT RSI(close,21) < 28 EXIT TP=5.2 SL=1.1
        # Original: RSI(close,21) < 30 EXIT TP=5.0 SL=1.0
        'cmaes_params': {
            'c0_left_RSI_period': 20.586,
            'c0_right_threshold': 28.262,
            'tp_mult': 5.176,
            'sl_mult': 1.081,
        },
    },
    # --- BNB/USDT: 3 LONG + 1 SHORT ---
    {
        # UPDATED — CMA-ES optimized (OTS +33.6% → +41.3%, PBO 0.226 → 0.000)
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed123_*',
        'seed': 123,
        'strategy_index': 18,
        'label': 'bnb_seed123_s18_cmaes',
        # LONG RSI(7)×STOCH_D(13,1h) & ROC(22,1h)<-0.7 & STOCH_D(5)>ADX(6)
        # Original: RSI(7)×STOCH_D(14,1h) & ROC(21,1h)<-0.5 & STOCH_D(5)>ADX(7)
        'cmaes_params': {
            'c0_left_RSI_period': 7.321,
            'c0_right_STOCH_D_period': 12.802,
            'c1_left_ROC_period': 22.189,
            'c1_right_threshold': -0.749,
            'c2_left_STOCH_D_period': 5.058,
            'c2_right_ADX_period': 6.254,
            'tp_mult': 3.214,
            'sl_mult': 2.449,
        },
    },
    {
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed777_*',
        'seed': 777,
        'strategy_index': 4,
        'label': 'bnb_seed777_s4',
        # SHORT ROC(5,1h)×1.0 & MACD_NORM(8,26,9,1h)>ROC(3,1h) & RSI(high,7,4h)>10
        # Original (CMA-ES = WORSE, keep original)
        # CAGR +21.4%, Sortino 0.505, 37 trades, Sharpe 2.05, PF 1.88
    },
    {
        # NEW — CMA-ES optimized (OTS +11.7% → +64.6%, PBO 0.238 → 0.004)
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed42_*',
        'seed': 42,
        'strategy_index': 13,
        'label': 'bnb_seed42_s13_cmaes',
        # LONG ADX(21,4h) > MFI(19,1h) EXIT TP=3.3 SL=2.5
        # Original: ADX(21,4h) > MFI(21,1h) EXIT TP=3.0 SL=2.5
        'cmaes_params': {
            'c0_left_ADX_period': 21.113,
            'c0_right_MFI_period': 19.118,
            'tp_mult': 3.256,
            'sl_mult': 2.478,
        },
    },
    {
        # NEW — CMA-ES optimized (OTS +8.7% → +48.6%, PBO 0.139 → 0.000)
        'symbol': 'BNB/USDT:USDT',
        'results_dir': 'experiment_BNB_USDT_seed777_*',
        'seed': 777,
        'strategy_index': 25,
        'label': 'bnb_seed777_s25_cmaes',
        # LONG MACD_NORM(15,20,9)>-2.3 & STOCH_D(20,1h)×RSI(8) & PRICE_POS(59)>0.1
        # Original: MACD_NORM(16,21,9)>-2.0 & STOCH_D(21,1h)×RSI(7) & PRICE_POS(55)>0.0
        'cmaes_params': {
            'c0_left_MACD_NORM_fast': 14.839,
            'c0_left_MACD_NORM_slow': 19.800,
            'c0_left_MACD_NORM_signal': 9.278,
            'c0_right_threshold': -2.264,
            'c1_left_STOCH_D_period': 20.253,
            'c1_right_RSI_period': 7.730,
            'c2_left_PRICE_POS_period': 58.737,
            'c2_right_threshold': 0.090,
            'tp_mult': 4.579,
            'sl_mult': 1.222,
        },
    },
]

# ============================================================================
# ROLLBACK: previous portfolio (v1) — 7 strategies, all original GE params
# To rollback: set PORTFOLIO = PORTFOLIO_V1_ROLLBACK and N_STRATEGIES = 7
# ============================================================================
PORTFOLIO_V1_ROLLBACK = [
    {'symbol': 'BTC/USDT:USDT', 'results_dir': 'experiment_seed123_*', 'seed': 123, 'strategy_index': 19, 'label': 'btc_seed123_s19'},
    {'symbol': 'BTC/USDT:USDT', 'results_dir': 'experiment_seed123_*', 'seed': 123, 'strategy_index': 6, 'label': 'btc_seed123_s6'},
    {'symbol': 'BTC/USDT:USDT', 'results_dir': 'experiment_seed42_*', 'seed': 42, 'strategy_index': 19, 'label': 'btc_seed42_s19'},
    {'symbol': 'ETH/USDT:USDT', 'results_dir': 'experiment_ETH_USDT_seed123_*', 'seed': 123, 'strategy_index': 7, 'label': 'eth_seed123_s7'},
    {'symbol': 'ETH/USDT:USDT', 'results_dir': 'experiment_ETH_USDT_seed777_*', 'seed': 777, 'strategy_index': 7, 'label': 'eth_seed777_s7'},
    {'symbol': 'BNB/USDT:USDT', 'results_dir': 'experiment_BNB_USDT_seed123_*', 'seed': 123, 'strategy_index': 18, 'label': 'bnb_seed123_s18'},
    {'symbol': 'BNB/USDT:USDT', 'results_dir': 'experiment_BNB_USDT_seed777_*', 'seed': 777, 'strategy_index': 4, 'label': 'bnb_seed777_s4'},
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

        cmaes = entry.get('cmaes_params', {})

        # CMA-ES may override TP/SL — use optimized values for order placement
        tp = cmaes.get('tp_mult', sd.get('tp_atr_mult', 0))
        sl = cmaes.get('sl_mult', sd.get('sl_atr_mult', 1.0))

        strategies.append(StrategyConfig(
            key=entry['label'],
            symbol=entry['symbol'],
            genome=sd['genome'],
            direction=direction or sd.get('direction', 'LONG'),
            tp_atr_mult=tp,
            sl_atr_mult=sl,
            trail_atr_mult=sd.get('trail_atr_mult', 0),
            expression=expr,
            weight=1.0 / n_strategies,
            cmaes_params=cmaes,
        ))

    return strategies


def load_strategies_from_portfolio_json() -> List[StrategyConfig]:
    """
    Load strategies from results/final_portfolio.json (v6-v9 engine).

    This is the new portfolio format: genome + conditions stored directly in JSON,
    no dependency on experiment result directories.
    """
    portfolio_path = Path(__file__).parent.parent / 'results' / 'final_portfolio.json'
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")

    with open(portfolio_path) as f:
        portfolio = json.load(f)

    n = len(portfolio)
    strategies = []

    for i, entry in enumerate(portfolio):
        sym = entry['symbol']
        symbol_ccxt = f"{sym}/USDT:USDT"
        direction = entry['direction']
        conditions = entry.get('conditions', [])
        conds_str = '; '.join(conditions) if isinstance(conditions, list) else str(conditions)

        strategies.append(StrategyConfig(
            key=f"v9_{sym.lower()}_{direction.lower()}_{i+1}",
            symbol=symbol_ccxt,
            genome=entry['genome'],
            direction=direction,
            tp_atr_mult=entry.get('tp_atr_mult', 0),
            sl_atr_mult=entry.get('sl_atr_mult', 1.0),
            trail_atr_mult=entry.get('trail_atr_mult', 0),
            expression=conds_str,
            weight=1.0 / n,
        ))

    logger.info(f"Loaded {len(strategies)} strategies from {portfolio_path.name}")
    return strategies


# Toggle between portfolio versions:
# "v2"  = old 10-strategy portfolio (experiment results dirs)
# "v3"  = new 18-strategy portfolio (final_portfolio.json, v6-v9 engine)
ACTIVE_PORTFOLIO = "v3"


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

    # Load strategies based on active portfolio version
    if ACTIVE_PORTFOLIO == "v3":
        config.strategies = load_strategies_from_portfolio_json()
    else:
        config.strategies = load_strategies_from_results()

    # Update risk: raise circuit breaker for 10x leverage on 18 strategies
    config.risk.max_portfolio_dd_pct = 30.0

    # Validate
    if not config.active_api_key or config.active_api_key.startswith('your_'):
        raise ValueError(
            f"API key not configured for mode '{config.trading_mode}'. "
            f"Edit .env file with your {'testnet' if config.is_testnet else 'live'} credentials."
        )

    if not config.strategies:
        raise ValueError("No strategies loaded. Check portfolio configuration.")

    return config
