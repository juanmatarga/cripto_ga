"""
Generate updated figures for the research paper.

Figure 1: Portfolio equity curve (10-strategy ensemble vs Buy & Hold)
Figure 2: Correlation heatmap between the 10 strategies

Uses OTS period: June 2025 to November 2025.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from grammar.mapper import decode
from evolution.fitness import _run_single_window
from evolution.param_extractor import extract_params, rebuild_strategy
from data.loader import load_data
from data.multi_timeframe import prepare_multi_tf_data
from live.config import PORTFOLIO, _find_experiment_dir

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'paper' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OTS_START = '2025-06-01'

# Costs config (matching live config)
COSTS_CONFIG = {
    'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
    'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
}
ATR_PERIOD = 14

# Short labels for the 10 strategies (order matches PORTFOLIO)
LABELS = [
    "BTC S1",    # btc_seed123_s19  SHORT
    "BTC S2",    # btc_seed123_s6   SHORT
    "BTC L1",    # btc_seed42_s19   LONG
    "ETH L1",    # eth_seed123_s7   LONG
    "ETH S1",    # eth_seed777_s7   SHORT
    "ETH S2*",   # eth_seed777_s26  SHORT CMA-ES
    "BNB L1*",   # bnb_seed123_s18  LONG  CMA-ES
    "BNB S1",    # bnb_seed777_s4   SHORT
    "BNB L2*",   # bnb_seed42_s13   LONG  CMA-ES
    "BNB L3*",   # bnb_seed777_s25  LONG  CMA-ES
]

# Unique symbols
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# Data config per symbol
DATA_CONFIGS = {
    'BTC/USDT:USDT': {
        'data': {
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
            'exchange': 'binance',
            'market_type': 'future',
        }
    },
    'ETH/USDT:USDT': {
        'data': {
            'symbol': 'ETH/USDT',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
            'exchange': 'binance',
            'market_type': 'future',
        }
    },
    'BNB/USDT:USDT': {
        'data': {
            'symbol': 'BNB/USDT',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
            'exchange': 'binance',
            'market_type': 'future',
        }
    },
}


def load_ots_data():
    """Load OTS data for each symbol. Returns dict of {symbol: ots_df}."""
    ots_data = {}
    tf_data_cache = {}
    for symbol, config in DATA_CONFIGS.items():
        logger.info(f"Loading data for {symbol}...")
        df = load_data(config, use_cache=True)
        ots = df[df.index >= OTS_START].copy()
        logger.info(f"  OTS: {len(ots)} bars ({ots.index.min()} to {ots.index.max()})")
        ots_data[symbol] = ots
        tf_data_cache[symbol] = prepare_multi_tf_data(ots)
    return ots_data, tf_data_cache


def load_strategy(entry):
    """Load and decode a strategy from experiment results, applying CMA-ES if needed."""
    exp_dir = _find_experiment_dir(
        RESULTS_DIR, entry['results_dir'], entry['seed']
    )
    with open(exp_dir / 'top_strategies.json') as f:
        top_strats = json.load(f)

    idx = entry['strategy_index']
    sd = top_strats[idx]
    strategy = decode(sd['genome'])

    if strategy is None:
        raise ValueError(f"Failed to decode genome for {entry['label']}")

    # Apply CMA-ES parameter overrides if present
    cmaes_params = entry.get('cmaes_params', {})
    if cmaes_params:
        params = extract_params(strategy)
        param_vector = [cmaes_params.get(p.name, p.value) for p in params]
        strategy = rebuild_strategy(strategy, param_vector, params)
        logger.info(f"  Applied CMA-ES params: {strategy.expression_raw}")

    return strategy


def run_all_strategies(ots_data, tf_data_cache):
    """
    Run all 10 strategies on OTS data.

    Returns:
        equity_dict: {label: pd.Series} equity curves (starting at 100)
        returns_dict: {label: pd.Series} bar-by-bar returns
    """
    equity_dict = {}
    returns_dict = {}

    for i, entry in enumerate(PORTFOLIO):
        label = LABELS[i]
        symbol = entry['symbol']
        logger.info(f"[{i+1}/10] Running {label} ({entry['label']}) on {symbol}...")

        strategy = load_strategy(entry)
        ots_df = ots_data[symbol]
        tf_data = tf_data_cache[symbol]

        equity, trades = _run_single_window(
            strategy, ots_df, COSTS_CONFIG, ATR_PERIOD, tf_data=tf_data,
        )

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        logger.info(f"  {label}: {len(trades)} trades, return={total_return:+.1f}%")

        equity_dict[label] = equity
        returns_dict[label] = equity.pct_change().fillna(0.0)

    return equity_dict, returns_dict


def compute_bh_curves(ots_data):
    """Compute Buy & Hold equity curves for each symbol, starting at 100."""
    bh_curves = {}
    for symbol_full, ots_df in ots_data.items():
        # Extract short name (BTC, ETH, BNB)
        short = symbol_full.split('/')[0]
        close = ots_df['Close']
        bh = (close / close.iloc[0]) * 100.0
        bh_curves[short] = bh
        total = (bh.iloc[-1] - 100.0)
        logger.info(f"  B&H {short}: {total:+.1f}%")
    return bh_curves


def plot_equity_curve(equity_dict, returns_dict, bh_curves):
    """
    Figure 1: Portfolio equity curve vs Buy & Hold.
    """
    # Combine all strategy returns into an equal-weight portfolio
    # Align all returns to a common index (use BTC's OTS index as reference)
    # Each strategy operates on its own symbol's data, so indices match per symbol.
    # We need a common 15m index. Since all OTS data covers same period, they share the index.
    all_returns = pd.DataFrame(returns_dict)
    portfolio_returns = all_returns.mean(axis=1)
    portfolio_equity = (1 + portfolio_returns).cumprod() * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))

    # Portfolio
    ax.plot(portfolio_equity.index, portfolio_equity.values,
            color='#1f77b4', linewidth=2.0, label='Portafolio (10 estrategias)', zorder=5)

    # B&H lines
    bh_styles = {
        'BTC': ('gray', '--', 'B&H BTC'),
        'ETH': ('gray', ':', 'B&H ETH'),
        'BNB': ('gray', '-.', 'B&H BNB'),
    }
    for asset, (color, ls, lbl) in bh_styles.items():
        if asset in bh_curves:
            curve = bh_curves[asset]
            ax.plot(curve.index, curve.values, color=color, linestyle=ls,
                    linewidth=1.2, label=lbl, alpha=0.7)

    # Horizontal line at 100
    ax.axhline(y=100, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

    ax.set_title('Portafolio combinado vs Buy & Hold (junio\u2013noviembre 2025)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Capital (base = 100)', fontsize=12)
    ax.set_xlabel('')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Final values annotation
    final_port = portfolio_equity.iloc[-1]
    ax.annotate(f'{final_port:.0f}',
                xy=(portfolio_equity.index[-1], final_port),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, color='#1f77b4', fontweight='bold')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'fig1_ensemble_equity.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return portfolio_equity


def plot_correlation_heatmap(returns_dict):
    """
    Figure 2: 10x10 correlation heatmap between strategy returns.
    """
    # Build DataFrame of returns, columns ordered by LABELS
    returns_df = pd.DataFrame(returns_dict)
    # Ensure column order matches LABELS
    returns_df = returns_df[LABELS]

    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = corr.values[i, j]
            # Choose text color based on background
            text_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    # Axis labels
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9)

    ax.set_title('Correlación entre estrategias', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Correlación', fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'fig3_correlation.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("Generating paper figures")
    logger.info("=" * 60)

    # 1. Load OTS data
    logger.info("\n--- Loading OTS data ---")
    ots_data, tf_data_cache = load_ots_data()

    # 2. Run all 10 strategies
    logger.info("\n--- Running strategies on OTS ---")
    equity_dict, returns_dict = run_all_strategies(ots_data, tf_data_cache)

    # 3. Compute B&H curves
    logger.info("\n--- Buy & Hold curves ---")
    bh_curves = compute_bh_curves(ots_data)

    # 4. Figure 1: Equity curve
    logger.info("\n--- Figure 1: Portfolio equity curve ---")
    portfolio_equity = plot_equity_curve(equity_dict, returns_dict, bh_curves)
    port_return = (portfolio_equity.iloc[-1] / 100.0 - 1) * 100
    logger.info(f"Portfolio total return: {port_return:+.1f}%")

    # 5. Figure 2: Correlation heatmap
    logger.info("\n--- Figure 2: Correlation heatmap ---")
    plot_correlation_heatmap(returns_dict)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Done! Figures saved to:")
    logger.info(f"  {FIGURES_DIR / 'fig1_ensemble_equity.png'}")
    logger.info(f"  {FIGURES_DIR / 'fig3_correlation.png'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
