"""
Generate 4-panel analysis figures for each of the 10 portfolio strategies.

For each strategy:
  Top-left:     Equity curve with Monte Carlo envelope (1000 sims)
  Top-right:    Monte Carlo distribution of final equity
  Bottom-left:  Trade PnL distribution (wins vs losses)
  Bottom-right: Performance metrics + Monte Carlo validation text

Extended OTS period: June 2025 to February 2026.
"""

import json
import logging
import traceback
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
from backtest.metrics import (
    calculate_returns, calculate_sortino_ratio, calculate_calmar_ratio,
    sharpe_ratio, cagr, max_drawdown,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'paper' / 'figures' / 'strategy_analysis'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OTS_START = '2025-06-01'
OTS_END = '2026-02-28'
INITIAL_CAPITAL = 1000.0
N_SIMULATIONS = 1000
BARS_PER_YEAR_15M = 35040

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

# Data configs with extended OTS end date
DATA_CONFIGS = {
    'BTC/USDT:USDT': {
        'data': {
            'symbol': 'BTC/USDT', 'timeframe': '15m',
            'start': '2022-01-01', 'end': OTS_END,
            'exchange': 'binance', 'market_type': 'future',
        }
    },
    'ETH/USDT:USDT': {
        'data': {
            'symbol': 'ETH/USDT', 'timeframe': '15m',
            'start': '2022-01-01', 'end': OTS_END,
            'exchange': 'binance', 'market_type': 'future',
        }
    },
    'BNB/USDT:USDT': {
        'data': {
            'symbol': 'BNB/USDT', 'timeframe': '15m',
            'start': '2022-01-01', 'end': OTS_END,
            'exchange': 'binance', 'market_type': 'future',
        }
    },
}


# ============================================================================
# Data & Strategy Loading
# ============================================================================

def load_ots_data():
    """Load OTS data for each symbol. Returns dict of {symbol: ots_df}."""
    ots_data = {}
    tf_data_cache = {}
    for symbol, config in DATA_CONFIGS.items():
        logger.info(f"Loading data for {symbol}...")
        df = load_data(config, use_cache=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        ots = df[df.index >= OTS_START].copy()
        logger.info(f"  OTS: {len(ots)} bars ({ots.index.min()} to {ots.index.max()})")
        ots_data[symbol] = ots
        tf_data_cache[symbol] = prepare_multi_tf_data(ots)
    return ots_data, tf_data_cache


def load_strategy(entry):
    """Load and decode a strategy from experiment results, applying CMA-ES if needed."""
    exp_dir = _find_experiment_dir(RESULTS_DIR, entry['results_dir'], entry['seed'])
    with open(exp_dir / 'top_strategies.json') as f:
        top_strats = json.load(f)

    idx = entry['strategy_index']
    sd = top_strats[idx]
    strategy = decode(sd['genome'])

    if strategy is None:
        raise ValueError(f"Failed to decode genome for {entry['label']}")

    cmaes_params = entry.get('cmaes_params', {})
    if cmaes_params:
        params = extract_params(strategy)
        param_vector = [cmaes_params.get(p.name, p.value) for p in params]
        strategy = rebuild_strategy(strategy, param_vector, params)

    return strategy


# ============================================================================
# Trade & Metrics Computation
# ============================================================================

def compute_trades_usd(trades, initial_capital=INITIAL_CAPITAL):
    """Convert pnl_pct trades to pnl_usd with compounding equity."""
    equity = initial_capital
    trades_usd = []
    for t in trades:
        pnl_usd = equity * t['pnl_pct']
        equity += pnl_usd
        trades_usd.append({**t, 'pnl_usd': pnl_usd, 'equity_after': equity})
    return trades_usd


def compute_metrics(trades_usd, equity_series, initial_capital=INITIAL_CAPITAL):
    """Compute comprehensive performance metrics."""
    if not trades_usd:
        return {}

    n_trades = len(trades_usd)
    final_equity = trades_usd[-1]['equity_after']
    total_return_pct = (final_equity / initial_capital - 1) * 100

    pnls_usd = [t['pnl_usd'] for t in trades_usd]
    pnls_pct = [t['pnl_pct'] for t in trades_usd]
    wins_usd = [p for p in pnls_usd if p > 0]
    losses_usd = [p for p in pnls_usd if p <= 0]

    win_rate = len(wins_usd) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins_usd) if wins_usd else 0
    avg_loss = np.mean(losses_usd) if losses_usd else 0

    gross_profit = sum(wins_usd) if wins_usd else 0
    gross_loss = abs(sum(losses_usd)) if losses_usd else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Max drawdown from trade-by-trade equity
    eq_arr = np.array([initial_capital] + [t['equity_after'] for t in trades_usd])
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd_pct = abs(dd.min()) * 100

    # Annualized metrics from bar-by-bar equity
    returns = calculate_returns(equity_series).dropna()
    sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
    sharpe = sharpe_ratio(equity_series, BARS_PER_YEAR_15M)
    cagr_val = cagr(equity_series, BARS_PER_YEAR_15M)
    calmar_val = calculate_calmar_ratio(cagr_val, max_drawdown(equity_series))

    return {
        'total_trades': n_trades,
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd_pct,
        'sortino': sortino,
        'sharpe': sharpe,
        'cagr': cagr_val,
        'calmar': calmar_val,
    }


# ============================================================================
# Monte Carlo Simulation (pnl_pct shuffling with compounding)
# ============================================================================

def run_monte_carlo_pct(trades, initial_capital=INITIAL_CAPITAL, n_simulations=N_SIMULATIONS):
    """
    Monte Carlo via shuffling pnl_pct and recomputing compounding equity.

    More correct than shuffling dollar amounts because it preserves
    the multiplicative nature of returns.
    """
    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    n_trades = len(pnl_pcts)

    # Actual equity curve (trade-by-trade, including initial)
    actual_equity = np.empty(n_trades + 1)
    actual_equity[0] = initial_capital
    for i, pnl in enumerate(pnl_pcts):
        actual_equity[i + 1] = actual_equity[i] * (1 + pnl)
    actual_final = actual_equity[-1]

    # Run simulations
    rng = np.random.default_rng(42)
    simulated_finals = np.empty(n_simulations)
    # Store curves including initial capital (n_trades + 1 points each)
    simulated_curves = np.empty((n_simulations, n_trades + 1))

    for s in range(n_simulations):
        shuffled = rng.permutation(pnl_pcts)
        simulated_curves[s, 0] = initial_capital
        for i, pnl in enumerate(shuffled):
            simulated_curves[s, i + 1] = simulated_curves[s, i] * (1 + pnl)
        simulated_finals[s] = simulated_curves[s, -1]

    # Percentiles
    percentiles = {
        'p5': float(np.percentile(simulated_finals, 5)),
        'p25': float(np.percentile(simulated_finals, 25)),
        'p50': float(np.percentile(simulated_finals, 50)),
        'p75': float(np.percentile(simulated_finals, 75)),
        'p95': float(np.percentile(simulated_finals, 95)),
    }

    actual_percentile = float((simulated_finals < actual_final).sum() / n_simulations * 100)

    return {
        'actual_equity': actual_equity,
        'actual_final': float(actual_final),
        'actual_return_pct': float((actual_final - initial_capital) / initial_capital * 100),
        'actual_percentile': actual_percentile,
        'simulated_finals': simulated_finals,
        'simulated_curves': simulated_curves,  # (n_sims, n_trades+1)
        'percentiles': percentiles,
        'mean_final': float(simulated_finals.mean()),
        'std_final': float(simulated_finals.std()),
        'best_case': float(simulated_finals.max()),
        'worst_case': float(simulated_finals.min()),
        'prob_profitable': float((simulated_finals > initial_capital).sum() / n_simulations),
    }


# ============================================================================
# Figure Generation
# ============================================================================

def plot_strategy_analysis(label, entry_key, strategy, trades_usd, mc_results,
                           metrics, output_path):
    """Create 4-panel analysis figure for a single strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    n_trades = len(trades_usd)
    trade_nums = np.arange(n_trades + 1)  # 0 to N (including initial capital)

    # ========================
    # Top-left: Equity Curve with Monte Carlo Envelope
    # ========================
    ax = axes[0, 0]

    mc_curves = mc_results['simulated_curves']  # (n_sims, n_trades+1)
    p5 = np.percentile(mc_curves, 5, axis=0)
    p25 = np.percentile(mc_curves, 25, axis=0)
    p50 = np.percentile(mc_curves, 50, axis=0)
    p75 = np.percentile(mc_curves, 75, axis=0)
    p95 = np.percentile(mc_curves, 95, axis=0)

    ax.fill_between(trade_nums, p5, p95, alpha=0.15, color='gray',
                     label='5th-95th percentile')
    ax.fill_between(trade_nums, p25, p75, alpha=0.3, color='gray',
                     label='25th-75th percentile')
    ax.plot(trade_nums, p50, '--', color='gray', linewidth=1.5,
            label='Median (Monte Carlo)')

    # Actual equity curve
    actual_eq = mc_results['actual_equity']
    ax.plot(trade_nums, actual_eq, 'b-', linewidth=2.5, label='Actual Strategy')
    ax.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5,
               label='Initial Capital')

    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Equity (USD)', fontsize=12)
    ax.set_title(f'Equity Curve with Monte Carlo Validation ({N_SIMULATIONS} sims)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========================
    # Top-right: Monte Carlo Distribution of Final Equity
    # ========================
    ax = axes[0, 1]

    ax.hist(mc_results['simulated_finals'], bins=50, color='skyblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(mc_results['actual_final'], color='blue', linewidth=3,
               label=f'Actual: ${mc_results["actual_final"]:.2f}')
    ax.axvline(mc_results['percentiles']['p50'], color='gray', linestyle='--',
               linewidth=2,
               label=f'Median: ${mc_results["percentiles"]["p50"]:.2f}')
    ax.axvline(INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5,
               label='Break-even')

    ax.set_xlabel('Final Equity (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(
        f'Monte Carlo Distribution (Actual at {mc_results["actual_percentile"]:.1f}th %ile)',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================
    # Bottom-left: Trade PnL Distribution
    # ========================
    ax = axes[1, 0]

    pnls = np.array([t['pnl_usd'] for t in trades_usd])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    if len(wins) > 0 and len(losses) > 0:
        all_pnls = np.concatenate([wins, losses])
        bins = np.linspace(all_pnls.min(), all_pnls.max(), 40)
        ax.hist(wins, bins=bins, color='green', alpha=0.7, label='Wins')
        ax.hist(losses, bins=bins, color='red', alpha=0.7, label='Losses')
    elif len(wins) > 0:
        ax.hist(wins, bins=30, color='green', alpha=0.7, label='Wins')
    elif len(losses) > 0:
        ax.hist(losses, bins=30, color='red', alpha=0.7, label='Losses')

    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Trade PnL (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Trade PnL Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================
    # Bottom-right: Metrics Text Panel
    # ========================
    ax = axes[1, 1]
    ax.axis('off')

    # Truncate long expressions
    expr = strategy.expression_raw
    if len(expr) > 70:
        expr = expr[:67] + '...'

    # Format CMA-ES tag
    cmaes_tag = " [CMA-ES]" if entry_key.endswith('_cmaes') else ""

    sep = '=' * 40
    metrics_text = (
        f"PATTERN: {strategy.direction}{cmaes_tag}\n"
        f"{expr}\n"
        f"\n"
        f"PERFORMANCE METRICS:\n"
        f"{sep}\n"
        f"Total Trades: {metrics['total_trades']}\n"
        f"Final Equity: ${metrics['final_equity']:.2f}\n"
        f"Total Return: {metrics['total_return_pct']:+.1f}%\n"
        f"\n"
        f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
        f"Avg Win: ${metrics['avg_win']:.2f}\n"
        f"Avg Loss: ${metrics['avg_loss']:.2f}\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%\n"
        f"\n"
        f"Sortino: {metrics['sortino']:.2f}\n"
        f"Sharpe:  {metrics['sharpe']:.2f}\n"
        f"CAGR:    {metrics['cagr']*100:+.1f}%\n"
        f"Calmar:  {metrics['calmar']:.2f}\n"
        f"\n"
        f"MONTE CARLO VALIDATION:\n"
        f"{sep}\n"
        f"Actual Percentile: {mc_results['actual_percentile']:.1f}th\n"
        f"Prob(Profitable): {mc_results['prob_profitable']*100:.1f}%\n"
        f"Expected Return: ${mc_results['mean_final']:.2f}\u00b1{mc_results['std_final']:.2f}\n"
        f"\n"
        f"Best Case:  ${mc_results['best_case']:.2f}\n"
        f"Worst Case: ${mc_results['worst_case']:.2f}"
    )

    ax.text(0.05, 0.95, metrics_text, fontsize=10.5, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(f'{label} ({entry_key}) \u2014 OTS Analysis (Jun 2025 \u2013 Feb 2026)',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(summary_data, output_path):
    """Create a visual summary table of all 10 strategies."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')

    headers = ['Strategy', 'Symbol', 'Dir', 'Trades', 'Return', 'Win%',
               'PF', 'MaxDD', 'Sortino', 'Sharpe', 'CAGR', 'MC %ile', 'P(>0)']

    rows = []
    for r in summary_data:
        rows.append([
            r['label'],
            r['symbol'],
            r['direction'],
            str(r['n_trades']),
            f"{r['total_return']:+.1f}%",
            f"{r['win_rate']*100:.1f}%",
            f"{r['profit_factor']:.2f}",
            f"{r['max_dd']:.1f}%",
            f"{r['sortino']:.2f}",
            f"{r['sharpe']:.2f}",
            f"{r['cagr']*100:+.1f}%",
            f"{r['mc_percentile']:.1f}%",
            f"{r['mc_prob_profitable']*100:.0f}%",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j, header in enumerate(headers):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color code returns
    for i, r in enumerate(summary_data):
        ret = r['total_return']
        color = '#C6EFCE' if ret > 0 else '#FFC7CE'
        table[i + 1, 4].set_facecolor(color)

        # Color code MC percentile
        pctile = r['mc_percentile']
        if pctile >= 75:
            table[i + 1, 11].set_facecolor('#C6EFCE')
        elif pctile <= 25:
            table[i + 1, 11].set_facecolor('#FFC7CE')

    ax.set_title(f'Portfolio Strategy Analysis Summary \u2014 OTS Jun 2025 to Feb 2026',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Generating strategy analysis figures (4-panel per strategy)")
    logger.info(f"OTS period: {OTS_START} to {OTS_END}")
    logger.info(f"Initial capital: ${INITIAL_CAPITAL:.0f}")
    logger.info(f"Monte Carlo simulations: {N_SIMULATIONS}")
    logger.info("=" * 70)

    # 1. Load OTS data
    logger.info("\n--- Loading OTS data ---")
    ots_data, tf_data_cache = load_ots_data()

    # 2. Analyze each strategy
    summary = []
    for i, entry in enumerate(PORTFOLIO):
        label = LABELS[i]
        symbol = entry['symbol']
        entry_key = entry['label']
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/10] {label} ({entry_key}) on {symbol}")
        logger.info(f"{'='*60}")

        try:
            # Load strategy
            strategy = load_strategy(entry)
            logger.info(f"  Expression: {strategy.expression_raw[:100]}")
            logger.info(f"  Direction: {strategy.direction}, "
                        f"TP={strategy.tp_atr_mult}, SL={strategy.sl_atr_mult}, "
                        f"Trail={strategy.trail_atr_mult}")

            # Run backtest on extended OTS
            ots_df = ots_data[symbol]
            tf_data = tf_data_cache[symbol]
            equity_series, trades = _run_single_window(
                strategy, ots_df, COSTS_CONFIG, ATR_PERIOD, tf_data=tf_data,
            )

            if not trades:
                logger.warning(f"  No trades for {label}. Skipping.")
                continue

            # Convert to USD (for PnL distribution and Monte Carlo)
            trades_usd = compute_trades_usd(trades)

            # Compute metrics (using both trade-level and bar-level equity)
            metrics = compute_metrics(trades_usd, equity_series)
            logger.info(f"  Trades: {metrics['total_trades']}, "
                        f"Return: {metrics['total_return_pct']:+.1f}%, "
                        f"WR: {metrics['win_rate']*100:.1f}%, "
                        f"PF: {metrics['profit_factor']:.2f}, "
                        f"Sortino: {metrics['sortino']:.2f}")

            # Run Monte Carlo (pnl_pct shuffling with compounding)
            logger.info(f"  Running Monte Carlo ({N_SIMULATIONS} simulations)...")
            mc_results = run_monte_carlo_pct(trades)
            logger.info(f"  MC: Actual at {mc_results['actual_percentile']:.1f}th %ile, "
                        f"P(profitable)={mc_results['prob_profitable']*100:.1f}%")

            # Generate 4-panel figure
            output_path = FIGURES_DIR / f'{entry_key}_analysis.png'
            plot_strategy_analysis(label, entry_key, strategy, trades_usd,
                                   mc_results, metrics, output_path)

            summary.append({
                'label': label,
                'key': entry_key,
                'symbol': symbol.split('/')[0],
                'direction': strategy.direction,
                'n_trades': metrics['total_trades'],
                'total_return': metrics['total_return_pct'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_dd': metrics['max_drawdown_pct'],
                'sortino': metrics['sortino'],
                'sharpe': metrics['sharpe'],
                'cagr': metrics['cagr'],
                'calmar': metrics['calmar'],
                'mc_percentile': mc_results['actual_percentile'],
                'mc_prob_profitable': mc_results['prob_profitable'],
                'mc_mean_final': mc_results['mean_final'],
                'mc_worst_case': mc_results['worst_case'],
            })

        except Exception as e:
            logger.error(f"  ERROR processing {label}: {e}")
            traceback.print_exc()
            continue

    if not summary:
        logger.error("No strategies processed successfully!")
        return

    # 3. Generate summary table figure
    logger.info("\n--- Generating summary table ---")
    plot_summary_table(summary, FIGURES_DIR / 'summary_table.png')

    # 4. Print console summary
    print(f"\n{'='*110}")
    print(f"STRATEGY ANALYSIS SUMMARY \u2014 OTS {OTS_START} to {OTS_END}")
    print(f"{'='*110}")
    print(f"{'Label':<12} {'Key':<28} {'Sym':>4} {'Dir':>6} {'#Tr':>4} "
          f"{'Return':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} "
          f"{'Sort':>6} {'Sharp':>6} {'CAGR':>7} {'MC%ile':>7} {'P(>0)':>6}")
    print(f"{'-'*110}")
    for r in summary:
        print(f"{r['label']:<12} {r['key']:<28} {r['symbol']:>4} {r['direction']:>6} "
              f"{r['n_trades']:>4} {r['total_return']:>+7.1f}% "
              f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['max_dd']:>6.1f}% {r['sortino']:>5.2f} "
              f"{r['sharpe']:>5.2f} {r['cagr']*100:>+6.1f}% "
              f"{r['mc_percentile']:>6.1f}% "
              f"{r['mc_prob_profitable']*100:>5.0f}%")

    # Aggregate stats
    positive = sum(1 for r in summary if r['total_return'] > 0)
    avg_return = np.mean([r['total_return'] for r in summary])
    avg_sortino = np.mean([r['sortino'] for r in summary])
    print(f"\n  Positive strategies: {positive}/{len(summary)} ({positive/len(summary)*100:.0f}%)")
    print(f"  Average return: {avg_return:+.1f}%")
    print(f"  Average Sortino: {avg_sortino:.2f}")

    print(f"\nFigures saved to: {FIGURES_DIR}")

    # 5. Save summary JSON
    output_json = FIGURES_DIR / 'analysis_summary.json'
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {output_json}")


if __name__ == '__main__':
    main()
