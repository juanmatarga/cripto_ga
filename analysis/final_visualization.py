"""
Professional visualization for final backtest results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict
import logging

sns.set_style("whitegrid")
logger = logging.getLogger(__name__)


def plot_equity_with_monte_carlo(backtest_results: Dict,
                                mc_results: Dict,
                                output_path: str):
    """
    Create professional plot with equity curve and Monte Carlo envelope.

    Args:
        backtest_results: From run_final_backtest
        mc_results: From run_monte_carlo
        output_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Equity Curve with Monte Carlo Envelope
    ax = axes[0, 0]

    equity_curve = backtest_results['equity_curve']

    # Plot Monte Carlo percentiles
    mc_curves = np.array(mc_results['simulated_curves'])
    trade_nums = range(1, len(mc_curves[0]) + 1)

    p5 = np.percentile(mc_curves, 5, axis=0)
    p25 = np.percentile(mc_curves, 25, axis=0)
    p50 = np.percentile(mc_curves, 50, axis=0)
    p75 = np.percentile(mc_curves, 75, axis=0)
    p95 = np.percentile(mc_curves, 95, axis=0)

    # Plot envelope
    ax.fill_between(trade_nums, p5, p95, alpha=0.2, color='gray', label='5th-95th percentile')
    ax.fill_between(trade_nums, p25, p75, alpha=0.3, color='gray', label='25th-75th percentile')
    ax.plot(trade_nums, p50, '--', color='gray', linewidth=2, label='Median (Monte Carlo)')

    # Plot actual
    ax.plot(equity_curve.index + 1, equity_curve['equity'], 'b-', linewidth=2.5, label='Actual Strategy')
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Initial Capital')

    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Equity (USD)', fontsize=12)
    ax.set_title(f'Equity Curve with Monte Carlo Validation (1000 sims)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of Final Equity
    ax = axes[0, 1]

    ax.hist(mc_results['simulated_finals'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mc_results['actual_final'], color='blue', linewidth=3, label=f'Actual: ${mc_results["actual_final"]:.2f}')
    ax.axvline(mc_results['percentiles']['p50'], color='gray', linestyle='--', linewidth=2, label=f'Median: ${mc_results["percentiles"]["p50"]:.2f}')
    ax.axvline(1000, color='red', linestyle='--', alpha=0.5, label='Break-even')

    ax.set_xlabel('Final Equity (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Monte Carlo Distribution (Actual at {mc_results["actual_percentile"]:.1f}th %ile)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Trade PnL Distribution
    ax = axes[1, 0]

    trades_df = pd.DataFrame(backtest_results['trades'])
    pnls = trades_df['pnl_usd'].values

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    ax.hist([wins, losses], bins=30, color=['green', 'red'], alpha=0.7, label=['Wins', 'Losses'])
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Trade PnL (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Metrics Table
    ax = axes[1, 1]
    ax.axis('off')

    metrics = backtest_results['metrics']
    pattern = backtest_results['pattern']

    metrics_text = f"""
PATTERN: {pattern.to_readable()}

PERFORMANCE METRICS:
{'='*35}
Total Trades: {metrics['total_trades']}
Final Equity: ${metrics['final_equity']:.2f}
Total Return: {metrics['total_return_pct']*100:+.1f}%

Win Rate: {metrics['win_rate']*100:.1f}%
Avg Win: ${metrics['avg_win']:.2f}
Avg Loss: ${metrics['avg_loss']:.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Max Drawdown: {metrics['max_drawdown_pct']*100:.1f}%

MONTE CARLO VALIDATION:
{'='*35}
Actual Percentile: {mc_results['actual_percentile']:.1f}th
Prob(Profitable): {mc_results['prob_profitable']*100:.1f}%
Expected Return: ${mc_results['mean_final']:.2f} ± ${mc_results['std_final']:.2f}

Best Case: ${mc_results['best_case']:.2f}
Worst Case: ${mc_results['worst_case']:.2f}
    """

    ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()
