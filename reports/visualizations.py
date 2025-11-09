"""
Visualization Suite - Publication-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Configuración estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def plot_equity_curves(portfolio_equity: pd.Series,
                      benchmark_equity: pd.Series,
                      output_path: Path):
    """
    Plot equity curves: portfolio vs benchmark.

    Args:
        portfolio_equity: Portfolio equity series
        benchmark_equity: Benchmark equity series
        output_path: Output file path
    """
    logger.info("Plotting equity curves...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot curves
    ax.plot(portfolio_equity.index, portfolio_equity.values,
           label='Portfolio', linewidth=2, color='#2E86AB')
    ax.plot(benchmark_equity.index, benchmark_equity.values,
           label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)

    # Styling
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.set_title('Portfolio Performance vs Buy & Hold Benchmark', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"[OK] Saved equity curve plot to {output_path}")

def plot_drawdown_analysis(equity: pd.Series, output_path: Path):
    """
    Plot drawdown analysis.

    Args:
        equity: Equity series
        output_path: Output file path
    """
    logger.info("Plotting drawdown analysis...")

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Equity
    ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86AB')
    ax1.fill_between(equity.index, equity.values, running_max.values,
                     alpha=0.3, color='#F18F01')
    ax1.set_ylabel('Equity', fontsize=12)
    ax1.set_title('Equity Curve with Drawdown Shading', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100,
                     alpha=0.5, color='#C73E1D')
    ax2.plot(drawdown.index, drawdown.values * 100, linewidth=1.5, color='#8B0000')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # Stats
    max_dd = drawdown.min() * 100
    ax2.text(0.02, 0.95, f'Max Drawdown: {max_dd:.2f}%',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"[OK] Saved drawdown plot to {output_path}")

def plot_evolution_fitness(evolution_tracker_data: Dict, output_path: Path):
    """
    Plot GA evolution (fitness progression).

    Args:
        evolution_tracker_data: Data from EvolutionTracker
        output_path: Output file path
    """
    logger.info("Plotting GA evolution...")

    # Extract data
    best_history = evolution_tracker_data.get('best_fitness_history', [])
    mean_history = evolution_tracker_data.get('mean_fitness_history', [])
    best_long = evolution_tracker_data.get('best_long_history', [])
    best_short = evolution_tracker_data.get('best_short_history', [])

    if not best_history:
        logger.warning("No evolution history available")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Extract generations and fitness
    generations_best = [h['generation'] for h in best_history]
    fitness_best = [h['fitness'] for h in best_history]

    generations_mean = [h['generation'] for h in mean_history]
    fitness_mean = [h['fitness'] for h in mean_history]

    # Plot
    ax.plot(generations_best, fitness_best, label='Best Fitness',
           linewidth=2.5, color='#2E86AB', marker='o', markersize=4)
    ax.plot(generations_mean, fitness_mean, label='Mean Fitness',
           linewidth=2, color='#A23B72', alpha=0.7, linestyle='--')

    # LONG/SHORT if available
    if best_long:
        gen_long = [h['generation'] for h in best_long]
        fit_long = [h['fitness'] for h in best_long]
        ax.plot(gen_long, fit_long, label='Best LONG',
               linewidth=1.5, color='#06A77D', alpha=0.6, linestyle=':')

    if best_short:
        gen_short = [h['generation'] for h in best_short]
        fit_short = [h['fitness'] for h in best_short]
        ax.plot(gen_short, fit_short, label='Best SHORT',
               linewidth=1.5, color='#F18F01', alpha=0.6, linestyle=':')

    # Styling
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Genetic Algorithm Evolution - Fitness Progression',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add improvement annotation
    if len(fitness_best) > 1:
        improvement = fitness_best[-1] - fitness_best[0]
        ax.text(0.02, 0.98, f'Total Improvement: {improvement:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"[OK] Saved evolution plot to {output_path}")

def plot_statistical_tests(hansen_results: Dict, white_results: Dict,
                          bootstrap_results: Dict, output_path: Path):
    """
    Plot statistical test results (p-values and confidence intervals).

    Args:
        hansen_results: Hansen SPA results
        white_results: White RC results
        bootstrap_results: Bootstrap results
        output_path: Output file path
    """
    logger.info("Plotting statistical test results...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========================================================================
    # LEFT: P-values
    # ========================================================================
    tests = []
    pvalues = []
    colors = []

    if hansen_results:
        tests.append('Hansen SPA')
        pvalues.append(hansen_results['p_value'])
        colors.append('#06A77D' if hansen_results['reject_null'] else '#C73E1D')

    if white_results:
        tests.append("White's RC")
        pvalues.append(white_results['p_value'])
        colors.append('#06A77D' if white_results['reject_null'] else '#C73E1D')

    if tests:
        bars = ax1.barh(tests, pvalues, color=colors, alpha=0.7)
        ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.set_xlabel('P-value', fontsize=12)
        ax1.set_title('Statistical Test P-values', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, max(pvalues) * 1.2 if pvalues else 0.1])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (test, pval) in enumerate(zip(tests, pvalues)):
            ax1.text(pval + 0.005, i, f'{pval:.4f}',
                    va='center', fontsize=10, fontweight='bold')

    # ========================================================================
    # RIGHT: Bootstrap Confidence Intervals
    # ========================================================================
    if bootstrap_results:
        metrics = []
        means = []
        ci_lower = []
        ci_upper = []

        for metric_name in ['upi', 'sharpe', 'cagr']:
            if metric_name in bootstrap_results:
                stats = bootstrap_results[metric_name]
                metrics.append(metric_name.upper())
                means.append(stats['mean'])
                ci_lower.append(stats['ci_lower'])
                ci_upper.append(stats['ci_upper'])

        if metrics:
            y_pos = np.arange(len(metrics))
            errors = [[m - l for m, l in zip(means, ci_lower)],
                     [u - m for m, u in zip(means, ci_upper)]]

            ax2.errorbar(means, y_pos, xerr=errors, fmt='o', markersize=8,
                        capsize=5, capthick=2, color='#2E86AB', ecolor='#A23B72',
                        linewidth=2)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.set_xlabel('Value', fontsize=12)
            ax2.set_title('Bootstrap Confidence Intervals (95%)',
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')

            # Add zero line for reference
            ax2.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"[OK] Saved statistical tests plot to {output_path}")

def plot_returns_distribution(returns: pd.Series, output_path: Path):
    """
    Plot returns distribution with statistics.

    Args:
        returns: Returns series
        output_path: Output file path
    """
    logger.info("Plotting returns distribution...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(returns * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    ax1.axvline(x=returns.mean() * 100, color='green', linestyle='--',
               linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax1.set_xlabel('Returns (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Stats box
    stats_text = f'Mean: {returns.mean()*100:.4f}%\n'
    stats_text += f'Std: {returns.std()*100:.4f}%\n'
    stats_text += f'Skew: {returns.skew():.4f}\n'
    stats_text += f'Kurt: {returns.kurtosis():.4f}'

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"[OK] Saved returns distribution plot to {output_path}")
