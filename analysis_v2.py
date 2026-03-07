"""
CriptoGA v2 — Final Analysis: Ensemble Construction + Paper Outputs.

Reads OTS results from all 3 seeds, builds optimized ensemble,
generates tables and figures for paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Experiment directories
EXPERIMENTS = {
    'seed42': 'results/experiment_seed42_20260307_112947',
    'seed123': 'results/experiment_seed123_20260307_113512',
    'seed777': 'results/experiment_seed777_20260307_112637',
}

OUTPUT_DIR = Path('reports/paper_v2')


def load_all_ots_strategies() -> List[Dict]:
    """Load all OTS-tested strategies from all seeds."""
    all_strats = []
    for seed_name, exp_dir in EXPERIMENTS.items():
        ots_path = Path(exp_dir) / 'ots_results.json'
        val_path = Path(exp_dir) / 'validation.json'
        if not ots_path.exists():
            continue

        with open(ots_path) as f:
            ots = json.load(f)
        with open(val_path) as f:
            val = json.load(f)

        # Build validation lookup
        val_map = {v['strategy_index']: v for v in val}

        for r in ots:
            m = r.get('metrics', {})
            idx = r.get('strategy_index', -1)
            v = val_map.get(idx, {})

            all_strats.append({
                'seed': seed_name,
                'index': idx,
                'expression': r.get('expression', ''),
                'direction': r.get('direction', ''),
                'n_trades': r.get('n_trades', 0),
                'cagr': m.get('cagr', 0),
                'sortino': m.get('sortino', 0),
                'max_dd': m.get('max_dd', 0),
                'profit_factor': m.get('profit_factor', 0),
                'win_rate': m.get('win_rate', 0),
                'upi': m.get('upi', 0),
                'sharpe': m.get('sharpe', 0),
                # Validation metrics
                'pbo': v.get('pbo', 1.0),
                'perm_p': v.get('perm_p_value', 1.0),
                'ttest_p': v.get('cpcv_ttest_pval', 1.0),
                'cpcv_mean_sortino': v.get('cpcv_mean_sortino', 0),
                'cpcv_pct_positive': v.get('cpcv_pct_positive', 0),
            })

    return all_strats


def build_ensemble_equity(strategies: List[Dict], ots_data: pd.DataFrame,
                          config: dict) -> Tuple[pd.Series, List[Dict]]:
    """
    Build ensemble equity curve from selected strategies.
    Equal-weight allocation, rebalanced per trade.
    """
    from grammar.mapper import decode
    from evolution.fitness import _run_single_window

    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    n = len(strategies)
    weight = 1.0 / n

    # Get individual equity curves
    curves = []
    for s in strategies:
        # Re-decode from top_strategies
        exp_dir = Path(EXPERIMENTS[s['seed']])
        with open(exp_dir / 'top_strategies.json') as f:
            all_strats = json.load(f)

        sd = all_strats[s['index']]
        strategy = decode(sd['genome'])
        if strategy is None:
            continue

        equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
        # Normalize to returns
        returns = equity.pct_change().fillna(0)
        curves.append(returns)

    if not curves:
        return pd.Series(dtype=float), []

    # Equal-weight ensemble returns
    returns_df = pd.DataFrame(curves).T
    ensemble_returns = returns_df.mean(axis=1)

    # Build equity from returns
    equity = (1 + ensemble_returns).cumprod() * 100

    return equity, curves


def analyze_results():
    """Full analysis pipeline."""
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open('config_v2.yaml') as f:
        config = yaml.safe_load(f)

    all_strats = load_all_ots_strategies()
    print(f"\nTotal OTS strategies: {len(all_strats)}")

    # Filter: positive CAGR and enough trades
    positive = [s for s in all_strats if s['cagr'] > 0 and s['n_trades'] >= 5]
    print(f"Positive CAGR (≥5 trades): {len(positive)}")

    # Sort by CAGR
    positive.sort(key=lambda x: x['cagr'], reverse=True)

    # ================================================================
    # TABLE 1: Top OTS strategies
    # ================================================================
    rows = []
    for s in positive:
        rows.append({
            'Seed': s['seed'],
            'Direction': s['direction'],
            'CAGR': f"{s['cagr']:.2%}",
            'Sortino': f"{s['sortino']:.3f}",
            'MaxDD': f"{s['max_dd']:.2%}",
            'PF': f"{s['profit_factor']:.2f}",
            'WinRate': f"{s['win_rate']:.0%}",
            'Trades': s['n_trades'],
            'PBO': f"{s['pbo']:.3f}",
            'PermP': f"{s['perm_p']:.3f}",
            'Expression': s['expression'][:80],
        })

    df_table1 = pd.DataFrame(rows)
    df_table1.to_csv(OUTPUT_DIR / 'table1_top_strategies.csv', index=False)
    print(f"\nTable 1: {len(rows)} OTS-positive strategies")
    print(df_table1[['Direction', 'CAGR', 'MaxDD', 'PF', 'Trades']].to_string())

    # ================================================================
    # ENSEMBLE: Select top uncorrelated strategies
    # ================================================================
    # Pick best by direction to diversify
    longs = [s for s in positive if s['direction'] == 'LONG']
    shorts = [s for s in positive if s['direction'] == 'SHORT']

    # Take top 5 LONG + top 5 SHORT (or all if fewer)
    ensemble_strats = longs[:5] + shorts[:5]
    print(f"\nEnsemble: {len(longs[:5])} LONG + {len(shorts[:5])} SHORT")

    # Load OTS data for ensemble equity
    from main_v2 import load_ots_data
    ots_data = load_ots_data(config)

    ensemble_equity, individual_curves = build_ensemble_equity(
        ensemble_strats, ots_data, config
    )

    if len(ensemble_equity) > 0:
        from backtest.metrics import (
            calculate_returns, calculate_sortino_ratio, cagr, max_drawdown
        )

        ens_returns = calculate_returns(ensemble_equity).dropna()
        ens_cagr = cagr(ensemble_equity, 35040)
        ens_dd = max_drawdown(ensemble_equity)
        ens_sortino = calculate_sortino_ratio(ens_returns, 35040)

        # B&H benchmark
        bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
        bh_cagr = cagr(bh_equity, 35040)
        bh_dd = max_drawdown(bh_equity)
        bh_returns = calculate_returns(bh_equity).dropna()
        bh_sortino = calculate_sortino_ratio(bh_returns, 35040)

        print(f"\n{'='*60}")
        print(f"ENSEMBLE vs BUY & HOLD (OTS: {ots_data.index.min().date()} to {ots_data.index.max().date()})")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Ensemble':>12} {'B&H':>12}")
        print(f"{'-'*44}")
        print(f"{'CAGR':<20} {ens_cagr:>12.2%} {bh_cagr:>12.2%}")
        print(f"{'Max Drawdown':<20} {ens_dd:>12.2%} {bh_dd:>12.2%}")
        print(f"{'Sortino':<20} {ens_sortino:>12.3f} {bh_sortino:>12.3f}")
        print(f"{'Final Equity':<20} {ensemble_equity.iloc[-1]:>12.2f} {bh_equity.iloc[-1]:>12.2f}")

        # ================================================================
        # TABLE 3: Ensemble vs B&H
        # ================================================================
        ens_table = {
            'Metric': ['CAGR', 'Max Drawdown', 'Sortino', 'Final Equity',
                       'Strategies', 'Total Trades'],
            'Ensemble': [f"{ens_cagr:.2%}", f"{ens_dd:.2%}", f"{ens_sortino:.3f}",
                         f"{ensemble_equity.iloc[-1]:.2f}",
                         len(ensemble_strats),
                         sum(s['n_trades'] for s in ensemble_strats)],
            'Buy&Hold': [f"{bh_cagr:.2%}", f"{bh_dd:.2%}", f"{bh_sortino:.3f}",
                         f"{bh_equity.iloc[-1]:.2f}", 1, 1],
        }
        pd.DataFrame(ens_table).to_csv(OUTPUT_DIR / 'table3_ensemble.csv', index=False)

        # ================================================================
        # FIGURE 1: Ensemble equity vs B&H
        # ================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ensemble_equity.index, ensemble_equity.values, 'b-',
                linewidth=2, label=f'Ensemble (CAGR={ens_cagr:.1%})')
        ax.plot(bh_equity.index, bh_equity.values, 'k--', alpha=0.7,
                label=f'Buy & Hold (CAGR={bh_cagr:.1%})')
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (base=100)')
        ax.set_title('OTS Performance: Ensemble vs Buy & Hold (Jun-Nov 2025)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'fig1_ensemble_equity.png', dpi=150)
        plt.close()

        # ================================================================
        # FIGURE 2: Individual strategy returns scatter
        # ================================================================
        fig, ax = plt.subplots(figsize=(10, 7))
        for s in all_strats:
            if s['n_trades'] >= 5:
                color = 'green' if s['cagr'] > 0 else 'red'
                marker = '^' if s['direction'] == 'LONG' else 'v'
                ax.scatter(abs(s['max_dd']) * 100, s['cagr'] * 100,
                           c=color, marker=marker, s=50, alpha=0.7)

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('CAGR (%)')
        ax.set_title('OTS: CAGR vs Max Drawdown (all validated strategies)')
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='LONG +'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='LONG -'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='green', markersize=10, label='SHORT +'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='SHORT -'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'fig2_ots_scatter.png', dpi=150)
        plt.close()

        # ================================================================
        # SUMMARY JSON
        # ================================================================
        summary = {
            'experiment_date': '2026-03-07',
            'grammar_version': 'v4 (scale-invariant + trailing stops + ADX/MFI)',
            'data_period': '2022-01-01 to 2025-11-21',
            'ots_period': '2025-06-01 to 2025-11-21',
            'total_strategies_evolved': sum(
                len(json.load(open(Path(d) / 'top_strategies.json')))
                for d in EXPERIMENTS.values()
            ),
            'passed_validation': len([s for s in all_strats]),
            'ots_positive_cagr': len(positive),
            'ensemble': {
                'n_strategies': len(ensemble_strats),
                'n_long': len([s for s in ensemble_strats if s['direction'] == 'LONG']),
                'n_short': len([s for s in ensemble_strats if s['direction'] == 'SHORT']),
                'cagr': float(ens_cagr),
                'max_dd': float(ens_dd),
                'sortino': float(ens_sortino),
            },
            'buy_and_hold': {
                'cagr': float(bh_cagr),
                'max_dd': float(bh_dd),
                'sortino': float(bh_sortino),
            },
            'top_strategies': [
                {
                    'seed': s['seed'],
                    'direction': s['direction'],
                    'cagr': s['cagr'],
                    'max_dd': s['max_dd'],
                    'sortino': s['sortino'],
                    'profit_factor': s['profit_factor'],
                    'n_trades': s['n_trades'],
                    'pbo': s['pbo'],
                    'expression': s['expression'][:100],
                }
                for s in positive[:10]
            ],
        }

        with open(OUTPUT_DIR / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nOutputs saved to {OUTPUT_DIR}/")
        print(f"  table1_top_strategies.csv")
        print(f"  table3_ensemble.csv")
        print(f"  fig1_ensemble_equity.png")
        print(f"  fig2_ots_scatter.png")
        print(f"  summary.json")


if __name__ == '__main__':
    analyze_results()
