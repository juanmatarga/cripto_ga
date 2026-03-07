"""
Final Analysis — Hansen SPA, White RC, Ensemble, Paper Tables & Figures.

Run after evolve → validate → ots pipeline.
Produces publication-ready output.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from grammar.mapper import decode
from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
from backtest.metrics import (
    calculate_returns, calculate_sortino_ratio, calculate_all_metrics,
    max_drawdown, cagr
)
from robustness.hansen_spa import hansen_spa_test
from robustness.white_rc import whites_reality_check

logging.basicConfig(level=logging.WARNING, format='%(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ots_positive_strategies(result_dirs):
    """Load all validated + OTS-positive strategies from multiple experiment dirs."""
    all_strategies = []

    for dir_path in result_dirs:
        d = Path(dir_path)
        with open(d / 'validation.json') as f:
            val = json.load(f)
        with open(d / 'ots_results.json') as f:
            ots = json.load(f)
        with open(d / 'top_strategies.json') as f:
            strats = json.load(f)

        ots_map = {r['strategy_index']: r for r in ots}

        for v in val:
            idx = v['strategy_index']
            if not v.get('passes_all'):
                continue
            o = ots_map.get(idx)
            if not o:
                continue
            m = o.get('metrics', {})
            if m.get('cagr', 0) <= 0 or o['n_trades'] < 5:
                continue

            s = strats[idx]
            strategy = decode(s['genome'])
            if strategy is None:
                continue

            all_strategies.append({
                'strategy': strategy,
                'genome': s['genome'],
                'expression': s['expression_raw'],
                'direction': s['direction'],
                'validation': v,
                'ots': o,
                'ots_metrics': m,
                'seed': d.name.split('seed')[1].split('_')[0],
            })

    # Sort by OTS trades descending
    all_strategies.sort(key=lambda x: x['ots']['n_trades'], reverse=True)
    return all_strategies


def load_data_splits(config_path='config_v2.yaml'):
    """Load evolution and OTS data."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from data.loader import load_data
    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ots_start = pd.Timestamp(config.get('data', {}).get('ots_start', '2025-06-01'))
    evo_data = df[df.index < ots_start]
    ots_data = df[df.index >= ots_start]

    return config, evo_data, ots_data


# ============================================================================
# STEP 1: HANSEN SPA + WHITE RC
# ============================================================================

def run_statistical_tests(strategies, ots_data, config):
    """Run Hansen SPA and White RC on OTS-positive strategies."""
    logger.info("=" * 70)
    logger.info("STEP 1: Hansen SPA + White's Reality Check")
    logger.info("=" * 70)

    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    # Benchmark: buy & hold returns on OTS
    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
    bh_returns = calculate_returns(bh_equity).dropna()

    # Get strategy equity curves and returns
    strategy_returns_list = []
    strategy_labels = []

    for i, s_info in enumerate(strategies):
        strategy = s_info['strategy']
        equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
        returns = calculate_returns(equity).dropna()
        strategy_returns_list.append(returns)
        strategy_labels.append(f"S{i}_{s_info['direction']}_{s_info['ots']['n_trades']}tr")

    # Hansen SPA: test best strategy individually
    best_idx = 0  # highest trade count
    hansen = hansen_spa_test(
        strategy_returns_list[best_idx], bh_returns,
        n_bootstrap=5000, alpha=0.05, seed=42
    )

    logger.info(f"\nHansen SPA (best strategy - {strategy_labels[best_idx]}):")
    logger.info(f"  t-stat={hansen['test_statistic']:.4f} p={hansen['p_value']:.4f} "
                f"reject_H0={hansen['reject_null']}")

    # White RC: test ALL strategies jointly (corrects for multiple testing)
    white = whites_reality_check(
        strategy_returns_list, bh_returns,
        n_bootstrap=5000, alpha=0.05, seed=42
    )

    logger.info(f"\nWhite's Reality Check ({len(strategy_returns_list)} strategies):")
    logger.info(f"  max_perf={white['max_performance']:.6f} p={white['p_value']:.4f} "
                f"reject_H0={white['reject_null']}")

    # Also test each individually
    individual_results = []
    for i, (ret, label) in enumerate(zip(strategy_returns_list, strategy_labels)):
        h = hansen_spa_test(ret, bh_returns, n_bootstrap=2000, alpha=0.05, seed=42+i)
        individual_results.append({
            'label': label,
            'expression': strategies[i]['expression'][:60],
            'ots_trades': strategies[i]['ots']['n_trades'],
            'ots_cagr': strategies[i]['ots_metrics'].get('cagr', 0),
            'hansen_p': h['p_value'],
            'hansen_reject': h['reject_null'],
            'mean_outperf': h['mean_outperformance'],
        })

    return hansen, white, individual_results, strategy_returns_list, bh_returns


# ============================================================================
# STEP 2: ENSEMBLE
# ============================================================================

def build_ensemble(strategies, ots_data, config):
    """Build ensemble combining top LONG + SHORT strategies."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Ensemble Strategy")
    logger.info("=" * 70)

    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    # Separate LONG and SHORT
    longs = [s for s in strategies if s['direction'] == 'LONG']
    shorts = [s for s in strategies if s['direction'] == 'SHORT']

    logger.info(f"  Available: {len(longs)} LONG, {len(shorts)} SHORT strategies")

    # Pick top strategies by trade count (statistical robustness)
    top_longs = longs[:min(5, len(longs))]
    top_shorts = shorts[:min(3, len(shorts))]
    ensemble_members = top_longs + top_shorts

    logger.info(f"  Ensemble: {len(top_longs)} LONG + {len(top_shorts)} SHORT = "
                f"{len(ensemble_members)} members")

    # Equal-weight ensemble equity curve
    equity_curves = []
    for s_info in ensemble_members:
        strategy = s_info['strategy']
        equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
        # Normalize to start at 100
        equity_norm = equity / equity.iloc[0] * 100
        equity_curves.append(equity_norm)

    if not equity_curves:
        logger.warning("No ensemble members!")
        return None

    # Equal-weight average
    ensemble_df = pd.DataFrame({f's{i}': eq for i, eq in enumerate(equity_curves)})
    ensemble_equity = ensemble_df.mean(axis=1)

    # Benchmark
    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100

    # Metrics
    ens_returns = calculate_returns(ensemble_equity).dropna()
    ens_sortino = calculate_sortino_ratio(ens_returns, BARS_PER_YEAR_15M)
    ens_cagr = cagr(ensemble_equity, BARS_PER_YEAR_15M)
    ens_dd = max_drawdown(ensemble_equity)

    bh_cagr = cagr(bh_equity, BARS_PER_YEAR_15M)
    bh_dd = max_drawdown(bh_equity)

    logger.info(f"\n  Ensemble OTS Performance:")
    logger.info(f"    Sortino: {ens_sortino:.3f}")
    logger.info(f"    CAGR:    {ens_cagr*100:.2f}% (B&H: {bh_cagr*100:.2f}%)")
    logger.info(f"    MaxDD:   {ens_dd:.2%} (B&H: {bh_dd:.2%})")
    logger.info(f"    Alpha:   {(ens_cagr - bh_cagr)*100:.2f}% annualized")

    return {
        'ensemble_equity': ensemble_equity,
        'bh_equity': bh_equity,
        'member_equities': equity_curves,
        'members': ensemble_members,
        'sortino': ens_sortino,
        'cagr': ens_cagr,
        'max_dd': ens_dd,
        'bh_cagr': bh_cagr,
        'bh_dd': bh_dd,
    }


# ============================================================================
# STEP 3: PAPER TABLES & FIGURES
# ============================================================================

def generate_paper_outputs(strategies, hansen, white, individual_results,
                           ensemble, strategy_returns, bh_returns,
                           ots_data, output_dir):
    """Generate publication-ready tables and figures."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Paper Tables & Figures")
    logger.info("=" * 70)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- TABLE 1: Top Strategies Summary ----
    rows = []
    for s in strategies:
        v = s['validation']
        o = s['ots']
        m = s['ots_metrics']
        rows.append({
            'Direction': s['direction'],
            'Expression': s['expression'][:55],
            'OTS Trades': o['n_trades'],
            'OTS CAGR (%)': f"{m.get('cagr', 0)*100:.1f}",
            'OTS MaxDD (%)': f"{abs(m.get('max_dd', 0))*100:.1f}",
            'OTS Sortino': f"{m.get('sortino', 0):.3f}" if m.get('sortino', 0) < 100 else ">10",
            'PBO': f"{v['pbo']:.3f}",
            'Perm p': f"{v['perm_p_value']:.4f}",
            'CPCV %+': f"{v['cpcv_pct_positive']:.0%}",
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(out / 'table1_top_strategies.csv', index=False)
    logger.info(f"  Table 1: {len(rows)} strategies -> {out / 'table1_top_strategies.csv'}")

    # ---- TABLE 2: Statistical Tests ----
    test_rows = []
    for r in individual_results:
        test_rows.append({
            'Strategy': r['label'],
            'OTS Trades': r['ots_trades'],
            'OTS CAGR (%)': f"{r['ots_cagr']*100:.1f}",
            'Hansen p': f"{r['hansen_p']:.4f}",
            'Reject H0': 'Yes' if r['hansen_reject'] else 'No',
            'Mean Outperf': f"{r['mean_outperf']*100:.4f}%",
        })
    table2 = pd.DataFrame(test_rows)
    table2.to_csv(out / 'table2_hansen_spa.csv', index=False)

    # White RC summary
    white_summary = {
        'Test': "White's Reality Check",
        'N Strategies': white['n_strategies_tested'],
        'p-value': f"{white['p_value']:.4f}",
        'Reject H0': 'Yes' if white['reject_null'] else 'No',
        'Max Mean Outperf': f"{white['max_performance']*100:.4f}%",
    }
    with open(out / 'table2b_white_rc.json', 'w') as f:
        json.dump(white_summary, f, indent=2)
    logger.info(f"  Table 2: Hansen SPA + White RC -> {out}")

    # ---- TABLE 3: Ensemble vs B&H ----
    if ensemble:
        ens_table = {
            'Metric': ['CAGR (%)', 'Max Drawdown (%)', 'Sortino', 'N Members'],
            'Ensemble': [
                f"{ensemble['cagr']*100:.2f}",
                f"{abs(ensemble['max_dd'])*100:.2f}",
                f"{ensemble['sortino']:.3f}",
                str(len(ensemble['members'])),
            ],
            'Buy & Hold': [
                f"{ensemble['bh_cagr']*100:.2f}",
                f"{abs(ensemble['bh_dd'])*100:.2f}",
                '-',
                '-',
            ],
        }
        pd.DataFrame(ens_table).to_csv(out / 'table3_ensemble.csv', index=False)
        logger.info(f"  Table 3: Ensemble -> {out / 'table3_ensemble.csv'}")

    # ---- FIGURE 1: Ensemble Equity vs B&H ----
    if ensemble:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Individual members in light gray
        for i, eq in enumerate(ensemble['member_equities']):
            member = ensemble['members'][i]
            color = '#2E86AB' if member['direction'] == 'LONG' else '#C73E1D'
            ax.plot(eq.index, eq.values, alpha=0.15, color=color, linewidth=0.8)

        # Ensemble
        ax.plot(ensemble['ensemble_equity'].index, ensemble['ensemble_equity'].values,
                label='Ensemble', linewidth=2.5, color='#06A77D')
        # B&H
        ax.plot(ensemble['bh_equity'].index, ensemble['bh_equity'].values,
                label='Buy & Hold', linewidth=2, color='#A23B72', linestyle='--')

        ax.axhline(y=100, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (start = 100)')
        ax.set_title('OTS Holdout (Jun-Nov 2025): Ensemble vs Buy & Hold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / 'fig1_ensemble_equity.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Figure 1: Equity curves -> {out / 'fig1_ensemble_equity.png'}")

    # ---- FIGURE 2: Hansen p-values bar chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r['label'] for r in individual_results]
    pvals = [r['hansen_p'] for r in individual_results]
    colors = ['#06A77D' if r['hansen_reject'] else '#C73E1D' for r in individual_results]

    y_pos = range(len(labels))
    ax.barh(y_pos, pvals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('p-value')
    ax.set_title(f"Hansen SPA Test — Individual Strategies on OTS\n"
                 f"White's RC joint p = {white['p_value']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    for i, pval in enumerate(pvals):
        ax.text(pval + 0.002, i, f'{pval:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(out / 'fig2_hansen_pvalues.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Figure 2: Hansen p-values -> {out / 'fig2_hansen_pvalues.png'}")

    # ---- FIGURE 3: Evolution fitness progression (from best seed) ----
    best_evo_log = None
    for dir_name in ['experiment_seed42_20260307_015843',
                     'experiment_seed777_20260307_015837',
                     'experiment_seed123_20260307_015430']:
        evo_path = Path('results') / dir_name / 'evolution_log.json'
        if evo_path.exists():
            with open(evo_path) as f:
                best_evo_log = json.load(f)
            break

    if best_evo_log:
        fig, ax = plt.subplots(figsize=(12, 5))
        gens = range(1, len(best_evo_log) + 1)

        for island_key in ['island_0', 'island_1', 'island_2']:
            best_fits = []
            for gen_data in best_evo_log:
                if island_key in gen_data:
                    best_fits.append(gen_data[island_key]['best_fitness'])
                else:
                    best_fits.append(None)

            sel_type = best_evo_log[0].get(island_key, {}).get('selection', island_key)
            ax.plot(gens, best_fits, label=f'{island_key} ({sel_type})',
                    linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Island Model Evolution — Fitness Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / 'fig3_evolution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Figure 3: Evolution -> {out / 'fig3_evolution.png'}")

    # ---- FIGURE 4: OTS strategy CAGR vs number of trades ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in strategies:
        o = s['ots']
        m = s['ots_metrics']
        color = '#2E86AB' if s['direction'] == 'LONG' else '#C73E1D'
        marker = '^' if s['direction'] == 'LONG' else 'v'
        ax.scatter(o['n_trades'], m.get('cagr', 0) * 100,
                   c=color, marker=marker, s=80, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='gray', linewidth=1)
    ax.set_xlabel('Number of OTS Trades')
    ax.set_ylabel('OTS CAGR (%)')
    ax.set_title('Strategy Performance on OTS Holdout')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2E86AB',
               markersize=10, label='LONG'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#C73E1D',
               markersize=10, label='SHORT'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig4_ots_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Figure 4: OTS scatter -> {out / 'fig4_ots_scatter.png'}")

    # ---- SUMMARY JSON ----
    summary = {
        'n_strategies_ots_positive': len(strategies),
        'n_long': sum(1 for s in strategies if s['direction'] == 'LONG'),
        'n_short': sum(1 for s in strategies if s['direction'] == 'SHORT'),
        'hansen_best_p': hansen['p_value'],
        'hansen_best_reject': hansen['reject_null'],
        'white_rc_p': white['p_value'],
        'white_rc_reject': white['reject_null'],
        'ensemble_cagr': ensemble['cagr'] if ensemble else None,
        'ensemble_sortino': ensemble['sortino'] if ensemble else None,
        'ensemble_max_dd': ensemble['max_dd'] if ensemble else None,
        'bh_cagr': ensemble['bh_cagr'] if ensemble else None,
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Summary -> {out / 'summary.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    result_dirs = [
        'results/experiment_seed42_20260307_015843',
        'results/experiment_seed123_20260307_015430',
        'results/experiment_seed777_20260307_015837',
    ]

    # Verify dirs exist
    for d in result_dirs:
        if not Path(d).exists():
            logger.error(f"Directory not found: {d}")
            return

    logger.info("Loading data and strategies...")
    config, evo_data, ots_data = load_data_splits()
    strategies = load_ots_positive_strategies(result_dirs)
    logger.info(f"Loaded {len(strategies)} OTS-positive validated strategies")

    # Step 1: Statistical tests
    hansen, white, individual, strat_returns, bh_returns = \
        run_statistical_tests(strategies, ots_data, config)

    # Step 2: Ensemble
    ensemble = build_ensemble(strategies, ots_data, config)

    # Step 3: Paper outputs
    output_dir = 'reports/paper_v2'
    generate_paper_outputs(
        strategies, hansen, white, individual, ensemble,
        strat_returns, bh_returns, ots_data, output_dir
    )

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Strategies tested:  {len(strategies)}")
    logger.info(f"  Hansen SPA (best):  p = {hansen['p_value']:.4f} "
                f"({'REJECT H0' if hansen['reject_null'] else 'FAIL'})")
    logger.info(f"  White RC (joint):   p = {white['p_value']:.4f} "
                f"({'REJECT H0' if white['reject_null'] else 'FAIL'})")
    if ensemble:
        logger.info(f"  Ensemble CAGR:      {ensemble['cagr']*100:.2f}%")
        logger.info(f"  Ensemble MaxDD:     {ensemble['max_dd']:.2%}")
        logger.info(f"  B&H CAGR:           {ensemble['bh_cagr']*100:.2f}%")
        logger.info(f"  Alpha (ann.):       {(ensemble['cagr']-ensemble['bh_cagr'])*100:.2f}%")
    n_hansen_pass = sum(1 for r in individual if r['hansen_reject'])
    logger.info(f"  Hansen pass indiv:  {n_hansen_pass}/{len(individual)}")
    logger.info(f"\nAll outputs saved to: {output_dir}/")


if __name__ == '__main__':
    main()
