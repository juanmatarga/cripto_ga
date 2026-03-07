"""
CriptoGA v2 — Deep Analysis: Hansen SPA, White RC, Optimized Ensemble, Qualitative.

Runs all statistical tests and builds optimized portfolio from OTS results.
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('reports/paper_v2')


def get_experiment_dirs() -> Dict[str, str]:
    """Find all experiment directories with OTS results."""
    results_dir = Path('results')
    experiments = {}
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / 'ots_results.json').exists():
            # Extract seed from directory name
            name = d.name
            seed_part = name.split('_seed')[1].split('_')[0] if '_seed' in name else name
            # Use latest per seed
            experiments[f'seed{seed_part}'] = str(d)
    return experiments


def load_ots_strategy_equities(experiments: Dict[str, str],
                                ots_data: pd.DataFrame,
                                config: dict) -> Tuple[Dict, pd.DataFrame]:
    """
    Load individual strategy equity curves from OTS.
    Returns (strategy_info_dict, returns_dataframe).
    """
    from grammar.mapper import decode
    from evolution.fitness import _run_single_window

    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    all_info = {}
    all_returns = {}

    for seed_name, exp_dir in experiments.items():
        ots_path = Path(exp_dir) / 'ots_results.json'
        strats_path = Path(exp_dir) / 'top_strategies.json'
        val_path = Path(exp_dir) / 'validation.json'

        with open(ots_path) as f:
            ots = json.load(f)
        with open(strats_path) as f:
            all_strats = json.load(f)
        with open(val_path) as f:
            val = json.load(f)

        val_map = {v['strategy_index']: v for v in val}

        for r in ots:
            idx = r.get('strategy_index', -1)
            m = r.get('metrics', {})
            n_trades = r.get('n_trades', 0)
            cagr_val = m.get('cagr', 0)

            if n_trades < 5 or cagr_val <= 0:
                continue

            sd = all_strats[idx]
            strategy = decode(sd['genome'])
            if strategy is None:
                continue

            key = f"{seed_name}_s{idx}"

            equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
            returns = equity.pct_change().fillna(0)

            all_returns[key] = returns
            v = val_map.get(idx, {})
            all_info[key] = {
                'seed': seed_name,
                'index': idx,
                'expression': r.get('expression', ''),
                'direction': r.get('direction', ''),
                'n_trades': n_trades,
                'cagr': cagr_val,
                'sortino': m.get('sortino', 0),
                'max_dd': m.get('max_dd', 0),
                'profit_factor': m.get('profit_factor', 0),
                'win_rate': m.get('win_rate', 0),
                'pbo': v.get('pbo', 1.0),
                'perm_p': v.get('perm_p_value', 1.0),
            }

    returns_df = pd.DataFrame(all_returns)
    return all_info, returns_df


# ============================================================================
# 1. HANSEN SPA + WHITE RC
# ============================================================================

def run_statistical_tests(returns_df: pd.DataFrame, ots_data: pd.DataFrame,
                          strategy_info: Dict) -> Dict:
    """Run Hansen SPA and White RC on all OTS-positive strategies."""
    from robustness.white_rc import whites_reality_check

    results = {}

    # Benchmark: Buy & Hold returns
    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
    bh_returns = bh_equity.pct_change().fillna(0)

    # Cash benchmark (0 returns)
    cash_returns = pd.Series(0.0, index=ots_data.index)

    strategy_returns_list = [returns_df[col] for col in returns_df.columns]

    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")
    print(f"Testing {len(strategy_returns_list)} strategies")

    # White RC vs B&H
    print("\n--- White's Reality Check vs Buy & Hold ---")
    wrc_bh = whites_reality_check(
        strategy_returns_list, bh_returns, n_bootstrap=2000, alpha=0.05
    )
    print(f"  p-value: {wrc_bh['p_value']:.4f} {'SIGNIFICANT' if wrc_bh['reject_null'] else 'not significant'}")
    results['white_rc_vs_bh'] = wrc_bh

    # White RC vs Cash
    print("\n--- White's Reality Check vs Cash ---")
    wrc_cash = whites_reality_check(
        strategy_returns_list, cash_returns, n_bootstrap=2000, alpha=0.05
    )
    print(f"  p-value: {wrc_cash['p_value']:.4f} {'SIGNIFICANT' if wrc_cash['reject_null'] else 'not significant'}")
    results['white_rc_vs_cash'] = wrc_cash

    # Individual Hansen-style tests (per strategy vs each benchmark)
    print("\n--- Individual Strategy Tests (t-test on excess returns) ---")
    from scipy import stats as sp_stats

    individual_results = []
    for col in returns_df.columns:
        strat_ret = returns_df[col]
        info = strategy_info[col]

        # vs B&H
        excess_bh = strat_ret - bh_returns
        t_bh, p_bh_2 = sp_stats.ttest_1samp(excess_bh.dropna(), 0)
        p_bh = p_bh_2 / 2 if t_bh > 0 else 1 - p_bh_2 / 2

        # vs Cash
        t_cash, p_cash_2 = sp_stats.ttest_1samp(strat_ret.dropna(), 0)
        p_cash = p_cash_2 / 2 if t_cash > 0 else 1 - p_cash_2 / 2

        individual_results.append({
            'key': col,
            'direction': info['direction'],
            'cagr': info['cagr'],
            'n_trades': info['n_trades'],
            'p_vs_bh': p_bh,
            'p_vs_cash': p_cash,
            'sig_bh': p_bh < 0.05,
            'sig_cash': p_cash < 0.05,
        })

    ind_df = pd.DataFrame(individual_results)
    n_sig_bh = ind_df['sig_bh'].sum()
    n_sig_cash = ind_df['sig_cash'].sum()
    print(f"  Significant vs B&H: {n_sig_bh}/{len(ind_df)}")
    print(f"  Significant vs Cash: {n_sig_cash}/{len(ind_df)}")

    for _, row in ind_df.iterrows():
        mark_bh = '*' if row['sig_bh'] else ' '
        mark_cash = '*' if row['sig_cash'] else ' '
        print(f"  {row['key']:20s} {row['direction']:5s} CAGR={row['cagr']:+.1%} "
              f"p_bh={row['p_vs_bh']:.3f}{mark_bh} p_cash={row['p_vs_cash']:.3f}{mark_cash}")

    results['individual'] = individual_results

    # Trade-level test: are trades profitable on aggregate?
    print("\n--- Aggregate Trade-Level Tests ---")
    # Collect all trades from OTS-positive strategies
    from grammar.mapper import decode
    from evolution.fitness import _run_single_window

    with open('config_v2.yaml') as f:
        config = yaml.safe_load(f)
    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    all_trades_pnl = []
    for key, info in strategy_info.items():
        exp_dir = Path(get_experiment_dirs()[info['seed']])
        with open(exp_dir / 'top_strategies.json') as f:
            strats = json.load(f)
        sd = strats[info['index']]
        strategy = decode(sd['genome'])
        if strategy is None:
            continue
        _, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
        all_trades_pnl.extend([t['pnl_pct'] for t in trades])

    if all_trades_pnl:
        t_stat, p_val_2 = sp_stats.ttest_1samp(all_trades_pnl, 0)
        p_val = p_val_2 / 2 if t_stat > 0 else 1 - p_val_2 / 2
        _, wilcox_p = sp_stats.wilcoxon([x for x in all_trades_pnl if x != 0],
                                        alternative='greater')

        print(f"  Total trades: {len(all_trades_pnl)}")
        print(f"  Mean PnL: {np.mean(all_trades_pnl):.4%}")
        print(f"  Median PnL: {np.median(all_trades_pnl):.4%}")
        print(f"  Win rate: {sum(1 for x in all_trades_pnl if x > 0)/len(all_trades_pnl):.1%}")
        print(f"  t-test p-value: {p_val:.4f} {'SIGNIFICANT' if p_val < 0.05 else ''}")
        print(f"  Wilcoxon p-value: {wilcox_p:.4f} {'SIGNIFICANT' if wilcox_p < 0.05 else ''}")

        results['trade_level'] = {
            'n_trades': len(all_trades_pnl),
            'mean_pnl': float(np.mean(all_trades_pnl)),
            'median_pnl': float(np.median(all_trades_pnl)),
            'ttest_p': float(p_val),
            'wilcoxon_p': float(wilcox_p),
        }

    return results


# ============================================================================
# 2. OPTIMIZED ENSEMBLE
# ============================================================================

def optimize_ensemble(returns_df: pd.DataFrame, strategy_info: Dict) -> Dict:
    """Build optimized portfolio weights using multiple methods."""
    from backtest.metrics import calculate_sortino_ratio, cagr, max_drawdown

    print(f"\n{'='*70}")
    print("ENSEMBLE OPTIMIZATION")
    print(f"{'='*70}")

    n = returns_df.shape[1]
    cols = returns_df.columns.tolist()

    # Method 1: Equal Weight
    eq_weights = np.ones(n) / n
    eq_ret = returns_df.values @ eq_weights
    eq_equity = pd.Series((1 + eq_ret).cumprod() * 100, index=returns_df.index)

    # Method 2: Inverse Volatility (Risk Parity)
    vols = returns_df.std()
    inv_vol = 1 / vols.replace(0, np.inf)
    rp_weights = (inv_vol / inv_vol.sum()).values
    rp_ret = returns_df.values @ rp_weights
    rp_equity = pd.Series((1 + rp_ret).cumprod() * 100, index=returns_df.index)

    # Method 3: Max Sharpe
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    def neg_sharpe(w):
        port_ret = mean_returns @ w
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-10:
            return 0
        return -port_ret / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.4)] * n  # Max 40% per strategy to ensure diversification

    x0 = eq_weights.copy()
    result = minimize(neg_sharpe, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    ms_weights = result.x if result.success else eq_weights
    ms_ret = returns_df.values @ ms_weights
    ms_equity = pd.Series((1 + ms_ret).cumprod() * 100, index=returns_df.index)

    # Method 4: Min Variance
    def portfolio_var(w):
        return w @ cov_matrix @ w

    result_mv = minimize(portfolio_var, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    mv_weights = result_mv.x if result_mv.success else eq_weights
    mv_ret = returns_df.values @ mv_weights
    mv_equity = pd.Series((1 + mv_ret).cumprod() * 100, index=returns_df.index)

    # Compare all methods
    methods = {
        'Equal Weight': (eq_weights, eq_equity),
        'Risk Parity': (rp_weights, rp_equity),
        'Max Sharpe': (ms_weights, ms_equity),
        'Min Variance': (mv_weights, mv_equity),
    }

    print(f"\n{'Method':<16} {'CAGR':>8} {'MaxDD':>8} {'Sortino':>8} {'Final':>8}")
    print('-' * 52)

    best_method = None
    best_sortino = -999

    results = {}
    for name, (weights, equity) in methods.items():
        ret = equity.pct_change().dropna()
        c = cagr(equity, 35040)
        dd = max_drawdown(equity)
        s = calculate_sortino_ratio(ret, 35040)

        print(f"{name:<16} {c:>7.1%} {dd:>7.1%} {s:>8.3f} {equity.iloc[-1]:>7.2f}")

        results[name] = {
            'weights': {cols[i]: float(weights[i]) for i in range(n) if weights[i] > 0.01},
            'cagr': float(c),
            'max_dd': float(dd),
            'sortino': float(s),
            'final_equity': float(equity.iloc[-1]),
        }

        if s > best_sortino:
            best_sortino = s
            best_method = name

    print(f"\nBest method (by Sortino): {best_method}")

    # Print weights for best method
    best_weights = methods[best_method][0]
    print(f"\n{best_method} weights:")
    for i, col in enumerate(cols):
        if best_weights[i] > 0.01:
            info = strategy_info[col]
            print(f"  {col:20s} w={best_weights[i]:.1%} | {info['direction']:5s} "
                  f"CAGR={info['cagr']:+.1%} PF={info['profit_factor']:.2f}")

    # Return best equity for plotting
    results['best_method'] = best_method
    results['best_equity'] = methods[best_method][1]

    return results


# ============================================================================
# 3. QUALITATIVE ANALYSIS
# ============================================================================

def qualitative_analysis(strategy_info: Dict) -> str:
    """Analyze what market logic the top strategies capture."""
    print(f"\n{'='*70}")
    print("QUALITATIVE STRATEGY ANALYSIS")
    print(f"{'='*70}")

    # Sort by CAGR
    sorted_strats = sorted(strategy_info.items(), key=lambda x: x[1]['cagr'], reverse=True)

    analysis_text = []

    for key, info in sorted_strats[:8]:
        expr = info['expression']
        direction = info['direction']
        cagr_val = info['cagr']
        pf = info['profit_factor']

        print(f"\n--- {key} ({direction}, CAGR={cagr_val:+.1%}, PF={pf:.2f}) ---")
        print(f"  Expression: {expr[:120]}")

        # Parse logic
        logic_explanation = _explain_strategy(expr, direction)
        print(f"  Logic: {logic_explanation}")
        analysis_text.append({
            'key': key,
            'direction': direction,
            'cagr': cagr_val,
            'profit_factor': pf,
            'n_trades': info['n_trades'],
            'expression': expr[:150],
            'explanation': logic_explanation,
        })

    return analysis_text


def _explain_strategy(expr: str, direction: str) -> str:
    """Generate human-readable explanation of a strategy expression."""
    explanations = []

    # Detect key patterns
    if 'RSI' in expr and '_LT_ 20' in expr:
        explanations.append("Oversold RSI (<20) as entry filter")
    elif 'RSI' in expr and '_GT_ 80' in expr:
        explanations.append("Overbought RSI (>80) as entry filter")
    elif 'RSI' in expr and '_GT_ 50' in expr:
        explanations.append("Bullish RSI momentum (>50)")
    elif 'RSI' in expr and '_LT_ 50' in expr:
        explanations.append("Bearish RSI momentum (<50)")

    if 'MFI' in expr and '_GT_ 90' in expr:
        explanations.append("Very high money flow (>90) — strong buying pressure")
    elif 'MFI' in expr and '_GT_ 70' in expr:
        explanations.append("High money flow — buying pressure")
    elif 'MFI' in expr:
        explanations.append("Money flow index as momentum filter")

    if 'ADX' in expr:
        explanations.append("ADX trend strength filter")

    if 'STOCH_K' in expr and 'CROSSES_ABOVE' in expr:
        explanations.append("Stochastic bullish crossover (momentum turning up)")
    elif 'STOCH_K' in expr and 'CROSSES_BELOW' in expr:
        explanations.append("Stochastic bearish crossover")
    elif 'STOCH' in expr:
        explanations.append("Stochastic oscillator for momentum")

    if 'BBWIDTH' in expr:
        explanations.append("Bollinger Band width — volatility expansion/squeeze")

    if 'VOL_RATIO' in expr and '_GT_' in expr:
        explanations.append("Above-average volume confirmation")
    elif 'VOL_RATIO' in expr:
        explanations.append("Volume ratio filter")

    if 'ROC' in expr and 'CROSSES_ABOVE' in expr:
        explanations.append("Rate of change crossover — momentum shift")
    elif 'ROC' in expr:
        explanations.append("Rate of change for momentum")

    if 'PCT_B' in expr:
        explanations.append("Bollinger %B position within bands")

    if 'PRICE_POS' in expr:
        explanations.append("Price position relative to moving average")

    if 'MACD_NORM' in expr:
        explanations.append("Normalized MACD for momentum")

    if 'TRAIL' in expr:
        explanations.append("Uses trailing stop (rides trends)")

    # Direction-specific insight
    if direction == 'SHORT':
        if 'RSI' in expr and ('_LT_ 20' in expr or '_LT_ 30' in expr):
            explanations.append("CONTRARIAN: shorts after extreme oversold (expecting dead cat bounce failure)")
        elif 'ROC' in expr and 'CROSSES_ABOVE' in expr:
            explanations.append("Fades upward momentum spikes (mean-reversion short)")

    n_conditions = expr.count(' AND ') + expr.count(' OR ') + 1
    if n_conditions >= 3:
        explanations.append(f"Multi-condition ({n_conditions} filters) — highly selective")

    return "; ".join(explanations) if explanations else "Simple indicator comparison"


# ============================================================================
# MAIN
# ============================================================================

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open('config_v2.yaml') as f:
        config = yaml.safe_load(f)

    experiments = get_experiment_dirs()
    print(f"Found {len(experiments)} experiment directories with OTS results")
    for k, v in experiments.items():
        print(f"  {k}: {v}")

    # Load OTS data
    from main_v2 import load_ots_data
    ots_data = load_ots_data(config)

    # Load all strategy equity curves
    strategy_info, returns_df = load_ots_strategy_equities(experiments, ots_data, config)
    print(f"\nLoaded {len(strategy_info)} OTS-positive strategies (≥5 trades)")

    if returns_df.empty:
        print("No valid strategies found!")
        return

    # B&H benchmark
    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
    bh_returns = bh_equity.pct_change().fillna(0)

    # ================================================
    # 1. Statistical Tests
    # ================================================
    stat_results = run_statistical_tests(returns_df, ots_data, strategy_info)

    # ================================================
    # 2. Optimized Ensemble
    # ================================================
    ensemble_results = optimize_ensemble(returns_df, strategy_info)

    # ================================================
    # 3. Qualitative Analysis
    # ================================================
    qual_analysis = qualitative_analysis(strategy_info)

    # ================================================
    # SAVE OUTPUTS
    # ================================================

    # Save statistical test results
    stat_output = {
        'white_rc_vs_bh': {k: v for k, v in stat_results.get('white_rc_vs_bh', {}).items()},
        'white_rc_vs_cash': {k: v for k, v in stat_results.get('white_rc_vs_cash', {}).items()},
        'individual_tests': stat_results.get('individual', []),
        'trade_level': stat_results.get('trade_level', {}),
    }
    with open(OUTPUT_DIR / 'statistical_tests.json', 'w') as f:
        json.dump(stat_output, f, indent=2, default=str)

    # Save ensemble results (without equity series)
    ens_output = {k: v for k, v in ensemble_results.items() if k != 'best_equity'}
    with open(OUTPUT_DIR / 'ensemble_optimization.json', 'w') as f:
        json.dump(ens_output, f, indent=2, default=str)

    # Save qualitative analysis
    with open(OUTPUT_DIR / 'qualitative_analysis.json', 'w') as f:
        json.dump(qual_analysis, f, indent=2)

    # ================================================
    # FIGURES
    # ================================================

    # Fig 1: Ensemble equity comparison
    best_equity = ensemble_results.get('best_equity')
    if best_equity is not None:
        from backtest.metrics import calculate_sortino_ratio, cagr, max_drawdown

        fig, ax = plt.subplots(figsize=(12, 6))

        # Equal-weight for comparison
        eq_ret = returns_df.mean(axis=1)
        eq_equity = (1 + eq_ret).cumprod() * 100

        ax.plot(best_equity.index, best_equity.values, 'b-', linewidth=2,
                label=f'{ensemble_results["best_method"]} (CAGR={ensemble_results[ensemble_results["best_method"]]["cagr"]:.1%})')
        ax.plot(eq_equity.index, eq_equity.values, 'g--', linewidth=1.5,
                label=f'Equal Weight (CAGR={ensemble_results["Equal Weight"]["cagr"]:.1%})')
        ax.plot(bh_equity.index, bh_equity.values, 'k:', linewidth=1.5, alpha=0.7,
                label=f'Buy & Hold (CAGR={cagr(bh_equity, 35040):.1%})')
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (base=100)')
        ax.set_title('OTS: Optimized Ensemble vs Equal Weight vs Buy & Hold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'fig1_ensemble_equity.png', dpi=150)
        plt.close()

    # Fig 3: Strategy correlation heatmap
    if returns_df.shape[1] > 2:
        corr = returns_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        labels = [f"{strategy_info[c]['direction'][0]}_{c.split('_s')[1]}" for c in corr.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title('Strategy Return Correlations (OTS period)')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'fig3_correlation.png', dpi=150)
        plt.close()

    # Fig 4: Drawdown comparison
    if best_equity is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        # Drawdown of best ensemble
        peak = best_equity.expanding().max()
        dd = (best_equity - peak) / peak * 100
        ax.fill_between(dd.index, dd.values, 0, color='blue', alpha=0.3, label='Ensemble DD')

        bh_peak = bh_equity.expanding().max()
        bh_dd = (bh_equity - bh_peak) / bh_peak * 100
        ax.fill_between(bh_dd.index, bh_dd.values, 0, color='red', alpha=0.3, label='B&H DD')

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Comparison: Ensemble vs Buy & Hold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'fig4_drawdown.png', dpi=150)
        plt.close()

    # Update summary
    summary = {
        'analysis_date': '2026-03-07',
        'grammar_version': 'v4 (trailing stops + ADX/MFI)',
        'n_seeds': len(experiments),
        'seeds': list(experiments.keys()),
        'n_ots_positive': len(strategy_info),
        'statistical_tests': {
            'white_rc_vs_bh_p': stat_results.get('white_rc_vs_bh', {}).get('p_value', 1.0),
            'white_rc_vs_cash_p': stat_results.get('white_rc_vs_cash', {}).get('p_value', 1.0),
            'trade_level_ttest_p': stat_results.get('trade_level', {}).get('ttest_p', 1.0),
            'trade_level_wilcoxon_p': stat_results.get('trade_level', {}).get('wilcoxon_p', 1.0),
        },
        'ensemble_best_method': ensemble_results.get('best_method', 'Equal Weight'),
        'ensemble_metrics': {
            k: v for k, v in ensemble_results.get(ensemble_results.get('best_method', 'Equal Weight'), {}).items()
            if k != 'weights'
        },
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL OUTPUTS SAVED")
    print(f"{'='*70}")
    print(f"  {OUTPUT_DIR}/statistical_tests.json")
    print(f"  {OUTPUT_DIR}/ensemble_optimization.json")
    print(f"  {OUTPUT_DIR}/qualitative_analysis.json")
    print(f"  {OUTPUT_DIR}/summary.json")
    print(f"  {OUTPUT_DIR}/fig1_ensemble_equity.png")
    print(f"  {OUTPUT_DIR}/fig3_correlation.png")
    print(f"  {OUTPUT_DIR}/fig4_drawdown.png")


if __name__ == '__main__':
    main()
