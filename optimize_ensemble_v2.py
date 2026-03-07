"""
Ensemble Optimization v2 — Pareto frontier approach.

Key insight from v1: adding weak strategies dilutes the top performers.
Strategy: find the Pareto-optimal ensembles across CAGR vs Sortino vs MaxDD,
then pick the best configurations for the paper.

Also tests: leverage effect on the concentrated portfolio (top 3-5).
"""

import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


OUTPUT_DIR = Path('reports/paper_v2')


def load_strategy_returns(config):
    """Load individual OTS equity curves for all positive strategies."""
    from grammar.mapper import decode
    from evolution.fitness import _run_single_window
    from main_v2 import load_ots_data

    ots_data = load_ots_data(config)
    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    results_dir = Path('results')
    experiments = {}
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / 'ots_results.json').exists():
            if '_seed' in d.name:
                seed_part = d.name.split('_seed')[1].split('_')[0]
                experiments[f'seed{seed_part}'] = str(d)

    strategies = []
    equities = {}

    for seed_name, exp_dir in experiments.items():
        exp_path = Path(exp_dir)
        with open(exp_path / 'ots_results.json') as f:
            ots_results = json.load(f)
        with open(exp_path / 'top_strategies.json') as f:
            top_strats = json.load(f)
        with open(exp_path / 'validation.json') as f:
            val_results = json.load(f)

        val_map = {v['strategy_index']: v for v in val_results}

        for r in ots_results:
            m = r.get('metrics', {})
            idx = r.get('strategy_index', -1)
            cagr_val = m.get('cagr', 0)
            n_trades = r.get('n_trades', 0)
            if cagr_val <= 0 or n_trades < 5:
                continue

            sd = top_strats[idx]
            strategy = decode(sd['genome'])
            if strategy is None:
                continue

            equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
            key = f"{seed_name}_s{idx}"
            equities[key] = equity

            v = val_map.get(idx, {})
            strategies.append({
                'key': key,
                'seed': seed_name,
                'index': idx,
                'direction': r.get('direction', ''),
                'cagr': cagr_val,
                'sortino': m.get('sortino', 0),
                'max_dd': m.get('max_dd', 0),
                'profit_factor': m.get('profit_factor', 0),
                'win_rate': m.get('win_rate', 0),
                'n_trades': n_trades,
                'expression': r.get('expression', ''),
                'pbo': v.get('pbo', 1.0),
                'perm_p': v.get('perm_p_value', 1.0),
            })

    # Build returns from equities
    returns_dict = {}
    for k, eq in equities.items():
        returns_dict[k] = eq.pct_change().fillna(0)
    returns_df = pd.DataFrame(returns_dict)

    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
    return strategies, returns_df, equities, bh_equity, ots_data


def deduplicate(strategies, returns_df, threshold=0.98):
    """Remove near-duplicates by return correlation."""
    corr = returns_df.corr()
    to_remove = set()
    keys = list(returns_df.columns)
    for i in range(len(keys)):
        if keys[i] in to_remove:
            continue
        for j in range(i + 1, len(keys)):
            if keys[j] in to_remove:
                continue
            if abs(corr.loc[keys[i], keys[j]]) > threshold:
                si = next(s for s in strategies if s['key'] == keys[i])
                sj = next(s for s in strategies if s['key'] == keys[j])
                loser = keys[j] if si['cagr'] >= sj['cagr'] else keys[i]
                to_remove.add(loser)
    kept = [s for s in strategies if s['key'] not in to_remove]
    kept_df = returns_df.drop(columns=list(to_remove))
    return kept, kept_df, to_remove


def metrics(returns_series, periods=35040):
    """Compute CAGR, MaxDD, Sortino, Calmar from returns series."""
    r = returns_series.replace([np.inf, -np.inf], 0).fillna(0)
    eq = (1 + r).cumprod() * 100
    n = len(eq)
    if n < 2 or eq.iloc[-1] <= 0:
        return {'cagr': -999, 'max_dd': -1, 'sortino': -999, 'calmar': -999, 'final': 0}
    years = n / periods
    cagr_v = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-6)) - 1
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    down = r[r < 0]
    if len(down) > 0:
        ds = np.sqrt((down ** 2).mean())
        sortino = (r.mean() / ds) * np.sqrt(periods) if ds > 0 else 999
    else:
        sortino = 999
    calmar = cagr_v / abs(max_dd) if abs(max_dd) > 1e-10 else 999
    return {'cagr': cagr_v, 'max_dd': max_dd, 'sortino': sortino,
            'calmar': calmar, 'final': eq.iloc[-1]}


def max_sharpe_optimize(returns_df, keys, min_w=0.01, max_w=0.60):
    """Find Max Sharpe weights with bounds."""
    if len(keys) < 2:
        return np.array([1.0])
    R = returns_df[keys].values
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(keys)

    def neg_sharpe(w):
        p = R @ w
        p = p[np.isfinite(p)]
        if len(p) < 10:
            return 1e6
        return -(np.mean(p) / max(np.std(p), 1e-12))

    bounds = [(min_w, max_w)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    best_w, best_v = np.ones(n) / n, neg_sharpe(np.ones(n) / n)

    for _ in range(30):
        w0 = np.random.dirichlet(np.ones(n))
        w0 = np.clip(w0, min_w, max_w)
        w0 /= w0.sum()
        try:
            res = minimize(neg_sharpe, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000})
            if res.success and res.fun < best_v:
                best_v = res.fun
                best_w = res.x
        except Exception:
            pass

    best_w = np.clip(best_w, min_w, max_w)
    best_w /= best_w.sum()
    return best_w


def greedy_forward(strategies, returns_df, max_n=10, metric='sortino'):
    """Greedy forward selection maximizing ensemble metric."""
    available = [s['key'] for s in strategies]
    selected = []
    best_score = -999

    while len(selected) < max_n and available:
        best_add, best_new = None, -999
        for c in available:
            trial = selected + [c]
            m = metrics(returns_df[trial].mean(axis=1))
            if m[metric] > best_new:
                best_new = m[metric]
                best_add = c
        if best_add is None or best_new < best_score * 0.95:
            break
        selected.append(best_add)
        available.remove(best_add)
        best_score = best_new
    return selected


def build_ensemble_equity(returns_df, keys, weights=None):
    """Build weighted ensemble returns."""
    if weights is None:
        return returns_df[keys].mean(axis=1)
    return pd.Series(returns_df[keys].values @ weights, index=returns_df.index)


def main():
    with open('config_v2.yaml') as f:
        config = yaml.safe_load(f)

    print("Loading strategies...")
    strategies, returns_df, equities, bh_equity, ots_data = load_strategy_returns(config)
    print(f"Loaded {len(strategies)} OTS-positive strategies")

    # Deduplicate
    strategies, returns_df, removed = deduplicate(strategies, returns_df)
    print(f"After dedup: {len(strategies)} (removed {len(removed)})")

    # B&H
    bh_ret = bh_equity.pct_change().fillna(0)
    bh_m = metrics(bh_ret)
    print(f"B&H: CAGR={bh_m['cagr']:.1%}, MaxDD={bh_m['max_dd']:.1%}, Sortino={bh_m['sortino']:.3f}")

    # ================================================================
    # Build all candidate ensembles
    # ================================================================
    candidates = {}

    # --- Single best ---
    best_cagr = max(strategies, key=lambda x: x['cagr'])
    candidates['Single_best'] = {
        'keys': [best_cagr['key']], 'weights': None,
        'desc': f"Best single strategy ({best_cagr['key']})"
    }

    # --- Top N by CAGR ---
    for n in [3, 5, 8]:
        ranked = sorted(strategies, key=lambda x: x['cagr'], reverse=True)
        keys = [s['key'] for s in ranked[:n]]
        candidates[f'Top{n}_EW'] = {'keys': keys, 'weights': None,
                                     'desc': f'Top {n} by CAGR, equal weight'}
        w = max_sharpe_optimize(returns_df, keys)
        candidates[f'Top{n}_MS'] = {'keys': keys, 'weights': w,
                                     'desc': f'Top {n} by CAGR, max Sharpe'}

    # --- Top N by Sortino ---
    for n in [5, 8]:
        ranked = sorted(strategies, key=lambda x: x['sortino'], reverse=True)
        keys = [s['key'] for s in ranked[:n]]
        candidates[f'Top{n}Sort_EW'] = {'keys': keys, 'weights': None,
                                         'desc': f'Top {n} by Sortino, equal weight'}
        w = max_sharpe_optimize(returns_df, keys)
        candidates[f'Top{n}Sort_MS'] = {'keys': keys, 'weights': w,
                                         'desc': f'Top {n} by Sortino, max Sharpe'}

    # --- Greedy forward ---
    for metric in ['sortino', 'cagr', 'calmar']:
        fwd = greedy_forward(strategies, returns_df, max_n=10, metric=metric)
        if fwd:
            candidates[f'Greedy_{metric}_EW'] = {'keys': fwd, 'weights': None,
                                                  'desc': f'Greedy fwd ({metric}), equal'}
            if len(fwd) >= 2:
                w = max_sharpe_optimize(returns_df, fwd)
                candidates[f'Greedy_{metric}_MS'] = {'keys': fwd, 'weights': w,
                                                      'desc': f'Greedy fwd ({metric}), max Sharpe'}

    # --- Quality filter: CAGR > 5% AND PF > 1.2 ---
    quality = [s for s in strategies if s['cagr'] > 0.05 and s['profit_factor'] > 1.2]
    if quality:
        qkeys = [s['key'] for s in quality]
        candidates['Quality_EW'] = {'keys': qkeys, 'weights': None,
                                    'desc': f'CAGR>5% & PF>1.2 ({len(qkeys)}), equal'}
        if len(qkeys) >= 2:
            w = max_sharpe_optimize(returns_df, qkeys)
            candidates['Quality_MS'] = {'keys': qkeys, 'weights': w,
                                        'desc': f'CAGR>5% & PF>1.2 ({len(qkeys)}), max Sharpe'}

    # --- Direction-balanced: top N LONG + top N SHORT ---
    for n in [3, 5]:
        longs = sorted([s for s in strategies if s['direction'] == 'LONG'],
                        key=lambda x: x['cagr'], reverse=True)
        shorts = sorted([s for s in strategies if s['direction'] == 'SHORT'],
                         key=lambda x: x['cagr'], reverse=True)
        keys = [s['key'] for s in longs[:n]] + [s['key'] for s in shorts[:n]]
        if keys:
            candidates[f'Balanced_{n}L{n}S_EW'] = {'keys': keys, 'weights': None,
                                                     'desc': f'Top {n} LONG + {n} SHORT, equal'}
            if len(keys) >= 2:
                w = max_sharpe_optimize(returns_df, keys)
                candidates[f'Balanced_{n}L{n}S_MS'] = {'keys': keys, 'weights': w,
                                                         'desc': f'Top {n} L + {n} S, max Sharpe'}

    # --- Exhaustive on top 12 (CAGR > 3%) ---
    strong = sorted([s for s in strategies if s['cagr'] > 0.03],
                     key=lambda x: x['cagr'], reverse=True)[:12]
    if len(strong) >= 2:
        print(f"\nExhaustive search on {len(strong)} strategies (CAGR > 3%)...")
        best_combo_sort, best_sort = None, -999
        best_combo_cagr, best_cagr_v = None, -999
        skeys = [s['key'] for s in strong]

        for size in range(2, min(9, len(skeys) + 1)):
            for combo in combinations(skeys, size):
                ens_ret = returns_df[list(combo)].mean(axis=1)
                m = metrics(ens_ret)
                if m['max_dd'] > -0.15:
                    if m['sortino'] > best_sort:
                        best_sort = m['sortino']
                        best_combo_sort = list(combo)
                    if m['cagr'] > best_cagr_v:
                        best_cagr_v = m['cagr']
                        best_combo_cagr = list(combo)

        if best_combo_sort:
            candidates['Exhaustive_Sortino_EW'] = {'keys': best_combo_sort, 'weights': None,
                                                    'desc': 'Exhaustive best Sortino, equal'}
            w = max_sharpe_optimize(returns_df, best_combo_sort)
            candidates['Exhaustive_Sortino_MS'] = {'keys': best_combo_sort, 'weights': w,
                                                    'desc': 'Exhaustive best Sortino, max Sharpe'}
        if best_combo_cagr:
            candidates['Exhaustive_CAGR_EW'] = {'keys': best_combo_cagr, 'weights': None,
                                                 'desc': 'Exhaustive best CAGR, equal'}

    # ================================================================
    # Evaluate all candidates
    # ================================================================
    print(f"\n{'='*80}")
    print(f"{'Method':<28} {'N':>3} {'CAGR':>7} {'MaxDD':>7} {'Sort':>7} {'Calm':>7} {'Final':>7}")
    print(f"{'='*80}")

    results = {}
    for name, cand in candidates.items():
        ens_ret = build_ensemble_equity(returns_df, cand['keys'], cand['weights'])
        m = metrics(ens_ret)
        results[name] = {**cand, 'metrics': m}

        calm_s = f"{m['calmar']:.1f}" if abs(m['calmar']) < 100 else ">100"
        print(f"{name:<28} {len(cand['keys']):>3} {m['cagr']:>6.1%} {m['max_dd']:>6.2%} "
              f"{m['sortino']:>7.3f} {calm_s:>7} {m['final']:>7.2f}")

    bh_calm = f"{bh_m['calmar']:.1f}" if abs(bh_m['calmar']) < 100 else ">100"
    print(f"{'B&H':<28} {'1':>3} {bh_m['cagr']:>6.1%} {bh_m['max_dd']:>6.2%} "
          f"{bh_m['sortino']:>7.3f} {bh_calm:>7} {bh_m['final']:>7.2f}")

    # ================================================================
    # Pick the Pareto-optimal portfolios
    # ================================================================
    print(f"\n{'='*80}")
    print("PARETO-OPTIMAL PORTFOLIOS")
    print(f"{'='*80}")

    # Pareto: a portfolio dominates another if it's better on ALL of (CAGR, Sortino, -MaxDD)
    pareto = []
    items = list(results.items())
    for i, (n1, r1) in enumerate(items):
        m1 = r1['metrics']
        dominated = False
        for j, (n2, r2) in enumerate(items):
            if i == j:
                continue
            m2 = r2['metrics']
            if (m2['cagr'] >= m1['cagr'] and m2['sortino'] >= m1['sortino']
                    and m2['max_dd'] >= m1['max_dd']  # less negative = better
                    and (m2['cagr'] > m1['cagr'] or m2['sortino'] > m1['sortino']
                         or m2['max_dd'] > m1['max_dd'])):
                dominated = True
                break
        if not dominated:
            pareto.append((n1, r1))

    pareto.sort(key=lambda x: x[1]['metrics']['cagr'], reverse=True)

    print(f"\n{'Name':<28} {'N':>3} {'CAGR':>7} {'MaxDD':>7} {'Sort':>7} {'Calm':>7} {'Profile'}")
    print("-" * 90)
    for name, data in pareto:
        m = data['metrics']
        profile = "HIGH RETURN" if m['cagr'] > 0.15 else "BALANCED" if m['cagr'] > 0.08 else "CONSERVATIVE"
        calm_s = f"{m['calmar']:.1f}" if abs(m['calmar']) < 100 else ">100"
        print(f"{name:<28} {len(data['keys']):>3} {m['cagr']:>6.1%} {m['max_dd']:>6.2%} "
              f"{m['sortino']:>7.3f} {calm_s:>7} {profile}")

    # ================================================================
    # Select 3 portfolios for the paper
    # ================================================================
    print(f"\n{'='*80}")
    print("RECOMMENDED PORTFOLIOS FOR PAPER")
    print(f"{'='*80}")

    # 1. Highest CAGR (concentrated, any size)
    highest_cagr = max(results.items(),
                       key=lambda x: x[1]['metrics']['cagr'])

    # 2. Best ensemble (N>=3) by CAGR with MaxDD < 5%
    multi_strat = [(n, d) for n, d in results.items() if len(d['keys']) >= 3]
    high_return = max(multi_strat,
                      key=lambda x: x[1]['metrics']['cagr']
                      if x[1]['metrics']['max_dd'] > -0.05 else -999)

    # 3. Highest Sortino (N>=3)
    best_sortino = max(multi_strat,
                       key=lambda x: x[1]['metrics']['sortino'])

    # 4. Best Calmar (risk-adjusted, N>=3)
    best_calmar = max(multi_strat,
                      key=lambda x: x[1]['metrics']['calmar']
                      if x[1]['metrics']['calmar'] < 100 else -999)

    picks = [
        ("AGGRESSIVE", highest_cagr),
        ("HIGH_RETURN", high_return),
        ("RISK_ADJUSTED", best_sortino),
        ("CONSERVATIVE", best_calmar),
    ]

    for label, (name, data) in picks:
        m = data['metrics']
        print(f"\n--- {label}: {name} ---")
        print(f"  CAGR:    {m['cagr']:.2%}")
        print(f"  MaxDD:   {m['max_dd']:.2%}")
        print(f"  Sortino: {m['sortino']:.3f}")
        calm = m['calmar']
        print(f"  Calmar:  {calm:.2f}" if abs(calm) < 100 else f"  Calmar:  >100")
        print(f"  Final:   {m['final']:.2f}")
        print(f"  Strategies ({len(data['keys'])}):")
        for k in data['keys']:
            s = next((s for s in strategies if s['key'] == k), None)
            if s:
                w_info = ""
                if data['weights'] is not None:
                    w_dict = dict(zip(data['keys'], data['weights']))
                    w_info = f" w={w_dict.get(k, 0):.1%}"
                print(f"    {k:<22} {s['direction']:<6} CAGR={s['cagr']:.1%} "
                      f"PF={s['profit_factor']:.2f} Trades={s['n_trades']}{w_info}")

    # ================================================================
    # Generate comparison figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Equity curves of the 4 recommended portfolios
    ax = axes[0, 0]
    colors = ['red', 'blue', 'green', 'purple']
    for i, (label, (name, data)) in enumerate(picks):
        ens_ret = build_ensemble_equity(returns_df, data['keys'], data['weights'])
        eq = (1 + ens_ret).cumprod() * 100
        ax.plot(eq.index, eq.values, color=colors[i], linewidth=2,
                label=f'{label} ({data["metrics"]["cagr"]:.1%})')

    ax.plot(bh_equity.index, bh_equity.values, 'k--', alpha=0.5,
            label=f'B&H ({bh_m["cagr"]:.1%})')
    ax.axhline(y=100, color='gray', ls=':', alpha=0.3)
    ax.set_title('Recommended Portfolios — OTS Equity Curves')
    ax.set_ylabel('Equity (base=100)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: CAGR vs Sortino scatter of all candidates
    ax = axes[0, 1]
    for name, data in results.items():
        m = data['metrics']
        n = len(data['keys'])
        color = 'green' if n <= 3 else 'blue' if n <= 8 else 'orange'
        ax.scatter(m['sortino'], m['cagr'] * 100, c=color, s=50, alpha=0.7)
        if name in [p[0] for _, p in picks]:
            ax.annotate(name, (m['sortino'], m['cagr'] * 100),
                        fontsize=6, ha='left')
    ax.set_xlabel('Sortino Ratio')
    ax.set_ylabel('CAGR (%)')
    ax.set_title('All Ensembles: CAGR vs Sortino')
    ax.grid(True, alpha=0.3)

    # Plot 3: MaxDD comparison bar chart
    ax = axes[1, 0]
    pick_names = [p[0] for p in picks]
    pick_dd = [abs(results[p[1][0]]['metrics']['max_dd']) * 100 for p in picks]
    pick_dd.append(abs(bh_m['max_dd']) * 100)
    pick_names.append('B&H')
    bars = ax.barh(pick_names, pick_dd, color=['red', 'blue', 'green', 'purple', 'black'])
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown Comparison')
    for i, v in enumerate(pick_dd):
        ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

    # Plot 4: CAGR comparison bar chart
    ax = axes[1, 1]
    pick_cagr = [results[p[1][0]]['metrics']['cagr'] * 100 for p in picks]
    pick_cagr.append(bh_m['cagr'] * 100)
    bars = ax.barh(pick_names, pick_cagr,
                   color=['red', 'blue', 'green', 'purple', 'black'])
    ax.set_xlabel('CAGR (%)')
    ax.set_title('CAGR Comparison')
    ax.axvline(x=0, color='gray', ls='-', alpha=0.3)
    for i, v in enumerate(pick_cagr):
        ax.text(v + 0.5 if v > 0 else v - 3, i, f'{v:.1f}%', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_ensemble_comparison.png', dpi=150)
    plt.close()

    # ================================================================
    # Save results
    # ================================================================
    output = {
        'recommended': {},
        'all_candidates': {},
        'pareto_frontier': [],
        'buy_and_hold': bh_m,
    }

    for label, (name, data) in picks:
        m = data['metrics']
        output['recommended'][label] = {
            'method': name,
            'n_strategies': len(data['keys']),
            'strategies': data['keys'],
            'weights': dict(zip(data['keys'], data['weights'].tolist())) if data['weights'] is not None else 'equal',
            'cagr': m['cagr'],
            'max_dd': m['max_dd'],
            'sortino': m['sortino'],
            'calmar': m['calmar'],
            'final_equity': m['final'],
        }

    for name, data in results.items():
        m = data['metrics']
        output['all_candidates'][name] = {
            'n': len(data['keys']),
            'cagr': m['cagr'],
            'max_dd': m['max_dd'],
            'sortino': m['sortino'],
            'calmar': m['calmar'],
        }

    for name, data in pareto:
        output['pareto_frontier'].append(name)

    with open(OUTPUT_DIR / 'ensemble_final.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Also save a clean CSV for the paper
    rows = []
    for label, (name, data) in picks:
        m = data['metrics']
        rows.append({
            'Portfolio': label,
            'Method': name,
            'N': len(data['keys']),
            'CAGR': f"{m['cagr']:.2%}",
            'Max Drawdown': f"{m['max_dd']:.2%}",
            'Sortino': f"{m['sortino']:.3f}",
            'Calmar': f"{m['calmar']:.1f}" if abs(m['calmar']) < 100 else ">100",
            'Final Equity': f"{m['final']:.2f}",
        })
    rows.append({
        'Portfolio': 'Buy & Hold',
        'Method': '-',
        'N': 1,
        'CAGR': f"{bh_m['cagr']:.2%}",
        'Max Drawdown': f"{bh_m['max_dd']:.2%}",
        'Sortino': f"{bh_m['sortino']:.3f}",
        'Calmar': f"{bh_m['calmar']:.1f}",
        'Final Equity': f"{bh_m['final']:.2f}",
    })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'table_final_portfolios.csv', index=False)

    print(f"\n\nOutputs saved:")
    print(f"  {OUTPUT_DIR / 'ensemble_final.json'}")
    print(f"  {OUTPUT_DIR / 'table_final_portfolios.csv'}")
    print(f"  {OUTPUT_DIR / 'fig5_ensemble_comparison.png'}")


if __name__ == '__main__':
    main()
