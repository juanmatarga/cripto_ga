"""
Exhaustive ensemble optimization — find the best subset + weights.

Approaches:
1. Top-N by CAGR (N=3,5,8,10)
2. Top-N by Sortino
3. Top-N by Profit Factor
4. Greedy forward selection (maximize ensemble Sortino)
5. Greedy backward elimination
6. Max Sharpe optimization on each subset
7. Deduplicate near-identical strategies
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import combinations
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
    returns_dict = {}

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
            cagr = m.get('cagr', 0)
            n_trades = r.get('n_trades', 0)
            if cagr <= 0 or n_trades < 5:
                continue

            # Get equity curve
            sd = top_strats[idx]
            strategy = decode(sd['genome'])
            if strategy is None:
                continue

            equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period)
            rets = equity.pct_change().fillna(0)

            key = f"{seed_name}_s{idx}"
            returns_dict[key] = rets

            v = val_map.get(idx, {})
            strategies.append({
                'key': key,
                'seed': seed_name,
                'index': idx,
                'direction': r.get('direction', ''),
                'cagr': cagr,
                'sortino': m.get('sortino', 0),
                'max_dd': m.get('max_dd', 0),
                'profit_factor': m.get('profit_factor', 0),
                'win_rate': m.get('win_rate', 0),
                'n_trades': n_trades,
                'expression': r.get('expression', ''),
                'pbo': v.get('pbo', 1.0),
                'perm_p': v.get('perm_p_value', 1.0),
            })

    # Build aligned returns DataFrame
    returns_df = pd.DataFrame(returns_dict)

    # B&H benchmark
    bh_equity = ots_data['Close'] / ots_data['Close'].iloc[0] * 100
    bh_returns = bh_equity.pct_change().fillna(0)

    return strategies, returns_df, bh_equity, bh_returns, ots_data


def deduplicate_strategies(strategies, returns_df, corr_threshold=0.98):
    """Remove near-duplicate strategies (correlation > threshold)."""
    corr = returns_df.corr()
    to_remove = set()
    keys = list(returns_df.columns)

    for i in range(len(keys)):
        if keys[i] in to_remove:
            continue
        for j in range(i + 1, len(keys)):
            if keys[j] in to_remove:
                continue
            if abs(corr.loc[keys[i], keys[j]]) > corr_threshold:
                # Remove the one with lower CAGR
                si = next(s for s in strategies if s['key'] == keys[i])
                sj = next(s for s in strategies if s['key'] == keys[j])
                loser = keys[j] if si['cagr'] >= sj['cagr'] else keys[i]
                to_remove.add(loser)

    kept = [s for s in strategies if s['key'] not in to_remove]
    kept_df = returns_df.drop(columns=list(to_remove))
    return kept, kept_df, to_remove


def ensemble_metrics(returns_series, periods_per_year=35040):
    """Calculate key metrics for an ensemble returns series."""
    returns_series = returns_series.replace([np.inf, -np.inf], 0).fillna(0)
    equity = (1 + returns_series).cumprod() * 100

    # CAGR
    n_periods = len(equity)
    if n_periods < 2 or equity.iloc[-1] <= 0 or equity.iloc[0] <= 0:
        return {'cagr': -999, 'max_dd': -1, 'sortino': -999, 'final': 0}

    years = n_periods / periods_per_year
    if years <= 0:
        return {'cagr': -999, 'max_dd': -1, 'sortino': -999, 'final': 0}

    cagr_val = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    # Max drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sortino
    downside = returns_series[returns_series < 0]
    if len(downside) > 0:
        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std > 0:
            sortino = (returns_series.mean() / downside_std) * np.sqrt(periods_per_year)
        else:
            sortino = 999.0
    else:
        sortino = 999.0

    # Calmar
    calmar = cagr_val / abs(max_dd) if abs(max_dd) > 1e-10 else 999.0

    return {
        'cagr': cagr_val,
        'max_dd': max_dd,
        'sortino': sortino,
        'calmar': calmar,
        'final': equity.iloc[-1],
    }


def equal_weight_ensemble(returns_df, keys):
    """Equal weight ensemble returns."""
    if not keys:
        return pd.Series(dtype=float)
    return returns_df[keys].mean(axis=1)


def max_sharpe_weights(returns_df, keys):
    """Optimize for maximum Sharpe ratio."""
    if len(keys) < 2:
        return np.array([1.0])

    R = returns_df[keys].values
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(keys)

    def neg_sharpe(w):
        port_ret = R @ w
        port_ret = port_ret[np.isfinite(port_ret)]
        if len(port_ret) < 10:
            return 1e6
        mu = np.mean(port_ret)
        std = np.std(port_ret)
        if std < 1e-12:
            return 1e6
        return -mu / std

    bounds = [(0.01, 0.5)] * n  # Min 1%, max 50% per strategy
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = np.ones(n) / n

    # Try multiple random starts
    best_w = w0
    best_val = neg_sharpe(w0)

    for _ in range(20):
        w_init = np.random.dirichlet(np.ones(n))
        w_init = np.clip(w_init, 0.01, 0.5)
        w_init /= w_init.sum()
        try:
            res = minimize(neg_sharpe, w_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 500})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w = res.x
        except Exception:
            pass

    best_w = np.clip(best_w, 0.01, 0.5)
    best_w /= best_w.sum()
    return best_w


def greedy_forward_selection(strategies, returns_df, max_size=10, metric='sortino'):
    """Greedy forward: add strategy that most improves ensemble metric."""
    available = [s['key'] for s in strategies]
    selected = []
    best_score = -999

    while len(selected) < max_size and available:
        best_add = None
        best_new_score = -999

        for candidate in available:
            trial = selected + [candidate]
            ens_ret = returns_df[trial].mean(axis=1)
            m = ensemble_metrics(ens_ret)
            score = m[metric]

            if score > best_new_score:
                best_new_score = score
                best_add = candidate

        if best_add is None or best_new_score <= best_score * 0.95:
            break  # Stop if no improvement (allow 5% tolerance)

        selected.append(best_add)
        available.remove(best_add)
        best_score = best_new_score

    return selected


def greedy_backward_elimination(strategies, returns_df, min_size=3, metric='sortino'):
    """Greedy backward: remove strategy that least hurts ensemble metric."""
    current = [s['key'] for s in strategies]
    ens_ret = returns_df[current].mean(axis=1)
    current_score = ensemble_metrics(ens_ret)[metric]

    while len(current) > min_size:
        best_remove = None
        best_score_after = -999

        for candidate in current:
            trial = [k for k in current if k != candidate]
            ens_ret = returns_df[trial].mean(axis=1)
            m = ensemble_metrics(ens_ret)
            score = m[metric]

            if score > best_score_after:
                best_score_after = score
                best_remove = candidate

        if best_score_after <= current_score:
            break  # Removing hurts — stop

        current.remove(best_remove)
        current_score = best_score_after

    return current


def exhaustive_search(strategies, returns_df, max_size=8):
    """Try all combinations up to max_size (if feasible)."""
    keys = [s['key'] for s in strategies]
    n = len(keys)

    best_result = None
    best_combo = None
    best_sortino = -999

    total_combos = 0
    for size in range(2, min(max_size + 1, n + 1)):
        combos = list(combinations(keys, size))
        total_combos += len(combos)

    if total_combos > 50000:
        print(f"  Too many combos ({total_combos}), skipping exhaustive search")
        return None, None

    print(f"  Testing {total_combos} combinations...")

    for size in range(2, min(max_size + 1, n + 1)):
        for combo in combinations(keys, size):
            ens_ret = returns_df[list(combo)].mean(axis=1)
            m = ensemble_metrics(ens_ret)
            if m['sortino'] > best_sortino and m['max_dd'] > -0.15:
                best_sortino = m['sortino']
                best_combo = list(combo)
                best_result = m

    return best_combo, best_result


def run_optimization():
    """Main optimization pipeline."""
    with open('config_v2.yaml') as f:
        config = yaml.safe_load(f)

    print("Loading strategy returns...")
    strategies, returns_df, bh_equity, bh_returns, ots_data = load_strategy_returns(config)
    print(f"Loaded {len(strategies)} OTS-positive strategies")

    # Print strategy summary
    print(f"\n{'Key':<22} {'Dir':<6} {'CAGR':>7} {'Sort':>7} {'MaxDD':>8} {'PF':>6} {'Trades':>7}")
    print("-" * 65)
    for s in sorted(strategies, key=lambda x: x['cagr'], reverse=True):
        print(f"{s['key']:<22} {s['direction']:<6} {s['cagr']:>6.1%} {s['sortino']:>7.3f} "
              f"{s['max_dd']:>7.2%} {s['profit_factor']:>6.2f} {s['n_trades']:>7}")

    # B&H metrics
    bh_m = ensemble_metrics(bh_returns)
    print(f"\nBuy & Hold: CAGR={bh_m['cagr']:.2%}, MaxDD={bh_m['max_dd']:.2%}, "
          f"Sortino={bh_m['sortino']:.3f}")

    # ================================================================
    # STEP 1: Deduplicate
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DEDUPLICATION (corr > 0.98)")
    print("=" * 70)
    deduped, deduped_df, removed = deduplicate_strategies(strategies, returns_df)
    if removed:
        print(f"Removed {len(removed)} duplicates: {removed}")
    else:
        print("No duplicates found")
    print(f"Remaining: {len(deduped)} strategies")

    results = {}

    # ================================================================
    # STEP 2: Top-N by different criteria
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TOP-N ENSEMBLES")
    print("=" * 70)

    for n in [3, 5, 8, 10, len(deduped)]:
        if n > len(deduped):
            continue

        for sort_key, label in [('cagr', 'CAGR'), ('sortino', 'Sortino'), ('profit_factor', 'PF')]:
            ranked = sorted(deduped, key=lambda x: x[sort_key], reverse=True)
            top_keys = [s['key'] for s in ranked[:n]]

            # Equal weight
            ew_ret = equal_weight_ensemble(deduped_df, top_keys)
            ew_m = ensemble_metrics(ew_ret)

            tag = f"Top{n}_by_{label}_EW"
            results[tag] = {'keys': top_keys, 'metrics': ew_m, 'weights': 'equal'}
            print(f"  {tag:<30} CAGR={ew_m['cagr']:>7.2%} MaxDD={ew_m['max_dd']:>7.2%} "
                  f"Sortino={ew_m['sortino']:>7.3f} Final={ew_m['final']:>7.2f}")

            # Max Sharpe (only for N >= 2)
            if n >= 2:
                ms_w = max_sharpe_weights(deduped_df, top_keys)
                ms_ret = (deduped_df[top_keys].values @ ms_w)
                ms_ret = pd.Series(ms_ret, index=deduped_df.index)
                ms_m = ensemble_metrics(ms_ret)

                tag_ms = f"Top{n}_by_{label}_MS"
                results[tag_ms] = {'keys': top_keys, 'metrics': ms_m,
                                   'weights': dict(zip(top_keys, ms_w.tolist()))}
                print(f"  {tag_ms:<30} CAGR={ms_m['cagr']:>7.2%} MaxDD={ms_m['max_dd']:>7.2%} "
                      f"Sortino={ms_m['sortino']:>7.3f} Final={ms_m['final']:>7.2f}")

    # ================================================================
    # STEP 3: Greedy selection
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: GREEDY SELECTION")
    print("=" * 70)

    for metric in ['sortino', 'cagr', 'calmar']:
        # Forward
        fwd = greedy_forward_selection(deduped, deduped_df, max_size=10, metric=metric)
        fwd_ret = equal_weight_ensemble(deduped_df, fwd)
        fwd_m = ensemble_metrics(fwd_ret)
        tag = f"Greedy_Fwd_{metric}_EW"
        results[tag] = {'keys': fwd, 'metrics': fwd_m, 'weights': 'equal'}
        print(f"  {tag:<30} N={len(fwd):>2} CAGR={fwd_m['cagr']:>7.2%} "
              f"MaxDD={fwd_m['max_dd']:>7.2%} Sortino={fwd_m['sortino']:>7.3f}")

        # Forward + Max Sharpe
        if len(fwd) >= 2:
            ms_w = max_sharpe_weights(deduped_df, fwd)
            ms_ret = pd.Series(deduped_df[fwd].values @ ms_w, index=deduped_df.index)
            ms_m = ensemble_metrics(ms_ret)
            tag_ms = f"Greedy_Fwd_{metric}_MS"
            results[tag_ms] = {'keys': fwd, 'metrics': ms_m,
                               'weights': dict(zip(fwd, ms_w.tolist()))}
            print(f"  {tag_ms:<30} N={len(fwd):>2} CAGR={ms_m['cagr']:>7.2%} "
                  f"MaxDD={ms_m['max_dd']:>7.2%} Sortino={ms_m['sortino']:>7.3f}")

    # Backward elimination
    for metric in ['sortino', 'cagr']:
        bwd = greedy_backward_elimination(deduped, deduped_df, min_size=3, metric=metric)
        bwd_ret = equal_weight_ensemble(deduped_df, bwd)
        bwd_m = ensemble_metrics(bwd_ret)
        tag = f"Greedy_Bwd_{metric}_EW"
        results[tag] = {'keys': bwd, 'metrics': bwd_m, 'weights': 'equal'}
        print(f"  {tag:<30} N={len(bwd):>2} CAGR={bwd_m['cagr']:>7.2%} "
              f"MaxDD={bwd_m['max_dd']:>7.2%} Sortino={bwd_m['sortino']:>7.3f}")

    # ================================================================
    # STEP 4: Exhaustive search (if feasible)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: EXHAUSTIVE SEARCH (all combos, MaxDD > -15%)")
    print("=" * 70)

    # Filter to strategies with CAGR > 3% to make exhaustive search feasible
    strong = [s for s in deduped if s['cagr'] > 0.03]
    strong_df = deduped_df[[s['key'] for s in strong]]
    print(f"  Using {len(strong)} strategies with CAGR > 3%")
    ex_combo, ex_m = exhaustive_search(strong, strong_df, max_size=8)
    if ex_combo:
        tag = "Exhaustive_best_EW"
        results[tag] = {'keys': ex_combo, 'metrics': ex_m, 'weights': 'equal'}
        print(f"  {tag:<30} N={len(ex_combo):>2} CAGR={ex_m['cagr']:>7.2%} "
              f"MaxDD={ex_m['max_dd']:>7.2%} Sortino={ex_m['sortino']:>7.3f}")
        print(f"  Strategies: {ex_combo}")

        # Max Sharpe on exhaustive best
        if len(ex_combo) >= 2:
            ms_w = max_sharpe_weights(deduped_df, ex_combo)
            ms_ret = pd.Series(deduped_df[ex_combo].values @ ms_w, index=deduped_df.index)
            ms_m = ensemble_metrics(ms_ret)
            tag_ms = "Exhaustive_best_MS"
            results[tag_ms] = {'keys': ex_combo, 'metrics': ms_m,
                               'weights': dict(zip(ex_combo, ms_w.tolist()))}
            print(f"  {tag_ms:<30} N={len(ex_combo):>2} CAGR={ms_m['cagr']:>7.2%} "
                  f"MaxDD={ms_m['max_dd']:>7.2%} Sortino={ms_m['sortino']:>7.3f}")

    # ================================================================
    # STEP 5: Single best strategy (benchmark)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SINGLE BEST (benchmark)")
    print("=" * 70)

    for metric in ['cagr', 'sortino', 'profit_factor']:
        best_s = max(deduped, key=lambda x: x[metric])
        single_ret = deduped_df[best_s['key']]
        single_m = ensemble_metrics(single_ret)
        tag = f"Single_best_{metric}"
        results[tag] = {'keys': [best_s['key']], 'metrics': single_m, 'weights': 'single'}
        print(f"  {tag:<30} {best_s['key']:<22} CAGR={single_m['cagr']:>7.2%} "
              f"MaxDD={single_m['max_dd']:>7.2%} Sortino={single_m['sortino']:>7.3f}")

    # ================================================================
    # FINAL RANKING
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL RANKING (by Sortino, then CAGR)")
    print("=" * 70)

    ranked = sorted(results.items(),
                    key=lambda x: (x[1]['metrics']['sortino'], x[1]['metrics']['cagr']),
                    reverse=True)

    print(f"\n{'Rank':<5} {'Method':<32} {'N':>3} {'CAGR':>8} {'MaxDD':>8} {'Sortino':>8} {'Calmar':>8} {'Final':>8}")
    print("-" * 85)
    for i, (tag, data) in enumerate(ranked[:20], 1):
        m = data['metrics']
        n = len(data['keys'])
        calmar = m.get('calmar', 0)
        if abs(calmar) > 100:
            calmar_s = ">100"
        else:
            calmar_s = f"{calmar:.1f}"
        print(f"{i:<5} {tag:<32} {n:>3} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
              f"{m['sortino']:>8.3f} {calmar_s:>8} {m['final']:>8.2f}")

    # Save best result
    best_tag, best_data = ranked[0]
    print(f"\n{'=' * 70}")
    print(f"WINNER: {best_tag}")
    print(f"{'=' * 70}")
    m = best_data['metrics']
    print(f"  CAGR:    {m['cagr']:.2%}")
    print(f"  MaxDD:   {m['max_dd']:.2%}")
    print(f"  Sortino: {m['sortino']:.3f}")
    print(f"  Calmar:  {m.get('calmar', 0):.2f}")
    print(f"  Final:   {m['final']:.2f}")
    print(f"  Strategies ({len(best_data['keys'])}):")
    for k in best_data['keys']:
        s = next((s for s in deduped if s['key'] == k), None)
        if s:
            w_info = ""
            if isinstance(best_data['weights'], dict):
                w_info = f" w={best_data['weights'].get(k, 0):.1%}"
            print(f"    {k:<22} {s['direction']:<6} CAGR={s['cagr']:.1%} PF={s['profit_factor']:.2f}{w_info}")

    # Save full results
    output = {
        'winner': best_tag,
        'winner_metrics': best_data['metrics'],
        'winner_keys': best_data['keys'],
        'winner_weights': best_data['weights'],
        'all_results': {tag: {'n': len(d['keys']), **d['metrics']}
                        for tag, d in ranked[:20]},
        'buy_and_hold': bh_m,
        'deduplication_removed': list(removed),
        'n_strategies_after_dedup': len(deduped),
    }

    output_path = Path('reports/paper_v2/ensemble_search.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    run_optimization()
