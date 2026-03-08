#!/usr/bin/env python3
"""Analyze CMA-ES results across all experiments."""
import json
from pathlib import Path


def analyze_experiment(results_dir: Path) -> dict:
    """Analyze a single experiment's CMA-ES results."""
    cmaes_path = results_dir / 'cmaes_results.json'
    if not cmaes_path.exists():
        return None

    with open(cmaes_path) as f:
        results = json.load(f)

    # Get asset from metadata
    meta_path = results_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        asset = meta.get('symbol', 'BTC/USDT').split('/')[0]
    else:
        asset = 'BTC'

    # Extract seed from dir name
    name = results_dir.name
    seed = 'unknown'
    for part in name.split('_'):
        if part.startswith('seed'):
            seed = part.replace('seed', '')
            break

    n_total = len(results)
    n_better = 0
    n_worse = 0
    n_same = 0
    total_orig_cagr = 0
    total_opt_cagr = 0

    details = []
    for r in results:
        idx = r['strategy_index']
        ots_o = r.get('ots_original', {}).get('metrics', {}).get('cagr', None)
        ots_n = r.get('ots_optimized', {}).get('metrics', {}).get('cagr', None)
        pbo_o = r['validation_original']['pbo']
        pbo_n = r['validation_optimized']['pbo']

        if ots_o is not None and ots_n is not None:
            total_orig_cagr += ots_o
            total_opt_cagr += ots_n
            if ots_n > ots_o + 0.001:
                n_better += 1
                status = 'BETTER'
            elif ots_n < ots_o - 0.001:
                n_worse += 1
                status = 'WORSE'
            else:
                n_same += 1
                status = 'SAME'
        else:
            status = '?'

        details.append({
            'idx': idx,
            'ots_orig': ots_o,
            'ots_opt': ots_n,
            'pbo_orig': pbo_o,
            'pbo_opt': pbo_n,
            'status': status,
        })

    return {
        'asset': asset,
        'seed': seed,
        'n_total': n_total,
        'n_better': n_better,
        'n_worse': n_worse,
        'n_same': n_same,
        'total_orig_cagr': total_orig_cagr,
        'total_opt_cagr': total_opt_cagr,
        'details': details,
    }


def main():
    experiments = [
        'experiment_BNB_USDT_seed42_20260307_211637',
        'experiment_BNB_USDT_seed123_20260308_010956',
        'experiment_BNB_USDT_seed777_20260308_011315',
        'experiment_ETH_USDT_seed42_20260307_202814',
        'experiment_ETH_USDT_seed777_20260308_010426',
        'experiment_seed123_20260307_193201',
        'experiment_seed777_20260307_195112',
    ]

    all_results = []
    for exp_name in experiments:
        results_dir = Path('results') / exp_name
        if not results_dir.exists():
            continue
        result = analyze_experiment(results_dir)
        if result:
            all_results.append(result)

    # Print summary
    print(f"\n{'='*90}")
    print("CMA-ES v5.1 COMPREHENSIVE RESULTS (multi-seed consensus + trade stability gate)")
    print(f"{'='*90}")

    grand_better = grand_worse = grand_same = grand_total = 0
    grand_orig_cagr = grand_opt_cagr = 0.0

    print(f"\n{'Asset':>5} {'Seed':>6} {'N':>3} {'BETTER':>7} {'SAME':>5} {'WORSE':>6} "
          f"{'OrigCAGR':>10} {'OptCAGR':>10} {'Delta':>8}")
    print(f"{'-'*70}")

    for r in all_results:
        delta = r['total_opt_cagr'] - r['total_orig_cagr']
        print(f"{r['asset']:>5} {r['seed']:>6} {r['n_total']:>3} "
              f"{r['n_better']:>7} {r['n_same']:>5} {r['n_worse']:>6} "
              f"{r['total_orig_cagr']:>+10.4f} {r['total_opt_cagr']:>+10.4f} "
              f"{delta:>+8.4f}")
        grand_better += r['n_better']
        grand_worse += r['n_worse']
        grand_same += r['n_same']
        grand_total += r['n_total']
        grand_orig_cagr += r['total_orig_cagr']
        grand_opt_cagr += r['total_opt_cagr']

    print(f"{'-'*70}")
    grand_delta = grand_opt_cagr - grand_orig_cagr
    print(f"{'TOTAL':>5} {'':>6} {grand_total:>3} "
          f"{grand_better:>7} {grand_same:>5} {grand_worse:>6} "
          f"{grand_orig_cagr:>+10.4f} {grand_opt_cagr:>+10.4f} "
          f"{grand_delta:>+8.4f}")

    pct_better = grand_better / max(grand_total, 1) * 100
    pct_portfolio = grand_delta / max(abs(grand_orig_cagr), 0.01) * 100
    print(f"\nStrategies improved: {grand_better}/{grand_total} ({pct_better:.0f}%)")
    print(f"Portfolio CAGR delta: {grand_delta:+.4f} ({pct_portfolio:+.1f}%)")

    # Print top improvements
    print(f"\n{'='*90}")
    print("TOP OTS IMPROVEMENTS")
    print(f"{'='*90}")

    all_details = []
    for r in all_results:
        for d in r['details']:
            d['asset'] = r['asset']
            d['seed'] = r['seed']
            all_details.append(d)

    improvements = [d for d in all_details
                    if d['status'] == 'BETTER' and d['ots_opt'] is not None]
    improvements.sort(key=lambda x: (x['ots_opt'] - x['ots_orig']), reverse=True)

    print(f"\n{'Asset':>5} {'Seed':>6} {'#':>3} {'Orig CAGR':>10} {'Opt CAGR':>10} "
          f"{'Delta':>8} {'PBO_o':>6} {'PBO_n':>6}")
    for d in improvements[:15]:
        delta = d['ots_opt'] - d['ots_orig']
        print(f"{d['asset']:>5} {d['seed']:>6} {d['idx']:>3} "
              f"{d['ots_orig']:>+10.4f} {d['ots_opt']:>+10.4f} {delta:>+8.4f} "
              f"{d['pbo_orig']:>6.3f} {d['pbo_opt']:>6.3f}")


if __name__ == '__main__':
    main()
