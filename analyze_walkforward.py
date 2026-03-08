#!/usr/bin/env python3
"""
Analyze Walk-Forward Results.

Reads walk-forward results JSON and produces detailed analysis:
  - Per-window breakdown with regime classification
  - Regime-filtered ensemble performance
  - Process consistency (does evolution reliably find alpha?)
  - Decay analysis across time
  - Comparison vs buy & hold

Usage:
    python3 analyze_walkforward.py results/walkforward_evolve_BTC_USDT_seed42_*/results.json
    python3 analyze_walkforward.py results/walkforward_stability_*/results.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def analyze_results(results_path: str):
    with open(results_path) as f:
        data = json.load(f)

    for symbol, agg in data.items():
        if not isinstance(agg, dict) or 'per_window' not in agg:
            print(f"Skipping {symbol}: no per_window data")
            continue

        windows = agg['per_window']
        if not windows:
            continue

        print(f"\n{'='*90}")
        print(f"  WALK-FORWARD ANALYSIS: {symbol}")
        print(f"{'='*90}")

        # ---- Per-window detail ----
        print(f"\n{'W':>3} {'Test Period':>26} {'Regime':>9} {'B&H':>8} "
              f"{'Ens.CAGR':>9} {'Sortino':>8} {'MaxDD':>7} "
              f"{'Trades':>7} {'Strats':>6}")
        print(f"{'-'*90}")

        total_raw = 1.0
        total_bh = 1.0
        n_positive = 0
        n_beat_bh = 0

        for pw in windows:
            cagr = pw.get('ensemble_cagr', 0)
            sortino = pw.get('ensemble_sortino', 0)
            maxdd = pw.get('ensemble_max_dd', 0)
            trades = pw.get('ensemble_n_trades', 0)
            bh = pw.get('test_bh_return', 0)
            regime = pw.get('test_regime', '?')
            n_tested = pw.get('n_tested', 0)

            # Compound 3-month return
            period_ret = (1 + cagr) ** (3/12) - 1
            total_raw *= (1 + period_ret)
            total_bh *= (1 + bh)

            if cagr > 0:
                n_positive += 1
            if cagr > bh:
                n_beat_bh += 1

            alpha_marker = '*' if cagr > bh else ' '
            print(f"W{pw['window_id']:>2} {pw['test_period']:>26} "
                  f"{regime:>9} {bh:>+7.1%} "
                  f"{cagr:>+8.1%} {sortino:>8.2f} {maxdd:>6.1%} "
                  f"{trades:>7} {n_tested:>6}{alpha_marker}")

        n_windows = len(windows)
        print(f"{'-'*90}")
        print(f"{'TOTAL':>32} {'':>9} {total_bh-1:>+7.1%} "
              f"{total_raw-1:>+8.1%}")

        # ---- Summary ----
        print(f"\n--- Summary ---")
        print(f"  Windows:     {n_windows}")
        print(f"  Positive:    {n_positive}/{n_windows} ({n_positive/n_windows:.0%})")
        print(f"  Beat B&H:    {n_beat_bh}/{n_windows} ({n_beat_bh/n_windows:.0%})")
        print(f"  Compounded:  {total_raw-1:+.1%} vs B&H {total_bh-1:+.1%}")
        excess = total_raw - total_bh
        print(f"  Excess:      {excess:+.4f} ({excess/(total_bh)*100:+.1f}% relative)")

        # ---- Regime breakdown ----
        print(f"\n--- By Regime ---")
        regime_data = {}
        for pw in windows:
            r = pw.get('test_regime', '?')
            if r not in regime_data:
                regime_data[r] = []
            regime_data[r].append(pw.get('ensemble_cagr', 0))

        for regime in ['bull', 'bear', 'sideways']:
            if regime in regime_data:
                cagrs = regime_data[regime]
                pos = sum(1 for c in cagrs if c > 0)
                print(f"  {regime:>8s}: {len(cagrs)} windows, "
                      f"mean CAGR={np.mean(cagrs):+.1%}, "
                      f"{pos}/{len(cagrs)} positive")

        # ---- Decay analysis ----
        if n_windows >= 4:
            half = n_windows // 2
            first_cagrs = [pw.get('ensemble_cagr', 0) for pw in windows[:half]]
            second_cagrs = [pw.get('ensemble_cagr', 0) for pw in windows[half:]]
            print(f"\n--- Decay Analysis ---")
            print(f"  First half:  mean CAGR={np.mean(first_cagrs):+.1%}")
            print(f"  Second half: mean CAGR={np.mean(second_cagrs):+.1%}")
            if np.mean(second_cagrs) < np.mean(first_cagrs) * 0.5:
                print(f"  WARNING: Performance decay detected!")
            else:
                print(f"  No significant decay.")

        # ---- Per-strategy analysis (stability mode) ----
        has_strategies = any(pw.get('strategy_results') for pw in windows)
        if has_strategies:
            print(f"\n--- Per-Strategy Consistency ---")
            # Collect each strategy's CAGR across windows
            strat_windows = {}
            for pw in windows:
                for sr in pw.get('strategy_results', []):
                    idx = sr.get('strategy_index', -1)
                    expr = sr.get('expression', '')[:60]
                    dirn = sr.get('direction', '?')
                    key = f"{idx}_{dirn}"
                    if key not in strat_windows:
                        strat_windows[key] = {
                            'idx': idx, 'dir': dirn, 'expr': expr,
                            'cagrs': [], 'trades': []
                        }
                    strat_windows[key]['cagrs'].append(sr.get('cagr', 0))
                    strat_windows[key]['trades'].append(sr.get('n_trades', 0))

            print(f"\n{'#':>3} {'Dir':>5} {'Wins':>4}/{n_windows} "
                  f"{'MeanCAGR':>9} {'StdCAGR':>8} {'MinCAGR':>8} "
                  f"{'Trades':>7} {'Expression':>40}")
            print(f"{'-'*90}")

            # Sort by consistency (n positive windows)
            sorted_strats = sorted(
                strat_windows.values(),
                key=lambda s: sum(1 for c in s['cagrs'] if c > 0),
                reverse=True
            )

            for s in sorted_strats[:20]:
                n_pos = sum(1 for c in s['cagrs'] if c > 0)
                mean_c = np.mean(s['cagrs'])
                std_c = np.std(s['cagrs'])
                min_c = np.min(s['cagrs'])
                tot_trades = sum(s['trades'])
                print(f"{s['idx']:>3} {s['dir']:>5} {n_pos:>4}/{len(s['cagrs'])} "
                      f"{mean_c:>+8.1%} {std_c:>7.1%} {min_c:>+7.1%} "
                      f"{tot_trades:>7} {s['expr']:>40}")


def main():
    if len(sys.argv) < 2:
        # Auto-find latest results
        results_dirs = sorted(Path('results').glob('walkforward_*'))
        if not results_dirs:
            print("No walk-forward results found. Run run_walkforward.py first.")
            return
        results_path = results_dirs[-1] / 'results.json'
        print(f"Using latest: {results_path}")
    else:
        results_path = sys.argv[1]

    analyze_results(results_path)


if __name__ == '__main__':
    main()
