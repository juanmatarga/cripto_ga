#!/usr/bin/env python3
"""
OTS Evaluation for Top 15 Walk-Forward Strategies.

Loads the same strategies used in the stability test, selects the 15 that showed
the best walk-forward consistency, and runs a one-shot backtest on the sacred
OTS holdout period (2025-06-01 to 2025-11-21).

This is a FINAL evaluation — no iteration, no tuning.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

OTS_START = '2025-06-01'
OTS_END = '2025-11-21'
CONFIG_PATH = 'config_v2.yaml'

# The top 15 strategies from the walk-forward stability analysis
# (indices into the list returned by load_existing_strategies('BTC/USDT'))
TARGET_INDICES = [87, 109, 34, 25, 117, 131, 88, 23, 121, 32, 169, 165, 178, 1, 9]


def main():
    t0 = time.time()

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # ---- Load strategies (same as run_walkforward.py stability mode) ----
    from run_walkforward import load_existing_strategies
    all_strategies = load_existing_strategies('BTC/USDT')
    logger.info(f"Loaded {len(all_strategies)} validated BTC strategies")

    if len(all_strategies) == 0:
        logger.error("No strategies found!")
        return

    # Verify we have enough strategies
    max_idx = max(TARGET_INDICES)
    if max_idx >= len(all_strategies):
        logger.error(f"Target index {max_idx} >= total strategies {len(all_strategies)}")
        return

    # Select target strategies
    selected = [(idx, all_strategies[idx]) for idx in TARGET_INDICES]
    logger.info(f"Selected {len(selected)} strategies for OTS evaluation")

    # ---- Load OTS data ----
    from data.loader import load_data

    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ots_data = df[(df.index >= pd.Timestamp(OTS_START)) &
                  (df.index <= pd.Timestamp(OTS_END))]

    logger.info(f"OTS data: {len(ots_data)} bars, "
                f"{ots_data.index.min()} to {ots_data.index.max()}")

    if len(ots_data) < 100:
        logger.error("Insufficient OTS data!")
        return

    # ---- Run OTS backtests ----
    from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
    from backtest.metrics import calculate_all_metrics
    from data.multi_timeframe import prepare_multi_tf_data

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    # Pre-compute multi-TF data once
    tf_data = prepare_multi_tf_data(ots_data)

    results = []

    for idx, strategy in selected:
        logger.info(f"\n--- Strategy {idx}: {strategy.direction} ---")
        logger.info(f"    {strategy.expression_raw[:100]}")

        try:
            equity, trades = _run_single_window(
                strategy, ots_data, costs_config, atr_period,
                tf_data=tf_data,
            )
            metrics = calculate_all_metrics(equity, BARS_PER_YEAR_15M)

            n_trades = len(trades)
            winning = sum(1 for t in trades if t['pnl_pct'] > 0)
            win_rate = winning / n_trades if n_trades > 0 else 0.0

            winning_pnl = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
            losing_pnl = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
            pf = winning_pnl / max(losing_pnl, 1e-10)

            result = {
                'strategy_index': idx,
                'expression': strategy.expression_raw,
                'direction': strategy.direction,
                'cagr': float(metrics.get('cagr', 0)),
                'sortino': float(metrics.get('sortino', 0)),
                'max_dd': float(metrics.get('max_dd', 0)),
                'n_trades': n_trades,
                'win_rate': win_rate,
                'profit_factor': pf,
                'equity_final': float(equity.iloc[-1]),
                'tp_sl': f"TP={strategy.tp_atr_mult} SL={strategy.sl_atr_mult}",
                'trail': strategy.trail_atr_mult,
            }
            results.append(result)

            logger.info(f"    CAGR={result['cagr']:+.2%} Sortino={result['sortino']:.3f} "
                        f"MaxDD={result['max_dd']:.1%} Trades={n_trades} "
                        f"WR={win_rate:.1%} PF={pf:.2f}")

        except Exception as e:
            logger.error(f"    OTS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'strategy_index': idx,
                'expression': strategy.expression_raw,
                'direction': strategy.direction,
                'error': str(e),
            })

    # ---- Print results table ----
    print(f"\n{'='*120}")
    print(f"  OTS RESULTS — TOP 15 WALK-FORWARD STRATEGIES")
    print(f"  Period: {OTS_START} to {OTS_END}")
    print(f"{'='*120}")

    # Sort by CAGR (descending)
    valid_results = [r for r in results if 'cagr' in r]
    valid_results.sort(key=lambda r: r['cagr'], reverse=True)

    print(f"\n{'#':>3} {'Idx':>4} {'Dir':>5} {'CAGR':>8} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Trades':>7} {'WinRate':>8} {'PF':>6} "
          f"{'EqFinal':>9} {'Pass':>5}  Expression")
    print(f"{'-'*120}")

    n_positive = 0
    n_pass = 0
    for rank, r in enumerate(valid_results, 1):
        cagr_val = r['cagr']
        sortino = r['sortino']
        max_dd = r['max_dd']
        n_trades = r['n_trades']
        wr = r['win_rate']
        pf = r['profit_factor']
        eq = r['equity_final']

        # OTS pass criteria: positive CAGR, reasonable drawdown, enough trades
        is_positive = cagr_val > 0
        passes = is_positive and abs(max_dd) < 0.30 and n_trades >= 5
        pass_str = 'YES' if passes else 'no'

        if is_positive:
            n_positive += 1
        if passes:
            n_pass += 1

        # Truncate expression for display
        expr = r['expression'][:55]

        print(f"{rank:>3} {r['strategy_index']:>4} {r['direction']:>5} "
              f"{cagr_val:>+7.1%} {sortino:>8.3f} "
              f"{max_dd:>6.1%} {n_trades:>7} {wr:>7.1%} {pf:>6.2f} "
              f"{eq:>9.2f} {pass_str:>5}  {expr}")

    # Errors
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nFailed strategies:")
        for r in error_results:
            print(f"  [{r['strategy_index']}] {r['error']}")

    # Summary
    print(f"\n{'='*120}")
    print(f"  SUMMARY")
    print(f"{'='*120}")
    print(f"  Strategies evaluated:  {len(valid_results)}/{len(TARGET_INDICES)}")
    print(f"  Positive CAGR (OTS):   {n_positive}/{len(valid_results)}")
    print(f"  Pass OTS criteria:     {n_pass}/{len(valid_results)} "
          f"(CAGR>0, DD<30%, trades>=5)")

    if valid_results:
        cagrs = [r['cagr'] for r in valid_results]
        sortinos = [r['sortino'] for r in valid_results]
        print(f"\n  CAGR range:    {min(cagrs):+.1%} to {max(cagrs):+.1%}")
        print(f"  Mean CAGR:     {np.mean(cagrs):+.1%}")
        print(f"  Median CAGR:   {np.median(cagrs):+.1%}")
        print(f"  Mean Sortino:  {np.mean(sortinos):.3f}")

    # B&H for reference
    bh_return = (ots_data['Close'].iloc[-1] / ots_data['Close'].iloc[0]) - 1
    print(f"\n  Buy & Hold:    {bh_return:+.1%} (BTC {OTS_START} to {OTS_END})")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*120}")

    # Save results
    output_path = Path('results') / 'ots_walkforward_top15.json'
    with open(output_path, 'w') as f:
        json.dump({
            'ots_period': f"{OTS_START} to {OTS_END}",
            'n_strategies': len(valid_results),
            'n_positive': n_positive,
            'n_pass': n_pass,
            'bh_return': float(bh_return),
            'results': valid_results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == '__main__':
    main()
