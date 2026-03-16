#!/usr/bin/env python3
"""
OTS (Out-of-Time-Sample) Evaluation for NSGA-II Pareto Front Strategies.

Loads pareto_front.json from each NSGA-II run directory, deduplicates strategies,
and runs a one-shot backtest on the sacred OTS holdout (2025-06-01 to 2025-11-21).

Usage:
    python3 run_ots_nsga2.py                    # All symbols
    python3 run_ots_nsga2.py --symbol BTC       # Single symbol
    python3 run_ots_nsga2.py --symbol ETH BNB   # Multiple symbols
"""

import argparse
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
RESULTS_DIR = Path('results')

# Symbol mapping: short name -> ccxt symbol
SYMBOL_MAP = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'BNB': 'BNB/USDT',
}


def find_nsga2_dirs(symbol_short: str) -> list:
    """Find all NSGA-II result directories for a given symbol."""
    pattern = f"nsga2_{symbol_short}_*"
    dirs = sorted(RESULTS_DIR.glob(pattern))
    # Filter to only those with pareto_front.json
    return [d for d in dirs if (d / 'pareto_front.json').exists()]


def load_pareto_strategies(result_dir: Path) -> list:
    """Load strategies from a pareto_front.json file."""
    pf_path = result_dir / 'pareto_front.json'
    with open(pf_path) as f:
        strategies_raw = json.load(f)
    return strategies_raw


def dedup_key(s: dict) -> tuple:
    """Create a deduplication key from conditions + exits."""
    conditions = tuple(sorted(s.get('conditions', [])))
    tp = s.get('tp_atr_mult', 0)
    sl = s.get('sl_atr_mult', 0)
    trail = s.get('trail_atr_mult', 0)
    direction = s.get('direction', '')
    return (direction, conditions, tp, sl, trail)


def reconstruct_strategy(s_dict: dict):
    """Reconstruct a Strategy object from a pareto_front.json entry."""
    from grammar.mapper import decode, _parse_conditions
    from strategy.phenotype import Strategy, Condition

    # First try decoding from genome
    genome = s_dict.get('genome', [])
    strategy = decode(genome)

    if strategy is not None:
        return strategy

    # Fallback: reconstruct directly from the stored fields
    logger.warning(f"Genome decode failed, reconstructing from stored fields")
    conditions = []
    for cond_str in s_dict.get('conditions', []):
        # Parse "RSI(close, 14) > 30" into Condition
        for comp in ['CROSSES_ABOVE', 'CROSSES_BELOW', '>', '<']:
            if f' {comp} ' in cond_str:
                parts = cond_str.split(f' {comp} ', 1)
                if len(parts) == 2:
                    conditions.append(Condition(
                        left=parts[0].strip(),
                        comparator=comp,
                        right=parts[1].strip(),
                    ))
                    break

    if not conditions:
        return None

    return Strategy(
        genome=genome,
        direction=s_dict['direction'],
        conditions=conditions,
        logic=s_dict.get('logic', 'c0'),
        tp_atr_mult=s_dict.get('tp_atr_mult', 0.0),
        sl_atr_mult=s_dict.get('sl_atr_mult', 1.0),
        trail_atr_mult=s_dict.get('trail_atr_mult', 0.0),
        expression_raw=s_dict.get('expression_raw', ''),
        n_nodes=s_dict.get('n_nodes', len(conditions)),
        codons_used=s_dict.get('codons_used', 0),
        wrapping_count=s_dict.get('wrapping_count', 0),
    )


def load_ots_data(symbol: str) -> pd.DataFrame:
    """Load OHLCV data for the OTS period."""
    from data.loader import load_data

    config = {
        'data': {
            'symbol': symbol,
            'market_type': 'future',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
        }
    }

    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter to OTS period
    ots = df[(df.index >= pd.Timestamp(OTS_START)) &
             (df.index <= pd.Timestamp(OTS_END))]

    logger.info(f"OTS data for {symbol}: {len(ots)} bars, "
                f"{ots.index.min()} to {ots.index.max()}")
    return ots


def evaluate_ots(strategies: list, ots_data: pd.DataFrame,
                 costs_config: dict, atr_period: int) -> list:
    """Run OTS backtest for each strategy."""
    from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
    from backtest.metrics import calculate_all_metrics
    from data.multi_timeframe import prepare_multi_tf_data

    # Pre-compute multi-TF data once
    tf_data = prepare_multi_tf_data(ots_data)

    results = []

    for i, (s_dict, strategy) in enumerate(strategies):
        label = f"[{i+1}/{len(strategies)}]"
        conds_str = ' ; '.join(s_dict.get('conditions', []))
        logger.info(f"{label} {s_dict['direction']} | {conds_str}")

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
                'direction': s_dict['direction'],
                'conditions': s_dict.get('conditions', []),
                'logic': s_dict.get('logic', ''),
                'tp_atr_mult': s_dict.get('tp_atr_mult', 0),
                'sl_atr_mult': s_dict.get('sl_atr_mult', 0),
                'trail_atr_mult': s_dict.get('trail_atr_mult', 0),
                'expression_raw': s_dict.get('expression_raw', ''),
                'n_nodes': s_dict.get('n_nodes', 0),
                'evo_objectives': s_dict.get('objectives', []),
                'evo_stability': s_dict.get('stability', 0),
                'evo_n_trades': s_dict.get('n_trades', 0),
                # OTS results
                'ots_cagr': float(metrics.get('cagr', 0)),
                'ots_sortino': float(metrics.get('sortino', 0)),
                'ots_max_dd': float(metrics.get('max_dd', 0)),
                'ots_n_trades': n_trades,
                'ots_win_rate': win_rate,
                'ots_profit_factor': pf,
                'ots_equity_final': float(equity.iloc[-1]),
            }
            results.append(result)

            logger.info(f"  -> CAGR={result['ots_cagr']:+.2%} "
                        f"Sortino={result['ots_sortino']:.3f} "
                        f"MaxDD={result['ots_max_dd']:.1%} "
                        f"Trades={n_trades} WR={win_rate:.1%} PF={pf:.2f}")

        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'direction': s_dict['direction'],
                'conditions': s_dict.get('conditions', []),
                'expression_raw': s_dict.get('expression_raw', ''),
                'error': str(e),
            })

    return results


def print_results_table(symbol_short: str, results: list, bh_return: float):
    """Print a formatted results table for one symbol."""
    valid = [r for r in results if 'ots_cagr' in r]
    valid.sort(key=lambda r: r['ots_cagr'], reverse=True)
    errors = [r for r in results if 'error' in r]

    print(f"\n{'='*140}")
    print(f"  OTS RESULTS — NSGA-II PARETO FRONT — {symbol_short}/USDT")
    print(f"  Period: {OTS_START} to {OTS_END}")
    print(f"{'='*140}")

    if not valid:
        print("  No valid results.")
        return 0, 0

    header = (f"{'#':>3} {'Dir':>5} {'CAGR':>8} {'Sortino':>8} {'MaxDD':>7} "
              f"{'Trades':>7} {'WinRate':>8} {'PF':>6} {'EqFin':>8} {'Pass':>5}  "
              f"{'TP':>4} {'SL':>5} {'TRL':>5}  Conditions")
    print(f"\n{header}")
    print(f"{'-'*140}")

    n_positive = 0
    n_pass = 0

    for rank, r in enumerate(valid, 1):
        cagr_val = r['ots_cagr']
        sortino = r['ots_sortino']
        max_dd = r['ots_max_dd']
        n_trades = r['ots_n_trades']
        wr = r['ots_win_rate']
        pf = r['ots_profit_factor']
        eq = r['ots_equity_final']

        tp = r.get('tp_atr_mult', 0)
        sl = r.get('sl_atr_mult', 0)
        trail = r.get('trail_atr_mult', 0)

        # OTS pass criteria: positive CAGR, reasonable drawdown, enough trades
        is_positive = cagr_val > 0
        passes = is_positive and abs(max_dd) < 0.30 and n_trades >= 5
        pass_str = 'YES' if passes else 'no'

        if is_positive:
            n_positive += 1
        if passes:
            n_pass += 1

        # Conditions string
        conds = ' ; '.join(r.get('conditions', []))
        if len(conds) > 55:
            conds = conds[:52] + '...'

        tp_str = f"{tp:.1f}" if tp > 0 else "-"
        trail_str = f"{trail:.1f}" if trail > 0 else "-"

        print(f"{rank:>3} {r['direction']:>5} "
              f"{cagr_val:>+7.1%} {sortino:>8.3f} "
              f"{max_dd:>6.1%} {n_trades:>7} {wr:>7.1%} {pf:>6.2f} "
              f"{eq:>8.2f} {pass_str:>5}  "
              f"{tp_str:>4} {sl:>5.2f} {trail_str:>5}  {conds}")

    if errors:
        print(f"\n  Failed strategies: {len(errors)}")
        for r in errors:
            print(f"    {r.get('direction', '?')}: {r.get('error', 'unknown')}")

    print(f"\n{'-'*140}")
    print(f"  {symbol_short} Summary:")
    print(f"    Total evaluated:     {len(valid)}")
    print(f"    Positive CAGR:       {n_positive}/{len(valid)} "
          f"({n_positive/len(valid)*100:.0f}%)")
    print(f"    Pass OTS criteria:   {n_pass}/{len(valid)} "
          f"(CAGR>0, DD<30%, trades>=5)")

    if valid:
        cagrs = [r['ots_cagr'] for r in valid]
        sortinos = [r['ots_sortino'] for r in valid]
        print(f"    CAGR range:          {min(cagrs):+.1%} to {max(cagrs):+.1%}")
        print(f"    Mean CAGR:           {np.mean(cagrs):+.1%}")
        print(f"    Median CAGR:         {np.median(cagrs):+.1%}")
        print(f"    Mean Sortino:        {np.mean(sortinos):.3f}")

    print(f"    Buy & Hold:          {bh_return:+.1%}")

    return n_positive, n_pass


def main():
    parser = argparse.ArgumentParser(description='OTS evaluation for NSGA-II Pareto front strategies')
    parser.add_argument('--symbol', nargs='*', default=None,
                        help='Symbol(s) to evaluate: BTC, ETH, BNB. Default: all found.')
    args = parser.parse_args()

    t0 = time.time()

    # Determine which symbols to process
    if args.symbol:
        symbols = [s.upper() for s in args.symbol]
    else:
        # Auto-detect from results directories
        symbols = []
        for short in ['BTC', 'ETH', 'BNB']:
            if find_nsga2_dirs(short):
                symbols.append(short)

    if not symbols:
        logger.error("No NSGA-II result directories found!")
        return

    logger.info(f"Symbols to evaluate: {symbols}")

    costs_config = {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    }
    atr_period = 14

    all_results = {}

    for symbol_short in symbols:
        ccxt_symbol = SYMBOL_MAP[symbol_short]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol_short} ({ccxt_symbol})")
        logger.info(f"{'='*60}")

        # Find all NSGA-II runs for this symbol
        nsga2_dirs = find_nsga2_dirs(symbol_short)
        if not nsga2_dirs:
            logger.warning(f"No NSGA-II results found for {symbol_short}")
            continue

        logger.info(f"Found {len(nsga2_dirs)} NSGA-II runs: {[d.name for d in nsga2_dirs]}")

        # Load and deduplicate strategies across all runs
        seen_keys = set()
        unique_strategies = []  # list of (raw_dict, Strategy)

        for d in nsga2_dirs:
            raw_strategies = load_pareto_strategies(d)
            logger.info(f"  {d.name}: {len(raw_strategies)} Pareto front strategies")

            for s_dict in raw_strategies:
                key = dedup_key(s_dict)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                strategy = reconstruct_strategy(s_dict)
                if strategy is None:
                    logger.warning(f"  Could not reconstruct strategy: "
                                   f"{s_dict.get('expression_raw', '?')[:60]}")
                    continue

                unique_strategies.append((s_dict, strategy))

        logger.info(f"Total unique strategies for {symbol_short}: {len(unique_strategies)}")

        if not unique_strategies:
            logger.warning(f"No valid strategies for {symbol_short}")
            continue

        # Load OTS data
        ots_data = load_ots_data(ccxt_symbol)
        if len(ots_data) < 100:
            logger.error(f"Insufficient OTS data for {symbol_short}: {len(ots_data)} bars")
            continue

        # Buy & Hold reference
        bh_return = (ots_data['Close'].iloc[-1] / ots_data['Close'].iloc[0]) - 1

        # Run OTS backtests
        results = evaluate_ots(unique_strategies, ots_data, costs_config, atr_period)

        # Print table
        n_pos, n_pass = print_results_table(symbol_short, results, bh_return)

        all_results[symbol_short] = {
            'symbol': ccxt_symbol,
            'ots_period': f"{OTS_START} to {OTS_END}",
            'n_runs': len(nsga2_dirs),
            'run_dirs': [d.name for d in nsga2_dirs],
            'n_total_pareto': sum(len(load_pareto_strategies(d)) for d in nsga2_dirs),
            'n_unique': len(unique_strategies),
            'n_positive': n_pos,
            'n_pass': n_pass,
            'bh_return': float(bh_return),
            'results': [r for r in results],
        }

    # ---- Grand summary ----
    elapsed = time.time() - t0
    print(f"\n\n{'='*140}")
    print(f"  GRAND SUMMARY — NSGA-II OTS EVALUATION")
    print(f"{'='*140}")

    total_unique = 0
    total_positive = 0
    total_pass = 0

    for sym, data in all_results.items():
        total_unique += data['n_unique']
        total_positive += data['n_positive']
        total_pass += data['n_pass']
        valid = [r for r in data['results'] if 'ots_cagr' in r]
        mean_cagr = np.mean([r['ots_cagr'] for r in valid]) if valid else 0
        print(f"  {sym:>4}: {data['n_unique']:>3} unique strategies, "
              f"{data['n_positive']:>2} positive ({data['n_positive']/max(data['n_unique'],1)*100:.0f}%), "
              f"{data['n_pass']:>2} pass OTS, "
              f"mean CAGR={mean_cagr:+.1%}, "
              f"B&H={data['bh_return']:+.1%}")

    print(f"\n  Total: {total_unique} unique, "
          f"{total_positive} positive ({total_positive/max(total_unique,1)*100:.0f}%), "
          f"{total_pass} pass OTS criteria")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*140}")

    # Save results
    output_path = RESULTS_DIR / 'ots_nsga2_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == '__main__':
    main()
