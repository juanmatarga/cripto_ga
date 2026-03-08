#!/usr/bin/env python3
"""
Walk-Forward Analysis Runner.

Usage:
    # Phase 1: Strategy stability (fast diagnostic, ~2 min)
    python3 run_walkforward.py stability --symbol BTC/USDT

    # Phase 2: Full walk-forward with re-evolution (~1h per asset)
    python3 run_walkforward.py evolve --symbol BTC/USDT --seed 42

    # Phase 3: CMA-ES walk-forward (re-optimize params per window, ~30 min)
    python3 run_walkforward.py cmaes --symbol BTC/USDT --seed 42

    # All assets
    python3 run_walkforward.py stability --symbol ALL
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.loader import load_data
from grammar.mapper import decode
from validation.walk_forward import (
    WalkForwardEngine, WFConfig, generate_windows,
    aggregate_wf_results, print_wf_summary
)

logger = logging.getLogger('cripto_ga')


def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'walkforward_{timestamp}.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_full_data(symbol: str, config: dict) -> pd.DataFrame:
    """Load full data range (including OTS — walk-forward uses its own splits)."""
    # Override symbol in config
    cfg = dict(config)
    cfg['data'] = dict(cfg.get('data', {}))
    cfg['data']['symbol'] = symbol

    df = load_data(cfg)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info(f"Loaded {symbol}: {len(df)} bars, "
                f"{df.index.min()} to {df.index.max()}")
    return df


def load_existing_strategies(symbol: str) -> list:
    """Load validated strategies from experiment results for a symbol."""
    results_dir = Path('results')
    safe = symbol.replace('/', '_').replace(':', '')

    # Find all experiment dirs for this symbol
    if 'BTC' in symbol:
        patterns = ['experiment_seed*']
    else:
        base = safe.split('_')[0]  # 'ETH', 'BNB', etc.
        patterns = [f'experiment_{base}_USDT_seed*']

    strategies = []
    seen_expressions = set()

    for pattern in patterns:
        for exp_dir in sorted(results_dir.glob(pattern)):
            top_path = exp_dir / 'top_strategies.json'
            val_path = exp_dir / 'validation.json'

            if not top_path.exists() or not val_path.exists():
                continue

            with open(top_path) as f:
                top_strats = json.load(f)
            with open(val_path) as f:
                validation = json.load(f)

            # Only take validated strategies
            passed = {v['strategy_index'] for v in validation
                      if v.get('passes_all', False)}

            for idx in passed:
                if idx >= len(top_strats):
                    continue
                sd = top_strats[idx]
                genome = sd['genome']
                strategy = decode(genome)
                if strategy is None:
                    continue

                # Dedup by expression
                expr = strategy.expression_raw
                if expr in seen_expressions:
                    continue
                seen_expressions.add(expr)
                strategies.append(strategy)

    logger.info(f"Loaded {len(strategies)} unique validated strategies for {symbol}")
    return strategies


def cmd_stability(args):
    """Run strategy stability analysis."""
    config = yaml.safe_load(open('config_v2.yaml'))

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'] if args.symbol == 'ALL' \
              else [args.symbol]

    wf_config = WFConfig(
        mode=args.window_mode,
        test_months=3,
        step_months=3,
        min_train_months=18,
        embargo_days=7,
    )

    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"STABILITY ANALYSIS: {symbol}")
        logger.info(f"{'='*60}")

        df = load_full_data(symbol, config)
        strategies = load_existing_strategies(symbol)

        if not strategies:
            logger.warning(f"No validated strategies for {symbol}, skipping")
            continue

        engine = WalkForwardEngine(config, wf_config)
        results = engine.run_stability(df, strategies)
        agg = aggregate_wf_results(results)
        print_wf_summary(agg)

        all_results[symbol] = agg

    # Save
    output_dir = Path('results') / f'walkforward_stability_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir}")


def cmd_evolve(args):
    """Run full walk-forward with re-evolution."""
    config = yaml.safe_load(open('config_v2.yaml'))

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'] if args.symbol == 'ALL' \
              else [args.symbol]

    wf_config = WFConfig(
        mode=args.window_mode,
        test_months=3,
        step_months=3,
        min_train_months=18,
        embargo_days=7,
        population=args.pop,
        generations=args.gens,
        patience=args.patience,
        n_top=args.n_top,
        cpcv_groups=6,
        pbo_threshold=0.50,
        min_trades_oos=3,
    )

    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD EVOLUTION: {symbol} (seed={args.seed})")
        logger.info(f"{'='*60}")

        # Override symbol in config
        cfg = dict(config)
        cfg['data'] = dict(cfg.get('data', {}))
        cfg['data']['symbol'] = symbol

        df = load_full_data(symbol, cfg)

        engine = WalkForwardEngine(cfg, wf_config)
        results = engine.run_evolve(df, seed=args.seed)
        agg = aggregate_wf_results(results)
        print_wf_summary(agg)

        all_results[symbol] = agg

    # Save
    safe = symbols[0].replace('/', '_') if len(symbols) == 1 else 'ALL'
    output_dir = Path('results') / f'walkforward_evolve_{safe}_seed{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir}")


def cmd_cmaes(args):
    """Run CMA-ES walk-forward re-optimization."""
    config = yaml.safe_load(open('config_v2.yaml'))

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'] if args.symbol == 'ALL' \
              else [args.symbol]

    wf_config = WFConfig(
        mode=args.window_mode,
        test_months=3,
        step_months=3,
        min_train_months=18,
        embargo_days=7,
        cmaes_sigma=0.05,
        cmaes_evals=args.cmaes_evals,
        min_trades_oos=3,
    )

    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD CMA-ES: {symbol} (seed={args.seed})")
        logger.info(f"{'='*60}")

        df = load_full_data(symbol, config)
        strategies = load_existing_strategies(symbol)

        if not strategies:
            logger.warning(f"No validated strategies for {symbol}, skipping")
            continue

        engine = WalkForwardEngine(config, wf_config)
        results = engine.run_cmaes(df, strategies, seed=args.seed)
        agg = aggregate_wf_results(results)
        print_wf_summary(agg)

        all_results[symbol] = agg

    # Save
    safe = symbols[0].replace('/', '_') if len(symbols) == 1 else 'ALL'
    output_dir = Path('results') / f'walkforward_cmaes_{safe}_seed{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='CriptoGA Walk-Forward Analysis'
    )
    subparsers = parser.add_subparsers(dest='command')

    # stability
    p_stab = subparsers.add_parser('stability',
        help='Backtest existing strategies across time windows (diagnostic)')
    p_stab.add_argument('--symbol', default='BTC/USDT',
        help='Symbol or ALL')
    p_stab.add_argument('--window-mode', default='expanding',
        choices=['expanding', 'rolling'])

    # evolve
    p_evo = subparsers.add_parser('evolve',
        help='Full re-evolution per window')
    p_evo.add_argument('--symbol', default='BTC/USDT')
    p_evo.add_argument('--seed', type=int, default=42)
    p_evo.add_argument('--pop', type=int, default=150)
    p_evo.add_argument('--gens', type=int, default=80)
    p_evo.add_argument('--patience', type=int, default=15)
    p_evo.add_argument('--n-top', type=int, default=10)
    p_evo.add_argument('--window-mode', default='expanding',
        choices=['expanding', 'rolling'])

    # cmaes
    p_cma = subparsers.add_parser('cmaes',
        help='CMA-ES re-optimization per window')
    p_cma.add_argument('--symbol', default='BTC/USDT')
    p_cma.add_argument('--seed', type=int, default=42)
    p_cma.add_argument('--cmaes-evals', type=int, default=200)
    p_cma.add_argument('--window-mode', default='expanding',
        choices=['expanding', 'rolling'])

    args = parser.parse_args()

    setup_logging()

    if args.command == 'stability':
        cmd_stability(args)
    elif args.command == 'evolve':
        cmd_evolve(args)
    elif args.command == 'cmaes':
        cmd_cmaes(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
