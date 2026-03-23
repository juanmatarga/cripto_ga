"""
NSGA-II Re-Evolution — Round 2 with fixed pipeline.

Training: 2022-01-01 to 2025-09-30
OTS: 2025-10-01 to 2026-02-28 (NOT used here — separate validation)

Usage:
    python3 run_nsga2_evolution.py --symbol BTC/USDT --seed 42
    python3 run_nsga2_evolution.py --symbol ETH/USDT --seed 42
    python3 run_nsga2_evolution.py --symbol BNB/USDT --seed 42
"""

import argparse
import json
import logging
import os
import random
import time
import yaml
from datetime import datetime

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='NSGA-II Evolution')
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop', type=int, default=200)
    parser.add_argument('--gen', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    symbol_short = args.symbol.replace('/USDT', '').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"nsga2_{symbol_short}_seed{args.seed}_{timestamp}"

    logger.info(f"=== NSGA-II Evolution: {args.symbol} seed={args.seed} ===")
    logger.info(f"Run: {run_name}")

    # Load base config (has pagination settings for loader)
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Override for Round 2
    config['data']['symbol'] = args.symbol
    config['data']['start'] = '2022-01-01 00:00:00'
    config['data']['end'] = '2025-09-30 23:59:59'

    # NSGA-II evolution params
    config['evolution'] = {
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'genome_length': 50,
        'window_bars': 11520,      # ~4 months — more trades per window for better stats
        'max_generations': args.gen,
        'archive_parent_pct': 0.10,
        'use_fixed_windows': False,  # WINDOW ROTATION — forces generalization
        'max_phenotype_copies': 2,   # strict clone limit (was 5)
        'keep_ratio': 0.6,          # 60% windows kept, 40% fresh each gen
    }
    config['fitness'] = {'parsimony_coefficient': 0.0}  # NO parsimony in objectives
    config['exits'] = {'atr_period': 14}
    config['costs'] = {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    }

    # Load data (from parquet cache if available, else Binance API)
    import pandas as pd
    cache_file = f"data/cache/{symbol_short}_USDT_15m_2022_2025.parquet"
    if os.path.exists(cache_file):
        logger.info(f"Loading {args.symbol} from cache: {cache_file}")
        data = pd.read_parquet(cache_file)
        logger.info(f"Loaded {len(data)} bars from cache ({data.index[0]} to {data.index[-1]})")
    else:
        logger.info(f"Loading {args.symbol} 15m data: 2022-01-01 to 2025-09-30...")
        from loader import load_binance_data
        data = load_binance_data(config)
        logger.info(f"Loaded {len(data)} bars ({data.index[0]} to {data.index[-1]})")

    # Use 10 windows per generation, rotating 40% each gen
    window_bars = config['evolution']['window_bars']
    max_possible = max(1, (len(data) - window_bars) // window_bars + 1)
    n_windows = min(10, max_possible)  # 10 windows of ~2 months, rotating
    config['evolution']['n_windows_per_gen'] = n_windows
    logger.info(f"Using {n_windows} rotating windows of {window_bars} bars (~{window_bars*15//(60*24)} days) from pool of {max_possible}")

    # Run evolution
    from evolution.engine import EvolutionEngine
    engine = EvolutionEngine(config, data)
    engine.initialize(pop_size=args.pop)

    logger.info(f"Starting: pop={args.pop}, gen={args.gen}, patience={args.patience}")
    logger.info(f"Windows: {config['evolution']['n_windows_per_gen']} x {config['evolution']['window_bars']} bars")
    t0 = time.time()

    result = engine.run(n_generations=args.gen, patience=args.patience)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {result.final_generation} gens in {elapsed/60:.1f} min")
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Pareto front: {len(result.pareto_front)} strategies")
    logger.info(f"Archive: {result.archive.n_occupied}/45 cells ({result.archive.coverage:.0%})")

    # Save results
    results_dir = f"results/{run_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save Pareto front strategies
    pareto_data = []
    for i, s in enumerate(result.pareto_front):
        entry = s.to_dict()
        entry['pareto_rank'] = i + 1
        pareto_data.append(entry)
        cf = s.objectives[0]
        con = s.objectives[1]
        sort_val = s.metrics.get('sortino', 0) if s.metrics else 0
        logger.info(
            f"  #{i+1} {s.direction} | CF={cf:+.2f} con={con:.0%} sortino={sort_val:+.2f} | "
            f"nodes={s.n_nodes} trades={s.n_trades} stab={s.stability:.2f} | "
            f"{s.to_readable()}"
        )

    with open(f"{results_dir}/pareto_front.json", 'w') as f:
        json.dump(pareto_data, f, indent=2, default=str)

    # Save evolution history
    history = [
        {
            'generation': s.generation,
            'best_sortino': s.best_sortino,
            'best_return': s.best_return,
            'median_sortino': s.median_sortino,
            'front1_size': s.front1_size,
            'valid_count': s.valid_count,
            'total_count': s.total_count,
            'cache_hit_rate': s.cache_hit_rate,
            'archive_coverage': s.archive_coverage,
            'elapsed_seconds': s.elapsed_seconds,
        }
        for s in result.history
    ]
    with open(f"{results_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(f"{results_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)

    logger.info(f"\nResults saved to {results_dir}/")


if __name__ == '__main__':
    main()
