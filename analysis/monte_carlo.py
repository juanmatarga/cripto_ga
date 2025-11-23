"""
Monte Carlo Simulation via Trade Shuffling.

Validates strategy robustness by:
1. Taking actual trade results
2. Shuffling order randomly
3. Recalculating equity curve
4. Repeating 1000 times
5. Comparing actual vs distribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def simulate_equity_curve_shuffled(trades: List[Dict],
                                   initial_capital: float = 1000.0,
                                   seed: int = None) -> pd.DataFrame:
    """
    Simulate equity curve with shuffled trade order.

    Args:
        trades: List of closed trades with 'pnl_usd' key
        initial_capital: Starting capital
        seed: Random seed for reproducibility

    Returns:
        DataFrame with equity curve
    """
    if seed is not None:
        np.random.seed(seed)

    # Shuffle trades
    shuffled_trades = trades.copy()
    np.random.shuffle(shuffled_trades)

    # Recalculate equity
    equity = [initial_capital]
    for trade in shuffled_trades:
        equity.append(equity[-1] + trade['pnl_usd'])

    return pd.DataFrame({
        'equity': equity[1:],  # Skip initial
        'trade_num': range(1, len(equity))
    })


def run_monte_carlo(trades: List[Dict],
                   initial_capital: float = 1000.0,
                   n_simulations: int = 1000) -> Dict:
    """
    Run Monte Carlo simulation via trade shuffling.

    Args:
        trades: List of closed trades
        initial_capital: Starting capital
        n_simulations: Number of simulations

    Returns:
        Dict with simulation results
    """
    logger.info(f"Running Monte Carlo: {n_simulations} simulations with {len(trades)} trades")

    # Actual equity curve
    actual_equity = [initial_capital]
    for trade in trades:
        actual_equity.append(actual_equity[-1] + trade['pnl_usd'])
    actual_final = actual_equity[-1]

    # Run simulations
    simulated_finals = []
    simulated_curves = []

    for i in range(n_simulations):
        sim_curve = simulate_equity_curve_shuffled(trades, initial_capital, seed=i)
        simulated_finals.append(sim_curve['equity'].iloc[-1])
        simulated_curves.append(sim_curve['equity'].values)

    simulated_finals = np.array(simulated_finals)

    # Calculate percentiles
    percentiles = {
        'p5': np.percentile(simulated_finals, 5),
        'p25': np.percentile(simulated_finals, 25),
        'p50': np.percentile(simulated_finals, 50),
        'p75': np.percentile(simulated_finals, 75),
        'p95': np.percentile(simulated_finals, 95)
    }

    # Calculate actual's percentile rank
    actual_percentile = (simulated_finals < actual_final).sum() / n_simulations * 100

    results = {
        'actual_final': actual_final,
        'actual_return_pct': (actual_final - initial_capital) / initial_capital * 100,
        'actual_percentile': actual_percentile,
        'simulated_finals': simulated_finals,
        'simulated_curves': simulated_curves,
        'percentiles': percentiles,
        'mean_final': simulated_finals.mean(),
        'std_final': simulated_finals.std(),
        'best_case': simulated_finals.max(),
        'worst_case': simulated_finals.min(),
        'prob_profitable': (simulated_finals > initial_capital).sum() / n_simulations
    }

    logger.info(f"Monte Carlo complete: Actual ${actual_final:.2f} at {actual_percentile:.1f}th percentile")
    logger.info(f"  Mean: ${results['mean_final']:.2f}, Std: ${results['std_final']:.2f}")
    logger.info(f"  P(profitable): {results['prob_profitable']*100:.1f}%")

    return results
