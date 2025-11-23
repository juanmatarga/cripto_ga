"""
Portfolio Selection - Decorrelation and Filtering
Greedy algorithm to build diversified pattern portfolio
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

from ga_patterns.chromosome import Pattern
from backtest.runner import run_backtest
from backtest.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

def calculate_correlation_matrix(equity_curves: List[pd.Series]) -> pd.DataFrame:
    """
    Calculate correlation matrix between equity curves.

    Args:
        equity_curves: List of equity Series (one per pattern)

    Returns:
        Correlation matrix (DataFrame)

    Notes:
        - Uses Pearson correlation on returns
        - Handles NaN values (fills with 0)
    """
    # Convert to returns
    returns_list = []
    for equity in equity_curves:
        returns = equity.pct_change().fillna(0)
        returns_list.append(returns)

    # Create DataFrame
    returns_df = pd.DataFrame(returns_list).T

    # Calculate correlation
    corr_matrix = returns_df.corr()

    logger.debug(f"Correlation matrix calculated: {corr_matrix.shape}")

    return corr_matrix

def filter_patterns_by_metrics(patterns: List[Pattern], data: pd.DataFrame,
                               config: dict) -> List[Tuple[Pattern, pd.Series, Dict]]:
    """
    Filter patterns by minimum quality thresholds.

    Args:
        patterns: List of patterns to filter
        data: OHLCV data for backtesting
        config: Config dict with selection.filters section

    Returns:
        List of (pattern, equity_curve, metrics) tuples that pass filters

    Filters applied:
        - upi_min
        - sharpe_min
        - cagr_min
        - max_drawdown_max (maximum allowed drawdown)
        - profit_factor_min
        - win_rate_min
        - min_trades_per_window

    Notes:
        - Patterns must pass ALL filters to be included
        - Runs full backtest for each pattern
    """
    logger.info(f"Filtering {len(patterns)} patterns by quality metrics...")

    filters = config['selection']['filters']
    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    passed_patterns = []

    for i, pattern in enumerate(patterns):
        try:
            # ALWAYS print pattern being evaluated
            logger.info(f"\n{'='*80}")
            logger.info(f"PORTFOLIO SELECTION - Evaluating Pattern {i+1}/{len(patterns)}:")
            logger.info(f"  Expression: {pattern.to_expression()}")
            logger.info(f"  Readable:   {pattern.to_readable()}")
            logger.info(f"  Direction:  {pattern.direction}")
            logger.info(f"  Modules:    {pattern.modules}")
            logger.info(f"  GA Fitness: {pattern.fitness:.4f}")
            logger.info(f"{'='*80}")

            # Run backtest
            equity, trades = run_backtest(pattern, data, config)

            # Calculate metrics (SPRINT 14 FIX: Pass trades for correct win rate)
            metrics = calculate_all_metrics(equity, periods_per_year, trades)

            # Apply filters
            logger.info(f"\nFilter Check:")
            logger.info(f"  UPI:           {metrics['upi']:.2f} (min: {filters['upi_min']:.2f})")
            logger.info(f"  Sharpe:        {metrics['sharpe']:.2f} (min: {filters['sharpe_min']:.2f})")
            logger.info(f"  CAGR:          {metrics['cagr']:.2%} (min: {filters['cagr_min']:.2%})")
            logger.info(f"  Max DD:        {abs(metrics['max_dd']):.2%} (max: {filters['max_drawdown_max']:.2%})")
            logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f} (min: {filters['profit_factor_min']:.2f})")
            logger.info(f"  Win Rate:      {metrics['win_rate']:.2%} (min: {filters['win_rate_min']:.2%})")
            logger.info(f"  Trades:        {len(trades)} (min: {filters['min_trades_per_window']}, max: {filters.get('max_trades_total', 1000)})")

            if metrics['upi'] < filters['upi_min']:
                logger.info(f"[ERROR] REJECTED: UPI too low")
                logger.info(f"{'='*80}\n")
                continue

            if metrics['sharpe'] < filters['sharpe_min']:
                logger.info(f"[ERROR] REJECTED: Sharpe too low")
                logger.info(f"{'='*80}\n")
                continue

            if metrics['cagr'] < filters['cagr_min']:
                logger.info(f"[ERROR] REJECTED: CAGR too low")
                logger.info(f"{'='*80}\n")
                continue

            if abs(metrics['max_dd']) > filters['max_drawdown_max']:
                logger.info(f"[ERROR] REJECTED: Drawdown too high")
                logger.info(f"{'='*80}\n")
                continue

            if metrics['profit_factor'] < filters['profit_factor_min']:
                logger.info(f"[ERROR] REJECTED: Profit factor too low")
                logger.info(f"{'='*80}\n")
                continue

            if metrics['win_rate'] < filters['win_rate_min']:
                logger.info(f"[ERROR] REJECTED: Win rate too low")
                logger.info(f"{'='*80}\n")
                continue

            if len(trades) < filters['min_trades_per_window']:
                logger.info(f"[ERROR] REJECTED: Not enough trades")
                logger.info(f"{'='*80}\n")
                continue

            # SPRINT 14 FIX: Max trades filter (prevent overtrading patterns)
            max_trades_total = filters.get('max_trades_total', 1000)
            if len(trades) > max_trades_total:
                logger.info(f"[ERROR] REJECTED: Too many trades (overtrading)")
                logger.info(f"{'='*80}\n")
                continue

            # All filters passed
            passed_patterns.append((pattern, equity, metrics))
            logger.info(f"[OK] PASSED ALL FILTERS - Added to portfolio candidates")
            logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"Pattern {i}: Error during filtering: {e}")
            continue

    logger.info(f"[OK] Filtering complete: {len(passed_patterns)}/{len(patterns)} patterns passed")

    return passed_patterns

def greedy_decorrelation(filtered_patterns: List[Tuple[Pattern, pd.Series, Dict]],
                        config: dict) -> List[Tuple[Pattern, pd.Series, Dict]]:
    """
    Select decorrelated portfolio using greedy algorithm.

    Args:
        filtered_patterns: List of (pattern, equity, metrics) that passed filters
        config: Config dict with selection section

    Returns:
        Selected portfolio (list of tuples)

    Algorithm:
        1. Sort patterns by fitness (descending)
        2. Select top pattern
        3. For each remaining pattern:
           - Calculate correlation with all selected patterns
           - If max_correlation < threshold: add to portfolio
           - Stop when portfolio reaches topk size

    Notes:
        - Greedy approach (not globally optimal, but fast)
        - Prioritizes high-fitness patterns
        - Ensures ρ < corr_threshold for all pairs
    """
    logger.info(f"Running greedy decorrelation algorithm...")

    corr_threshold = config['selection']['corr_threshold']
    topk = config['selection']['topk']

    # Sort by fitness (using pattern.fitness)
    sorted_patterns = sorted(filtered_patterns, key=lambda x: x[0].fitness, reverse=True)

    # Initialize portfolio with best pattern
    portfolio = [sorted_patterns[0]]
    portfolio_equities = [sorted_patterns[0][1]]

    logger.info(f"Portfolio seed: Pattern with fitness {sorted_patterns[0][0].fitness:.4f}")

    # Greedy selection
    for candidate_pattern, candidate_equity, candidate_metrics in sorted_patterns[1:]:
        if len(portfolio) >= topk:
            logger.info(f"Portfolio full ({topk} patterns). Stopping.")
            break

        # Calculate correlation with all portfolio patterns
        max_corr = 0.0
        for portfolio_pattern, portfolio_equity, portfolio_metrics in portfolio:
            # Calculate returns correlation
            candidate_returns = candidate_equity.pct_change().fillna(0)
            portfolio_returns = portfolio_equity.pct_change().fillna(0)

            # Align indices (in case of different lengths)
            min_len = min(len(candidate_returns), len(portfolio_returns))
            corr = np.corrcoef(
                candidate_returns.iloc[:min_len],
                portfolio_returns.iloc[:min_len]
            )[0, 1]

            if np.isnan(corr):
                corr = 0.0

            max_corr = max(max_corr, abs(corr))

        # Add if below threshold
        if max_corr < corr_threshold:
            portfolio.append((candidate_pattern, candidate_equity, candidate_metrics))
            portfolio_equities.append(candidate_equity)
            logger.info(f"Added pattern (fitness={candidate_pattern.fitness:.4f}, "
                       f"max_corr={max_corr:.3f}) - Portfolio size: {len(portfolio)}")
        else:
            logger.debug(f"Rejected pattern (fitness={candidate_pattern.fitness:.4f}, "
                        f"max_corr={max_corr:.3f} >= {corr_threshold})")

    logger.info(f"[OK] Decorrelation complete: {len(portfolio)} patterns selected")

    # Calculate final correlation matrix for reporting
    if len(portfolio) > 1:
        portfolio_equities_series = [p[1] for p in portfolio]
        final_corr_matrix = calculate_correlation_matrix(portfolio_equities_series)

        logger.info(f"Final portfolio correlation matrix:")
        logger.info(f"\n{final_corr_matrix}")

        avg_corr = final_corr_matrix.values[np.triu_indices_from(final_corr_matrix.values, k=1)].mean()
        logger.info(f"Average pairwise correlation: {avg_corr:.3f}")

    return portfolio

def select_portfolio(patterns: List[Pattern], data: pd.DataFrame,
                    config: dict) -> List[Tuple[Pattern, pd.Series, Dict]]:
    """
    Complete portfolio selection pipeline.

    Args:
        patterns: Raw patterns from GA (e.g., top 50)
        data: OHLCV data for backtesting
        config: Config dict

    Returns:
        Final portfolio (list of tuples)

    Pipeline:
        1. Filter by quality metrics
        2. Greedy decorrelation
        3. Return final portfolio

    Notes:
        - Combines filtering + decorrelation
        - Returns patterns ready for validation
    """
    logger.info("="*80)
    logger.info("PORTFOLIO SELECTION PIPELINE")
    logger.info("="*80)

    # Step 1: Filter
    filtered = filter_patterns_by_metrics(patterns, data, config)

    if len(filtered) == 0:
        logger.error("No patterns passed quality filters!")
        return []

    # Step 2: Decorrelate
    portfolio = greedy_decorrelation(filtered, config)

    logger.info(f"\n[OK] Portfolio selection complete: {len(portfolio)} patterns")

    # Print portfolio summary
    logger.info("\nFINAL PORTFOLIO:")
    for i, (pattern, equity, metrics) in enumerate(portfolio, 1):
        logger.info(f"{i}. {pattern.direction} | UPI={metrics['upi']:.2f} | "
                   f"Sharpe={metrics['sharpe']:.2f} | CAGR={metrics['cagr']:.2%} | "
                   f"MDD={abs(metrics['max_dd']):.2%}")
        logger.info(f"   {pattern.expression}")

    return portfolio
