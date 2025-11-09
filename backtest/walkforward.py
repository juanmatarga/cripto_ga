"""
Walk-Forward Analysis - Anti-Lookahead Validation
Supports stratified sampling for fast mode
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class WalkForwardWindow:
    """
    Represents a single train/test split with anti-lookahead guarantees.
    """
    def __init__(self, window_id: int,
                 train_start: pd.Timestamp, train_end: pd.Timestamp,
                 test_start: pd.Timestamp, test_end: pd.Timestamp,
                 train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.window_id = window_id
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.train_data = train_data
        self.test_data = test_data

        # CRITICAL: Anti-lookahead assertion
        assert test_start > train_end, \
            f"LOOKAHEAD DETECTED: test_start ({test_start}) <= train_end ({train_end})"

        logger.debug(f"Window {window_id} created: "
                    f"Train [{train_start} to {train_end}], "
                    f"Test [{test_start} to {test_end}]")

    def __repr__(self):
        return (f"WalkForwardWindow(id={self.window_id}, "
               f"train={self.train_start.date()} to {self.train_end.date()}, "
               f"test={self.test_start.date()} to {self.test_end.date()})")

def create_walkforward_windows(data: pd.DataFrame, config: dict,
                               fast_mode: bool = False) -> List[WalkForwardWindow]:
    """
    Create walk-forward windows with anti-lookahead guarantees.

    Args:
        data: Full OHLCV DataFrame with DatetimeIndex
        config: Config dict with walkforward section
        fast_mode: If True, use stratified sampling

    Returns:
        List of WalkForwardWindow objects

    Algorithm:
        1. Generate all possible windows based on train/test/step months
        2. If fast_mode: Select representative windows via stratified sampling
        3. For each window: Assert test_start > train_end

    Example:
        Data: 2020-01-01 to 2025-01-01
        train_months=6, test_months=2, step_months=1
        - Window 1: Train 2020-01 to 2020-06, Test 2020-07 to 2020-08
        - Window 2: Train 2020-02 to 2020-07, Test 2020-08 to 2020-09
        - ...
        If fast_mode with n_windows=5: Select 5 representative windows

    Notes:
        - ALWAYS asserts test_start > train_end
        - Stratified sampling ensures coverage across market regimes
    """
    logger.info("Creating walk-forward windows...")

    train_months = config['walkforward']['train_months']
    test_months = config['walkforward']['test_months']
    step_months = config['walkforward']['step_months']

    data_start = data.index.min()
    data_end = data.index.max()

    logger.info(f"Data range: {data_start} to {data_end}")
    logger.info(f"Walk-forward params: train={train_months}m, test={test_months}m, step={step_months}m")

    # Generate all possible windows
    all_windows = []
    window_id = 0
    current_start = data_start

    while True:
        # Calculate window boundaries
        train_start = current_start
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.DateOffset(days=1)  # +1 day gap to ensure no overlap
        test_end = test_start + pd.DateOffset(months=test_months)

        # Check if we have enough data
        if test_end > data_end:
            break

        # Extract data slices
        train_data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
        test_data = data[(data.index >= test_start) & (data.index <= test_end)].copy()

        # Validate data
        if len(train_data) < 100:
            logger.warning(f"Window {window_id} has too few train samples ({len(train_data)}). Skipping.")
            current_start = current_start + pd.DateOffset(months=step_months)
            continue

        if len(test_data) < 20:
            logger.warning(f"Window {window_id} has too few test samples ({len(test_data)}). Skipping.")
            current_start = current_start + pd.DateOffset(months=step_months)
            continue

        # Create window (with anti-lookahead assertion)
        window = WalkForwardWindow(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_data=train_data,
            test_data=test_data
        )

        all_windows.append(window)
        window_id += 1

        # Move to next window
        current_start = current_start + pd.DateOffset(months=step_months)

    logger.info(f"Generated {len(all_windows)} total windows")

    # Fast mode: Stratified sampling
    if fast_mode and config['ga']['fast_mode']['enabled']:
        n_windows = config['ga']['fast_mode']['n_windows']
        strategy = config['ga']['fast_mode']['strategy']

        if len(all_windows) <= n_windows:
            logger.info(f"Fast mode: Using all {len(all_windows)} windows (less than target {n_windows})")
            selected_windows = all_windows
        else:
            logger.info(f"Fast mode: Selecting {n_windows} windows via {strategy} sampling")
            selected_windows = _stratified_sampling(all_windows, n_windows, strategy)
    else:
        selected_windows = all_windows

    logger.info(f"[OK] Final windows: {len(selected_windows)}")
    for window in selected_windows:
        logger.info(f"  {window}")

    return selected_windows

def _stratified_sampling(windows: List[WalkForwardWindow], n_windows: int,
                        strategy: str) -> List[WalkForwardWindow]:
    """
    Select representative windows via stratified sampling.

    Args:
        windows: All available windows
        n_windows: Number of windows to select
        strategy: Sampling strategy ('stratified', 'uniform', 'recent')

    Returns:
        Selected windows

    Strategies:
        - 'stratified': Evenly spaced across time periods
        - 'uniform': Random uniform sampling
        - 'recent': Bias towards recent data

    Notes:
        - Ensures coverage of different market regimes
        - Stratified is recommended for crypto (high regime changes)
    """
    if strategy == 'stratified':
        # Evenly spaced indices
        indices = np.linspace(0, len(windows) - 1, n_windows, dtype=int)
        selected = [windows[i] for i in indices]

        logger.debug(f"Stratified sampling: Selected indices {indices}")

    elif strategy == 'uniform':
        # Random uniform
        np.random.seed(42)  # Reproducibility
        indices = np.random.choice(len(windows), size=n_windows, replace=False)
        indices = sorted(indices)
        selected = [windows[i] for i in indices]

        logger.debug(f"Uniform sampling: Selected indices {indices}")

    elif strategy == 'recent':
        # Bias towards recent (last 60% of windows)
        recent_cutoff = int(len(windows) * 0.4)
        recent_windows = windows[recent_cutoff:]

        if len(recent_windows) >= n_windows:
            # Stratify within recent windows
            indices = np.linspace(0, len(recent_windows) - 1, n_windows, dtype=int)
            selected = [recent_windows[i] for i in indices]
        else:
            # Not enough recent windows, use all recent + some old
            n_old_needed = n_windows - len(recent_windows)
            old_windows = windows[:recent_cutoff]
            old_indices = np.linspace(0, len(old_windows) - 1, n_old_needed, dtype=int)
            selected = [old_windows[i] for i in old_indices] + recent_windows

        logger.debug(f"Recent sampling: {len(selected)} windows (bias towards recent)")

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return selected

def validate_no_lookahead(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Explicit validation that test data comes strictly after train data.

    Args:
        train_data: Training DataFrame
        test_data: Testing DataFrame

    Raises:
        AssertionError if lookahead detected

    Notes:
        - Called by backtest to ensure no future data leakage
        - Should NEVER fail if using WalkForwardWindow
    """
    train_end = train_data.index.max()
    test_start = test_data.index.min()

    assert test_start > train_end, \
        f"LOOKAHEAD DETECTED: test_start ({test_start}) <= train_end ({train_end})"

    logger.debug(f"Anti-lookahead validated: train_end={train_end}, test_start={test_start}")

def get_window_market_regime(window: WalkForwardWindow) -> Dict[str, float]:
    """
    Characterize market regime for a window (for analysis/debugging).

    Args:
        window: WalkForwardWindow

    Returns:
        Dict with regime metrics:
        - volatility: Annualized volatility
        - trend: Price change %
        - regime: 'bull', 'bear', or 'sideways'

    Notes:
        - Used for stratified sampling analysis
        - Helps ensure diverse regime coverage
    """
    train_data = window.train_data

    # Calculate metrics
    returns = train_data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(365 * 24 * 4)  # Annualized for 15m data

    total_return = (train_data['Close'].iloc[-1] / train_data['Close'].iloc[0]) - 1

    # Classify regime
    if total_return > 0.20:  # >20% gain
        regime = 'bull'
    elif total_return < -0.20:  # >20% loss
        regime = 'bear'
    else:
        regime = 'sideways'

    return {
        'volatility': volatility,
        'total_return': total_return,
        'regime': regime,
        'train_start': window.train_start,
        'train_end': window.train_end
    }
