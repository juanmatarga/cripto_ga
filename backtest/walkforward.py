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

def create_walkforward_windows(data: pd.DataFrame, train_months: int,
                               test_months: int, step_months: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward windows with anti-lookahead guarantees.

    Args:
        data: Full OHLCV DataFrame with DatetimeIndex
        train_months: Training window size in months
        test_months: Testing window size in months
        step_months: Step size in months

    Returns:
        List of (train_df, test_df) tuples

    Algorithm:
        1. Generate all possible windows based on train/test/step months
        2. For each window: Assert test_start > train_end

    Example:
        Data: 2020-01-01 to 2025-01-01
        train_months=6, test_months=2, step_months=1
        - Window 1: Train 2020-01 to 2020-06, Test 2020-07 to 2020-08
        - Window 2: Train 2020-02 to 2020-07, Test 2020-08 to 2020-09
        - ...

    Notes:
        - ALWAYS asserts test_start > train_end
        - Returns simple (train_df, test_df) tuples for easy iteration
    """
    logger.info("Creating walk-forward windows...")

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

        # Anti-lookahead assertion
        assert test_start > train_end, \
            f"LOOKAHEAD DETECTED: test_start ({test_start}) <= train_end ({train_end})"

        all_windows.append((train_data, test_data))
        window_id += 1

        logger.debug(f"Window {window_id}: Train [{train_start.date()} to {train_end.date()}], "
                    f"Test [{test_start.date()} to {test_end.date()}]")

        # Move to next window
        current_start = current_start + pd.DateOffset(months=step_months)

    logger.info(f"[OK] Generated {len(all_windows)} total windows")

    return all_windows

def stratified_sampling_windows(windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
                                n_sample: int, seed: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Selecciona ventanas representativas por régimen de mercado.

    Estrategia INTELIGENTE:
    1. Calcula volatilidad, dirección, año para cada ventana
    2. Divide en bins por régimen
    3. Sample proporcional de cada bin

    Args:
        windows: Lista de (train_df, test_df) tuples
        n_sample: Número de ventanas a seleccionar
        seed: Random seed

    Returns:
        Lista de ventanas seleccionadas

    Example:
        Si tenemos 40 ventanas y queremos 5:
        - Calcular volatility bins: low, medium, high
        - Calcular direction bins: bear, sideways, bull
        - Calcular year bins: 2020, 2021, 2022, ...
        - Seleccionar 1-2 ventanas de cada combinación
    """
    np.random.seed(seed)

    if len(windows) <= n_sample:
        logger.info(f"Stratified sampling: Using all {len(windows)} windows (less than target {n_sample})")
        return windows

    logger.info(f"Stratified sampling: Selecting {n_sample}/{len(windows)} windows by market regime")

    # ========================================================================
    # 1. CARACTERIZAR RÉGIMEN DE CADA VENTANA
    # ========================================================================
    regime_data = []

    for i, (train_df, test_df) in enumerate(windows):
        # Volatilidad (std de returns)
        returns = train_df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualizado

        # Dirección (total return)
        total_return = (train_df['Close'].iloc[-1] / train_df['Close'].iloc[0]) - 1

        # Año
        year = train_df.index[0].year

        regime_data.append({
            'index': i,
            'volatility': volatility,
            'total_return': total_return,
            'year': year
        })

    regime_df = pd.DataFrame(regime_data)

    # ========================================================================
    # 2. BINNING POR RÉGIMEN
    # ========================================================================
    # Volatility bins: terciles
    regime_df['vol_bin'] = pd.qcut(regime_df['volatility'], q=3, labels=['low_vol', 'med_vol', 'high_vol'], duplicates='drop')

    # Direction bins: terciles
    regime_df['dir_bin'] = pd.qcut(regime_df['total_return'], q=3, labels=['bear', 'sideways', 'bull'], duplicates='drop')

    # Year bins
    regime_df['year_bin'] = regime_df['year']

    # Combinar en un solo régimen
    regime_df['regime'] = (regime_df['vol_bin'].astype(str) + '_' +
                           regime_df['dir_bin'].astype(str) + '_' +
                           regime_df['year_bin'].astype(str))

    logger.debug(f"Identified {regime_df['regime'].nunique()} unique regimes")

    # ========================================================================
    # 3. STRATIFIED SAMPLING
    # ========================================================================
    regime_counts = regime_df['regime'].value_counts()

    # Calcular cuántas ventanas tomar de cada régimen (proporcional)
    samples_per_regime = {}
    for regime, count in regime_counts.items():
        proportion = count / len(regime_df)
        n_from_regime = max(1, int(np.ceil(proportion * n_sample)))
        samples_per_regime[regime] = min(n_from_regime, count)

    # Ajustar si nos pasamos
    total_samples = sum(samples_per_regime.values())
    if total_samples > n_sample:
        # Reducir regímenes más grandes
        regimes_sorted = sorted(samples_per_regime.items(), key=lambda x: x[1], reverse=True)
        excess = total_samples - n_sample

        for regime, n in regimes_sorted:
            if excess == 0:
                break
            reduction = min(excess, samples_per_regime[regime] - 1)
            if reduction > 0:
                samples_per_regime[regime] -= reduction
                excess -= reduction

    # Seleccionar ventanas
    selected_indices = []

    for regime, n_to_select in samples_per_regime.items():
        regime_indices = regime_df[regime_df['regime'] == regime]['index'].values

        if len(regime_indices) <= n_to_select:
            selected_indices.extend(regime_indices)
        else:
            # Sample random de este régimen
            sampled = np.random.choice(regime_indices, size=n_to_select, replace=False)
            selected_indices.extend(sampled)

    # Ordenar para mantener orden temporal
    selected_indices = sorted(selected_indices)

    # Limitar a n_sample exacto
    if len(selected_indices) > n_sample:
        selected_indices = selected_indices[:n_sample]

    selected_windows = [windows[i] for i in selected_indices]

    logger.info(f"[OK] Selected {len(selected_windows)} windows covering {len(set(regime_df.loc[selected_indices, 'regime']))} regimes")

    # Log régimen distribution
    selected_regimes = regime_df.loc[selected_indices, 'regime'].value_counts()
    for regime, count in selected_regimes.items():
        logger.debug(f"  Regime {regime}: {count} windows")

    return selected_windows

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
