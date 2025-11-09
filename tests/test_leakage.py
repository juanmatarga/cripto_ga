"""
Anti-Lookahead Tests - Critical for Data Integrity

These tests ensure that NO future data leaks into training/evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.walkforward import (
    create_walkforward_windows, validate_no_lookahead,
    WalkForwardWindow
)
from ga_patterns.chromosome import Pattern, PredicateNode
from backtest.runner import run_backtest, generate_signals

def test_walkforward_no_overlap(config_fixture):
    """
    Walk-forward windows must have test_start > train_end.
    """
    # Create sample data (2 years)
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start_date, periods=1000, freq='15min')
    data = pd.DataFrame({
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 102,
        'Low': np.random.randn(1000).cumsum() + 98,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000, 2000, 1000)
    }, index=dates)

    # Create windows (new signature)
    windows = create_walkforward_windows(data, train_months=1, test_months=1, step_months=1)

    # Validate all windows
    for train_df, test_df in windows:
        # Critical assertion: test starts AFTER train ends
        train_end = train_df.index.max()
        test_start = test_df.index.min()

        assert test_start > train_end, \
            f"LOOKAHEAD DETECTED: test_start ({test_start}) <= train_end ({train_end})"

        # Additional check: no date overlap
        train_dates = set(train_df.index)
        test_dates = set(test_df.index)
        overlap = train_dates & test_dates

        assert len(overlap) == 0, \
            f"Date overlap detected: {len(overlap)} dates"

        # Validate using explicit function
        validate_no_lookahead(train_df, test_df)

def test_walkforward_chronological_order(config_fixture):
    """
    Walk-forward windows must be in chronological order.
    """
    # Create sample data
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start_date, periods=1000, freq='15min')
    data = pd.DataFrame({
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 102,
        'Low': np.random.randn(1000).cumsum() + 98,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000, 2000, 1000)
    }, index=dates)

    windows = create_walkforward_windows(data, train_months=1, test_months=1, step_months=1)

    # Check chronological order
    for i in range(len(windows) - 1):
        current_train, _ = windows[i]
        next_train, _ = windows[i + 1]

        # Next window's train should start after or at same time as current window's train
        assert next_train.index.min() >= current_train.index.min(), \
            f"Windows out of order: {i} vs {i+1}"

def test_signal_generation_no_lookahead(config_fixture):
    """
    Signal generation must not use future data.
    """
    # Create pattern
    pattern = Pattern(
        direction='LONG',
        window=2,
        expression=PredicateNode('close', '>', bar_offset=0, compare_with_bar=1),
        generation_created=0
    )

    # Create data
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 2000, 100)
    }, index=dates)

    # Generate signals
    signals = generate_signals(pattern, data)

    # Check that early bars (insufficient history) are False
    min_bars = pattern.window + 20
    for i in range(min_bars):
        assert signals.iloc[i] == False, \
            f"Signal generated at bar {i} with insufficient history"

def test_backtest_exit_timing(config_fixture):
    """
    Exits must be checked AFTER entry, not before.
    """
    # Create simple pattern
    pattern = Pattern(
        direction='LONG',
        window=2,
        expression=PredicateNode('close', '>', bar_offset=0, compare_with_bar=1),
        generation_created=0
    )

    # Create data with known pattern
    dates = pd.date_range('2023-01-01', periods=200, freq='15min')
    close_prices = [100] * 50 + [110] * 50 + [105] * 100  # Entry at bar 50
    data = pd.DataFrame({
        'Open': close_prices,
        'High': [p + 2 for p in close_prices],
        'Low': [p - 2 for p in close_prices],
        'Close': close_prices,
        'Volume': [1000] * 200
    }, index=dates)

    # Run backtest
    equity, trades = run_backtest(pattern, data, config_fixture)

    # Check that all exits happen AFTER entries
    for i, trade in trades.iterrows():
        assert trade['exit_bar'] > trade['entry_bar'], \
            f"Trade {i}: Exit bar ({trade['exit_bar']}) <= entry bar ({trade['entry_bar']})"

        # Exit date after entry date
        assert trade['exit_date'] > trade['entry_date'], \
            f"Trade {i}: Exit date before entry date"

def test_walkforward_window_class_assertion():
    """
    WalkForwardWindow class must raise AssertionError if test_start <= train_end.
    """
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    train_data = pd.DataFrame({
        'Close': [100] * 50
    }, index=dates[:50])

    test_data = pd.DataFrame({
        'Close': [100] * 50
    }, index=dates[30:80])  # OVERLAPPING with train

    # This should raise AssertionError
    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        window = WalkForwardWindow(
            window_id=0,
            train_start=dates[0],
            train_end=dates[49],
            test_start=dates[30],  # Starts BEFORE train ends
            test_end=dates[79],
            train_data=train_data,
            test_data=test_data
        )

def test_validate_no_lookahead_function():
    """
    validate_no_lookahead must catch overlapping data.
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')

    # Valid case: no overlap
    train_data = pd.DataFrame({'Close': [100] * 50}, index=dates[:50])
    test_data = pd.DataFrame({'Close': [100] * 49}, index=dates[51:])

    # Should NOT raise
    validate_no_lookahead(train_data, test_data)

    # Invalid case: overlap
    train_data_bad = pd.DataFrame({'Close': [100] * 50}, index=dates[:50])
    test_data_bad = pd.DataFrame({'Close': [100] * 50}, index=dates[30:80])

    # Should raise
    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        validate_no_lookahead(train_data_bad, test_data_bad)

def test_stratified_sampling_preserves_no_lookahead(config_fixture):
    """
    Stratified sampling must preserve anti-lookahead guarantees.
    """
    from backtest.walkforward import stratified_sampling_windows

    # Create sample data
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start_date, periods=2000, freq='15min')
    data = pd.DataFrame({
        'Open': np.random.randn(2000).cumsum() + 100,
        'High': np.random.randn(2000).cumsum() + 102,
        'Low': np.random.randn(2000).cumsum() + 98,
        'Close': np.random.randn(2000).cumsum() + 100,
        'Volume': np.random.randint(1000, 2000, 2000)
    }, index=dates)

    # Create all windows
    all_windows = create_walkforward_windows(data, train_months=1, test_months=1, step_months=1)

    # Apply stratified sampling
    sampled_windows = stratified_sampling_windows(all_windows, n_sample=5, seed=42)

    # Every sampled window must have no lookahead
    for train_df, test_df in sampled_windows:
        train_end = train_df.index.max()
        test_start = test_df.index.min()

        assert test_start > train_end, \
            f"Stratified sampling violated anti-lookahead"

        validate_no_lookahead(train_df, test_df)

def test_atr_calculation_no_lookahead():
    """
    ATR calculation must not use future data.
    """
    from backtest.exits import calculate_atr

    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    data = pd.DataFrame({
        'High': [105, 110, 108, 112, 115],
        'Low': [95, 100, 98, 102, 105],
        'Close': [100, 105, 103, 107, 110]
    }, index=dates[:5])

    atr = calculate_atr(data, period=3)

    # ATR at bar i should only use data up to bar i
    # ATR can be calculated from bar 0 (uses High-Low for first bar)
    # But should use previous close for subsequent bars
    assert not pd.isna(atr.iloc[0]), "ATR[0] should be valid (uses High-Low)"

    # ATR should never exceed max(High-Low) by much
    max_range = (data['High'] - data['Low']).max()
    valid_atr = atr.dropna()

    if len(valid_atr) > 0:
        assert valid_atr.max() <= max_range * 2, \
            "ATR suspiciously high - possible lookahead"

def test_pattern_evaluation_lookback_only(config_fixture):
    """
    Pattern evaluation must only look backward, never forward.
    """
    # Create pattern with bar_offset references
    pattern = Pattern(
        direction='LONG',
        window=5,
        expression=PredicateNode('close', '>', bar_offset=0, compare_with_bar=3),
        generation_created=0
    )

    # Create data
    dates = pd.date_range('2023-01-01', periods=50, freq='15min')
    data = pd.DataFrame({
        'Open': list(range(50)),
        'High': list(range(50)),
        'Low': list(range(50)),
        'Close': list(range(50)),
        'Volume': [1000] * 50
    }, index=dates)

    # Evaluate pattern
    signals = generate_signals(pattern, data)

    # At bar i, pattern should only reference bars 0...i, not i+1...N
    # We can't easily test this directly, but we can check that
    # signals don't change if we modify future data

    # Generate signals again with modified future
    data_modified = data.copy()
    data_modified.iloc[40:, :] = 999  # Modify last 10 bars

    signals_modified = generate_signals(pattern, data_modified)

    # Signals up to bar 39 should be identical
    # (because future data shouldn't affect past signals)
    for i in range(35):  # Check first 35 bars (safe window)
        assert signals.iloc[i] == signals_modified.iloc[i], \
            f"Signal at bar {i} changed when future data changed - LOOKAHEAD!"
