"""Tests for window sampling with partial rotation."""

import pandas as pd
import numpy as np
import pytest


def _make_data(n_bars=50000):
    """Create synthetic OHLCV data for sampling tests."""
    idx = pd.date_range('2022-01-01', periods=n_bars, freq='15min')
    np.random.seed(42)
    return pd.DataFrame({
        'Open': np.random.randn(n_bars).cumsum() + 100,
        'High': np.random.randn(n_bars).cumsum() + 101,
        'Low': np.random.randn(n_bars).cumsum() + 99,
        'Close': np.random.randn(n_bars).cumsum() + 100,
        'Volume': np.random.rand(n_bars) * 1000,
    }, index=idx)


class TestSampleWindowsWithRotation:
    def test_returns_correct_count(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=5, window_bars=8640)
        assert len(result) == 5

    def test_returns_tuples_with_ids(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=3, window_bars=4320)
        for df, wid in result:
            assert isinstance(df, pd.DataFrame)
            assert isinstance(wid, str)
            assert wid.startswith("w_")

    def test_partial_rotation_keeps_some(self):
        import random
        random.seed(42)
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        gen1 = sample_windows_with_rotation(data, n_windows=5, window_bars=4320)
        gen2 = sample_windows_with_rotation(
            data, n_windows=5, window_bars=4320,
            previous_windows=gen1, keep_ratio=0.6
        )
        gen1_ids = {wid for _, wid in gen1}
        gen2_ids = {wid for _, wid in gen2}
        kept = gen1_ids & gen2_ids
        assert len(kept) >= 2

    def test_window_size_correct(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=3, window_bars=8640)
        for df, _ in result:
            assert len(df) == 8640

    def test_short_data_returns_single(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(1000)
        result = sample_windows_with_rotation(data, n_windows=5, window_bars=8640)
        assert len(result) == 1
