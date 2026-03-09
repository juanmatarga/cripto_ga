"""Tests for HMM volatility regime detector and combined detector."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from data.regime_detector_hmm import (
    HMMVolatilityDetector, _compute_features, _compute_sizing_multiplier,
    detect_regime_combined,
)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data with distinct volatility regimes."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range('2024-01-01', periods=n, freq='15min', tz='UTC')

    # Create three distinct periods: calm, normal, volatile
    prices = [100.0]
    for i in range(1, n):
        if i < 700:
            ret = np.random.normal(0.0001, 0.002)  # Calm
        elif i < 1400:
            ret = np.random.normal(0, 0.005)        # Normal
        else:
            ret = np.random.normal(-0.0001, 0.015)  # Volatile
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.001 for p in prices],
        'Low': [p * 0.999 for p in prices],
        'Close': prices,
        'Volume': np.random.uniform(100, 1000, n),
    }, index=dates)
    return df


class TestComputeFeatures:
    def test_returns_all_columns(self, sample_ohlcv):
        features = _compute_features(sample_ohlcv)
        expected = {'log_ret', 'real_vol', 'vol_ratio', 'abs_sma_slope', 'vol_of_vol'}
        assert set(features.columns) == expected

    def test_same_length_as_input(self, sample_ohlcv):
        features = _compute_features(sample_ohlcv)
        assert len(features) == len(sample_ohlcv)

    def test_no_extreme_outliers(self, sample_ohlcv):
        """Clipping should remove extreme values."""
        features = _compute_features(sample_ohlcv)
        for col in features.columns:
            valid = features[col].dropna()
            if len(valid) > 0:
                assert valid.max() < 1e6
                assert valid.min() > -1e6


class TestHMMVolatilityDetector:
    def test_fit_and_predict(self, sample_ohlcv):
        detector = HMMVolatilityDetector(n_states=3)
        detector.fit(sample_ohlcv)
        vol_states = detector.predict_vol_state(sample_ohlcv)
        assert len(vol_states) == len(sample_ohlcv)
        assert set(vol_states.unique()).issubset({'calm', 'normal', 'volatile'})

    def test_stability_output_keys(self, sample_ohlcv):
        detector = HMMVolatilityDetector(n_states=3)
        detector.fit(sample_ohlcv)
        result = detector.get_regime_stability(sample_ohlcv.tail(200))
        assert 'vol_state' in result
        assert 'stability' in result
        assert 'transition_risk' in result
        assert 'vol_state_probs' in result
        assert 0 <= result['stability'] <= 1
        assert abs(result['stability'] + result['transition_risk'] - 1.0) < 0.01

    def test_save_and_load(self, sample_ohlcv, tmp_path):
        detector = HMMVolatilityDetector(n_states=3)
        detector.fit(sample_ohlcv)
        path = str(tmp_path / 'test_hmm.pkl')
        detector.save(path)

        loaded = HMMVolatilityDetector.load(path)
        orig = detector.predict_vol_state(sample_ohlcv)
        loaded_pred = loaded.predict_vol_state(sample_ohlcv)
        assert (orig == loaded_pred).all()

    def test_deterministic_across_seeds(self, sample_ohlcv):
        """Same seed → same results."""
        d1 = HMMVolatilityDetector(n_states=3, random_seed=42)
        d1.fit(sample_ohlcv)
        d2 = HMMVolatilityDetector(n_states=3, random_seed=42)
        d2.fit(sample_ohlcv)
        r1 = d1.predict_vol_state(sample_ohlcv)
        r2 = d2.predict_vol_state(sample_ohlcv)
        assert (r1 == r2).all()

    def test_short_data_raises(self):
        """Too few bars should raise."""
        short_df = pd.DataFrame({
            'Close': [100] * 50,
            'Volume': [1000] * 50,
        }, index=pd.date_range('2024-01-01', periods=50, freq='15min'))
        detector = HMMVolatilityDetector()
        with pytest.raises(ValueError, match="at least 100"):
            detector.fit(short_df)


class TestSizingMultiplier:
    def test_full_confidence_calm(self):
        assert _compute_sizing_multiplier(1.0, 1.0, 'calm') == 1.0

    def test_floor_at_03(self):
        assert _compute_sizing_multiplier(0.0, 0.0, 'volatile') == 0.3

    def test_volatile_penalty(self):
        normal = _compute_sizing_multiplier(0.8, 0.8, 'normal')
        volatile = _compute_sizing_multiplier(0.8, 0.8, 'volatile')
        assert volatile < normal

    def test_calm_bonus(self):
        normal = _compute_sizing_multiplier(0.8, 0.8, 'normal')
        calm = _compute_sizing_multiplier(0.8, 0.8, 'calm')
        assert calm >= normal


class TestCombinedDetector:
    def test_output_keys(self, sample_ohlcv):
        hmm = HMMVolatilityDetector(n_states=3)
        hmm.fit(sample_ohlcv)
        result = detect_regime_combined(sample_ohlcv.tail(200), hmm=hmm)

        # SMA keys
        assert 'regime' in result
        assert 'confidence' in result
        assert 'slope' in result
        # HMM keys
        assert 'vol_state' in result
        assert 'stability' in result
        assert 'transition_risk' in result
        # Sizing key
        assert 'sizing_mult' in result
        assert 0.3 <= result['sizing_mult'] <= 1.0

    def test_regime_is_from_sma(self, sample_ohlcv):
        """regime should come from SMA, not HMM."""
        from data.regime_detector import detect_regime_with_confidence
        hmm = HMMVolatilityDetector(n_states=3)
        hmm.fit(sample_ohlcv)
        combined = detect_regime_combined(sample_ohlcv.tail(200), hmm=hmm)
        sma_only = detect_regime_with_confidence(sample_ohlcv.tail(200))
        assert combined['regime'] == sma_only['regime']
        assert combined['confidence'] == sma_only['confidence']
