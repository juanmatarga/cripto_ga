"""
Tests for Sprint 7: Multi-Asset Support.
"""

import numpy as np
import pandas as pd
import pytest

from data.multi_asset import (
    ASSETS, make_asset_config, validate_asset,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_config():
    return {
        'data': {
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'market_type': 'future',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
            'ots_start': '2025-06-01',
        },
        'costs': {
            'fees_bps_long': 1.0,
            'fees_bps_short': 1.0,
            'slippage_bps_long': 1.0,
            'slippage_bps_short': 1.0,
        },
        'evolution': {
            'population': 300,
            'generations_max': 150,
            'seed': 42,
        },
    }


@pytest.fixture
def df_sample():
    """Generic OHLCV data for validation tests."""
    np.random.seed(42)
    n = 60000
    dates = pd.date_range('2022-01-01', periods=n, freq='15min')
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    price = np.maximum(price, 1)  # No negative prices
    return pd.DataFrame({
        'Open': price,
        'High': price + np.abs(np.random.randn(n) * 0.3),
        'Low': price - np.abs(np.random.randn(n) * 0.3),
        'Close': price + np.random.randn(n) * 0.2,
        'Volume': np.abs(np.random.randn(n) * 1000) + 500,
    }, index=dates)


# ============================================================================
# Asset Definitions
# ============================================================================

class TestAssetDefinitions:
    """Test asset configuration and definitions."""

    def test_assets_defined(self):
        assert 'BTC/USDT' in ASSETS
        assert 'ETH/USDT' in ASSETS
        assert 'SOL/USDT' in ASSETS
        assert 'BNB/USDT' in ASSETS

    def test_assets_have_required_fields(self):
        for symbol, info in ASSETS.items():
            assert 'start' in info, f"{symbol} missing 'start'"
            assert 'min_bars' in info, f"{symbol} missing 'min_bars'"

    def test_make_asset_config(self, base_config):
        eth_config = make_asset_config('ETH/USDT', base_config)
        assert eth_config['data']['symbol'] == 'ETH/USDT'
        # Should not mutate original
        assert base_config['data']['symbol'] == 'BTC/USDT'

    def test_make_asset_config_preserves_costs(self, base_config):
        eth_config = make_asset_config('ETH/USDT', base_config)
        assert eth_config['costs'] == base_config['costs']

    def test_make_asset_config_unknown_symbol(self, base_config):
        """Unknown symbols should still work (no crash)."""
        config = make_asset_config('DOGE/USDT', base_config)
        assert config['data']['symbol'] == 'DOGE/USDT'


# ============================================================================
# Data Validation
# ============================================================================

class TestValidation:
    """Test data validation for evolution readiness."""

    def test_validate_good_data(self, df_sample):
        result = validate_asset(df_sample, 'TEST/USDT')
        assert result['valid']
        assert result['n_bars'] == 60000
        assert result['coverage'] > 0.95
        assert result['zero_vol_pct'] < 5.0

    def test_validate_data_with_gaps(self, df_sample):
        """Data with many gaps should fail validation."""
        # Remove 10% of bars
        df_gapped = df_sample.iloc[::2]  # 50% coverage
        result = validate_asset(df_gapped, 'GAP/USDT')
        assert not result['valid']

    def test_validate_data_with_zero_volume(self, df_sample):
        df_zero = df_sample.copy()
        df_zero.loc[df_zero.index[:6000], 'Volume'] = 0  # 10% zero vol
        result = validate_asset(df_zero, 'ZERO/USDT')
        assert not result['valid']
