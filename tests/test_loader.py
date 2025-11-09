"""Tests para Binance data loader."""

import pytest
import pandas as pd
from datetime import datetime, timezone
from loader import load_binance_data, check_binance_connection, _timeframe_to_milliseconds

def test_connection_to_binance(config_fixture):
    """Test básico de conexión."""
    # Este test requiere conexión real a Binance
    result = check_binance_connection(config_fixture)
    assert isinstance(result, bool)

def test_timeframe_conversion():
    """Test de conversión de timeframes."""
    assert _timeframe_to_milliseconds('1m') == 60 * 1000
    assert _timeframe_to_milliseconds('15m') == 15 * 60 * 1000
    assert _timeframe_to_milliseconds('1h') == 60 * 60 * 1000
    assert _timeframe_to_milliseconds('1d') == 24 * 60 * 60 * 1000

def test_timeframe_invalid():
    """Test de timeframe inválido."""
    with pytest.raises(ValueError):
        _timeframe_to_milliseconds('5s')

# NOTA: Tests de load_binance_data requieren conexión real y son lentos
# Para testing local, usar pytest -m "not slow"
# Marcar tests lentos con @pytest.mark.slow

@pytest.mark.slow
def test_load_small_dataset(config_fixture):
    """Test de carga de dataset pequeño (1 día)."""
    # Modificar config para solo 1 día
    config_fixture['data']['start'] = '2024-10-01 00:00:00'
    config_fixture['data']['end'] = '2024-10-02 00:00:00'

    df = load_binance_data(config_fixture)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert df.index.name == 'timestamp'
    assert 'symbol' in df.attrs
