"""
Pattern Grammar - Direct OHLCV Predicates
Evolutionary approach: Direct comparisons → Ratios → Indicators
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# REGISTRY
# ============================================================================

PREDICATE_REGISTRY: Dict[str, Dict] = {}

def register_predicate(name: str, func: Callable,
                      threshold_range: Tuple[float, float],
                      description: str,
                      category: str = 'direct',
                      allows_comparison: bool = False):
    """
    Registra predicado.

    Args:
        allows_comparison: Si True, permite comparar con otra barra (ej: C[0] vs C[1])
    """
    PREDICATE_REGISTRY[name] = {
        'func': func,
        'threshold_range': threshold_range,
        'description': description,
        'category': category,
        'allows_comparison': allows_comparison
    }

# ============================================================================
# FASE 1: PREDICADOS DIRECTOS (Gen 1-50)
# Comparaciones OHLCV puras sin normalizar
# ============================================================================

def close_price(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Precio de cierre en barra específica."""
    idx = -(bar_offset + 1)
    if abs(idx) > len(data):
        return 0.0
    return data.iloc[idx]['Close']

register_predicate(
    'close',
    close_price,
    threshold_range=(0.0, 100000.0),  # Amplio para BTC
    description="Close price at bar",
    category='direct',
    allows_comparison=True
)

def open_price(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Precio de apertura."""
    idx = -(bar_offset + 1)
    if abs(idx) > len(data):
        return 0.0
    return data.iloc[idx]['Open']

register_predicate(
    'open',
    open_price,
    threshold_range=(0.0, 100000.0),
    description="Open price at bar",
    category='direct',
    allows_comparison=True
)

def high_price(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Precio máximo."""
    idx = -(bar_offset + 1)
    if abs(idx) > len(data):
        return 0.0
    return data.iloc[idx]['High']

register_predicate(
    'high',
    high_price,
    threshold_range=(0.0, 100000.0),
    description="High price at bar",
    category='direct',
    allows_comparison=True
)

def low_price(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Precio mínimo."""
    idx = -(bar_offset + 1)
    if abs(idx) > len(data):
        return 0.0
    return data.iloc[idx]['Low']

register_predicate(
    'low',
    low_price,
    threshold_range=(0.0, 100000.0),
    description="Low price at bar",
    category='direct',
    allows_comparison=True
)

def volume(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Volumen en barra."""
    idx = -(bar_offset + 1)
    if abs(idx) > len(data):
        return 0.0
    return data.iloc[idx]['Volume']

register_predicate(
    'volume',
    volume,
    threshold_range=(0.0, 1e12),  # Amplio
    description="Volume at bar",
    category='direct',
    allows_comparison=True
)

# ============================================================================
# FASE 2: PREDICADOS NORMALIZADOS (Gen 51-100)
# Ratios y cambios porcentuales
# ============================================================================

def price_change_pct(data: pd.DataFrame, bar_offset: int = 0, lookback: int = 1) -> float:
    """
    Cambio % del precio.

    Formula: (C[i] - C[i-lookback]) / C[i-lookback]
    """
    idx = -(bar_offset + 1)
    idx_prev = idx - lookback

    if abs(idx_prev) > len(data):
        return 0.0

    current = data.iloc[idx]['Close']
    previous = data.iloc[idx_prev]['Close']

    if previous == 0:
        return 0.0

    return (current - previous) / previous

register_predicate(
    'price_change_pct',
    price_change_pct,
    threshold_range=(-0.10, 0.10),  # ±10%
    description="Price % change: (C[i] - C[i-k]) / C[i-k]",
    category='ratio',
    allows_comparison=False
)

def body_pct(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """
    Tamaño del cuerpo como % del precio.

    Formula: abs(C - O) / C
    """
    idx = -(bar_offset + 1)
    row = data.iloc[idx]

    if row['Close'] == 0:
        return 0.0

    return abs(row['Close'] - row['Open']) / row['Close']

register_predicate(
    'body_pct',
    body_pct,
    threshold_range=(0.0, 0.05),  # 0-5%
    description="Body size: |C - O| / C",
    category='ratio',
    allows_comparison=False
)

def range_pct(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """
    Rango total como % del precio.

    Formula: (H - L) / C
    """
    idx = -(bar_offset + 1)
    row = data.iloc[idx]

    if row['Close'] == 0:
        return 0.0

    return (row['High'] - row['Low']) / row['Close']

register_predicate(
    'range_pct',
    range_pct,
    threshold_range=(0.0, 0.10),  # 0-10%
    description="Range: (H - L) / C",
    category='ratio',
    allows_comparison=False
)

def volume_change_pct(data: pd.DataFrame, bar_offset: int = 0, lookback: int = 1) -> float:
    """Cambio % del volumen."""
    idx = -(bar_offset + 1)
    idx_prev = idx - lookback

    if abs(idx_prev) > len(data):
        return 0.0

    current = data.iloc[idx]['Volume']
    previous = data.iloc[idx_prev]['Volume']

    if previous == 0:
        return 0.0

    return (current - previous) / previous

register_predicate(
    'volume_change_pct',
    volume_change_pct,
    threshold_range=(-0.50, 1.0),  # -50% a +100%
    description="Volume % change",
    category='ratio',
    allows_comparison=False
)

def close_position_in_range(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """
    Posición del cierre en el rango [0, 1].

    Formula: (C - L) / (H - L)
    """
    idx = -(bar_offset + 1)
    row = data.iloc[idx]

    range_val = row['High'] - row['Low']
    if range_val == 0:
        return 0.5

    return (row['Close'] - row['Low']) / range_val

register_predicate(
    'close_position_in_range',
    close_position_in_range,
    threshold_range=(0.0, 1.0),
    description="Close position: (C - L) / (H - L)",
    category='ratio',
    allows_comparison=False
)

def body_ratio(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """
    Proporción del cuerpo en el rango [0, 1].

    Formula: |C - O| / (H - L)
    """
    idx = -(bar_offset + 1)
    row = data.iloc[idx]

    body = abs(row['Close'] - row['Open'])
    range_val = row['High'] - row['Low']

    if range_val == 0:
        return 0.0

    return body / range_val

register_predicate(
    'body_ratio',
    body_ratio,
    threshold_range=(0.0, 1.0),
    description="Body / Range",
    category='ratio',
    allows_comparison=False
)

# ============================================================================
# FASE 3: INDICADORES TÉCNICOS (Gen 101-150)
# ============================================================================

def rsi_value(data: pd.DataFrame, bar_offset: int = 0, period: int = 14) -> float:
    """RSI(14)."""
    idx = -(bar_offset + 1)

    if abs(idx) + period > len(data):
        return 50.0

    closes = data['Close'].iloc[idx - period + 1: idx + 1]
    delta = closes.diff()

    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]

    if last_loss == 0:
        return 100.0

    rs = last_gain / last_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

register_predicate(
    'rsi',
    rsi_value,
    threshold_range=(0.0, 100.0),
    description="RSI(14)",
    category='indicator',
    allows_comparison=False
)

def price_vs_ma_pct(data: pd.DataFrame, bar_offset: int = 0, period: int = 20) -> float:
    """
    Distancia a MA como %.

    Formula: (C - MA) / MA
    """
    idx = -(bar_offset + 1)

    if abs(idx) + period > len(data):
        return 0.0

    current_price = data.iloc[idx]['Close']
    price_series = data['Close'].iloc[idx - period + 1: idx + 1]
    ma = price_series.mean()

    if ma == 0:
        return 0.0

    return (current_price - ma) / ma

register_predicate(
    'price_vs_ma_pct',
    price_vs_ma_pct,
    threshold_range=(-0.10, 0.10),
    description="Price vs MA(20): (C - MA) / MA",
    category='indicator',
    allows_comparison=False
)

# ============================================================================
# OPERADORES
# ============================================================================

class ComparisonOperator:
    """Operadores de comparación."""

    OPERATORS = {
        '>': lambda x, y: x > y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
    }

    @staticmethod
    def evaluate(operator: str, value1: float, value2: float) -> bool:
        """Evalúa comparación entre dos valores."""
        return ComparisonOperator.OPERATORS[operator](value1, value2)


class LogicalOperator:
    """Operadores lógicos."""

    @staticmethod
    def AND(*conditions: bool) -> bool:
        return all(conditions)

    @staticmethod
    def OR(*conditions: bool) -> bool:
        return any(conditions)

    @staticmethod
    def NOT(condition: bool) -> bool:
        return not condition

# ============================================================================
# HELPERS
# ============================================================================

def get_available_predicates(generation: int, allow_indicators: bool = False) -> List[str]:
    """
    Predicados disponibles según generación (evolución progresiva).

    Gen 1-50:   Direct (close, open, high, low, volume)
    Gen 51-100: Direct + Ratios
    Gen 101+:   Todos (si allow_indicators=True)
    """
    direct = [name for name, info in PREDICATE_REGISTRY.items()
              if info['category'] == 'direct']
    ratios = [name for name, info in PREDICATE_REGISTRY.items()
              if info['category'] == 'ratio']
    indicators = [name for name, info in PREDICATE_REGISTRY.items()
                  if info['category'] == 'indicator']

    if generation <= 50:
        available = direct
    elif generation <= 100:
        available = direct + ratios
    else:
        if allow_indicators:
            available = direct + ratios + indicators
        else:
            available = direct + ratios

    logger.debug(f"Gen {generation}: {len(available)} predicates ({len([p for p in available if p in direct])} direct)")
    return available

def get_common_offsets() -> List[int]:
    """Offsets comunes para comparaciones."""
    return [1, 2, 3, 5, 10]  # Comparar con 1, 2, 3, 5, 10 barras atrás

def get_random_offset() -> int:
    """Offset aleatorio para comparaciones."""
    import random
    return random.randint(1, 10)
