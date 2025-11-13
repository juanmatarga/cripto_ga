"""
Building Blocks for Pattern Generation

This module defines pre-built trading pattern modules organized by complexity.
Each module represents a testable hypothesis about market behavior.

Expression Notation:
    - C[n], O[n], H[n], L[n], V[n]: OHLCV at offset n (0=current, 1=prev, etc.)
    - Body%[n], Range%[n], ClosePos[n]: Derived OHLCV features
    - RSI[period][n], SMA[period][n]: Indicators (Gen 30+)
    - MACD[n], BB_Upper[n], ATR[period][n]: Advanced indicators (Gen 80+)
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# BASE MODULES (min_generation=0) - Available from start
# ============================================================================

BASE_MODULES = {
    # MOMENTUM (6 modules)
    'momentum_up_2bar': {
        'expression': 'C[0] > C[1] AND C[1] > C[2]',
        'description': 'Price rising for 2 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_up_3bar': {
        'expression': 'C[0] > C[1] AND C[1] > C[2] AND C[2] > C[3]',
        'description': 'Price rising for 3 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_up_5bar': {
        'expression': 'C[0] > C[1] AND C[1] > C[2] AND C[2] > C[3] AND C[3] > C[4] AND C[4] > C[5]',
        'description': 'Price rising for 5 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_down_2bar': {
        'expression': 'C[0] < C[1] AND C[1] < C[2]',
        'description': 'Price falling for 2 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_down_3bar': {
        'expression': 'C[0] < C[1] AND C[1] < C[2] AND C[2] < C[3]',
        'description': 'Price falling for 3 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_down_5bar': {
        'expression': 'C[0] < C[1] AND C[1] < C[2] AND C[2] < C[3] AND C[3] < C[4] AND C[4] < C[5]',
        'description': 'Price falling for 5 consecutive bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },

    # VOLUME (4 modules)
    'volume_spike_short': {
        'expression': 'V[0] > V[3] AND V[0] > V[5]',
        'description': 'Volume spike vs recent bars (3-5 bars back)',
        'category': 'volume',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volume_spike_long': {
        'expression': 'V[0] > V[10] AND V[0] > V[15]',
        'description': 'Volume spike vs longer lookback (10-15 bars)',
        'category': 'volume',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volume_declining': {
        'expression': 'V[0] < V[1] AND V[1] < V[2] AND V[2] < V[3]',
        'description': 'Volume declining for 3 bars',
        'category': 'volume',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volume_climax': {
        'expression': 'V[0] > V[1] * 2.0',
        'description': 'Volume climax (>2x previous bar)',
        'category': 'volume',
        'complexity': 'simple',
        'min_generation': 0
    },

    # BREAKOUT (4 modules)
    'breakout_high_short': {
        'expression': 'H[0] > H[1] AND H[0] > H[2] AND H[0] > H[3]',
        'description': 'Breakout above recent highs (3 bars)',
        'category': 'breakout',
        'complexity': 'simple',
        'min_generation': 0
    },
    'breakout_high_long': {
        'expression': 'H[0] >= H[5] AND H[0] >= H[10] AND H[0] >= H[15]',
        'description': 'Breakout above longer-term highs (15 bars)',
        'category': 'breakout',
        'complexity': 'simple',
        'min_generation': 0
    },
    'breakout_low_short': {
        'expression': 'L[0] < L[1] AND L[0] < L[2] AND L[0] < L[3]',
        'description': 'Breakdown below recent lows (3 bars)',
        'category': 'breakout',
        'complexity': 'simple',
        'min_generation': 0
    },
    'breakout_low_long': {
        'expression': 'L[0] <= L[5] AND L[0] <= L[10] AND L[0] <= L[15]',
        'description': 'Breakdown below longer-term lows (15 bars)',
        'category': 'breakout',
        'complexity': 'simple',
        'min_generation': 0
    },

    # BODY (3 modules)
    'large_body': {
        'expression': 'Body%[0] > 0.012',
        'description': 'Large candle body (>1.2% of price)',
        'category': 'body',
        'complexity': 'simple',
        'min_generation': 0
    },
    'medium_body': {
        'expression': 'Body%[0] >= 0.006 AND Body%[0] <= 0.012',
        'description': 'Medium candle body (0.6-1.2% of price)',
        'category': 'body',
        'complexity': 'simple',
        'min_generation': 0
    },
    'small_body': {
        'expression': 'Body%[0] < 0.004',
        'description': 'Small candle body (<0.4% of price)',
        'category': 'body',
        'complexity': 'simple',
        'min_generation': 0
    },

    # GAP (2 modules)
    'gap_up': {
        'expression': 'L[0] > H[1]',
        'description': 'Gap up (low above previous high)',
        'category': 'gap',
        'complexity': 'simple',
        'min_generation': 0
    },
    'gap_down': {
        'expression': 'H[0] < L[1]',
        'description': 'Gap down (high below previous low)',
        'category': 'gap',
        'complexity': 'simple',
        'min_generation': 0
    },

    # VOLATILITY (4 modules)
    'volatility_high': {
        'expression': 'Range%[0] > 0.02',
        'description': 'High volatility (range >2% of price)',
        'category': 'volatility',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volatility_low': {
        'expression': 'Range%[0] < 0.008',
        'description': 'Low volatility (range <0.8% of price)',
        'category': 'volatility',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volatility_expansion': {
        'expression': 'Range%[0] > Range%[1] AND Range%[1] > Range%[2]',
        'description': 'Volatility expanding (2 bars)',
        'category': 'volatility',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volatility_contraction': {
        'expression': 'Range%[0] < Range%[1] AND Range%[1] < Range%[2]',
        'description': 'Volatility contracting (2 bars)',
        'category': 'volatility',
        'complexity': 'simple',
        'min_generation': 0
    },

    # POSITION (3 modules)
    'close_near_high': {
        'expression': 'ClosePos[0] > 0.75',
        'description': 'Close near high of bar (>75% of range)',
        'category': 'position',
        'complexity': 'simple',
        'min_generation': 0
    },
    'close_near_low': {
        'expression': 'ClosePos[0] < 0.25',
        'description': 'Close near low of bar (<25% of range)',
        'category': 'position',
        'complexity': 'simple',
        'min_generation': 0
    },
    'close_middle': {
        'expression': 'ClosePos[0] >= 0.4 AND ClosePos[0] <= 0.6',
        'description': 'Close in middle of bar (40-60% of range)',
        'category': 'position',
        'complexity': 'simple',
        'min_generation': 0
    },

    # ========================================================================
    # SPRINT 12: MEAN-REVERSION SHORT MODULES (8 modules)
    # Designed specifically for 15min timeframe where trends are noisy
    # ========================================================================
    'overbought_pullback_short': {
        'expression': 'C[0] > C[1] AND C[1] > C[2] AND RSI[14][0] > 65 AND C[0] < H[0] * 0.998',
        'description': 'Overbought pullback - price rising but RSI high and failing at highs',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'exhaustion_top_short': {
        'expression': 'C[0] < O[0] AND Body%[0] > 0.008 AND H[0] > H[1] AND H[0] > H[2]',
        'description': 'Bearish engulfing at new high - exhaustion',
        'category': 'pattern',
        'complexity': 'simple',
        'min_generation': 0
    },
    'failed_breakout_short': {
        'expression': 'H[0] > H[1] AND H[1] > H[2] AND C[0] < C[1] AND V[0] < V[1]',
        'description': 'Failed breakout - new high but close lower on decreasing volume',
        'category': 'pattern',
        'complexity': 'simple',
        'min_generation': 0
    },
    'volume_climax_short': {
        'expression': 'V[0] > SMA_V[20][0] * 2.0 AND C[0] < O[0] AND Range%[0] > 0.015',
        'description': 'Volume climax down - selling pressure spike',
        'category': 'volume',
        'complexity': 'simple',
        'min_generation': 0
    },
    'rejection_from_resistance_short': {
        'expression': 'H[0] > SMA[20][0] AND C[0] < SMA[20][0] AND Body%[0] > 0.006',
        'description': 'Price rejected from SMA20 resistance with strong bearish candle',
        'category': 'pattern',
        'complexity': 'simple',
        'min_generation': 0
    },
    'momentum_down_strong': {
        'expression': 'C[0] < C[1] * 0.995 AND C[1] < C[2] * 0.995',
        'description': 'Strong momentum down - 0.5%+ drops for 2 bars',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'weak_bounce_short': {
        'expression': 'C[0] > C[1] AND C[1] < C[2] AND C[0] < C[2] AND V[0] < V[1]',
        'description': 'Weak bounce after drop - lower volume retracement',
        'category': 'momentum',
        'complexity': 'simple',
        'min_generation': 0
    },
    'lower_highs_short': {
        'expression': 'H[0] < H[1] AND H[1] < H[2]',
        'description': 'Lower highs pattern - bearish structure',
        'category': 'pattern',
        'complexity': 'simple',
        'min_generation': 0
    },
}

# ============================================================================
# INDICATOR MODULES (min_generation=30) - Available from Gen 30+
# ============================================================================

INDICATOR_MODULES = {
    # RSI (6 modules)
    'rsi_oversold_30': {
        'expression': 'RSI[14][0] < 30',
        'description': 'RSI below 30 (deeply oversold)',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },
    'rsi_oversold_40': {
        'expression': 'RSI[14][0] < 40',
        'description': 'RSI below 40 (oversold)',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },
    'rsi_overbought_60': {
        'expression': 'RSI[14][0] > 60',
        'description': 'RSI above 60 (overbought)',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },
    'rsi_overbought_70': {
        'expression': 'RSI[14][0] > 70',
        'description': 'RSI above 70 (deeply overbought)',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },
    'rsi_rising': {
        'expression': 'RSI[14][0] > RSI[14][1] AND RSI[14][1] > RSI[14][2]',
        'description': 'RSI rising for 2 bars',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },
    'rsi_falling': {
        'expression': 'RSI[14][0] < RSI[14][1] AND RSI[14][1] < RSI[14][2]',
        'description': 'RSI falling for 2 bars',
        'category': 'rsi',
        'complexity': 'medium',
        'min_generation': 30
    },

    # MOVING AVERAGE (5 modules)
    'price_above_sma20': {
        'expression': 'C[0] > SMA[20][0]',
        'description': 'Price above 20-period SMA',
        'category': 'ma',
        'complexity': 'medium',
        'min_generation': 30
    },
    'price_below_sma20': {
        'expression': 'C[0] < SMA[20][0]',
        'description': 'Price below 20-period SMA',
        'category': 'ma',
        'complexity': 'medium',
        'min_generation': 30
    },
    'price_above_sma50': {
        'expression': 'C[0] > SMA[50][0]',
        'description': 'Price above 50-period SMA',
        'category': 'ma',
        'complexity': 'medium',
        'min_generation': 30
    },
    'price_below_sma50': {
        'expression': 'C[0] < SMA[50][0]',
        'description': 'Price below 50-period SMA',
        'category': 'ma',
        'complexity': 'medium',
        'min_generation': 30
    },
    'volume_above_sma': {
        'expression': 'V[0] > SMA_V[20][0]',
        'description': 'Volume above its 20-period SMA',
        'category': 'ma',
        'complexity': 'medium',
        'min_generation': 30
    },
}

# ============================================================================
# ADVANCED MODULES (min_generation=80) - Available from Gen 80+
# ============================================================================

ADVANCED_MODULES = {
    # MACD (4 modules)
    'macd_bullish_cross': {
        'expression': 'MACD[0] > Signal[0] AND MACD[1] <= Signal[1]',
        'description': 'MACD crosses above signal line',
        'category': 'macd',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'macd_bearish_cross': {
        'expression': 'MACD[0] < Signal[0] AND MACD[1] >= Signal[1]',
        'description': 'MACD crosses below signal line',
        'category': 'macd',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'macd_positive': {
        'expression': 'MACD[0] > 0',
        'description': 'MACD above zero line',
        'category': 'macd',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'macd_histogram_growing': {
        'expression': 'MACDHist[0] > MACDHist[1] AND MACDHist[1] > MACDHist[2]',
        'description': 'MACD histogram growing for 2 bars',
        'category': 'macd',
        'complexity': 'advanced',
        'min_generation': 80
    },

    # BOLLINGER BANDS (4 modules)
    'bb_lower_touch': {
        'expression': 'C[0] < BB_Lower[0] * 1.01',
        'description': 'Price touching lower Bollinger Band',
        'category': 'bollinger',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'bb_upper_touch': {
        'expression': 'C[0] > BB_Upper[0] * 0.99',
        'description': 'Price touching upper Bollinger Band',
        'category': 'bollinger',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'bb_squeeze': {
        'expression': 'BB_Width[0] < BB_Width_SMA[20][0] * 0.7',
        'description': 'Bollinger Bands squeeze (width <70% of average)',
        'category': 'bollinger',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'bb_expansion': {
        'expression': 'BB_Width[0] > BB_Width_SMA[20][0] * 1.3',
        'description': 'Bollinger Bands expansion (width >130% of average)',
        'category': 'bollinger',
        'complexity': 'advanced',
        'min_generation': 80
    },

    # ATR (3 modules)
    'atr_high': {
        'expression': 'ATR[14][0] > ATR_SMA[20][0] * 1.5',
        'description': 'High ATR (>150% of its average)',
        'category': 'atr',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'atr_low': {
        'expression': 'ATR[14][0] < ATR_SMA[20][0] * 0.7',
        'description': 'Low ATR (<70% of its average)',
        'category': 'atr',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'atr_expanding': {
        'expression': 'ATR[14][0] > ATR[14][1] AND ATR[14][1] > ATR[14][2]',
        'description': 'ATR expanding for 2 bars',
        'category': 'atr',
        'complexity': 'advanced',
        'min_generation': 80
    },

    # STOCHASTIC (3 modules)
    'stoch_oversold': {
        'expression': 'Stoch_K[0] < 20 AND Stoch_D[0] < 20',
        'description': 'Stochastic oversold (K and D < 20)',
        'category': 'stochastic',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'stoch_overbought': {
        'expression': 'Stoch_K[0] > 80 AND Stoch_D[0] > 80',
        'description': 'Stochastic overbought (K and D > 80)',
        'category': 'stochastic',
        'complexity': 'advanced',
        'min_generation': 80
    },
    'stoch_cross_bull': {
        'expression': 'Stoch_K[0] > Stoch_D[0] AND Stoch_K[1] <= Stoch_D[1]',
        'description': 'Stochastic K crosses above D (bullish)',
        'category': 'stochastic',
        'complexity': 'advanced',
        'min_generation': 80
    },
}

# ============================================================================
# MERGED DICTIONARY
# ============================================================================

ALL_MODULES = {**BASE_MODULES, **INDICATOR_MODULES, **ADVANCED_MODULES}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_modules(generation: int, allow_indicators: bool = True) -> Dict[str, Dict]:
    """
    Return modules available at given generation.

    Args:
        generation: Current generation number (0-150)
        allow_indicators: If False, only return BASE_MODULES

    Returns:
        Dict of module_name → module_info

    Example:
        >>> modules = get_available_modules(0)
        >>> len(modules)  # Only BASE_MODULES
        25

        >>> modules = get_available_modules(50)
        >>> len(modules)  # BASE + INDICATOR
        36

        >>> modules = get_available_modules(100)
        >>> len(modules)  # All modules
        50
    """
    available = {**BASE_MODULES}

    if allow_indicators:
        if generation >= 30:
            available.update(INDICATOR_MODULES)
        if generation >= 80:
            available.update(ADVANCED_MODULES)

    logger.debug(f"Generation {generation}: {len(available)} modules available")
    return available


def get_module_family(module_name: str) -> List[str]:
    """
    Return modules in same category (for intelligent mutation).

    Args:
        module_name: Name of module

    Returns:
        List of module names with same 'category' field

    Example:
        >>> get_module_family('momentum_up_2bar')
        ['momentum_up_2bar', 'momentum_up_3bar', 'momentum_up_5bar',
         'momentum_down_2bar', 'momentum_down_3bar', 'momentum_down_5bar']

        >>> get_module_family('rsi_oversold_30')
        ['rsi_oversold_30', 'rsi_oversold_40', 'rsi_overbought_60',
         'rsi_overbought_70', 'rsi_rising', 'rsi_falling']
    """
    module_info = get_module_info(module_name)
    if not module_info:
        logger.warning(f"Unknown module: {module_name}")
        return []

    category = module_info['category']

    # Find all modules with same category
    family = [
        name for name, info in ALL_MODULES.items()
        if info['category'] == category
    ]

    logger.debug(f"Module '{module_name}' family ({category}): {len(family)} modules")
    return family


def get_module_info(module_name: str) -> Dict:
    """
    Return info dict for specific module.

    Args:
        module_name: Name of module

    Returns:
        Module dict or {} if not found

    Example:
        >>> info = get_module_info('momentum_up_2bar')
        >>> info['expression']
        'C[0] > C[1] AND C[1] > C[2]'
        >>> info['category']
        'momentum'
    """
    return ALL_MODULES.get(module_name, {})


def get_modules_by_category(category: str) -> List[str]:
    """
    Return all module names in a specific category.

    Args:
        category: Category name (e.g., 'momentum', 'rsi', 'macd')

    Returns:
        List of module names

    Example:
        >>> get_modules_by_category('momentum')
        ['momentum_up_2bar', 'momentum_up_3bar', 'momentum_up_5bar',
         'momentum_down_2bar', 'momentum_down_3bar', 'momentum_down_5bar']
    """
    return [
        name for name, info in ALL_MODULES.items()
        if info['category'] == category
    ]


def get_modules_by_complexity(complexity: str) -> List[str]:
    """
    Return all module names with specific complexity level.

    Args:
        complexity: 'simple', 'medium', or 'advanced'

    Returns:
        List of module names

    Example:
        >>> len(get_modules_by_complexity('simple'))
        25
        >>> len(get_modules_by_complexity('medium'))
        11
        >>> len(get_modules_by_complexity('advanced'))
        14
    """
    return [
        name for name, info in ALL_MODULES.items()
        if info['complexity'] == complexity
    ]


# ============================================================================
# MODULE STATISTICS
# ============================================================================

def print_module_stats():
    """Print statistics about available modules."""
    print(f"\n{'='*70}")
    print("BUILDING BLOCKS STATISTICS")
    print(f"{'='*70}")

    print(f"\nTotal modules: {len(ALL_MODULES)}")
    print(f"  - BASE (Gen 0+):     {len(BASE_MODULES)}")
    print(f"  - INDICATOR (Gen 30+): {len(INDICATOR_MODULES)}")
    print(f"  - ADVANCED (Gen 80+):  {len(ADVANCED_MODULES)}")

    # By complexity
    print(f"\nBy complexity:")
    print(f"  - Simple:   {len(get_modules_by_complexity('simple'))}")
    print(f"  - Medium:   {len(get_modules_by_complexity('medium'))}")
    print(f"  - Advanced: {len(get_modules_by_complexity('advanced'))}")

    # By category
    print(f"\nBy category:")
    categories = set(info['category'] for info in ALL_MODULES.values())
    for cat in sorted(categories):
        modules = get_modules_by_category(cat)
        print(f"  - {cat:12s}: {len(modules):2d} modules")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Print statistics when run directly
    print_module_stats()

    # Test functions
    print("Testing get_available_modules():")
    print(f"  Gen 0:   {len(get_available_modules(0))} modules")
    print(f"  Gen 30:  {len(get_available_modules(30))} modules")
    print(f"  Gen 80:  {len(get_available_modules(80))} modules")
    print(f"  Gen 150: {len(get_available_modules(150))} modules")

    print("\nTesting get_module_family():")
    family = get_module_family('momentum_up_2bar')
    print(f"  momentum_up_2bar family: {family}")

    print("\nTesting get_module_info():")
    info = get_module_info('large_body')
    print(f"  large_body: {info['expression']}")
    print(f"  description: {info['description']}")
