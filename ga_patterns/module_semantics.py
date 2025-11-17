"""
Module Semantic Classification - Directional Bias Analysis

This module classifies trading pattern modules by their directional bias:
- BULLISH: Modules that signal upward price movement (for LONG patterns)
- BEARISH: Modules that signal downward price movement (for SHORT patterns)
- NEUTRAL: Modules that work for both directions (volume, volatility, etc.)

Used by genetic operators to prevent nonsense patterns like:
- LONG patterns with all bearish modules
- SHORT patterns with all bullish modules
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DIRECTIONAL BIAS CLASSIFICATION
# ============================================================================

BULLISH_MODULES = {
    # Upward momentum
    'momentum_up_2bar',
    'momentum_up_3bar',
    'momentum_up_5bar',

    # Breakouts above
    'breakout_high_short',
    'breakout_high_long',

    # Mean reversion from oversold
    'rsi_oversold_30',
    'rsi_oversold_40',
    'rsi_rising',
    'stoch_oversold',

    # Price above MA (bullish trend)
    'price_above_sma20',
    'price_above_sma50',
    'golden_cross',

    # MACD signals
    'macd_bullish_cross',
    'macd_histogram_rising',

    # Bollinger
    'bb_lower_touch',  # Bounce from lower band

    # Gap up
    'gap_up',

    # Candle position
    'close_near_high',  # Bullish candle
}

BEARISH_MODULES = {
    # Downward momentum
    'momentum_down_2bar',
    'momentum_down_3bar',
    'momentum_down_5bar',
    'momentum_down_strong',

    # Breakdowns below
    'breakout_low_short',
    'breakout_low_long',

    # Mean reversion from overbought
    'rsi_overbought_60',
    'rsi_overbought_70',
    'rsi_falling',
    'stoch_overbought',

    # Price below MA (bearish trend)
    'price_below_sma20',
    'price_below_sma50',
    'death_cross',

    # MACD signals
    'macd_bearish_cross',
    'macd_histogram_falling',

    # Bollinger
    'bb_upper_touch',  # Rejection from upper band

    # Gap down
    'gap_down',

    # Candle position
    'close_near_low',  # Bearish candle

    # Short-specific mean reversion patterns
    'overbought_pullback_short',
    'exhaustion_top_short',
    'failed_breakout_short',
    'volume_climax_short',
    'rejection_from_resistance_short',
    'weak_bounce_short',
    'lower_highs_short',
}

NEUTRAL_MODULES = {
    # Volume (can work both ways)
    'volume_spike_short',
    'volume_spike_long',
    'volume_declining',
    'volume_climax',

    # Volatility (context-dependent)
    'volatility_high',
    'volatility_low',
    'volatility_expansion',
    'volatility_contraction',

    # Body size (depends on direction context)
    'large_body',
    'medium_body',
    'small_body',

    # Candle position (can be used contextually)
    'close_middle',

    # ATR
    'atr_expanding',
    'atr_contracting',
}


def get_module_bias(module_name: str) -> str:
    """
    Return directional bias of a module.

    Args:
        module_name: Name of module

    Returns:
        'BULLISH', 'BEARISH', or 'NEUTRAL'

    Example:
        >>> get_module_bias('momentum_up_2bar')
        'BULLISH'
        >>> get_module_bias('momentum_down_strong')
        'BEARISH'
        >>> get_module_bias('volume_climax')
        'NEUTRAL'
    """
    if module_name in BULLISH_MODULES:
        return 'BULLISH'
    elif module_name in BEARISH_MODULES:
        return 'BEARISH'
    elif module_name in NEUTRAL_MODULES:
        return 'NEUTRAL'
    else:
        logger.warning(f"Unknown module bias for '{module_name}', assuming NEUTRAL")
        return 'NEUTRAL'


def get_compatible_modules(direction: str, available_modules: List[str]) -> List[str]:
    """
    Filter modules compatible with pattern direction.

    Args:
        direction: 'LONG' or 'SHORT'
        available_modules: List of module names to filter

    Returns:
        List of compatible module names

    Logic:
        - LONG: Can use BULLISH + NEUTRAL modules
        - SHORT: Can use BEARISH + NEUTRAL modules

    Example:
        >>> modules = ['momentum_up_2bar', 'momentum_down_2bar', 'volume_climax']
        >>> get_compatible_modules('LONG', modules)
        ['momentum_up_2bar', 'volume_climax']
        >>> get_compatible_modules('SHORT', modules)
        ['momentum_down_2bar', 'volume_climax']
    """
    compatible = []

    for module in available_modules:
        bias = get_module_bias(module)

        if direction == 'LONG':
            # LONG patterns use bullish and neutral modules
            if bias in ['BULLISH', 'NEUTRAL']:
                compatible.append(module)
        elif direction == 'SHORT':
            # SHORT patterns use bearish and neutral modules
            if bias in ['BEARISH', 'NEUTRAL']:
                compatible.append(module)

    return compatible


def calculate_pattern_bias_score(modules: List[str], direction: str) -> float:
    """
    Calculate how well modules align with pattern direction.

    Args:
        modules: List of module names
        direction: 'LONG' or 'SHORT'

    Returns:
        Score between 0.0 and 1.0:
        - 1.0 = Perfect alignment (all modules support direction)
        - 0.5 = Neutral (no bias either way)
        - 0.0 = Complete contradiction (all modules oppose direction)

    Logic:
        - For LONG: +1 for bullish, +0.5 for neutral, 0 for bearish
        - For SHORT: +1 for bearish, +0.5 for neutral, 0 for bullish

    Example:
        >>> # LONG with all bullish modules
        >>> calculate_pattern_bias_score(['momentum_up_2bar', 'rsi_oversold_30'], 'LONG')
        1.0

        >>> # LONG with bearish modules (nonsense pattern)
        >>> calculate_pattern_bias_score(['momentum_down_2bar', 'rsi_overbought_70'], 'LONG')
        0.0

        >>> # LONG with mixed modules
        >>> calculate_pattern_bias_score(['momentum_up_2bar', 'volume_climax'], 'LONG')
        0.75
    """
    if not modules:
        return 0.5

    total_score = 0.0

    for module in modules:
        bias = get_module_bias(module)

        if direction == 'LONG':
            if bias == 'BULLISH':
                total_score += 1.0
            elif bias == 'NEUTRAL':
                total_score += 0.5
            # BEARISH gets 0.0
        elif direction == 'SHORT':
            if bias == 'BEARISH':
                total_score += 1.0
            elif bias == 'NEUTRAL':
                total_score += 0.5
            # BULLISH gets 0.0

    # Normalize to 0-1
    return total_score / len(modules)


def is_pattern_semantically_valid(modules: List[str], direction: str,
                                  min_bias_score: float = 0.5) -> bool:
    """
    Check if pattern makes semantic sense.

    Args:
        modules: List of module names
        direction: 'LONG' or 'SHORT'
        min_bias_score: Minimum required bias score (default 0.5 = at least neutral)

    Returns:
        True if pattern is valid, False if nonsense

    Example:
        >>> # Valid LONG pattern
        >>> is_pattern_semantically_valid(['momentum_up_2bar', 'volume_climax'], 'LONG')
        True

        >>> # Invalid LONG pattern (all bearish modules)
        >>> is_pattern_semantically_valid(['momentum_down_2bar', 'rsi_overbought_70'], 'LONG')
        False
    """
    bias_score = calculate_pattern_bias_score(modules, direction)

    if bias_score < min_bias_score:
        logger.debug(f"Pattern fails semantic validation: "
                    f"{direction} with modules {modules} (bias score: {bias_score:.2f})")
        return False

    return True


def check_redundant_modules(modules: List[str]) -> List[str]:
    """
    Find redundant modules from same family.

    Args:
        modules: List of module names

    Returns:
        List of redundant module names (empty if none found)

    Example:
        >>> # Redundant: 3 momentum variants
        >>> check_redundant_modules(['momentum_up_2bar', 'momentum_up_3bar', 'momentum_up_5bar'])
        ['momentum_up_3bar', 'momentum_up_5bar']

        >>> # Valid: different families
        >>> check_redundant_modules(['momentum_up_2bar', 'volume_climax', 'rsi_oversold_30'])
        []
    """
    from collections import defaultdict

    # Group modules by family prefix
    families = defaultdict(list)

    for module in modules:
        # Extract family prefix (before last underscore+number)
        # e.g., 'momentum_up_2bar' -> 'momentum_up'
        # e.g., 'volume_climax' -> 'volume_climax'

        # Special handling for numbered variants
        if any(module.endswith(f'_{n}bar') for n in [2, 3, 5]):
            # momentum_up_2bar -> momentum_up
            family = '_'.join(module.split('_')[:-1])
        elif any(module.endswith(f'_{n}') for n in [20, 30, 40, 50, 60, 70]):
            # rsi_oversold_30 -> rsi_oversold
            family = '_'.join(module.split('_')[:-1])
        else:
            # No variant, use full name as family
            family = module

        families[family].append(module)

    # Find families with >1 module (redundant)
    redundant = []
    for family, members in families.items():
        if len(members) > 1:
            # Keep first, mark rest as redundant
            redundant.extend(members[1:])

    if redundant:
        logger.debug(f"Found redundant modules: {redundant}")

    return redundant


# ============================================================================
# MODULE FILTERING FOR GENETIC OPERATORS
# ============================================================================

def filter_modules_for_direction(modules: List[str], direction: str,
                                 max_opposite_ratio: float = 0.2) -> List[str]:
    """
    Remove modules that contradict the pattern direction.

    Args:
        modules: List of module names
        direction: 'LONG' or 'SHORT'
        max_opposite_ratio: Maximum allowed ratio of opposite-bias modules (default 0.2 = 20%)

    Returns:
        Filtered list with contradictory modules removed

    Example:
        >>> # LONG pattern with 1 bearish module
        >>> filter_modules_for_direction(
        ...     ['momentum_up_2bar', 'momentum_down_2bar', 'volume_climax'],
        ...     'LONG'
        ... )
        ['momentum_up_2bar', 'volume_climax']
    """
    compatible = []
    incompatible = []

    for module in modules:
        bias = get_module_bias(module)

        if direction == 'LONG':
            if bias in ['BULLISH', 'NEUTRAL']:
                compatible.append(module)
            else:
                incompatible.append(module)
        elif direction == 'SHORT':
            if bias in ['BEARISH', 'NEUTRAL']:
                compatible.append(module)
            else:
                incompatible.append(module)

    # Allow small number of opposite modules (noise tolerance)
    if len(modules) > 0:
        opposite_ratio = len(incompatible) / len(modules)
        if opposite_ratio > max_opposite_ratio:
            logger.debug(f"Filtered {len(incompatible)} incompatible modules from {direction} pattern")
            return compatible

    # If ratio is acceptable, keep all modules
    return modules


if __name__ == '__main__':
    # Test semantic validation
    print("="*70)
    print("MODULE SEMANTIC CLASSIFICATION TEST")
    print("="*70)

    # Test 1: Valid LONG pattern
    print("\nTest 1: Valid LONG pattern")
    modules = ['momentum_up_2bar', 'rsi_oversold_30', 'volume_climax']
    direction = 'LONG'
    bias_score = calculate_pattern_bias_score(modules, direction)
    is_valid = is_pattern_semantically_valid(modules, direction)
    print(f"  Modules: {modules}")
    print(f"  Direction: {direction}")
    print(f"  Bias score: {bias_score:.2f}")
    print(f"  Valid: {is_valid}")

    # Test 2: Invalid LONG pattern (bearish modules)
    print("\nTest 2: Invalid LONG pattern (all bearish)")
    modules = ['momentum_down_2bar', 'rsi_overbought_70', 'close_near_low']
    direction = 'LONG'
    bias_score = calculate_pattern_bias_score(modules, direction)
    is_valid = is_pattern_semantically_valid(modules, direction)
    print(f"  Modules: {modules}")
    print(f"  Direction: {direction}")
    print(f"  Bias score: {bias_score:.2f}")
    print(f"  Valid: {is_valid}")

    # Test 3: Valid SHORT pattern
    print("\nTest 3: Valid SHORT pattern")
    modules = ['momentum_down_strong', 'failed_breakout_short', 'volume_climax']
    direction = 'SHORT'
    bias_score = calculate_pattern_bias_score(modules, direction)
    is_valid = is_pattern_semantically_valid(modules, direction)
    print(f"  Modules: {modules}")
    print(f"  Direction: {direction}")
    print(f"  Bias score: {bias_score:.2f}")
    print(f"  Valid: {is_valid}")

    # Test 4: Redundant modules
    print("\nTest 4: Redundant modules")
    modules = ['momentum_up_2bar', 'momentum_up_3bar', 'momentum_up_5bar']
    redundant = check_redundant_modules(modules)
    print(f"  Modules: {modules}")
    print(f"  Redundant: {redundant}")

    # Test 5: Compatible module filtering
    print("\nTest 5: Compatible module filtering")
    all_modules = ['momentum_up_2bar', 'momentum_down_2bar', 'volume_climax',
                   'rsi_oversold_30', 'rsi_overbought_70']
    long_compatible = get_compatible_modules('LONG', all_modules)
    short_compatible = get_compatible_modules('SHORT', all_modules)
    print(f"  All modules: {all_modules}")
    print(f"  LONG compatible: {long_compatible}")
    print(f"  SHORT compatible: {short_compatible}")

    print(f"\n{'='*70}\n")
