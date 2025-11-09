"""
Pattern Templates - Seed population with trading logic
"""

from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode
import random
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# TRADING TEMPLATES
# ============================================================================

def template_momentum_volume(generation: int) -> Pattern:
    """Template: Momentum + Volume confirmation"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=1)
    else:
        pred1 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=1)

    expression = LogicalNode('AND', [pred1, pred2])

    return Pattern(
        direction=direction,
        window=random.randint(3, 6),
        expression=expression,
        generation_created=generation
    )


def template_breakout(generation: int) -> Pattern:
    """Template: Breakout with volume confirmation"""
    direction = random.choice(['LONG', 'SHORT'])
    lookback = random.randint(3, 10)

    if direction == 'LONG':
        pred1 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=lookback)
        pred2 = PredicateNode('high', '>', threshold=None, bar_offset=0, compare_with_bar=1)
    else:
        pred1 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=lookback)
        pred2 = PredicateNode('low', '<', threshold=None, bar_offset=0, compare_with_bar=1)

    pred3 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=5)
    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(4, 7),
        expression=expression,
        generation_created=generation
    )


def template_mean_reversion(generation: int) -> Pattern:
    """Template: Mean reversion with consolidation"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=5)
        pred2 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=10)
    else:
        pred1 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=5)
        pred2 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=10)

    pred3 = PredicateNode('body_pct', '<', threshold=0.01, bar_offset=0)
    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(3, 6),
        expression=expression,
        generation_created=generation
    )


def template_trend_continuation(generation: int) -> Pattern:
    """Template: Trend continuation with persistence"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '>', threshold=None, bar_offset=1, compare_with_bar=2)
        pred3 = PredicateNode('close', '>', threshold=None, bar_offset=2, compare_with_bar=3)
    else:
        pred1 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '<', threshold=None, bar_offset=1, compare_with_bar=2)
        pred3 = PredicateNode('close', '<', threshold=None, bar_offset=2, compare_with_bar=3)

    pred4 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=5)
    expression = LogicalNode('AND', [pred1, pred2, pred3, pred4])

    return Pattern(
        direction=direction,
        window=random.randint(4, 7),
        expression=expression,
        generation_created=generation
    )


def template_volume_spike(generation: int) -> Pattern:
    """Template: Volume spike with direction"""
    direction = random.choice(['LONG', 'SHORT'])

    pred1 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=5)
    pred2 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=10)

    if direction == 'LONG':
        pred3 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
    else:
        pred3 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)

    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(3, 6),
        expression=expression,
        generation_created=generation
    )


def template_range_breakout(generation: int) -> Pattern:
    """Template: Range breakout with large body"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('high', '>', threshold=None, bar_offset=0, compare_with_bar=5)
        pred2 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=5)
    else:
        pred1 = PredicateNode('low', '<', threshold=None, bar_offset=0, compare_with_bar=5)
        pred2 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=5)

    pred3 = PredicateNode('body_pct', '>', threshold=0.015, bar_offset=0)
    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(4, 7),
        expression=expression,
        generation_created=generation
    )


def template_gap_move(generation: int) -> Pattern:
    """Template: Gap behavior with volume"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('open', '>', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
    else:
        pred1 = PredicateNode('open', '<', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)

    pred3 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=3)
    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(3, 5),
        expression=expression,
        generation_created=generation
    )


def template_volatility_expansion(generation: int) -> Pattern:
    """Template: Volatility expansion with direction"""
    direction = random.choice(['LONG', 'SHORT'])

    pred1 = PredicateNode('high', '>', threshold=None, bar_offset=0, compare_with_bar=3)
    pred2 = PredicateNode('low', '<', threshold=None, bar_offset=0, compare_with_bar=3)

    if direction == 'LONG':
        pred3 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
    else:
        pred3 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)

    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(4, 6),
        expression=expression,
        generation_created=generation
    )


def template_consecutive_closes(generation: int) -> Pattern:
    """Template: Multiple consecutive closes"""
    direction = random.choice(['LONG', 'SHORT'])

    if direction == 'LONG':
        pred1 = PredicateNode('close', '>', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '>', threshold=None, bar_offset=1, compare_with_bar=2)
    else:
        pred1 = PredicateNode('close', '<', threshold=None, bar_offset=0, compare_with_bar=1)
        pred2 = PredicateNode('close', '<', threshold=None, bar_offset=1, compare_with_bar=2)

    pred3 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=5)
    expression = LogicalNode('AND', [pred1, pred2, pred3])

    return Pattern(
        direction=direction,
        window=random.randint(3, 5),
        expression=expression,
        generation_created=generation
    )


def template_climax_volume(generation: int) -> Pattern:
    """Template: Climax volume with range"""
    direction = random.choice(['LONG', 'SHORT'])

    pred1 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=10)
    pred2 = PredicateNode('volume', '>', threshold=None, bar_offset=0, compare_with_bar=20)
    pred3 = PredicateNode('range_pct', '>', threshold=0.02, bar_offset=0)

    if direction == 'LONG':
        pred4 = PredicateNode('close_position_in_range', '>', threshold=0.7, bar_offset=0)
    else:
        pred4 = PredicateNode('close_position_in_range', '<', threshold=0.3, bar_offset=0)

    expression = LogicalNode('AND', [pred1, pred2, pred3, pred4])

    return Pattern(
        direction=direction,
        window=random.randint(4, 7),
        expression=expression,
        generation_created=generation
    )


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

TEMPLATE_FUNCTIONS = [
    template_momentum_volume,
    template_breakout,
    template_mean_reversion,
    template_trend_continuation,
    template_volume_spike,
    template_range_breakout,
    template_gap_move,
    template_volatility_expansion,
    template_consecutive_closes,
    template_climax_volume
]

def generate_from_template(generation: int) -> Pattern:
    """
    Genera patrón desde template random.

    Returns:
        Pattern válido basado en lógica de trading
    """
    template_func = random.choice(TEMPLATE_FUNCTIONS)
    pattern = template_func(generation)

    logger.debug(f"Generated from template: {template_func.__name__}")

    return pattern
