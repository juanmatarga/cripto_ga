"""
Pattern Generator V2 - Building Blocks System

SPRINT 13: Added semantic validation to prevent nonsense patterns.

Generates random PatternChromosomes using building block modules.
Respects progressive complexity unlocking.

Key improvements:
- Filters modules by direction compatibility before sampling
- Validates semantic coherence (LONG uses bullish, SHORT uses bearish)
- Prevents redundant modules from same family
"""

import random
from typing import List
import logging

from ga_patterns.chromosome_v2 import PatternChromosome, validate_chromosome
from ga_patterns.building_blocks import get_available_modules
from ga_patterns.module_semantics import (
    get_compatible_modules,
    is_pattern_semantically_valid,
    check_redundant_modules
)

logger = logging.getLogger(__name__)


def validate_pattern_logic(pattern: PatternChromosome) -> bool:
    """
    Validate that pattern modules make logical sense together (SPRINT 12).

    Rules:
        1. Direction coherence (LONG shouldn't have bearish modules)
        2. No redundancy (max one from each module family)
        3. Avoid overly simple patterns

    Returns:
        bool: True if pattern is logically valid

    Example:
        >>> pattern = PatternChromosome(direction='LONG',
        ...     modules=['momentum_up_2bar', 'rsi_oversold_30'], logic='AND')
        >>> validate_pattern_logic(pattern)
        True
        >>> bad = PatternChromosome(direction='LONG',
        ...     modules=['momentum_down_3bar', 'rsi_overbought_70'], logic='AND')
        >>> validate_pattern_logic(bad)
        False
    """
    # Define module classifications
    BULLISH_MODULES = [
        'momentum_up_2bar', 'momentum_up_3bar', 'momentum_up_5bar',
        'rsi_oversold_30', 'rsi_oversold_40', 'rsi_rising',
        'price_above_sma20', 'price_above_sma50',
        'breakout_high_long', 'breakout_high_short',
        'macd_bullish_cross', 'bb_lower_touch',
        'close_near_high'
    ]

    BEARISH_MODULES = [
        'momentum_down_2bar', 'momentum_down_3bar', 'momentum_down_5bar',
        'rsi_overbought_60', 'rsi_overbought_70', 'rsi_falling',
        'price_below_sma20', 'price_below_sma50',
        'breakout_low_short', 'momentum_down_strong',
        'macd_bearish_cross', 'bb_upper_touch',
        'overbought_pullback_short', 'exhaustion_top_short',
        'failed_breakout_short', 'rejection_from_resistance_short',
        'close_near_low', 'weak_bounce_short', 'lower_highs_short'
    ]

    # Rule 1: Direction coherence
    if pattern.direction == 'LONG':
        # Should not have primarily bearish modules
        bearish_count = sum(1 for m in pattern.modules if m in BEARISH_MODULES)
        if bearish_count > len(pattern.modules) // 2:
            logger.debug(f"Rejected LONG pattern with {bearish_count}/{len(pattern.modules)} bearish modules")
            return False

    elif pattern.direction == 'SHORT':
        # Should not have primarily bullish modules
        bullish_count = sum(1 for m in pattern.modules if m in BULLISH_MODULES)
        if bullish_count > len(pattern.modules) // 2:
            logger.debug(f"Rejected SHORT pattern with {bullish_count}/{len(pattern.modules)} bullish modules")
            return False

    # Rule 2: No redundancy (max one momentum_up variant, etc.)
    momentum_up = [m for m in pattern.modules if 'momentum_up' in m]
    if len(momentum_up) > 1:
        logger.debug(f"Rejected pattern with multiple momentum_up modules: {momentum_up}")
        return False

    momentum_down = [m for m in pattern.modules if 'momentum_down' in m]
    if len(momentum_down) > 1:
        logger.debug(f"Rejected pattern with multiple momentum_down modules: {momentum_down}")
        return False

    volume_spike = [m for m in pattern.modules if 'volume_spike' in m]
    if len(volume_spike) > 1:
        logger.debug(f"Rejected pattern with multiple volume_spike modules: {volume_spike}")
        return False

    # Rule 3: Avoid overly simple patterns
    if len(pattern.modules) < 2:
        logger.debug(f"Rejected pattern with only {len(pattern.modules)} module(s)")
        return False

    return True


def generate_random_chromosome(generation: int, config: dict) -> PatternChromosome:
    """
    Generate a random PatternChromosome using available modules.

    Args:
        generation: Current generation number (determines available modules)
        config: Config dict with GA settings

    Returns:
        PatternChromosome: Valid random chromosome

    Process:
        1. Get available modules for this generation
        2. Randomly select 2-5 modules
        3. Choose random direction, logic, window
        4. Create and validate chromosome
        5. Retry if invalid (shouldn't happen)
    """
    max_attempts = 100

    for attempt in range(max_attempts):
        # SPRINT 13: Choose direction FIRST, then filter compatible modules
        direction = random.choice(['LONG', 'SHORT'])

        # Get available modules
        allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
        available_modules = get_available_modules(generation, allow_indicators, config)  # AUDIT FIX: Pass config

        # SPRINT 13: Filter modules compatible with direction
        compatible_modules = get_compatible_modules(direction, list(available_modules.keys()))

        if len(compatible_modules) < 2:
            logger.warning(f"Not enough compatible modules for {direction}, attempt {attempt+1}")
            continue

        # Determine number of modules based on generation
        if generation < 30:
            n_modules = random.randint(2, 3)
        elif generation < 80:
            n_modules = random.randint(2, 4)
        else:
            n_modules = random.randint(3, 5)

        # Ensure we have enough modules
        n_modules = min(n_modules, len(compatible_modules))

        # Sample COMPATIBLE modules only
        module_names = random.sample(compatible_modules, n_modules)

        # Random logic
        logic_options = ['AND', 'OR']
        if n_modules >= 2:
            logic_options.append('2of3')
        if n_modules >= 3:
            logic_options.append('3of4')
        logic = random.choice(logic_options)

        # SPRINT 12: Dynamic window sizing based on pattern complexity
        # Simpler patterns (2 modules) -> smaller windows (less overfitting)
        # Complex patterns (4-5 modules) -> larger windows (need more data)
        if n_modules == 2:
            window = random.randint(3, 5)
        elif n_modules == 3:
            window = random.randint(4, 7)
        elif n_modules == 4:
            window = random.randint(5, 9)
        else:  # 5+ modules
            window = random.randint(6, 10)

        # SPRINT 13: Check for redundant modules before creating chromosome
        redundant = check_redundant_modules(module_names)
        if redundant:
            # Remove redundant modules
            module_names = [m for m in module_names if m not in redundant]
            if len(module_names) == 0:
                logger.debug(f"Attempt {attempt + 1}: All modules redundant, retrying...")
                continue

        # Create chromosome
        chromosome = PatternChromosome(
            direction=direction,
            modules=module_names,
            logic=logic,
            window=window,
            generation_created=generation,
            fitness=-999.0,
            fitness_long=-999.0,
            fitness_short=-999.0
        )

        # SPRINT 13: Validate syntactic AND semantic correctness
        if validate_chromosome(chromosome) and validate_pattern_logic(chromosome):
            # Check semantic validity
            if is_pattern_semantically_valid(module_names, direction, min_bias_score=0.5):
                logger.debug(f"Generated: {chromosome.to_readable()}")
                return chromosome
            else:
                logger.debug(f"Attempt {attempt + 1}: Semantically invalid, retrying...")
        else:
            logger.debug(f"Attempt {attempt + 1}: Syntactically invalid, retrying...")

    # Fallback: return simple valid chromosome
    logger.warning("Failed to generate valid random chromosome after max attempts. Using fallback.")
    return PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short'],
        logic='AND',
        window=5,
        generation_created=generation
    )


def create_seed_patterns(generation: int, config: dict) -> List[PatternChromosome]:
    """
    Create seed patterns with known trading logic (SPRINT 12).

    Purpose: Initialize 30% of population with known-good pattern archetypes
    to give GA a head start, especially for SHORT patterns.

    Args:
        generation: Current generation (determines available modules)
        config: Config dict

    Returns:
        List of ~30 seed patterns covering:
            - Mean reversion (RSI oversold/overbought)
            - Trend following (SMA + momentum)
            - Breakout (volume + range expansion)

    Example:
        >>> seeds = create_seed_patterns(0, config)
        >>> len(seeds)
        30
        >>> sum(1 for p in seeds if p.direction == 'SHORT')
        15
    """
    seeds = []

    # Get available modules
    allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
    available = get_available_modules(generation, allow_indicators, config)  # AUDIT FIX: Pass config

    # LONG seeds (15 patterns) - proven strategies
    long_seeds = [
        # Mean reversion LONG
        (['rsi_oversold_30', 'volume_spike_short'], 'AND', 5),
        (['rsi_oversold_40', 'momentum_up_2bar'], 'AND', 5),
        (['rsi_oversold_30', 'close_near_low'], 'AND', 4),

        # Trend following LONG
        (['momentum_up_3bar', 'volume_above_sma'], 'AND', 7),
        (['momentum_up_5bar', 'volume_spike_long', 'close_near_high'], 'AND', 8),

        # Breakout LONG
        (['breakout_high_long', 'volume_spike_long'], 'AND', 6),
        (['breakout_high_short', 'momentum_up_3bar', 'volume_spike_short'], 'AND', 7),

        # Volatility LONG
        (['volatility_expansion', 'momentum_up_3bar', 'volume_above_sma'], 'AND', 6),

        # Combinations LONG
        (['large_body', 'close_near_high', 'volume_spike_short'], 'AND', 5),
        (['momentum_up_3bar', 'close_near_high', 'volume_spike_short'], 'AND', 6),
        (['rsi_oversold_40', 'volume_above_sma', 'large_body'], 'AND', 5),
        (['momentum_up_2bar', 'volume_spike_short', 'volatility_expansion'], 'AND', 5),
        (['breakout_high_short', 'volume_spike_short', 'large_body'], 'AND', 6),
        (['rsi_oversold_30', 'momentum_up_3bar', 'volume_spike_short'], 'AND', 5),
        (['momentum_up_2bar', 'close_near_high'], 'AND', 4),
    ]

    # SHORT seeds (15 patterns) - mean-reversion focused for 15min
    short_seeds = [
        # Mean reversion SHORT (primary edge)
        (['rsi_overbought_70', 'volume_spike_short'], 'AND', 5),
        (['rsi_overbought_60', 'overbought_pullback_short'], 'AND', 5),
        (['rsi_overbought_70', 'exhaustion_top_short'], 'AND', 5),
        (['overbought_pullback_short', 'volume_spike_short'], 'AND', 5),
        (['rsi_overbought_60', 'rejection_from_resistance_short'], 'AND', 6),

        # Failed breakouts SHORT
        (['failed_breakout_short', 'volume_climax_short'], 'AND', 5),
        (['exhaustion_top_short', 'volume_spike_short'], 'AND', 5),
        (['failed_breakout_short', 'momentum_down_strong'], 'AND', 5),

        # Trend following SHORT (weaker but still valid)
        (['momentum_down_strong', 'volume_spike_short'], 'AND', 5),
        (['lower_highs_short', 'momentum_down_3bar'], 'AND', 5),

        # Breakout SHORT
        (['breakout_low_short', 'volume_spike_short'], 'AND', 6),

        # Combinations SHORT
        (['rsi_overbought_70', 'large_body', 'close_near_low'], 'AND', 5),
        (['overbought_pullback_short', 'rejection_from_resistance_short'], 'AND', 6),
        (['exhaustion_top_short', 'failed_breakout_short'], 'AND', 5),
        (['rsi_overbought_60', 'volume_climax_short', 'large_body'], 'AND', 5),
    ]

    # Create LONG seed patterns
    for modules, logic, window in long_seeds:
        # Check if all modules are available
        if all(m in available for m in modules):
            pattern = PatternChromosome(
                direction='LONG',
                modules=modules,
                logic=logic,
                window=window,
                generation_created=generation
            )
            seeds.append(pattern)
        else:
            logger.debug(f"Skipping LONG seed {modules} - modules not available at gen {generation}")

    # Create SHORT seed patterns
    for modules, logic, window in short_seeds:
        if all(m in available for m in modules):
            pattern = PatternChromosome(
                direction='SHORT',
                modules=modules,
                logic=logic,
                window=window,
                generation_created=generation
            )
            seeds.append(pattern)
        else:
            logger.debug(f"Skipping SHORT seed {modules} - modules not available at gen {generation}")

    logger.info(f"[SPRINT 12] Created {len(seeds)} seed patterns with known edge")
    long_seeds_count = sum(1 for p in seeds if p.direction == 'LONG')
    short_seeds_count = len(seeds) - long_seeds_count
    logger.info(f"  LONG seeds: {long_seeds_count}, SHORT seeds: {short_seeds_count}")

    return seeds


def initialize_population(population_size: int, generation: int,
                         config: dict) -> List[PatternChromosome]:
    """
    Initialize population with mix of seeds and random patterns (SPRINT 12).

    Strategy:
        - 30% seeds (known-good patterns with proven edge)
        - 70% random (exploration)

    Args:
        population_size: Number of chromosomes to generate
        generation: Generation number (should be 0 for initial population)
        config: Config dict

    Returns:
        List[PatternChromosome]: List of valid chromosomes

    Example:
        >>> config = {'ga': {'window_min': 3, 'window_max': 10}}
        >>> population = initialize_population(100, generation=0, config=config)
        >>> len(population)
        100
        >>> all(isinstance(p, PatternChromosome) for p in population)
        True
    """
    logger.info(f"Initializing population of {population_size} patterns (Generation {generation})...")

    population = []

    # SPRINT 12: Create seeds first (30% of population)
    seeds = create_seed_patterns(generation, config)
    target_seeds = min(len(seeds), int(population_size * 0.3))

    # Sample seeds if we have too many
    if len(seeds) > target_seeds:
        seeds = random.sample(seeds, target_seeds)

    population.extend(seeds)
    logger.info(f"[SPRINT 12] Added {len(seeds)} seed patterns (30% of population)")

    # Fill rest with random patterns (70%)
    remaining = population_size - len(population)
    for i in range(remaining):
        chromosome = generate_random_chromosome(generation, config)
        population.append(chromosome)

        if (i + 1) % 20 == 0:
            logger.debug(f"  Generated {len(seeds) + i + 1}/{population_size} patterns")

    # Verify all are valid
    valid_count = sum(1 for p in population if validate_chromosome(p))

    logger.info(f"[OK] Population initialized: {valid_count}/{population_size} valid patterns")

    # Log statistics
    from collections import Counter
    all_modules = [m for p in population for m in p.modules]
    module_counts = Counter(all_modules)
    top_5 = module_counts.most_common(5)

    logger.info(f"Initial module distribution:")
    for module, count in top_5:
        logger.info(f"  {module}: {count}")

    direction_counts = Counter([p.direction for p in population])
    logger.info(f"Direction split: {direction_counts['LONG']} LONG, {direction_counts['SHORT']} SHORT")

    return population


if __name__ == '__main__':
    # Test population generation
    logging.basicConfig(level=logging.INFO)

    config = {
        'ga': {
            'window_min': 3,
            'window_max': 10,
            'building_blocks': {
                'allow_indicators': True
            }
        }
    }

    print("="*70)
    print("TESTING POPULATION GENERATION")
    print("="*70)

    # Test Gen 0
    print("\nGeneration 0 (Base modules only):")
    pop_0 = initialize_population(20, generation=0, config=config)
    print(f"Sample patterns:")
    for i, p in enumerate(pop_0[:5]):
        print(f"  {i+1}. {p.to_readable()}")

    # Test Gen 30
    print("\nGeneration 30 (With indicators):")
    pop_30 = initialize_population(20, generation=30, config=config)
    print(f"Sample patterns:")
    for i, p in enumerate(pop_30[:5]):
        print(f"  {i+1}. {p.to_readable()}")

    # Test Gen 80
    print("\nGeneration 80 (With advanced modules):")
    pop_80 = initialize_population(20, generation=80, config=config)
    print(f"Sample patterns:")
    for i, p in enumerate(pop_80[:5]):
        print(f"  {i+1}. {p.to_readable()}")

    print(f"\n{'='*70}\n")
