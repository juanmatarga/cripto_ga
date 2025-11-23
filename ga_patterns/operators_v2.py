"""
Genetic Operators V2 - Semantic-Aware Crossover and Mutation

SPRINT 13: Added semantic constraints to prevent nonsense patterns.

Key improvements:
    - Crossover: Preserves building blocks + semantic coherence
    - Mutation: Direction-aware (LONG uses bullish, SHORT uses bearish)
    - Validation: Rejects patterns with contradictory modules
    - Redundancy check: Prevents duplicate modules from same family

Prevents patterns like:
    - LONG: AND(momentum_down_strong, rsi_overbought_70) [ERROR]
    - SHORT: OR(momentum_up_2bar, rsi_oversold_30) [ERROR]
"""

import random
import copy
from typing import List
from ga_patterns.chromosome_v2 import PatternChromosome, validate_chromosome
from ga_patterns.building_blocks import get_available_modules, get_module_family
from ga_patterns.module_semantics import (
    get_compatible_modules,
    is_pattern_semantically_valid,
    filter_modules_for_direction,
    check_redundant_modules
)
import logging

logger = logging.getLogger(__name__)


def crossover(parent1: PatternChromosome, parent2: PatternChromosome,
              generation: int, config: dict) -> PatternChromosome:
    """
    SPRINT 13: Semantic-aware crossover that preserves building blocks.

    CRITICAL CHANGES:
    1. Only cross parents with SAME direction (prevent LONG/SHORT mixing)
    2. If different directions, use dominant parent's direction
    3. Filter incompatible modules after crossover
    4. Remove redundant modules from same family
    5. Validate semantic coherence before returning

    Strategy:
        1. Determine offspring direction (prefer parent1's direction)
        2. Uniform crossover on module positions
        3. Filter modules incompatible with offspring direction
        4. Remove redundant modules (e.g., momentum_up_2bar + momentum_up_3bar)
        5. Validate semantic coherence (≥50% modules support direction)

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        generation: Current generation number
        config: Config dict with GA settings

    Returns:
        PatternChromosome: Valid offspring or copy of parent1 if crossover fails

    Example:
        >>> p1 = PatternChromosome(direction='LONG', modules=['momentum_up_2bar', 'volume_spike'], logic='AND')
        >>> p2 = PatternChromosome(direction='LONG', modules=['rsi_oversold_30', 'large_body'], logic='AND')
        >>> offspring = crossover(p1, p2, generation=0, config={})
        >>> offspring.direction
        'LONG'
    """
    logger.debug(f"Crossover: P1={parent1.direction} ({len(parent1.modules)} modules), "
                f"P2={parent2.direction} ({len(parent2.modules)} modules)")

    # STEP 1: Determine offspring direction
    # Prefer same-direction crossover (80% of time use parent1's direction)
    if random.random() < 0.8:
        offspring_direction = parent1.direction
    else:
        # 20% chance to flip (exploration)
        offspring_direction = parent2.direction

    logger.debug(f"Offspring direction: {offspring_direction}")

    # STEP 2: Uniform crossover on modules (position-by-position)
    offspring_modules = []
    max_len = max(len(parent1.modules), len(parent2.modules))

    for i in range(max_len):
        if random.random() < 0.5:
            # Inherit from parent1
            if i < len(parent1.modules):
                module = parent1.modules[i]
                if module not in offspring_modules:  # No duplicates
                    offspring_modules.append(module)
        else:
            # Inherit from parent2
            if i < len(parent2.modules):
                module = parent2.modules[i]
                if module not in offspring_modules:
                    offspring_modules.append(module)

    # Fallback if no modules (shouldn't happen)
    if len(offspring_modules) == 0:
        parent = random.choice([parent1, parent2])
        if parent.modules:
            offspring_modules.append(parent.modules[0])

    logger.debug(f"After crossover: {len(offspring_modules)} modules: {offspring_modules}")

    # STEP 3: Filter incompatible modules
    # Remove modules that contradict offspring direction
    offspring_modules = filter_modules_for_direction(
        offspring_modules,
        offspring_direction,
        max_opposite_ratio=0.2  # Allow up to 20% opposite modules (noise tolerance)
    )

    logger.debug(f"After filtering: {len(offspring_modules)} modules: {offspring_modules}")

    # STEP 4: Remove redundant modules from same family
    redundant = check_redundant_modules(offspring_modules)
    if redundant:
        # Remove redundant modules
        offspring_modules = [m for m in offspring_modules if m not in redundant]
        logger.debug(f"Removed redundant modules: {redundant}")

    # Ensure at least 1 module remains
    if len(offspring_modules) == 0:
        # Emergency fallback: take first compatible module from parent1
        compatible = filter_modules_for_direction(parent1.modules, offspring_direction)
        if compatible:
            offspring_modules = [compatible[0]]
        else:
            # Last resort: use parent1's first module
            offspring_modules = [parent1.modules[0]]
        logger.warning(f"All modules filtered out, using fallback: {offspring_modules}")

    # Cap at 5 modules to prevent bloat
    if len(offspring_modules) > 5:
        offspring_modules = random.sample(offspring_modules, 5)

    # STEP 5: Inherit logic
    if random.random() < 0.7:
        offspring_logic = random.choice([parent1.logic, parent2.logic])
    else:
        # Mutate logic
        offspring_logic = random.choice(['AND', 'OR', '2of3', '3of4'])

    # Validate logic makes sense for module count
    if offspring_logic == '2of3' and len(offspring_modules) < 2:
        offspring_logic = 'AND'
    elif offspring_logic == '3of4' and len(offspring_modules) < 3:
        offspring_logic = 'AND'

    # STEP 6: Inherit window
    rand = random.random()
    if rand < 0.5:
        offspring_window = parent1.window
    elif rand < 0.8:
        offspring_window = parent2.window
    else:
        # Interpolate windows
        offspring_window = (parent1.window + parent2.window) // 2

    # STEP 7: Inherit TP/SL parameters (blend from parents)
    if random.random() < 0.7:
        # Take from one parent
        source_parent = random.choice([parent1, parent2])
        offspring_tp = source_parent.tp_atr_mult
        offspring_sl = source_parent.sl_atr_mult
    else:
        # Average parents' TP/SL
        offspring_tp = (parent1.tp_atr_mult + parent2.tp_atr_mult) / 2
        offspring_sl = (parent1.sl_atr_mult + parent2.sl_atr_mult) / 2

    # Create offspring
    offspring = PatternChromosome(
        direction=offspring_direction,
        modules=offspring_modules,
        logic=offspring_logic,
        window=offspring_window,
        generation_created=generation,
        fitness=-999.0,
        fitness_long=-999.0,
        fitness_short=-999.0,
        tp_atr_mult=offspring_tp,
        sl_atr_mult=offspring_sl
    )

    # STEP 8: Validate syntactic correctness
    if not validate_chromosome(offspring):
        logger.warning("Crossover produced invalid offspring, returning parent1")
        return copy.deepcopy(parent1)

    # STEP 9: Validate semantic coherence
    if not is_pattern_semantically_valid(offspring.modules, offspring.direction, min_bias_score=0.5):
        logger.warning(f"Crossover produced semantically invalid pattern: {offspring.to_readable()}")
        # Try to fix by filtering again with stricter threshold
        offspring.modules = filter_modules_for_direction(
            offspring.modules,
            offspring.direction,
            max_opposite_ratio=0.0  # No opposite modules
        )

        # If still empty, return parent1
        if len(offspring.modules) == 0:
            logger.warning("Cannot fix semantic issues, returning parent1")
            return copy.deepcopy(parent1)

    logger.debug(f"Crossover successful: {offspring.to_readable()}")
    return offspring


def mutate(chromosome: PatternChromosome, generation: int, config: dict) -> PatternChromosome:
    """
    SPRINT 13: Semantic-aware mutation with direction constraints.

    CRITICAL CHANGES:
    1. add_module: Only adds modules COMPATIBLE with pattern direction
    2. flip_direction: Replaces ALL modules with opposite-direction modules
    3. replace_module: Tries to use direction-compatible alternatives
    4. Validates semantic coherence after mutation

    Mutation types (with probabilities):
        - add_module (30%): Add direction-compatible module
        - remove_module (20%): Remove one module (keep minimum 1)
        - replace_module (30%): Replace with direction-compatible module
        - flip_logic (15%): Change combination logic (AND ↔ OR)
        - flip_direction (5%): Change direction + replace modules

    Args:
        chromosome: Chromosome to mutate
        generation: Current generation number
        config: Config dict

    Returns:
        PatternChromosome: Mutated chromosome or original if mutation invalid

    Example:
        >>> chrom = PatternChromosome(direction='LONG', modules=['momentum_up_2bar'], logic='AND')
        >>> mutated = mutate(chrom, generation=0, config={})
        >>> # If mutated to SHORT, modules should be bearish
        >>> mutated.direction in ['LONG', 'SHORT']
        True
    """
    mutated = copy.deepcopy(chromosome)

    # Choose mutation type (reduced flip_direction probability)
    mutation_types = ['add_module', 'remove_module', 'replace_module', 'flip_logic', 'flip_direction']
    mutation_weights = [0.30, 0.20, 0.30, 0.15, 0.05]  # flip_direction reduced to 5%
    mutation_type = random.choices(mutation_types, weights=mutation_weights)[0]

    logger.debug(f"Applying {mutation_type} mutation to {mutated.direction} pattern with {len(mutated.modules)} modules")

    try:
        if mutation_type == 'add_module':
            # SPRINT 13: Only add modules COMPATIBLE with pattern direction
            allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
            available = get_available_modules(generation, allow_indicators, config)  # AUDIT FIX: Pass config

            # Filter by direction compatibility
            compatible = get_compatible_modules(mutated.direction, list(available.keys()))

            # Exclude modules already in pattern
            candidates = [m for m in compatible if m not in mutated.modules]

            if candidates:
                new_module = random.choice(candidates)
                mutated.modules.append(new_module)
                logger.debug(f"Added direction-compatible module: {new_module}")
            else:
                logger.debug("No new compatible modules to add")

        elif mutation_type == 'remove_module':
            # Remove random module (keep at least 1)
            if len(mutated.modules) > 1:
                removed = mutated.modules.pop(random.randint(0, len(mutated.modules) - 1))
                logger.debug(f"Removed module: {removed}")

                # Adjust logic if needed
                if mutated.logic == '2of3' and len(mutated.modules) < 2:
                    mutated.logic = 'AND'
                    logger.debug("Adjusted logic to AND (insufficient modules for 2of3)")
                elif mutated.logic == '3of4' and len(mutated.modules) < 3:
                    mutated.logic = 'AND'
                    logger.debug("Adjusted logic to AND (insufficient modules for 3of4)")
            else:
                logger.debug("Cannot remove module - only 1 remaining")

        elif mutation_type == 'replace_module':
            # SPRINT 13: Replace with direction-compatible module from same family
            if mutated.modules:
                idx = random.randint(0, len(mutated.modules) - 1)
                old_module = mutated.modules[idx]

                # Get modules from same family
                family = get_module_family(old_module)

                # Filter by direction compatibility
                compatible_family = get_compatible_modules(mutated.direction, family)

                # Remove old module from candidates
                candidates = [m for m in compatible_family if m != old_module]

                if candidates:
                    new_module = random.choice(candidates)
                    mutated.modules[idx] = new_module
                    logger.debug(f"Replaced {old_module} with compatible {new_module}")
                else:
                    # Fallback: try any compatible module
                    allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
                    available = get_available_modules(generation, allow_indicators, config)  # AUDIT FIX: Pass config
                    compatible_all = get_compatible_modules(mutated.direction, list(available.keys()))
                    candidates = [m for m in compatible_all if m not in mutated.modules]

                    if candidates:
                        new_module = random.choice(candidates)
                        mutated.modules[idx] = new_module
                        logger.debug(f"Replaced {old_module} with any compatible {new_module}")
                    else:
                        logger.debug(f"No compatible replacements for {old_module}")

        elif mutation_type == 'flip_logic':
            # Change logic operator
            logic_options = ['AND', 'OR', '2of3', '3of4']

            # Filter valid options based on module count
            n_modules = len(mutated.modules)
            valid_options = []

            if n_modules >= 1:
                valid_options.extend(['AND', 'OR'])
            if n_modules >= 2:
                valid_options.append('2of3')
            if n_modules >= 3:
                valid_options.append('3of4')

            # Remove current logic from options
            valid_options = [l for l in valid_options if l != mutated.logic]

            if valid_options:
                old_logic = mutated.logic
                mutated.logic = random.choice(valid_options)
                logger.debug(f"Changed logic from {old_logic} to {mutated.logic}")

        elif mutation_type == 'flip_direction':
            # SPRINT 13: Flip direction AND replace modules with opposite-direction modules
            old_direction = mutated.direction
            mutated.direction = 'SHORT' if mutated.direction == 'LONG' else 'LONG'
            logger.debug(f"Flipping direction from {old_direction} to {mutated.direction}")

            # Replace ALL modules with opposite-direction compatible modules
            allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
            available = get_available_modules(generation, allow_indicators, config)  # AUDIT FIX: Pass config
            compatible = get_compatible_modules(mutated.direction, list(available.keys()))

            if len(compatible) > 0:
                # Keep same number of modules (or max 3)
                n_modules = min(len(mutated.modules), 3)
                mutated.modules = random.sample(compatible, min(n_modules, len(compatible)))
                logger.debug(f"Replaced modules with {mutated.direction}-compatible: {mutated.modules}")
            else:
                # Revert direction if no compatible modules
                mutated.direction = old_direction
                logger.warning(f"No compatible modules for {mutated.direction}, reverting direction")

        # Reset fitness
        mutated.fitness = -999.0
        mutated.fitness_long = -999.0
        mutated.fitness_short = -999.0
        mutated.generation_created = generation

        # SPRINT 13: Validate syntactic correctness
        if not validate_chromosome(mutated):
            logger.warning("Mutation produced syntactically invalid chromosome, returning original")
            return chromosome

        # SPRINT 13: Validate semantic coherence
        if not is_pattern_semantically_valid(mutated.modules, mutated.direction, min_bias_score=0.5):
            logger.warning(f"Mutation produced semantically invalid pattern: {mutated.to_readable()}")
            # Try to fix by filtering incompatible modules
            mutated.modules = filter_modules_for_direction(
                mutated.modules,
                mutated.direction,
                max_opposite_ratio=0.0
            )

            # If no modules left, return original
            if len(mutated.modules) == 0:
                logger.warning("Cannot fix semantic issues, returning original")
                return chromosome

        logger.debug(f"Mutation successful: {mutated.to_readable()}")
        return mutated

    except Exception as e:
        logger.error(f"Error during {mutation_type} mutation: {e}")
        return chromosome


def mutate_window(chromosome: PatternChromosome) -> PatternChromosome:
    """
    Mutate the lookback window parameter.

    Args:
        chromosome: Chromosome to mutate

    Returns:
        PatternChromosome: Mutated chromosome with new window

    Example:
        >>> chrom = PatternChromosome(direction='LONG', modules=['momentum_up_2bar'], logic='AND', window=5)
        >>> mutated = mutate_window(chrom)
        >>> 2 <= mutated.window <= 20
        True
    """
    mutated = copy.deepcopy(chromosome)

    # Small random adjustment to window
    adjustment = random.choice([-2, -1, 1, 2])
    new_window = mutated.window + adjustment

    # Clamp to valid range [2, 20]
    new_window = max(2, min(20, new_window))

    old_window = mutated.window
    mutated.window = new_window

    logger.debug(f"Changed window from {old_window} to {new_window}")

    # Reset fitness
    mutated.fitness = -999.0
    mutated.fitness_long = -999.0
    mutated.fitness_short = -999.0

    return mutated


if __name__ == '__main__':
    # Test genetic operators
    print("="*70)
    print("TESTING GENETIC OPERATORS V2")
    print("="*70)

    # Create test parents
    p1 = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short', 'large_body'],
        logic='AND',
        window=5
    )

    p2 = PatternChromosome(
        direction='SHORT',
        modules=['breakout_high_short', 'large_body', 'close_near_high'],
        logic='OR',
        window=10
    )

    config = {
        'ga': {
            'building_blocks': {
                'allow_indicators': True
            }
        }
    }

    # Test crossover
    print("\nTest 1: Crossover")
    print(f"  Parent 1: {p1.to_readable()}")
    print(f"  Parent 2: {p2.to_readable()}")

    offspring = crossover(p1, p2, generation=0, config=config)
    print(f"  Offspring: {offspring.to_readable()}")
    print(f"  Valid: {validate_chromosome(offspring)}")

    # Test multiple crossovers
    print("\n  Running 5 more crossovers:")
    for i in range(5):
        off = crossover(p1, p2, generation=0, config=config)
        print(f"    {i+1}. {off.to_readable()}")

    # Test mutations
    print("\nTest 2: Mutations")
    test_chrom = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short', 'large_body'],
        logic='AND',
        window=7
    )
    print(f"  Original: {test_chrom.to_readable()}")

    print("\n  Running 10 mutations:")
    for i in range(10):
        mutated = mutate(test_chrom, generation=0, config=config)
        print(f"    {i+1}. {mutated.to_readable()}")

    # Test window mutation
    print("\nTest 3: Window mutation")
    print(f"  Original: {test_chrom.to_readable()}")
    for i in range(5):
        mutated = mutate_window(test_chrom)
        print(f"    {i+1}. Window changed to: {mutated.window}")

    # Test with advanced generation (Gen 100)
    print("\nTest 4: Crossover with advanced modules (Gen 100)")
    p3 = PatternChromosome(
        direction='LONG',
        modules=['macd_bullish_cross', 'rsi_oversold_30', 'bb_lower_touch'],
        logic='2of3',
        window=15
    )

    p4 = PatternChromosome(
        direction='SHORT',
        modules=['momentum_down_3bar', 'stoch_overbought'],
        logic='AND',
        window=12
    )

    print(f"  Parent 3: {p3.to_readable()}")
    print(f"  Parent 4: {p4.to_readable()}")

    offspring2 = crossover(p3, p4, generation=100, config=config)
    print(f"  Offspring: {offspring2.to_readable()}")

    print(f"\n{'='*70}\n")
