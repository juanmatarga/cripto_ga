"""
Genetic Operators V2 - Intelligent crossover and mutation for building blocks.

These operators maintain the semantic meaning of patterns by operating on
module-level rather than arbitrary subtree manipulation.

Key improvements:
    - Crossover: Uniform module mixing with 30% chance of adding new module
    - Mutation: 5 intelligent strategies (add, remove, replace, flip_logic, flip_direction)
    - Family-based replacement: Replace modules with others from same category
    - Validation: All offspring are validated before returning
"""

import random
import copy
from typing import List
from ga_patterns.chromosome_v2 import PatternChromosome, validate_chromosome
from ga_patterns.building_blocks import get_available_modules, get_module_family
import logging

logger = logging.getLogger(__name__)


def crossover(parent1: PatternChromosome, parent2: PatternChromosome,
              generation: int, config: dict) -> PatternChromosome:
    """
    Uniform crossover of modules between parents.

    Strategy:
        1. Pool modules from both parents
        2. Sample subset of modules
        3. Inherit direction/logic/window from parents
        4. 30% chance: add new module from current generation
        5. Validate offspring

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        generation: Current generation number
        config: Config dict with GA settings

    Returns:
        PatternChromosome: Valid offspring or copy of parent1 if invalid

    Example:
        >>> p1 = PatternChromosome(direction='LONG', modules=['momentum_up_2bar', 'volume_spike'], logic='AND')
        >>> p2 = PatternChromosome(direction='SHORT', modules=['breakout_high_short', 'large_body'], logic='AND')
        >>> offspring = crossover(p1, p2, generation=0, config={})
        >>> len(offspring.modules) >= 2
        True
    """
    logger.debug(f"Crossover between parents with {len(parent1.modules)} and {len(parent2.modules)} modules")

    # Combine modules from both parents
    all_modules = parent1.modules + parent2.modules

    # Remove duplicates while preserving order
    seen = set()
    unique_modules = []
    for module in all_modules:
        if module not in seen:
            seen.add(module)
            unique_modules.append(module)

    # Determine number of modules for offspring based on generation complexity
    if generation < 30:
        n_modules = random.randint(2, 3)
    elif generation < 80:
        n_modules = random.randint(2, 4)
    else:
        n_modules = random.randint(3, 5)

    # Ensure we don't exceed available modules
    n_modules = min(n_modules, len(unique_modules))

    # Sample modules
    if len(unique_modules) <= n_modules:
        offspring_modules = unique_modules.copy()
    else:
        offspring_modules = random.sample(unique_modules, n_modules)

    # 30% chance to add a new module available at this generation
    if random.random() < 0.3:
        allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
        available = get_available_modules(generation, allow_indicators)

        # Find modules not already in offspring
        new_candidates = [m for m in available.keys() if m not in offspring_modules]

        if new_candidates:
            new_module = random.choice(new_candidates)
            offspring_modules.append(new_module)
            logger.debug(f"Added new module: {new_module}")

    # Inherit direction (random choice from parents)
    offspring_direction = random.choice([parent1.direction, parent2.direction])

    # Inherit logic (random choice from parents)
    offspring_logic = random.choice([parent1.logic, parent2.logic])

    # Validate logic makes sense for module count
    if offspring_logic == '2of3' and len(offspring_modules) < 2:
        offspring_logic = 'AND'
    elif offspring_logic == '3of4' and len(offspring_modules) < 3:
        offspring_logic = 'AND'

    # Inherit window (random choice from parents)
    offspring_window = random.choice([parent1.window, parent2.window])

    # Create offspring
    offspring = PatternChromosome(
        direction=offspring_direction,
        modules=offspring_modules,
        logic=offspring_logic,
        window=offspring_window,
        generation_created=generation,
        fitness=-999.0,
        fitness_long=-999.0,
        fitness_short=-999.0
    )

    # Validate
    if validate_chromosome(offspring):
        logger.debug(f"Crossover successful: {offspring.to_readable()}")
        return offspring
    else:
        logger.warning("Crossover produced invalid offspring, returning parent1")
        return copy.deepcopy(parent1)


def mutate(chromosome: PatternChromosome, generation: int, config: dict) -> PatternChromosome:
    """
    Mutate chromosome using one of 5 strategies.

    Mutation types (with probabilities):
        - add_module (25%): Add new module from available set
        - remove_module (20%): Remove one module (keep minimum 1)
        - replace_module (30%): Replace module with one from same family
        - flip_logic (15%): Change combination logic (AND ↔ OR)
        - flip_direction (10%): Change direction (LONG ↔ SHORT)

    Args:
        chromosome: Chromosome to mutate
        generation: Current generation number
        config: Config dict

    Returns:
        PatternChromosome: Mutated chromosome or original if mutation invalid

    Example:
        >>> chrom = PatternChromosome(direction='LONG', modules=['momentum_up_2bar'], logic='AND')
        >>> mutated = mutate(chrom, generation=0, config={})
        >>> mutated.direction in ['LONG', 'SHORT']
        True
    """
    mutated = copy.deepcopy(chromosome)

    # Choose mutation type
    mutation_types = ['add_module', 'remove_module', 'replace_module', 'flip_logic', 'flip_direction']
    mutation_weights = [0.25, 0.20, 0.30, 0.15, 0.10]
    mutation_type = random.choices(mutation_types, weights=mutation_weights)[0]

    logger.debug(f"Applying {mutation_type} mutation to pattern with {len(mutated.modules)} modules")

    try:
        if mutation_type == 'add_module':
            # Add new module available at this generation
            allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
            available = get_available_modules(generation, allow_indicators)

            # Find modules not already in pattern
            candidates = [m for m in available.keys() if m not in mutated.modules]

            if candidates:
                new_module = random.choice(candidates)
                mutated.modules.append(new_module)
                logger.debug(f"Added module: {new_module}")
            else:
                logger.debug("No new modules to add")

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
            # Replace module with one from same family (category)
            if mutated.modules:
                idx = random.randint(0, len(mutated.modules) - 1)
                old_module = mutated.modules[idx]

                # Get modules from same family
                family = get_module_family(old_module)

                # Remove old module from candidates
                candidates = [m for m in family if m != old_module]

                if candidates:
                    new_module = random.choice(candidates)
                    mutated.modules[idx] = new_module
                    logger.debug(f"Replaced {old_module} with {new_module}")
                else:
                    logger.debug(f"No alternatives in family for {old_module}")

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
            # Flip direction
            old_direction = mutated.direction
            mutated.direction = 'SHORT' if mutated.direction == 'LONG' else 'LONG'
            logger.debug(f"Changed direction from {old_direction} to {mutated.direction}")

        # Reset fitness
        mutated.fitness = -999.0
        mutated.fitness_long = -999.0
        mutated.fitness_short = -999.0
        mutated.generation_created = generation

        # Validate
        if validate_chromosome(mutated):
            logger.debug(f"Mutation successful: {mutated.to_readable()}")
            return mutated
        else:
            logger.warning("Mutation produced invalid chromosome, returning original")
            return chromosome

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
