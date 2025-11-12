"""
Pattern Generator V2 - Building Blocks System

Generates random PatternChromosomes using building block modules.
Respects progressive complexity unlocking.
"""

import random
from typing import List
import logging

from ga_patterns.chromosome_v2 import PatternChromosome, validate_chromosome
from ga_patterns.building_blocks import get_available_modules

logger = logging.getLogger(__name__)


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
        # Get available modules
        allow_indicators = config.get('ga', {}).get('building_blocks', {}).get('allow_indicators', True)
        available_modules = get_available_modules(generation, allow_indicators)

        # Determine number of modules based on generation
        if generation < 30:
            n_modules = random.randint(2, 3)
        elif generation < 80:
            n_modules = random.randint(2, 4)
        else:
            n_modules = random.randint(3, 5)

        # Ensure we have enough modules
        n_modules = min(n_modules, len(available_modules))

        # Sample modules
        module_names = random.sample(list(available_modules.keys()), n_modules)

        # Random direction
        direction = random.choice(['LONG', 'SHORT'])

        # Random logic
        logic_options = ['AND', 'OR']
        if n_modules >= 2:
            logic_options.append('2of3')
        if n_modules >= 3:
            logic_options.append('3of4')
        logic = random.choice(logic_options)

        # Random window
        window_min = config.get('ga', {}).get('window_min', 3)
        window_max = config.get('ga', {}).get('window_max', 10)
        window = random.randint(window_min, window_max)

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

        # Validate
        if validate_chromosome(chromosome):
            logger.debug(f"Generated: {chromosome.to_readable()}")
            return chromosome
        else:
            logger.debug(f"Attempt {attempt + 1}: Invalid chromosome, retrying...")

    # Fallback: return simple valid chromosome
    logger.warning("Failed to generate valid random chromosome after max attempts. Using fallback.")
    return PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short'],
        logic='AND',
        window=5,
        generation_created=generation
    )


def initialize_population(population_size: int, generation: int,
                         config: dict) -> List[PatternChromosome]:
    """
    Initialize population with random PatternChromosomes.

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

    for i in range(population_size):
        chromosome = generate_random_chromosome(generation, config)
        population.append(chromosome)

        if (i + 1) % 20 == 0:
            logger.debug(f"  Generated {i + 1}/{population_size} patterns")

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
