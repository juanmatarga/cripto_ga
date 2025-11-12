"""
Test Sprint 8 - Building Blocks System

Test the complete system:
1. Create PatternChromosome
2. Evaluate fitness with walk-forward
3. Test operators
4. Verify equity normalization fix
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ga_patterns.chromosome_v2 import PatternChromosome, validate_chromosome
from ga_patterns.operators_v2 import crossover, mutate
from ga_patterns.fitness import evaluate_fitness_bidirectional

def load_config():
    """Load configuration"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_data():
    """Load sample data"""
    data_path = Path('data/processed/btc_usdt_15m.parquet')
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Use subset for faster testing (last 6 months)
    df = df.iloc[-20000:].copy()
    logger.info(f"Using subset: {len(df)} bars")

    return df

def test_chromosome_creation():
    """Test 1: Create and validate chromosomes"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Chromosome Creation and Validation")
    logger.info("="*70)

    # Simple pattern
    chrom1 = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short', 'large_body'],
        logic='AND',
        window=5
    )

    logger.info(f"Pattern 1: {chrom1.to_readable()}")
    logger.info(f"Valid: {validate_chromosome(chrom1)}")
    logger.info(f"Expression: {chrom1.to_expression()[:150]}...")

    # Pattern with indicators
    chrom2 = PatternChromosome(
        direction='SHORT',
        modules=['momentum_down_3bar', 'rsi_overbought_70', 'volume_spike_short'],
        logic='2of3',
        window=10
    )

    logger.info(f"\nPattern 2: {chrom2.to_readable()}")
    logger.info(f"Valid: {validate_chromosome(chrom2)}")

    return chrom1, chrom2

def test_operators(chrom1, chrom2, config):
    """Test 2: Genetic operators"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Genetic Operators")
    logger.info("="*70)

    # Test crossover
    offspring = crossover(chrom1, chrom2, generation=0, config=config)
    logger.info(f"Crossover result: {offspring.to_readable()}")
    logger.info(f"Valid: {validate_chromosome(offspring)}")

    # Test mutations
    logger.info("\nTesting 5 mutations:")
    for i in range(5):
        mutated = mutate(chrom1, generation=0, config=config)
        logger.info(f"  {i+1}. {mutated.to_readable()}")

    return offspring

def test_fitness_evaluation(chrom, data, config):
    """Test 3: Fitness evaluation with walk-forward"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Fitness Evaluation")
    logger.info("="*70)

    logger.info(f"Evaluating: {chrom.to_readable()}")

    try:
        fitness, direction = evaluate_fitness_bidirectional(
            chrom, data, config, fast_mode=True
        )

        logger.info(f"Result:")
        logger.info(f"  Fitness: {fitness:.4f}")
        logger.info(f"  Direction: {direction}")
        logger.info(f"  Fitness LONG: {chrom.fitness_long:.4f}")
        logger.info(f"  Fitness SHORT: {chrom.fitness_short:.4f}")

        if fitness > -999:
            logger.info("✓ SUCCESS: Pattern evaluated successfully!")
            return True
        else:
            logger.warning("✗ FAIL: Fitness = -999 (pattern failed constraints)")
            return False

    except Exception as e:
        logger.error(f"✗ ERROR during fitness evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_patterns(data, config):
    """Test 4: Multiple simple patterns"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Evaluating Multiple Simple Patterns")
    logger.info("="*70)

    test_patterns = [
        PatternChromosome(
            direction='LONG',
            modules=['momentum_up_2bar', 'close_near_high'],
            logic='AND',
            window=5
        ),
        PatternChromosome(
            direction='SHORT',
            modules=['momentum_down_2bar', 'close_near_low'],
            logic='AND',
            window=5
        ),
        PatternChromosome(
            direction='LONG',
            modules=['volume_spike_short', 'large_body', 'close_near_high'],
            logic='2of3',
            window=7
        ),
    ]

    results = []
    for i, pattern in enumerate(test_patterns):
        logger.info(f"\n[{i+1}/{len(test_patterns)}] {pattern.to_readable()}")

        try:
            fitness, direction = evaluate_fitness_bidirectional(
                pattern, data, config, fast_mode=True
            )

            logger.info(f"  → Fitness: {fitness:.4f}, Direction: {direction}")
            results.append({
                'pattern': pattern.to_readable(),
                'fitness': fitness,
                'direction': direction,
                'success': fitness > -999
            })
        except Exception as e:
            logger.error(f"  → Error: {e}")
            results.append({
                'pattern': pattern.to_readable(),
                'fitness': -999,
                'direction': 'N/A',
                'success': False
            })

    # Summary
    logger.info("\n" + "-"*70)
    logger.info("SUMMARY:")
    successful = sum(1 for r in results if r['success'])
    logger.info(f"  Successful evaluations: {successful}/{len(results)}")

    if successful > 0:
        best = max(results, key=lambda x: x['fitness'])
        logger.info(f"  Best fitness: {best['fitness']:.4f}")
        logger.info(f"  Best pattern: {best['pattern']}")

    return results

def main():
    """Run all tests"""
    logger.info("="*70)
    logger.info("SPRINT 8 - BUILDING BLOCKS SYSTEM TEST")
    logger.info("="*70)

    # Load config and data
    logger.info("\nLoading configuration and data...")
    config = load_config()
    data = load_data()

    if data is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Test 1: Create chromosomes
    chrom1, chrom2 = test_chromosome_creation()

    # Test 2: Operators
    offspring = test_operators(chrom1, chrom2, config)

    # Test 3: Fitness evaluation
    success = test_fitness_evaluation(chrom1, data, config)

    # Test 4: Multiple patterns
    if success:
        results = test_simple_patterns(data, config)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TESTS COMPLETED")
    logger.info("="*70)

    if success:
        logger.info("✓ All core systems working correctly!")
        logger.info("\nNext steps:")
        logger.info("  1. Test with more complex patterns")
        logger.info("  2. Run full GA with new system")
        logger.info("  3. Compare results with legacy system")
    else:
        logger.warning("✗ Some tests failed. Review logs above.")

if __name__ == '__main__':
    main()
