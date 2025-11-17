"""
Genetic Algorithm Pattern Discovery for Crypto Trading
======================================================

Main execution script for end-to-end experiment.

Author: Juan Manuel [Last Name]
Institution: Universidad del CEMA (UCEMA)
Course: Advanced Business Analytics
Date: January 2025

Description:
    This script implements a complete genetic algorithm system for discovering
    profitable trading patterns in cryptocurrency time series data. The system
    includes walk-forward validation, statistical testing, and publication-ready
    reporting.

Usage:
    python main.py

Expected Runtime:
    2-4 hours depending on hardware and configuration

Outputs:
    All results saved to output_reports/ directory

For more information:
    See README.md and docs/USER_GUIDE.md
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import List
import sys
import argparse
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ASCII art banner
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧬  GENETIC ALGORITHM PATTERN DISCOVERY  🧬               ║
║                                                              ║
║     Cryptocurrency Trading Pattern Evolution                ║
║     with Statistical Validation                             ║
║                                                              ║
║     UCEMA - Advanced Business Analytics                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# Setup logging antes de cualquier import
def setup_logging(config: dict):
    """Configura logging estructurado."""
    log_dir = Path(config['output']['logs_dir'])
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"experiment_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO if config['output']['verbose_logging'] else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    """Carga configuración desde YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: dict):
    """Valida que config tenga todas las secciones requeridas."""
    required_sections = ['data', 'costs', 'walkforward', 'ga', 'exits',
                        'selection', 'robustness', 'output']
    for section in required_sections:
        assert section in config, f"Missing config section: {section}"

    # Validar timeframe en TIME_MAP
    timeframe = config['data']['timeframe']
    assert timeframe in config['data']['time_map'], \
        f"Timeframe '{timeframe}' not in TIME_MAP. Available: {list(config['data']['time_map'].keys())}"

def calculate_diversity(population):
    """
    SPRINT 12.6: Calculate population genetic diversity.

    Diversity metric: Percentage of unique module sets in population.
    Higher diversity = more exploration potential.

    Args:
        population: List of PatternChromosome

    Returns:
        float: Diversity percentage (0.0 to 1.0)

    Example:
        >>> # Population with 100 patterns
        >>> # If 50 have unique module combinations, diversity = 0.50
        >>> diversity = calculate_diversity(population)
        >>> diversity
        0.50
    """
    if len(population) == 0:
        return 0.0

    # Count unique module sets
    unique_module_sets = set()
    for p in population:
        # Convert modules list to frozenset for hashing
        module_set = frozenset(p.modules)
        unique_module_sets.add(module_set)

    # Diversity = unique patterns / total patterns
    diversity = len(unique_module_sets) / len(population)

    return diversity

def inject_immigrants(population: List, generation: int, config: dict, n_immigrants: int = 20):
    """
    SPRINT 12.6: Inject fresh random patterns to combat stagnation.

    Replaces worst patterns with new random patterns to restore genetic diversity.
    This prevents premature convergence to local optima.

    Args:
        population: Current population (List of PatternChromosome)
        generation: Current generation number
        config: Config dict
        n_immigrants: Number of fresh patterns to inject (default 20)

    Returns:
        Updated population with immigrants injected

    Example:
        If population has converged and best fitness hasn't improved for 2 gens,
        inject 20 fresh patterns to bring new genetic material.
    """
    from ga_patterns.generator_v2 import generate_random_chromosome
    from ga_patterns.chromosome_v2 import validate_chromosome

    logger = logging.getLogger(__name__)
    logger.info(f"IMMIGRATION: Injecting {n_immigrants} fresh patterns to restore diversity...")

    # Generate immigrants
    immigrants = []
    attempts = 0
    max_attempts = n_immigrants * 10  # Try up to 10x to get valid patterns

    while len(immigrants) < n_immigrants and attempts < max_attempts:
        immigrant = generate_random_chromosome(generation, config)
        if validate_chromosome(immigrant):
            immigrant.fitness = -999.0  # Will be evaluated next generation
            immigrants.append(immigrant)
        attempts += 1

    if len(immigrants) < n_immigrants:
        logger.warning(f"Only generated {len(immigrants)}/{n_immigrants} valid immigrants after {attempts} attempts")

    # Sort population by fitness (ascending)
    population.sort(key=lambda p: p.fitness)

    # Replace worst N patterns with immigrants
    # Keep top (population_size - n_immigrants), add n_immigrants new patterns
    n_replace = min(len(immigrants), len(population))
    new_population = population[n_replace:] + immigrants

    logger.info(f"[OK] Injected {len(immigrants)} immigrants, replacing worst {n_replace} patterns")
    logger.info(f"     Population size: {len(population)} → {len(new_population)}")

    return new_population

def maintain_diversity(population, max_similarity: float = 0.8):
    """
    Remove duplicate/highly similar patterns to maintain diversity (SPRINT 12).

    Similarity metric: Jaccard index on modules set.
    If two patterns have >80% module overlap, keep the one with better fitness.

    Args:
        population: List of PatternChromosome
        max_similarity: Maximum allowed similarity threshold (not currently used)

    Returns:
        List of unique patterns
    """
    from collections import defaultdict

    logger = logging.getLogger(__name__)

    # Group patterns by module set
    module_groups = defaultdict(list)
    for pattern in population:
        module_key = frozenset(pattern.modules)
        module_groups[module_key].append(pattern)

    # For each group, keep best pattern
    diverse_population = []
    for module_set, patterns in module_groups.items():
        # Sort by fitness (descending)
        best_pattern = max(patterns, key=lambda p: p.fitness)
        diverse_population.append(best_pattern)

    removed = len(population) - len(diverse_population)
    if removed > 0:
        logger.info(f"[DIVERSITY] Removed {removed} duplicate patterns, {len(diverse_population)} unique patterns remain")

    return diverse_population

def main():
    """
    Pipeline principal del experimento.

    Fases:
    0. Setup (config, logging, directorios)
    1. Data loading (Binance API con paginación)
    2. GA Evolution (con tracking de evolución)
    3. Pattern selection (decorrelación)
    4. Statistical validation (Hansen SPA, White RC)
    5. Report generation
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Genetic Algorithm Pattern Discovery')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Override max generations from config')
    args = parser.parse_args()

    config_path = args.config

    # FASE 0: Setup
    try:
        print(BANNER)
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print("\n" + "="*60)
        print("  GENETIC ALGORITHM PATTERN DISCOVERY")
        print("  Cryptocurrency Trading Pattern Evolution")
        print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = datetime.now()  # Track total runtime

    print("="*80)
    print("BTC/USDT PATTERN DISCOVERY - GENETIC ALGORITHM")
    print("="*80)
    print(f"Using config: {config_path}")

    config = load_config(config_path)
    validate_config(config)

    # Override generations if provided
    if args.generations is not None:
        print(f"Overriding generations: {config['ga']['generations_max']} -> {args.generations}")
        config['ga']['generations_max'] = args.generations

    logger = setup_logging(config)

    logger.info("Configuration loaded successfully")
    logger.info(f"Exchange: {config['data']['exchange']}")
    logger.info(f"Symbol: {config['data']['symbol']}")
    logger.info(f"Timeframe: {config['data']['timeframe']}")
    logger.info(f"Period: {config['data']['start']} to {config['data']['end']}")
    logger.info(f"GA Population: {config['ga']['population']}")
    logger.info(f"Max Generations: {config['ga']['generations_max']}")

    # Crear directorios de output
    Path(config['output']['reports_dir']).mkdir(exist_ok=True)
    Path(config['output']['evolution_dir']).mkdir(exist_ok=True)

    # FASE 1: Data Loading
    logger.info("\n" + "="*80)
    logger.info("FASE 1: DATA LOADING (BINANCE API)")
    logger.info("="*80)

    from loader import load_binance_data, check_binance_connection

    # Test de conexión primero
    logger.info("Testing Binance connection...")
    if not check_binance_connection(config):
        logger.error("Cannot connect to Binance. Check your internet connection.")
        sys.exit(1)

    # Cargar datos
    try:
        data = load_binance_data(config)
        logger.info(f"[OK] Data loaded successfully: {len(data)} candles")
        logger.info(f"[OK] Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"[OK] Metadata: {data.attrs}")

        # Obtener periods_per_year desde TIME_MAP
        timeframe = config['data']['timeframe']
        periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']
        logger.info(f"[OK] Periods per year ({timeframe}): {periods_per_year}")

    except Exception as e:
        logger.error(f"[FAIL] Data loading failed: {e}")
        raise

    # Demo: Calcular métricas en price series (solo para verificar)
    logger.info("\nDemo: Calculating metrics on price series...")
    from backtest.metrics import calculate_all_metrics

    demo_equity = data['Close'].copy()  # Usar precios como equity demo
    demo_metrics = calculate_all_metrics(demo_equity, periods_per_year)

    logger.info(f"Demo Metrics (on raw BTC prices):")
    for key, value in demo_metrics.items():
        if isinstance(value, float):
            if key in ['cagr', 'total_return', 'max_dd']:
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # SPRINT 9: Preprocess indicators ONCE
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING INDICATORS (ONCE)")
    logger.info("="*80)

    from ga_patterns.evaluator import preprocess_indicators
    data = preprocess_indicators(data)

    indicator_cols = [c for c in data.columns if c not in ['Open','High','Low','Close','Volume']]
    logger.info(f"[OK] Added {len(indicator_cols)} indicator columns")
    logger.info(f"Indicators: {', '.join(indicator_cols[:5])}... (+{len(indicator_cols)-5} more)")

    # FASE 2: GA Evolution
    logger.info("\n" + "="*80)
    logger.info("FASE 2: GENETIC ALGORITHM EVOLUTION (BIDIRECTIONAL)")
    logger.info("="*80)

    import random
    import copy
    # SPRINT 8: Building Blocks System
    from ga_patterns.generator_v2 import initialize_population
    from ga_patterns.generator import tournament_selection  # Still use legacy for tournament
    from ga_patterns.operators_v2 import crossover, mutate
    from ga_patterns.chromosome_v2 import PatternChromosome
    from ga_patterns.fitness import evaluate_fitness_bidirectional
    from ga_patterns.evolution_tracker import EvolutionTracker

    # Seeds
    random.seed(config['ga']['seed'])
    np.random.seed(config['ga']['seed'])

    # Tracker
    tracker = EvolutionTracker(config)

    # Parameters
    population_size = config['ga']['population']
    max_generations = config['ga']['generations_max']
    patience = config['ga']['patience_no_improve']
    elitism = config['ga']['elitism']
    mutation_rate = config['ga']['mutation_rate']
    crossover_rate = config['ga']['crossover_rate']

    # Initialize
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING POPULATION")
    logger.info(f"{'='*80}")

    population = initialize_population(population_size, generation=0, config=config)

    # SPRINT 11: CREATE EVALUATION WINDOWS ONCE (GLOBAL)
    logger.info(f"\n{'='*80}")
    logger.info("CREATING EVALUATION WINDOWS (ONCE)")
    logger.info(f"{'='*80}")

    from backtest.simple_sampling import create_simple_windows
    from ga_patterns.fitness import evaluate_fitness_unidirectional

    global_windows = create_simple_windows(
        data,
        n_windows=config['ga']['fast_mode']['n_windows'],
        window_months=config['ga']['fast_mode']['window_months'],
        seed=config['ga']['seed']
    )

    logger.info(f"[OK] Created {len(global_windows)} global windows")
    logger.info("These windows will be reused for ALL pattern evaluations")
    logger.info("")

    # Evaluate initial - SPRINT 11: Unidirectional with global windows
    logger.info("Evaluating initial population...")
    for i, pattern in enumerate(population):
        fitness, direction = evaluate_fitness_unidirectional(
            pattern, global_windows, config
        )
        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i+1}/{len(population)} patterns")

    logger.info(f"[OK] Initial population evaluated")
    valid_count = sum(1 for p in population if p.fitness > -999)
    logger.info(f"Valid patterns: {valid_count}/{len(population)}")

    best_fitness_history = []
    best_pattern = max(population, key=lambda p: p.fitness)
    generations_without_improvement = 0

    # Track initial
    mean_fitness = np.mean([p.fitness for p in population])
    tracker.track_generation(0, population, best_pattern, mean_fitness)

    # Evolution
    logger.info(f"\n{'='*80}")
    logger.info("STARTING EVOLUTION")
    logger.info(f"{'='*80}")

    for generation in range(1, max_generations + 1):
        logger.info("")
        logger.info("="*80)
        logger.info(f"GENERATION {generation}/{max_generations}")
        logger.info("="*80)

        # SPRINT 11: Log module unlocks with emoji-free version for Windows
        if generation == 30:
            logger.info("UNLOCKED: Indicator modules (RSI, SMA, MACD)")
        elif generation == 80:
            logger.info("UNLOCKED: Advanced modules (Bollinger Bands, ATR, Stochastic)")

        # New population
        new_population = []

        # Elitism
        population_sorted = sorted(population, key=lambda p: p.fitness, reverse=True)
        for i in range(elitism):
            new_population.append(copy.deepcopy(population_sorted[i]))

        # Generate offspring - SPRINT 8: Use new operators
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                offspring = crossover(parent1, parent2, generation, config)
            else:
                offspring = copy.deepcopy(parent1)

            if random.random() < mutation_rate:
                offspring = mutate(offspring, generation, config)

            new_population.append(offspring)

        population = new_population

        # SPRINT 12: Maintain diversity
        population = maintain_diversity(population)

        # AUDIT FIX: Direction quotas to ensure SHORT survival
        # Target: 30% minimum for each direction (15 SHORT, 15 LONG out of 50)
        min_quota_per_direction = int(population_size * 0.30)

        long_count = sum(1 for p in population if p.direction == 'LONG')
        short_count = sum(1 for p in population if p.direction == 'SHORT')

        logger.debug(f"Direction distribution: {long_count} LONG, {short_count} SHORT")

        from ga_patterns.generator_v2 import generate_random_chromosome
        from ga_patterns.chromosome_v2 import PatternChromosome

        # Enforce SHORT quota
        if short_count < min_quota_per_direction:
            shortage = min_quota_per_direction - short_count
            logger.info(f"[QUOTA] SHORT shortage: {shortage} patterns. Generating SHORT patterns to meet quota...")

            # Remove worst LONG patterns to make room
            if long_count > min_quota_per_direction:
                long_patterns = [p for p in population if p.direction == 'LONG']
                long_sorted = sorted(long_patterns, key=lambda p: p.fitness)
                to_remove = long_sorted[:shortage]
                population = [p for p in population if p not in to_remove]
                logger.debug(f"  Removed {len(to_remove)} worst LONG patterns to make room")

            # Generate SHORT patterns
            for _ in range(shortage):
                # Generate random pattern and force SHORT direction
                new_pattern = generate_random_chromosome(generation, config)
                new_pattern.direction = 'SHORT'
                # Re-filter modules to be SHORT-compatible
                from ga_patterns.module_semantics import get_compatible_modules
                compatible = get_compatible_modules('SHORT', new_pattern.modules)
                if compatible:
                    new_pattern.modules = compatible[:len(new_pattern.modules)]
                population.append(new_pattern)

            logger.info(f"[QUOTA] Added {shortage} SHORT patterns")

        # Enforce LONG quota
        if long_count < min_quota_per_direction:
            shortage = min_quota_per_direction - long_count
            logger.info(f"[QUOTA] LONG shortage: {shortage} patterns. Generating LONG patterns to meet quota...")

            # Remove worst SHORT patterns to make room
            if short_count > min_quota_per_direction:
                short_patterns = [p for p in population if p.direction == 'SHORT']
                short_sorted = sorted(short_patterns, key=lambda p: p.fitness)
                to_remove = short_sorted[:shortage]
                population = [p for p in population if p not in to_remove]
                logger.debug(f"  Removed {len(to_remove)} worst SHORT patterns to make room")

            # Generate LONG patterns
            for _ in range(shortage):
                new_pattern = generate_random_chromosome(generation, config)
                new_pattern.direction = 'LONG'
                from ga_patterns.module_semantics import get_compatible_modules
                compatible = get_compatible_modules('LONG', new_pattern.modules)
                if compatible:
                    new_pattern.modules = compatible[:len(new_pattern.modules)]
                population.append(new_pattern)

            logger.info(f"[QUOTA] Added {shortage} LONG patterns")

        # Refill if diversity maintenance removed too many patterns
        if len(population) < population_size:
            n_to_add = population_size - len(population)
            logger.info(f"[DIVERSITY] Refilling {n_to_add} patterns to reach population size")
            for _ in range(n_to_add):
                new_pattern = generate_random_chromosome(generation, config)
                population.append(new_pattern)

        # SPRINT 11: Evaluate new patterns only with unidirectional
        patterns_to_eval = [p for p in population if p.fitness == -999.0]
        logger.info(f"\nEvaluating {len(patterns_to_eval)} new patterns...")

        for i, pattern in enumerate(patterns_to_eval):
            if (i+1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(patterns_to_eval)} patterns evaluated")

            fitness, direction = evaluate_fitness_unidirectional(
                pattern,
                global_windows,
                config
            )

        # SPRINT 11: Enhanced statistics
        valid_patterns = [p for p in population if p.fitness > -999]
        long_patterns = [p for p in population if p.direction == 'LONG']
        short_patterns = [p for p in population if p.direction == 'SHORT']

        valid_long = [p for p in long_patterns if p.fitness > -999]
        valid_short = [p for p in short_patterns if p.fitness > -999]

        best_long = max(long_patterns, key=lambda p: p.fitness) if long_patterns else None
        best_short = max(short_patterns, key=lambda p: p.fitness) if short_patterns else None

        # SPRINT 14: Enhanced dashboard logging
        logger.info("")
        logger.info("="*80)
        logger.info(f"GENERATION {generation}/{max_generations} DASHBOARD")
        logger.info("="*80)

        # Quick stats
        current_best = max(population, key=lambda p: p.fitness)
        mean_fit = np.mean([p.fitness for p in valid_patterns]) if valid_patterns else 0

        logger.info(f"VALID: {len(valid_patterns)}/{len(population)} ({len(valid_patterns)*100//len(population)}%)")
        logger.info(f"BEST: {current_best.fitness:.4f} ({current_best.direction})")
        logger.info(f"MEAN: {mean_fit:.4f}")
        logger.info(f"DIVERSITY: {calculate_diversity(population)*100:.1f}%")
        logger.info("")
        logger.info(f"LONG: {len(valid_long)}/{len(long_patterns)} valid ({len(valid_long)*100//len(long_patterns) if long_patterns else 0}%)")
        logger.info(f"SHORT: {len(valid_short)}/{len(short_patterns)} valid ({len(valid_short)*100//len(short_patterns) if short_patterns else 0}%)")

        # AUDIT: Enhanced diagnostics for SHORT pattern tracking
        logger.info("")
        logger.info("DIAGNOSTIC: SHORT Pattern Analysis")
        logger.info("-" * 40)

        # Analyze why SHORT patterns are failing
        failed_short = [p for p in short_patterns if p.fitness == -999]
        if failed_short:
            logger.info(f"Failed SHORT patterns: {len(failed_short)}")
            # Check failure reasons by examining fitness components
            zero_trades_count = sum(1 for p in failed_short if hasattr(p, 'n_trades') and p.n_trades == 0)
            if zero_trades_count > 0:
                logger.info(f"  - Zero trades: {zero_trades_count}/{len(failed_short)}")

        # Show module usage breakdown by direction
        long_modules = [m for p in long_patterns for m in p.modules]
        short_modules = [m for p in short_patterns for m in p.modules]

        from collections import Counter
        if short_modules:
            short_module_counts = Counter(short_modules).most_common(3)
            logger.info(f"Top 3 SHORT modules: {', '.join([f'{m}({c})' for m, c in short_module_counts])}")

        if long_modules:
            long_module_counts = Counter(long_modules).most_common(3)
            logger.info(f"Top 3 LONG modules: {', '.join([f'{m}({c})' for m, c in long_module_counts])}")

        # Show best SHORT performance metrics
        if valid_short:
            best_valid_short = max(valid_short, key=lambda p: p.fitness)
            logger.info("")
            logger.info(f"Best valid SHORT fitness: {best_valid_short.fitness:.4f}")
            if hasattr(best_valid_short, 'fitness_components'):
                logger.info(f"  Components: Sortino={best_valid_short.fitness_components.get('sortino_norm', 0):.2f}, "
                          f"Calmar={best_valid_short.fitness_components.get('calmar_norm', 0):.2f}, "
                          f"WinRate={best_valid_short.fitness_components.get('win_rate', 0):.2%}")
        else:
            logger.warning("NO VALID SHORT PATTERNS FOUND!")

        logger.info("-" * 40)
        logger.info("")

        if best_long and best_long.fitness > -999:
            logger.info("")
            logger.info("BEST LONG:")
            logger.info(f"  {best_long.to_readable()[:80]}")
            if hasattr(best_long, 'metrics') and best_long.metrics:
                logger.info(f"  Fitness={best_long.fitness:.4f}, Sharpe={best_long.metrics.get('sharpe', 0):.2f}, CAGR={best_long.metrics.get('cagr', 0)*100:.1f}%, Trades={best_long.n_trades}")

        if best_short and best_short.fitness > -999:
            logger.info("")
            logger.info("BEST SHORT:")
            logger.info(f"  {best_short.to_readable()[:80]}")
            if hasattr(best_short, 'metrics') and best_short.metrics:
                logger.info(f"  Fitness={best_short.fitness:.4f}, Sharpe={best_short.metrics.get('sharpe', 0):.2f}, CAGR={best_short.metrics.get('cagr', 0)*100:.1f}%, Trades={best_short.n_trades}")

        # Module usage
        from collections import Counter
        all_modules = [m for p in population for m in p.modules]
        top_modules = Counter(all_modules).most_common(5)
        logger.info(f"\nTop 5 modules:")
        for module, count in top_modules:
            logger.info(f"  {module}: {count}")

        logger.info("")

        # Tracking
        current_best = max(population, key=lambda p: p.fitness)
        mean_fitness = np.mean([p.fitness for p in population])
        best_fitness_history.append(current_best.fitness)

        # Track
        tracker.track_generation(generation, population, current_best, mean_fitness)

        # SPRINT 12.6: Track genetic diversity
        diversity = calculate_diversity(population)
        logger.info(f"Population diversity: {diversity:.2%} ({int(diversity * len(population))} unique patterns)")

        if diversity < 0.30:
            logger.warning("LOW DIVERSITY (<30% unique patterns) - population may be converging")

        # SPRINT 12.6: Track improvement and trigger immigration
        if current_best.fitness > best_pattern.fitness + 0.01:  # Minimum 0.01 improvement
            best_pattern = current_best
            generations_without_improvement = 0
            logger.info("[OK] New best!")
        else:
            generations_without_improvement += 1
            logger.info(f"No improvement ({generations_without_improvement}/{patience})")

        # SPRINT 12.6: Immigration trigger when stagnation detected
        if generations_without_improvement >= 2 and generation < max_generations - 5:
            logger.info("")
            logger.info("STAGNATION DETECTED - Triggering immigration to restore diversity")
            population = inject_immigrants(population, generation, config, n_immigrants=20)
            # Don't reset counter - let patience still work for final early stop
            logger.info("")

        # Early stopping (with higher patience now)
        if generations_without_improvement >= patience:
            logger.info(f"\n[OK] Early stopping at gen {generation}")
            break

    # Top patterns
    population_sorted = sorted(population, key=lambda p: p.fitness, reverse=True)
    top_patterns = population_sorted[:20]

    logger.info(f"\n{'='*80}")
    logger.info("EVOLUTION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Generations: {generation}")
    logger.info(f"Best: {best_pattern.fitness:.4f} ({best_pattern.direction})")

    # Save summary
    tracker.save_final_summary(generation, top_patterns)

    # Display top 5 - SPRINT 8: Use readable format
    logger.info(f"\nTop 5 Patterns:")
    for i, pattern in enumerate(top_patterns[:5], 1):
        logger.info(f"\n{i}. {pattern.to_readable()} | Fit: {pattern.fitness:.4f} | L:{pattern.fitness_long:.4f} S:{pattern.fitness_short:.4f}")
        if hasattr(pattern, 'to_expression'):
            logger.info(f"   Modules: {', '.join(pattern.modules)}")
        else:
            logger.info(f"   {pattern.expression}")

    # FASE 3: Pattern Selection
    logger.info("\n" + "="*80)
    logger.info("FASE 3: PORTFOLIO SELECTION")
    logger.info("="*80)

    from backtest.correlation import select_portfolio

    # Select decorrelated portfolio from top patterns
    portfolio = select_portfolio(top_patterns, data, config)

    if len(portfolio) == 0:
        logger.error("Portfolio selection failed - no patterns passed filters!")
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT STOPPED (No valid portfolio)")
        logger.info("="*80)
        return

    logger.info(f"\n[OK] Selected {len(portfolio)} decorrelated patterns for validation")

    # FASE 4: Statistical Validation
    logger.info("\n" + "="*80)
    logger.info("FASE 4: STATISTICAL VALIDATION")
    logger.info("="*80)

    if len(portfolio) > 0:
        from robustness import run_robustness_tests

        logger.info("Running robustness tests on final portfolio...")
        logger.info("WARNING: This may take 10-20 minutes...")

        robustness_results = run_robustness_tests(
            portfolio_patterns=portfolio,
            data=data,
            config=config
        )

        # Guardar resultados
        import json

        output_dir = Path(config['output']['reports_dir'])
        output_dir.mkdir(exist_ok=True)

        # Guardar Hansen SPA
        if robustness_results['hansen_spa']:
            hansen_file = output_dir / 'hansen_spa_results.json'
            with open(hansen_file, 'w') as f:
                json.dump(robustness_results['hansen_spa'], f, indent=2)
            logger.info(f"[OK] Saved Hansen SPA results to {hansen_file}")

        # Guardar White RC
        if robustness_results['white_rc']:
            white_file = output_dir / 'white_rc_results.json'
            with open(white_file, 'w') as f:
                json.dump(robustness_results['white_rc'], f, indent=2)
            logger.info(f"[OK] Saved White RC results to {white_file}")

        # Guardar Bootstrap
        if robustness_results['bootstrap']:
            bootstrap_file = output_dir / 'bootstrap_results.json'
            # Convertir a JSON-serializable (sin arrays numpy)
            bootstrap_json = {}
            for metric, stats in robustness_results['bootstrap'].items():
                bootstrap_json[metric] = {
                    'mean': float(stats['mean']),
                    'median': float(stats['median']),
                    'std': float(stats['std']),
                    'ci_lower': float(stats['ci_lower']),
                    'ci_upper': float(stats['ci_upper'])
                }
            with open(bootstrap_file, 'w') as f:
                json.dump(bootstrap_json, f, indent=2)
            logger.info(f"[OK] Saved Bootstrap results to {bootstrap_file}")

        # Guardar equity curves para visualización
        portfolio_equity = robustness_results['portfolio_equity']
        benchmark_equity = robustness_results['benchmark_equity']

        equity_df = pd.DataFrame({
            'portfolio': portfolio_equity,
            'benchmark': benchmark_equity
        })
        equity_file = output_dir / 'equity_curves.csv'
        equity_df.to_csv(equity_file)
        logger.info(f"[OK] Saved equity curves to {equity_file}")

    else:
        logger.warning("No portfolio to validate (empty)")

    # FASE 5: Report Generation
    logger.info("\n" + "="*80)
    logger.info("FASE 5: REPORT GENERATION")
    logger.info("="*80)

    if len(portfolio) > 0 and robustness_results:
        from reports import visualizations, report_generator, latex_exporter

        output_dir = Path(config['output']['reports_dir'])

        # Load evolution tracker data
        evolution_dir = Path(config['output']['evolution_dir'])
        evolution_file = evolution_dir / 'evolution_summary.json'

        evolution_data = {}
        if evolution_file.exists():
            with open(evolution_file, 'r') as f:
                evolution_data = json.load(f)
            logger.info(f"[OK] Loaded evolution data from {evolution_file}")
        else:
            logger.warning("Evolution summary not found, skipping evolution plot")

        # 1. Generate visualizations
        logger.info("\n--- Generating Visualizations ---")

        try:
            # Equity curves
            visualizations.plot_equity_curves(
                portfolio_equity=portfolio_equity,
                benchmark_equity=benchmark_equity,
                output_path=output_dir / 'equity_performance.png'
            )

            # Drawdown analysis
            visualizations.plot_drawdown_analysis(
                equity=portfolio_equity,
                output_path=output_dir / 'drawdown_analysis.png'
            )

            # Evolution fitness (if data available)
            if evolution_data:
                visualizations.plot_evolution_fitness(
                    evolution_tracker_data=evolution_data,
                    output_path=output_dir / 'evolution_fitness.png'
                )

            # Statistical tests
            visualizations.plot_statistical_tests(
                hansen_results=robustness_results['hansen_spa'],
                white_results=robustness_results['white_rc'],
                bootstrap_results=robustness_results['bootstrap'],
                output_path=output_dir / 'statistical_tests.png'
            )

            # Returns distribution
            returns = portfolio_equity.pct_change().dropna()
            visualizations.plot_returns_distribution(
                returns=returns,
                output_path=output_dir / 'returns_distribution.png'
            )

            logger.info("[OK] All visualizations generated successfully")

        except Exception as e:
            logger.error(f"[FAIL] Visualization generation failed: {e}")

        # 2. Generate Markdown report
        logger.info("\n--- Generating Markdown Report ---")

        try:
            report_generator.generate_report(
                portfolio=portfolio,
                portfolio_equity=portfolio_equity,
                benchmark_equity=benchmark_equity,
                evolution_data=evolution_data,
                final_generation=generation,
                hansen_results=robustness_results['hansen_spa'],
                white_results=robustness_results['white_rc'],
                bootstrap_results=robustness_results['bootstrap'],
                data=data,
                config=config,
                output_path=output_dir / 'experiment_report.md'
            )

            logger.info("[OK] Markdown report generated successfully")

        except Exception as e:
            logger.error(f"[FAIL] Report generation failed: {e}")

        # 3. Export LaTeX tables
        logger.info("\n--- Exporting LaTeX Tables ---")

        try:
            # Calculate metrics for tables
            portfolio_metrics = calculate_all_metrics(portfolio_equity, periods_per_year)
            benchmark_metrics = calculate_all_metrics(benchmark_equity, periods_per_year)

            latex_exporter.export_all_latex_tables(
                portfolio=portfolio,
                portfolio_metrics=portfolio_metrics,
                benchmark_metrics=benchmark_metrics,
                hansen_results=robustness_results['hansen_spa'],
                white_results=robustness_results['white_rc'],
                bootstrap_results=robustness_results['bootstrap'],
                output_dir=output_dir
            )

            logger.info("[OK] LaTeX tables exported successfully")

        except Exception as e:
            logger.error(f"[FAIL] LaTeX export failed: {e}")

        # Summary of outputs
        logger.info("\n--- Output Files Generated ---")
        logger.info(f"Reports directory: {output_dir}")
        logger.info(f"  - equity_performance.png")
        logger.info(f"  - drawdown_analysis.png")
        logger.info(f"  - evolution_fitness.png")
        logger.info(f"  - statistical_tests.png")
        logger.info(f"  - returns_distribution.png")
        logger.info(f"  - experiment_report.md")
        logger.info(f"  - patterns_table.tex")
        logger.info(f"  - metrics_table.tex")
        logger.info(f"  - statistical_tests_table.tex")
        logger.info(f"  - hansen_spa_results.json")
        logger.info(f"  - white_rc_results.json")
        logger.info(f"  - bootstrap_results.json")
        logger.info(f"  - equity_curves.csv")

    else:
        logger.warning("Skipping report generation (no portfolio or robustness results)")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*80}")

    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds() / 60

    logger.info(f"\nExperiment Summary:")
    logger.info(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Total runtime: {total_runtime:.1f} minutes ({total_runtime/60:.1f} hours)")

    logger.info(f"\nData:")
    logger.info(f"  Symbol: {config['data']['symbol']}")
    logger.info(f"  Timeframe: {config['data']['timeframe']}")
    logger.info(f"  Bars: {len(data):,}")

    logger.info(f"\nGenetic Algorithm:")
    logger.info(f"  Generations run: {generation}")
    logger.info(f"  Final population: {len(population)}")
    logger.info(f"  Best fitness: {best_pattern.fitness:.4f}")

    logger.info(f"\nPortfolio:")
    if len(portfolio) > 0:
        # Extract patterns from portfolio tuples
        if isinstance(portfolio[0], tuple):
            portfolio_patterns = [p[0] for p in portfolio]
        else:
            portfolio_patterns = portfolio

        logger.info(f"  Size: {len(portfolio)} patterns")
        logger.info(f"  LONG count: {sum(1 for p in portfolio_patterns if p.direction == 'LONG')}")
        logger.info(f"  SHORT count: {sum(1 for p in portfolio_patterns if p.direction == 'SHORT')}")
    else:
        logger.info(f"  Size: 0 patterns (empty)")

    if 'robustness_results' in locals() and robustness_results:
        logger.info(f"\nStatistical Validation:")
        if robustness_results.get('hansen_spa'):
            hansen = robustness_results['hansen_spa']
            status = "✓ PASSED" if hansen['reject_null'] else "✗ FAILED"
            logger.info(f"  Hansen SPA: {status} (p={hansen['p_value']:.4f})")

        if robustness_results.get('white_rc'):
            white = robustness_results['white_rc']
            status = "✓ PASSED" if white['reject_null'] else "✗ FAILED"
            logger.info(f"  White RC: {status} (p={white['p_value']:.4f})")

    logger.info(f"\nOutputs:")
    logger.info(f"  Reports directory: {Path(config['output']['reports_dir'])}")
    evolution_dir = Path(config['output'].get('evolution_dir', 'output_evolution'))
    if evolution_dir.exists():
        logger.info(f"  Evolution snapshots: {evolution_dir}")

    logger.info(f"\n{'='*80}")
    logger.info("Next Steps:")
    logger.info("  1. Review: output_reports/experiment_report.md")
    logger.info("  2. Check visualizations: output_reports/*.png")
    logger.info("  3. Import LaTeX tables into your paper")
    logger.info("  4. Run validation: pytest tests/")
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # SPRINT 14: Auto-run Evolution Analytics
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("GENERATING EVOLUTION ANALYTICS")
    logger.info("="*80)

    try:
        from analysis.evolution_analytics import EvolutionAnalyzer

        # Use tracking config from config.yaml
        snapshots_dir = config['ga']['evolution_tracking'].get('output_dir', './evolution_snapshots')

        analyzer = EvolutionAnalyzer(snapshots_dir=snapshots_dir)
        results_df = analyzer.run_full_analysis(output_dir="./analysis_output")

        logger.info("[OK] Evolution analytics generated successfully")
        logger.info("    - Plots: ./analysis_output/*.png")
        logger.info("    - Report: ./analysis_output/evolution_report.md")

        # Generate presentation
        logger.info("")
        logger.info("Generating HTML presentation...")
        try:
            from analysis.generate_presentation import generate_html_presentation

            success = generate_html_presentation(
                report_path="./analysis_output/evolution_report.md",
                images_dir="./analysis_output",
                output_path="./analysis_output/presentation.html"
            )

            if success:
                logger.info("")
                logger.info("="*80)
                logger.info("PRESENTATION READY!")
                logger.info("="*80)
                logger.info(f"Open in browser: {Path('./analysis_output/presentation.html').absolute()}")
                logger.info("Share this self-contained HTML file with your professor.")
                logger.info("="*80)

        except Exception as e:
            logger.warning(f"Could not generate presentation: {e}")

    except Exception as e:
        logger.error(f"Failed to generate analytics: {e}")
        logger.error("Continuing without analytics...")

    logger.info("")

    print(BANNER)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_runtime:.1f} minutes\n")

if __name__ == '__main__':
    main()
