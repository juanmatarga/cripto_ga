"""
BTC/USDT Pattern Discovery - Main Orchestrator
Genetic Algorithm + Walk-Forward + Statistical Validation
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

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
    config_path = 'config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # FASE 0: Setup
    print("="*80)
    print("BTC/USDT PATTERN DISCOVERY - GENETIC ALGORITHM")
    print("="*80)
    print(f"Using config: {config_path}")

    config = load_config(config_path)
    validate_config(config)
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

    # FASE 2: GA Evolution
    logger.info("\n" + "="*80)
    logger.info("FASE 2: GENETIC ALGORITHM EVOLUTION (BIDIRECTIONAL)")
    logger.info("="*80)

    import random
    import copy
    from ga_patterns.generator import (
        initialize_population, tournament_selection,
        subtree_crossover, mutate_pattern
    )
    from ga_patterns.fitness import evaluate_population
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

    population = initialize_population(population_size, generation=0, config=config['ga'])

    # Evaluate initial
    evaluate_population(population, data, config)

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
        logger.info(f"\n--- Generation {generation}/{max_generations} ---")

        # New population
        new_population = []

        # Elitism
        population_sorted = sorted(population, key=lambda p: p.fitness, reverse=True)
        for i in range(elitism):
            new_population.append(copy.deepcopy(population_sorted[i]))

        # Generate offspring
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                offspring = subtree_crossover(parent1, parent2, generation, config['ga'])
            else:
                offspring = copy.deepcopy(parent1)

            if random.random() < mutation_rate:
                offspring = mutate_pattern(offspring, generation, config['ga'])

            new_population.append(offspring)

        population = new_population

        # Evaluate
        evaluate_population(population, data, config)

        # Tracking
        current_best = max(population, key=lambda p: p.fitness)
        mean_fitness = np.mean([p.fitness for p in population])
        best_fitness_history.append(current_best.fitness)

        logger.info(f"Best: {current_best.fitness:.4f} ({current_best.direction})")
        logger.info(f"Mean: {mean_fitness:.4f}")

        # Track
        tracker.track_generation(generation, population, current_best, mean_fitness)

        # Early stopping
        if current_best.fitness > best_pattern.fitness:
            best_pattern = current_best
            generations_without_improvement = 0
            logger.info("[OK] New best!")
        else:
            generations_without_improvement += 1
            logger.info(f"No improvement ({generations_without_improvement}/{patience})")

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

    # Display top 5
    logger.info(f"\nTop 5 Patterns:")
    for i, pattern in enumerate(top_patterns[:5], 1):
        logger.info(f"\n{i}. {pattern.direction} | Fit: {pattern.fitness:.4f} | L:{pattern.fitness_long:.4f} S:{pattern.fitness_short:.4f}")
        logger.info(f"   {pattern.expression}")

    # FASE 3: Pattern Selection
    logger.info("\n" + "="*80)
    logger.info("FASE 3: PORTFOLIO SELECTION")
    logger.info("="*80)

    # TODO: Implementar en Sprint 3

    # FASE 4: Statistical Validation
    logger.info("\n" + "="*80)
    logger.info("FASE 4: STATISTICAL VALIDATION")
    logger.info("="*80)

    # TODO: Implementar en Sprint 4

    # FASE 5: Report Generation
    logger.info("\n" + "="*80)
    logger.info("FASE 5: REPORT GENERATION")
    logger.info("="*80)

    # TODO: Implementar en Sprint 5

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*80)

if __name__ == '__main__':
    main()
