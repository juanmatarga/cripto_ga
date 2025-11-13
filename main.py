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

        logger.info("")
        logger.info("GENERATION SUMMARY")
        logger.info("-" * 80)
        logger.info(f"Valid patterns: {len(valid_patterns)}/100 ({len(valid_patterns)}%)")
        logger.info(f"  LONG: {len(valid_long)}/{len(long_patterns)} ({len(valid_long)*100//len(long_patterns) if long_patterns else 0}%)")
        logger.info(f"  SHORT: {len(valid_short)}/{len(short_patterns)} ({len(valid_short)*100//len(short_patterns) if short_patterns else 0}%)")

        if best_long and best_long.fitness > -999:
            logger.info(f"\nBest LONG pattern (fitness={best_long.fitness:.4f}):")
            logger.info(f"  {best_long.to_readable()}")
            if hasattr(best_long, 'metrics') and best_long.metrics:
                logger.info(f"  Sharpe: {best_long.metrics.get('sharpe', 0):.2f}, CAGR: {best_long.metrics.get('cagr', 0)*100:.1f}%, Trades: {best_long.n_trades}")

        if best_short and best_short.fitness > -999:
            logger.info(f"\nBest SHORT pattern (fitness={best_short.fitness:.4f}):")
            logger.info(f"  {best_short.to_readable()}")
            if hasattr(best_short, 'metrics') and best_short.metrics:
                logger.info(f"  Sharpe: {best_short.metrics.get('sharpe', 0):.2f}, CAGR: {best_short.metrics.get('cagr', 0)*100:.1f}%, Trades: {best_short.n_trades}")

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
    logger.info("🎉 EXPERIMENT COMPLETE 🎉")
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

    print(BANNER)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_runtime:.1f} minutes\n")

if __name__ == '__main__':
    main()
