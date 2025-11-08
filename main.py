"""
BTC/USDT Pattern Discovery - Main Orchestrator
Genetic Algorithm + Walk-Forward + Statistical Validation
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime
import sys

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
    # FASE 0: Setup
    print("="*80)
    print("BTC/USDT PATTERN DISCOVERY - GENETIC ALGORITHM")
    print("="*80)

    config = load_config()
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

    # TODO: Implementar en Sprint 1
    # from loader import load_binance_data
    # data = load_binance_data(config)
    # logger.info(f"✓ Data loaded: {len(data)} candles")

    # FASE 2: GA Evolution
    logger.info("\n" + "="*80)
    logger.info("FASE 2: GENETIC ALGORITHM EVOLUTION")
    logger.info("="*80)

    # TODO: Implementar en Sprint 2
    # from ga_patterns.generator import run_genetic_algorithm
    # top_patterns = run_genetic_algorithm(data, config)

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
