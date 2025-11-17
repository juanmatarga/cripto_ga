"""
Evolution Tracker - Enhanced with LONG/SHORT tracking
"""

import json
import yaml
from pathlib import Path
from typing import List
from datetime import datetime
import logging

from ga_patterns.chromosome import Pattern

logger = logging.getLogger(__name__)

class EvolutionTracker:
    """Trackea evolución con best LONG/SHORT por generación."""

    def __init__(self, config: dict):
        self.config = config
        self.tracking_config = config['ga']['evolution_tracking']
        self.enabled = self.tracking_config.get('enabled', True)

        if not self.enabled:
            logger.info("Evolution tracking DISABLED")
            return

        self.output_dir = Path(self.tracking_config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.sample_size = self.tracking_config.get('sample_size_per_generation', 5)
        self.save_every = self.tracking_config.get('save_every_n_generations', 10)

        # History
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_long_history = []
        self.best_short_history = []
        self.generation_samples = {}

        logger.info(f"[OK] Evolution tracking ENABLED")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Sample size/gen: {self.sample_size}")
        logger.info(f"  Save every: {self.save_every} gens")

    def track_generation(self, generation: int, population: List[Pattern],
                       best_pattern: Pattern, mean_fitness: float):
        """
        SPRINT 14: Enhanced tracking with comprehensive evolution analytics.

        Tracks:
        - Fitness statistics (mean, std, min, max, median)
        - Diversity metrics (unique patterns, module diversity)
        - Performance metrics (Sharpe, CAGR, trades distribution)
        - Evolution indicators (improvement rate, stagnation)
        """
        if not self.enabled:
            return

        import numpy as np

        # Separar por dirección
        long_patterns = [p for p in population if p.direction == 'LONG']
        short_patterns = [p for p in population if p.direction == 'SHORT']

        # Valid patterns only (fitness > -999)
        valid_patterns = [p for p in population if p.fitness > -999]
        valid_long = [p for p in long_patterns if p.fitness > -999]
        valid_short = [p for p in short_patterns if p.fitness > -999]

        # Best LONG
        best_long = max(long_patterns, key=lambda p: p.fitness) if long_patterns else None

        # Best SHORT
        best_short = max(short_patterns, key=lambda p: p.fitness) if short_patterns else None

        # SPRINT 14: Fitness statistics (comprehensive)
        if valid_patterns:
            fitness_values = [p.fitness for p in valid_patterns]
            fitness_stats = {
                'mean': float(np.mean(fitness_values)),
                'std': float(np.std(fitness_values)),
                'min': float(np.min(fitness_values)),
                'max': float(np.max(fitness_values)),
                'median': float(np.median(fitness_values)),
                'q25': float(np.percentile(fitness_values, 25)),
                'q75': float(np.percentile(fitness_values, 75))
            }
        else:
            fitness_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'q25': 0, 'q75': 0}

        # SPRINT 14: Diversity metrics
        unique_module_sets = set()
        all_modules = []
        for p in population:
            unique_module_sets.add(tuple(sorted(p.modules)))
            all_modules.extend(p.modules)

        module_diversity = len(unique_module_sets) / len(population) if population else 0

        # SPRINT 14: Performance metrics aggregation
        performance_metrics = {
            'sharpe': [],
            'cagr': [],
            'trades': [],
            'win_rate': []
        }

        for p in valid_patterns:
            if hasattr(p, 'metrics') and p.metrics:
                performance_metrics['sharpe'].append(p.metrics.get('sharpe', 0))
                performance_metrics['cagr'].append(p.metrics.get('cagr', 0))
            if hasattr(p, 'n_trades'):
                performance_metrics['trades'].append(p.n_trades)
            if hasattr(p, 'fitness_components') and 'win_rate' in p.fitness_components:
                performance_metrics['win_rate'].append(p.fitness_components['win_rate'])

        avg_performance = {
            'avg_sharpe': float(np.mean(performance_metrics['sharpe'])) if performance_metrics['sharpe'] else 0,
            'avg_cagr': float(np.mean(performance_metrics['cagr'])) if performance_metrics['cagr'] else 0,
            'avg_trades': float(np.mean(performance_metrics['trades'])) if performance_metrics['trades'] else 0,
            'avg_win_rate': float(np.mean(performance_metrics['win_rate'])) if performance_metrics['win_rate'] else 0
        }

        # Guardar history (legacy)
        self.best_fitness_history.append({
            'generation': generation,
            'fitness': best_pattern.fitness,
            'direction': best_pattern.direction
        })

        self.mean_fitness_history.append({
            'generation': generation,
            'fitness': mean_fitness
        })

        if best_long:
            self.best_long_history.append({
                'generation': generation,
                'fitness': best_long.fitness
            })

        if best_short:
            self.best_short_history.append({
                'generation': generation,
                'fitness': best_short.fitness
            })

        # SPRINT 11: Save top 5 LONG + top 5 SHORT instead of random samples
        # Sort patterns by fitness
        long_sorted = sorted(long_patterns, key=lambda p: p.fitness, reverse=True)
        short_sorted = sorted(short_patterns, key=lambda p: p.fitness, reverse=True)

        # Take top 5 of each (or less if not available)
        top_long = long_sorted[:5]
        top_short = short_sorted[:5]

        # Filter out invalid patterns (fitness = -999)
        top_long_valid = [p for p in top_long if p.fitness > -999]
        top_short_valid = [p for p in top_short if p.fitness > -999]

        # Module usage statistics
        from collections import Counter
        module_usage = dict(Counter(all_modules).most_common(10))

        # SPRINT 14: Enhanced snapshot with comprehensive metrics
        self.generation_samples[generation] = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),

            # Best patterns
            'best_overall': best_pattern.to_dict(),
            'best_long': best_long.to_dict() if best_long else None,
            'best_short': best_short.to_dict() if best_short else None,

            # SPRINT 14: Population statistics
            'population_stats': {
                'total': len(population),
                'valid': len(valid_patterns),
                'valid_pct': (len(valid_patterns) / len(population) * 100) if population else 0,
                'long_total': len(long_patterns),
                'short_total': len(short_patterns),
                'valid_long': len(valid_long),
                'valid_short': len(valid_short),
                'valid_long_pct': (len(valid_long) / len(long_patterns) * 100) if long_patterns else 0,
                'valid_short_pct': (len(valid_short) / len(short_patterns) * 100) if short_patterns else 0
            },

            # SPRINT 14: Fitness statistics
            'fitness_stats': fitness_stats,

            # SPRINT 14: Diversity metrics
            'diversity': {
                'unique_patterns_pct': module_diversity * 100,
                'unique_patterns_count': len(unique_module_sets),
                'top_modules': module_usage
            },

            # SPRINT 14: Performance metrics
            'performance': avg_performance,
            # SPRINT 11: Top performers instead of random samples
            'top_long': [
                {
                    'rank': i+1,
                    'fitness': p.fitness,
                    'readable': p.to_readable(),
                    'modules': p.modules,
                    'logic': p.logic,
                    'window': p.window,
                    'generation_created': p.generation_created,
                    'expression': p.to_expression(),
                    'sharpe': p.metrics.get('sharpe', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'cagr': p.metrics.get('cagr', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'upi': p.metrics.get('upi', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'max_dd': p.metrics.get('max_dd', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'n_trades': p.n_trades if hasattr(p, 'n_trades') else None
                }
                for i, p in enumerate(top_long_valid)
            ],
            'top_short': [
                {
                    'rank': i+1,
                    'fitness': p.fitness,
                    'readable': p.to_readable(),
                    'modules': p.modules,
                    'logic': p.logic,
                    'window': p.window,
                    'generation_created': p.generation_created,
                    'expression': p.to_expression(),
                    'sharpe': p.metrics.get('sharpe', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'cagr': p.metrics.get('cagr', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'upi': p.metrics.get('upi', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'max_dd': p.metrics.get('max_dd', None) if hasattr(p, 'metrics') and p.metrics else None,
                    'n_trades': p.n_trades if hasattr(p, 'n_trades') else None
                }
                for i, p in enumerate(top_short_valid)
            ]
        }

        logger.info(f"[OK] Tracked generation {generation}: {len(top_long_valid)} top LONG + {len(top_short_valid)} top SHORT")

        # Snapshot
        if generation % self.save_every == 0:
            self._save_snapshot(generation)

    def _save_snapshot(self, generation: int):
        """Guarda snapshot."""
        if not self.enabled:
            return

        filename = self.output_dir / f"generation_{generation:04d}.json"

        snapshot = {
            'generation': generation,
            'data': self.generation_samples.get(generation, {}),
            'metadata': {
                'config': {
                    'population': self.config['ga']['population'],
                    'mutation_rate': self.config['ga']['mutation_rate'],
                    'crossover_rate': self.config['ga']['crossover_rate'],
                },
                'timestamp': datetime.now().isoformat()
            }
        }

        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"  [OK] Saved snapshot: {filename.name}")

    def save_final_summary(self, final_generation: int, top_patterns: List[Pattern]):
        """Guarda resumen final con análisis LONG/SHORT."""
        if not self.enabled:
            return

        summary_file = self.output_dir / "evolution_summary.yaml"

        # Separar top patterns
        top_long = [p for p in top_patterns if p.direction == 'LONG'][:5]
        top_short = [p for p in top_patterns if p.direction == 'SHORT'][:5]

        summary = {
            'evolution_metadata': {
                'total_generations': final_generation,
                'population_size': self.config['ga']['population'],
                'mutation_rate': self.config['ga']['mutation_rate'],
                'crossover_rate': self.config['ga']['crossover_rate'],
                'timestamp': datetime.now().isoformat()
            },
            'fitness_progression': {
                'best_overall': self.best_fitness_history,
                'mean': self.mean_fitness_history,
                'best_long': self.best_long_history,
                'best_short': self.best_short_history
            },
            'top_patterns_overall': [
                {'rank': i + 1, 'fitness': p.fitness, 'direction': p.direction, 'pattern': p.to_dict()}
                for i, p in enumerate(top_patterns[:10])
            ],
            'top_patterns_long': [
                {'rank': i + 1, 'fitness': p.fitness, 'pattern': p.to_dict()}
                for i, p in enumerate(top_long)
            ],
            'top_patterns_short': [
                {'rank': i + 1, 'fitness': p.fitness, 'pattern': p.to_dict()}
                for i, p in enumerate(top_short)
            ],
            'statistics': {
                'best_fitness_ever': max([h['fitness'] for h in self.best_fitness_history]),
                'final_best_fitness': self.best_fitness_history[-1]['fitness'] if self.best_fitness_history else -999,
                'improvement': self.best_fitness_history[-1]['fitness'] - self.best_fitness_history[0]['fitness'] if len(self.best_fitness_history) > 1 else 0,
                'best_long_fitness': max([h['fitness'] for h in self.best_long_history]) if self.best_long_history else -999,
                'best_short_fitness': max([h['fitness'] for h in self.best_short_history]) if self.best_short_history else -999
            }
        }

        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        logger.info(f"\n{'='*80}")
        logger.info(f"EVOLUTION SUMMARY SAVED")
        logger.info(f"{'='*80}")
        logger.info(f"File: {summary_file}")
        logger.info(f"Total generations: {final_generation}")
        logger.info(f"Best fitness ever: {summary['statistics']['best_fitness_ever']:.4f}")
        logger.info(f"Best LONG: {summary['statistics']['best_long_fitness']:.4f}")
        logger.info(f"Best SHORT: {summary['statistics']['best_short_fitness']:.4f}")
