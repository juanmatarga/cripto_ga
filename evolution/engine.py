"""
Evolution Engine — main GA loop as a testable class.

Replaces the inline loop in main.py.
"""

import random
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from grammar.mapper import decode
from strategy.phenotype import Strategy
from strategy.parameters import random_genome, GENOME_LENGTH
from evolution.operators import crossover, mutate
from evolution.selection import tournament_select
from evolution.fitness import evaluate_strategy, FAIL_FITNESS
from backtest.sampling import sample_evolution_windows

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Stats for one generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    valid_count: int
    total_count: int
    best_direction: str
    best_n_trades: int
    best_n_nodes: int
    elapsed_seconds: float


@dataclass
class EvolutionResult:
    """Result of a full evolution run."""
    best_strategies: List[Strategy]
    history: List[GenerationStats]
    total_evaluations: int
    final_generation: int


class EvolutionEngine:
    """
    Grammatical Evolution engine.

    Usage:
        engine = EvolutionEngine(config, data)
        engine.initialize(pop_size=200)
        result = engine.run(n_generations=100, patience=20)
    """

    def __init__(self, config: dict, data):
        """
        Args:
            config: Full config dict (needs 'evolution', 'costs', 'exits', 'fitness' sections)
            data: OHLCV DataFrame (evolution period only, excluding OTS)
        """
        self.config = config
        self.data = data
        self.population: List[Strategy] = []
        self.generation: int = 0
        self.history: List[GenerationStats] = []
        self.total_evaluations: int = 0

        # Evolution params
        evo_cfg = config.get('evolution', {})
        self.mutation_rate = evo_cfg.get('mutation_rate', 0.1)
        self.crossover_rate = evo_cfg.get('crossover_rate', 0.8)
        self.elitism_pct = evo_cfg.get('elitism_pct', 0.05)
        self.genome_length = evo_cfg.get('genome_length', GENOME_LENGTH)
        self.tournament_k = evo_cfg.get('tournament_k', 3)
        self.n_windows = evo_cfg.get('n_windows_per_gen', 10)
        self.window_bars = evo_cfg.get('window_bars', 2880)

    def initialize(self, pop_size: int):
        """Generate initial random population."""
        self.population = []
        attempts = 0
        max_attempts = pop_size * 5

        while len(self.population) < pop_size and attempts < max_attempts:
            genome = random_genome(self.genome_length)
            strategy = decode(genome)
            if strategy is not None:
                self.population.append(strategy)
            attempts += 1

        if len(self.population) < pop_size:
            logger.warning(
                f"Could only generate {len(self.population)}/{pop_size} valid strategies"
            )

        logger.info(f"Initialized population: {len(self.population)} strategies")

    def step(self) -> GenerationStats:
        """Execute one generation. Returns stats."""
        t0 = time.time()

        # 1. Sample fresh windows (window rotation)
        windows = sample_evolution_windows(
            self.data, n_windows=self.n_windows, window_bars=self.window_bars
        )

        # 2. Evaluate ALL strategies on this generation's windows.
        # Every strategy must be re-evaluated because windows change each gen
        # (window rotation for anti-overfitting). No caching of old fitness.
        for strategy in self.population:
            strategy.fitness = FAIL_FITNESS  # Reset before re-eval
            evaluate_strategy(strategy, windows, self.config)
            self.total_evaluations += 1

        # 3. Stats
        valid = [s for s in self.population if s.fitness != FAIL_FITNESS]
        all_sortinos = [s.fitness[0] for s in self.population]

        if valid:
            best = max(valid, key=lambda s: s.fitness[0])
            mean_fit = sum(s.fitness[0] for s in valid) / len(valid)
            sorted_fits = sorted(s.fitness[0] for s in valid)
            median_fit = sorted_fits[len(sorted_fits) // 2]
        else:
            best = self.population[0] if self.population else None
            mean_fit = -999.0
            median_fit = -999.0

        elapsed = time.time() - t0

        stats = GenerationStats(
            generation=self.generation,
            best_fitness=best.fitness[0] if best else -999.0,
            mean_fitness=mean_fit,
            median_fitness=median_fit,
            valid_count=len(valid),
            total_count=len(self.population),
            best_direction=best.direction if best else 'N/A',
            best_n_trades=best.n_trades if best else 0,
            best_n_nodes=best.n_nodes if best else 0,
            elapsed_seconds=elapsed,
        )
        self.history.append(stats)

        logger.info(
            f"Gen {self.generation:3d} | "
            f"best={stats.best_fitness:+.3f} mean={stats.mean_fitness:+.3f} | "
            f"valid={stats.valid_count}/{stats.total_count} | "
            f"trades={stats.best_n_trades} nodes={stats.best_n_nodes} | "
            f"{elapsed:.1f}s"
        )

        # 4. Selection + reproduction
        n_elite = max(2, int(len(self.population) * self.elitism_pct))
        sorted_pop = sorted(self.population, key=lambda s: s.fitness[0], reverse=True)
        elite = sorted_pop[:n_elite]

        new_pop: List[Strategy] = list(elite)
        target_size = len(self.population)

        while len(new_pop) < target_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                p1 = tournament_select(self.population, self.tournament_k)
                p2 = tournament_select(self.population, self.tournament_k)
                c1_genome, c2_genome = crossover(p1.genome, p2.genome)
            else:
                p1 = tournament_select(self.population, self.tournament_k)
                c1_genome = p1.genome[:]
                c2_genome = None

            # Mutate
            c1_genome = mutate(c1_genome, self.mutation_rate)
            s1 = decode(c1_genome)
            if s1 is not None and len(new_pop) < target_size:
                new_pop.append(s1)

            if c2_genome is not None:
                c2_genome = mutate(c2_genome, self.mutation_rate)
                s2 = decode(c2_genome)
                if s2 is not None and len(new_pop) < target_size:
                    new_pop.append(s2)

        self.population = new_pop
        self.generation += 1
        return stats

    def run(self, n_generations: int, patience: int = 20) -> EvolutionResult:
        """
        Run full evolution loop.

        Args:
            n_generations: Maximum generations
            patience: Stop after this many generations without improvement

        Returns:
            EvolutionResult with best strategies and history
        """
        best_fitness = -999.0
        stagnation = 0

        logger.info(f"Starting evolution: pop={len(self.population)}, "
                    f"max_gen={n_generations}, patience={patience}")

        for gen in range(n_generations):
            stats = self.step()

            if stats.best_fitness > best_fitness + 1e-6:
                best_fitness = stats.best_fitness
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (no improvement for {patience} gens)")
                break

        # Collect best strategies
        valid = [s for s in self.population if s.fitness != FAIL_FITNESS]
        best = sorted(valid, key=lambda s: s.fitness[0], reverse=True)

        logger.info(f"Evolution complete: {self.generation} generations, "
                    f"{self.total_evaluations} evaluations, "
                    f"best fitness={best[0].fitness[0]:.4f}" if best else "no valid strategies")

        return EvolutionResult(
            best_strategies=best[:20],
            history=self.history,
            total_evaluations=self.total_evaluations,
            final_generation=self.generation,
        )
