"""
Evolution Engine v2 — NSGA-II with evaluation caching and MAP-Elites.

Replaces v1's scalar-fitness tournament-selection loop.
"""

import random
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from grammar.mapper import decode
from strategy.phenotype import Strategy
from strategy.parameters import random_genome, GENOME_LENGTH
from evolution.operators import crossover, mutate
from evolution.nsga2 import non_dominated_sort, select_parents, select_by_stability
from evolution.fitness import evaluate_single_window, compute_objectives, FAIL_FITNESS
from evolution.cache import EvalCache
from evolution.archive import MAPElitesArchive
from backtest.sampling import sample_windows_with_rotation
from data.multi_timeframe import prepare_multi_tf_data

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Stats for one generation."""
    generation: int
    best_sortino: float
    best_return: float
    median_sortino: float
    front1_size: int
    valid_count: int
    total_count: int
    cache_hit_rate: float
    archive_coverage: float
    elapsed_seconds: float


@dataclass
class EvolutionResult:
    """Result of a full evolution run."""
    pareto_front: List[Strategy]
    all_strategies: List[Strategy]
    archive: MAPElitesArchive
    history: List[GenerationStats]
    total_evaluations: int
    final_generation: int


class EvolutionEngine:
    """
    NSGA-II Grammatical Evolution engine.

    Usage:
        engine = EvolutionEngine(config, data)
        engine.initialize(pop_size=200)
        result = engine.run(n_generations=100, patience=30)
    """

    def __init__(self, config: dict, data):
        self.config = config
        self.data = data
        self.population: List[Strategy] = []
        self.generation: int = 0
        self.history: List[GenerationStats] = []
        self.total_evaluations: int = 0

        evo_cfg = config.get('evolution', {})
        self.mutation_rate = evo_cfg.get('mutation_rate', 0.1)
        self.crossover_rate = evo_cfg.get('crossover_rate', 0.8)
        self.genome_length = evo_cfg.get('genome_length', GENOME_LENGTH)
        self.n_windows = evo_cfg.get('n_windows_per_gen', 5)
        self.window_bars = evo_cfg.get('window_bars', 8640)
        self.max_generations = evo_cfg.get('max_generations', 100)
        self.archive_parent_pct = evo_cfg.get('archive_parent_pct', 0.10)
        self.parsimony_coeff = config.get('fitness', {}).get('parsimony_coefficient', 0.02)

        self.cache = EvalCache()
        self.archive = MAPElitesArchive()
        self._prev_windows: List[Tuple] = []

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
            logger.warning(f"Only {len(self.population)}/{pop_size} valid strategies")

        logger.info(f"Initialized: {len(self.population)} strategies")

    def _evaluate_population(self, strategies: List[Strategy],
                             windows: List[Tuple]) -> int:
        """
        Evaluate strategies on windows using cache.
        Sets objectives, stability, constraint_violation in-place.
        Returns number of new evaluations (cache misses).
        """
        new_evals = 0

        for strategy in strategies:
            if not strategy.conditions:
                strategy.objectives = (-999.0, -999.0)
                strategy.stability = -999.0
                strategy.constraint_violation = 100.0
                continue

            genome_key = tuple(strategy.genome)
            window_metrics = []

            for window_df, wid in windows:
                cached = self.cache.get(genome_key, wid)
                if cached is not None:
                    self.cache.record_hit()
                    window_metrics.append(cached)
                else:
                    self.cache.record_miss()
                    tf_data = prepare_multi_tf_data(window_df)
                    metrics = evaluate_single_window(
                        strategy, window_df, self.config, tf_data=tf_data
                    )
                    if metrics is not None:
                        self.cache.put(genome_key, wid, metrics)
                        window_metrics.append(metrics)
                    new_evals += 1

            compute_objectives(strategy, window_metrics, self.parsimony_coeff)

        return new_evals

    def _breed_offspring(self, parents: List[Strategy], n: int) -> List[Strategy]:
        """Generate n offspring from parents via crossover + mutation."""
        offspring = []
        while len(offspring) < n:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                c1_genome, c2_genome = crossover(p1.genome, p2.genome)
            else:
                p1 = random.choice(parents)
                c1_genome = p1.genome[:]
                c2_genome = None

            c1_genome = mutate(c1_genome, self.mutation_rate,
                               self.generation, self.max_generations)
            s1 = decode(c1_genome)
            if s1 is not None:
                offspring.append(s1)

            if c2_genome is not None and len(offspring) < n:
                c2_genome = mutate(c2_genome, self.mutation_rate,
                                   self.generation, self.max_generations)
                s2 = decode(c2_genome)
                if s2 is not None:
                    offspring.append(s2)

        return offspring[:n]

    def step(self) -> GenerationStats:
        """Execute one NSGA-II generation."""
        t0 = time.time()
        pop_size = len(self.population)

        # 1. Sample windows with partial rotation
        windows = sample_windows_with_rotation(
            self.data, n_windows=self.n_windows,
            window_bars=self.window_bars,
            previous_windows=self._prev_windows,
        )
        self._prev_windows = windows

        # Evict stale cache entries
        active_ids = {wid for _, wid in windows}
        self.cache.evict_except(active_ids)
        self.cache.reset_counters()

        # 2. Select parents: 90% via binary tournament, 10% from archive
        n_archive = max(1, int(pop_size * self.archive_parent_pct))
        n_tournament = pop_size - n_archive

        tournament_parents = select_parents(self.population, n_tournament)
        archive_parents = self.archive.sample_for_reproduction(n_archive)
        all_parents = tournament_parents + archive_parents

        # 3. Breed offspring (lambda = mu)
        offspring = self._breed_offspring(all_parents, pop_size)

        # 4. Evaluate offspring
        new_evals = self._evaluate_population(offspring, windows)
        self.total_evaluations += new_evals

        # Re-evaluate parents on new windows (cache hits for kept windows)
        parent_evals = self._evaluate_population(self.population, windows)
        self.total_evaluations += parent_evals

        # 5. Merge parents + offspring (mu+lambda)
        combined = list(self.population) + offspring

        # 6. Non-dominated sort
        fronts = non_dominated_sort(combined)

        # 7. Fill new population from fronts
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                remaining = pop_size - len(new_pop)
                selected = select_by_stability(front, remaining)
                new_pop.extend(selected)
                break

        # 8. Update MAP-Elites archive
        for s in new_pop:
            if s.constraint_violation <= 0 and s.metrics:
                trades_per_month = s.n_trades / max(len(windows), 1) / 3.0
                self.archive.try_add(s, trades_per_month, regime_sortinos=None)

        # Stats
        cache_stats = self.cache.stats()
        front1 = fronts[0] if fronts else []
        feasible = [s for s in new_pop if s.constraint_violation <= 0]

        if feasible:
            best = max(feasible, key=lambda s: s.objectives[0])
            best_return = max(feasible, key=lambda s: s.objectives[1])
            sortinos = [s.objectives[0] for s in feasible]
            median_s = sorted(sortinos)[len(sortinos) // 2]
        else:
            best = new_pop[0] if new_pop else None
            best_return = best
            median_s = -999.0

        elapsed = time.time() - t0
        stats = GenerationStats(
            generation=self.generation,
            best_sortino=best.objectives[0] if best else -999.0,
            best_return=best_return.objectives[1] if best_return else -999.0,
            median_sortino=median_s,
            front1_size=len(front1),
            valid_count=len(feasible),
            total_count=len(new_pop),
            cache_hit_rate=cache_stats['hit_rate'],
            archive_coverage=self.archive.coverage,
            elapsed_seconds=elapsed,
        )
        self.history.append(stats)

        logger.info(
            f"Gen {self.generation:3d} | "
            f"F1={stats.front1_size} feasible={stats.valid_count}/{stats.total_count} | "
            f"best_S={stats.best_sortino:+.2f} best_R={stats.best_return:+.1f}% | "
            f"cache={stats.cache_hit_rate:.0%} archive={stats.archive_coverage:.0%} | "
            f"{elapsed:.1f}s"
        )

        self.population = new_pop
        self.generation += 1
        return stats

    def run(self, n_generations: int, patience: int = 30) -> EvolutionResult:
        """
        Run full NSGA-II evolution.

        Patience tracks stagnation of front1 size + best Sortino.
        """
        best_metric = -999.0
        stagnation = 0

        logger.info(f"Starting NSGA-II evolution: pop={len(self.population)}, "
                    f"max_gen={n_generations}, patience={patience}")

        for gen in range(n_generations):
            stats = self.step()

            current_metric = stats.best_sortino + stats.front1_size * 0.01
            if current_metric > best_metric + 0.01:
                best_metric = current_metric
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (stagnation={patience})")
                break

        # Collect Pareto front
        fronts = non_dominated_sort(self.population)
        pareto = fronts[0] if fronts else []
        pareto_feasible = [s for s in pareto if s.constraint_violation <= 0]
        pareto_sorted = sorted(pareto_feasible, key=lambda s: s.objectives[0], reverse=True)

        logger.info(f"Evolution complete: {self.generation} gens, "
                    f"{self.total_evaluations} evals, "
                    f"Pareto front: {len(pareto_sorted)} strategies")

        return EvolutionResult(
            pareto_front=pareto_sorted,
            all_strategies=self.population,
            archive=self.archive,
            history=self.history,
            total_evaluations=self.total_evaluations,
            final_generation=self.generation,
        )
