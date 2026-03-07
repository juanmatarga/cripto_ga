"""
Island Model — parallel sub-populations with different selection pressures.

3 islands:
- Island 0 (Exploitation): Tournament selection
- Island 1 (Diversity):    Lexicase selection
- Island 2 (Exploration):  Random selection

Migration: every M generations, top K individuals from each island
migrate to the others.
"""

import random
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

from grammar.mapper import decode
from strategy.phenotype import Strategy
from strategy.parameters import random_genome, GENOME_LENGTH
from evolution.operators import crossover, mutate
from evolution.selection import tournament_select, lexicase_select
from evolution.fitness import evaluate_strategy, FAIL_FITNESS
from evolution.archive import MAPElitesArchive, TOTAL_CELLS
from backtest.sampling import sample_evolution_windows
from data.regime_detector import detect_regime

logger = logging.getLogger(__name__)


@dataclass
class IslandStats:
    """Stats for one island in one generation."""
    island_id: int
    selection_type: str
    best_fitness: float
    mean_fitness: float
    valid_count: int
    total_count: int


def _random_select(population: List[Strategy]) -> Strategy:
    """Random selection — pure exploration."""
    return random.choice(population)


SELECTION_FUNCTIONS = {
    'tournament': tournament_select,
    'lexicase': lexicase_select,
    'random': _random_select,
}


class IslandModel:
    """
    Multi-island evolution with different selection pressures.

    Usage:
        model = IslandModel(config, data)
        model.initialize(total_pop_size=300)
        result = model.run(n_generations=100)
    """

    def __init__(self, config: dict, data):
        self.config = config
        self.data = data

        evo_cfg = config.get('evolution', {})
        island_cfg = config.get('islands', {})

        self.n_islands = island_cfg.get('n_islands', 3)
        self.migration_interval = island_cfg.get('migration_interval', 10)
        self.migration_size = island_cfg.get('migration_size', 5)
        self.mutation_rate = evo_cfg.get('mutation_rate', 0.1)
        self.crossover_rate = evo_cfg.get('crossover_rate', 0.8)
        self.elitism_pct = evo_cfg.get('elitism_pct', 0.05)
        self.genome_length = evo_cfg.get('genome_length', GENOME_LENGTH)
        self.tournament_k = evo_cfg.get('tournament_k', 3)
        self.n_windows = evo_cfg.get('n_windows_per_gen', 10)
        self.window_bars = evo_cfg.get('window_bars', 2880)

        # Selection types per island
        self.selection_types = island_cfg.get(
            'selection_types',
            ['tournament', 'lexicase', 'random']
        )[:self.n_islands]
        # Pad if fewer types than islands
        while len(self.selection_types) < self.n_islands:
            self.selection_types.append('tournament')

        self.islands: List[List[Strategy]] = [[] for _ in range(self.n_islands)]
        self.archive = MAPElitesArchive()
        self.regime_labels = detect_regime(data)
        self.generation: int = 0
        self.total_evaluations: int = 0
        self.unique_phenotypes: set = set()  # Track unique expressions evaluated
        self.history: List[List[IslandStats]] = []

    def initialize(self, total_pop_size: int):
        """Split population evenly across islands."""
        per_island = total_pop_size // self.n_islands
        remainder = total_pop_size % self.n_islands

        for island_id in range(self.n_islands):
            target = per_island + (1 if island_id < remainder else 0)
            pop = []
            attempts = 0
            while len(pop) < target and attempts < target * 5:
                genome = random_genome(self.genome_length)
                s = decode(genome)
                if s is not None:
                    pop.append(s)
                attempts += 1
            self.islands[island_id] = pop
            logger.info(f"Island {island_id} ({self.selection_types[island_id]}): "
                        f"{len(pop)} strategies")

    def step(self) -> List[IslandStats]:
        """Execute one generation across all islands."""
        # Shared windows for all islands this generation
        windows = sample_evolution_windows(
            self.data, n_windows=self.n_windows, window_bars=self.window_bars
        )

        gen_stats = []

        for island_id in range(self.n_islands):
            population = self.islands[island_id]
            if not population:
                continue

            # Evaluate
            for s in population:
                s.fitness = FAIL_FITNESS
                evaluate_strategy(s, windows, self.config,
                                  regime_labels=self.regime_labels)
                self.total_evaluations += 1
                self.unique_phenotypes.add(s.expression_raw)

            # Update archive with regime info
            for s in population:
                if s.fitness != FAIL_FITNESS and s.metrics:
                    n_trades = s.metrics.get('n_trades', 0)
                    n_windows = s.metrics.get('n_windows', 1)
                    tpm = n_trades / max(n_windows, 1)
                    regime_sortinos = self._estimate_regime_affinity(s)
                    self.archive.try_add(s, trades_per_month=tpm,
                                         regime_sortinos=regime_sortinos)

            # Stats
            valid = [s for s in population if s.fitness != FAIL_FITNESS]
            stats = IslandStats(
                island_id=island_id,
                selection_type=self.selection_types[island_id],
                best_fitness=max(s.fitness[0] for s in valid) if valid else -999.0,
                mean_fitness=(sum(s.fitness[0] for s in valid) / len(valid)) if valid else -999.0,
                valid_count=len(valid),
                total_count=len(population),
            )
            gen_stats.append(stats)

            # Selection + reproduction
            select_fn = SELECTION_FUNCTIONS[self.selection_types[island_id]]
            n_elite = max(1, int(len(population) * self.elitism_pct))
            sorted_pop = sorted(population, key=lambda s: s.fitness[0], reverse=True)
            elite = sorted_pop[:n_elite]

            new_pop: List[Strategy] = list(elite)
            target_size = len(population)

            while len(new_pop) < target_size:
                if random.random() < self.crossover_rate and len(population) >= 2:
                    p1 = select_fn(population) if self.selection_types[island_id] != 'tournament' \
                        else tournament_select(population, self.tournament_k)
                    p2 = select_fn(population) if self.selection_types[island_id] != 'tournament' \
                        else tournament_select(population, self.tournament_k)
                    c1_genome, c2_genome = crossover(p1.genome, p2.genome)
                else:
                    p1 = select_fn(population) if self.selection_types[island_id] != 'tournament' \
                        else tournament_select(population, self.tournament_k)
                    c1_genome = p1.genome[:]
                    c2_genome = None

                c1_genome = mutate(c1_genome, self.mutation_rate)
                s1 = decode(c1_genome)
                if s1 is not None and len(new_pop) < target_size:
                    new_pop.append(s1)

                if c2_genome is not None:
                    c2_genome = mutate(c2_genome, self.mutation_rate)
                    s2 = decode(c2_genome)
                    if s2 is not None and len(new_pop) < target_size:
                        new_pop.append(s2)

            self.islands[island_id] = new_pop

        # Migration
        if (self.generation + 1) % self.migration_interval == 0:
            self._migrate()

        # Archive injection: every 5 gens, inject archive strategies
        if (self.generation + 1) % 5 == 0 and self.archive.n_occupied > 0:
            self._inject_from_archive()

        self.history.append(gen_stats)
        self.generation += 1

        # Log summary
        best_overall = max(
            (s.best_fitness for s in gen_stats), default=-999.0
        )
        logger.info(
            f"Gen {self.generation:3d} | "
            + " | ".join(
                f"I{s.island_id}({s.selection_type[:4]})={s.best_fitness:+.2f}"
                for s in gen_stats
            )
            + f" | archive={self.archive.n_occupied}/{TOTAL_CELLS}"
        )

        return gen_stats

    def _estimate_regime_affinity(self, strategy: Strategy) -> Dict[str, float]:
        """
        Estimate which regime a strategy is best suited for.

        Uses signal distribution across regimes as a proxy:
        the regime where the strategy fires most signals gets
        the strategy's actual Sortino; others get a lower estimate.
        """
        from strategy.vectorized_eval import generate_signals

        try:
            signals = generate_signals(strategy, self.data)
            signal_count = {}
            for regime in ['bull', 'bear', 'sideways']:
                mask = self.regime_labels == regime
                signal_count[regime] = int(signals[mask].sum())

            total_signals = sum(signal_count.values())
            if total_signals == 0:
                return {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0}

            # Distribute the strategy's Sortino proportional to signal concentration
            base_sortino = strategy.fitness[0]
            result = {}
            for regime in ['bull', 'bear', 'sideways']:
                proportion = signal_count[regime] / total_signals
                result[regime] = base_sortino * proportion

            return result
        except Exception:
            return {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0}

    def _migrate(self):
        """Migrate top individuals between islands."""
        n = self.migration_size
        logger.info(f"Migration: sending top {n} from each island")

        # Collect top-n from each island
        migrants = []
        for island_id in range(self.n_islands):
            pop = self.islands[island_id]
            sorted_pop = sorted(pop, key=lambda s: s.fitness[0], reverse=True)
            migrants.append(sorted_pop[:n])

        # Send migrants to other islands (replace worst individuals)
        for island_id in range(self.n_islands):
            incoming = []
            for src_id in range(self.n_islands):
                if src_id != island_id:
                    incoming.extend(migrants[src_id])

            if incoming and len(self.islands[island_id]) > len(incoming):
                # Replace worst individuals
                pop = sorted(self.islands[island_id],
                             key=lambda s: s.fitness[0], reverse=True)
                pop = pop[:len(pop) - len(incoming)] + incoming
                self.islands[island_id] = pop

    def _inject_from_archive(self):
        """Inject archive strategies into islands as quality immigrants."""
        per_island = max(1, self.migration_size // 2)
        archive_strategies = self.archive.sample_for_reproduction(
            per_island * self.n_islands
        )

        if not archive_strategies:
            return

        for island_id in range(self.n_islands):
            start = island_id * per_island
            end = start + per_island
            immigrants = archive_strategies[start:end]

            if immigrants and len(self.islands[island_id]) > len(immigrants):
                # Replace worst
                pop = sorted(self.islands[island_id],
                             key=lambda s: s.fitness[0], reverse=True)
                pop = pop[:len(pop) - len(immigrants)] + immigrants
                self.islands[island_id] = pop

        logger.debug(f"Injected {len(archive_strategies)} archive strategies")

    def run(self, n_generations: int, patience: int = 20):
        """
        Run full island model evolution.

        Returns dict with best strategies, archive, and history.
        """
        best_fitness = -999.0
        stagnation = 0

        total_pop = sum(len(island) for island in self.islands)
        logger.info(f"Starting island model: {self.n_islands} islands, "
                    f"total_pop={total_pop}, max_gen={n_generations}")

        for gen in range(n_generations):
            gen_stats = self.step()

            current_best = max(
                (s.best_fitness for s in gen_stats), default=-999.0
            )
            if current_best > best_fitness + 1e-6:
                best_fitness = current_best
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (no improvement for {patience} gens)")
                break

        # Collect diverse strategies: de-duplicated population + archive
        all_valid = []
        for island in self.islands:
            all_valid.extend(s for s in island if s.fitness != FAIL_FITNESS)
        all_valid.sort(key=lambda s: s.fitness[0], reverse=True)

        # De-duplicate by expression
        seen = set()
        unique_pop = []
        for s in all_valid:
            if s.expression_raw not in seen:
                seen.add(s.expression_raw)
                unique_pop.append(s)

        # Add archive strategies (diverse by definition — one per niche)
        archive_strats = self.archive.get_all_strategies()
        for s in archive_strats:
            if s.expression_raw not in seen:
                seen.add(s.expression_raw)
                unique_pop.append(s)

        unique_pop.sort(key=lambda s: s.fitness[0], reverse=True)

        return {
            'best_strategies': unique_pop[:30],
            'archive': self.archive,
            'history': self.history,
            'total_evaluations': self.total_evaluations,
            'unique_phenotypes': len(self.unique_phenotypes),
            'final_generation': self.generation,
        }
