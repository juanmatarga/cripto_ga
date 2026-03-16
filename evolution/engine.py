"""
Evolution Engine v5 — Complexity-Niched NSGA-II + Island Mechanisms.

Combines v4 complexity niches with the old island model's diversity:
1. Complexity-niched population (1-node, 2-node, 3-node)
2. Hybrid selection per niche: NSGA-II tournament + lexicase + random
3. Structure-preserving crossover/mutation
4. Migration between niches every N gens
5. Archive injection every 5 gens
6. Regime-aware archive
"""

import random
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from grammar.mapper import decode
from grammar.bnf import GRAMMAR
from strategy.phenotype import Strategy
from strategy.parameters import random_genome, GENOME_LENGTH
from evolution.operators import crossover, mutate
from evolution.selection import lexicase_select
from evolution.nsga2 import (
    non_dominated_sort, select_parents, select_by_crowding,
    compute_crowding_distance,
)
from evolution.fitness import evaluate_single_window, compute_objectives, FAIL_FITNESS
from evolution.cache import EvalCache
from evolution.archive import MAPElitesArchive
from backtest.sampling import sample_windows_with_rotation
from data.multi_timeframe import prepare_multi_tf_data

logger = logging.getLogger(__name__)

# The structure codon determines entry_rule complexity.
# Grammar expansion order: <strategy>(codon 0) → <direction>(codon 1) → <entry_rule>(codon 2)
# entry_rule has 4 productions: [0]=1-node, [1]=2-node, [2]=3-node, [3]=3-node(grouped)
STRUCTURE_CODON_IDX = 2

# Population split across complexity niches
NICHE_PROPORTIONS = {1: 0.25, 2: 0.375, 3: 0.375}


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
    complexity_dist: Dict[int, int] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result of a full evolution run."""
    pareto_front: List[Strategy]
    all_strategies: List[Strategy]
    archive: MAPElitesArchive
    history: List[GenerationStats]
    total_evaluations: int
    final_generation: int


def _fix_structure_codon(genome: List[int], target_nodes: int):
    """Force the structure codon to produce the target complexity."""
    base = (genome[STRUCTURE_CODON_IDX] // 4) * 4
    if target_nodes == 1:
        genome[STRUCTURE_CODON_IDX] = base  # mod 4 = 0
    elif target_nodes == 2:
        genome[STRUCTURE_CODON_IDX] = base + 1  # mod 4 = 1
    else:  # 3-node
        genome[STRUCTURE_CODON_IDX] = base + random.choice([2, 3])


def _get_complexity(strategy: Strategy) -> int:
    """Get the complexity niche: 1, 2, or 3."""
    return min(strategy.n_nodes, 3)


class EvolutionEngine:
    """
    Complexity-Niched NSGA-II engine v4.

    Population is divided into 3 niches by strategy complexity.
    NSGA-II selection happens WITHIN each niche, preventing simple
    strategies from outcompeting complex ones.
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
        self.n_windows = evo_cfg.get('n_windows_per_gen', 10)
        self.window_bars = evo_cfg.get('window_bars', 5760)
        self.max_generations = evo_cfg.get('max_generations', 100)
        self.archive_parent_pct = evo_cfg.get('archive_parent_pct', 0.10)
        self.parsimony_coeff = config.get('fitness', {}).get('parsimony_coefficient', 0.0)
        self.max_phenotype_copies = evo_cfg.get('max_phenotype_copies', 2)
        self.keep_ratio = evo_cfg.get('keep_ratio', 0.6)

        # Window rotation is the default
        self.use_fixed_windows = evo_cfg.get('use_fixed_windows', False)

        # Migration and injection settings (from old island model)
        self.migration_interval = evo_cfg.get('migration_interval', 10)
        self.migration_size = evo_cfg.get('migration_size', 3)
        self.archive_inject_interval = evo_cfg.get('archive_inject_interval', 5)

        self.cache = EvalCache()
        self.archive = MAPElitesArchive()
        self._prev_windows: List[Tuple] = []
        self._fixed_windows: Optional[List[Tuple]] = None
        self._tf_data_cache: Dict[str, dict] = {}

    def initialize(self, pop_size: int):
        """Generate initial population with BALANCED complexity."""
        self.population = []

        for target_nodes, proportion in NICHE_PROPORTIONS.items():
            n = int(pop_size * proportion)
            niche_pop = self._generate_with_complexity(n, target_nodes)
            self.population.extend(niche_pop)

        # Fill remaining slots randomly
        while len(self.population) < pop_size:
            genome = random_genome(self.genome_length)
            s = decode(genome)
            if s is not None:
                self.population.append(s)

        dist = Counter(_get_complexity(s) for s in self.population)
        logger.info(f"Initialized: {len(self.population)} strategies — "
                    f"1n={dist[1]} 2n={dist[2]} 3n={dist[3]}")

    def _generate_with_complexity(self, n: int, target_nodes: int) -> List[Strategy]:
        """Generate n strategies with specific complexity."""
        strategies = []
        attempts = 0
        max_attempts = n * 10

        while len(strategies) < n and attempts < max_attempts:
            genome = random_genome(self.genome_length)
            _fix_structure_codon(genome, target_nodes)
            s = decode(genome)
            if s is not None and s.n_nodes == target_nodes:
                strategies.append(s)
            attempts += 1

        if len(strategies) < n:
            logger.warning(f"Only generated {len(strategies)}/{n} "
                           f"strategies for {target_nodes}-node niche")
        return strategies

    def _evaluate_population(self, strategies: List[Strategy],
                             windows: List[Tuple]) -> int:
        """Evaluate strategies on windows. Returns number of new evaluations."""
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
                    tf_data = self._tf_data_cache.get(wid)
                    if tf_data is None:
                        tf_data = prepare_multi_tf_data(window_df)
                        self._tf_data_cache[wid] = tf_data
                    metrics = evaluate_single_window(
                        strategy, window_df, self.config, tf_data=tf_data
                    )
                    if metrics is not None:
                        self.cache.put(genome_key, wid, metrics)
                        window_metrics.append(metrics)
                    new_evals += 1

            compute_objectives(strategy, window_metrics, self.parsimony_coeff)

        return new_evals

    def _hybrid_select(self, parents: List[Strategy]) -> Strategy:
        """
        Hybrid selection from old island model:
        - 60% binary tournament (exploitation)
        - 30% lexicase (diversity — selects specialists in different metrics)
        - 10% random (exploration)
        """
        r = random.random()
        if r < 0.60:
            # Binary tournament
            a = random.choice(parents)
            b = random.choice(parents)
            return a if a.objectives[0] > b.objectives[0] else b
        elif r < 0.90:
            # Lexicase — promotes diverse specialists
            return lexicase_select(parents)
        else:
            # Random — pure exploration
            return random.choice(parents)

    def _breed_within_niche(self, parents: List[Strategy], n: int,
                            target_nodes: int) -> List[Strategy]:
        """
        Breed n offspring within a complexity niche.

        Uses hybrid selection (tournament + lexicase + random).
        Structure codon is protected — offspring maintain the same complexity.
        """
        offspring = []
        max_attempts = n * 5
        attempts = 0

        # Filter parents to matching complexity (with fallback)
        matching = [p for p in parents if _get_complexity(p) == target_nodes]
        if len(matching) < 2:
            matching = parents  # fallback to all parents

        while len(offspring) < n and attempts < max_attempts:
            attempts += 1

            if random.random() < self.crossover_rate and len(matching) >= 2:
                p1 = self._hybrid_select(matching)
                p2 = self._hybrid_select(matching)
                active = max(p1.codons_used, p2.codons_used)
                c1_genome, c2_genome = crossover(p1.genome, p2.genome,
                                                  active_codons=active)
            else:
                p1 = self._hybrid_select(matching)
                c1_genome = p1.genome[:]
                c2_genome = None
                active = p1.codons_used

            # Mutate
            c1_genome = mutate(c1_genome, self.mutation_rate,
                               self.generation, self.max_generations,
                               active_codons=active)
            # PROTECT structure codon
            _fix_structure_codon(c1_genome, target_nodes)

            s1 = decode(c1_genome)
            if s1 is not None and s1.n_nodes == target_nodes:
                offspring.append(s1)

            if c2_genome is not None and len(offspring) < n:
                c2_genome = mutate(c2_genome, self.mutation_rate,
                                   self.generation, self.max_generations,
                                   active_codons=active)
                _fix_structure_codon(c2_genome, target_nodes)
                s2 = decode(c2_genome)
                if s2 is not None and s2.n_nodes == target_nodes:
                    offspring.append(s2)

        return offspring[:n]

    def _select_within_niche(self, strategies: List[Strategy], n: int) -> List[Strategy]:
        """Run NSGA-II selection within a single complexity niche."""
        if not strategies:
            return []

        if len(strategies) <= n:
            return strategies[:]

        # Non-dominated sort within niche
        fronts = non_dominated_sort(strategies)

        selected = []
        phenotype_counts = {}

        for front in fronts:
            if len(selected) >= n:
                break

            compute_crowding_distance(front)
            sorted_front = sorted(front, key=lambda s: s.crowding_distance, reverse=True)

            for s in sorted_front:
                if len(selected) >= n:
                    break
                key = s.expression_raw
                count = phenotype_counts.get(key, 0)
                if count < self.max_phenotype_copies:
                    selected.append(s)
                    phenotype_counts[key] = count + 1

        return selected

    def _compute_regime_sortinos(self, strategy: Strategy) -> Optional[Dict[str, float]]:
        """Estimate regime performance from per-window metrics."""
        if not strategy.window_metrics:
            return None

        regime_sortinos = {'bull': [], 'bear': [], 'sideways': []}
        for m in strategy.window_metrics:
            ret = m.get('return_pct', 0.0)
            sortino = m.get('sortino', 0.0)
            if ret > 3.0:
                regime_sortinos['bull'].append(sortino)
            elif ret < -3.0:
                regime_sortinos['bear'].append(sortino)
            else:
                regime_sortinos['sideways'].append(sortino)

        result = {}
        for regime, vals in regime_sortinos.items():
            result[regime] = sum(vals) / len(vals) if vals else 0.0

        return result if any(v != 0.0 for v in result.values()) else None

    def _migrate_between_niches(self, population: List[Strategy],
                                windows: List[Tuple]):
        """
        Migrate top strategies between complexity niches.
        Re-decodes genomes with different structure codon to create cross-niche migrants.
        """
        # Build index-based niche mapping
        niche_indices = {1: [], 2: [], 3: []}
        for i, s in enumerate(population):
            niche_indices[_get_complexity(s)].append(i)

        n_migrate = self.migration_size
        replaced = 0

        for src_nodes in [1, 2, 3]:
            src_idxs = niche_indices.get(src_nodes, [])
            if not src_idxs:
                continue

            # Get top by composite fitness
            src_sorted = sorted(src_idxs,
                                key=lambda i: population[i].objectives[0]
                                if hasattr(population[i], 'objectives') else -999,
                                reverse=True)
            top_idxs = src_sorted[:n_migrate]

            for dst_nodes in [1, 2, 3]:
                if dst_nodes == src_nodes:
                    continue
                dst_idxs = niche_indices.get(dst_nodes, [])
                if not dst_idxs:
                    continue

                # Find worst in destination niche
                dst_sorted = sorted(dst_idxs,
                                    key=lambda i: population[i].objectives[0]
                                    if hasattr(population[i], 'objectives') else -999)

                for k, src_i in enumerate(top_idxs):
                    if k >= len(dst_sorted):
                        break
                    s = population[src_i]
                    new_genome = s.genome[:]
                    _fix_structure_codon(new_genome, dst_nodes)
                    new_genome = mutate(new_genome, self.mutation_rate * 0.5,
                                        self.generation, self.max_generations,
                                        active_codons=s.codons_used)
                    _fix_structure_codon(new_genome, dst_nodes)
                    migrant = decode(new_genome)
                    if migrant is not None and migrant.n_nodes == dst_nodes:
                        population[dst_sorted[k]] = migrant
                        replaced += 1

        if replaced:
            logger.debug(f"Migration: replaced {replaced} strategies across niches")

    def _inject_from_archive(self, population: List[Strategy],
                              windows: List[Tuple]):
        """Inject archive elite strategies into each complexity niche."""
        archive_strats = self.archive.sample_for_reproduction(
            self.migration_size * 3
        )
        if not archive_strats:
            return

        injected = 0
        for s in archive_strats:
            target = _get_complexity(s)
            # Find worst in that niche
            niche = [(i, p) for i, p in enumerate(population)
                     if _get_complexity(p) == target]
            if not niche:
                continue
            worst_idx = min(niche, key=lambda x: x[1].objectives[0]
                            if hasattr(x[1], 'objectives') and
                            x[1].objectives != (-999.0, -999.0) else -9999)[0]
            population[worst_idx] = s
            injected += 1

        if injected:
            logger.debug(f"Injected {injected} archive strategies")

    def step(self) -> GenerationStats:
        """Execute one complexity-niched NSGA-II generation."""
        t0 = time.time()
        pop_size = len(self.population)

        # 1. Sample windows with rotation
        if self.use_fixed_windows and self._fixed_windows is not None:
            windows = self._fixed_windows
        else:
            windows = sample_windows_with_rotation(
                self.data, n_windows=self.n_windows,
                window_bars=self.window_bars,
                previous_windows=self._prev_windows,
                keep_ratio=self.keep_ratio,
            )
            self._prev_windows = windows
            if self.use_fixed_windows:
                self._fixed_windows = windows

        if not self.use_fixed_windows:
            active_ids = {wid for _, wid in windows}
            self.cache.evict_except(active_ids)
            stale_keys = [k for k in self._tf_data_cache if k not in active_ids]
            for k in stale_keys:
                del self._tf_data_cache[k]

        self.cache.reset_counters()

        # 2. Group current population by complexity
        niches = {1: [], 2: [], 3: []}
        for s in self.population:
            niches[_get_complexity(s)].append(s)

        # 3. Breed offspring WITHIN each niche
        all_offspring = []
        for target_nodes, proportion in NICHE_PROPORTIONS.items():
            niche_size = int(pop_size * proportion)
            niche_parents = niches.get(target_nodes, [])

            # Also pull parents from archive
            n_archive = max(1, int(niche_size * self.archive_parent_pct))
            archive_parents = self.archive.sample_for_reproduction(n_archive)
            # Filter archive parents to matching complexity
            archive_matching = [p for p in archive_parents
                                if _get_complexity(p) == target_nodes]
            niche_parents = niche_parents + archive_matching

            if not niche_parents:
                # No parents — generate fresh
                fresh = self._generate_with_complexity(niche_size, target_nodes)
                all_offspring.extend(fresh)
                continue

            # Select parents within niche via tournament
            tournament_parents = select_parents(niche_parents,
                                                min(niche_size, len(niche_parents)))
            offspring = self._breed_within_niche(tournament_parents, niche_size,
                                                target_nodes)
            all_offspring.extend(offspring)

        # 4. Evaluate offspring
        new_evals = self._evaluate_population(all_offspring, windows)
        self.total_evaluations += new_evals

        # Re-evaluate parents on current windows
        parent_evals = self._evaluate_population(self.population, windows)
        self.total_evaluations += parent_evals

        # 5. Select survivors WITHIN each niche (mu+lambda per niche)
        new_pop = []
        niche_stats = {}

        for target_nodes, proportion in NICHE_PROPORTIONS.items():
            niche_target = int(pop_size * proportion)

            # Combine parents + offspring for this niche
            niche_combined = [s for s in self.population
                              if _get_complexity(s) == target_nodes]
            niche_combined += [s for s in all_offspring
                               if _get_complexity(s) == target_nodes]

            # Select within niche
            selected = self._select_within_niche(niche_combined, niche_target)

            # If not enough, generate fresh immigrants for this niche
            if len(selected) < niche_target:
                n_fresh = niche_target - len(selected)
                fresh = self._generate_with_complexity(n_fresh, target_nodes)
                if fresh:
                    fresh_evals = self._evaluate_population(fresh, windows)
                    self.total_evaluations += fresh_evals
                    selected.extend(fresh)

            niche_feasible = sum(1 for s in selected if s.constraint_violation <= 0)
            niche_stats[target_nodes] = (len(selected), niche_feasible)

            new_pop.extend(selected[:niche_target])

        # 6. Update MAP-Elites archive
        for s in new_pop:
            if s.constraint_violation <= 0 and s.metrics:
                trades_per_month = s.n_trades / max(len(windows), 1) / 2.0
                regime_sortinos = self._compute_regime_sortinos(s)
                self.archive.try_add(s, trades_per_month, regime_sortinos=regime_sortinos)

        # 7. Migration between niches (from old island model)
        if (self.generation + 1) % self.migration_interval == 0:
            self._migrate_between_niches(new_pop, windows)

        # 8. Archive injection (from old island model)
        if (self.generation + 1) % self.archive_inject_interval == 0:
            self._inject_from_archive(new_pop, windows)

        # 7. Stats
        cache_stats = self.cache.stats()
        all_feasible = [s for s in new_pop if s.constraint_violation <= 0]
        dist = Counter(_get_complexity(s) for s in new_pop)

        if all_feasible:
            best = max(all_feasible, key=lambda s: s.objectives[0])
            best_consistency = max(all_feasible, key=lambda s: s.objectives[1])
            composites = [s.objectives[0] for s in all_feasible]
            median_cf = sorted(composites)[len(composites) // 2]
        else:
            best = new_pop[0] if new_pop else None
            best_consistency = best
            median_cf = -999.0

        elapsed = time.time() - t0
        stats = GenerationStats(
            generation=self.generation,
            best_sortino=best.objectives[0] if best else -999.0,
            best_return=best_consistency.objectives[1] if best_consistency else -999.0,
            median_sortino=median_cf,
            front1_size=sum(1 for s in all_feasible if s.objectives[0] > median_cf),
            valid_count=len(all_feasible),
            total_count=len(new_pop),
            cache_hit_rate=cache_stats['hit_rate'],
            archive_coverage=self.archive.coverage,
            elapsed_seconds=elapsed,
            complexity_dist=dict(dist),
        )
        self.history.append(stats)

        n_unique = len(set(s.expression_raw for s in new_pop))

        # Per-niche best
        niche_bests = []
        for nodes in [1, 2, 3]:
            niche_f = [s for s in all_feasible if _get_complexity(s) == nodes]
            if niche_f:
                b = max(niche_f, key=lambda s: s.objectives[0])
                niche_bests.append(f"{nodes}n:{b.objectives[0]:+.1f}")
            else:
                niche_bests.append(f"{nodes}n:---")

        logger.info(
            f"Gen {self.generation:3d} | "
            f"1n={dist.get(1,0)} 2n={dist.get(2,0)} 3n={dist.get(3,0)} "
            f"feas={stats.valid_count}/{stats.total_count} uniq={n_unique} | "
            f"best=[{' '.join(niche_bests)}] | "
            f"cache={stats.cache_hit_rate:.0%} arch={stats.archive_coverage:.0%} | "
            f"{elapsed:.1f}s"
        )

        self.population = new_pop
        self.generation += 1
        return stats

    def run(self, n_generations: int, patience: int = 30) -> EvolutionResult:
        """Run full complexity-niched NSGA-II evolution."""
        best_ever = -999.0
        stagnation = 0

        logger.info(f"Starting NSGA-II v4 (complexity-niched): pop={len(self.population)}, "
                    f"max_gen={n_generations}, patience={patience}, "
                    f"rotation={'OFF' if self.use_fixed_windows else 'ON'}")

        for gen in range(n_generations):
            stats = self.step()

            if stats.best_sortino > best_ever + 0.01:
                best_ever = stats.best_sortino
                stagnation = 0
            else:
                stagnation += 1

            # Diversity-reactive: boost mutation when diversity drops
            n_unique = len(set(s.expression_raw for s in self.population))
            diversity_ratio = n_unique / max(len(self.population), 1)
            if diversity_ratio < 0.40:
                self.mutation_rate = min(self.mutation_rate * 1.2, 0.30)
                logger.info(f"  Low diversity ({diversity_ratio:.0%}) — "
                            f"boosted mutation to {self.mutation_rate:.2f}")
            elif self.mutation_rate > self.config.get('evolution', {}).get('mutation_rate', 0.15):
                self.mutation_rate = self.config.get('evolution', {}).get('mutation_rate', 0.15)

            # Stagnation-triggered immigration (from old island model)
            if stagnation > 0 and stagnation % 10 == 0:
                n_immigrants = 10  # per niche
                for target_nodes in [1, 2, 3]:
                    fresh = self._generate_with_complexity(n_immigrants, target_nodes)
                    # Replace worst in that niche
                    niche_idxs = [i for i, s in enumerate(self.population)
                                  if _get_complexity(s) == target_nodes]
                    if niche_idxs and fresh:
                        worst_idxs = sorted(niche_idxs,
                                            key=lambda i: self.population[i].objectives[0]
                                            if hasattr(self.population[i], 'objectives')
                                            else -999)[:len(fresh)]
                        for idx, f in zip(worst_idxs, fresh):
                            self.population[idx] = f
                logger.info(f"  Stagnation immigration: injected {n_immigrants*3} fresh strategies")

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (stagnation={patience})")
                break

        # Collect Pareto front across ALL complexity levels
        all_feasible = [s for s in self.population if s.constraint_violation <= 0]
        if all_feasible:
            fronts = non_dominated_sort(all_feasible)
            pareto = fronts[0] if fronts else []
        else:
            pareto = []

        # Also add best from each niche (even if not on global Pareto front)
        pareto_set = set(id(s) for s in pareto)
        for nodes in [1, 2, 3]:
            niche_f = [s for s in all_feasible if _get_complexity(s) == nodes]
            if niche_f:
                best_in_niche = sorted(niche_f, key=lambda s: s.objectives[0], reverse=True)
                for s in best_in_niche[:3]:  # Top 3 from each niche
                    if id(s) not in pareto_set:
                        pareto.append(s)
                        pareto_set.add(id(s))

        pareto_sorted = sorted(pareto, key=lambda s: s.objectives[0], reverse=True)

        # Log summary per niche
        for nodes in [1, 2, 3]:
            niche_p = [s for s in pareto_sorted if _get_complexity(s) == nodes]
            if niche_p:
                logger.info(f"  Niche {nodes}-node: {len(niche_p)} Pareto strategies")

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
