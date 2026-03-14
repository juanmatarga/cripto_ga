"""
MAP-Elites Archive — Quality-Diversity for strategy exploration.

Maintains a grid of niches defined by behavioral dimensions.
Each cell holds the best strategy found for that niche.

Grid dimensions:
- Frequency: trades/month -> low / medium / high
- Complexity: n_nodes -> 1 / 2 / 3 / 4 / 5+
- Regime: best-performing regime -> bull / bear / sideways
"""

import random
import logging
from typing import Dict, List, Optional, Tuple

from strategy.phenotype import Strategy

logger = logging.getLogger(__name__)

# Bin definitions
FREQ_BINS = ['low', 'medium', 'high']       # 3 bins
COMPLEXITY_BINS = [1, 2, 3, 4, 5]            # 5 bins (5 means 5+)
REGIME_BINS = ['bull', 'bear', 'sideways']   # 3 bins

# Total cells: 3 * 5 * 3 = 45
TOTAL_CELLS = len(FREQ_BINS) * len(COMPLEXITY_BINS) * len(REGIME_BINS)


def _freq_bin(trades_per_month: float) -> str:
    """Classify trade frequency."""
    if trades_per_month <= 5:
        return 'low'
    elif trades_per_month <= 20:
        return 'medium'
    else:
        return 'high'


def _complexity_bin(n_nodes: int) -> int:
    """Classify complexity."""
    return min(n_nodes, 5) if n_nodes >= 1 else 1


def _regime_bin(regime_sortinos: Optional[Dict[str, float]]) -> str:
    """
    Determine which regime the strategy performs best in.
    Falls back to 'sideways' if no regime data available.
    """
    if not regime_sortinos:
        return 'sideways'
    return max(regime_sortinos, key=regime_sortinos.get)


class MAPElitesArchive:
    """
    MAP-Elites grid archive.

    Stores one strategy per niche cell. A new strategy replaces the
    resident only if it has higher fitness.
    """

    def __init__(self):
        self.grid: Dict[Tuple[str, int, str], Strategy] = {}

    def try_add(self, strategy: Strategy,
                trades_per_month: float = 0.0,
                regime_sortinos: Optional[Dict[str, float]] = None) -> bool:
        """
        Attempt to add a strategy to the archive.

        Args:
            strategy: Strategy with valid fitness
            trades_per_month: Trading frequency metric
            regime_sortinos: {'bull': sortino, 'bear': sortino, 'sideways': sortino}

        Returns:
            True if strategy was added (cell was empty or strategy beat resident)
        """
        if strategy.objectives[0] <= -999.0:
            return False

        cell = (
            _freq_bin(trades_per_month),
            _complexity_bin(strategy.n_nodes),
            _regime_bin(regime_sortinos),
        )

        if cell not in self.grid or strategy.objectives[0] > self.grid[cell].objectives[0]:
            self.grid[cell] = strategy
            return True
        return False

    def sample_for_reproduction(self, n: int) -> List[Strategy]:
        """Sample n strategies from occupied cells (with replacement)."""
        occupied = list(self.grid.values())
        if not occupied:
            return []
        return random.choices(occupied, k=min(n, len(occupied) * 3))

    @property
    def n_occupied(self) -> int:
        return len(self.grid)

    @property
    def coverage(self) -> float:
        """Fraction of total cells that are occupied."""
        return self.n_occupied / TOTAL_CELLS

    def best_per_regime(self) -> Dict[str, Optional[Strategy]]:
        """Return best strategy for each regime."""
        result = {}
        for regime in REGIME_BINS:
            cells = [(k, v) for k, v in self.grid.items() if k[2] == regime]
            if cells:
                result[regime] = max(cells, key=lambda x: x[1].objectives[0])[1]
            else:
                result[regime] = None
        return result

    def get_all_strategies(self) -> List[Strategy]:
        """Return all strategies in the archive."""
        return list(self.grid.values())

    def summary(self) -> Dict:
        """Archive stats for logging."""
        if not self.grid:
            return {'n_occupied': 0, 'coverage': 0.0, 'best_fitness': -999.0}

        fitnesses = [s.objectives[0] for s in self.grid.values()]
        return {
            'n_occupied': self.n_occupied,
            'coverage': self.coverage,
            'total_cells': TOTAL_CELLS,
            'best_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'regimes': {
                regime: sum(1 for k in self.grid if k[2] == regime)
                for regime in REGIME_BINS
            },
        }
