"""
Strategy Phenotype — decoded representation of a GE genome.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Condition:
    """A single condition in a strategy's entry rule."""
    left: str           # e.g. "RSI(close, 14)" or "close"
    comparator: str     # ">", "<", "CROSSES_ABOVE", "CROSSES_BELOW"
    right: str          # e.g. "30" or "SMA(close, 50)"

    def __str__(self):
        return f"{self.left} {self.comparator} {self.right}"


@dataclass
class Strategy:
    """
    Phenotype decoded from a GE genome.

    Created by grammar.mapper.decode(). Evaluated by strategy.vectorized_eval.
    """
    genome: List[int]
    direction: str                              # "LONG" or "SHORT"
    conditions: List[Condition]                  # Parsed entry conditions
    logic: str                                   # Raw logic string: "c0 AND c1", "(c0 AND c1) OR c2", etc.
    tp_atr_mult: float                           # Take profit in ATR multiples (0 = no fixed TP)
    sl_atr_mult: float                           # Stop loss in ATR multiples
    trail_atr_mult: float = 0.0                  # Trailing stop in ATR multiples (0 = no trail)
    expression_raw: str = ""                     # Full decoded expression string
    n_nodes: int = 0                             # Complexity (number of conditions)
    codons_used: int = 0                         # How many codons were consumed
    wrapping_count: int = 0                      # How many times genome was wrapped

    # Filled post-evaluation
    fitness: Tuple[float, float] = (-999.0, -999.0)  # (sortino, calmar)

    # NSGA-II fields (filled during evaluation/selection)
    objectives: Tuple[float, float] = (-999.0, -999.0)  # (median_sortino, median_return)
    stability: float = -999.0                             # -std(sortino across windows)
    constraint_violation: float = 0.0                      # 0.0 = feasible, >0 = infeasible
    rank: int = 999                                        # Pareto front rank (1 = best)
    crowding_distance: float = 0.0                          # NSGA-II crowding distance
    window_metrics: Optional[List[Dict]] = None            # Per-window metrics for analysis

    metrics: Optional[Dict] = None
    n_trades: int = 0

    def __str__(self):
        conds = " ; ".join(str(c) for c in self.conditions)
        trail = f" TRAIL={self.trail_atr_mult}" if self.trail_atr_mult > 0 else ""
        tp = f"TP={self.tp_atr_mult} " if self.tp_atr_mult > 0 else ""
        return f"{self.direction} | {conds} | {tp}SL={self.sl_atr_mult}{trail}"

    def to_readable(self) -> str:
        trail = f", TRAIL={self.trail_atr_mult}xATR" if self.trail_atr_mult > 0 else ""
        tp = f"TP={self.tp_atr_mult}xATR, " if self.tp_atr_mult > 0 else ""
        return f"{self.direction} when {self.logic} ({tp}SL={self.sl_atr_mult}xATR{trail})"

    def to_dict(self) -> dict:
        return {
            'genome': self.genome,
            'direction': self.direction,
            'conditions': [str(c) for c in self.conditions],
            'logic': self.logic,
            'tp_atr_mult': self.tp_atr_mult,
            'sl_atr_mult': self.sl_atr_mult,
            'trail_atr_mult': self.trail_atr_mult,
            'n_nodes': self.n_nodes,
            'codons_used': self.codons_used,
            'wrapping_count': self.wrapping_count,
            'fitness': list(self.fitness),
            'objectives': list(self.objectives),
            'rank': self.rank,
            'stability': self.stability,
            'n_trades': self.n_trades,
            'expression_raw': self.expression_raw,
            'metrics': self.metrics,
        }
