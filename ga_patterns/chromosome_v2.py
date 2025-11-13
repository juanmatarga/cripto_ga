"""
Pattern Chromosome V2 - Building Blocks Architecture

This module defines the new pattern representation as a combination of
pre-built modules, replacing the old tree-based Pattern class.

Key improvements:
    - Human-readable: Patterns are lists of named modules
    - Interpretable: Each module has clear trading meaning
    - Evolvable: Operators preserve semantic validity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

@dataclass
class PatternChromosome:
    """
    Pattern represented as combination of building block modules.

    This replaces the old Pattern class with PredicateNode/LogicalNode trees.

    Attributes:
        direction: 'LONG' or 'SHORT'
        modules: List of module names, e.g., ['momentum_up_2bar', 'volume_spike']
        logic: How to combine modules - 'AND', 'OR', '2of3', '3of4'
        window: Lookback window size (2-20)
        generation_created: Generation when pattern was created
        fitness: Combined fitness score
        fitness_long: Fitness when evaluated as LONG
        fitness_short: Fitness when evaluated as SHORT
        metrics: Dict of performance metrics (sharpe, cagr, upi, max_dd, etc.)
        n_trades: Total number of trades across all evaluation windows

    Example:
        >>> chrom = PatternChromosome(
        ...     direction='LONG',
        ...     modules=['momentum_up_2bar', 'volume_spike_short', 'large_body'],
        ...     logic='AND',
        ...     window=5
        ... )
        >>> print(chrom.to_readable())
        LONG (w=5): AND(momentum_up_2bar, volume_spike_short, large_body)
    """
    direction: str
    modules: List[str]
    logic: str = 'AND'
    window: int = 5
    generation_created: int = 0
    fitness: float = -999.0
    fitness_long: float = -999.0
    fitness_short: float = -999.0
    metrics: Dict = field(default_factory=dict)  # Performance metrics
    n_trades: int = 0  # Total trades across windows

    def to_readable(self) -> str:
        """
        Convert to human-readable format for logging and snapshots.

        Returns:
            String format: "LONG (w=5): AND(momentum_up_2bar, volume_spike, large_body)"

        Example:
            >>> chrom = PatternChromosome(
            ...     direction='LONG',
            ...     modules=['momentum_up_2bar', 'volume_spike_short'],
            ...     logic='OR',
            ...     window=10
            ... )
            >>> chrom.to_readable()
            'LONG (w=10): OR(momentum_up_2bar, volume_spike_short)'
        """
        modules_str = ", ".join(self.modules)
        return f"{self.direction} (w={self.window}): {self.logic}({modules_str})"

    def to_expression(self) -> str:
        """
        Convert modules to full executable expression.

        Returns:
            String with all module expressions combined

        Example:
            Input:
                modules = ['momentum_up_2bar', 'volume_spike_short']
                logic = 'AND'

            Process:
                momentum_up_2bar = "C[0] > C[1] AND C[1] > C[2]"
                volume_spike_short = "V[0] > V[3] AND V[0] > V[5]"

            Output:
                "(C[0] > C[1] AND C[1] > C[2]) AND (V[0] > V[3] AND V[0] > V[5])"
        """
        from ga_patterns.building_blocks import get_module_info

        # Get expressions for each module
        module_expressions = []
        for module_name in self.modules:
            info = get_module_info(module_name)
            if info:
                module_expressions.append(f"({info['expression']})")
            else:
                logger.warning(f"Unknown module in to_expression: {module_name}")

        if not module_expressions:
            logger.error("No valid module expressions found")
            return "False"  # Invalid pattern

        # Combine based on logic
        if self.logic == 'AND':
            return " AND ".join(module_expressions)

        elif self.logic == 'OR':
            return " OR ".join(module_expressions)

        elif self.logic == '2of3':
            # At least 2 of N modules must be True
            # Convert to: (A AND B) OR (A AND C) OR (B AND C)
            if len(module_expressions) < 2:
                return " AND ".join(module_expressions)
            elif len(module_expressions) == 2:
                return " AND ".join(module_expressions)
            else:
                # Generate all 2-combinations
                combos = list(combinations(module_expressions, 2))
                combo_strs = [f"({a} AND {b})" for a, b in combos]
                result = " OR ".join(combo_strs)
                logger.debug(f"2of3 with {len(module_expressions)} modules → {len(combos)} combinations")
                return result

        elif self.logic == '3of4':
            # At least 3 of N modules must be True
            if len(module_expressions) < 3:
                return " AND ".join(module_expressions)
            else:
                combos = list(combinations(module_expressions, 3))
                combo_strs = [f"({a} AND {b} AND {c})" for a, b, c in combos]
                result = " OR ".join(combo_strs)
                logger.debug(f"3of4 with {len(module_expressions)} modules → {len(combos)} combinations")
                return result

        else:
            # Default to AND
            logger.warning(f"Unknown logic '{self.logic}', defaulting to AND")
            return " AND ".join(module_expressions)

    def __str__(self) -> str:
        """String representation uses readable format."""
        return self.to_readable()

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"PatternChromosome(direction={self.direction}, "
                f"modules={self.modules}, logic={self.logic}, "
                f"window={self.window}, fitness={self.fitness:.4f})")

    def to_dict(self) -> dict:
        """
        Convert chromosome to dictionary for serialization.

        Used by evolution tracker for saving snapshots.

        Returns:
            dict: All chromosome attributes

        Example:
            >>> chrom = PatternChromosome(
            ...     direction='LONG',
            ...     modules=['momentum_up_2bar'],
            ...     logic='AND'
            ... )
            >>> d = chrom.to_dict()
            >>> d['direction']
            'LONG'
        """
        return {
            'direction': self.direction,
            'modules': self.modules,
            'logic': self.logic,
            'window': self.window,
            'generation_created': self.generation_created,
            'fitness': self.fitness,
            'fitness_long': self.fitness_long,
            'fitness_short': self.fitness_short,
            'metrics': self.metrics,  # SPRINT 11: Performance metrics
            'n_trades': self.n_trades,  # SPRINT 11: Trade count
            'expression_readable': self.to_readable(),
            'expression_full': self.to_expression()
        }


def validate_chromosome(chrom: PatternChromosome) -> bool:
    """
    Validate that chromosome is well-formed.

    Args:
        chrom: PatternChromosome to validate

    Returns:
        bool: True if valid, False otherwise

    Checks:
        1. direction in ['LONG', 'SHORT']
        2. All modules exist in ALL_MODULES
        3. logic in ['AND', 'OR', '2of3', '3of4']
        4. Sufficient modules for logic:
           - '2of3' requires len(modules) >= 2
           - '3of4' requires len(modules) >= 3
        5. window in range [2, 20]

    Example:
        >>> chrom = PatternChromosome(
        ...     direction='LONG',
        ...     modules=['momentum_up_2bar', 'volume_spike_short'],
        ...     logic='AND',
        ...     window=5
        ... )
        >>> validate_chromosome(chrom)
        True

        >>> bad_chrom = PatternChromosome(
        ...     direction='INVALID',
        ...     modules=['momentum_up_2bar'],
        ...     logic='AND',
        ...     window=5
        ... )
        >>> validate_chromosome(bad_chrom)
        False
    """
    from ga_patterns.building_blocks import ALL_MODULES

    # Check direction
    if chrom.direction not in ['LONG', 'SHORT']:
        logger.error(f"Invalid direction: {chrom.direction}")
        return False

    # Check modules exist
    if not chrom.modules:
        logger.error("Empty modules list")
        return False

    for module in chrom.modules:
        if module not in ALL_MODULES:
            logger.error(f"Unknown module: {module}")
            return False

    # Check logic
    if chrom.logic not in ['AND', 'OR', '2of3', '3of4']:
        logger.error(f"Invalid logic: {chrom.logic}")
        return False

    # Check sufficient modules for logic
    if chrom.logic == '2of3' and len(chrom.modules) < 2:
        logger.error("'2of3' requires at least 2 modules")
        return False

    if chrom.logic == '3of4' and len(chrom.modules) < 3:
        logger.error("'3of4' requires at least 3 modules")
        return False

    # Check window range
    if not (2 <= chrom.window <= 20):
        logger.error(f"Invalid window: {chrom.window} (must be 2-20)")
        return False

    logger.debug(f"Chromosome validated successfully: {chrom.to_readable()}")
    return True


def get_chromosome_complexity(chrom: PatternChromosome) -> str:
    """
    Determine the complexity level of a chromosome based on its modules.

    Args:
        chrom: PatternChromosome to analyze

    Returns:
        str: 'simple', 'medium', or 'advanced'

    Logic:
        - If any module is 'advanced' → 'advanced'
        - Else if any module is 'medium' → 'medium'
        - Else → 'simple'

    Example:
        >>> chrom = PatternChromosome(
        ...     direction='LONG',
        ...     modules=['momentum_up_2bar', 'volume_spike_short'],
        ...     logic='AND'
        ... )
        >>> get_chromosome_complexity(chrom)
        'simple'
    """
    from ga_patterns.building_blocks import get_module_info

    complexities = []
    for module_name in chrom.modules:
        info = get_module_info(module_name)
        if info:
            complexities.append(info['complexity'])

    if 'advanced' in complexities:
        return 'advanced'
    elif 'medium' in complexities:
        return 'medium'
    else:
        return 'simple'


def get_chromosome_min_generation(chrom: PatternChromosome) -> int:
    """
    Get the minimum generation at which this chromosome could be created.

    Args:
        chrom: PatternChromosome to analyze

    Returns:
        int: Minimum generation (0, 30, or 80)

    Logic:
        Return max(min_generation) across all modules

    Example:
        >>> # Chromosome with only BASE modules
        >>> chrom1 = PatternChromosome(
        ...     direction='LONG',
        ...     modules=['momentum_up_2bar', 'volume_spike_short'],
        ...     logic='AND'
        ... )
        >>> get_chromosome_min_generation(chrom1)
        0

        >>> # Chromosome with INDICATOR module
        >>> chrom2 = PatternChromosome(
        ...     direction='LONG',
        ...     modules=['momentum_up_2bar', 'rsi_oversold_40'],
        ...     logic='AND'
        ... )
        >>> get_chromosome_min_generation(chrom2)
        30
    """
    from ga_patterns.building_blocks import get_module_info

    min_gens = []
    for module_name in chrom.modules:
        info = get_module_info(module_name)
        if info:
            min_gens.append(info['min_generation'])

    return max(min_gens) if min_gens else 0


if __name__ == '__main__':
    # Test PatternChromosome
    print("="*70)
    print("TESTING PatternChromosome")
    print("="*70)

    # Test 1: Simple AND pattern
    chrom1 = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short', 'large_body'],
        logic='AND',
        window=5
    )
    print(f"\nTest 1: Simple AND pattern")
    print(f"  Readable: {chrom1.to_readable()}")
    print(f"  Valid: {validate_chromosome(chrom1)}")
    print(f"  Complexity: {get_chromosome_complexity(chrom1)}")
    print(f"  Min generation: {get_chromosome_min_generation(chrom1)}")
    print(f"  Expression: {chrom1.to_expression()[:100]}...")

    # Test 2: OR pattern with indicators
    chrom2 = PatternChromosome(
        direction='SHORT',
        modules=['momentum_down_3bar', 'rsi_overbought_70'],
        logic='OR',
        window=10
    )
    print(f"\nTest 2: OR pattern with indicators")
    print(f"  Readable: {chrom2.to_readable()}")
    print(f"  Valid: {validate_chromosome(chrom2)}")
    print(f"  Complexity: {get_chromosome_complexity(chrom2)}")
    print(f"  Min generation: {get_chromosome_min_generation(chrom2)}")

    # Test 3: 2of3 pattern
    chrom3 = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar', 'volume_spike_short', 'close_near_high'],
        logic='2of3',
        window=7
    )
    print(f"\nTest 3: 2of3 pattern")
    print(f"  Readable: {chrom3.to_readable()}")
    print(f"  Valid: {validate_chromosome(chrom3)}")
    print(f"  Expression combinations: ", end="")
    expr = chrom3.to_expression()
    print(f"{expr.count(' OR ') + 1} total")

    # Test 4: Invalid pattern
    chrom4 = PatternChromosome(
        direction='INVALID_DIR',
        modules=['momentum_up_2bar'],
        logic='AND',
        window=5
    )
    print(f"\nTest 4: Invalid pattern")
    print(f"  Readable: {chrom4.to_readable()}")
    print(f"  Valid: {validate_chromosome(chrom4)}")

    # Test 5: Advanced pattern
    chrom5 = PatternChromosome(
        direction='LONG',
        modules=['macd_bullish_cross', 'bb_lower_touch', 'rsi_oversold_30'],
        logic='AND',
        window=15
    )
    print(f"\nTest 5: Advanced pattern")
    print(f"  Readable: {chrom5.to_readable()}")
    print(f"  Valid: {validate_chromosome(chrom5)}")
    print(f"  Complexity: {get_chromosome_complexity(chrom5)}")
    print(f"  Min generation: {get_chromosome_min_generation(chrom5)}")

    print(f"\n{'='*70}\n")
