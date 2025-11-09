"""
Chromosome Representation - Expression Trees with Direct Comparisons
"""

from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredicateNode:
    """
    Nodo predicado con soporte para comparaciones directas.

    Puede ser:
    1. Comparación con threshold: close[0] > 50000
    2. Comparación con otra barra: close[0] > close[1]
    """
    predicate_name: str
    operator: str  # '>', '>=', '<='

    # Para comparación con threshold
    threshold: Optional[float] = None
    bar_offset: int = 0
    params: Optional[Dict] = None

    # Para comparación con otra barra (ej: C[0] vs C[1])
    compare_with_bar: Optional[int] = None  # Si es != None, comparar con esta barra

    def evaluate(self, data: pd.DataFrame) -> bool:
        """Evalúa predicado."""
        from ga_patterns.grammar import PREDICATE_REGISTRY, ComparisonOperator

        predicate_info = PREDICATE_REGISTRY.get(self.predicate_name)
        if not predicate_info:
            logger.error(f"Predicate not found: {self.predicate_name}")
            return False

        predicate_func = predicate_info['func']

        try:
            # Calcular valor en bar_offset
            value1 = predicate_func(data, bar_offset=self.bar_offset, **(self.params or {}))

            # Determinar valor de comparación
            if self.compare_with_bar is not None:
                # Comparación con otra barra
                value2 = predicate_func(data, bar_offset=self.compare_with_bar, **(self.params or {}))
            else:
                # Comparación con threshold
                value2 = self.threshold

            # Evaluar
            result = ComparisonOperator.evaluate(self.operator, value1, value2)
            return result

        except Exception as e:
            logger.error(f"Error evaluating {self.predicate_name}: {e}")
            return False

    def __repr__(self):
        if self.compare_with_bar is not None:
            # Comparación directa: close[0] > close[1]
            return f"{self.predicate_name}[{self.bar_offset}] {self.operator} {self.predicate_name}[{self.compare_with_bar}]"
        else:
            # Comparación con threshold: price_change_pct[0] > 0.02
            return f"{self.predicate_name}[{self.bar_offset}] {self.operator} {self.threshold:.3f}"


@dataclass
class LogicalNode:
    """Nodo lógico."""
    operator: str  # 'AND', 'OR', 'NOT'
    children: List[Union['LogicalNode', PredicateNode]] = field(default_factory=list)

    def evaluate(self, data: pd.DataFrame) -> bool:
        """Evalúa subárbol lógico."""
        from ga_patterns.grammar import LogicalOperator

        child_results = [child.evaluate(data) for child in self.children]

        if self.operator == 'AND':
            return LogicalOperator.AND(*child_results)
        elif self.operator == 'OR':
            return LogicalOperator.OR(*child_results)
        elif self.operator == 'NOT':
            if len(child_results) != 1:
                return False
            return LogicalOperator.NOT(child_results[0])
        else:
            return False

    def __repr__(self):
        children_str = ', '.join([str(c) for c in self.children])
        return f"{self.operator}({children_str})"


@dataclass
class Pattern:
    """Patrón completo."""
    direction: str  # "LONG" o "SHORT"
    window: int  # 2-8
    expression: Union[LogicalNode, PredicateNode]
    fitness: float = -999.0
    generation_created: int = 0

    # Metadata de evaluación bidireccional
    fitness_long: float = -999.0
    fitness_short: float = -999.0

    def evaluate_on_data(self, data: pd.DataFrame) -> bool:
        """Evalúa patrón en ventana."""
        if len(data) < self.window:
            return False

        window_data = data.tail(self.window)

        try:
            return self.expression.evaluate(window_data)
        except Exception as e:
            logger.error(f"Error evaluating pattern: {e}")
            return False

    def to_dict(self) -> dict:
        """Serializa patrón."""
        return {
            'direction': self.direction,
            'window': self.window,
            'expression': self._serialize_node(self.expression),
            'fitness': self.fitness,
            'fitness_long': self.fitness_long,
            'fitness_short': self.fitness_short,
            'generation': self.generation_created
        }

    def _serialize_node(self, node):
        """Serializa árbol."""
        if isinstance(node, PredicateNode):
            return {
                'type': 'predicate',
                'name': node.predicate_name,
                'operator': node.operator,
                'threshold': node.threshold,
                'bar_offset': node.bar_offset,
                'compare_with_bar': node.compare_with_bar,
                'params': node.params
            }
        elif isinstance(node, LogicalNode):
            return {
                'type': 'logical',
                'operator': node.operator,
                'children': [self._serialize_node(c) for c in node.children]
            }

    def __repr__(self):
        return f"Pattern({self.direction}, w={self.window}, fit={self.fitness:.4f}, gen={self.generation_created})\n  {self.expression}"


# ============================================================================
# VALIDACIÓN CON CONSTRAINTS ADAPTATIVOS
# ============================================================================

def validate_pattern(pattern: Pattern) -> bool:
    """
    Valida patrón con constraints adaptativos.

    Constraints:
    1. Window: 2 <= window <= 8
    2. Direction: 'LONG' or 'SHORT'
    3. Depth: max_depth <= 4
    4. Predicates: max adapta según generación
       - Gen 1-50:   max 4 predicados
       - Gen 51-100: max 6 predicados
       - Gen 101+:   max 8 predicados
    """
    if not 2 <= pattern.window <= 8:
        return False

    if pattern.direction not in ['LONG', 'SHORT']:
        return False

    depth = _calculate_depth(pattern.expression)
    if depth > 4:
        return False

    # Constraint adaptativo de predicados
    predicate_count = _count_predicates(pattern.expression)
    generation = pattern.generation_created

    if generation <= 50:
        max_predicates = 4
    elif generation <= 100:
        max_predicates = 6
    else:
        max_predicates = 8

    if predicate_count > max_predicates:
        logger.debug(f"Too many predicates: {predicate_count} > {max_predicates} (gen {generation})")
        return False

    return True

def _calculate_depth(node, current_depth: int = 0) -> int:
    """Calcula profundidad."""
    if isinstance(node, PredicateNode):
        return current_depth
    else:
        if not node.children:
            return current_depth
        return max([_calculate_depth(c, current_depth + 1) for c in node.children])

def _count_predicates(node) -> int:
    """Cuenta predicados."""
    if isinstance(node, PredicateNode):
        return 1
    else:
        return sum([_count_predicates(c) for c in node.children])
