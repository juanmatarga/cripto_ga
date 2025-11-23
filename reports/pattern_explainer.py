"""
Pattern Explainer - Convert expression trees to natural language
"""

from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode
import logging

logger = logging.getLogger(__name__)

def explain_predicate(node: PredicateNode) -> str:
    """
    Convierte PredicateNode a descripción en lenguaje natural.
    
    Args:
        node: PredicateNode
    
    Returns:
        str: Descripción legible
    
    Examples:
        close[0] > close[1] -> "Current close is greater than previous close"
        price_change_pct[0] > 0.02 -> "Price change is above 2.00%"
    """
    predicate_name = node.predicate_name
    operator = node.operator
    bar_offset = node.bar_offset
    
    # Mapeo de nombres técnicos a legibles
    name_mapping = {
        'close': 'close price',
        'open': 'open price',
        'high': 'high price',
        'low': 'low price',
        'volume': 'volume',
        'price_change_pct': 'price change',
        'body_pct': 'candle body',
        'range_pct': 'candle range',
        'volume_change_pct': 'volume change',
        'close_position_in_range': 'close position in range',
        'body_ratio': 'body-to-range ratio',
        'rsi': 'RSI',
        'price_vs_ma_pct': 'price vs moving average'
    }
    
    readable_name = name_mapping.get(predicate_name, predicate_name)
    
    # Mapeo de operadores
    operator_mapping = {
        '>': 'is greater than',
        '>=': 'is at least',
        '<=': 'is at most',
        '<': 'is less than',
        '==': 'equals'
    }
    
    readable_operator = operator_mapping.get(operator, operator)
    
    # Offset description
    if bar_offset == 0:
        offset_desc = "current"
    elif bar_offset == 1:
        offset_desc = "previous"
    elif bar_offset == 2:
        offset_desc = "2 bars ago"
    else:
        offset_desc = f"{bar_offset} bars ago"
    
    # Construir descripción
    if node.compare_with_bar is not None:
        # Comparación con otra barra
        compare_offset = node.compare_with_bar
        
        if compare_offset == 1:
            compare_desc = "previous"
        elif compare_offset == 2:
            compare_desc = "2 bars ago"
        else:
            compare_desc = f"{compare_offset} bars ago"
        
        description = f"{readable_name} ({offset_desc}) {readable_operator} {readable_name} ({compare_desc})"
    
    else:
        # Comparación con threshold
        threshold = node.threshold
        
        # Formatear threshold según tipo de predicado
        if 'pct' in predicate_name or 'change' in predicate_name:
            threshold_str = f"{threshold*100:.2f}%"
        elif 'ratio' in predicate_name or 'position' in predicate_name:
            threshold_str = f"{threshold:.3f}"
        elif 'rsi' in predicate_name:
            threshold_str = f"{threshold:.1f}"
        else:
            threshold_str = f"{threshold:.2f}"
        
        description = f"{readable_name} ({offset_desc}) {readable_operator} {threshold_str}"
    
    return description

def explain_logical_node(node: LogicalNode, indent: int = 0) -> str:
    """
    Convierte LogicalNode a descripción recursiva.
    
    Args:
        node: LogicalNode
        indent: Nivel de indentación
    
    Returns:
        str: Descripción con estructura
    """
    indent_str = "  " * indent
    operator = node.operator
    
    # Explicar cada hijo
    child_explanations = []
    for child in node.children:
        if isinstance(child, PredicateNode):
            child_explanations.append(explain_predicate(child))
        elif isinstance(child, LogicalNode):
            child_explanations.append(explain_logical_node(child, indent + 1))
    
    # Construir explicación
    if operator == 'AND':
        if len(child_explanations) == 2:
            explanation = f"{child_explanations[0]} AND {child_explanations[1]}"
        else:
            explanation = "ALL of the following:\n" + "\n".join([f"{indent_str}  - {exp}" for exp in child_explanations])
    
    elif operator == 'OR':
        if len(child_explanations) == 2:
            explanation = f"{child_explanations[0]} OR {child_explanations[1]}"
        else:
            explanation = "ANY of the following:\n" + "\n".join([f"{indent_str}  - {exp}" for exp in child_explanations])
    
    elif operator == 'NOT':
        explanation = f"NOT ({child_explanations[0]})"
    
    else:
        explanation = f"{operator}({', '.join(child_explanations)})"
    
    return explanation

def explain_pattern(pattern: Pattern) -> str:
    """
    Explica patrón completo en lenguaje natural.
    
    Args:
        pattern: Pattern
    
    Returns:
        str: Explicación completa multi-línea
    """
    direction = pattern.direction
    window = pattern.window
    fitness = pattern.fitness
    
    # Header
    explanation = f"Pattern Description\n"
    explanation += f"{'='*60}\n"
    explanation += f"Direction: {direction}\n"
    explanation += f"Window: {window} bars\n"
    explanation += f"Fitness: {fitness:.4f}\n"
    
    if hasattr(pattern, 'fitness_long') and hasattr(pattern, 'fitness_short'):
        explanation += f"Fitness (LONG): {pattern.fitness_long:.4f}\n"
        explanation += f"Fitness (SHORT): {pattern.fitness_short:.4f}\n"
    
    explanation += f"\nEntry Condition:\n"
    explanation += f"{'-'*60}\n"
    
    # Explicar expression
    if isinstance(pattern.expression, PredicateNode):
        explanation += explain_predicate(pattern.expression)
    elif isinstance(pattern.expression, LogicalNode):
        explanation += explain_logical_node(pattern.expression)
    
    explanation += f"\n{'-'*60}\n"
    
    # Interpretación
    explanation += f"\nInterpretation:\n"
    if direction == 'LONG':
        explanation += f"When this condition is met, the strategy opens a LONG position.\n"
    else:
        explanation += f"When this condition is met, the strategy opens a SHORT position.\n"
    
    explanation += f"The pattern looks at the last {window} bars to evaluate the condition.\n"
    
    return explanation

def explain_portfolio(patterns: list) -> str:
    """
    Explica portfolio completo.
    
    Args:
        patterns: Lista de Pattern (o tuplas de (Pattern, equity, metrics))
    
    Returns:
        str: Explicación del portfolio
    """
    # Extraer patterns si son tuplas
    if len(patterns) > 0 and isinstance(patterns[0], tuple):
        actual_patterns = [p[0] for p in patterns]
    else:
        actual_patterns = patterns
    
    explanation = f"\n{'='*80}\n"
    explanation += f"PORTFOLIO EXPLANATION\n"
    explanation += f"{'='*80}\n\n"
    explanation += f"Total Patterns: {len(actual_patterns)}\n"
    
    long_count = sum(1 for p in actual_patterns if p.direction == 'LONG')
    short_count = sum(1 for p in actual_patterns if p.direction == 'SHORT')
    
    explanation += f"LONG Patterns: {long_count}\n"
    explanation += f"SHORT Patterns: {short_count}\n"
    explanation += f"\n{'-'*80}\n\n"
    
    for i, pattern in enumerate(actual_patterns, 1):
        explanation += f"\nPattern #{i}\n"
        explanation += explain_pattern(pattern)
        explanation += f"\n{'-'*80}\n"
    
    return explanation
