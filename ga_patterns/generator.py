"""
Genetic Operators - Advanced mutation with term substitution
"""

import random
import numpy as np
from typing import List
import copy
import logging

from ga_patterns.grammar import (
    get_available_predicates, PREDICATE_REGISTRY,
    get_common_offsets, get_random_offset
)
from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode, validate_pattern

logger = logging.getLogger(__name__)

def generate_random_pattern(generation: int, config: dict) -> Pattern:
    """
    Genera patrón random con mínimo 2 predicados.
    Garantiza complejidad suficiente para capturar lógica real.
    """
    direction = random.choice(['LONG', 'SHORT'])
    window = random.randint(config['window_min'], config['window_max'])

    available_predicates = get_available_predicates(generation, config.get('allow_indicators', False))

    # FORZAR mínimo 2 predicados
    operator = random.choice(['AND', 'OR'])
    num_predicates = random.randint(2, 4)

    predicates = [
        _generate_random_predicate(available_predicates)
        for _ in range(num_predicates)
    ]

    expression = LogicalNode(operator=operator, children=predicates)

    pattern = Pattern(
        direction=direction,
        window=window,
        expression=expression,
        generation_created=generation
    )

    if not validate_pattern(pattern):
        # Fallback: AND con 2 predicados simples
        expression = LogicalNode(
            operator='AND',
            children=[
                _generate_random_predicate(available_predicates),
                _generate_random_predicate(available_predicates)
            ]
        )
        pattern.expression = expression

        if not validate_pattern(pattern):
            raise ValueError("Could not generate valid pattern")

    return pattern

def _generate_random_tree(available_predicates: List[str], depth: int, max_depth: int):
    """Genera árbol random."""
    if depth == max_depth:
        return _generate_random_predicate(available_predicates)

    # 30% terminar temprano
    if random.random() < 0.3:
        return _generate_random_predicate(available_predicates)

    operator = random.choice(['AND', 'OR'])
    num_children = random.randint(2, 3)

    children = [
        _generate_random_tree(available_predicates, depth + 1, max_depth)
        for _ in range(num_children)
    ]

    return LogicalNode(operator=operator, children=children)

def _generate_random_predicate(available_predicates: List[str]) -> PredicateNode:
    """
    Genera PredicateNode con offsets mixtos.

    Para predicados 'direct' (C, O, H, L, V):
    - 70% comparación con otra barra (C[0] > C[1])
    - 30% comparación con threshold

    Para predicados 'ratio':
    - 100% comparación con threshold
    """
    predicate_name = random.choice(available_predicates)
    predicate_info = PREDICATE_REGISTRY[predicate_name]

    operator = random.choice(['>', '>=', '<', '<='])
    bar_offset = random.randint(0, 5)

    # Determinar tipo de comparación
    if predicate_info['allows_comparison'] and random.random() < 0.7:
        # Comparación con otra barra (offsets mixtos)
        if random.random() < 0.7:
            # 70%: Offset común
            compare_offset = bar_offset + random.choice(get_common_offsets())
        else:
            # 30%: Offset aleatorio
            compare_offset = bar_offset + get_random_offset()

        return PredicateNode(
            predicate_name=predicate_name,
            operator=operator,
            threshold=None,
            bar_offset=bar_offset,
            compare_with_bar=compare_offset
        )
    else:
        # Comparación con threshold
        threshold_min, threshold_max = predicate_info['threshold_range']
        threshold = random.uniform(threshold_min, threshold_max)

        return PredicateNode(
            predicate_name=predicate_name,
            operator=operator,
            threshold=threshold,
            bar_offset=bar_offset,
            compare_with_bar=None
        )

def initialize_population(population_size: int, generation: int, config: dict) -> List[Pattern]:
    """
    Genera población inicial con 50% templates, 50% random.
    Templates garantizan calidad inicial y lógica de trading.
    Random permite exploración y descubrimiento.
    """
    from ga_patterns.templates import generate_from_template

    logger.info(f"Initializing population of {population_size} patterns...")
    logger.info(f"  50% from templates, 50% random")
    population = []

    n_templates = population_size // 2
    n_random = population_size - n_templates

    # 1. Generate from templates
    for i in range(n_templates):
        try:
            pattern = generate_from_template(generation)
            population.append(pattern)

            if (i + 1) % 20 == 0:
                logger.info(f"  Templates: {i+1}/{n_templates}")
        except Exception as e:
            logger.error(f"Failed to generate template pattern {i}: {e}")
            continue

    # 2. Generate random
    for i in range(n_random):
        try:
            pattern = generate_random_pattern(generation, config)
            population.append(pattern)

            if (i + 1) % 20 == 0:
                logger.info(f"  Random: {i+1}/{n_random}")
        except Exception as e:
            logger.error(f"Failed to generate random pattern {i}: {e}")
            continue

    logger.info(f"[OK] Population initialized: {len(population)} ({n_templates} templates, {len(population) - n_templates} random)")
    return population

def tournament_selection(population: List[Pattern], tournament_size: int = 3) -> Pattern:
    """Selección por torneo."""
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda p: p.fitness)
    return winner

def subtree_crossover(parent1: Pattern, parent2: Pattern,
                     generation: int, config: dict) -> Pattern:
    """
    Crossover inteligente: preserva estructura lógica.

    Estrategias:
    1. Intercambio de predicados (preserva AND/OR)
    2. Intercambio de subárboles completos
    3. Mezcla de parámetros (window, thresholds)
    """
    offspring = copy.deepcopy(parent1)

    # 60% predicado exchange (inteligente)
    # 40% subtree exchange (clásico)
    if random.random() < 0.6:
        # STRATEGY 1: Predicate exchange (preserva LogicalNode)
        predicates_p1 = _get_all_predicates(offspring.expression)
        predicates_p2 = _get_all_predicates(parent2.expression)

        if len(predicates_p1) > 0 and len(predicates_p2) > 0:
            # Seleccionar predicado random de cada padre
            pred_from_p1 = random.choice(predicates_p1)
            pred_from_p2 = copy.deepcopy(random.choice(predicates_p2))

            # Reemplazar pred_from_p1 con pred_from_p2 en offspring
            offspring.expression = _replace_predicate_in_tree(offspring.expression, pred_from_p1, pred_from_p2)

            logger.debug("Crossover: predicate exchange")
    else:
        # STRATEGY 2: Subtree exchange (clásico)
        subtree_p2 = _select_random_subtree(copy.deepcopy(parent2.expression))
        offspring.expression = _replace_random_subtree(offspring.expression, subtree_p2)
        logger.debug("Crossover: subtree exchange")

    # STRATEGY 3: Parameter mixing
    offspring.direction = random.choice([parent1.direction, parent2.direction])
    offspring.window = random.choice([parent1.window, parent2.window])
    offspring.generation_created = generation
    offspring.fitness = -999.0
    offspring.fitness_long = -999.0
    offspring.fitness_short = -999.0

    if not validate_pattern(offspring):
        logger.debug("Crossover produced invalid pattern, returning parent1")
        return copy.deepcopy(parent1)

    return offspring

def _select_random_subtree(node):
    """Selecciona subtree random."""
    all_nodes = []

    def collect_nodes(n):
        all_nodes.append(n)
        if isinstance(n, LogicalNode):
            for child in n.children:
                collect_nodes(child)

    collect_nodes(node)
    selected = random.choice(all_nodes)
    return copy.deepcopy(selected)

def _replace_random_subtree(tree, new_subtree):
    """Reemplaza subtree random."""
    if isinstance(tree, PredicateNode):
        return new_subtree

    if random.random() < 0.3:
        return new_subtree

    if tree.children:
        child_idx = random.randint(0, len(tree.children) - 1)
        tree.children[child_idx] = _replace_random_subtree(tree.children[child_idx], new_subtree)

    return tree

def _get_all_predicates(node) -> List[PredicateNode]:
    """
    Extrae todos los PredicateNode de un árbol.
    Útil para crossover inteligente.
    """
    predicates = []

    def collect(n):
        if isinstance(n, PredicateNode):
            predicates.append(n)
        elif isinstance(n, LogicalNode):
            for child in n.children:
                collect(child)

    collect(node)
    return predicates

def _replace_predicate_in_tree(tree, old_pred: PredicateNode, new_pred: PredicateNode):
    """
    Reemplaza old_pred con new_pred en el árbol.
    Preserva estructura lógica (AND/OR).
    """
    if isinstance(tree, PredicateNode):
        if tree is old_pred:
            return copy.deepcopy(new_pred)
        else:
            return tree

    if isinstance(tree, LogicalNode):
        tree.children = [
            _replace_predicate_in_tree(child, old_pred, new_pred)
            for child in tree.children
        ]

    return tree

def mutate_pattern(pattern: Pattern, generation: int, config: dict) -> Pattern:
    """
    Mutación con 5 tipos:
    1. Subtree (30%)
    2. Threshold (25%)
    3. Operator (20%)
    4. Simplify (10%)
    5. Add Predicate (15%) - NUEVO
    """
    mutated = copy.deepcopy(pattern)

    mutation_type = random.choices(
        ['subtree', 'threshold', 'operator', 'simplify', 'add_predicate'],
        weights=[0.3, 0.25, 0.2, 0.10, 0.15]
    )[0]

    logger.debug(f"Applying {mutation_type} mutation")

    if mutation_type == 'subtree':
        # Reemplazar subtree con predicado random
        available_predicates = get_available_predicates(generation, config.get('allow_indicators', False))
        new_subtree = _generate_random_predicate(available_predicates)
        mutated.expression = _replace_random_subtree(mutated.expression, new_subtree)

    elif mutation_type == 'threshold':
        # Ajustar threshold o cambiar offset
        predicate = _select_random_predicate_node(mutated.expression)
        if predicate:
            if predicate.compare_with_bar is not None:
                # Cambiar offset de comparación
                if random.random() < 0.5:
                    predicate.compare_with_bar += random.choice([-1, 1])
                    predicate.compare_with_bar = max(0, predicate.compare_with_bar)
                else:
                    predicate.bar_offset += random.choice([-1, 1])
                    predicate.bar_offset = max(0, predicate.bar_offset)
            else:
                # Ajustar threshold
                adjustment = random.uniform(0.8, 1.2)
                predicate.threshold *= adjustment

                pred_info = PREDICATE_REGISTRY.get(predicate.predicate_name)
                if pred_info:
                    min_t, max_t = pred_info['threshold_range']
                    predicate.threshold = max(min_t, min(max_t, predicate.threshold))

    elif mutation_type == 'operator':
        # Flip operator
        node = _select_random_node(mutated.expression)
        if isinstance(node, PredicateNode):
            flip_map = {'>': '<=', '>=': '<', '<': '>=', '<=': '>'}
            node.operator = flip_map.get(node.operator, node.operator)
        elif isinstance(node, LogicalNode):
            node.operator = 'OR' if node.operator == 'AND' else 'AND'

    elif mutation_type == 'simplify':
        # Simplify - reduce complejidad
        if isinstance(mutated.expression, LogicalNode):
            if mutated.expression.children:
                mutated.expression = copy.deepcopy(random.choice(mutated.expression.children))
                logger.debug("Simplified: replaced LogicalNode with child")

    elif mutation_type == 'add_predicate':
        # NUEVO: Add predicate - incrementa complejidad
        available_predicates = get_available_predicates(generation, config.get('allow_indicators', False))
        new_predicate = _generate_random_predicate(available_predicates)

        if isinstance(mutated.expression, LogicalNode):
            # Agregar a LogicalNode existente
            mutated.expression.children.append(new_predicate)
            logger.debug(f"Added predicate to LogicalNode (now {len(mutated.expression.children)} children)")
        else:
            # Wrap en LogicalNode con nuevo predicado
            operator = random.choice(['AND', 'OR'])
            mutated.expression = LogicalNode(
                operator=operator,
                children=[mutated.expression, new_predicate]
            )
            logger.debug(f"Wrapped in LogicalNode {operator} with new predicate")

    mutated.generation_created = generation
    mutated.fitness = -999.0
    mutated.fitness_long = -999.0
    mutated.fitness_short = -999.0

    if not validate_pattern(mutated):
        logger.debug("Mutation produced invalid pattern, returning original")
        return copy.deepcopy(pattern)

    return mutated

def _select_random_predicate_node(node) -> PredicateNode:
    """Selecciona PredicateNode random."""
    predicates = []

    def collect_predicates(n):
        if isinstance(n, PredicateNode):
            predicates.append(n)
        elif isinstance(n, LogicalNode):
            for child in n.children:
                collect_predicates(child)

    collect_predicates(node)
    return random.choice(predicates) if predicates else None

def _select_random_node(node):
    """Selecciona nodo random."""
    all_nodes = []

    def collect_all(n):
        all_nodes.append(n)
        if isinstance(n, LogicalNode):
            for child in n.children:
                collect_all(child)

    collect_all(node)
    return random.choice(all_nodes) if all_nodes else None
