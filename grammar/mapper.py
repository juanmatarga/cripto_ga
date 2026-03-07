"""
Genotype-to-Phenotype Mapper for Grammatical Evolution.

Maps a list of integer codons to a Strategy phenotype using the BNF grammar.
"""

import re
import logging
from typing import List, Optional, Tuple

from grammar.bnf import GRAMMAR, START_SYMBOL, MAX_WRAPS
from strategy.phenotype import Strategy, Condition

logger = logging.getLogger(__name__)

# Regex to find the first non-terminal in a string
_NT_RE = re.compile(r'<[^>]+>')

# Comparator tokens used in grammar (avoid < > conflict with non-terminal syntax)
_COMPARATORS = ('CROSSES_ABOVE', 'CROSSES_BELOW', '_GT_', '_LT_')

# Map grammar tokens to actual comparator strings for Condition objects
_COMP_MAP = {
    '_GT_': '>',
    '_LT_': '<',
    'CROSSES_ABOVE': 'CROSSES_ABOVE',
    'CROSSES_BELOW': 'CROSSES_BELOW',
}


def decode(genome: List[int]) -> Optional[Strategy]:
    """
    Decode a genome (integer codons) into a Strategy phenotype.

    Returns Strategy if decoding succeeds, None if genome is invalid.
    """
    if not genome:
        return None

    expr = START_SYMBOL
    codon_idx = 0
    wraps = 0
    codons_used = 0

    max_expansions = len(genome) * (MAX_WRAPS + 1) * 2
    expansions = 0

    while True:
        match = _NT_RE.search(expr)
        if match is None:
            break

        expansions += 1
        if expansions > max_expansions:
            return None

        nt = match.group()
        productions = GRAMMAR.get(nt)
        if productions is None:
            return None

        if codon_idx >= len(genome):
            wraps += 1
            if wraps > MAX_WRAPS:
                return None
            codon_idx = 0

        codon = genome[codon_idx]
        codon_idx += 1
        codons_used += 1

        chosen = productions[codon % len(productions)]
        expr = expr[:match.start()] + chosen + expr[match.end():]

    return _parse_expression(expr, genome, codons_used, wraps)


def _parse_expression(expr: str, genome: List[int],
                      codons_used: int, wraps: int) -> Optional[Strategy]:
    """
    Parse fully-expanded expression into a Strategy.

    Expected format: "LONG WHEN <conditions> EXIT TP=2.0 SL=1.0"
    """
    when_idx = expr.find(' WHEN ')
    exit_idx = expr.find(' EXIT ')

    if when_idx == -1 or exit_idx == -1:
        return None

    direction = expr[:when_idx].strip()
    conditions_str = expr[when_idx + 6:exit_idx].strip()
    exit_str = expr[exit_idx + 6:].strip()

    if direction not in ('LONG', 'SHORT'):
        return None

    tp_mult, sl_mult, trail_mult = _parse_exits(exit_str)
    if sl_mult is None:
        return None

    conditions, logic = _parse_conditions(conditions_str)
    if not conditions:
        return None

    return Strategy(
        genome=genome,
        direction=direction,
        conditions=conditions,
        logic=logic,
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        trail_atr_mult=trail_mult,
        expression_raw=expr,
        n_nodes=len(conditions),
        codons_used=codons_used,
        wrapping_count=wraps,
    )


def _parse_exits(exit_str: str) -> Tuple[float, Optional[float], float]:
    """Parse exit params into (tp, sl, trail).

    Supports:
      'TP=2.0 SL=1.0'              -> (2.0, 1.0, 0.0)
      'SL=1.0 TRAIL=2.0'           -> (0.0, 1.0, 2.0)
      'TP=4.0 SL=1.0 TRAIL=2.0'   -> (4.0, 1.0, 2.0)
    """
    try:
        tp = 0.0
        sl = None
        trail = 0.0
        for part in exit_str.split():
            if part.startswith('TP='):
                tp = float(part[3:])
            elif part.startswith('SL='):
                sl = float(part[3:])
            elif part.startswith('TRAIL='):
                trail = float(part[6:])
        if sl is None:
            return 0.0, None, 0.0
        # Must have either TP or TRAIL (or both)
        if tp == 0.0 and trail == 0.0:
            return 0.0, None, 0.0
        return tp, sl, trail
    except (ValueError, IndexError):
        return 0.0, None, 0.0


def _parse_conditions(conditions_str: str) -> Tuple[List[Condition], str]:
    """
    Parse condition string into Condition list and logic string.

    The grammar produces tokens like:
        "RSI(close, 14) _GT_ 30 AND SMA(close, 20) _LT_ close"

    We split on AND/OR (respecting parens), then parse each atomic condition.
    """
    # Step 1: Split into atomic conditions and build logic string
    atoms, logic = _split_into_atoms(conditions_str)

    if not atoms:
        return [], ""

    # Step 2: Parse each atom into a Condition
    conditions = []
    for atom in atoms:
        cond = _parse_single_condition(atom.strip())
        if cond is None:
            return [], ""
        conditions.append(cond)

    return conditions, logic


def _split_into_atoms(text: str) -> Tuple[List[str], str]:
    """
    Split "A _GT_ B AND C _LT_ D OR E CROSSES_ABOVE F"
    into atoms ["A _GT_ B", "C _LT_ D", "E CROSSES_ABOVE F"]
    and logic "c0 AND c1 OR c2".

    Respects parentheses grouping.
    """
    atoms = []
    logic_parts = []
    current = []
    depth = 0
    tokens = text.split()
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == '(':
            depth += 1
            if depth == 1:
                logic_parts.append('(')
                i += 1
                continue
            else:
                current.append(token)
        elif token == ')':
            if depth == 1:
                # Close a group — flush current atom if any
                if current:
                    atom_str = ' '.join(current)
                    cond = _parse_single_condition(atom_str)
                    if cond is not None:
                        atoms.append(atom_str)
                        logic_parts.append(f"c{len(atoms)-1}")
                    current = []
                logic_parts.append(')')
                depth -= 1
                i += 1
                continue
            else:
                depth -= 1
                current.append(token)
        elif token in ('AND', 'OR') and depth <= 1:
            # Logic operator — flush current atom
            if current:
                atom_str = ' '.join(current)
                atoms.append(atom_str)
                logic_parts.append(f"c{len(atoms)-1}")
                current = []
            logic_parts.append(token)
        else:
            current.append(token)

        i += 1

    # Flush remaining
    if current:
        atom_str = ' '.join(current)
        atoms.append(atom_str)
        logic_parts.append(f"c{len(atoms)-1}")

    logic = ' '.join(logic_parts)
    return atoms, logic


def _parse_single_condition(atom: str) -> Optional[Condition]:
    """
    Parse a single atomic condition like "RSI(close, 14) _GT_ 30"
    into a Condition object.
    """
    # Try each comparator (longest first to avoid partial matches)
    for token in _COMPARATORS:
        # Find the comparator token, but not inside parentheses
        pos = _find_token_outside_parens(atom, token)
        if pos != -1:
            left = atom[:pos].strip()
            right = atom[pos + len(token):].strip()
            if left and right:
                actual_comp = _COMP_MAP[token]
                return Condition(left=left, comparator=actual_comp, right=right)

    return None


def _find_token_outside_parens(text: str, token: str) -> int:
    """Find position of token in text, but only at parenthesis depth 0."""
    depth = 0
    tlen = len(token)
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0:
            # Check if token starts here
            if text[i:i+tlen] == token:
                # Verify it's surrounded by spaces or at string boundary
                before_ok = (i == 0 or text[i-1] == ' ')
                after_ok = (i + tlen >= len(text) or text[i+tlen] == ' ')
                if before_ok and after_ok:
                    return i
        i += 1
    return -1
