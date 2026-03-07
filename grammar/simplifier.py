"""
Expression Simplifier — canonicalize decoded strategies.

Removes degenerate conditions (e.g., SMA(close,20) > SMA(close,20))
and deduplicates identical conditions.
"""

from typing import List
from strategy.phenotype import Condition


def simplify_conditions(conditions: List[Condition]) -> List[Condition]:
    """
    Remove degenerate and duplicate conditions.

    Returns simplified list. Returns empty list if all conditions are degenerate.
    """
    result = []
    seen = set()

    for cond in conditions:
        # Skip tautologies: same indicator on both sides with > or <
        if cond.left == cond.right and cond.comparator in ('>', '<'):
            continue

        # Skip CROSSES with same indicator
        if cond.left == cond.right and cond.comparator in ('CROSSES_ABOVE', 'CROSSES_BELOW'):
            continue

        # Deduplicate: canonical form is sorted (left, comp, right)
        key = (cond.left, cond.comparator, cond.right)
        if key in seen:
            continue
        seen.add(key)

        result.append(cond)

    return result


def update_logic_after_simplification(logic: str, original_count: int,
                                       kept_indices: List[int]) -> str:
    """
    Update logic string after removing conditions.

    If conditions c1 and c3 were removed from "c0 AND c1 AND c2 AND c3",
    result should be "c0 AND c1" (renumbered).
    """
    if not kept_indices:
        return ""

    # Build mapping: old index -> new index
    mapping = {}
    for new_idx, old_idx in enumerate(kept_indices):
        mapping[f"c{old_idx}"] = f"c{new_idx}"

    result = logic
    # Replace in reverse order to avoid c1 matching in c10
    for old_idx in range(original_count - 1, -1, -1):
        old_key = f"c{old_idx}"
        if old_key in mapping:
            result = result.replace(old_key, f"__{mapping[old_key]}__")
        else:
            # This condition was removed — replace with TRUE
            result = result.replace(old_key, "TRUE")

    # Clean up temp markers
    for new_idx in range(len(kept_indices)):
        result = result.replace(f"__c{new_idx}__", f"c{new_idx}")

    # Simplify logic with TRUE:
    # "X AND TRUE" -> "X", "TRUE AND X" -> "X"
    # "X OR TRUE" -> "TRUE" (but we just drop the whole thing)
    while " AND TRUE" in result:
        result = result.replace(" AND TRUE", "")
    while "TRUE AND " in result:
        result = result.replace("TRUE AND ", "")
    while "(TRUE)" in result:
        result = result.replace("(TRUE)", "TRUE")

    # If everything simplified to TRUE or empty, flag it
    result = result.strip()
    if result in ("TRUE", ""):
        return ""

    return result
