"""
Probability of Backtest Overfitting (PBO)
Bailey & Lopez de Prado, 2014.

Uses CPCV results to estimate the probability that the best in-sample
strategy underperforms out-of-sample.
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def calculate_pbo(cpcv_results: Dict) -> Dict:
    """
    Calculate PBO from CPCV results.

    PBO = proportion of OOS splits where the strategy's Sortino is <= 0.
    A strategy that truly has alpha should have positive Sortino in most splits.

    Args:
        cpcv_results: Output of cpcv_evaluate()

    Returns:
        Dict with:
        - pbo: Probability of Backtest Overfitting (0 to 1)
        - interpretation: Human-readable assessment
        - n_splits: Number of splits used
        - n_negative: Number of splits with Sortino <= 0
    """
    sortinos = cpcv_results.get('oos_sortinos', [])

    if not sortinos:
        return {
            'pbo': 1.0,
            'interpretation': 'No CPCV data available',
            'n_splits': 0,
            'n_negative': 0,
        }

    n_negative = sum(1 for s in sortinos if s <= 0)
    pbo = n_negative / len(sortinos)

    if pbo < 0.30:
        interp = 'Strong evidence of real alpha (PBO < 0.30)'
    elif pbo < 0.50:
        interp = 'Moderate evidence, warrants further testing (PBO < 0.50)'
    elif pbo < 0.70:
        interp = 'Weak evidence, likely some overfitting (PBO < 0.70)'
    else:
        interp = 'Strong evidence of overfitting (PBO >= 0.70)'

    result = {
        'pbo': pbo,
        'interpretation': interp,
        'n_splits': len(sortinos),
        'n_negative': n_negative,
    }

    logger.info(f"PBO = {pbo:.3f} ({interp})")
    return result
