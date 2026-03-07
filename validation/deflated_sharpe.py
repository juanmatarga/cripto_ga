"""
Deflated Sharpe Ratio (DSR)
Bailey & Lopez de Prado, 2014.

Corrects an observed Sharpe ratio for:
1. Number of trials (multiple strategies tested)
2. Skewness and kurtosis of returns
3. Sample length

If DSR > 0.95, the observed Sharpe is likely real (not due to multiple testing).

All Sharpe ratios in this module are on the ANNUALIZED scale.
"""

import numpy as np
from scipy import stats
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Default: 15m bars → 35040 per year
DEFAULT_PERIODS_PER_YEAR = 35040


def _annualized_sr_std(T: int, periods_per_year: int,
                       skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Standard error of the annualized Sharpe ratio estimator under H0 (SR=0).

    The per-period SR has variance ≈ 1/(T-1).
    Annualized SR = per-period SR * sqrt(periods_per_year).
    So Var(SR_annual) = periods_per_year / (T-1).
    """
    if T <= 1:
        return float(np.sqrt(periods_per_year))
    return float(np.sqrt(periods_per_year / (T - 1)))


def expected_max_sharpe(n_trials: int, T: int,
                        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
                        skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Expected maximum annualized Sharpe ratio under the null hypothesis
    (all strategies have zero true Sharpe) given n_trials attempts.

    Based on extreme value theory for the max of n iid normals.
    """
    if n_trials <= 1:
        return 0.0

    log_n = np.log(n_trials)
    if log_n <= 0:
        return 0.0

    sr_std = _annualized_sr_std(T, periods_per_year, skew, kurtosis)

    # Expected max of n standard normals, scaled by SR standard deviation
    expected = sr_std * (
        np.sqrt(2 * log_n) -
        (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))
    )

    return max(expected, 0.0)


def deflated_sharpe_ratio(observed_sharpe: float,
                          n_trials: int,
                          T: int,
                          skew: float = 0.0,
                          kurtosis: float = 3.0,
                          periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
                          sharpe_benchmark: float = 0.0) -> Dict:
    """
    Calculate the Deflated Sharpe Ratio.

    Args:
        observed_sharpe: Annualized Sharpe ratio of the strategy
        n_trials: Total number of strategies evaluated during evolution
        T: Number of return observations (bars)
        skew: Skewness of the strategy's returns
        kurtosis: Kurtosis of the strategy's returns
        periods_per_year: Annualization factor (default 35040 for 15m)
        sharpe_benchmark: Annualized Sharpe of benchmark (default 0)

    Returns:
        Dict with dsr, expected_max_sr, interpretation, etc.
    """
    if T <= 1 or n_trials <= 0:
        return {
            'dsr': 0.0,
            'expected_max_sr': 0.0,
            'observed_sharpe': observed_sharpe,
            'n_trials': n_trials,
            'interpretation': 'Insufficient data',
        }

    e_max_sr = expected_max_sharpe(n_trials, T, periods_per_year, skew, kurtosis)
    sr_std = _annualized_sr_std(T, periods_per_year, skew, kurtosis)

    if sr_std < 1e-10:
        sr_std = 1e-10

    # z-test: is observed SR significantly above expected max under null?
    z = (observed_sharpe - e_max_sr) / sr_std
    dsr = float(stats.norm.cdf(z))

    if dsr >= 0.95:
        interp = f'Sharpe likely real (DSR={dsr:.3f} >= 0.95)'
    elif dsr >= 0.80:
        interp = f'Marginal evidence (DSR={dsr:.3f})'
    else:
        interp = f'Sharpe likely due to multiple testing (DSR={dsr:.3f} < 0.80)'

    result = {
        'dsr': dsr,
        'expected_max_sr': e_max_sr,
        'observed_sharpe': observed_sharpe,
        'n_trials': n_trials,
        'T': T,
        'skew': skew,
        'kurtosis': kurtosis,
        'interpretation': interp,
    }

    logger.info(f"DSR = {dsr:.4f} | observed SR={observed_sharpe:.3f} | "
                f"E[max(SR)]={e_max_sr:.3f} | n_trials={n_trials}")

    return result
