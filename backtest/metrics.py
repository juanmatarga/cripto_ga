"""
Performance Metrics - Crypto-adapted calculations
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calcula returns desde equity curve.

    Args:
        equity_curve: Series con valores de equity

    Returns:
        pd.Series: Returns (pct_change)
    """
    return equity_curve.pct_change()

def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calcula serie de drawdowns desde peak histórico.

    Formula: DD[t] = (Equity[t] - Peak[t]) / Peak[t]

    Returns:
        pd.Series: Drawdowns (valores negativos o cero)
    """
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown

def ulcer_index(equity_curve: pd.Series) -> float:
    """
    Calcula Ulcer Index (medida de dolor de drawdown).

    Formula: UI = sqrt(mean(drawdown^2))

    Mejor que max_drawdown porque penaliza drawdowns prolongados.

    Returns:
        float: Ulcer Index (>= 0, menor es mejor)
    """
    dd_series = calculate_drawdown_series(equity_curve)

    # Ulcer = sqrt(mean(DD^2))
    ui = np.sqrt((dd_series ** 2).mean())

    return ui

def upi_ratio(equity_curve: pd.Series, periods_per_year: int) -> float:
    """
    Calcula UPI (Ulcer Performance Index / Martin Ratio).

    Formula: UPI = CAGR / Ulcer Index

    Args:
        equity_curve: Series con equity values
        periods_per_year: Número de períodos en un año (de TIME_MAP)

    Returns:
        float: UPI (mayor es mejor, puede ser negativo si CAGR < 0)
    """
    cagr_val = cagr(equity_curve, periods_per_year)
    ui = ulcer_index(equity_curve)

    if ui == 0:
        # Sin drawdowns → UPI infinito (o muy alto)
        # Decisión: retornar 100 (cap alto)
        return 100.0 if cagr_val > 0 else 0.0

    return cagr_val / ui

def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int,
                 risk_free_rate: float = 0.0) -> float:
    """
    Calcula Sharpe Ratio anualizado.

    Formula: Sharpe = (mean(returns) - rf) / std(returns) * sqrt(periods_per_year)

    Args:
        periods_per_year: De TIME_MAP (ej: 35040 para 15m)
        risk_free_rate: Tasa libre de riesgo anualizada (default 0 para cripto)

    Returns:
        float: Sharpe ratio anualizado
    """
    returns = calculate_returns(equity_curve).dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Anualizar
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)

    sharpe = (mean_return - risk_free_rate) / std_return

    return sharpe

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calcula máximo drawdown.

    Returns:
        float: Max DD (<=0, ej: -0.25 para 25% drawdown)
    """
    dd_series = calculate_drawdown_series(equity_curve)
    max_dd = dd_series.min()
    return max_dd

def cagr(equity_curve: pd.Series, periods_per_year: int) -> float:
    """
    Calcula Compound Annual Growth Rate.

    Formula: CAGR = (final/initial)^(periods_per_year/n) - 1

    Returns:
        float: CAGR anualizado (ej: 0.15 para 15%)
    """
    if len(equity_curve) < 2:
        return 0.0

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    n_periods = len(equity_curve)

    if initial <= 0:
        return 0.0

    cagr_val = (final / initial) ** (periods_per_year / n_periods) - 1

    return cagr_val

def calculate_all_metrics(equity_curve: pd.Series, periods_per_year: int) -> Dict[str, float]:
    """
    Calcula bundle completo de métricas.

    Args:
        equity_curve: Series con equity values
        periods_per_year: De config['data']['time_map'][timeframe]['bars_per_year']

    Returns:
        dict con keys: upi, sharpe, cagr, max_dd, ulcer_index, total_return, volatility, num_periods
    """
    if len(equity_curve) < 2:
        logger.warning("Equity curve too short (<2 points), returning zero metrics")
        return {
            'upi': 0.0,
            'sharpe': 0.0,
            'cagr': 0.0,
            'max_dd': 0.0,
            'ulcer_index': 0.0,
            'total_return': 0.0,
            'volatility': 0.0,
            'num_periods': len(equity_curve)
        }

    returns = calculate_returns(equity_curve).dropna()

    metrics = {
        'upi': upi_ratio(equity_curve, periods_per_year),
        'sharpe': sharpe_ratio(equity_curve, periods_per_year),
        'cagr': cagr(equity_curve, periods_per_year),
        'max_dd': max_drawdown(equity_curve),
        'ulcer_index': ulcer_index(equity_curve),
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'volatility': returns.std() * np.sqrt(periods_per_year),  # Anualizada
        'num_periods': len(equity_curve)
    }

    return metrics
