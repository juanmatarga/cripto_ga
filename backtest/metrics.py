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
        # Sin drawdowns -> UPI infinito (o muy alto)
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

def calculate_all_metrics(equity_curve: pd.Series, periods_per_year: int,
                         trades: list = None) -> Dict[str, float]:
    """
    Calcula bundle completo de métricas.

    Args:
        equity_curve: Series con equity values
        periods_per_year: De config['data']['time_map'][timeframe]['bars_per_year']
        trades: Optional list of trade dicts (for accurate win rate / profit factor)
                Each trade dict should have 'pnl_pct' key

    Returns:
        dict con keys: upi, sharpe, cagr, max_dd, ulcer_index, total_return, volatility, num_periods,
                      win_rate, profit_factor

    Notes:
        - If trades provided: win_rate and profit_factor calculated from trade outcomes (CORRECT)
        - If trades not provided: calculated from equity curve returns (APPROXIMATE - less accurate)
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
            'num_periods': len(equity_curve),
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

    returns = calculate_returns(equity_curve).dropna()

    # SPRINT 14 FIX: Calculate win rate and profit factor from trades if available
    if trades is not None and len(trades) > 0:
        # [PASS] CORRECT: Calculate from trade outcomes
        winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl_pct', 0) < 0]

        win_rate = len(winning_trades) / len(trades)

        # Profit factor from actual trade P&L
        total_gains = sum(t['pnl_pct'] for t in winning_trades)
        total_losses = abs(sum(t['pnl_pct'] for t in losing_trades))

        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = 999.0  # All wins, no losses
        else:
            profit_factor = 0.0  # No trades

        logger.debug(f"Win rate calculated from {len(trades)} trades: {win_rate:.2%} "
                    f"({len(winning_trades)}W / {len(losing_trades)}L)")
    else:
        # FALLBACK: Calculate from equity curve returns (APPROXIMATE - less accurate)
        # This counts bars where equity increased, not winning trades
        logger.debug("No trades provided - calculating win rate from equity curve (less accurate)")

        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

        # Profit factor = sum(positive returns) / abs(sum(negative returns))
        total_gains = positive_returns.sum() if len(positive_returns) > 0 else 0.0
        total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0

        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = 999.0  # All wins, no losses
        else:
            profit_factor = 0.0  # No trades

    metrics = {
        'upi': upi_ratio(equity_curve, periods_per_year),
        'sharpe': sharpe_ratio(equity_curve, periods_per_year),
        'cagr': cagr(equity_curve, periods_per_year),
        'max_dd': max_drawdown(equity_curve),
        'ulcer_index': ulcer_index(equity_curve),
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'volatility': returns.std() * np.sqrt(periods_per_year),  # Anualizada
        'num_periods': len(equity_curve),
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

    return metrics


def calculate_sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """
    Calculate Sortino ratio (return / downside deviation) - SPRINT 12.

    Like Sharpe but only penalizes downside volatility, which is more
    relevant for traders who don't mind upside volatility.

    Args:
        returns: Series of returns (pct_change)
        periods_per_year: Number of periods per year for annualization

    Returns:
        float: Sortino ratio (higher is better)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sortino = calculate_sortino_ratio(returns, 252)
        >>> sortino > 0
        True

    Notes:
        - Only negative returns are considered in volatility calculation
        - Returns 999.0 if there is no downside volatility (perfect strategy)
        - Annualized using periods_per_year
    """
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()

    # Only negative returns for downside risk
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        # No downside = perfect strategy
        return 999.0

    downside_std = downside_returns.std()

    if downside_std == 0 or np.isnan(downside_std):
        return 999.0

    # Annualize
    sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_calmar_ratio(cagr_value: float, max_dd_value: float) -> float:
    """
    Calculate Calmar ratio (CAGR / |max drawdown|) - SPRINT 12.

    Measures return per unit of worst drawdown.
    Higher is better.

    Args:
        cagr_value: Compound annual growth rate (decimal, e.g., 0.15 for 15%)
        max_dd_value: Maximum drawdown (decimal, e.g., -0.30 for -30%)

    Returns:
        float: Calmar ratio (higher is better)

    Example:
        >>> calmar = calculate_calmar_ratio(0.25, -0.10)
        >>> calmar
        2.5

    Notes:
        - If max_dd is essentially 0, returns a very high score
        - If CAGR is negative, returns 0 (invalid strategy)
    """
    if cagr_value <= 0:
        # Negative or zero CAGR = bad strategy
        return 0.0

    if max_dd_value >= -0.01:
        # Essentially no drawdown = excellent
        return cagr_value * 100

    calmar = cagr_value / abs(max_dd_value)

    return calmar
