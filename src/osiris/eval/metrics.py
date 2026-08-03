"""Performance metrics. Risk-adjusted, not total return.

Total return over a bull market is not evidence of skill. Every metric here
exists to separate selection from beta, and skill from luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class PerfSummary:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    n_periods: int


def to_array(returns: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(returns), dtype=float)


def total_return(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def cagr(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0:
        return -1.0
    years = returns.size / periods_per_year
    return growth ** (1.0 / years) - 1.0 if years > 0 else 0.0


def volatility(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.size < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def sharpe(
    returns: np.ndarray, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino(
    returns: np.ndarray, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Penalizes only downside deviation, which is what actually hurts."""
    if returns.size < 2:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if float(np.mean(excess)) > 0 else 0.0
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / dd * math.sqrt(periods_per_year))


def equity_curve(returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    if returns.size == 0:
        return np.array([start])
    return start * np.cumprod(1.0 + returns)


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    curve = equity_curve(returns)
    peak = np.maximum.accumulate(curve)
    return (curve - peak) / peak


def max_drawdown(returns: np.ndarray) -> float:
    dd = drawdown_series(returns)
    return float(abs(dd.min())) if dd.size else 0.0


def calmar(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd <= 1e-12:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def win_rate(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.mean(returns > 0))


def summarize(
    returns: list[float] | np.ndarray, periods_per_year: int = TRADING_DAYS
) -> PerfSummary:
    r = to_array(returns)
    return PerfSummary(
        total_return=total_return(r),
        cagr=cagr(r, periods_per_year),
        volatility=volatility(r, periods_per_year),
        sharpe=sharpe(r, periods_per_year=periods_per_year),
        sortino=sortino(r, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(r),
        calmar=calmar(r, periods_per_year),
        win_rate=win_rate(r),
        n_periods=int(r.size),
    )


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_periods: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe adjusted for multiple testing.

    Testing many strategy variants inflates the best observed Sharpe by chance
    alone. Reporting the raw maximum is the single most common way a backtest
    lies. This applies the Bailey / Lopez de Prado deflation: subtract the
    expected maximum Sharpe of `n_trials` null strategies.
    """
    if n_trials < 1 or n_periods < 2:
        return observed_sharpe
    if n_trials == 1:
        return observed_sharpe

    # Expected maximum of n_trials draws from a standard normal.
    euler = 0.5772156649015329
    e_max = (1 - euler) * _norm_ppf(1 - 1.0 / n_trials) + euler * _norm_ppf(
        1 - 1.0 / (n_trials * math.e)
    )

    # Variance of the Sharpe estimator, adjusted for higher moments.
    sr = observed_sharpe
    var_sr = (
        1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2
    ) / max(1, n_periods - 1)
    sd_sr = math.sqrt(max(var_sr, 1e-12))
    return observed_sharpe - e_max * sd_sr


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        return 0.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
