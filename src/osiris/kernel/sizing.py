"""Position sizing. Volatility-targeted with a fractional-Kelly ceiling.

Never fixed share counts: those silently size up as price rises, so a position
grows its risk contribution exactly as it becomes more expensive.
"""

from __future__ import annotations

import math


def atr_from_bars(highs: list[float], lows: list[float], closes: list[float]) -> float:
    """Average True Range. Wilder's true range, simple mean."""
    if len(closes) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else 0.0


def realized_vol(returns: list[float], annualize: bool = True) -> float:
    """Standard deviation of returns, optionally annualized (252 trading days)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    vol = math.sqrt(var)
    return vol * math.sqrt(252) if annualize else vol


def vol_target_notional(
    equity: float,
    *,
    symbol_vol: float,
    target_portfolio_vol: float = 0.15,
    position_count: int = 20,
    max_weight: float = 0.10,
) -> float:
    """Size so each name contributes comparable risk, not comparable dollars.

    A 60%-vol name gets a smaller allocation than a 15%-vol name. Without this,
    the highest-volatility holding silently dominates portfolio outcomes.
    """
    if equity <= 0 or position_count <= 0:
        return 0.0

    equal_weight = 1.0 / position_count
    if symbol_vol <= 1e-9:
        weight = equal_weight
    else:
        # Per-name vol budget, scaled by how far this name's vol sits from it.
        per_name_vol_budget = target_portfolio_vol / math.sqrt(position_count)
        weight = per_name_vol_budget / symbol_vol
        # Never let a low-vol name dominate merely because it is quiet.
        weight = min(weight, equal_weight * 2.0)

    weight = min(weight, max_weight)
    return max(0.0, equity * weight)


def fractional_kelly_cap(
    win_rate: float, win_loss_ratio: float, fraction: float = 0.25
) -> float:
    """Fractional Kelly as a WEIGHT CEILING, never as a target.

    Full Kelly maximizes long-run log growth but tolerates ruinous drawdowns and
    assumes the edge estimate is correct. It never is. Quarter-Kelly is the
    conventional compromise; the result here is only ever used as an upper bound.
    """
    if win_loss_ratio <= 0 or not (0.0 < win_rate < 1.0):
        return 0.0
    kelly = win_rate - (1.0 - win_rate) / win_loss_ratio
    return max(0.0, kelly * fraction)


def clamp_to_adv(notional: float, adv_usd: float, max_participation: float) -> float:
    """Cap order size as a fraction of average daily dollar volume.

    Keeps the strategy viable as capital grows: an order that is a large share of
    ADV moves the price against itself, and the modeled slippage stops being real.
    """
    if adv_usd <= 0:
        return 0.0
    return min(notional, adv_usd * max_participation)
