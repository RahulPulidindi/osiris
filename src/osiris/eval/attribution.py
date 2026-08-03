"""Performance attribution and live factor exposure.

Answers the question most retail systems never ask: **am I running a strategy, or
am I long the market with extra steps?**

Total return does not distinguish these. A 20-name equal-weight book of high-beta
growth names will beat the S&P in a rising market with zero skill involved. That
shows up here as a beta loading rather than as alpha, which is exactly the point.
Both cited research systems ran market beta below 1.0 -- that is the evidence that
*selection* was doing the work rather than exposure.

Brinson decomposition splits excess return into:
  - **selection**  -- picked better names inside a sector (this is the edge)
  - **allocation** -- overweighted sectors that outperformed (this is a bet)
  - **interaction** -- the cross term
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FactorExposure:
    """Live regression of realized returns on factor returns."""

    alpha_annualized: float
    alpha_t_stat: float
    loadings: dict[str, float]
    r_squared: float
    n_periods: int

    @property
    def market_beta(self) -> float:
        return self.loadings.get("market", 0.0)

    @property
    def is_alpha_significant(self) -> bool:
        return self.alpha_t_stat >= 2.0

    @property
    def verdict(self) -> str:
        """Plain-language read, since this is the panel that matters most."""
        if self.n_periods < 20:
            return "insufficient history to distinguish alpha from noise"
        if self.is_alpha_significant:
            return f"selection is adding value (alpha t={self.alpha_t_stat:.1f})"
        if abs(self.market_beta) > 1.0:
            return (
                f"returns explained by market exposure (beta {self.market_beta:.2f}), "
                "not selection"
            )
        return f"no significant alpha yet (t={self.alpha_t_stat:.1f})"


def compute_factor_exposure(
    returns: list[float] | np.ndarray,
    factors: dict[str, list[float] | np.ndarray],
    *,
    periods_per_year: int = 252,
) -> FactorExposure:
    """OLS of strategy returns on factor returns, with a t-stat on the intercept."""
    y = np.asarray(list(returns), dtype=float)
    names = list(factors)
    if not names or y.size == 0:
        return FactorExposure(0.0, 0.0, {}, 0.0, int(y.size))

    X = np.column_stack([np.asarray(list(factors[n]), dtype=float) for n in names])
    if X.shape[0] != y.size or y.size <= X.shape[1] + 1:
        return FactorExposure(0.0, 0.0, dict.fromkeys(names, 0.0), 0.0, int(y.size))

    design = np.column_stack([np.ones(y.size), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = y.size - design.shape[1]
    mse = float(resid @ resid) / dof if dof > 0 else 0.0

    try:
        se = np.sqrt(np.diag(mse * np.linalg.inv(design.T @ design)))
    except np.linalg.LinAlgError:
        return FactorExposure(0.0, 0.0, dict.fromkeys(names, 0.0), 0.0, int(y.size))

    total_ss = float(((y - y.mean()) ** 2).sum())
    return FactorExposure(
        alpha_annualized=float(coef[0]) * periods_per_year,
        alpha_t_stat=float(coef[0] / se[0]) if se[0] > 0 else 0.0,
        loadings={n: float(c) for n, c in zip(names, coef[1:], strict=True)},
        r_squared=float(1.0 - (resid @ resid) / total_ss) if total_ss > 0 else 0.0,
        n_periods=int(y.size),
    )


@dataclass(frozen=True)
class SectorDeviation:
    sector: str
    portfolio_weight: float
    benchmark_weight: float

    @property
    def deviation(self) -> float:
        return self.portfolio_weight - self.benchmark_weight


def sector_deviations(
    portfolio_weights: dict[str, float], benchmark_weights: dict[str, float]
) -> list[SectorDeviation]:
    """Active sector bets, largest absolute deviation first."""
    sectors = set(portfolio_weights) | set(benchmark_weights)
    out = [
        SectorDeviation(s, portfolio_weights.get(s, 0.0), benchmark_weights.get(s, 0.0))
        for s in sectors
    ]
    return sorted(out, key=lambda d: -abs(d.deviation))


@dataclass(frozen=True)
class BrinsonAttribution:
    selection: float
    allocation: float
    interaction: float
    total_excess: float
    by_sector: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def selection_share(self) -> float:
        """Fraction of excess return coming from stock picking."""
        if abs(self.total_excess) < 1e-12:
            return 0.0
        return self.selection / self.total_excess

    @property
    def verdict(self) -> str:
        if abs(self.total_excess) < 1e-9:
            return "no excess return to attribute"
        if self.selection_share > 0.6:
            return "excess return is mostly stock selection"
        if abs(self.allocation) > abs(self.selection):
            return "excess return is mostly sector allocation, not selection"
        return "excess return is mixed between selection and allocation"


def brinson_attribution(
    portfolio_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_weights: dict[str, float],
    benchmark_returns: dict[str, float],
) -> BrinsonAttribution:
    """Brinson-Hood-Beebower decomposition at the sector level.

    All four inputs are keyed by sector. Portfolio returns are the realized return
    of *our* holdings in that sector; benchmark returns are the sector index.
    """
    sectors = set(portfolio_weights) | set(benchmark_weights)
    total_bench = sum(
        benchmark_weights.get(s, 0.0) * benchmark_returns.get(s, 0.0) for s in sectors
    )
    total_port = sum(
        portfolio_weights.get(s, 0.0) * portfolio_returns.get(s, 0.0) for s in sectors
    )

    selection = allocation = interaction = 0.0
    by_sector: dict[str, dict[str, float]] = {}

    for s in sectors:
        wp = portfolio_weights.get(s, 0.0)
        wb = benchmark_weights.get(s, 0.0)
        rp = portfolio_returns.get(s, 0.0)
        rb = benchmark_returns.get(s, 0.0)

        sel = wb * (rp - rb)
        alloc = (wp - wb) * rb
        inter = (wp - wb) * (rp - rb)

        selection += sel
        allocation += alloc
        interaction += inter
        by_sector[s] = {
            "selection": sel,
            "allocation": alloc,
            "interaction": inter,
            "portfolio_weight": wp,
            "benchmark_weight": wb,
        }

    return BrinsonAttribution(
        selection=selection,
        allocation=allocation,
        interaction=interaction,
        total_excess=total_port - total_bench,
        by_sector=by_sector,
    )


@dataclass(frozen=True)
class SlippageReport:
    """Realized vs modeled execution cost. The earliest edge-decay warning.

    When realized slippage drifts above the model, the backtest is no longer
    describing the strategy being run -- and it drifts before returns do.
    """

    realized_bps: float
    modeled_bps: float
    n_fills: int

    @property
    def excess_bps(self) -> float:
        return self.realized_bps - self.modeled_bps

    @property
    def is_degrading(self) -> bool:
        return self.n_fills >= 20 and self.excess_bps > self.modeled_bps


def slippage_report(
    realized_bps: list[float], modeled_bps: float
) -> SlippageReport:
    vals = [v for v in realized_bps if v is not None]
    return SlippageReport(
        realized_bps=float(np.mean(vals)) if vals else 0.0,
        modeled_bps=modeled_bps,
        n_fills=len(vals),
    )
