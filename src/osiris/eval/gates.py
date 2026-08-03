"""The four evaluation gates. All must pass before capital is committed.

This module exists to falsify the strategy cheaply. If these gates fail, the
correct outcome is to buy an index fund and keep the dashboard -- a real result,
not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from osiris.eval.metrics import sharpe, total_return


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    statistic: float
    detail: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- Gate 1
def monte_carlo_gate(
    strategy_returns: list[float] | np.ndarray,
    universe_returns: np.ndarray,
    *,
    n_holdings: int,
    n_trials: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
) -> GateResult:
    """Beat random same-sized portfolios from the same universe on the same dates.

    Beating the index proves nothing: an equal-weight basket of 20 names in a
    rising market beats a cap-weighted index routinely. The question is whether
    *this selection* beats an arbitrary one drawn from the same opportunity set.

    universe_returns: (n_periods, n_symbols) matrix of per-period returns.
    """
    strat = np.asarray(list(strategy_returns), dtype=float)
    if universe_returns.ndim != 2:
        raise ValueError("universe_returns must be 2-D (periods x symbols)")
    n_periods, n_symbols = universe_returns.shape
    if n_symbols < n_holdings:
        raise ValueError(f"universe has {n_symbols} symbols, need >= {n_holdings}")
    if strat.size != n_periods:
        raise ValueError(
            f"strategy has {strat.size} periods, universe has {n_periods}"
        )

    rng = np.random.default_rng(seed)
    observed = total_return(strat)

    null = np.empty(n_trials, dtype=float)
    for i in range(n_trials):
        # Fresh random basket each period, equal-weighted: matches the actual
        # portfolio on every dimension except which names were chosen.
        picks = rng.integers(0, n_symbols, size=(n_periods, n_holdings))
        basket = np.take_along_axis(universe_returns, picks, axis=1).mean(axis=1)
        null[i] = total_return(basket)

    percentile = float((null < observed).mean())
    p_value = 1.0 - percentile
    return GateResult(
        name="monte_carlo",
        passed=p_value < alpha,
        statistic=p_value,
        detail=(
            f"return {observed:.2%} at {percentile:.1%} percentile of {n_trials} "
            f"random {n_holdings}-name portfolios (p={p_value:.4f})"
        ),
        extra={
            "percentile": percentile,
            "observed_return": observed,
            "null_median": float(np.median(null)),
            "null_distribution": null.tolist(),
        },
    )


# ---------------------------------------------------------------- Gate 2
def factor_attribution_gate(
    strategy_returns: list[float] | np.ndarray,
    factors: dict[str, list[float] | np.ndarray],
    *,
    alpha_t_threshold: float = 2.0,
    max_beta: float = 1.2,
) -> GateResult:
    """Regress on Fama-French 5 + momentum. The number that matters is alpha.

    If the book is simply long high-beta growth in a rising market, that shows up
    as factor loadings rather than alpha, and it is not a strategy -- it is
    market exposure with extra steps. Both cited research systems ran beta below
    1.0, which is the evidence that selection was doing the work.
    """
    y = np.asarray(list(strategy_returns), dtype=float)
    names = list(factors)
    if not names:
        raise ValueError("at least one factor required (market)")

    X = np.column_stack([np.asarray(list(factors[n]), dtype=float) for n in names])
    if X.shape[0] != y.size:
        raise ValueError(f"factor rows {X.shape[0]} != strategy periods {y.size}")
    if y.size <= X.shape[1] + 1:
        return GateResult(
            name="factor_attribution",
            passed=False,
            statistic=0.0,
            detail=f"insufficient data: {y.size} periods for {X.shape[1]} factors",
        )

    design = np.column_stack([np.ones(y.size), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = y.size - design.shape[1]
    mse = float(resid @ resid) / dof
    try:
        cov = mse * np.linalg.inv(design.T @ design)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return GateResult(
            name="factor_attribution",
            passed=False,
            statistic=0.0,
            detail="singular design matrix (collinear factors)",
        )

    alpha_per_period = float(coef[0])
    alpha_t = float(coef[0] / se[0]) if se[0] > 0 else 0.0
    loadings = {n: float(c) for n, c in zip(names, coef[1:], strict=True)}
    market_beta = loadings.get("market", loadings.get(names[0], 0.0))

    alpha_significant = alpha_t >= alpha_t_threshold
    beta_ok = abs(market_beta) <= max_beta
    passed = alpha_significant and beta_ok

    reasons = []
    if not alpha_significant:
        reasons.append(f"alpha t={alpha_t:.2f} below {alpha_t_threshold}")
    if not beta_ok:
        reasons.append(f"market beta {market_beta:.2f} exceeds {max_beta}")

    return GateResult(
        name="factor_attribution",
        passed=passed,
        statistic=alpha_t,
        detail=(
            f"alpha {alpha_per_period * 252:.2%}/yr (t={alpha_t:.2f}), "
            f"beta {market_beta:.2f}"
            + ("; " + "; ".join(reasons) if reasons else "")
        ),
        extra={
            "alpha_per_period": alpha_per_period,
            "alpha_annualized": alpha_per_period * 252,
            "alpha_t_stat": alpha_t,
            "loadings": loadings,
            "r_squared": float(
                1.0 - (resid @ resid) / (((y - y.mean()) ** 2).sum() or 1.0)
            ),
        },
    )


# ---------------------------------------------------------------- Gate 3
def funnel_fidelity_gate(
    funnel_top: list[str],
    full_width_top: list[str],
    *,
    min_overlap: float = 0.60,
) -> GateResult:
    """Overlap between funnel output and a full-width deep pass.

    The published alpha came from BREADTH. If the cheap pre-rank discards a name
    the deep pass would have ranked top-N, the funnel destroys the very signal
    being paid for. Low fidelity means the funnel is the bug, not the model.
    """
    if not full_width_top:
        return GateResult(
            name="funnel_fidelity",
            passed=False,
            statistic=0.0,
            detail="no full-width reference pass supplied",
        )
    a, b = set(funnel_top), set(full_width_top)
    overlap = len(a & b) / len(b)
    return GateResult(
        name="funnel_fidelity",
        passed=overlap >= min_overlap,
        statistic=overlap,
        detail=(
            f"{len(a & b)}/{len(b)} overlap ({overlap:.0%}), "
            f"min {min_overlap:.0%}; missed: {sorted(b - a)[:8]}"
        ),
        extra={"missed_names": sorted(b - a), "overlap": overlap},
    )


# ---------------------------------------------------------------- Gate 4
def cost_sensitivity_gate(
    returns_by_multiplier: dict[float, list[float] | np.ndarray],
    *,
    required_multipliers: tuple[float, ...] = (2.0, 5.0),
    min_sharpe_at_2x: float = 0.5,
) -> GateResult:
    """Re-run at 2x and 5x assumed slippage.

    An edge that dies at 2x was never an edge, it was an accounting artifact.
    Real costs are always worse than modeled.
    """
    missing = [m for m in required_multipliers if m not in returns_by_multiplier]
    if missing:
        return GateResult(
            name="cost_sensitivity",
            passed=False,
            statistic=0.0,
            detail=f"missing runs at multipliers {missing}",
        )

    sharpes = {m: sharpe(np.asarray(list(r), dtype=float)) for m, r in returns_by_multiplier.items()}
    s2 = sharpes[2.0]
    s5 = sharpes[5.0]
    passed = s2 >= min_sharpe_at_2x and s5 > 0.0
    return GateResult(
        name="cost_sensitivity",
        passed=passed,
        statistic=s2,
        detail=(
            "sharpe " + ", ".join(f"{m}x={s:.2f}" for m, s in sorted(sharpes.items()))
            + f"; need 2x>={min_sharpe_at_2x} and 5x>0"
        ),
        extra={"sharpe_by_multiplier": {str(k): v for k, v in sharpes.items()}},
    )


def run_all_gates(results: list[GateResult]) -> tuple[bool, str]:
    """Aggregate. ALL must pass; there is no partial credit before capital."""
    failed = [r for r in results if not r.passed]
    if not failed:
        return True, f"All {len(results)} gates passed."
    lines = [f"{len(failed)}/{len(results)} gates FAILED:"]
    lines += [f"  {r.name}: {r.detail}" for r in failed]
    lines.append("")
    lines.append(
        "Do not commit capital. If these cannot be passed, buying a low-cost "
        "index fund is the rational outcome."
    )
    return False, "\n".join(lines)
