"""Offline strategy validation against a simulated market.

Deliberately separate from anything the dashboard reads. This module fabricates
prices, which is legitimate for falsifying a strategy and unacceptable for
displaying to an operator -- so it is only ever invoked by the CLI gate, never by
the API.

The distinction matters: `python -m osiris.runner.gate` answers "does this
strategy survive contact with costs and randomness?", while the dashboard answers
"what did the agent just do with my money?". Mixing simulated numbers into the
second question is how someone ends up trusting a figure that was invented.
"""

from __future__ import annotations

import contextlib

from osiris.eval.gates import (
    cost_sensitivity_gate,
    factor_attribution_gate,
    monte_carlo_gate,
    run_all_gates,
)
from osiris.eval.metrics import deflated_sharpe, summarize
from osiris.logging import get_logger
from osiris.runner.paper import PaperRunner
from osiris.runner.synthetic import SyntheticMarket

log = get_logger(__name__)


async def seed(state, *, sessions: int, seed_value: int) -> None:
    """Run simulated sessions and compute the evaluation gates over the result."""
    market = SyntheticMarket(n_symbols=120, n_days=320, seed=seed_value)
    runner = PaperRunner(state=state, market=market)

    log.info("backtest.running", sessions=sessions, symbols=len(market.symbols))
    results = await runner.run(sessions)
    if not results:
        log.warning("backtest.no_sessions")
        return

    returns = state.daily_returns
    bench = state.benchmark_returns
    if len(returns) < 10:
        log.warning("backtest.too_few_returns", n=len(returns))
        return

    n = min(len(returns), len(bench))
    strategy = returns[-n:]
    benchmark = bench[-n:]

    gates = []
    start = max(runner.warmup_days, 130)
    universe_matrix = _universe_return_matrix(market, start, n)
    if universe_matrix is not None:
        with contextlib.suppress(ValueError):
            gates.append(
                monte_carlo_gate(
                    strategy,
                    universe_matrix,
                    n_holdings=state.limits.target_position_count,
                    n_trials=2_000,
                )
            )

    with contextlib.suppress(ValueError):
        gates.append(factor_attribution_gate(strategy, {"market": benchmark}))

    modeled_drag = 0.0002
    gates.append(
        cost_sensitivity_gate(
            {
                1.0: strategy,
                2.0: [r - modeled_drag for r in strategy],
                5.0: [r - modeled_drag * 4 for r in strategy],
            }
        )
    )

    all_passed, verdict = run_all_gates(gates)
    perf = summarize(strategy)
    mc = next((g for g in gates if g.name == "monte_carlo"), None)

    state.evaluation = {
        "gates": [
            {
                "name": g.name,
                "passed": g.passed,
                "statistic": g.statistic,
                "detail": g.detail,
            }
            for g in gates
        ],
        "all_passed": all_passed,
        "verdict": verdict,
        "sharpe": perf.sharpe,
        "deflated_sharpe": deflated_sharpe(
            perf.sharpe, n_trials=8, n_periods=perf.n_periods
        ),
        "sortino": perf.sortino if perf.sortino != float("inf") else 0.0,
        "max_drawdown": perf.max_drawdown,
        "total_return": perf.total_return,
        "after_tax_return": perf.total_return * 0.63,
        "cagr": perf.cagr,
        "win_rate": perf.win_rate,
        "monte_carlo_percentile": mc.extra.get("percentile", 0.0) if mc else 0.0,
        "monte_carlo_p_value": mc.statistic if mc else 1.0,
        "null_distribution": mc.extra.get("null_distribution", []) if mc else [],
        "observed_return": mc.extra.get("observed_return", 0.0) if mc else 0.0,
        "walk_forward": [],
        "equity_curve": [],
        "funnel_fidelity": 0.0,
        "cost_sensitivity": next(
            (
                g.extra.get("sharpe_by_multiplier", {})
                for g in gates
                if g.name == "cost_sensitivity"
            ),
            {},
        ),
    }
    log.info("backtest.complete", sessions=len(results), gates_passed=all_passed)


def _universe_return_matrix(market: SyntheticMarket, start: int, n: int):
    """(periods x symbols) return matrix over the same dates as the strategy."""
    import numpy as np

    rows = []
    for i in range(start, start + n):
        if i <= 0 or i >= market.n_days:
            continue
        prev = np.array([market.closes[s][i - 1] for s in market.symbols])
        curr = np.array([market.closes[s][i] for s in market.symbols])
        rows.append(curr / prev - 1.0)
    return np.array(rows) if len(rows) == n else None
