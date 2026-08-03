"""Evaluation gate tests.

The gates must reject a strategy with no edge and accept one with a real edge.
A gate that passes everything is worse than no gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from osiris.eval.gates import (
    cost_sensitivity_gate,
    factor_attribution_gate,
    funnel_fidelity_gate,
    monte_carlo_gate,
    run_all_gates,
)
from osiris.eval.metrics import deflated_sharpe, max_drawdown, sharpe, summarize

RNG = np.random.default_rng(42)


class TestMonteCarloGate:
    def test_rejects_no_skill_selection(self) -> None:
        """A random basket must NOT pass. This is the gate's whole purpose."""
        universe = RNG.normal(0.0005, 0.02, size=(250, 100))
        strategy = universe[:, :20].mean(axis=1)  # arbitrary fixed slice
        result = monte_carlo_gate(strategy, universe, n_holdings=20, n_trials=500)
        assert not result.passed, f"random selection should fail: {result.detail}"

    def test_accepts_genuine_skill(self) -> None:
        """A book that reliably holds the best names should pass."""
        universe = RNG.normal(0.0002, 0.02, size=(250, 100))
        # Simulate real foresight: hold the top performers each period.
        picks = np.argsort(-universe, axis=1)[:, :20]
        strategy = np.take_along_axis(universe, picks, axis=1).mean(axis=1)
        result = monte_carlo_gate(strategy, universe, n_holdings=20, n_trials=500)
        assert result.passed, result.detail
        assert result.statistic < 0.05

    def test_percentile_reported(self) -> None:
        universe = RNG.normal(0.0005, 0.02, size=(120, 60))
        strategy = universe[:, :10].mean(axis=1)
        r = monte_carlo_gate(strategy, universe, n_holdings=10, n_trials=200)
        assert 0.0 <= r.extra["percentile"] <= 1.0
        assert len(r.extra["null_distribution"]) == 200

    def test_rejects_mismatched_periods(self) -> None:
        universe = RNG.normal(0, 0.02, size=(100, 50))
        with pytest.raises(ValueError, match="periods"):
            monte_carlo_gate([0.01] * 50, universe, n_holdings=10, n_trials=10)

    def test_rejects_too_few_symbols(self) -> None:
        universe = RNG.normal(0, 0.02, size=(100, 5))
        with pytest.raises(ValueError, match="need >="):
            monte_carlo_gate([0.01] * 100, universe, n_holdings=20, n_trials=10)


class TestFactorAttributionGate:
    def test_rejects_pure_beta_book(self) -> None:
        """Long high-beta in a rising market is not a strategy.

        This is the most important rejection in the suite: it catches the book
        that looks brilliant purely because the market went up.
        """
        market = RNG.normal(0.0006, 0.012, size=300)
        strategy = 1.6 * market + RNG.normal(0.0, 0.001, size=300)  # no alpha
        r = factor_attribution_gate(strategy, {"market": market})
        assert not r.passed
        assert "beta" in r.detail.lower()

    def test_accepts_genuine_alpha_with_low_beta(self) -> None:
        market = RNG.normal(0.0004, 0.012, size=500)
        strategy = 0.7 * market + 0.0010 + RNG.normal(0.0, 0.002, size=500)
        r = factor_attribution_gate(strategy, {"market": market})
        assert r.passed, r.detail
        assert r.extra["alpha_t_stat"] >= 2.0
        assert abs(r.extra["loadings"]["market"] - 0.7) < 0.15

    def test_multifactor_loadings_recovered(self) -> None:
        n = 400
        f = {
            "market": RNG.normal(0.0004, 0.012, n),
            "size": RNG.normal(0.0, 0.008, n),
            "value": RNG.normal(0.0, 0.007, n),
            "momentum": RNG.normal(0.0, 0.009, n),
        }
        strategy = (
            0.9 * f["market"] - 0.4 * f["value"] + 0.0008 + RNG.normal(0, 0.002, n)
        )
        r = factor_attribution_gate(strategy, f)
        assert r.extra["loadings"]["value"] < -0.2
        assert 0.0 <= r.extra["r_squared"] <= 1.0

    def test_insufficient_data_fails_closed(self) -> None:
        r = factor_attribution_gate([0.01, 0.02], {"market": [0.01, 0.02]})
        assert not r.passed
        assert "insufficient" in r.detail


class TestFunnelFidelityGate:
    def test_high_overlap_passes(self) -> None:
        full = [f"S{i}" for i in range(20)]
        funnel = full[:17] + ["X1", "X2", "X3"]
        r = funnel_fidelity_gate(funnel, full, min_overlap=0.60)
        assert r.passed
        assert r.statistic == pytest.approx(0.85)

    def test_low_overlap_fails(self) -> None:
        """Funnel leakage: the pre-rank is discarding the names that matter."""
        full = [f"S{i}" for i in range(20)]
        funnel = [f"T{i}" for i in range(20)]
        r = funnel_fidelity_gate(funnel, full)
        assert not r.passed
        assert len(r.extra["missed_names"]) == 20

    def test_missing_reference_fails_closed(self) -> None:
        assert not funnel_fidelity_gate(["A"], []).passed


class TestCostSensitivityGate:
    def test_fragile_edge_rejected(self) -> None:
        """An edge that dies at 2x costs was an accounting artifact."""
        base = RNG.normal(0.0008, 0.01, 250)
        runs = {
            1.0: base,
            2.0: base - 0.0008,   # wiped out
            5.0: base - 0.0020,   # negative
        }
        assert not cost_sensitivity_gate(runs).passed

    def test_robust_edge_accepted(self) -> None:
        base = RNG.normal(0.0025, 0.008, 400)
        runs = {1.0: base, 2.0: base - 0.0002, 5.0: base - 0.0006}
        r = cost_sensitivity_gate(runs)
        assert r.passed, r.detail

    def test_missing_multiplier_fails_closed(self) -> None:
        assert not cost_sensitivity_gate({1.0: [0.01] * 10}).passed


class TestAggregate:
    def test_all_must_pass(self) -> None:
        from osiris.eval.gates import GateResult

        ok = GateResult("a", True, 1.0, "fine")
        bad = GateResult("b", False, 0.0, "broken")
        passed, msg = run_all_gates([ok, bad])
        assert not passed
        assert "index fund" in msg, "failure guidance should be explicit"

    def test_all_pass_message(self) -> None:
        from osiris.eval.gates import GateResult

        passed, msg = run_all_gates([GateResult("a", True, 1.0, "fine")])
        assert passed and "passed" in msg


class TestMetrics:
    def test_deflated_sharpe_penalizes_search(self) -> None:
        """Testing 500 variants inflates the best Sharpe by chance alone."""
        raw = 2.5
        one = deflated_sharpe(raw, n_trials=1, n_periods=250)
        many = deflated_sharpe(raw, n_trials=500, n_periods=250)
        assert one == raw
        assert many < raw

    def test_max_drawdown_positive_fraction(self) -> None:
        returns = np.array([0.1, -0.2, -0.1, 0.05])
        mdd = max_drawdown(returns)
        assert 0 < mdd < 1

    def test_sharpe_zero_for_flat(self) -> None:
        assert sharpe(np.zeros(100)) == 0.0

    def test_summary_fields(self) -> None:
        s = summarize(RNG.normal(0.0005, 0.01, 300))
        assert s.n_periods == 300
        assert s.volatility > 0
