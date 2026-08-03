"""Go-live arming checks.

These tests matter more than their line count suggests: preflight is the last
thing between a defect and real money. The failure mode they exist to prevent is
a check that *reports* pass while the underlying condition is unverified -- an
arming gate that always says yes is worse than none, because it is trusted.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from osiris.api.state import build_runtime_state
from osiris.config import AccountType, Mode, RiskLimits, Settings
from osiris.execution.broker import PaperBroker
from osiris.execution.journal import EventType, Journal
from osiris.execution.killswitch import KillSwitch
from osiris.kernel.state import BreakerState
from osiris.runner.preflight import (
    MIN_RETURN_SAMPLES,
    Severity,
    check_account_type,
    check_breakers_clear,
    check_evaluation_gates,
    check_explicit_arming,
    check_kill_switch_clear,
    check_kill_switch_writable,
    check_mcp_snapshot,
    check_no_fabricated_returns,
    check_no_kernel_bypasses,
    check_no_reconciliation_breaks,
    check_paper_matches_backtest,
    check_return_sample_size,
    check_risk_limits_coherent,
    check_vetoes_are_visible,
    run_preflight,
)
from osiris.types import VetoCode


@pytest.fixture
def journal(tmp_path: pathlib.Path) -> Journal:
    return Journal(tmp_path / "journal.jsonl", fsync=False)


@pytest.fixture
def state(tmp_path: pathlib.Path):
    st = build_runtime_state(
        settings=Settings(
            mode=Mode.PAPER,
            account_type=AccountType.MARGIN,
            account_equity_usd=100_000.0,
        ),
        limits=RiskLimits(),
        journal_path=tmp_path / "journal.jsonl",
        broker=PaperBroker(starting_cash=100_000.0),
    )
    st.killswitch = KillSwitch(tmp_path / "KILL_SWITCH")
    return st


def healthy_journal(journal: Journal) -> Journal:
    """A journal that reflects a clean, productive paper run."""
    for i in range(40):
        journal.append(
            EventType.CYCLE_END, {"as_of": f"2025-09-{(i % 28) + 1:02d}", "equity": 100.0}
        )
        journal.append(EventType.RECONCILIATION, {"clean": True})
        journal.append(EventType.INTENT_EMITTED, {"symbol": "AAA"})
        journal.append(EventType.REVIEW_PASSED, {"symbol": "AAA"})
        journal.append(EventType.ORDER_PLACED, {"symbol": "AAA"})
        journal.append(EventType.FILL, {"symbol": "AAA"})
    return journal


class TestConfigChecks:
    def test_unknown_account_type_blocks(self):
        check = check_account_type(Settings(account_type=AccountType.UNKNOWN))

        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_known_account_type_passes(self):
        assert check_account_type(Settings(account_type=AccountType.CASH)).passed

    def test_paper_mode_needs_no_arming(self):
        assert check_explicit_arming(Settings(mode=Mode.PAPER)).passed

    def test_live_without_acknowledgement_blocks(self):
        """Mode alone must not arm real money."""
        settings = Settings(
            mode=Mode.LIVE,
            account_type=AccountType.MARGIN,
            account_equity_usd=1_000.0,
            i_understand_the_risk="no",
        )
        check = check_explicit_arming(settings)

        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_live_with_acknowledgement_passes(self):
        settings = Settings(
            mode=Mode.LIVE,
            account_type=AccountType.MARGIN,
            account_equity_usd=1_000.0,
            i_understand_the_risk="yes",
        )

        assert check_explicit_arming(settings).passed

    def test_default_limits_are_coherent(self):
        assert check_risk_limits_coherent(RiskLimits()).passed

    def test_per_order_cap_above_symbol_cap_is_incoherent(self):
        """A per-order cap wider than the position cap makes the gates disagree."""
        limits = RiskLimits(max_trade_notional_pct=0.15, max_symbol_weight=0.10)
        check = check_risk_limits_coherent(limits)

        assert not check.passed
        assert "per-order cap exceeds" in check.detail


class TestSafetyChecks:
    def test_engaged_kill_switch_blocks(self, tmp_path):
        ks = KillSwitch(tmp_path / "KILL_SWITCH")
        ks.engage("manual halt")

        check = check_kill_switch_clear(ks)
        assert not check.passed
        assert "manual halt" in check.detail

    def test_clear_kill_switch_passes(self, tmp_path):
        assert check_kill_switch_clear(KillSwitch(tmp_path / "KILL_SWITCH")).passed

    def test_writability_is_actually_probed(self, tmp_path):
        """A switch you cannot engage during an incident is not a switch."""
        ks = KillSwitch(tmp_path / "KILL_SWITCH")

        assert check_kill_switch_writable(ks).passed
        # The probe must not leave anything behind, or it becomes a phantom halt.
        assert not ks.path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_unwritable_kill_switch_path_blocks(self, tmp_path):
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        ks = KillSwitch(readonly / "KILL_SWITCH")
        readonly.chmod(0o500)
        try:
            check = check_kill_switch_writable(ks)
        finally:
            readonly.chmod(0o700)

        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_tripped_breaker_blocks(self):
        breakers = BreakerState().trip(VetoCode.BREAKER_TRIPPED, "daily loss 3.2%")
        check = check_breakers_clear(breakers)

        assert not check.passed
        assert "daily loss 3.2%" in check.detail

    def test_clear_breakers_pass(self):
        assert check_breakers_clear(BreakerState()).passed

    def test_missing_mcp_snapshot_blocks(self, tmp_path):
        check = check_mcp_snapshot(tmp_path / "absent.json")

        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_snapshot_with_tools_passes(self, tmp_path):
        path = tmp_path / "snap.json"
        path.write_text(json.dumps({"tool_count": 42, "captured_at": "2026-01-01"}))

        assert check_mcp_snapshot(path).passed

    def test_empty_snapshot_blocks(self, tmp_path):
        """Zero tools means drift detection has no baseline."""
        path = tmp_path / "snap.json"
        path.write_text(json.dumps({"tool_count": 0}))

        assert not check_mcp_snapshot(path).passed


class TestReconciliationGate:
    """The Phase 5 gate: zero reconciliation breaks. No exceptions."""

    def test_clean_run_passes(self, journal):
        healthy_journal(journal)

        assert check_no_reconciliation_breaks(journal).passed

    def test_a_single_break_blocks(self, journal):
        """One break means the ledger may describe a book that does not exist."""
        healthy_journal(journal)
        journal.append(EventType.RECONCILIATION_BREAK, {"clean": False})

        check = check_no_reconciliation_breaks(journal)
        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_never_reconciling_is_not_a_pass(self, journal):
        """Zero breaks with zero reconciliations is absence of evidence.

        The naive implementation -- `breaks == 0` -- would clear a system that
        has never once compared its ledger to a broker.
        """
        journal.append(EventType.CYCLE_END, {"as_of": "2025-09-01"})

        assert not check_no_reconciliation_breaks(journal).passed


class TestKernelBypassGate:
    def test_reviews_at_least_matching_placements_passes(self, journal):
        healthy_journal(journal)

        assert check_no_kernel_bypasses(journal).passed

    def test_more_placements_than_reviews_is_a_bypass(self, journal):
        """The structural invariant. If this can break, the kernel is advisory."""
        journal.append(EventType.REVIEW_PASSED, {})
        for _ in range(3):
            journal.append(EventType.ORDER_PLACED, {})

        check = check_no_kernel_bypasses(journal)
        assert not check.passed
        assert check.severity is Severity.BLOCKING
        assert "unsimulated" in check.detail


class TestVetoVisibility:
    def test_a_productive_run_passes(self, journal):
        healthy_journal(journal)

        assert check_vetoes_are_visible(journal).passed

    def test_intents_with_zero_fills_blocks(self, journal):
        """The exact failure this build hit: 280 intents, zero fills.

        Every unit test passed while the assembled system did nothing, because
        the planner and the kernel disagreed on every single order.
        """
        for _ in range(50):
            journal.append(EventType.INTENT_EMITTED, {"symbol": "AAA"})
            journal.append(EventType.KERNEL_VETO, {"vetoes": ["notional_cap"]})

        check = check_vetoes_are_visible(journal)
        assert not check.passed
        assert check.severity is Severity.BLOCKING
        assert "ZERO fills" in check.detail

    def test_no_intents_at_all_blocks(self, journal):
        journal.append(EventType.CYCLE_END, {"as_of": "2025-09-01"})

        assert not check_vetoes_are_visible(journal).passed


class TestEvaluationGates:
    def test_missing_evaluation_blocks(self):
        check = check_evaluation_gates(None)

        assert not check.passed
        assert "unfalsified" in check.detail

    def test_all_required_gates_passing_clears(self):
        evaluation = {
            "gates": [
                {"name": "monte_carlo", "passed": True},
                {"name": "factor_attribution", "passed": True},
                {"name": "cost_sensitivity", "passed": True},
            ],
            "monte_carlo_percentile": 0.98,
            "sharpe": 1.5,
        }

        assert check_evaluation_gates(evaluation).passed

    def test_a_failing_gate_blocks(self):
        evaluation = {
            "gates": [
                {"name": "monte_carlo", "passed": False},
                {"name": "factor_attribution", "passed": True},
                {"name": "cost_sensitivity", "passed": True},
            ]
        }
        check = check_evaluation_gates(evaluation)

        assert not check.passed
        assert "monte_carlo" in check.detail

    def test_a_missing_gate_blocks_even_when_the_rest_pass(self):
        """A gate that silently stopped running must not lower the bar.

        Checking only that present gates passed would clear a harness whose
        Monte Carlo test had been dropped -- the most important gate of the four,
        since it is the one that separates selection from beta.
        """
        evaluation = {
            "gates": [
                {"name": "factor_attribution", "passed": True},
                {"name": "cost_sensitivity", "passed": True},
            ]
        }
        check = check_evaluation_gates(evaluation)

        assert not check.passed
        assert "monte_carlo" in check.detail
        assert "never ran" in check.detail


class TestReturnSeriesIntegrity:
    def test_sufficient_samples_pass(self):
        assert check_return_sample_size([0.001] * MIN_RETURN_SAMPLES).passed

    def test_tiny_sample_warns_without_blocking(self):
        check = check_return_sample_size([0.01, 0.02])

        assert not check.passed
        assert check.severity is Severity.ADVISORY

    def test_plausible_returns_pass(self):
        assert check_no_fabricated_returns([0.01, -0.02, 0.005]).passed

    def test_a_fabricated_total_loss_blocks(self):
        """A -100% day on a long-only book is an accounting artifact.

        It comes from a zero equity mark on a skipped session and silently
        corrupts Sharpe, drawdown, and every gate computed downstream.
        """
        check = check_no_fabricated_returns([0.01, -1.0, 0.005])

        assert not check.passed
        assert check.severity is Severity.BLOCKING

    def test_nan_returns_block(self):
        assert not check_no_fabricated_returns([0.01, float("nan")]).passed


class TestPaperVsBacktest:
    def test_similar_sharpe_passes(self):
        rng = __import__("numpy").random.default_rng(3)
        base = list(rng.normal(0.001, 0.01, 120))

        assert check_paper_matches_backtest(base, base).passed

    def test_wide_divergence_warns(self):
        """If backtest is great and paper is bad, the backtest is wrong."""
        import numpy as np

        rng = np.random.default_rng(5)
        paper = list(rng.normal(-0.002, 0.02, 200))
        backtest = list(rng.normal(0.004, 0.005, 200))

        check = check_paper_matches_backtest(paper, backtest)
        assert not check.passed
        assert check.severity is Severity.ADVISORY

    def test_missing_backtest_is_advisory_not_blocking(self):
        """A missing comparison is a known gap, not evidence of a defect."""
        check = check_paper_matches_backtest([0.01] * 50, None)

        assert not check.passed
        assert check.severity is Severity.ADVISORY


class TestAggregateReport:
    """The whole-report behavior. Default must be refusal."""

    def test_a_fresh_unproven_system_is_not_cleared(self, state, tmp_path):
        """Nothing has been run, so nothing has been proven."""
        report = run_preflight(state, snapshot_path=tmp_path / "absent.json")

        assert not report.armed
        assert report.blocking_failures

    def test_advisories_alone_do_not_block(self, state, tmp_path):
        """Advisory warnings are judgement calls, not hard stops."""
        healthy_journal(state.journal)
        state.evaluation = {
            "gates": [
                {"name": "monte_carlo", "passed": True},
                {"name": "factor_attribution", "passed": True},
                {"name": "cost_sensitivity", "passed": True},
            ]
        }
        state.daily_returns = [0.001] * 60
        snapshot = tmp_path / "snap.json"
        snapshot.write_text(json.dumps({"tool_count": 30}))

        report = run_preflight(state, snapshot_path=snapshot)

        assert report.armed, report.describe()
        # No backtest was supplied, so at least one advisory must be surfaced
        # rather than silently absorbed by a passing aggregate.
        assert any(c.name == "paper_matches_backtest" for c in report.advisories)

    def test_a_check_that_raises_counts_as_a_failure(self, state, tmp_path):
        """An inconclusive safety check is not a passing one.

        If an exception propagated instead, a crashing preflight would be
        indistinguishable from one that was never run.
        """

        class Exploding:
            path = tmp_path / "KILL_SWITCH"

            def check(self):
                raise OSError("cannot stat kill switch")

        state.killswitch = Exploding()
        report = run_preflight(state, snapshot_path=tmp_path / "absent.json")

        assert not report.armed
        failed = {c.name for c in report.blocking_failures}
        assert "kill_switch_clear" in failed
        assert any("OSError" in c.detail for c in report.checks)

    def test_an_empty_report_is_never_armed(self):
        from osiris.runner.preflight import PreflightReport

        assert not PreflightReport().armed

    def test_report_serializes_for_the_api(self, state, tmp_path):
        payload = run_preflight(state, snapshot_path=tmp_path / "absent.json").to_dict()

        assert payload["armed"] is False
        assert payload["checks"]
        assert all({"name", "passed", "severity", "detail"} <= set(c) for c in payload["checks"])

    def test_describe_refuses_capital_when_blocked(self, state, tmp_path):
        text = run_preflight(state, snapshot_path=tmp_path / "absent.json").describe()

        assert "NOT CLEARED" in text
        assert "Do not commit capital" in text
