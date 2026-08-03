"""Alerting.

The invariant under test throughout: **an alerting failure must never affect
trading.** A webhook that times out, returns 500, or raises must be absorbed. The
inverse -- an order that fails because its notification failed -- is a far worse
outcome than a missed notification, so every sink swallows its own errors.
"""

from __future__ import annotations

import pytest

from osiris.runner.alerts import (
    Alert,
    Alerter,
    LogSink,
    Severity,
    WebhookSink,
    alerts_for_cycle,
    breaker_tripped,
    build_alerter,
    fill_alert,
    kill_switch_engaged,
    reconciliation_break,
)


class CapturingSink:
    name = "capture"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


class ExplodingSink:
    name = "exploding"

    def send(self, alert: Alert) -> bool:
        raise RuntimeError("webhook is down")


class TestSeverityRouting:
    def test_reconciliation_break_is_critical(self):
        """The system's model of reality is wrong. Nothing ranks higher."""
        assert reconciliation_break("AAPL: ledger=10 broker=0").severity is Severity.CRITICAL

    def test_breaker_trip_is_critical(self):
        assert breaker_tripped(["daily loss 3.1%"]).severity is Severity.CRITICAL

    def test_kill_switch_is_critical(self):
        assert kill_switch_engaged("manual halt").severity is Severity.CRITICAL

    def test_a_fill_is_only_info(self):
        """Fills are expected behavior.

        Paging on expected behavior trains an operator to mute the channel, which
        loses the critical alerts too.
        """
        assert fill_alert("AAPL", "buy", 10.0, 150.0).severity is Severity.INFO

    def test_below_threshold_alerts_are_dropped(self):
        sink = CapturingSink()
        alerter = Alerter(sinks=[sink], min_severity=Severity.CRITICAL)

        assert not alerter.send(fill_alert("AAPL", "buy", 1.0, 100.0))
        assert alerter.send(breaker_tripped(["halt"]))
        assert len(sink.sent) == 1


class TestFailureIsolation:
    def test_a_raising_sink_does_not_propagate(self):
        """An alerting failure must never reach the trading path."""
        alerter = Alerter(sinks=[ExplodingSink()])

        # Must not raise.
        assert alerter.send(breaker_tripped(["halt"])) is False

    def test_a_broken_sink_does_not_starve_the_others(self):
        good = CapturingSink()
        alerter = Alerter(sinks=[ExplodingSink(), good])

        assert alerter.send(breaker_tripped(["halt"]))
        assert len(good.sent) == 1

    def test_webhook_absorbs_a_connection_failure(self):
        """An unreachable host returns False rather than raising."""
        sink = WebhookSink("http://127.0.0.1:1/never-listening", timeout=0.25)

        assert sink.send(breaker_tripped(["halt"])) is False

    def test_webhook_absorbs_a_malformed_url(self):
        assert WebhookSink("not-a-url").send(fill_alert("A", "buy", 1.0, 1.0)) is False


class TestDeduplication:
    def test_repeats_are_suppressed(self):
        """A tripped breaker is re-evaluated every cycle.

        Without suppression a single halt emits one alert per cycle until a human
        intervenes, and the operator mutes the channel -- strictly worse than no
        alerting at all.
        """
        sink = CapturingSink()
        alerter = Alerter(sinks=[sink])

        assert alerter.send(breaker_tripped(["daily loss 3.1%"]))
        assert not alerter.send(breaker_tripped(["daily loss 3.1%"]))
        assert len(sink.sent) == 1

    def test_distinct_alerts_are_not_suppressed(self):
        sink = CapturingSink()
        alerter = Alerter(sinks=[sink])

        alerter.send(fill_alert("AAPL", "buy", 1.0, 100.0))
        alerter.send(fill_alert("MSFT", "buy", 1.0, 100.0))

        assert len(sink.sent) == 2

    def test_suppression_can_be_reset_after_acknowledgement(self):
        sink = CapturingSink()
        alerter = Alerter(sinks=[sink])
        alerter.send(breaker_tripped(["halt"]))

        alerter.reset_suppression("breaker_tripped")

        assert alerter.send(breaker_tripped(["halt"]))
        assert len(sink.sent) == 2

    def test_suppression_can_be_disabled(self):
        sink = CapturingSink()
        alerter = Alerter(sinks=[sink], suppress_repeats=False)

        alerter.send(breaker_tripped(["halt"]))
        alerter.send(breaker_tripped(["halt"]))

        assert len(sink.sent) == 2


class TestFormatting:
    def test_critical_alerts_are_visibly_marked(self):
        text = reconciliation_break("AAPL mismatch").format_text()

        assert "CRITICAL" in text
        assert "AAPL mismatch" in text

    def test_breaker_alert_states_the_manual_reset_path(self):
        """An alert that does not say what to do is only noise."""
        body = breaker_tripped(["drawdown 10.2%"]).body

        assert "reset" in body.lower()
        assert "drawdown 10.2%" in body

    def test_breaker_alert_states_that_exits_continue(self):
        """Halt means take no NEW risk, never abandon existing risk."""
        assert "exits" in breaker_tripped(["halt"]).body.lower()

    def test_context_is_rendered(self):
        text = fill_alert("AAPL", "buy", 10.0, 150.0).format_text()

        assert "AAPL" in text
        assert "1,500" in text


class TestBuildAlerter:
    def test_defaults_to_logging_only(self, monkeypatch):
        monkeypatch.delenv("OSIRIS_ALERT_WEBHOOK", raising=False)

        alerter = build_alerter()

        assert len(alerter.sinks) == 1
        assert isinstance(alerter.sinks[0], LogSink)

    def test_webhook_is_added_when_configured(self, monkeypatch):
        monkeypatch.setenv("OSIRIS_ALERT_WEBHOOK", "https://example.invalid/hook")

        alerter = build_alerter()

        assert any(isinstance(s, WebhookSink) for s in alerter.sinks)

    def test_a_bad_severity_falls_back_rather_than_crashing(self, monkeypatch):
        """Misconfigured alerting must not prevent the process from starting."""
        monkeypatch.delenv("OSIRIS_ALERT_WEBHOOK", raising=False)
        monkeypatch.setenv("OSIRIS_ALERT_MIN_SEVERITY", "nonsense")

        assert build_alerter().min_severity is Severity.INFO

    def test_logging_sink_is_always_present(self, monkeypatch):
        """Structured logs are the floor and must never be absent."""
        monkeypatch.setenv("OSIRIS_ALERT_WEBHOOK", "https://example.invalid/hook")

        assert any(isinstance(s, LogSink) for s in build_alerter().sinks)


class TestAlertsForCycle:
    """Mapping a cycle outcome to alerts, without sending anything."""

    @pytest.fixture
    def cycle(self):
        import datetime as dt

        from osiris.execution.loop import CycleResult

        return CycleResult(
            correlation_id="abc123",
            as_of=dt.date(2026, 3, 2),
            ran=True,
            equity=101_000.0,
        )

    def test_a_skipped_cycle_emits_nothing(self):
        import datetime as dt

        from osiris.execution.loop import CycleResult

        result = CycleResult(
            correlation_id="x", as_of=dt.date(2026, 3, 1), ran=False, reason="weekend"
        )

        assert alerts_for_cycle(result) == []

    def test_a_clean_cycle_emits_only_a_summary(self, cycle):
        alerts = alerts_for_cycle(cycle)

        assert [a.kind for a in alerts] == ["cycle"]
        assert alerts[0].severity is Severity.INFO

    def test_a_reconciliation_break_emits_a_critical(self, cycle):
        import dataclasses

        alerts = alerts_for_cycle(dataclasses.replace(cycle, reconciled_clean=False))

        critical = [a for a in alerts if a.kind == "reconciliation_break"]
        assert len(critical) == 1
        assert critical[0].severity is Severity.CRITICAL

    def test_a_halt_emits_a_breaker_alert(self, cycle):
        import dataclasses

        alerts = alerts_for_cycle(dataclasses.replace(cycle, halted=True))

        assert any(a.kind == "breaker_tripped" for a in alerts)

    def test_a_halt_without_recorded_reasons_still_alerts(self, cycle):
        """Never swallow a halt just because its cause was not recorded."""
        import dataclasses

        alerts = alerts_for_cycle(dataclasses.replace(cycle, halted=True))
        breaker = next(a for a in alerts if a.kind == "breaker_tripped")

        assert breaker.body.strip()

    def test_fills_are_reported_individually(self, cycle):
        import dataclasses
        from datetime import UTC, datetime

        from osiris.execution.executor import ExecutionReport
        from osiris.types import Fill, Side

        report = ExecutionReport(
            fills=[
                Fill(
                    symbol=sym,
                    side=Side.BUY,
                    quantity=2.0,
                    price=100.0,
                    ts=datetime.now(UTC),
                    order_id=f"o-{sym}",
                )
                for sym in ("AAA", "BBB")
            ]
        )
        alerts = alerts_for_cycle(dataclasses.replace(cycle, report=report))

        fills = [a for a in alerts if a.kind == "fill"]
        assert {a.context["symbol"] for a in fills} == {"AAA", "BBB"}
