"""Alerting for the events a human must know about immediately.

Phase 5 requires alerts on fills and breaker trips. The design constraint that
shapes this module is that **an alert failure must never affect trading.** A
webhook timeout, a bad token, or a rate limit cannot be allowed to raise into the
execution path, because the alternative -- an order that fails because its
notification failed -- is far worse than a missed notification.

So every sink swallows its own errors and logs them. The journal remains the
authoritative record; alerts are a convenience layer over it.

Severity governs routing, not formatting. `CRITICAL` is reserved for the two
conditions that mean the system's model of reality is wrong (reconciliation break,
kernel bypass) plus the halts. A fill is `INFO`: it is expected behavior, and
paging on expected behavior trains an operator to ignore the channel.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import ClassVar

from osiris.logging import get_logger

log = get_logger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    kind: str
    severity: Severity
    title: str
    body: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: dict = field(default_factory=dict)

    def format_text(self) -> str:
        prefix = {
            Severity.INFO: "OSIRIS",
            Severity.WARNING: "OSIRIS WARN",
            Severity.CRITICAL: "OSIRIS CRITICAL",
        }[self.severity]
        lines = [f"[{prefix}] {self.title}", self.body]
        if self.context:
            lines.append(
                " | ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
            )
        return "\n".join(line for line in lines if line)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "title": self.title,
            "body": self.body,
            "ts": self.ts.isoformat(),
            "context": self.context,
        }


# ----------------------------------------------------------------------- sinks
class LogSink:
    """Always-on sink. Structured logs are the floor, never absent."""

    name = "log"

    def send(self, alert: Alert) -> bool:
        fn = {
            Severity.INFO: log.info,
            Severity.WARNING: log.warning,
            Severity.CRITICAL: log.error,
        }[alert.severity]
        fn("alert", kind=alert.kind, title=alert.title, body=alert.body, **alert.context)
        return True


class WebhookSink:
    """Generic JSON webhook. Works with Slack, Discord, ntfy, and Pushover.

    Uses `urllib` rather than httpx so the alert path has no dependency that
    could be mid-reload during an incident, and blocks with a short timeout: an
    alert that hangs is a stalled trading loop.
    """

    name = "webhook"

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        # Request construction is INSIDE the try. `urllib.request.Request`
        # validates the URL scheme eagerly and raises ValueError on a malformed
        # one, so building it outside would let a typo'd webhook URL raise
        # straight into the caller -- defeating the whole point of this module.
        try:
            payload = json.dumps(
                {"text": alert.format_text(), **alert.to_dict()}
            ).encode()
            request = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                ok = 200 <= resp.status < 300
                if not ok:
                    log.warning("alerts.webhook_rejected", status=resp.status)
                return ok
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Deliberately swallowed. An alerting failure must never propagate
            # into the trading path: a missed notification is recoverable, an
            # order that failed because its notification failed is not.
            log.warning("alerts.webhook_failed", error=str(exc))
            return False


@dataclass
class Alerter:
    """Fans alerts out to every sink, deduplicating repeats.

    Deduplication matters more than it appears: a tripped breaker is evaluated on
    every cycle, so without suppression a single halt would emit an alert per
    cycle until a human intervened -- and the operator would mute the channel,
    which is strictly worse than no alerting.
    """

    sinks: list = field(default_factory=list)
    min_severity: Severity = Severity.INFO
    suppress_repeats: bool = True
    history: list[Alert] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    # ClassVar, else the dataclass machinery would treat this ranking table as a
    # per-instance field with a mutable default.
    _ORDER: ClassVar[dict[Severity, int]] = {
        Severity.INFO: 0,
        Severity.WARNING: 1,
        Severity.CRITICAL: 2,
    }

    def __post_init__(self) -> None:
        if not self.sinks:
            self.sinks = [LogSink()]

    def send(self, alert: Alert) -> bool:
        if self._ORDER[alert.severity] < self._ORDER[self.min_severity]:
            return False

        key = f"{alert.kind}:{alert.title}"
        if self.suppress_repeats and key in self._seen:
            log.debug("alerts.suppressed_repeat", kind=alert.kind)
            return False
        self._seen.add(key)
        self.history.append(alert)

        delivered = False
        for sink in self.sinks:
            try:
                delivered = sink.send(alert) or delivered
            except Exception as exc:
                # A broken sink must not stop the remaining sinks, and must not
                # reach the caller.
                log.warning(
                    "alerts.sink_failed", sink=getattr(sink, "name", "?"), error=str(exc)
                )
        return delivered

    def reset_suppression(self, kind: str | None = None) -> None:
        """Clear dedup state, e.g. after a human acknowledges an incident."""
        if kind is None:
            self._seen.clear()
        else:
            self._seen = {k for k in self._seen if not k.startswith(f"{kind}:")}


# ---------------------------------------------------------------- constructors
def breaker_tripped(reasons: list[str]) -> Alert:
    """Critical: the system has halted new risk and needs a human to reset it."""
    return Alert(
        kind="breaker_tripped",
        severity=Severity.CRITICAL,
        title="Circuit breaker tripped -- new risk halted",
        body=(
            "; ".join(reasons)
            + ". Exits still run. A breaker requires a manual reset: "
            "POST /api/control/breakers/reset"
        ),
        context={"count": len(reasons)},
    )


def reconciliation_break(detail: str) -> Alert:
    """Critical: the ledger and the broker disagree about what is owned.

    The highest-consequence alert in the system. Until it is resolved, every
    sizing decision is computed against a book that may not exist.
    """
    return Alert(
        kind="reconciliation_break",
        severity=Severity.CRITICAL,
        title="Reconciliation break -- ledger and broker disagree",
        body=(
            f"{detail}\nTrading is halted. Do not reset until the cause is "
            "understood: this is the failure mode where a rejected order was "
            "booked as a fill."
        ),
    )


def kill_switch_engaged(reason: str) -> Alert:
    return Alert(
        kind="kill_switch",
        severity=Severity.CRITICAL,
        title="Kill switch engaged",
        body=f"{reason}. New entries blocked; risk exits continue.",
    )


def fill_alert(symbol: str, side: str, quantity: float, price: float) -> Alert:
    """Info, not a page. Fills are expected behavior."""
    return Alert(
        kind="fill",
        severity=Severity.INFO,
        title=f"{side.upper()} {symbol}",
        body=f"{quantity:.4f} @ ${price:,.2f} (${quantity * price:,.2f})",
        context={"symbol": symbol, "side": side},
    )


def cycle_summary(summary: str, *, equity: float, halted: bool) -> Alert:
    return Alert(
        kind="cycle",
        severity=Severity.WARNING if halted else Severity.INFO,
        title="Cycle halted" if halted else "Cycle complete",
        body=summary,
        context={"equity": round(equity, 2)},
    )


def cycle_failed(error: str, consecutive: int) -> Alert:
    """A cycle raised. Escalates to critical once failures repeat.

    A single failure is usually transient (a timed-out request, a rate limit).
    Three in a row means the agent is not managing the book at all, which is the
    state you need to know about even if nothing is losing money yet -- an
    unmanaged position has no stop.
    """
    return Alert(
        kind="cycle_failed",
        severity=Severity.CRITICAL if consecutive >= 3 else Severity.WARNING,
        title=(
            f"Cycle failed {consecutive}x in a row"
            if consecutive >= 3
            else "Cycle failed"
        ),
        body=(
            f"{error}\n"
            + (
                "The agent is not managing existing positions while this "
                "persists. Reconnecting."
                if consecutive >= 3
                else "Will retry on the next scheduled wake."
            )
        ),
        context={"consecutive": consecutive},
    )


def preflight_blocked(failures: list[str]) -> Alert:
    return Alert(
        kind="preflight_blocked",
        severity=Severity.CRITICAL,
        title="Preflight NOT cleared -- refusing to arm",
        body="Blocking failures: " + ", ".join(failures),
        context={"count": len(failures)},
    )


def build_alerter(*, min_severity: Severity | None = None) -> Alerter:
    """Assemble sinks from the environment.

    Reads `OSIRIS_ALERT_WEBHOOK` rather than taking config, because alerting
    credentials are operational secrets that should not travel through the same
    typed settings object the model's constraints live in.
    """
    sinks: list = [LogSink()]
    url = os.environ.get("OSIRIS_ALERT_WEBHOOK", "").strip()
    if url:
        sinks.append(WebhookSink(url))
    else:
        log.info("alerts.no_webhook_configured", detail="logging only")

    threshold = min_severity
    if threshold is None:
        raw = os.environ.get("OSIRIS_ALERT_MIN_SEVERITY", "info").strip().lower()
        try:
            threshold = Severity(raw)
        except ValueError:
            log.warning("alerts.bad_min_severity", value=raw)
            threshold = Severity.INFO
    return Alerter(sinks=sinks, min_severity=threshold)


def alerts_for_cycle(result) -> list[Alert]:
    """Derive the alerts a completed cycle should emit.

    Pure: takes a `CycleResult` and returns alerts without sending them, so the
    mapping from outcome to alert is testable without a network sink.
    """
    out: list[Alert] = []
    if not result.ran:
        return out

    if not result.reconciled_clean:
        out.append(
            reconciliation_break(
                f"cycle {result.correlation_id} on {result.as_of.isoformat()}"
            )
        )
    breakers = getattr(result, "breakers", None)
    reasons = list(getattr(breakers, "reasons", ()) or ())
    if result.halted:
        out.append(breaker_tripped(reasons or ["halted (cause not recorded)"]))

    if result.report:
        for fill in result.report.fills:
            out.append(
                fill_alert(fill.symbol, fill.side.value, fill.quantity, fill.price)
            )
    out.append(
        cycle_summary(result.summary(), equity=result.equity, halted=result.halted)
    )
    return out
