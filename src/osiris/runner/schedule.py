"""When the agent should wake up.

The naive loop -- `run_cycle(); sleep(3600)` -- is wrong in both directions. It
burns LLM budget researching a market that is closed, and it drifts: a cycle
started at 09:31 puts the next at 10:31, then 11:31, so the agent never trades
the open again after its first day.

This module answers one question: *how long until the next moment worth waking
for?* Everything is computed in market time, because that is the only timezone
the exchange has.

Design decision worth stating: the schedule is a pure function of the clock. It
performs no I/O and holds no state, so "what would it do at 3am on a holiday?"
is answerable in a test rather than by waiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from osiris.data.macro import (
    MARKET_CLOSE,
    MARKET_OPEN,
    is_trading_session,
    market_now,
)
from osiris.logging import get_logger

log = get_logger(__name__)

# Trade shortly AFTER the bell, not at it. The first minutes of a session carry
# the widest spreads of the day as overnight orders clear; entering into that is
# paying the spread gate's worst case for no reason. Ten minutes is enough for
# the book to settle while still being "the open" in any meaningful sense.
OPEN_OFFSET = timedelta(minutes=10)

# Stop opening new positions before the close so an order has time to fill and
# reconcile within the session. An unfilled order at 16:00 becomes tomorrow's
# problem.
CLOSE_BUFFER = timedelta(minutes=20)


@dataclass(frozen=True)
class Wake:
    """A scheduled wake-up: when, and whether to trade on arrival."""

    at: datetime
    should_trade: bool
    reason: str

    def seconds_from(self, now: datetime | None = None) -> float:
        now = market_now(now)
        return max(0.0, (self.at - now).total_seconds())


def _at(day: datetime, t: time) -> datetime:
    return day.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)


def next_wake(now: datetime | None = None, *, interval_minutes: int = 0) -> Wake:
    """The next moment the agent should act.

    Three cases:

    1. Inside the tradable window -> wake now (or after `interval_minutes` if an
       intraday cadence is configured).
    2. Before the open today, on a trading day -> wake just after the bell.
    3. Otherwise (after the close, weekend, holiday) -> wake at the next
       session's open.

    `interval_minutes=0` means one cycle per session, at the open. That is the
    correct default for this strategy: it rebalances a daily-ranked book, so
    running it hourly multiplies cost and turnover without adding signal.
    """
    et = market_now(now)
    today_open = _at(et, MARKET_OPEN) + OPEN_OFFSET
    today_last = _at(et, MARKET_CLOSE) - CLOSE_BUFFER

    if is_trading_session(et.date()):
        if et < today_open:
            return Wake(today_open, True, "waiting for the opening bell")
        if et <= today_last:
            if interval_minutes > 0:
                nxt = et + timedelta(minutes=interval_minutes)
                if nxt <= today_last:
                    return Wake(nxt, True, f"next intraday cycle in {interval_minutes}m")
            else:
                return Wake(et, True, "market open")

    # After the close, or not a trading day at all. Walk forward to the next
    # session rather than adding 24h, so holidays and weekends are skipped
    # correctly rather than producing a wake on a closed day.
    probe = et
    for _ in range(10):
        probe = _at(probe + timedelta(days=1), MARKET_OPEN)
        if is_trading_session(probe.date()):
            return Wake(
                probe + OPEN_OFFSET,
                True,
                f"market closed; next session {probe.date().isoformat()}",
            )
    # Ten consecutive non-trading days is not a real calendar. Fail loudly rather
    # than sleeping forever on a stale holiday table.
    raise RuntimeError(
        "No trading session found within 10 days. The market holiday table in "
        "osiris/data/macro.py is probably stale."
    )


def describe_wait(seconds: float) -> str:
    """Human-readable duration, for a log line an operator will actually read."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def market_status(now: datetime | None = None) -> dict:
    """Current schedule state, for the dashboard and for logs."""
    et = market_now(now)
    wake = next_wake(et)
    return {
        "market_time": et.isoformat(),
        "is_trading_day": is_trading_session(et.date()),
        "next_wake": wake.at.isoformat(),
        "next_wake_in": describe_wait(wake.seconds_from(et)),
        "reason": wake.reason,
    }


def utc_now() -> datetime:
    return datetime.now(UTC)
