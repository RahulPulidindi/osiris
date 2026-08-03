"""Macro calendar and trading blackout windows.

CPI, FOMC, and NFP days dominate everything else that happens in the market.
Entering a new position into a scheduled macro print is an uncompensated
variance bet that swamps the daily alpha the strategy is trying to harvest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo


class EventKind(str, Enum):
    CPI = "CPI"
    FOMC = "FOMC"
    NFP = "NFP"
    PPI = "PPI"
    GDP = "GDP"
    JOLTS = "JOLTS"
    OTHER = "OTHER"


# Blackout minutes before/after each event, by importance.
DEFAULT_WINDOWS: dict[EventKind, tuple[int, int]] = {
    EventKind.FOMC: (180, 120),   # rate decisions reprice everything
    EventKind.CPI: (120, 60),
    EventKind.NFP: (120, 60),
    EventKind.PPI: (60, 30),
    EventKind.GDP: (60, 30),
    EventKind.JOLTS: (30, 15),
    EventKind.OTHER: (30, 15),
}


@dataclass(frozen=True)
class MacroEvent:
    kind: EventKind
    at: datetime
    description: str = ""

    def blackout_window(self) -> tuple[datetime, datetime]:
        before, after = DEFAULT_WINDOWS.get(self.kind, (30, 15))
        at = self.at if self.at.tzinfo else self.at.replace(tzinfo=UTC)
        return at - timedelta(minutes=before), at + timedelta(minutes=after)

    def is_blackout(self, now: datetime) -> bool:
        start, stop = self.blackout_window()
        now = now if now.tzinfo else now.replace(tzinfo=UTC)
        return start <= now <= stop


class MacroCalendar:
    """Holds scheduled events and answers the blackout question."""

    def __init__(self, events: list[MacroEvent] | None = None) -> None:
        self.events = sorted(events or [], key=lambda e: e.at)

    def add(self, event: MacroEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.at)

    def active_blackout(self, now: datetime | None = None) -> MacroEvent | None:
        now = now or datetime.now(UTC)
        return next((e for e in self.events if e.is_blackout(now)), None)

    def is_blackout(self, now: datetime | None = None) -> tuple[bool, str]:
        event = self.active_blackout(now)
        if event is None:
            return False, ""
        return True, f"{event.kind.value} at {event.at.isoformat()}"

    def events_on(self, d: date) -> list[MacroEvent]:
        return [e for e in self.events if e.at.date() == d]

    def next_event(self, now: datetime | None = None) -> MacroEvent | None:
        now = now or datetime.now(UTC)
        return next((e for e in self.events if e.at > now), None)


# US market hours in Eastern Time, expressed as naive times for comparison.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Full-day US market holidays. Extend annually; a stale list silently permits
# trading on a closed day, so this is checked in tests.
MARKET_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Jr. Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    }
)


MARKET_TZ = ZoneInfo("America/New_York")


def market_now(now: datetime | None = None) -> datetime:
    """Current time in Eastern, which is the only timezone the market has."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(MARKET_TZ)


def session_date(now: datetime | None = None) -> date:
    """Today's SESSION date, in market time rather than UTC.

    Using `datetime.now(UTC).date()` is wrong for four hours of every weekday:
    after 8pm ET (7pm in winter) the UTC date has already rolled to tomorrow. Two
    consequences, both silent --

      1. A cycle run in the evening stamps tomorrow's date on its snapshot,
         journal entries, and equity marks.
      2. `OrderIntent.idempotency_key` embeds the date, so it changes mid-session:
         the same logical order gets two different keys, and duplicate protection
         quietly stops working across that boundary.
    """
    return market_now(now).date()


def is_market_holiday(d: date) -> bool:
    return d in MARKET_HOLIDAYS_2026


def is_trading_session(d: date) -> bool:
    return d.weekday() < 5 and not is_market_holiday(d)


def is_market_open(now: datetime | None = None) -> bool:
    """True only during regular trading hours on a trading day."""
    et = market_now(now)
    return is_trading_session(et.date()) and MARKET_OPEN <= et.time() <= MARKET_CLOSE


def describe_session(now: datetime | None = None) -> str:
    """Human-readable reason a session is or is not open.

    Exists because "not a trading session" left the operator to work out whether
    it was a weekend, a holiday, or simply after hours.
    """
    et = market_now(now)
    stamp = et.strftime("%a %b %d, %H:%M ET")
    if is_market_holiday(et.date()):
        return f"{stamp} — US market holiday"
    if et.weekday() >= 5:
        return f"{stamp} — weekend; next session {next_trading_day(et.date())}"
    if et.time() < MARKET_OPEN:
        return f"{stamp} — pre-market; opens 09:30 ET"
    if et.time() > MARKET_CLOSE:
        return f"{stamp} — after close; next session {next_trading_day(et.date())}"
    return f"{stamp} — market open"


def next_trading_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while not is_trading_session(nxt):
        nxt += timedelta(days=1)
    return nxt
