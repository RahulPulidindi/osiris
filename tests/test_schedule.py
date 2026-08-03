"""When the agent wakes.

The bug this module exists to prevent is DRIFT. A plain `sleep(interval)` loop
starts a cycle at 09:31, the next at 10:31, then 11:31 -- so after the first day
the agent never trades the open again. It also spends LLM budget researching a
closed market, which is money for nothing.

Every case here is a pure function of the clock, so "what does it do at 3am on
Thanksgiving?" is answerable in a test rather than by waiting until November.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from osiris.runner.schedule import (
    CLOSE_BUFFER,
    OPEN_OFFSET,
    describe_wait,
    market_status,
    next_wake,
)

ET = ZoneInfo("America/New_York")


def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


class TestWakeDuringTheSession:
    def test_before_the_bell_waits_for_the_open(self):
        wake = next_wake(at(2026, 8, 3, 9, 29))

        assert wake.at.hour == 9
        assert wake.at.minute == 30 + int(OPEN_OFFSET.total_seconds() // 60)
        assert wake.should_trade

    def test_it_waits_past_the_bell_not_at_it(self):
        """The first minutes carry the widest spreads of the day.

        Entering into that pays the spread gate's worst case for no reason.
        """
        wake = next_wake(at(2026, 8, 3, 9, 0))

        assert wake.at > at(2026, 8, 3, 9, 30)

    def test_inside_the_window_wakes_immediately(self):
        now = at(2026, 8, 3, 11, 0)

        assert next_wake(now).seconds_from(now) == 0

    def test_after_the_close_buffer_defers_to_tomorrow(self):
        """An order placed at 15:59 has no time to fill and reconcile."""
        close_minus = at(2026, 8, 3, 16, 0) - CLOSE_BUFFER
        wake = next_wake(close_minus.replace(minute=close_minus.minute + 1))

        assert wake.at.date() == at(2026, 8, 4, 9).date()


class TestWakeOutsideTheSession:
    def test_after_the_close_targets_the_next_morning(self):
        wake = next_wake(at(2026, 8, 3, 16, 30))

        assert wake.at.date().isoformat() == "2026-08-04"
        assert wake.at.hour == 9

    def test_friday_evening_skips_the_weekend(self):
        """The specific case a naive `+1 day` gets wrong."""
        wake = next_wake(at(2026, 7, 31, 17, 0))

        assert wake.at.date().isoformat() == "2026-08-03"
        assert wake.at.weekday() == 0

    def test_saturday_targets_monday(self):
        assert next_wake(at(2026, 8, 1, 12)).at.date().isoformat() == "2026-08-03"

    def test_sunday_targets_monday(self):
        assert next_wake(at(2026, 8, 2, 12)).at.date().isoformat() == "2026-08-03"

    def test_a_holiday_is_skipped(self):
        """Thanksgiving 2026 falls on a Thursday; the next session is Friday."""
        wake = next_wake(at(2026, 11, 26, 10))

        assert wake.at.date().isoformat() == "2026-11-27"

    def test_christmas_skips_to_the_next_open_day(self):
        """Dec 25 2026 is a Friday, so the next session is the following Monday."""
        wake = next_wake(at(2026, 12, 25, 10))

        assert wake.at.date().isoformat() == "2026-12-28"

    def test_the_reason_is_stated(self):
        """A silent sleep is indistinguishable from a hang."""
        assert "next session" in next_wake(at(2026, 8, 2, 12)).reason


class TestIntradayCadence:
    def test_zero_means_once_per_session(self):
        """The right default for a daily-ranked book.

        Hourly cycles multiply cost and turnover without adding signal.
        """
        now = at(2026, 8, 3, 11, 0)

        assert next_wake(now, interval_minutes=0).seconds_from(now) == 0

    def test_an_interval_schedules_the_next_cycle(self):
        now = at(2026, 8, 3, 11, 0)
        wake = next_wake(now, interval_minutes=60)

        assert wake.at == at(2026, 8, 3, 12, 0)

    def test_an_interval_does_not_run_past_the_close(self):
        """The last intraday cycle must leave time to fill and reconcile."""
        wake = next_wake(at(2026, 8, 3, 15, 50), interval_minutes=60)

        assert wake.at.date().isoformat() == "2026-08-04"


class TestNoDrift:
    def test_the_open_is_hit_every_day_not_progressively_later(self):
        """The core regression.

        Simulate a week of "cycle ran, now what?" and assert every wake lands at
        the same time each morning. A fixed-interval loop would walk forward by
        an hour a day.
        """
        wakes = []
        probe = at(2026, 8, 3, 9, 45)
        for _ in range(5):
            # Pretend the cycle finished and the day is done.
            wake = next_wake(probe.replace(hour=16, minute=30))
            wakes.append(wake.at)
            probe = wake.at

        assert len({(w.hour, w.minute) for w in wakes}) == 1
        assert [w.weekday() for w in wakes] == [1, 2, 3, 4, 0]


class TestDurationFormatting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0s"), (45, "45s"), (600, "10m"), (7200, "2.0h"), (300_000, "3.5d")],
    )
    def test_reads_naturally(self, seconds, expected):
        assert describe_wait(seconds) == expected


class TestMarketStatus:
    def test_reports_the_schedule_for_the_dashboard(self):
        status = market_status(at(2026, 8, 2, 12))

        assert status["is_trading_day"] is False
        assert "next_wake_in" in status
        assert status["reason"]

    def test_a_trading_day_is_reported_as_such(self):
        assert market_status(at(2026, 8, 3, 11))["is_trading_day"] is True
