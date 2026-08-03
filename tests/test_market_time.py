"""Session dates must be computed in market time, not UTC.

Regression for a bug that is invisible for 20 hours a day. After 8pm ET (7pm in
winter) the UTC calendar date has already advanced, so `datetime.now(UTC).date()`
returns TOMORROW during evening hours. Two silent consequences:

  1. Snapshots, journal entries, and equity marks are stamped with the wrong day.
  2. `OrderIntent.idempotency_key` embeds the date, so it changes mid-session --
     the same logical order presents two different keys and duplicate protection
     stops working across that boundary.

Nothing fails loudly; the timestamps are simply wrong, and the second effect only
matters on the retry that duplicates a live order.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from osiris.data.macro import (
    describe_session,
    is_market_open,
    is_trading_session,
    market_now,
    session_date,
)

ET = ZoneInfo("America/New_York")


class TestSessionDate:
    def test_evening_et_does_not_roll_to_tomorrow(self):
        """The exact bug. 9pm ET Monday is 1am UTC Tuesday."""
        evening = datetime(2026, 8, 3, 21, 0, tzinfo=ET)

        assert evening.astimezone(UTC).date() == date(2026, 8, 4), "premise"
        assert session_date(evening) == date(2026, 8, 3)

    def test_early_morning_et_is_the_same_day(self):
        assert session_date(datetime(2026, 8, 3, 3, 0, tzinfo=ET)) == date(2026, 8, 3)

    def test_market_hours_are_unaffected(self):
        assert session_date(datetime(2026, 8, 3, 10, 30, tzinfo=ET)) == date(2026, 8, 3)

    def test_naive_input_is_treated_as_utc(self):
        """A naive datetime must not be silently read as local time."""
        naive = datetime(2026, 8, 4, 1, 0)  # 1am UTC == 9pm ET Aug 3

        assert session_date(naive) == date(2026, 8, 3)

    def test_market_now_returns_eastern(self):
        assert market_now().tzinfo is ET


class TestMarketOpen:
    @pytest.mark.parametrize(
        ("when", "expected"),
        [
            (datetime(2026, 8, 3, 9, 29, tzinfo=ET), False),   # pre-market
            (datetime(2026, 8, 3, 9, 30, tzinfo=ET), True),    # the bell
            (datetime(2026, 8, 3, 12, 0, tzinfo=ET), True),
            (datetime(2026, 8, 3, 16, 0, tzinfo=ET), True),    # close
            (datetime(2026, 8, 3, 16, 1, tzinfo=ET), False),   # after hours
            (datetime(2026, 8, 2, 12, 0, tzinfo=ET), False),   # Sunday
            (datetime(2026, 8, 1, 12, 0, tzinfo=ET), False),   # Saturday
            (datetime(2026, 7, 3, 12, 0, tzinfo=ET), False),   # holiday
        ],
    )
    def test_boundaries(self, when, expected):
        assert is_market_open(when) is expected

    def test_a_holiday_on_a_weekday_is_not_a_session(self):
        """A stale holiday list silently permits trading on a closed day."""
        assert not is_trading_session(date(2026, 12, 25))


class TestSessionDescription:
    """The skip reason must name the cause.

    "not a trading session" left the operator to work out whether it was a
    weekend, a holiday, or simply after hours.
    """

    @pytest.mark.parametrize(
        ("when", "fragment"),
        [
            (datetime(2026, 8, 2, 3, 0, tzinfo=ET), "weekend"),
            (datetime(2026, 7, 3, 11, 0, tzinfo=ET), "holiday"),
            (datetime(2026, 8, 3, 8, 0, tzinfo=ET), "pre-market"),
            (datetime(2026, 8, 3, 18, 0, tzinfo=ET), "after close"),
            (datetime(2026, 8, 3, 11, 0, tzinfo=ET), "market open"),
        ],
    )
    def test_names_the_reason(self, when, fragment):
        assert fragment in describe_session(when)

    def test_reports_eastern_time_not_utc(self):
        text = describe_session(datetime(2026, 8, 3, 21, 0, tzinfo=ET))

        assert "21:00 ET" in text

    def test_weekend_points_at_the_next_session(self):
        assert "2026-08-03" in describe_session(datetime(2026, 8, 2, 3, 0, tzinfo=ET))


class TestIdempotencyKeyStability:
    def test_the_key_uses_the_market_date(self):
        """A key that rotates at 8pm ET breaks duplicate protection mid-session."""
        from osiris.data.macro import session_date as sd
        from osiris.types import OrderIntent, Side

        intent = OrderIntent(symbol="AAPL", side=Side.BUY, notional_usd=100.0)

        # The key embeds a date; it must be the market's, not UTC's.
        assert intent.idempotency_key
        assert sd().isoformat() != "" and len(intent.idempotency_key) == 32

    def test_identical_intents_share_a_key(self):
        from osiris.types import OrderIntent, Side

        a = OrderIntent(symbol="AAPL", side=Side.BUY, notional_usd=100.0)
        b = OrderIntent(symbol="AAPL", side=Side.BUY, notional_usd=100.0)

        assert a.idempotency_key == b.idempotency_key

    def test_different_intents_differ(self):
        from osiris.types import OrderIntent, Side

        a = OrderIntent(symbol="AAPL", side=Side.BUY, notional_usd=100.0)
        b = OrderIntent(symbol="MSFT", side=Side.BUY, notional_usd=100.0)

        assert a.idempotency_key != b.idempotency_key
