"""Ledger accounting and reconciliation.

These matter more than they look. The ledger is what every risk limit is computed
against, so a ledger that drifts from the broker makes every downstream gate wrong
in the same direction while still reporting "approved."
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from osiris.execution.ledger import DailyPnL, Ledger
from osiris.types import Fill, Side

TS = datetime(2026, 7, 31, 14, 30, tzinfo=UTC)


def fill(
    symbol: str,
    side: Side,
    quantity: float,
    price: float,
    *,
    key: str = "",
    order_id: str = "o1",
    ts: datetime | None = None,
) -> Fill:
    return Fill(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        ts=ts or TS,
        order_id=order_id,
        idempotency_key=key,
    )


class TestFillAccounting:
    def test_buy_establishes_position_and_debits_cash(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        assert led.quantity_of("AAPL") == 10
        assert led.cash == 9_000.0
        assert led.positions["AAPL"].avg_cost == 100.0

    def test_average_cost_blends_across_buys(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="a"))
        led.apply_fill(fill("AAPL", Side.BUY, 10, 120.0, key="b"))

        assert led.quantity_of("AAPL") == 20
        assert led.positions["AAPL"].avg_cost == 110.0

    def test_sell_realizes_pnl_against_average_cost(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="a"))
        led.apply_fill(fill("AAPL", Side.SELL, 10, 130.0, key="b"))

        assert led.quantity_of("AAPL") == 0
        assert led.realized_pnl == 300.0
        assert led.cash == 10_300.0

    def test_full_exit_clears_cost_basis(self):
        """A residual basis on a closed position corrupts the next entry."""
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="a"))
        led.apply_fill(fill("AAPL", Side.SELL, 10, 90.0, key="b"))

        pos = led.positions["AAPL"]
        assert pos.quantity == 0.0
        assert pos.cost_basis_total == 0.0
        assert "AAPL" not in led.open_symbols

    def test_partial_sell_keeps_remaining_basis(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="a"))
        led.apply_fill(fill("AAPL", Side.SELL, 4, 150.0, key="b"))

        assert led.quantity_of("AAPL") == 6
        assert led.realized_pnl == 200.0
        assert led.positions["AAPL"].avg_cost == 100.0

    def test_oversell_is_clamped_to_holding(self):
        """A broker echo claiming more shares than held must not go negative.

        Long-only: a negative quantity is a short position the venue does not
        even support, so clamping is the only correct behavior.
        """
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 5, 100.0, key="a"))
        led.apply_fill(fill("AAPL", Side.SELL, 50, 110.0, key="b"))

        assert led.quantity_of("AAPL") == 0.0

    def test_duplicate_fill_is_ignored(self):
        """A retry returning the same fill must not double-count."""
        led = Ledger(starting_cash=10_000.0)
        first = led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="same-key"))
        second = led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="same-key"))

        assert first is True
        assert second is False
        assert led.quantity_of("AAPL") == 10
        assert led.cash == 9_000.0


class TestPortfolioProjection:
    def test_equity_is_cash_plus_marked_positions(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        assert led.equity({"AAPL": 120.0}) == 10_000.0 + 200.0

    def test_unrealized_pnl_tracks_marks(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        assert led.unrealized_pnl({"AAPL": 115.0}) == 150.0

    def test_to_portfolio_excludes_closed_positions(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0, key="a"))
        led.apply_fill(fill("MSFT", Side.BUY, 5, 200.0, key="b"))
        led.apply_fill(fill("AAPL", Side.SELL, 10, 100.0, key="c"))

        pf = led.to_portfolio({"AAPL": 100.0, "MSFT": 200.0})
        assert [p.symbol for p in pf.positions] == ["MSFT"]

    def test_metadata_flows_into_portfolio(self):
        """Sector and beta must reach the kernel or its exposure gates are blind."""
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))
        led.set_metadata({"AAPL": "Technology"}, {"AAPL": 1.3})

        pf = led.to_portfolio({"AAPL": 100.0})
        assert pf.positions[0].sector == "Technology"
        assert pf.positions[0].beta == 1.3


class TestReconciliation:
    def test_matching_state_is_clean(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        result = led.reconcile(
            {"AAPL": 10.0}, broker_equity=led.equity({"AAPL": 100.0}),
            prices={"AAPL": 100.0},
        )
        assert result.clean
        assert result.divergences == ()

    def test_phantom_position_is_detected(self):
        """We believe we hold it; the broker does not. The isError-200 failure."""
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        result = led.reconcile({}, broker_equity=10_000.0, prices={"AAPL": 100.0})
        assert not result.clean
        assert result.divergences[0].kind == "phantom_position"
        assert result.divergences[0].symbol == "AAPL"

    def test_unrecorded_position_is_detected(self):
        """The broker holds something we never booked. Equally dangerous."""
        led = Ledger(starting_cash=10_000.0)

        result = led.reconcile({"TSLA": 3.0}, broker_equity=10_900.0, prices={"TSLA": 300.0})
        assert not result.clean
        assert result.divergences[0].kind == "unrecorded_position"

    def test_quantity_mismatch_is_detected_with_delta(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        result = led.reconcile(
            {"AAPL": 7.0}, broker_equity=led.equity({"AAPL": 100.0}),
            prices={"AAPL": 100.0},
        )
        div = result.divergences[0]
        assert div.kind == "quantity_mismatch"
        assert div.delta == -3.0

    def test_float_noise_is_not_a_divergence(self):
        led = Ledger(starting_cash=10_000.0)
        led.apply_fill(fill("AAPL", Side.BUY, 10, 100.0))

        result = led.reconcile(
            {"AAPL": 10.0 + 1e-9}, broker_equity=led.equity({"AAPL": 100.0}),
            prices={"AAPL": 100.0},
        )
        assert result.clean

    def test_equity_divergence_beyond_tolerance_fails(self):
        led = Ledger(starting_cash=10_000.0)

        result = led.reconcile({}, broker_equity=12_000.0, prices={})
        assert not result.equity_matches
        assert not result.clean

    def test_equity_within_tolerance_passes(self):
        led = Ledger(starting_cash=10_000.0)

        result = led.reconcile({}, broker_equity=10_020.0, prices={}, equity_tolerance=0.005)
        assert result.equity_matches


class TestDailyPnL:
    """Session accounting.

    Written around the REAL call sequence -- `roll_day` at the open,
    `close_session` at the close -- because that is what the loop does and
    because a shortcut here is what let a dead breaker ship.
    """

    @staticmethod
    def run_sessions(pnl: DailyPnL, closes: list[float], *, start_day: int = 5) -> None:
        """Replay consecutive sessions with the given closing equities.

        Ordering matters and is easy to get wrong: `roll_day` attributes P&L to
        the session it is CLOSING, so a session's result is only counted when the
        next one opens. Opening and closing at the same value in a single step
        would shift every result one session late.
        """
        for i, close in enumerate(closes):
            pnl.roll_day(close, as_of=date(2026, 1, start_day + i))
            pnl.close_session(close)
        # Open one more session so the final close is attributed.
        pnl.roll_day(closes[-1], as_of=date(2026, 1, start_day + len(closes)))

    def test_first_roll_sets_baseline_without_counting_a_loss(self):
        pnl = DailyPnL()
        pnl.roll_day(100_000.0, as_of=date(2026, 1, 5))

        assert pnl.day_start_equity == 100_000.0
        assert pnl.consecutive_losses == 0
        assert pnl.history == [], "a fresh account has no completed session yet"

    def test_consecutive_losses_accumulate(self):
        pnl = DailyPnL()
        self.run_sessions(pnl, [100_000.0, 99_000.0, 98_000.0])

        assert pnl.consecutive_losses == 2

    def test_a_winning_day_resets_the_streak(self):
        """The breaker counts CONSECUTIVE losses; one win must clear it."""
        pnl = DailyPnL()
        self.run_sessions(pnl, [100_000.0, 99_000.0, 101_000.0])

        assert pnl.consecutive_losses == 0

    def test_peak_equity_is_monotonic(self):
        """Drawdown is measured from the peak, so the peak must never fall."""
        pnl = DailyPnL()
        pnl.roll_day(100_000.0, as_of=date(2026, 1, 5))
        pnl.mark(120_000.0)
        pnl.mark(90_000.0)

        assert pnl.peak_equity == 120_000.0

    def test_day_start_is_the_prior_close_not_the_current_equity(self):
        """The overnight gap must be inside the day's P&L.

        This book trades once at the open, so equity barely moves during a
        session. Anchoring `day_start_equity` intra-session would report only
        slippage as the daily P&L -- and the daily-loss breaker reads that same
        number, so a large gap down would register as roughly zero.
        """
        pnl = DailyPnL()
        pnl.roll_day(100_000.0, as_of=date(2026, 1, 5))
        pnl.close_session(100_000.0)

        # Next session opens after a -5% overnight gap.
        pnl.roll_day(95_000.0, as_of=date(2026, 1, 6))

        assert pnl.day_start_equity == 100_000.0
        assert (95_000.0 - pnl.day_start_equity) / pnl.day_start_equity == pytest.approx(-0.05)

    def test_is_new_session_distinguishes_the_open_session(self):
        pnl = DailyPnL()
        pnl.roll_day(100_000.0, as_of=date(2026, 1, 5))

        assert not pnl.is_new_session(date(2026, 1, 5))
        assert pnl.is_new_session(date(2026, 1, 6))

    def test_history_uses_the_session_date_not_wall_clock(self):
        """A replay must not stamp every simulated session with today's date."""
        pnl = DailyPnL()
        self.run_sessions(pnl, [100_000.0, 99_000.0])

        assert [d for d, _ in pnl.history] == ["2026-01-05", "2026-01-06"]


def test_wash_sale_window_uses_calendar_days():
    """Documents the 30-day rule the tax accounting depends on."""
    from osiris.eval.backtest import TaxAccount

    tax = TaxAccount()
    d0 = TS.date()
    tax.buy("AAPL", 10, 100.0, d0)
    tax.sell("AAPL", 10, 90.0, d0 + timedelta(days=1))
    tax.buy("AAPL", 10, 92.0, d0 + timedelta(days=10))

    assert tax.wash_sale_count == 1
