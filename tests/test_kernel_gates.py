"""Risk Kernel gate tests.

The kernel is the only thing standing between a captured model and the account,
so these tests are the most consequential in the project. Every gate is tested
for both the veto and the pass, and critically for the asymmetry between
entries and exits.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from osiris.kernel.state import BreakerState
from osiris.types import Side, VetoCode
from tests.conftest import (
    NOW,
    diversified_positions,
    make_intent,
    make_portfolio,
    make_position,
    make_state,
)


class TestKillSwitch:
    def test_blocks_entry(self, kernel) -> None:
        state = make_state(kill_switch_engaged=True)
        d = kernel.evaluate(make_intent(), state)
        assert not d.approved
        assert VetoCode.KILL_SWITCH in d.vetoes

    def test_permits_exit(self, kernel) -> None:
        """A halted agent must still be able to close positions.

        Freezing while holding losers is worse than never having traded.
        """
        book = diversified_positions(20)
        state = make_state(make_portfolio(positions=book), kill_switch_engaged=True)
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="rank_exit"), state
        )
        assert VetoCode.KILL_SWITCH not in d.vetoes


class TestBreakers:
    def test_blocks_entry_when_tripped(self, kernel) -> None:
        state = make_state(
            breakers=BreakerState(tripped=(VetoCode.BREAKER_TRIPPED,), reasons=("daily loss",))
        )
        d = kernel.evaluate(make_intent(), state)
        assert not d.approved
        assert VetoCode.BREAKER_TRIPPED in d.vetoes

    def test_permits_risk_exit_when_tripped(self, kernel) -> None:
        book = diversified_positions(20)
        state = make_state(
            make_portfolio(positions=book),
            breakers=BreakerState(tripped=(VetoCode.BREAKER_TRIPPED,), reasons=("dd",)),
        )
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="risk_exit"), state
        )
        assert d.approved, d.notes


class TestMacroBlackout:
    def test_blocks_entry(self, kernel) -> None:
        state = make_state(macro_blackout=True, macro_blackout_reason="FOMC")
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.MACRO_BLACKOUT in d.vetoes
        assert any("FOMC" in n for n in d.notes)

    def test_permits_exit(self, kernel) -> None:
        book = diversified_positions(20)
        state = make_state(make_portfolio(positions=book), macro_blackout=True)
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="rank_exit"), state
        )
        assert VetoCode.MACRO_BLACKOUT not in d.vetoes


class TestEarningsBlackout:
    def test_blocks_entry_inside_window(self, kernel) -> None:
        state = make_state(earnings={"AAPL": NOW + timedelta(hours=12)})
        d = kernel.evaluate(make_intent("AAPL"), state)
        assert VetoCode.EARNINGS_BLACKOUT in d.vetoes

    def test_permits_entry_outside_window(self, kernel) -> None:
        state = make_state(earnings={"AAPL": NOW + timedelta(hours=100)})
        d = kernel.evaluate(make_intent("AAPL"), state)
        assert VetoCode.EARNINGS_BLACKOUT not in d.vetoes

    def test_ignores_past_earnings(self, kernel) -> None:
        state = make_state(earnings={"AAPL": NOW - timedelta(hours=5)})
        d = kernel.evaluate(make_intent("AAPL"), state)
        assert VetoCode.EARNINGS_BLACKOUT not in d.vetoes

    def test_permits_exit_into_earnings(self, kernel) -> None:
        """De-risking before a report must never be blocked."""
        book = diversified_positions(20)
        state = make_state(
            make_portfolio(positions=book),
            earnings={book[0].symbol: NOW + timedelta(hours=2)},
        )
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="risk_exit"), state
        )
        assert d.approved, d.notes


class TestInvalidationRequirement:
    def test_entry_without_invalidation_vetoed(self, kernel) -> None:
        state = make_state()
        intent = make_intent(invalidation="")
        d = kernel.evaluate(intent, state)
        assert VetoCode.MISSING_INVALIDATION in d.vetoes

    def test_whitespace_invalidation_rejected(self, kernel) -> None:
        state = make_state()
        d = kernel.evaluate(make_intent(invalidation="   \n  "), state)
        assert VetoCode.MISSING_INVALIDATION in d.vetoes

    def test_exit_needs_no_invalidation(self, kernel) -> None:
        book = diversified_positions(20)
        state = make_state(make_portfolio(positions=book))
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="rank_exit"), state
        )
        assert VetoCode.MISSING_INVALIDATION not in d.vetoes


class TestDataQualityFailsClosed:
    def test_unknown_tradability_vetoed(self, kernel) -> None:
        state = make_state()
        d = kernel.evaluate(make_intent("UNKNOWN_SYM"), state)
        assert VetoCode.NOT_TRADABLE in d.vetoes

    def test_not_tradable_vetoed(self, kernel) -> None:
        state = make_state(tradable=False)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.NOT_TRADABLE in d.vetoes

    def test_stale_quote_vetoed(self, kernel) -> None:
        state = make_state(quote_age_s=600)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.STALE_DATA in d.vetoes

    def test_fresh_quote_passes(self, kernel) -> None:
        state = make_state(quote_age_s=10)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.STALE_DATA not in d.vetoes

    def test_missing_quote_vetoed(self, kernel) -> None:
        state = make_state(symbols=("MSFT",))
        d = kernel.evaluate(make_intent("AAPL"), state)
        assert VetoCode.STALE_DATA in d.vetoes


class TestSpreadGate:
    def test_wide_spread_blocks_entry(self, kernel) -> None:
        state = make_state(spread_bps=80.0)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.SPREAD_TOO_WIDE in d.vetoes

    def test_tight_spread_passes(self, kernel) -> None:
        state = make_state(spread_bps=2.0)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.SPREAD_TOO_WIDE not in d.vetoes

    def test_wide_spread_does_not_block_exit(self, kernel) -> None:
        """Exits matter most exactly when the market is ugly."""
        book = diversified_positions(20)
        state = make_state(make_portfolio(positions=book), spread_bps=200.0)
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="risk_exit"), state
        )
        assert VetoCode.SPREAD_TOO_WIDE not in d.vetoes


class TestOrderBudgetAndIdempotency:
    def test_budget_exhausted_blocks(self, kernel, limits) -> None:
        state = make_state(orders_placed_today=limits.daily_order_budget)
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.ORDER_BUDGET in d.vetoes

    def test_duplicate_key_blocked(self, kernel) -> None:
        intent = make_intent()
        state = make_state(used_idempotency_keys=frozenset({intent.idempotency_key}))
        d = kernel.evaluate(intent, state)
        assert VetoCode.DUPLICATE_ORDER in d.vetoes

    def test_idempotency_key_is_stable(self) -> None:
        a, b = make_intent(), make_intent()
        assert a.idempotency_key == b.idempotency_key

    def test_idempotency_key_differs_by_side(self) -> None:
        buy = make_intent(side=Side.BUY)
        sell = make_intent(side=Side.SELL, reason="rank_exit")
        assert buy.idempotency_key != sell.idempotency_key

    def test_idempotency_key_differs_by_notional(self) -> None:
        assert make_intent(notional=1000).idempotency_key != make_intent(notional=2000).idempotency_key


class TestMandatoryReview:
    def test_place_blocked_without_review(self, kernel) -> None:
        state = make_state()
        d = kernel.evaluate_before_place(make_intent(), state)
        assert not d.approved
        assert VetoCode.REVIEW_NOT_RUN in d.vetoes

    def test_place_allowed_after_review(self, kernel) -> None:
        intent = make_intent()
        state = make_state(reviewed=frozenset({intent.idempotency_key}))
        d = kernel.evaluate_before_place(intent, state)
        assert d.approved, d.notes

    def test_review_of_different_intent_does_not_authorize(self, kernel) -> None:
        """Reviewing AAPL must not authorize placing MSFT."""
        reviewed = make_intent("AAPL")
        other = make_intent("MSFT")
        state = make_state(
            symbols=("AAPL", "MSFT"), reviewed=frozenset({reviewed.idempotency_key})
        )
        d = kernel.evaluate_before_place(other, state)
        assert VetoCode.REVIEW_NOT_RUN in d.vetoes
