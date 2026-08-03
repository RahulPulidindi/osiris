"""The order pipeline. The most important tests after the kernel itself.

What is being proven here is structural, not behavioral: there must be **no path**
from an intent to the venue that skips review or skips the kernel. Behavior can be
fixed later; a missing gate places a real order.
"""

from __future__ import annotations

import pytest

from osiris.config import AccountType, RiskLimits
from osiris.execution.broker import (
    Broker,
    OrderRejected,
    PlaceResult,
    ReviewResult,
)
from osiris.execution.executor import Executor
from osiris.execution.journal import EventType, Journal
from osiris.execution.ledger import Ledger
from osiris.execution.mcp_broker import AmbiguousOrderState
from osiris.kernel.kernel import RiskKernel
from osiris.types import Fill, Side, VetoCode
from tests.conftest import NOW, make_intent, make_state


class SpyBroker(Broker):
    """Records the call sequence so ordering invariants can be asserted."""

    def __init__(
        self,
        *,
        review_ok: bool = True,
        place_ok: bool = True,
        raise_ambiguous: bool = False,
        fill_price: float = 100.0,
        positions: dict[str, float] | None = None,
        equity: float = 100_000.0,
    ) -> None:
        self.calls: list[str] = []
        self.review_ok = review_ok
        self.place_ok = place_ok
        self.raise_ambiguous = raise_ambiguous
        self.fill_price = fill_price
        self._positions = positions if positions is not None else {}
        self._equity = equity
        self.placed_keys: list[str] = []

    @property
    def name(self) -> str:
        return "spy"

    async def review(self, request):
        self.calls.append(f"review:{request.symbol}")
        if not self.review_ok:
            return ReviewResult(False, message="simulated review rejection")
        qty = request.notional_usd / self.fill_price
        return ReviewResult(
            True, estimated_price=self.fill_price, estimated_quantity=qty
        )

    async def place(self, request):
        self.calls.append(f"place:{request.symbol}")
        self.placed_keys.append(request.idempotency_key)
        if self.raise_ambiguous:
            raise AmbiguousOrderState("transport died after send")
        if not self.place_ok:
            raise OrderRejected("simulated venue rejection")
        qty = request.notional_usd / self.fill_price
        fill = Fill(
            symbol=request.symbol,
            side=request.side,
            quantity=qty,
            price=self.fill_price,
            ts=NOW,
            order_id="spy-1",
            idempotency_key=request.idempotency_key,
            intended_price=request.limit_price or self.fill_price,
        )
        return PlaceResult("spy-1", True, fills=(fill,))

    async def get_positions(self):
        return dict(self._positions)

    async def get_account_equity(self):
        return self._equity


@pytest.fixture
def wiring(tmp_path, limits: RiskLimits):
    """An armed executor with a spy broker, fresh journal, and empty ledger."""

    def build(**broker_kwargs):
        broker = SpyBroker(**broker_kwargs)
        journal = Journal(tmp_path / "j.jsonl", fsync=False)
        ledger = Ledger(starting_cash=100_000.0)
        kernel = RiskKernel(limits, account_type=AccountType.MARGIN)
        executor = Executor(broker, kernel, journal, ledger, armed=True)
        return executor, broker, journal, ledger

    return build


class TestPipelineOrdering:
    async def test_review_precedes_place(self, wiring):
        """Simulation is mandatory and must happen first."""
        executor, broker, _, _ = wiring()
        await executor.execute([make_intent()], make_state())

        assert broker.calls == ["review:AAPL", "place:AAPL"]

    async def test_failed_review_prevents_place(self, wiring):
        executor, broker, journal, ledger = wiring(review_ok=False)
        report = await executor.execute([make_intent()], make_state())

        assert broker.calls == ["review:AAPL"]
        assert not report.fills
        assert len(report.rejected) == 1
        assert journal.read(event=EventType.REVIEW_REJECTED)
        assert ledger.open_symbols == []

    async def test_venue_rejection_books_no_fill(self, wiring):
        """The isError-200 trap: a rejected order must never book as filled."""
        executor, _, journal, ledger = wiring(place_ok=False)
        report = await executor.execute([make_intent()], make_state())

        assert not report.fills
        assert ledger.open_symbols == []
        assert journal.read(event=EventType.ORDER_FAILED)

    async def test_successful_order_books_to_ledger(self, wiring):
        executor, _, journal, ledger = wiring()
        report = await executor.execute([make_intent(notional=1_500.0)], make_state())

        assert len(report.fills) == 1
        assert ledger.quantity_of("AAPL") == pytest.approx(15.0)
        assert journal.read(event=EventType.FILL)

    async def test_journal_records_placement_before_fill(self, wiring):
        """A crash between the two must leave evidence, not a phantom position."""
        executor, _, journal, _ = wiring()
        await executor.execute([make_intent()], make_state())

        events = [e.event for e in journal.read()]
        assert events.index(EventType.ORDER_PLACED) < events.index(EventType.FILL)


class TestKernelCannotBeBypassed:
    async def test_vetoed_intent_never_reaches_the_broker(self, wiring):
        """The whole architecture rests on this."""
        executor, broker, journal, _ = wiring()
        # No invalidation condition: an automatic veto.
        intent = make_intent(invalidation="")
        report = await executor.execute([intent], make_state())

        assert broker.calls == []
        assert len(report.vetoed) == 1
        assert VetoCode.MISSING_INVALIDATION in report.vetoed[0].vetoes
        assert journal.read(event=EventType.KERNEL_VETO)

    async def test_kill_switch_blocks_entries(self, wiring):
        executor, broker, _, _ = wiring()
        state = make_state(kill_switch_engaged=True)
        report = await executor.execute([make_intent()], state)

        assert broker.calls == []
        assert VetoCode.KILL_SWITCH in report.vetoed[0].vetoes

    async def test_tripped_breaker_blocks_entries_but_allows_risk_exits(self, wiring):
        """Halt stops new risk; it must never trap an existing position."""
        from osiris.kernel.state import BreakerState
        from tests.conftest import make_portfolio, make_position

        breakers = BreakerState().trip(VetoCode.BREAKER_TRIPPED, "daily loss")
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        state = make_state(portfolio, breakers=breakers)

        executor, broker, _, _ = wiring()
        entry = make_intent(side=Side.BUY, reason="rank_entry")
        risk_exit = make_intent(
            symbol="AAPL", side=Side.SELL, notional=5_000.0, reason="risk_exit"
        )
        report = await executor.execute([entry, risk_exit], state)

        assert "place:AAPL" in broker.calls
        assert len(report.fills) == 1
        assert report.fills[0].side is Side.SELL
        assert any(VetoCode.BREAKER_TRIPPED in d.vetoes for d in report.vetoed)

    async def test_unarmed_live_broker_refuses_to_place(self, tmp_path, limits):
        """Two independent affirmations are required before real money moves."""
        broker = SpyBroker()
        executor = Executor(
            broker,
            RiskKernel(limits, account_type=AccountType.MARGIN),
            Journal(tmp_path / "j.jsonl", fsync=False),
            Ledger(starting_cash=100_000.0),
            armed=False,
        )
        report = await executor.execute([make_intent()], make_state())

        assert "place:AAPL" not in broker.calls
        assert report.rejected[0][1] == "live path not armed"


class TestExitsAreNeverTrapped:
    """Regression guard. A position that cannot be closed is the worst bug here.

    Positions may grow to max_symbol_weight (10% of equity) while the per-trade
    notional cap is 2%. If that cap applied to sells, any position above 2% would
    be permanently unclosable -- and the position most needing an exit is exactly
    the one that has grown largest.
    """

    async def test_large_position_can_be_fully_exited(self, wiring):
        from tests.conftest import diversified_positions, make_portfolio, make_position

        # A full book so the diversification floor is satisfied, plus one
        # oversized name at 9% of equity: legal to hold, but 4.5x the 2%
        # per-trade cap.
        portfolio = make_portfolio(
            positions=(*diversified_positions(16), make_position("AAPL", 9_000.0))
        )
        state = make_state(portfolio)

        executor, broker, _, _ = wiring()
        exit_intent = make_intent(
            symbol="AAPL", side=Side.SELL, notional=9_000.0, reason="rank_exit"
        )
        report = await executor.execute([exit_intent], state)

        assert "place:AAPL" in broker.calls
        assert len(report.fills) == 1

    async def test_entry_above_the_cap_is_still_vetoed(self, wiring):
        """The exemption must apply to exits ONLY, never to entries."""
        executor, broker, _, _ = wiring()
        report = await executor.execute([make_intent(notional=9_000.0)], make_state())

        assert broker.calls == []
        assert VetoCode.NOTIONAL_CAP in report.vetoed[0].vetoes

    async def test_risk_exit_bypasses_the_diversification_floor(self, wiring):
        """Honoring a stop matters more than holding 15 names."""
        from tests.conftest import make_portfolio, make_position

        # Only 3 positions: well under min_position_count of 15.
        portfolio = make_portfolio(
            positions=(
                make_position("AAPL", 5_000.0),
                make_position("MSFT", 5_000.0),
                make_position("JPM", 5_000.0),
            )
        )
        executor, broker, _, _ = wiring()
        report = await executor.execute(
            [make_intent(symbol="AAPL", side=Side.SELL, notional=5_000.0, reason="risk_exit")],
            make_state(portfolio),
        )

        assert "place:AAPL" in broker.calls
        assert len(report.fills) == 1


class TestAmbiguityHalts:
    async def test_unknown_venue_state_halts_the_cycle(self, wiring):
        """Both guesses corrupt the ledger, so neither is made."""
        executor, _, journal, ledger = wiring(raise_ambiguous=True)
        report = await executor.execute([make_intent()], make_state())

        assert report.halted_reason is not None
        assert ledger.open_symbols == []
        assert journal.read(event=EventType.ERROR)

    async def test_halt_stops_processing_remaining_intents(self, wiring):
        """Continuing past an unknown position would compound the ambiguity."""
        executor, broker, _, _ = wiring(raise_ambiguous=True)
        intents = [
            make_intent(symbol="AAPL"),
            make_intent(symbol="MSFT"),
            make_intent(symbol="JPM"),
        ]
        await executor.execute(intents, make_state(symbols=("AAPL", "MSFT", "JPM")))

        assert len([c for c in broker.calls if c.startswith("place:")]) == 1


class TestBatchAccumulation:
    async def test_batch_evaluates_exits_before_entries(self, wiring):
        """Exits free buying power, so they must be processed first."""
        from dataclasses import replace as dc_replace

        from tests.conftest import diversified_positions, make_portfolio, make_position

        # Enough names that closing one still clears the diversification floor.
        # The entry is put in a sector the fixture book is light in, so only the
        # ordering is under test rather than the sector cap.
        portfolio = make_portfolio(
            positions=(*diversified_positions(16), make_position("MSFT", 1_500.0))
        )
        executor, broker, _, _ = wiring()
        intents = [
            make_intent(symbol="NEWCO", side=Side.BUY, notional=1_500.0),
            make_intent(symbol="MSFT", side=Side.SELL, notional=1_500.0, reason="rank_exit"),
        ]
        state = make_state(portfolio, symbols=("NEWCO", "MSFT"))
        state = dc_replace(state, sectors={**state.sectors, "NEWCO": "Utilities"})

        await executor.execute(intents, state)

        places = [c for c in broker.calls if c.startswith("place:")]
        assert places.index("place:MSFT") < places.index("place:NEWCO")

    async def test_collective_exposure_is_bounded(self, wiring):
        """Twenty individually-compliant buys must not breach a portfolio cap.

        Evaluating each intent against the INITIAL state would let the batch pass
        collectively while each member looks fine alone.
        """
        from dataclasses import replace as dc_replace

        executor, _, _, _ = wiring()
        # Each buy is 2% of equity, exactly at the per-trade cap, so every intent
        # is individually legal. Twenty of them in one sector is 40%, which must
        # breach the 25% sector ceiling collectively.
        symbols = [f"TECH{i}" for i in range(20)]
        intents = [make_intent(symbol=s, notional=2_000.0) for s in symbols]
        state = make_state(symbols=tuple(symbols))
        state = dc_replace(state, sectors=dict.fromkeys(symbols, "Technology"))

        report = await executor.execute(intents, state)

        approved_notional = sum(f.quantity * f.price for f in report.fills)
        assert approved_notional <= state.portfolio.equity * 0.25 + 1.0
        assert report.vetoed, "sector cap must bind on the batch"
        assert any(
            VetoCode.SECTOR_WEIGHT_CAP in d.vetoes for d in report.vetoed
        )


class TestReconciliation:
    async def test_clean_reconciliation_reports_clean(self, wiring):
        executor, _, journal, ledger = wiring(positions={}, equity=100_000.0)
        _, clean = await executor.reconcile(make_state(), prices={})

        assert clean
        assert journal.read(event=EventType.RECONCILIATION)

    async def test_divergence_trips_a_breaker(self, wiring):
        """Ledger drift is a correctness failure, not a warning to log."""
        executor, _, journal, _ = wiring(
            positions={"TSLA": 5.0}, equity=100_000.0
        )
        state, clean = await executor.reconcile(
            make_state(), prices={"TSLA": 300.0}
        )

        assert not clean
        assert state.breakers.is_tripped
        assert journal.read(event=EventType.RECONCILIATION_BREAK)
        assert journal.read(event=EventType.BREAKER_TRIPPED)

    async def test_slippage_is_measured_against_intent(self, wiring):
        """Realized vs modeled slippage is the earliest edge-decay signal."""
        executor, _, _, _ = wiring(fill_price=101.0)
        intent = make_intent()
        report = await executor.execute([intent], make_state())

        assert report.realized_slippage_bps is not None
