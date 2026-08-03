"""The guardian: continuous intraday risk enforcement.

What must be proven:
  1. A breached stop SELLS, through the full pipeline (review then place).
  2. A healthy position is left alone -- the guardian can never add risk.
  3. A stop fires ONCE, not on every tick.
  4. The guardian sells even while breakers are tripped (exits survive halts).
  5. An empty book does nothing and clears state.
"""

from __future__ import annotations

import pytest

from osiris.config import AccountType, RiskLimits, Settings
from osiris.execution.executor import Executor
from osiris.execution.guardian import Guardian
from osiris.execution.journal import Journal
from osiris.execution.ledger import DailyPnL, Ledger
from osiris.kernel.kernel import RiskKernel
from osiris.types import Fill, Side
from tests.conftest import NOW, make_quote
from tests.test_executor import SpyBroker


def seeded_ledger(*, symbol: str = "AAPL", qty: float = 100.0, cost: float = 100.0) -> Ledger:
    """A ledger holding one position bought at `cost`."""
    ledger = Ledger(starting_cash=100_000.0)
    ledger.apply_fill(
        Fill(
            symbol=symbol,
            side=Side.BUY,
            quantity=qty,
            price=cost,
            ts=NOW,
            order_id="seed-1",
            idempotency_key="seed-key",
        )
    )
    return ledger


@pytest.fixture
def build(tmp_path, limits: RiskLimits):
    def _build(ledger: Ledger | None = None, *, fill_price: float = 80.0):
        broker = SpyBroker(fill_price=fill_price)
        journal = Journal(tmp_path / "j.jsonl", fsync=False)
        ledger = ledger if ledger is not None else seeded_ledger()
        kernel = RiskKernel(limits, account_type=AccountType.MARGIN)
        executor = Executor(broker, kernel, journal, ledger, armed=True)
        guardian = Guardian(
            settings=Settings(),
            limits=limits,
            executor=executor,
            kernel=kernel,
            journal=journal,
            ledger=ledger,
            pnl=DailyPnL(day_start_equity=110_000.0, peak_equity=110_000.0),
            stop_loss_pct=0.15,
        )
        return guardian, broker, journal, ledger

    return _build


class TestStopFires:
    async def test_breached_stop_sells_through_full_pipeline(self, build):
        guardian, broker, _, _ = build()
        # Bought at 100; quote at 80 is a -20% drawdown against a 15% stop.
        quotes = {"AAPL": make_quote("AAPL", price=80.0)}

        intents = await guardian.tick(quotes, now=NOW)

        assert [i.symbol for i in intents] == ["AAPL"]
        assert intents[0].side is Side.SELL
        # Review before place: the guardian gets no shortcut past simulation.
        assert broker.calls == ["review:AAPL", "place:AAPL"]

    async def test_healthy_position_is_left_alone(self, build):
        guardian, broker, _, _ = build()
        quotes = {"AAPL": make_quote("AAPL", price=98.0)}  # -2%, inside the stop

        intents = await guardian.tick(quotes, now=NOW)

        assert intents == []
        assert broker.calls == []

    async def test_stop_fires_once_not_every_tick(self, build):
        guardian, broker, _, ledger = build()
        quotes = {"AAPL": make_quote("AAPL", price=80.0)}

        first = await guardian.tick(quotes, now=NOW)
        # Simulate the fill NOT emptying the position (partial fill): the
        # ledger still holds shares, price still below stop.
        second = await guardian.tick(quotes, now=NOW)

        assert len(first) == 1
        assert second == [], "a fired stop must not re-fire the same session"

    async def test_new_session_rearms_stops(self, build, tmp_path, limits):
        """A stop whose order was REJECTED still holds the position. It must
        not retry the same session (order-budget burn), but a new session
        re-arms it."""
        broker = SpyBroker(place_ok=False, fill_price=80.0)
        journal = Journal(tmp_path / "j2.jsonl", fsync=False)
        ledger = seeded_ledger()
        kernel = RiskKernel(limits, account_type=AccountType.MARGIN)
        executor = Executor(broker, kernel, journal, ledger, armed=True)
        guardian = Guardian(
            settings=Settings(),
            limits=limits,
            executor=executor,
            kernel=kernel,
            journal=journal,
            ledger=ledger,
            pnl=DailyPnL(day_start_equity=110_000.0, peak_equity=110_000.0),
            stop_loss_pct=0.15,
        )
        quotes = {"AAPL": make_quote("AAPL", price=80.0)}

        first = await guardian.tick(quotes, now=NOW)
        same_session = await guardian.tick(quotes, now=NOW)
        guardian.on_new_session()
        next_session = await guardian.tick(quotes, now=NOW)

        assert len(first) == 1
        assert same_session == []
        assert len(next_session) == 1

    async def test_exit_runs_even_with_intraday_breaker_loss(self, build):
        """A crash day trips the daily-loss breaker AND fires stops. The exit
        must still go out -- halts stop new risk, never risk reduction."""
        guardian, broker, _, _ = build()
        # Equity collapse: 100 sh from 100 -> 40 plus cash = big daily loss.
        quotes = {"AAPL": make_quote("AAPL", price=40.0)}

        intents = await guardian.tick(quotes, now=NOW)

        assert len(intents) == 1
        assert "place:AAPL" in broker.calls


class TestQuietBook:
    async def test_empty_book_is_a_noop(self, build):
        guardian, broker, _, _ = build(Ledger(starting_cash=1_000.0))

        intents = await guardian.tick({}, now=NOW)

        assert intents == []
        assert broker.calls == []

    async def test_no_quotes_is_a_noop_not_a_crash(self, build):
        guardian, broker, _, _ = build()

        intents = await guardian.tick({}, now=NOW)

        assert intents == []
        assert broker.calls == []


class TestMarks:
    async def test_tick_marks_equity_for_breakers(self, build):
        guardian, _, _, _ = build()
        quotes = {"AAPL": make_quote("AAPL", price=120.0)}

        await guardian.tick(quotes, now=NOW)

        # 100_000 cash - 10_000 cost + 100 sh * 120 = 102_000
        assert guardian.pnl.peak_equity >= 102_000.0
