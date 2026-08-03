"""Execution pipeline invariants: journal, kill switch, and the paper broker.

The pipeline tests in test_executor.py cover the kernel/review/place ordering.
This file covers the substrate those guarantees rest on.
"""

from __future__ import annotations

import pytest

from osiris.execution.broker import (
    OrderRejected,
    OrderRequest,
    PaperBroker,
    PaperFillModel,
)
from osiris.execution.journal import EventType, Journal
from osiris.execution.killswitch import KillSwitch
from osiris.types import Side
from tests.conftest import make_quote


def paper(**kwargs) -> PaperBroker:
    """A deterministic paper broker: no random rejections or partials."""
    model = PaperFillModel(
        partial_fill_probability=0.0, rejection_probability=0.0, seed=1
    )
    return PaperBroker(
        starting_cash=kwargs.pop("starting_cash", 100_000.0),
        quotes=kwargs.pop("quotes", {"AAPL": make_quote("AAPL", 100.0)}),
        adv=kwargs.pop("adv", {"AAPL": 500_000_000.0}),
        fill_model=kwargs.pop("fill_model", model),
    )


def request(
    symbol: str = "AAPL",
    side: Side = Side.BUY,
    notional: float = 1_000.0,
    key: str = "k1",
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol, side=side, notional_usd=notional, idempotency_key=key
    )


class TestJournal:
    def test_append_is_sequenced_and_persistent(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl", fsync=False)
        j.append(EventType.CYCLE_START, {"a": 1})
        j.append(EventType.CYCLE_END, {"b": 2})

        events = j.read()
        assert [e.seq for e in events] == [1, 2]
        assert events[0].event is EventType.CYCLE_START

    def test_sequence_survives_reopen(self, tmp_path):
        """A restart must not restart numbering, or ordering becomes ambiguous."""
        path = tmp_path / "j.jsonl"
        Journal(path, fsync=False).append(EventType.FILL, {})

        reopened = Journal(path, fsync=False)
        rec = reopened.append(EventType.FILL, {})
        assert rec.seq == 2

    def test_append_never_rewrites_history(self, tmp_path):
        """Append-only is the property that makes the journal evidence."""
        path = tmp_path / "j.jsonl"
        j = Journal(path, fsync=False)
        j.append(EventType.ORDER_PLACED, {"symbol": "AAPL"})
        first_line = path.read_text().splitlines()[0]

        j.append(EventType.ORDER_PLACED, {"symbol": "MSFT"})
        lines = path.read_text().splitlines()

        assert lines[0] == first_line
        assert len(lines) == 2

    def test_filters_by_event_type(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl", fsync=False)
        j.append(EventType.FILL, {})
        j.append(EventType.KERNEL_VETO, {"vetoes": ["notional_cap"]})
        j.append(EventType.FILL, {})

        assert len(j.read(event=EventType.FILL)) == 2

    def test_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        """A truncated final write must not make the whole journal unreadable."""
        path = tmp_path / "j.jsonl"
        j = Journal(path, fsync=False)
        j.append(EventType.FILL, {})
        with path.open("a") as fh:
            fh.write("{not json\n")
        j.append(EventType.FILL, {})

        assert len(j.read()) == 2

    def test_veto_summary_ranks_causes(self):
        """Answers 'why is nothing trading' -- the question vetoes exist to answer."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            j = Journal(Path(d) / "j.jsonl", fsync=False)
            j.append(EventType.KERNEL_VETO, {"vetoes": ["spread_too_wide"]})
            j.append(EventType.KERNEL_VETO, {"vetoes": ["spread_too_wide", "stale_data"]})

            summary = j.veto_summary()
            assert summary["spread_too_wide"] == 2
            assert summary["stale_data"] == 1

    def test_serializes_domain_types(self, tmp_path):
        """Enums, datetimes, and pydantic models must round-trip."""
        from tests.conftest import make_intent

        j = Journal(tmp_path / "j.jsonl", fsync=False)
        j.append(EventType.INTENT_EMITTED, {"intent": make_intent(), "side": Side.BUY})

        payload = j.read()[0].payload
        assert payload["side"] == "buy"
        assert payload["intent"]["symbol"] == "AAPL"


class TestKillSwitch:
    def test_absent_file_means_not_engaged(self, tmp_path):
        assert not KillSwitch(tmp_path / "KILL").engaged

    def test_engage_records_reason(self, tmp_path):
        ks = KillSwitch(tmp_path / "KILL")
        state = ks.engage("drawdown breach")

        assert state.engaged
        assert "drawdown breach" in state.reason

    def test_engaged_switch_blocks_new_risk_but_allows_exits(self, tmp_path):
        """Halt means 'take no new risk', never 'stop managing existing risk'.

        A bot that freezes while holding losers is worse than one that never traded.
        """
        ks = KillSwitch(tmp_path / "KILL")
        state = ks.engage("manual")

        assert not state.allows_new_risk
        assert state.allows_exits

    def test_release_requires_acknowledgement(self, tmp_path):
        """A fuse that resets itself is not a fuse."""
        ks = KillSwitch(tmp_path / "KILL")
        ks.engage("manual")

        with pytest.raises(ValueError, match="acknowledgement"):
            ks.release("")
        assert ks.engaged

    def test_release_with_acknowledgement_clears(self, tmp_path):
        ks = KillSwitch(tmp_path / "KILL")
        ks.engage("manual")

        assert not ks.release("rahul").engaged

    def test_survives_process_restart(self, tmp_path):
        """File-based so a supervisor cannot restart into an un-halted state."""
        path = tmp_path / "KILL"
        KillSwitch(path).engage("halt")

        assert KillSwitch(path).engaged


class TestPaperBroker:
    async def test_review_accepts_a_fundable_order(self):
        result = await paper().review(request())

        assert result.accepted
        assert result.estimated_quantity > 0

    async def test_buy_fills_above_the_mid(self):
        """Crossing the spread is not optional. Mid fills invent alpha."""
        broker = paper()
        result = await broker.place(request())

        quote = broker.quotes["AAPL"]
        assert result.fills[0].price > quote.mid

    async def test_sell_fills_below_the_mid(self):
        broker = paper()
        await broker.place(request(key="buy"))
        result = await broker.place(request(side=Side.SELL, notional=500.0, key="sell"))

        assert result.fills[0].price < broker.quotes["AAPL"].mid

    async def test_review_rejects_unfundable_buy(self):
        broker = paper(starting_cash=100.0)
        result = await broker.review(request(notional=50_000.0))

        assert not result.accepted
        assert "insufficient cash" in result.message

    async def test_review_rejects_selling_what_is_not_held(self):
        """Long-only: there is no short-sell tool on this venue."""
        result = await paper().review(request(side=Side.SELL))

        assert not result.accepted
        assert "no position" in result.message

    async def test_place_raises_when_review_fails(self):
        """No code path reaches the venue unsimulated."""
        broker = paper(starting_cash=10.0)

        with pytest.raises(OrderRejected):
            await broker.place(request(notional=50_000.0))

    async def test_missing_quote_is_rejected_not_guessed(self):
        """Fails closed: absent data is a rejection, never an assumed price."""
        result = await paper().review(request(symbol="UNKNOWN"))

        assert not result.accepted
        assert "no quote" in result.message

    async def test_idempotency_key_prevents_duplicate_orders(self):
        """A retry of the same logical order must not double-fill."""
        broker = paper()
        first = await broker.place(request(key="same"))
        second = await broker.place(request(key="same"))

        assert first.order_id == second.order_id
        assert broker.order_seq == 1

    async def test_positions_and_equity_track_fills(self):
        broker = paper()
        await broker.place(request(notional=1_000.0))

        positions = await broker.get_positions()
        assert positions["AAPL"] > 0
        # Equity dips slightly: the spread is a real cost paid on entry.
        assert await broker.get_account_equity() < 100_000.0

    async def test_selling_everything_closes_the_position(self):
        broker = paper()
        await broker.place(request(notional=1_000.0, key="b"))
        held = (await broker.get_positions())["AAPL"]
        quote = broker.quotes["AAPL"]

        await broker.place(
            request(side=Side.SELL, notional=held * quote.bid * 2, key="s")
        )
        assert "AAPL" not in await broker.get_positions()

    async def test_impact_scales_with_participation(self):
        """A large order relative to ADV must fill worse. Otherwise size is free."""
        model = PaperFillModel(partial_fill_probability=0.0, rejection_probability=0.0)
        small = model.effective_price(Side.BUY, 100.0, 1_000.0, 500_000_000.0)
        large = model.effective_price(Side.BUY, 100.0, 5_000_000.0, 500_000_000.0)

        assert large > small

    async def test_partial_fills_are_possible(self):
        broker = PaperBroker(
            quotes={"AAPL": make_quote("AAPL", 100.0)},
            adv={"AAPL": 5e8},
            fill_model=PaperFillModel(
                partial_fill_probability=1.0, rejection_probability=0.0, seed=3
            ),
        )
        result = await broker.place(request(notional=1_000.0))

        assert 0 < result.filled_quantity < 1_000.0 / 100.0
