"""API contract tests.

The most important assertion here is a NEGATIVE one: the API must expose no way
to place an order, mutate a risk limit, or arm the live path. A dashboard that can
trade is an attack surface pointed at the account.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import pytest

from osiris.api import create_app
from osiris.api.app import set_state
from osiris.api.events import Channel, EventBus
from osiris.api.state import build_runtime_state
from osiris.config import AccountType, Mode, RiskLimits, Settings
from osiris.execution.broker import PaperBroker
from osiris.types import Fill, Side
from tests.conftest import NOW, make_quote


@pytest.fixture
def state(tmp_path):
    settings = Settings(
        mode=Mode.PAPER, account_type=AccountType.MARGIN, account_equity_usd=100_000.0
    )
    st = build_runtime_state(
        settings=settings,
        limits=RiskLimits(),
        journal_path=tmp_path / "journal.jsonl",
        broker=PaperBroker(
            starting_cash=100_000.0, quotes={"AAPL": make_quote("AAPL", 100.0)}
        ),
    )
    st.bus = EventBus()
    st.killswitch.path = tmp_path / "KILL_SWITCH"
    set_state(st)
    return st


@pytest.fixture
def app(state):
    """The ASGI app, wired to the isolated per-test state."""
    return create_app()


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_reports_mode_and_arming(self, client):
        body = (await client.get("/api/health")).json()

        assert body["status"] == "ok"
        assert body["mode"] == "paper"
        assert body["armed"] is False

    async def test_reports_kill_switch_state(self, client, state):
        state.killswitch.engage("test halt")
        body = (await client.get("/api/health")).json()

        assert body["kill_switch_engaged"] is True


class TestApiCannotTrade:
    """The load-bearing negative test."""

    async def test_no_order_endpoint_exists(self, client):
        spec = (await client.get("/openapi.json")).json()
        paths = " ".join(spec["paths"]).lower()

        for forbidden in ("order", "place", "trade", "buy", "sell", "execute"):
            assert forbidden not in paths, f"API exposes a {forbidden!r} path"

    async def test_no_endpoint_mutates_risk_limits(self, client):
        spec = (await client.get("/openapi.json")).json()
        writable = [
            path
            for path, ops in spec["paths"].items()
            if {"post", "put", "patch", "delete"} & set(ops)
        ]

        # Only the human-safety controls may accept writes. `/api/control/arm`
        # records the two live-trading affirmations in .env; it cannot arm the
        # RUNNING process (the dry-run flag is welded to the MCP adapter at
        # connect time), so a restart remains the second factor.
        assert sorted(writable) == [
            "/api/control/arm",
            "/api/control/breakers/reset",
            "/api/control/kill-switch",
            "/api/control/kill-switch/release",
        ]


class TestActivityFeed:
    """The journal projected into something a human can read.

    The point of this endpoint is that an action and its reason arrive on the same
    row. Tests here assert that property rather than the wording, so copy can be
    reworded without breaking them.
    """

    @pytest.fixture
    def seeded(self, state):
        from osiris.execution.journal import EventType

        j = state.journal
        j.append(
            EventType.ORDER_PLACED,
            {
                "symbol": "AAPL",
                "side": "buy",
                "order_id": "o-1",
                "idempotency_key": "k-1",
                "reason": "rank_entry",
                "thesis": "momentum leader",
                "invalidation": "drops out of top 20",
            },
        )
        j.append(
            EventType.FILL,
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10.0,
                "price": 150.0,
                "order_id": "o-1",
            },
        )
        j.append(
            EventType.KERNEL_VETO,
            {
                "symbol": "MSFT",
                "side": "buy",
                "vetoes": ["sector_weight_cap"],
                "notes": ["sector Technology would reach 26.8%, cap 25.0%"],
            },
        )
        j.append(EventType.BREAKER_TRIPPED, {"reasons": ["daily loss 3.2%"]})
        return state

    async def test_a_fill_reports_what_and_why_together(self, client, seeded):
        rows = (await client.get("/api/activity")).json()
        buy = next(r for r in rows if r["kind"] == "bought")

        assert buy["symbol"] == "AAPL"
        assert buy["notional_usd"] == pytest.approx(1500.0)
        assert buy["reason"], "a trade with no reason is what we are fixing"
        assert "momentum leader" in buy["detail"]

    async def test_a_veto_is_translated_out_of_enum_speak(self, client, seeded):
        rows = (await client.get("/api/activity")).json()
        blocked = next(r for r in rows if r["kind"] == "blocked")

        assert blocked["symbol"] == "MSFT"
        assert "sector_weight_cap" not in blocked["reason"]
        assert "sector" in blocked["reason"].lower()
        # The kernel's own note is more specific than the code.
        assert "26.8%" in blocked["detail"]

    async def test_blocked_orders_appear_beside_trades(self, client, seeded):
        """A kernel blocking everything must not look like a quiet market."""
        kinds = {r["kind"] for r in (await client.get("/api/activity")).json()}

        assert {"bought", "blocked"} <= kinds

    async def test_a_halt_is_surfaced(self, client, seeded):
        rows = (await client.get("/api/activity")).json()
        halt = next(r for r in rows if r["kind"] == "halted")

        assert "daily loss 3.2%" in halt["reason"]

    async def test_newest_first(self, client, seeded):
        rows = (await client.get("/api/activity")).json()
        seqs = [r["seq"] for r in rows]

        assert seqs == sorted(seqs, reverse=True)

    async def test_kind_filter_narrows_the_feed(self, client, seeded):
        rows = (await client.get("/api/activity?kind=blocked")).json()

        assert rows
        assert all(r["kind"] == "blocked" for r in rows)

    async def test_an_exit_explains_the_trigger_not_the_thesis(self, client, state):
        """Showing a bull thesis next to a sell is actively misleading."""
        from osiris.execution.journal import EventType

        state.journal.append(
            EventType.ORDER_PLACED,
            {
                "symbol": "TSLA",
                "side": "sell",
                "order_id": "o-2",
                "idempotency_key": "k-2",
                "reason": "rank_exit",
                "thesis": "strong momentum, vol 30%",
                "invalidation": "dropped out of target ranking",
            },
        )
        state.journal.append(
            EventType.FILL,
            {
                "symbol": "TSLA",
                "side": "sell",
                "quantity": 5.0,
                "price": 200.0,
                "order_id": "o-2",
            },
        )

        rows = (await client.get("/api/activity")).json()
        sell = next(r for r in rows if r["kind"] == "sold")

        assert "dropped out" in sell["detail"]
        assert "strong momentum" not in sell["detail"]

    async def test_an_empty_journal_yields_an_empty_feed(self, client):
        assert (await client.get("/api/activity")).json() == []


class TestPreflight:
    """Go-live readiness, exposed read-only."""

    async def test_an_unproven_system_is_not_cleared(self, client):
        body = (await client.get("/api/preflight")).json()

        assert body["armed"] is False
        assert body["blocking_failures"]

    async def test_every_check_is_reported_with_its_severity(self, client):
        """The operator must see WHICH check blocks, not just that one did."""
        body = (await client.get("/api/preflight")).json()

        assert body["checks"]
        for check in body["checks"]:
            assert check["severity"] in {"blocking", "advisory"}
            assert check["detail"].strip()

    async def test_reporting_readiness_is_a_read(self, client):
        """Preflight must not be able to arm anything as a side effect."""
        before = (await client.get("/api/health")).json()["armed"]
        await client.get("/api/preflight")
        after = (await client.get("/api/health")).json()["armed"]

        assert before is False
        assert after is False

    async def test_an_engaged_kill_switch_blocks_clearance(self, client, state):
        state.killswitch.engage("operator halt")
        body = (await client.get("/api/preflight")).json()

        assert "kill_switch_clear" in body["blocking_failures"]


class TestPortfolio:
    async def test_empty_portfolio_reports_starting_equity(self, client):
        body = (await client.get("/api/portfolio")).json()

        assert body["equity"] == pytest.approx(100_000.0)
        assert body["position_count"] == 0

    async def test_positions_appear_with_weights_and_pnl(self, client, state):
        state.ledger.apply_fill(
            Fill(
                symbol="AAPL",
                side=Side.BUY,
                quantity=10,
                price=100.0,
                ts=NOW,
                order_id="o1",
                idempotency_key="k1",
            )
        )
        state.mark_prices({"AAPL": 110.0})
        state.ledger.set_metadata({"AAPL": "Technology"}, {"AAPL": 1.2})

        body = (await client.get("/api/portfolio")).json()
        position = body["positions"][0]

        assert position["symbol"] == "AAPL"
        assert position["unrealized_pnl"] == pytest.approx(100.0)
        assert position["sector"] == "Technology"
        assert body["portfolio_beta"] > 0

    async def test_numbers_are_returned_as_numbers(self, client):
        """Formatting is presentation; strings would break charting and sorting."""
        body = (await client.get("/api/portfolio")).json()

        assert isinstance(body["equity"], int | float)
        assert isinstance(body["daily_pnl_pct"], int | float)


class TestBreakers:
    async def test_reports_headroom_not_just_a_flag(self, client):
        """'3% from a halt' is actionable; 'not tripped' is not."""
        rows = (await client.get("/api/breakers")).json()
        names = {r["name"] for r in rows}

        assert {"daily_loss", "max_drawdown", "consecutive_losses"} <= names
        for row in rows:
            assert "threshold" in row and "value" in row

    async def test_daily_loss_trips_at_the_threshold(self, client, state):
        state.pnl.day_start_equity = 100_000.0
        state.ledger.cash = 96_000.0  # -4%, past the 3% halt

        rows = (await client.get("/api/breakers")).json()
        daily = next(r for r in rows if r["name"] == "daily_loss")

        assert daily["tripped"] is True


class TestKillSwitchControl:
    async def test_engage_then_release_round_trip(self, client):
        engaged = await client.post(
            "/api/control/kill-switch", json={"reason": "manual halt"}
        )
        assert engaged.json()["kill_switch_engaged"] is True

        released = await client.post(
            "/api/control/kill-switch/release", json={"acknowledged_by": "rahul"}
        )
        assert released.json()["kill_switch_engaged"] is False

    async def test_release_requires_an_acknowledgement(self, client):
        """A fuse that resets itself is not a fuse."""
        await client.post("/api/control/kill-switch", json={"reason": "halt"})
        resp = await client.post(
            "/api/control/kill-switch/release", json={"acknowledged_by": ""}
        )

        assert resp.status_code == 422

    async def test_engage_requires_a_reason(self, client):
        resp = await client.post("/api/control/kill-switch", json={"reason": ""})

        assert resp.status_code == 422


class TestJournalAndVetoes:
    async def test_journal_returns_appended_events(self, client, state):
        from osiris.execution.journal import EventType

        state.journal.append(EventType.CYCLE_START, {"as_of": "2026-07-31"})
        rows = (await client.get("/api/journal")).json()

        assert rows[-1]["event"] == "cycle_start"

    async def test_vetoes_are_visible_alongside_fills(self, client, state):
        """A kernel blocking everything must not look like a quiet market."""
        from osiris.execution.journal import EventType

        state.journal.append(EventType.KERNEL_VETO, {"vetoes": ["spread_too_wide"]})
        state.journal.append(EventType.KERNEL_VETO, {"vetoes": ["spread_too_wide"]})

        summary = (await client.get("/api/journal/veto-summary")).json()
        assert summary["spread_too_wide"] == 2

    async def test_journal_filters_by_event_type(self, client, state):
        from osiris.execution.journal import EventType

        state.journal.append(EventType.FILL, {"symbol": "AAPL"})
        state.journal.append(EventType.CYCLE_START, {})

        rows = (await client.get("/api/journal?event=fill")).json()
        assert all(r["event"] == "fill" for r in rows)


class TestFactorExposure:
    async def test_insufficient_history_says_so_rather_than_guessing(self, client):
        body = (await client.get("/api/factor-exposure")).json()

        assert body["n_periods"] < 3
        assert "insufficient" in body["verdict"]

    async def test_pure_beta_book_is_not_credited_with_alpha(self, client, state):
        """A book that just tracks the market must show beta, not alpha."""
        bench = [0.01, -0.005, 0.008, 0.002, -0.011, 0.006, 0.004, -0.002] * 5
        state.benchmark_returns = list(bench)
        state.daily_returns = [r * 1.05 for r in bench]

        body = (await client.get("/api/factor-exposure")).json()

        assert body["market_beta"] == pytest.approx(1.05, abs=0.02)
        assert body["is_significant"] is False


class TestEventBus:
    def test_publish_fans_out_to_subscribers(self):
        bus = EventBus()
        sub = bus.subscribe(replay=False)
        bus.publish(Channel.FILL, {"symbol": "AAPL"})

        assert len(sub.queue) == 1

    def test_late_joiner_receives_replay(self):
        """An empty dashboard on open is worse than a slightly stale one."""
        bus = EventBus()
        bus.publish(Channel.FILL, {"symbol": "AAPL"})
        sub = bus.subscribe(replay=True)

        assert len(sub.queue) == 1

    def test_slow_subscriber_drops_frames_instead_of_blocking(self):
        """Dropping dashboard frames is fine; delaying an order is not."""
        bus = EventBus()
        sub = bus.subscribe(replay=False)
        sub.queue = type(sub.queue)(maxlen=2)

        for i in range(5):
            bus.publish(Channel.PNL, {"i": i})

        assert len(sub.queue) == 2
        assert sub.dropped == 3

    def test_heartbeats_are_not_replayed(self):
        """Replaying stale keepalives would waste the buffer."""
        bus = EventBus()
        bus.publish(Channel.HEARTBEAT, {})
        sub = bus.subscribe(replay=True)

        assert len(sub.queue) == 0

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        sub = bus.subscribe(replay=False)
        bus.unsubscribe(sub)
        bus.publish(Channel.FILL, {})

        assert len(sub.queue) == 0

    def test_sequence_numbers_are_monotonic(self):
        """The client uses these to detect gaps after a reconnect."""
        bus = EventBus()
        first = bus.publish(Channel.FILL, {})
        second = bus.publish(Channel.PNL, {})

        assert second.seq == first.seq + 1


async def read_sse_lines(app, *, want: int, timeout: float = 3.0) -> list[str]:
    """Read up to `want` SSE lines by driving the ASGI app directly.

    Deliberately NOT via httpx.ASGITransport: that transport buffers the entire
    response body before returning, so a stream that never ends yields nothing and
    the test times out rather than failing informatively. Calling the ASGI
    interface lets us take the first N `http.response.body` messages and stop.

    The wall-clock bound is mandatory for the same reason: an SSE stream has no
    end, so a reader waiting for exactly N lines would hang whenever fewer arrive.
    """
    chunks: list[bytes] = []
    status: dict[str, object] = {}

    async def receive() -> dict:
        # A real client sends its request and then only reads.
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
            status["headers"] = {
                k.decode().lower(): v.decode() for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            text = b"".join(chunks).decode(errors="replace")
            if len([ln for ln in text.splitlines() if ln]) >= want:
                raise _StopStream

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/stream",
        "raw_path": b"/api/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test"), (b"accept", b"text/event-stream")],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }

    with contextlib.suppress(TimeoutError, _StopStream):
        await asyncio.wait_for(app(scope, receive, send), timeout=timeout)

    assert status.get("code") == 200, status
    content_type = (status.get("headers") or {}).get("content-type", "")
    assert "text/event-stream" in content_type, content_type

    return [ln for ln in b"".join(chunks).decode(errors="replace").splitlines() if ln]


class _StopStream(Exception):
    """Internal sentinel: enough frames collected."""


class TestSSEStream:
    """Regression guards for two SSE bugs that both fail silently on the server.

    Bug 1: returning `EventSourceResponse(...)` instead of declaring it as
    `response_class` on an async-generator path operation. In FastAPI 0.141 the
    encoding lives in the routing layer, so returning the class directly yields a
    200 with a correct `text/event-stream` header and an immediately closed body.

    Bug 2: passing pre-serialized JSON via `data` instead of `raw_data`. The
    `data` field is ALWAYS JSON-encoded, so a JSON string is double-encoded and
    the client's JSON.parse returns a string rather than an object.
    """

    async def test_stream_body_is_not_immediately_closed(self, app, state):
        """Bug 1. The stream must actually emit frames, not just headers."""
        state.publish(Channel.FILL, {"symbol": "AAPL", "side": "buy"})
        lines = await read_sse_lines(app, want=5)

        assert lines, "stream returned headers but no body"
        assert any(line.startswith("event:") for line in lines), lines
        assert any(line.startswith("data:") for line in lines), lines

    async def test_data_decodes_to_an_object_not_a_string(self, app, state):
        """Bug 2. `json.loads(data)` must yield a dict, never a str."""
        state.publish(Channel.FILL, {"symbol": "AAPL", "side": "buy", "quantity": 3})
        lines = await read_sse_lines(app, want=8)

        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in lines
            if line.startswith("data: ")
        ]
        assert payloads, "no data frames received"
        for payload in payloads:
            assert isinstance(payload, dict), f"double-encoded: {payload!r}"

    async def test_events_carry_channel_names_for_client_demux(self, app, state):
        """One connection is multiplexed, so the event name is load-bearing."""
        state.publish(Channel.VETO, {"symbol": "AAPL", "vetoes": ["notional_cap"]})
        lines = await read_sse_lines(app, want=8)

        names = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
        assert "heartbeat" in names
        assert "veto" in names

    async def test_replayed_events_carry_monotonic_ids(self, app, state):
        """`id:` becomes Last-Event-ID, which the client uses to detect gaps."""
        state.publish(Channel.FILL, {"symbol": "AAPL"})
        state.publish(Channel.FILL, {"symbol": "MSFT"})
        lines = await read_sse_lines(app, want=10)

        ids = [int(line.removeprefix("id: ")) for line in lines if line.startswith("id: ")]
        assert len(ids) >= 2
        assert ids == sorted(ids)
