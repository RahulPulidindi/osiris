"""Live data plane and connection layer.

These test the code that reads a real brokerage account, using a fake adapter
rather than a live session. The property under test throughout is
**never substitute data**: a missing quote must drop a symbol, not default a
price, because an invented price flows straight into position sizing and the
resulting order is wrong in a way nothing downstream can detect.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from osiris.data.live import (
    BENCHMARK_SECTOR_WEIGHTS,
    LiveMarket,
    _estimate_betas,
    _rows,
    normalize_sector,
)

# Shaped like a real Robinhood account number. Placeholders such as "ACC1" are
# rejected by the finder's validation on purpose, since a malformed number fails
# server-side with the same unhelpful error as sending none at all.
ACCOUNT = "123456789"


class FakeResult:
    """Mimics an MCP CallToolResult carrying JSON in a text block."""

    def __init__(self, payload: Any) -> None:
        class Block:
            text = json.dumps(payload)

        self.content = [Block()]
        self.isError = False


class FakeAdapter:
    """Records calls and replays canned responses per capability."""

    def __init__(self, responses: dict[str, Any], *, available: set[str] | None = None):
        self.responses = responses
        self.available = available if available is not None else set(responses)
        self.calls: list[tuple[str, dict]] = []

    def has(self, capability: str) -> bool:
        return capability in self.available

    async def call(self, capability: str, args: dict | None = None) -> FakeResult:
        self.calls.append((capability, args or {}))
        if capability not in self.responses:
            raise RuntimeError(f"tool unavailable: {capability}")
        payload = self.responses[capability]
        if callable(payload):
            payload = payload(args or {})
        return FakeResult(payload)


def bars(
    closes: list[float], volume: float | None = 1_000_000.0, symbols: list[str] | None = None
) -> dict:
    """A multi-symbol historicals response, shaped like the real one.

    The original fixture returned a flat `{"historicals": [...]}` with no symbol
    attribution -- which is what a SINGLE-symbol call returns. That encoded the
    wrong API into the tests: they passed while the live call failed, because
    `get_equity_historicals` takes a `symbols` array and groups its response.
    """
    rows = [{"close_price": c} | ({"volume": volume} if volume else {}) for c in closes]
    return {
        "data": [
            {"symbol": s, "historicals": rows} for s in (symbols or ["AAPL"])
        ]
    }


class TestSectorNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Information Technology", "Technology"),
            ("consumer cyclical", "Discretionary"),
            ("Health Care", "Healthcare"),
            ("Financial Services", "Financials"),
            ("Communication Services", "Technology"),
            (None, "Unknown"),
            ("", "Unknown"),
        ],
    )
    def test_maps_vendor_labels_to_benchmark_vocabulary(self, raw, expected):
        """The deviation gate compares against benchmark weights.

        If sector strings do not match the benchmark's vocabulary, every position
        lands in an unknown bucket and the gate silently compares nothing.
        """
        assert normalize_sector(raw) == expected

    def test_normalized_labels_exist_in_the_benchmark(self):
        for raw in ("Information Technology", "consumer staples", "energy"):
            assert normalize_sector(raw) in BENCHMARK_SECTOR_WEIGHTS


class TestPayloadShapes:
    def test_reads_rows_under_varying_envelope_keys(self):
        """MCP tools differ on whether results sit under results/data/items."""
        for key in ("results", "data", "items", "quotes"):
            assert _rows({key: [{"symbol": "AAPL"}]}, "quotes") == [{"symbol": "AAPL"}]

    def test_accepts_a_bare_list(self):
        assert _rows([{"symbol": "X"}]) == [{"symbol": "X"}]

    def test_a_single_bare_record_is_treated_as_one_row(self):
        assert _rows({"symbol": "X"}) == [{"symbol": "X"}]

    def test_garbage_yields_no_rows(self):
        assert _rows(None) == []
        assert _rows("not a payload") == []


class TestQuotes:
    async def test_reads_bid_ask_last(self):
        market = LiveMarket(
            FakeAdapter(
                {
                    "getQuotes": {
                        "results": [
                            {
                                "symbol": "AAPL",
                                "bid_price": 99.5,
                                "ask_price": 100.5,
                                "last_trade_price": 100.0,
                            }
                        ]
                    }
                }
            )
        )
        quotes = await market.quotes(["AAPL"])

        assert quotes["AAPL"].bid == 99.5
        assert quotes["AAPL"].ask == 100.5
        assert quotes["AAPL"].mid == pytest.approx(100.0)

    async def test_a_symbol_without_a_price_is_dropped_not_defaulted(self):
        """The core safety property of this module.

        Defaulting a price would let the spread and staleness gates pass on a
        number nobody observed, and the order would be sized against fiction.
        """
        market = LiveMarket(
            FakeAdapter(
                {
                    "getQuotes": {
                        "results": [
                            {"symbol": "AAPL", "last_trade_price": 100.0},
                            {"symbol": "BROKEN"},
                        ]
                    }
                }
            )
        )
        quotes = await market.quotes(["AAPL", "BROKEN"])

        assert "AAPL" in quotes
        assert "BROKEN" not in quotes

    async def test_mid_is_reconstructed_only_from_real_observations(self):
        """Missing `last` may be derived from a real bid and ask, nothing else."""
        market = LiveMarket(
            FakeAdapter(
                {"getQuotes": {"results": [{"symbol": "X", "bid": 10.0, "ask": 12.0}]}}
            )
        )
        quotes = await market.quotes(["X"])

        assert quotes["X"].last == pytest.approx(11.0)

    async def test_inverted_bid_ask_is_corrected(self):
        """Quote() rejects ask < bid, which would abort the whole cycle."""
        market = LiveMarket(
            FakeAdapter(
                {
                    "getQuotes": {
                        "results": [
                            {"symbol": "X", "bid": 12.0, "ask": 10.0, "last": 11.0}
                        ]
                    }
                }
            )
        )
        quotes = await market.quotes(["X"])

        assert quotes["X"].bid <= quotes["X"].ask

    async def test_a_failed_quote_call_degrades_to_empty(self):
        """A transport failure must not abort the cycle and strand positions."""
        market = LiveMarket(FakeAdapter({}))

        assert await market.quotes(["AAPL"]) == {}

    async def test_no_symbols_makes_no_calls(self):
        adapter = FakeAdapter({})
        await LiveMarket(adapter).quotes([])

        assert adapter.calls == []


class TestHistory:
    async def test_loads_closes_oldest_first(self):
        market = LiveMarket(
            FakeAdapter({"getHistoricals": bars([float(i) for i in range(1, 61)])})
        )
        out = await market.history(["AAPL"])

        assert out["AAPL"].size == 60
        assert out["AAPL"][0] < out["AAPL"][-1]

    async def test_too_short_a_series_is_omitted(self):
        """Ranking needs enough history to compute momentum and volatility.

        A 5-bar series would produce a momentum number that is noise, and the
        ranker cannot tell the difference.
        """
        market = LiveMarket(FakeAdapter({"getHistoricals": bars([1.0, 2.0, 3.0])}))

        assert await market.history(["AAPL"]) == {}

    async def test_nonpositive_prices_are_discarded(self):
        market = LiveMarket(
            FakeAdapter({"getHistoricals": bars([0.0, -1.0, *range(1, 60)])})
        )
        out = await market.history(["AAPL"])

        assert (out["AAPL"] > 0).all()


class TestADV:
    async def test_computes_average_dollar_volume(self):
        market = LiveMarket(
            FakeAdapter({"getHistoricals": bars([100.0] * 10, volume=1_000.0)})
        )
        adv = await market._advs(["AAPL"])

        assert adv["AAPL"] == pytest.approx(100_000.0)

    async def test_absent_volume_yields_no_adv_rather_than_a_guess(self):
        """The ADV gate must abstain rather than be fed an invented denominator.

        A price-derived proxy is not average dollar volume. Supplying one would
        make the participation gate approve exactly the oversized orders it exists
        to block, while appearing to function.
        """
        market = LiveMarket(
            FakeAdapter({"getHistoricals": bars([100.0] * 10, volume=None)})
        )

        assert await market._advs(["AAPL"]) == {}


class TestUniverse:
    """`run_scan` requires a scan_id obtained from `get_scans`.

    Calling it bare fails validation, which is what happened live: the fixture
    omitted the id because the old code did too.
    """

    def scanner(self, symbols: list[str]):
        return FakeAdapter(
            {
                "listScans": {"scans": [{"scan_id": "scan-1"}]},
                "runScan": {"results": [{"symbol": s} for s in symbols]},
            }
        )

    async def test_prefers_the_brokers_scanner(self):
        market = LiveMarket(self.scanner([f"S{i}" for i in range(30)]))

        assert len(await market.universe(fallback=["AAPL"])) == 30

    async def test_resolves_the_scan_id_before_running(self):
        adapter = self.scanner([f"S{i}" for i in range(30)])
        await LiveMarket(adapter).universe(fallback=["AAPL"])

        assert dict(adapter.calls)["runScan"] == {"scan_id": "scan-1"}

    async def test_falls_back_when_no_saved_scan_exists(self):
        """No scan configured is normal on a new account, not an error."""
        adapter = FakeAdapter({"listScans": {"scans": []}, "runScan": {"results": []}})
        universe = await LiveMarket(adapter).universe(fallback=["AAPL", "MSFT"])

        assert universe == ["AAPL", "MSFT"]

    async def test_falls_back_when_no_scanner_capability_exists(self):
        market = LiveMarket(FakeAdapter({}, available=set()))
        universe = await market.universe(fallback=["AAPL", "MSFT"])

        assert universe == ["AAPL", "MSFT"]

    async def test_falls_back_when_the_scan_returns_too_few(self):
        """A scan returning 1 name is a broken scan, not a narrow market."""
        market = LiveMarket(self.scanner(["A"]))
        universe = await market.universe(fallback=["AAPL", "MSFT"])

        assert universe == ["AAPL", "MSFT"]

    async def test_respects_the_limit(self):
        market = LiveMarket(self.scanner([f"S{i}" for i in range(500)]))

        assert len(await market.universe(fallback=[], limit=50)) == 50


class TestSnapshot:
    @pytest.fixture
    def adapter(self):
        """Responds for whichever symbols are requested, as the real API does."""
        closes = [100.0 + i * 0.1 for i in range(200)]
        return FakeAdapter(
            {
                "getQuotes": lambda args: {
                    "results": [
                        {
                            "symbol": s,
                            "bid": 99.0,
                            "ask": 101.0,
                            "last_trade_price": 100.0,
                        }
                        for s in args.get("symbols", [])
                    ]
                },
                "getHistoricals": lambda args: bars(
                    closes, symbols=args.get("symbols", [])
                ),
                "getFundamentals": lambda args: {
                    "fundamentals": [
                        {"symbol": s, "sector": "Information Technology"}
                        for s in args.get("symbols", [])
                    ]
                },
            }
        )

    async def test_produces_a_usable_snapshot(self, adapter):
        snap = await LiveMarket(adapter).snapshot(universe=["AAPL", "MSFT"])

        assert set(snap.universe) == {"AAPL", "MSFT"}
        assert snap.quotes and snap.closes
        assert snap.benchmark_sector_weights == BENCHMARK_SECTOR_WEIGHTS

    async def test_sectors_are_normalized_in_the_snapshot(self, adapter):
        snap = await LiveMarket(adapter).snapshot(universe=["AAPL"])

        assert snap.sectors["AAPL"] == "Technology"

    async def test_held_symbols_are_always_included(self, adapter):
        """A held name excluded from the universe becomes un-exitable.

        The ranker can only sell what it can see, so dropping a holding for
        failing a screen would strand the position permanently.
        """
        snap = await LiveMarket(adapter).snapshot(universe=["AAPL"], held=["TSLA"])

        assert "TSLA" in snap.universe

    async def test_symbols_missing_history_are_excluded(self):
        adapter = FakeAdapter(
            {
                "getQuotes": lambda args: {
                    "results": [
                        {"symbol": s, "last": 100.0} for s in args.get("symbols", [])
                    ]
                },
                # Too short for any symbol.
                "getHistoricals": bars([1.0, 2.0]),
            }
        )
        snap = await LiveMarket(adapter).snapshot(universe=["AAPL"])

        assert snap.universe == []

    async def test_an_empty_snapshot_is_returned_not_raised(self):
        """A dead cycle must still return, so exits can be evaluated next time."""
        snap = await LiveMarket(FakeAdapter({})).snapshot(universe=["AAPL"])

        assert snap.universe == []
        assert snap.quotes == {}


class TestAccountScopedReads:
    """Regression: Robinhood requires `account_number` on account reads.

    The original code threaded it into ORDERS only, and never fetched it at all,
    so `get_portfolio` and `get_equity_positions` both failed server-side
    validation. Nothing client-side caught it because the tool schemas were being
    read from the wrong attribute and came back empty.
    """

    def broker(self, responses: dict):
        from osiris.execution.mcp_broker import MCPBroker

        return MCPBroker(FakeAdapter(responses))

    async def test_account_number_is_resolved_from_the_account_list(self):
        b = self.broker({"listAccounts": {"results": [{"account_number": "1234567"}]}})

        assert await b.resolve_account() == "1234567"

    async def test_resolution_is_cached(self):
        adapter = FakeAdapter(
            {"listAccounts": {"results": [{"account_number": ACCOUNT}]}}
        )
        from osiris.execution.mcp_broker import MCPBroker

        b = MCPBroker(adapter)
        await b.resolve_account()
        await b.resolve_account()

        assert len(adapter.calls) == 1

    async def test_equity_read_sends_the_account_number(self):
        adapter = FakeAdapter(
            {
                "listAccounts": {"results": [{"account_number": ACCOUNT}]},
                "getPortfolio": {"equity": 5000.0},
            }
        )
        from osiris.execution.mcp_broker import MCPBroker

        await MCPBroker(adapter).get_account_equity()
        portfolio_args = dict(adapter.calls)["getPortfolio"]

        assert portfolio_args["account_number"] == ACCOUNT

    async def test_position_read_sends_the_account_number(self):
        adapter = FakeAdapter(
            {
                "listAccounts": {"results": [{"account_number": ACCOUNT}]},
                "listPositions": {"results": [{"symbol": "AAPL", "quantity": 3.0}]},
            }
        )
        from osiris.execution.mcp_broker import MCPBroker

        positions = await MCPBroker(adapter).get_positions()

        assert positions == {"AAPL": 3.0}
        assert dict(adapter.calls)["listPositions"]["account_number"] == ACCOUNT

    async def test_unresolvable_account_returns_none_rather_than_guessing(self):
        b = self.broker({"listAccounts": {"results": []}})

        assert await b.resolve_account() is None


class TestAccountSelection:
    """Regression: a login can see several accounts, and 'first in the listing'
    once resolved a user's Individual account ($369) instead of the Agentic
    account ($100) the agent was meant to trade. Every risk limit was being
    computed against money the agent should never touch.
    """

    def broker(self, responses: dict):
        from osiris.execution.mcp_broker import MCPBroker

        return MCPBroker(FakeAdapter(responses))

    @staticmethod
    def listing() -> dict:
        return {
            "results": [
                {"account_number": "434306221", "type": "individual"},
                {"account_number": "990088777", "type": "agentic brokerage"},
            ]
        }

    async def test_agentic_account_wins_over_individual(self):
        b = self.broker({"listAccounts": self.listing()})

        assert await b.resolve_account() == "990088777"

    async def test_env_pin_beats_everything(self, monkeypatch):
        monkeypatch.setenv("OSIRIS_ACCOUNT_NUMBER", "434306221")
        b = self.broker({"listAccounts": self.listing()})

        assert await b.resolve_account() == "434306221"

    async def test_pin_not_in_listing_refuses(self, monkeypatch):
        monkeypatch.setenv("OSIRIS_ACCOUNT_NUMBER", "111111111")
        b = self.broker({"listAccounts": self.listing()})

        assert await b.resolve_account() is None

    async def test_multiple_unidentifiable_accounts_refuse_to_guess(self):
        b = self.broker(
            {
                "listAccounts": {
                    "results": [
                        {"account_number": "111222333"},
                        {"account_number": "444555666"},
                    ]
                }
            }
        )

        assert await b.resolve_account() is None

    async def test_single_account_is_used_without_markers(self):
        b = self.broker(
            {"listAccounts": {"results": [{"account_number": "777888999"}]}}
        )

        assert await b.resolve_account() == "777888999"

    async def test_equity_is_found_inside_an_envelope(self):
        b = self.broker(
            {
                "listAccounts": {"results": [{"account_number": "A"}]},
                "getPortfolio": {"portfolio": {"portfolio_value": 12_345.0}},
            }
        )

        assert await b.get_account_equity() == pytest.approx(12_345.0)

    async def test_missing_equity_raises_rather_than_defaulting(self):
        """Every risk limit is a fraction of equity.

        Defaulting to any number would silently produce invented position sizes.
        """
        b = self.broker(
            {
                "listAccounts": {"results": [{"account_number": "A"}]},
                "getPortfolio": {"unrelated": 1},
            }
        )
        with pytest.raises(RuntimeError, match="no recognizable equity field"):
            await b.get_account_equity()


class TestAccountNumberDiscovery:
    """Regression: the response envelope is not in the tool's schema.

    A tool's `inputSchema` documents only what it ACCEPTS. Robinhood returns
    accounts under `{"data": [...], "guide": ...}`, while the obvious guesses were
    `accounts` and `results`. Rather than chase envelope names one failure at a
    time, the finder walks the structure and validates candidates by shape.
    """

    def find(self, payload):
        from osiris.execution.mcp_broker import _find_account_number

        return _find_account_number(payload)

    def test_finds_the_number_in_robinhoods_data_envelope(self):
        payload = {"data": [{"account_number": "123456789"}], "guide": "some text"}

        assert self.find(payload) == "123456789"

    @pytest.mark.parametrize(
        "payload",
        [
            {"account_number": "123456789"},
            {"results": [{"account_number": "123456789"}]},
            {"accounts": [{"account_number": "123456789"}]},
            {"data": {"accounts": [{"account_number": "123456789"}]}},
            [{"account_number": "123456789"}],
        ],
    )
    def test_survives_envelope_variation(self, payload):
        assert self.find(payload) == "123456789"

    def test_a_uuid_is_not_mistaken_for_an_account_number(self):
        """Records carry both `id` (a UUID) and `account_number`.

        Sending a UUID fails server-side with the same unhelpful "missing
        required" error as sending nothing, so accepting one would convert a clear
        failure into a confusing one.
        """
        payload = {
            "data": [
                {
                    "id": "2f9a1c44-7b3e-4d21-9c88-abc123456789",
                    "url": "https://api.robinhood.com/accounts/x/",
                }
            ]
        }

        assert self.find(payload) is None

    def test_prefers_account_number_over_a_sibling_id(self):
        payload = {
            "data": [
                {"id": "2f9a1c44-7b3e-4d21-9c88-abc", "account_number": "555000111"}
            ]
        }

        assert self.find(payload) == "555000111"

    def test_an_empty_account_list_yields_none(self):
        assert self.find({"data": [], "guide": "text"}) is None

    def test_recursion_is_bounded(self):
        """A deeply nested or cyclic-looking payload must not hang the connect."""
        payload: dict = {"data": {}}
        node = payload["data"]
        for _ in range(40):
            node["data"] = {}
            node = node["data"]
        node["account_number"] = "123456789"

        # Depth-limited, so this returns rather than recursing forever.
        assert self.find(payload) is None

    async def test_resolution_uses_the_recursive_finder(self):
        from osiris.execution.mcp_broker import MCPBroker

        broker = MCPBroker(
            FakeAdapter(
                {"listAccounts": {"data": [{"account_number": "987654321"}], "guide": "x"}}
            )
        )

        assert await broker.resolve_account() == "987654321"


class TestEquityDiscovery:
    """Finding total account value in an undocumented response envelope.

    `get_portfolio` is documented as a "market value breakdown by asset type and
    buying power", so several plausible numbers coexist in one response. Choosing
    among them is not cosmetic: every risk limit is a fraction of equity, so the
    wrong field silently rescales every position size.
    """

    def find(self, payload):
        from osiris.execution.mcp_broker import EQUITY_KEYS, _find_number

        return _find_number(payload, EQUITY_KEYS)

    def test_finds_equity_in_robinhoods_data_envelope(self):
        assert self.find({"data": {"total_equity": 12_345.67}, "guide": "t"}) == pytest.approx(
            12_345.67
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"equity": 1000.0},
            {"data": {"equity": 1000.0}},
            {"data": [{"equity": 1000.0}]},
            {"results": [{"portfolio_value": 1000.0}]},
        ],
    )
    def test_survives_envelope_variation(self, payload):
        assert self.find(payload) == pytest.approx(1000.0)

    def test_parses_decimal_strings(self):
        """Robinhood returns money as strings in several places."""
        assert self.find({"data": {"equity": "1,234.56"}}) == pytest.approx(1234.56)

    def test_prefers_total_equity_over_holdings_market_value(self):
        """A market-value field excludes cash and understates the account.

        Sizing against it would under-invest silently, and the number looks
        entirely plausible.
        """
        payload = {"data": {"market_value": 500.0, "total_equity": 1500.0}}

        assert self.find(payload) == pytest.approx(1500.0)

    def test_buying_power_is_never_used_as_equity(self):
        """On margin, buying power can be ~2x equity.

        Treating it as equity would double every position size while every
        percentage-based gate still reported compliance.
        """
        assert self.find({"data": {"buying_power": 999.0}, "guide": "t"}) is None

    def test_booleans_are_not_treated_as_numbers(self):
        assert self.find({"data": {"equity": True}}) is None

    async def test_a_missing_equity_field_reports_the_response_shape(self):
        """The error must name the paths that exist, not just top-level keys.

        `Received keys: ['data', 'guide']` was true and useless; identifying the
        real field took another full round trip because of it.
        """
        from osiris.execution.mcp_broker import MCPBroker

        broker = MCPBroker(
            FakeAdapter(
                {
                    "listAccounts": {"data": [{"account_number": ACCOUNT}]},
                    "getPortfolio": {"data": {"buying_power": 100.0}, "guide": "t"},
                }
            )
        )
        with pytest.raises(RuntimeError) as exc:
            await broker.get_account_equity()

        message = str(exc.value)
        assert "buying_power" in message, "must show what the response DID contain"
        assert "Looked for" in message

    def test_shape_description_omits_values(self):
        """Structure is the diagnostic; values would leak balances into logs."""
        from osiris.execution.mcp_broker import describe_shape

        lines = describe_shape({"data": {"equity": 12345.67}})

        assert any("equity: float" in line for line in lines)
        assert not any("12345" in line for line in lines)


class TestToolSchemaExtraction:
    """Regression: the root cause that hid the account_number failure.

    SDK v2 exposes the schema as `input_schema` (wire alias `inputSchema`).
    Reading only the camelCase attribute produced an EMPTY schema for all 53
    tools, which disabled client-side argument validation completely. The snapshot
    then claimed nothing was required while the server rejected calls for missing
    required properties.
    """

    class Tool:
        def __init__(self, name: str, schema: dict, attr: str) -> None:
            self.name = name
            self.description = ""
            setattr(self, attr, schema)

    class Page:
        # Name dictated by the MCP wire protocol, not our style.
        nextCursor = None  # noqa: N815

        def __init__(self, tools) -> None:
            self.tools = tools

    class Session:
        def __init__(self, page) -> None:
            self.page = page

        async def list_tools(self, params=None):
            return self.page

    async def _enumerate(self, attr: str):
        from osiris.mcp.client import enumerate_tools

        schema = {
            "required": ["account_number"],
            "properties": {"account_number": {"type": "string"}},
        }
        page = self.Page([self.Tool("get_portfolio", schema, attr)])
        return await enumerate_tools(self.Session(page))

    async def test_reads_the_snake_case_attribute(self):
        tools = await self._enumerate("input_schema")

        assert tools[0].required == ["account_number"]

    async def test_still_reads_the_camel_case_alias(self):
        """Compatibility with SDK 1.x, which used the wire name directly."""
        tools = await self._enumerate("inputSchema")

        assert tools[0].required == ["account_number"]

    async def test_a_required_property_is_actually_enforced(self):
        """Proves the schema is wired to validation, not merely stored."""
        from osiris.mcp.capabilities import CapabilityRegistry

        tools = await self._enumerate("input_schema")
        registry = CapabilityRegistry(tools)

        with pytest.raises(ValueError, match="account_number"):
            registry.validate_args(tools[0], {})


class TestCapabilityResolution:
    """Regression: capability predicates must select the EQUITY tool."""

    def registry(self, names: list[str]):
        from osiris.mcp.capabilities import CapabilityRegistry, ToolSpec

        return CapabilityRegistry([ToolSpec(n, "", {}) for n in names])

    def test_historicals_prefers_equity_over_index(self):
        """`get_index_historicals` is shorter and won the tiebreak.

        That would have fed index prices into every symbol's momentum, volatility,
        and beta -- all wrong, all plausible-looking.
        """
        reg = self.registry(
            ["get_equity_historicals", "get_index_historicals", "get_option_historicals"]
        )

        assert reg.resolve("getHistoricals").name == "get_equity_historicals"

    def test_quotes_prefers_equity_over_index_and_options(self):
        reg = self.registry(
            ["get_equity_quotes", "get_index_quotes", "get_option_quotes"]
        )

        assert reg.resolve("getQuotes").name == "get_equity_quotes"

    def test_place_order_never_resolves_to_options(self):
        reg = self.registry(["place_equity_order", "place_option_order"])

        assert reg.resolve("placeOrder").name == "place_equity_order"

    def test_review_is_required_and_resolves_to_equity(self):
        reg = self.registry(["review_equity_order", "review_option_order"])

        assert reg.resolve("reviewOrder").name == "review_equity_order"

    def test_the_real_robinhood_surface_resolves_correctly(self):
        """Pinned against the actual 53-tool surface from a live account."""
        reg = self.registry(
            [
                "get_accounts",
                "get_portfolio",
                "get_equity_positions",
                "get_equity_quotes",
                "get_index_quotes",
                "get_equity_historicals",
                "get_index_historicals",
                "get_option_historicals",
                "review_equity_order",
                "place_equity_order",
                "place_option_order",
                "cancel_equity_order",
            ]
        )
        expected = {
            "listAccounts": "get_accounts",
            "getPortfolio": "get_portfolio",
            "listPositions": "get_equity_positions",
            "getQuotes": "get_equity_quotes",
            "getHistoricals": "get_equity_historicals",
            "reviewOrder": "review_equity_order",
            "placeOrder": "place_equity_order",
            "cancelOrder": "cancel_equity_order",
        }
        for capability, tool in expected.items():
            assert reg.resolve(capability).name == tool, capability


class TestQuoteEnvelopeParsing:
    """Regression: quotes returned 0/61 while history returned 60/60.

    Same account, same envelope. `_rows` checked a fixed set of top-level keys and
    silently returned nothing when quotes were nested -- and the caller logged that
    as "quotes missing", which is indistinguishable from a market with no data.
    """

    def rows(self, payload):
        from osiris.data.live import _quote_rows

        return _quote_rows(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            {"results": [{"symbol": "AAPL", "last_trade_price": 100.0}]},
            {"data": [{"symbol": "AAPL", "last_trade_price": 100.0}], "guide": "t"},
            {"quotes": [{"symbol": "AAPL", "last_trade_price": 100.0}]},
            {"data": {"quotes": [{"symbol": "AAPL", "mark_price": 100.0}]}},
            [{"symbol": "AAPL", "price": 100.0}],
        ],
    )
    def test_survives_envelope_variation(self, payload):
        rows = self.rows(payload)

        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"

    def test_a_symbol_keyed_mapping_recovers_the_symbol_from_the_key(self):
        """`{"AAPL": {...prices...}}` carries no symbol inside the record.

        Without lifting the key, the quote parses and is then discarded for
        lacking a symbol -- a silent loss rather than an error.
        """
        rows = self.rows(
            {"data": {"AAPL": {"last_trade_price": 100.0}, "MSFT": {"last_trade_price": 200.0}}}
        )

        assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}

    def test_a_record_without_prices_is_not_a_quote(self):
        """Envelope objects that merely mention a symbol must not be collected."""
        assert self.rows({"data": [{"symbol": "AAPL", "name": "Apple Inc"}]}) == []

    async def test_an_unparseable_response_is_logged_as_a_parse_failure(self):
        """A parse bug and an empty market must not look the same.

        This is what cost a full round trip: 0 quotes was reported as "missing"
        when the real cause was an unrecognized envelope.
        """
        adapter = FakeAdapter({"getQuotes": {"unexpected": {"nested": "shape"}}})
        market = LiveMarket(adapter)

        assert await market.quotes(["AAPL"]) == {}
        # The call succeeded, so the failure is in parsing, not availability.
        assert dict(adapter.calls)["getQuotes"]["symbols"] == ["AAPL"]


class TestRequestShapes:
    """Regression: request arguments must match the live tool schemas.

    The first real cycle resolved 0 quotes and 0 history because the code sent
    `symbol` (singular) and a `span` parameter that does not exist, and omitted the
    REQUIRED `start_time`. Tests passed throughout because the fixtures encoded
    the same wrong shapes.

    These validate against the captured snapshot where available, so they break if
    Robinhood changes the contract rather than silently going stale.
    """

    @pytest.fixture
    def registry(self):
        from osiris.mcp.capabilities import CapabilityRegistry
        from osiris.mcp.client import load_snapshot

        try:
            return CapabilityRegistry(load_snapshot())
        except FileNotFoundError:
            pytest.skip("no MCP snapshot; run `python -m osiris.connect`")

    async def _args_for(self, capability: str, call) -> dict:
        adapter = FakeAdapter({capability: {"results": []}})
        await call(LiveMarket(adapter))
        return dict(adapter.calls)[capability]

    async def test_quotes_send_a_symbols_array(self, registry):
        args = await self._args_for(
            "getQuotes", lambda m: m.quotes(["AAPL", "MSFT"])
        )

        assert args["symbols"] == ["AAPL", "MSFT"]
        assert "symbol" not in args
        registry.validate_args(registry.resolve("getQuotes"), args)

    async def test_history_sends_symbols_and_required_start_time(self, registry):
        args = await self._args_for("getHistoricals", lambda m: m.history(["AAPL"]))

        assert args["symbols"] == ["AAPL"]
        assert "start_time" in args, "start_time is REQUIRED by the schema"
        # `span` was invented; the schema has no such parameter.
        assert "span" not in args
        registry.validate_args(registry.resolve("getHistoricals"), args)

    async def test_history_start_time_is_rfc3339_utc(self):
        from datetime import datetime

        args = await self._args_for("getHistoricals", lambda m: m.history(["AAPL"]))

        # Raises if the format is not what the schema documents.
        datetime.strptime(args["start_time"], "%Y-%m-%dT%H:%M:%SZ")

    async def test_history_requests_split_adjusted_prices(self):
        """Unadjusted prices contain synthetic gaps at every split.

        Momentum and volatility would read those gaps as real moves.
        """
        args = await self._args_for("getHistoricals", lambda m: m.history(["AAPL"]))

        assert args["adjustment_type"] == "split"

    async def test_fundamentals_send_a_symbols_array(self, registry):
        args = await self._args_for(
            "getFundamentals", lambda m: m.sectors(["AAPL", "MSFT"])
        )

        assert args["symbols"] == ["AAPL", "MSFT"]
        registry.validate_args(registry.resolve("getFundamentals"), args)

    async def test_run_scan_sends_a_scan_id(self, registry):
        adapter = FakeAdapter(
            {
                "listScans": {"scans": [{"scan_id": "scan-1"}]},
                "runScan": {"results": []},
            }
        )
        await LiveMarket(adapter).universe(fallback=["AAPL"])
        args = dict(adapter.calls)["runScan"]

        assert args["scan_id"] == "scan-1"
        registry.validate_args(registry.resolve("runScan"), args)


class TestBatching:
    """Batch ceilings are per-tool and enforced server-side.

    Historicals and fundamentals cap at 10 symbols; quotes above 20 silently omit
    `closes`. Exceeding a ceiling returns partial data rather than an error, which
    is the harder failure to notice.
    """

    async def test_quotes_batch_at_twenty(self):
        adapter = FakeAdapter({"getQuotes": {"results": []}})
        await LiveMarket(adapter).quotes([f"S{i}" for i in range(45)])

        sizes = [len(a["symbols"]) for c, a in adapter.calls if c == "getQuotes"]
        assert len(sizes) == 3
        assert max(sizes) <= 20

    async def test_history_batches_at_ten(self):
        adapter = FakeAdapter({"getHistoricals": {"results": []}})
        await LiveMarket(adapter).history([f"S{i}" for i in range(25)])

        sizes = [len(a["symbols"]) for c, a in adapter.calls if c == "getHistoricals"]
        assert len(sizes) == 3
        assert max(sizes) <= 10

    async def test_fundamentals_batch_at_ten(self):
        adapter = FakeAdapter({"getFundamentals": {"results": []}})
        await LiveMarket(adapter).sectors([f"S{i}" for i in range(22)])

        sizes = [len(a["symbols"]) for c, a in adapter.calls if c == "getFundamentals"]
        assert max(sizes) <= 10


class TestMultiSymbolParsing:
    """The response groups bars per symbol; a flat parse loses attribution."""

    def parse(self, payload, requested):
        from osiris.data.live import _parse_history

        return _parse_history(payload, requested)

    def test_parses_records_keyed_by_symbol(self):
        closes = [{"close_price": 100.0 + i} for i in range(40)]
        payload = {
            "data": [
                {"symbol": "AAPL", "historicals": closes},
                {"symbol": "MSFT", "historicals": closes},
            ]
        }
        out = self.parse(payload, ["AAPL", "MSFT"])

        assert set(out) == {"AAPL", "MSFT"}
        assert out["AAPL"].size == 40

    def test_parses_a_symbol_to_bars_mapping(self):
        closes = [{"close_price": 100.0 + i} for i in range(40)]
        out = self.parse({"AAPL": closes}, ["AAPL"])

        assert out["AAPL"].size == 40

    def test_never_merges_two_symbols_into_one_series(self):
        """The worst failure mode: one symbol's prices attributed to another.

        Momentum would be computed across a discontinuity between two unrelated
        instruments, and the number would look ordinary.
        """
        a = [{"close_price": 100.0} for _ in range(40)]
        b = [{"close_price": 500.0} for _ in range(40)]
        out = self.parse(
            {"data": [{"symbol": "AAPL", "historicals": a},
                      {"symbol": "MSFT", "historicals": b}]},
            ["AAPL", "MSFT"],
        )

        assert out["AAPL"].max() == 100.0
        assert out["MSFT"].min() == 500.0

    def test_short_series_are_dropped(self):
        payload = {"data": [{"symbol": "AAPL", "historicals": [{"close_price": 1.0}]}]}

        assert self.parse(payload, ["AAPL"]) == {}


class TestBetaEstimation:
    def test_defaults_to_one_without_a_benchmark(self):
        """Neutral, not optimistic.

        Guessing a low beta would let the beta budget be breached silently, which
        is the direction that costs money.
        """
        import numpy as np

        closes = {"A": np.linspace(100, 120, 100)}
        assert _estimate_betas(closes, None) == {"A": 1.0}

    def test_recovers_a_known_beta(self):
        import numpy as np

        rng = np.random.default_rng(4)
        market_returns = rng.normal(0.0004, 0.01, 300)
        bench = 100 * np.cumprod(1 + market_returns)
        levered = 100 * np.cumprod(1 + 2.0 * market_returns)

        beta = _estimate_betas({"L": levered}, bench)["L"]
        assert 1.7 < beta < 2.3

    def test_clamps_absurd_estimates(self):
        import numpy as np

        rng = np.random.default_rng(5)
        market_returns = rng.normal(0, 0.001, 200)
        bench = 100 * np.cumprod(1 + market_returns)
        wild = 100 * np.cumprod(1 + 60.0 * market_returns)

        assert _estimate_betas({"W": wild}, bench)["W"] <= 3.0
