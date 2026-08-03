"""Capability resolution must survive tool renames and never misclassify a write."""

from __future__ import annotations

import pytest

from osiris.mcp.capabilities import (
    CapabilityRegistry,
    ToolSpec,
    ToolUnavailable,
    is_write,
)
from tests.fixtures.documented_tools import DOCUMENTED_TOOLS


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry(DOCUMENTED_TOOLS)


class TestWriteClassification:
    """Misclassifying a write as a read skips the risk kernel entirely."""

    @pytest.mark.parametrize(
        "name",
        [
            "place_equity_order",
            "place_option_order",
            "cancel_equity_order",
            "cancel_option_order",
            "create_scan",
            "add_to_watchlist",
            "remove_from_watchlist",
            "update_scan_filters",
        ],
    )
    def test_writes_detected(self, name: str) -> None:
        tool = next(t for t in DOCUMENTED_TOOLS if t.name == name)
        assert is_write(tool), f"{name} must be classified as a write"

    @pytest.mark.parametrize(
        "name",
        [
            "get_equity_quotes",
            "get_portfolio",
            "get_equity_positions",
            "review_equity_order",  # simulation only, places nothing
            "review_option_order",
            "run_scan",
            "search",
            "get_equity_orders",
        ],
    )
    def test_reads_not_flagged_as_writes(self, name: str) -> None:
        tool = next(t for t in DOCUMENTED_TOOLS if t.name == name)
        assert not is_write(tool)

    def test_unknown_tool_defaults_conservatively(self) -> None:
        """An unrecognized name containing a write hint counts as a write."""
        mystery = ToolSpec(name="execute_basket_trade", description="", input_schema={})
        assert is_write(mystery)


class TestCapabilityResolution:
    def test_critical_capabilities_resolve(self, registry: CapabilityRegistry) -> None:
        for cap in [
            "placeOrder",
            "reviewOrder",
            "cancelOrder",
            "listPositions",
            "getPortfolio",
            "getQuotes",
            "runScan",
            "getTradability",
            "getEarningsCalendar",
            "getTaxLots",
        ]:
            assert registry.has(cap), f"{cap} must resolve against documented surface"

    def test_equity_preferred_over_option(self, registry: CapabilityRegistry) -> None:
        """v1 is equities-only; option tools must not satisfy equity capabilities."""
        assert registry.resolve("placeOrder").name == "place_equity_order"
        assert registry.resolve("reviewOrder").name == "review_equity_order"
        assert registry.resolve("cancelOrder").name == "cancel_equity_order"

    def test_resolves_by_shape_after_rename(self) -> None:
        """The whole point: a renamed tool still resolves."""
        renamed = [
            ToolSpec(
                name="submit_equity_trade_v2",
                description="Place an equity order",
                input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
            ),
            ToolSpec(name="preview_equity_trade_v2", description="Simulate", input_schema={}),
        ]
        reg = CapabilityRegistry(renamed)
        assert reg.resolve("placeOrder").name == "submit_equity_trade_v2"
        assert reg.resolve("reviewOrder").name == "preview_equity_trade_v2"

    def test_missing_capability_raises_with_context(self) -> None:
        reg = CapabilityRegistry([ToolSpec(name="get_portfolio", description="", input_schema={})])
        with pytest.raises(ToolUnavailable) as exc:
            reg.resolve("placeOrder")
        assert "placeOrder" in str(exc.value)
        assert not reg.has("placeOrder")

    def test_unknown_capability_name_is_an_error(self, registry: CapabilityRegistry) -> None:
        with pytest.raises(ValueError, match="Unknown capability"):
            registry.resolve("teleportFunds")

    def test_resolution_is_deterministic(self, registry: CapabilityRegistry) -> None:
        assert registry.resolve("getQuotes").name == registry.resolve("getQuotes").name


class TestArgumentValidation:
    def test_missing_required_rejected(self, registry: CapabilityRegistry) -> None:
        tool = registry.resolve("placeOrder")
        with pytest.raises(ValueError, match="missing required"):
            registry.validate_args(tool, {"symbol": "AAPL"})  # no side

    def test_enum_violation_rejected(self, registry: CapabilityRegistry) -> None:
        tool = registry.resolve("placeOrder")
        with pytest.raises(ValueError, match="must be one of"):
            registry.validate_args(tool, {"symbol": "AAPL", "side": "short"})

    def test_type_mismatch_rejected(self, registry: CapabilityRegistry) -> None:
        tool = registry.resolve("placeOrder")
        with pytest.raises(ValueError, match="should be number"):
            registry.validate_args(tool, {"symbol": "AAPL", "side": "buy", "quantity": "ten"})

    def test_bool_is_not_a_number(self, registry: CapabilityRegistry) -> None:
        """Python bool is an int subclass; a quantity of True must not pass."""
        tool = registry.resolve("placeOrder")
        with pytest.raises(ValueError):
            registry.validate_args(tool, {"symbol": "AAPL", "side": "buy", "quantity": True})

    def test_valid_args_pass(self, registry: CapabilityRegistry) -> None:
        tool = registry.resolve("placeOrder")
        args = {"symbol": "AAPL", "side": "buy", "notional_usd": 500.0, "order_type": "limit"}
        assert registry.validate_args(tool, args) == args
