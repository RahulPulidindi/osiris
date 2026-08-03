"""Order arguments must match the account's real `place_equity_order` schema.

Every assertion here was derived from the live schema captured off a real
Robinhood account, not from assumption. The original code sent
`amount_in_dollars` and `client_order_id`, neither of which exists; with
`additionalProperties: false` that rejects the entire order. Nothing caught it
because the schema was being read from the wrong attribute and came back empty,
so client-side validation had nothing to check against.

The tests validate against the captured schema where it is available, which means
they fail if Robinhood changes the contract -- the point of keeping a snapshot.
"""

from __future__ import annotations

import pytest

from osiris.execution.broker import OrderRejected, OrderRequest
from osiris.execution.mcp_broker import MCPBroker, _as_uuid
from osiris.types import OrderKind, Side

ACCOUNT = "123456789"


@pytest.fixture
def broker():
    return MCPBroker(None, account_number=ACCOUNT)


@pytest.fixture
def schema():
    """The live `place_equity_order` schema, if a snapshot exists."""
    from osiris.mcp.capabilities import CapabilityRegistry
    from osiris.mcp.client import load_snapshot

    try:
        registry = CapabilityRegistry(load_snapshot())
        return registry, registry.resolve("placeOrder")
    except FileNotFoundError:
        pytest.skip("no MCP snapshot; run `python -m osiris.connect` to capture one")


def market(**kw) -> OrderRequest:
    return OrderRequest(
        symbol=kw.get("symbol", "AAPL"),
        side=kw.get("side", Side.BUY),
        notional_usd=kw.get("notional_usd", 100.0),
        kind=OrderKind.MARKET,
        idempotency_key=kw.get("idempotency_key", "key-1"),
    )


def limit(**kw) -> OrderRequest:
    return OrderRequest(
        symbol=kw.get("symbol", "AAPL"),
        side=kw.get("side", Side.BUY),
        notional_usd=kw.get("notional_usd", 500.0),
        kind=OrderKind.LIMIT,
        limit_price=kw.get("limit_price", 100.0),
        idempotency_key=kw.get("idempotency_key", "key-1"),
    )


class TestFieldNames:
    def test_does_not_send_nonexistent_fields(self, broker):
        """`additionalProperties: false` means an unknown key kills the order."""
        args = broker._order_args(market())

        assert "amount_in_dollars" not in args
        assert "client_order_id" not in args

    def test_sends_the_required_fields(self, broker):
        args = broker._order_args(market())

        for field in ("account_number", "symbol", "side", "type"):
            assert field in args

    def test_validates_against_the_live_schema(self, broker, schema):
        registry, tool = schema

        registry.validate_args(tool, broker._order_args(market()))
        registry.validate_args(tool, broker._order_args(limit()))


class TestStringTyping:
    """The schema types every numeric order field as a STRING, not a number."""

    def test_dollar_amount_is_a_string(self, broker):
        args = broker._order_args(market(notional_usd=7.32))

        assert args["dollar_amount"] == "7.32"

    def test_quantity_is_a_string(self, broker):
        args = broker._order_args(limit(notional_usd=500.0, limit_price=100.0))

        assert args["quantity"] == "5"

    def test_limit_price_is_a_string_with_two_decimals(self, broker):
        args = broker._order_args(limit(limit_price=100.0))

        assert args["limit_price"] == "100.00"


class TestOrderTypeRules:
    def test_market_orders_use_dollar_amount(self, broker):
        """Dollar notional is how fractional shares are bought."""
        args = broker._order_args(market(notional_usd=7.32))

        assert args["type"] == "market"
        assert args["dollar_amount"] == "7.32"
        assert "quantity" not in args

    def test_limit_orders_use_share_quantity(self, broker):
        """`dollar_amount` is documented as valid only with type=market."""
        args = broker._order_args(limit(notional_usd=500.0, limit_price=100.0))

        assert args["type"] == "limit"
        assert "dollar_amount" not in args
        assert args["quantity"] == "5"

    def test_share_count_floors_rather_than_rounds_up(self, broker):
        """Rounding up would exceed the notional the kernel approved."""
        args = broker._order_args(limit(notional_usd=550.0, limit_price=100.0))

        assert args["quantity"] == "5"


class TestSmallOrderHandling:
    """A $366 account places $7 orders; most shares cost more than that."""

    def test_an_undersized_limit_becomes_a_market_order(self, broker):
        """Fractional quantities require a market order, so convert.

        Without this, every entry on a small account is rejected: one share of a
        $250 stock cannot be bought with $7.32, and quantity=0 is refused.
        """
        args = broker._order_args(limit(notional_usd=7.32, limit_price=250.0))

        assert args["type"] == "market"
        assert args["dollar_amount"] == "7.32"
        assert "limit_price" not in args
        assert "quantity" not in args

    def test_the_converted_order_still_validates(self, broker, schema):
        registry, tool = schema
        args = broker._order_args(limit(notional_usd=7.32, limit_price=250.0))

        registry.validate_args(tool, args)

    def test_strict_mode_refuses_instead_of_converting(self):
        """Opt out when silently changing order type is unacceptable."""
        strict = MCPBroker(
            None, account_number=ACCOUNT, allow_fractional_fallback=False
        )
        with pytest.raises(OrderRejected, match="less than one share"):
            strict._order_args(limit(notional_usd=7.32, limit_price=250.0))

    def test_a_sufficient_limit_order_is_left_alone(self, broker):
        args = broker._order_args(limit(notional_usd=500.0, limit_price=100.0))

        assert args["type"] == "limit"


class TestIdempotency:
    def test_ref_id_is_a_valid_uuid(self, broker):
        """The venue's duplicate protection requires UUID form."""
        import uuid

        args = broker._order_args(market(idempotency_key="abc123"))

        uuid.UUID(args["ref_id"])  # raises if malformed

    def test_ref_id_is_deterministic(self):
        """A retry must present the SAME ref_id to be deduplicated.

        A random UUID per attempt would defeat the entire mechanism and allow a
        retry to place a second live order.
        """
        assert _as_uuid("abc123") == _as_uuid("abc123")

    def test_different_orders_get_different_ref_ids(self):
        assert _as_uuid("order-1") != _as_uuid("order-2")

    def test_ref_id_is_omitted_without_an_idempotency_key(self, broker):
        request = OrderRequest(
            symbol="AAPL",
            side=Side.BUY,
            notional_usd=100.0,
            kind=OrderKind.MARKET,
            idempotency_key="",
        )

        assert "ref_id" not in broker._order_args(request)


class TestSellOrders:
    def test_sell_side_is_passed_through(self, broker):
        args = broker._order_args(market(side=Side.SELL))

        assert args["side"] == "sell"

    def test_a_sell_validates_against_the_schema(self, broker, schema):
        registry, tool = schema

        registry.validate_args(tool, broker._order_args(market(side=Side.SELL)))
