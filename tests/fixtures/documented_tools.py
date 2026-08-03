"""The Robinhood Agentic tool surface as publicly documented (2026-05-26).

This is a FIXTURE for offline testing, not a source of truth. The live surface is
account-specific and must be enumerated via `python -m osiris.mcp.enumerate`.
Its purpose is to prove the capability resolver works against a realistic shape
without a live connection.
"""

from __future__ import annotations

from osiris.mcp.capabilities import ToolSpec

_SYMBOL_SCHEMA = {
    "type": "object",
    "properties": {"symbols": {"type": "array"}},
    "required": ["symbols"],
}
_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "quantity": {"type": "number"},
        "notional_usd": {"type": "number"},
        "order_type": {"type": "string", "enum": ["market", "limit"]},
        "limit_price": {"type": "number"},
        "time_in_force": {"type": "string"},
    },
    "required": ["symbol", "side"],
}

_NAMES: list[tuple[str, str, dict]] = [
    # Account / portfolio
    ("get_accounts", "View all your Robinhood accounts", {"type": "object", "properties": {}}),
    ("get_portfolio", "Portfolio snapshot incl. value and buying power", {"type": "object", "properties": {}}),
    ("get_realized_pnl", "Realized P&L over a window", {"type": "object", "properties": {"span": {"type": "string"}}}),
    ("get_pnl_trade_history", "Trade-by-trade realized P&L", {"type": "object", "properties": {}}),
    ("search", "Map company name to ticker", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    # Market data
    ("get_equity_historicals", "OHLCV price bars across a time range", _SYMBOL_SCHEMA),
    ("get_equity_fundamentals", "Valuation ratios, market cap, 52wk range", _SYMBOL_SCHEMA),
    ("get_financials", "Reported financials over time", _SYMBOL_SCHEMA),
    ("get_equity_price_book", "Real-time Level 2 order book, up to 4 stocks", _SYMBOL_SCHEMA),
    ("get_equity_technical_indicators", "Compute RSI, MACD, Bollinger, MAs", _SYMBOL_SCHEMA),
    ("get_earnings_results", "Earnings history and next report", _SYMBOL_SCHEMA),
    ("get_earnings_calendar", "Earnings scheduled across the market", {"type": "object", "properties": {"start_date": {"type": "string"}}}),
    ("get_indexes", "Look up market indexes", _SYMBOL_SCHEMA),
    ("get_index_quotes", "Real-time index values", _SYMBOL_SCHEMA),
    # Equities
    ("get_equity_positions", "Open equity positions with cost basis", {"type": "object", "properties": {}}),
    ("get_equity_tax_lots", "Open tax lots for an equity holding", _SYMBOL_SCHEMA),
    ("get_equity_quotes", "Real-time quotes for up to 20 symbols", _SYMBOL_SCHEMA),
    ("get_equity_orders", "Equity order status history", {"type": "object", "properties": {}}),
    ("get_equity_tradability", "Whether a symbol can be traded", _SYMBOL_SCHEMA),
    ("review_equity_order", "Simulate an equity order, pre-trade warnings", _ORDER_SCHEMA),
    ("place_equity_order", "Place an equity order", _ORDER_SCHEMA),
    ("cancel_equity_order", "Cancel an open equity order", {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}),
    # Options
    ("get_option_chains", "Load option chains", _SYMBOL_SCHEMA),
    ("get_option_quotes", "Real-time option quotes", _SYMBOL_SCHEMA),
    ("get_option_positions", "Open or closed options positions", {"type": "object", "properties": {}}),
    ("get_option_orders", "Options order history", {"type": "object", "properties": {}}),
    ("review_option_order", "Simulate an options order", _ORDER_SCHEMA),
    ("place_option_order", "Place a real options order", _ORDER_SCHEMA),
    ("cancel_option_order", "Cancel an open options order", {"type": "object", "properties": {"order_id": {"type": "string"}}}),
    # Scanner
    ("get_scans", "List your saved scans", {"type": "object", "properties": {}}),
    ("get_scanner_filter_specs", "List available scanner filters", {"type": "object", "properties": {}}),
    ("create_scan", "Create a new scan", {"type": "object", "properties": {"name": {"type": "string"}, "filters": {"type": "object"}}}),
    ("run_scan", "Run a saved scan for live results", {"type": "object", "properties": {"scan_id": {"type": "string"}}}),
    ("update_scan_filters", "Change filters on a saved scan", {"type": "object", "properties": {"scan_id": {"type": "string"}}}),
    # Watchlists
    ("get_watchlists", "List watchlists", {"type": "object", "properties": {}}),
    ("add_to_watchlist", "Add symbols to a watchlist", _SYMBOL_SCHEMA),
    ("remove_from_watchlist", "Remove symbols from a watchlist", _SYMBOL_SCHEMA),
]

DOCUMENTED_TOOLS: list[ToolSpec] = [
    ToolSpec(name=n, description=d, input_schema=s) for n, d, s in _NAMES
]
