"""Capability resolution against the live MCP schema.

The Robinhood Trading MCP tool surface is account-specific, paginated, and
actively changing as asset classes are added. Binding a method to a tool *name*
is therefore a latent break: `place_equity_order` works until the day it is
renamed, and then fails at trade time.

So we match on capability -- a predicate over the advertised schema -- and fail
loudly when nothing satisfies it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A tool as advertised by `tools/list`."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.input_schema.get("properties", {}))


# Conservative write detection. Unknown counts as a write.
#
# A misclassification toward "write" costs only a redundant guard check; a
# misclassification toward "read" skips the risk kernel entirely. The asymmetry
# dictates the default.
_READ_PREFIX = re.compile(
    r"^(get|list|read|fetch|search|review|simulate|preview|run)_", re.I
)
_WRITE_HINT = re.compile(
    r"(place|submit|buy|sell|cancel|create|update|follow|unfollow|add|remove|order|trade|execute)",
    re.I,
)


def is_write(tool: ToolSpec) -> bool:
    if _READ_PREFIX.match(tool.name):
        return False
    return bool(_WRITE_HINT.search(tool.name))


# Asset classes this agent does NOT trade.
#
# Osiris is a US EQUITY agent. A tool scoped to another asset class can satisfy
# a capability predicate by name alone and then be silently wrong, which is the
# worst available outcome. Live incident (2026-09-16): Robinhood grew the tool
# surface from 53 to 73 tools by adding crypto. `get_crypto_positions` and
# `get_equity_positions` are both 20 characters, so the shortest-name tiebreak
# fell through to alphabetical order and "crypto" won. The agent resolved
# `placeOrder` to `place_crypto_order` -- one step from submitting crypto
# orders for stock tickers. Only a differing required-argument name
# (`rhs_account_number`) made the position read fail loudly enough to abort.
_OTHER_ASSET_CLASSES = (
    "crypto",
    "option",
    "forex",
    "currency",
    "futures",
    "index",
    "bond",
    "treasury",
)

# Positive identifiers for the class we DO trade, used to break ties in favour
# of an explicitly-equity tool rather than relying on name length.
_EQUITY_HINT = re.compile(r"equity|stock", re.I)


def is_equity_scoped(tool: ToolSpec) -> bool:
    """True unless the tool name marks it as another asset class."""
    name = tool.name.lower()
    return not any(cls in name for cls in _OTHER_ASSET_CLASSES)


def _read(pattern: str) -> Callable[[ToolSpec], bool]:
    rx = re.compile(pattern, re.I)
    return lambda t: bool(rx.search(t.name)) and not is_write(t) and is_equity_scoped(t)


CAPABILITIES: dict[str, Callable[[ToolSpec], bool]] = {
    # Reads
    "listAccounts": _read(r"account"),
    "getPortfolio": _read(r"portfolio"),
    "listPositions": lambda t: bool(re.search(r"position|holding", t.name, re.I))
    and not is_write(t)
    and is_equity_scoped(t),
    "listOrders": lambda t: bool(re.search(r"order", t.name, re.I))
    and bool(re.search(r"get|list|search|history", t.name, re.I))
    and not is_write(t)
    and is_equity_scoped(t),
    "getQuotes": lambda t: bool(re.search(r"quote", t.name, re.I))
    and is_equity_scoped(t),
    # Asset-class scoping is load-bearing here. The shortest-name tiebreak would
    # otherwise select `get_index_historicals` over `get_equity_historicals`,
    # silently returning INDEX prices for every stock symbol -- momentum,
    # volatility, and beta all computed from the wrong series while looking valid.
    "getHistoricals": lambda t: bool(re.search(r"historical", t.name, re.I))
    and is_equity_scoped(t),
    "getFundamentals": _read(r"fundamental"),
    "getFinancials": _read(r"financial"),
    "getTechnicalIndicators": _read(r"technical_indicator"),
    "getEarningsCalendar": _read(r"earnings_calendar"),
    "getEarningsResults": _read(r"earnings_result"),
    "getTradability": _read(r"tradability"),
    "getTaxLots": _read(r"tax_lot"),
    "getRealizedPnl": _read(r"realized_pnl"),
    "getPnlHistory": _read(r"pnl_trade_history"),
    "getPriceBook": _read(r"price_book"),
    "getScannerFilterSpecs": _read(r"scanner_filter_spec"),
    "listScans": _read(r"get_scans"),
    "runScan": lambda t: bool(re.search(r"run_scan", t.name, re.I)),
    # Simulation, mandatory before any write
    "reviewOrder": lambda t: bool(
        re.search(r"review|simulate|preview|validate", t.name, re.I)
    )
    and bool(re.search(r"order|trade", t.name, re.I))
    and is_equity_scoped(t),
    # Writes. Match order|trade: a rename from `place_equity_order` to
    # `submit_equity_trade_v2` is a plausible drift and must still resolve.
    "placeOrder": lambda t: bool(re.search(r"order|trade", t.name, re.I))
    and is_write(t)
    and not re.search(r"cancel|review|watchlist|scan", t.name, re.I)
    and is_equity_scoped(t),
    "cancelOrder": lambda t: bool(re.search(r"cancel", t.name, re.I))
    and is_equity_scoped(t),
    "createScan": lambda t: bool(re.search(r"create_scan", t.name, re.I)),
}


class ToolUnavailable(RuntimeError):
    def __init__(self, capability: str, available: list[str]) -> None:
        self.capability = capability
        self.available = available
        super().__init__(
            f"No tool satisfies capability {capability!r}. "
            f"{len(available)} tools advertised: {', '.join(sorted(available)[:12])}..."
        )


class ToolCallFailed(RuntimeError):
    """Raised when a tool returns isError, which arrives with an HTTP 200.

    Code that only catches exceptions will otherwise book a *rejected order* as
    a successful fill. This is the highest-consequence silent bug in the stack.
    """

    def __init__(self, tool: str, message: str, raw: Any = None) -> None:
        self.tool = tool
        self.raw = raw
        super().__init__(f"{tool} failed: {message}")


class CapabilityRegistry:
    """Resolves capabilities against a concrete advertised tool set."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def write_tools(self) -> set[str]:
        return {n for n, t in self._tools.items() if is_write(t)}

    def resolve(self, capability: str) -> ToolSpec:
        predicate = CAPABILITIES.get(capability)
        if predicate is None:
            raise ValueError(f"Unknown capability: {capability}")
        # Deterministic, and EXPLICIT about asset class before length.
        #
        # Length alone was not enough: `get_crypto_positions` and
        # `get_equity_positions` are the same length, so ordering fell through
        # to alphabetical and selected crypto. An explicitly-equity name now
        # always outranks an unqualified one, and length only breaks ties
        # within a tier.
        matches = sorted(
            (t for t in self._tools.values() if predicate(t)),
            key=lambda t: (
                0 if _EQUITY_HINT.search(t.name) else 1,
                len(t.name),
                t.name,
            ),
        )
        if not matches:
            raise ToolUnavailable(capability, self.tool_names)
        return matches[0]

    def has(self, capability: str) -> bool:
        try:
            self.resolve(capability)
            return True
        except (ToolUnavailable, ValueError):
            return False

    def validate_args(self, tool: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        """Validate against the advertised inputSchema before sending."""
        schema = tool.input_schema or {}
        props = schema.get("properties", {})
        problems: list[str] = []

        for key in schema.get("required", []):
            if args.get(key) is None:
                problems.append(f'missing required "{key}"')

        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                if schema.get("additionalProperties") is False:
                    problems.append(f'unknown argument "{key}"')
                continue
            expected = spec.get("type")
            if expected and not _type_matches(expected, value):
                problems.append(f'"{key}" should be {expected}, got {type(value).__name__}')
            enum = spec.get("enum")
            if isinstance(enum, list) and value not in enum:
                problems.append(f'"{key}" must be one of {enum}')

        if problems:
            raise ValueError(f"invalid arguments for {tool.name}: {'; '.join(problems)}")
        return args

    def capability_report(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for cap in CAPABILITIES:
            try:
                out[cap] = self.resolve(cap).name
            except ToolUnavailable:
                out[cap] = None
        return out


def _type_matches(expected: str | list[str], value: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(e, value) for e in expected)
    match expected:
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "array":
            return isinstance(value, (list, tuple))
        case "object":
            return isinstance(value, dict)
        case "null":
            return value is None
        case _:
            return True
