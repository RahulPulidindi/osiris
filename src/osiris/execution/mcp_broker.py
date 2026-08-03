"""Live broker backed by the Robinhood MCP.

Implements the same `Broker` interface as the paper broker, so the execution
pipeline is byte-identical across modes.

Three hazards this module exists to contain:

  1. **`isError: true` arrives with HTTP 200.** A logically rejected order does
     not raise. `MCPAdapter.call` converts that into `ToolCallFailed`; here we
     make sure a failure NEVER returns an accepted PlaceResult.

  2. **Review is mandatory.** `place` calls `review` itself rather than trusting
     the caller, so there is no code path that reaches the venue unsimulated.

  3. **Ambiguous placement is not success.** If the place call fails *after* the
     order may have reached the venue, we raise `AmbiguousOrderState` rather than
     guessing. Booking a maybe-order as filled or as absent are both corrupting;
     the only safe response is to stop and reconcile.
"""

from __future__ import annotations

import json
import re
from typing import Any

from osiris.execution.broker import (
    Broker,
    OrderRejected,
    OrderRequest,
    PlaceResult,
    ReviewResult,
)
from osiris.logging import get_logger
from osiris.mcp.capabilities import ToolCallFailed
from osiris.mcp.client import text_of
from osiris.types import Fill, OrderKind

log = get_logger(__name__)


class AmbiguousOrderState(RuntimeError):
    """The order may or may not exist at the venue. Halt and reconcile.

    Deliberately not retried: a retry can duplicate a live order, and assuming
    failure can leave an unmanaged position. A human must look.
    """


def _extract_payload(result: Any) -> dict:
    """MCP results carry text content; most tools return JSON inside it."""
    text = text_of(result)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    return payload if isinstance(payload, dict) else {"data": payload}


# Keys that carry an account number, in preference order.
_ACCOUNT_KEYS = ("account_number", "accountNumber", "account_id", "number")

# Fields that may carry total account value, in PREFERENCE order.
#
# Ordering is load-bearing. `get_portfolio` is documented as a "market value
# breakdown by asset type and buying power", so several plausible numbers coexist
# in one response and picking the wrong one misstates equity:
#   - a *_market_value field covers only holdings, excluding cash
#   - buying_power on a margin account can be roughly 2x equity
# Total-account fields are therefore tried first, and buying power is never used
# as a stand-in for equity.
EQUITY_KEYS: tuple[str, ...] = (
    "total_equity",
    "equity",
    "portfolio_equity",
    "portfolio_value",
    "total_account_value",
    "account_value",
    "total_value",
    "net_liquidation_value",
    "net_liquidity",
    "last_core_equity",
    "extended_hours_equity",
    "total_market_value",
    "market_value",
)

# An account number is digits, sometimes with a letter prefix. Excludes UUIDs and
# URLs, which appear under `id`/`url` on the same records and are not what the
# `account_number` parameter accepts.
_ACCOUNT_PATTERN = re.compile(r"^[A-Z]{0,3}\d{6,}$")


def _find_accounts(payload: Any, *, depth: int = 0) -> list[dict]:
    """Collect EVERY account record in a payload of unknown shape.

    Returns dicts of `{"number": ..., "record": <the dict it was found in>}` so
    the caller can inspect sibling metadata (type, name) when choosing.

    Collecting all rather than returning the first is the fix for a real
    incident: a user with an Individual account ($369) and an Agentic account
    ($100) had the agent silently resolve the INDIVIDUAL account, because it
    happened to appear first in the listing. Trading the wrong account is not a
    small bug -- every risk limit was being computed against money the agent
    was never meant to touch.
    """
    if depth > 6:
        return []

    found: list[dict] = []
    if isinstance(payload, dict):
        for key in _ACCOUNT_KEYS:
            value = payload.get(key)
            if isinstance(value, str | int):
                candidate = str(value).strip()
                if _ACCOUNT_PATTERN.match(candidate):
                    found.append({"number": candidate, "record": payload})
                    break  # one number per record
        for key in ("data", *[k for k in payload if k != "data"]):
            child = payload.get(key)
            if isinstance(child, dict | list):
                found.extend(_find_accounts(child, depth=depth + 1))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_find_accounts(item, depth=depth + 1))

    # De-duplicate on number, keeping the record with the most metadata.
    by_number: dict[str, dict] = {}
    for entry in found:
        existing = by_number.get(entry["number"])
        if existing is None or len(entry["record"]) > len(existing["record"]):
            by_number[entry["number"]] = entry
    return list(by_number.values())


# Markers that identify the account Robinhood provisions for agent trading.
# Checked against every string field of the account record, case-insensitively.
_AGENTIC_MARKERS = ("agentic", "agent")


def _is_agentic(record: dict) -> bool:
    for value in record.values():
        if isinstance(value, str) and any(
            m in value.lower() for m in _AGENTIC_MARKERS
        ):
            return True
    return False


def _find_account_number(payload: Any, *, depth: int = 0) -> str | None:
    """First account number in the payload. Kept for connect-time diagnostics;
    trading code must go through `resolve_account`, which chooses deliberately."""
    accounts = _find_accounts(payload, depth=depth)
    return accounts[0]["number"] if accounts else None


def _as_uuid(key: str) -> str:
    """Derive a stable UUID from our idempotency key.

    `ref_id` is the venue's duplicate-protection key and must be a UUID. Our key
    is a 32-char hex digest, so it is reshaped DETERMINISTICALLY -- a random UUID
    would defeat the purpose, since a retry has to present the same value to be
    recognized as the same order.
    """
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_OID, key))


def _find_number(payload: Any, keys: tuple[str, ...], *, depth: int = 0) -> float | None:
    """Search a payload of unknown shape for the first of `keys` holding a number.

    Recursive for the same reason `_find_account_number` is: the response envelope
    is not described by the tool's input schema, so it has to be discovered rather
    than assumed. Robinhood nests portfolio figures under `data`.

    `keys` is ordered by preference, and each level is checked against ALL keys
    before descending -- otherwise a less-preferred key on a shallow node would
    beat the exact field sitting one level down.
    """
    if depth > 6:
        return None

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            number = _coerce_number(value)
            if number is not None:
                return number
        for key in ("data", *[k for k in payload if k != "data"]):
            found = _find_number(payload.get(key), keys, depth=depth + 1)
            if found is not None:
                return found
        return None

    if isinstance(payload, list):
        for item in payload:
            found = _find_number(item, keys, depth=depth + 1)
            if found is not None:
                return found
    return None


def _coerce_number(value: Any) -> float | None:
    """Numbers arrive as floats, ints, or decimal strings depending on the tool."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("$", "").strip())
        except ValueError:
            return None
    return None


def describe_shape(payload: Any, *, depth: int = 0, max_depth: int = 4) -> list[str]:
    """Flatten a payload into `path: type` lines, for diagnostics.

    Exists so an unrecognized response can be identified in one round trip rather
    than by guessing field names across several. Values are omitted; only the
    structure is reported.
    """
    lines: list[str] = []
    if depth > max_depth:
        return lines
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict | list):
                lines.append(f"{'  ' * depth}{key}:")
                lines.extend(describe_shape(value, depth=depth + 1, max_depth=max_depth))
            else:
                lines.append(f"{'  ' * depth}{key}: {type(value).__name__}")
    elif isinstance(payload, list):
        if payload:
            lines.append(f"{'  ' * depth}[0 of {len(payload)}]:")
            lines.extend(describe_shape(payload[0], depth=depth + 1, max_depth=max_depth))
        else:
            lines.append(f"{'  ' * depth}[] (empty)")
    return lines


def _first_number(payload: dict, *keys: str) -> float | None:
    for key in keys:
        val = payload.get(key)
        if isinstance(val, int | float):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                continue
    return None


class MCPBroker(Broker):
    """Capability-oriented live broker. Holds the only broker connection."""

    def __init__(
        self,
        adapter: Any,
        *,
        account_number: str | None = None,
        allow_fractional_fallback: bool = True,
    ) -> None:
        self.adapter = adapter
        self.account_number = account_number
        # Robinhood permits fractional quantities on MARKET orders only. A small
        # account therefore cannot place a limit order at all: at $366 equity a 2%
        # order is $7.32, which is under one share of most stocks, and the venue
        # rejects quantity=0.
        #
        # With this enabled, such orders convert to a market order sized in
        # dollars. That trades away limit-price protection for the ability to
        # trade, which is the right trade at $7 notional -- crossing a 2bp spread
        # costs fractions of a cent, while not trading costs the whole strategy.
        # Disable it to make undersized limit orders fail loudly instead.
        self.allow_fractional_fallback = allow_fractional_fallback

    @property
    def name(self) -> str:
        return "robinhood-mcp"

    async def resolve_account(self) -> str | None:
        """Discover and cache the account number, choosing DELIBERATELY.

        Robinhood requires `account_number` on portfolio and position reads, not
        just on orders. But a login can see several accounts (Individual, Roth,
        Agentic), and "first one in the listing" once resolved a user's
        Individual account instead of the Agentic account the agent was meant to
        trade. Selection order:

          1. `OSIRIS_ACCOUNT_NUMBER` from the environment -- an explicit pin
             always wins, and is the recommended production setting.
          2. The account whose record identifies it as Agentic.
          3. A single account, if only one exists.
          4. Multiple accounts and no way to choose -> REFUSE. Trading an
             arbitrary account is worse than not starting.
        """
        if self.account_number:
            return self.account_number

        import os

        pinned = os.environ.get("OSIRIS_ACCOUNT_NUMBER", "").strip()

        try:
            result = await self.adapter.call("listAccounts", {})
        except Exception as exc:
            log.error("mcp_broker.account_lookup_failed", error=str(exc))
            return None

        payload = _extract_payload(result)
        accounts = _find_accounts(payload)

        if pinned:
            if any(a["number"] == pinned for a in accounts) or not accounts:
                self.account_number = pinned
                log.info(
                    "mcp_broker.account_resolved",
                    account=f"...{pinned[-4:]}",
                    via="OSIRIS_ACCOUNT_NUMBER",
                )
                return self.account_number
            log.error(
                "mcp_broker.pinned_account_not_visible",
                pinned=f"...{pinned[-4:]}",
                visible=[f"...{a['number'][-4:]}" for a in accounts],
                hint="the pinned account is not in listAccounts; check the number",
            )
            return None

        if not accounts:
            log.error(
                "mcp_broker.no_account_found",
                payload_keys=sorted(payload)[:8] if isinstance(payload, dict) else "?",
                hint="run `python -m osiris.connect --debug-accounts` to dump the payload",
            )
            return None

        agentic = [a for a in accounts if _is_agentic(a["record"])]
        if len(agentic) == 1:
            self.account_number = agentic[0]["number"]
            log.info(
                "mcp_broker.account_resolved",
                # Last four only: the full number is account-identifying and ends
                # up in logs that get pasted into bug reports.
                account=f"...{self.account_number[-4:]}",
                via="agentic marker",
            )
            return self.account_number

        if len(accounts) == 1:
            self.account_number = accounts[0]["number"]
            log.info(
                "mcp_broker.account_resolved",
                account=f"...{self.account_number[-4:]}",
                via="only account",
            )
            return self.account_number

        # Several accounts, none identifiable as Agentic: refuse to guess.
        log.error(
            "mcp_broker.ambiguous_accounts",
            visible=[f"...{a['number'][-4:]}" for a in accounts],
            hint=(
                "multiple accounts visible; set OSIRIS_ACCOUNT_NUMBER in .env "
                "to the account the agent should trade"
            ),
        )
        return None

    def _account_args(self, **extra: Any) -> dict[str, Any]:
        """Arguments for an account-scoped read."""
        args: dict[str, Any] = dict(extra)
        if self.account_number:
            args["account_number"] = self.account_number
        return args

    def _order_args(self, request: OrderRequest) -> dict[str, Any]:
        """Build `place_equity_order` arguments from the account's real schema.

        Four details are load-bearing, each verified against the live tool schema
        rather than assumed:

        1. **Every value is a STRING.** The schema types `quantity`,
           `dollar_amount`, and `limit_price` as strings, not numbers. Sending
           JSON numbers fails validation.
        2. **`dollar_amount` is market-only.** "Only valid with type=market." A
           limit order must therefore be expressed in shares, so notional is
           converted using the limit price.
        3. **`ref_id` must be a UUID**, not our 32-char hex idempotency key. The
           field is documented as the upstream idempotency key, so a malformed
           value forfeits duplicate protection on retry.
        4. **`amount_in_dollars` and `client_order_id` do not exist.** With
           `additionalProperties: false`, sending them rejects the whole order.
        """
        args: dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.value,
            "type": request.kind.value,
            "time_in_force": "gfd",
        }
        if self.account_number:
            args["account_number"] = self.account_number

        if request.kind is OrderKind.LIMIT and request.limit_price:
            price = round(request.limit_price, 2)
            args["limit_price"] = f"{price:.2f}"
            # Shares, because dollar_amount is rejected for limit orders.
            # Fractional quantities are allowed for market/regular-hours only, so
            # a limit order must use a whole number of shares.
            shares = int(request.notional_usd // price) if price > 0 else 0
            if shares < 1:
                if not self.allow_fractional_fallback:
                    raise OrderRejected(
                        f"{request.symbol}: ${request.notional_usd:,.2f} is less "
                        f"than one share at ${price:,.2f}. Use a market order for "
                        "fractional sizing, or increase the order size."
                    )
                # Fall back to a dollar-denominated MARKET order, which is the
                # only way to buy a fraction of a share. Logged loudly because
                # the order type is not what the planner asked for.
                log.info(
                    "mcp_broker.limit_to_market_fallback",
                    symbol=request.symbol,
                    notional=round(request.notional_usd, 2),
                    price=price,
                    reason="below one share; fractional requires a market order",
                )
                args["type"] = OrderKind.MARKET.value
                args.pop("limit_price", None)
                args["dollar_amount"] = f"{request.notional_usd:.2f}"
            else:
                args["quantity"] = str(shares)
        else:
            args["dollar_amount"] = f"{request.notional_usd:.2f}"

        if request.idempotency_key:
            args["ref_id"] = _as_uuid(request.idempotency_key)
        return args

    async def review(self, request: OrderRequest) -> ReviewResult:
        """Mandatory simulation. A failed review is a hard stop, not a warning."""
        try:
            result = await self.adapter.call("reviewOrder", self._order_args(request))
        except ToolCallFailed as exc:
            log.warning("mcp_broker.review_rejected", symbol=request.symbol, error=str(exc))
            return ReviewResult(False, message=f"review rejected: {exc}")
        except Exception as exc:
            log.error("mcp_broker.review_error", symbol=request.symbol, error=str(exc))
            return ReviewResult(False, message=f"review transport error: {exc}")

        payload = _extract_payload(result)
        price = _first_number(payload, "estimated_price", "price", "ask_price")
        quantity = _first_number(payload, "estimated_quantity", "quantity", "shares")
        if quantity is None and price and price > 0:
            quantity = request.notional_usd / price

        # Fail closed: an unparseable review is not an approval.
        if price is None and quantity is None:
            return ReviewResult(
                False,
                message="review returned no usable price or quantity",
                raw=payload,
            )
        return ReviewResult(
            accepted=True,
            estimated_price=price,
            estimated_quantity=quantity,
            estimated_cost=_first_number(payload, "estimated_cost", "total_cost")
            or request.notional_usd,
            message="ok",
            raw=payload,
        )

    async def place(self, request: OrderRequest) -> PlaceResult:
        review = await self.review(request)
        if not review.accepted:
            raise OrderRejected(review.message)

        args = self._order_args(request)
        try:
            result = await self.adapter.call("placeOrder", args, retries=1)
        except ToolCallFailed as exc:
            # A logical rejection is definitive: the venue said no.
            log.warning("mcp_broker.place_rejected", symbol=request.symbol, error=str(exc))
            return PlaceResult(
                order_id="", accepted=False, message=f"order rejected: {exc}"
            )
        except Exception as exc:
            # Transport failure after send. State is genuinely unknown.
            log.error("mcp_broker.place_ambiguous", symbol=request.symbol, error=str(exc))
            raise AmbiguousOrderState(
                f"place_equity_order for {request.symbol} failed in transport: {exc}. "
                "Order may exist at the venue. Reconcile before trading again."
            ) from exc

        payload = _extract_payload(result)
        order_id = str(
            payload.get("id") or payload.get("order_id") or payload.get("ref_id") or ""
        )
        state = str(payload.get("state") or payload.get("status") or "").lower()
        if state in {"rejected", "cancelled", "canceled", "failed"}:
            return PlaceResult(
                order_id=order_id,
                accepted=False,
                message=f"venue state={state}",
                raw=payload,
            )

        fills = self._parse_fills(payload, request)
        log.info(
            "mcp_broker.placed",
            symbol=request.symbol,
            side=request.side.value,
            order_id=order_id,
            state=state or "unknown",
            fills=len(fills),
        )
        return PlaceResult(
            order_id=order_id,
            accepted=True,
            fills=fills,
            message=state or "submitted",
            raw=payload,
        )

    def _parse_fills(self, payload: dict, request: OrderRequest) -> tuple[Fill, ...]:
        """Only booked as filled when the venue reports an executed quantity.

        A submitted-but-unfilled order books NO fill. Reconciliation picks it up
        once it executes; inventing a fill here is how the ledger drifts.
        """
        from datetime import UTC, datetime

        executions = payload.get("executions") or payload.get("fills") or []
        out: list[Fill] = []
        order_id = str(payload.get("id") or payload.get("order_id") or "")

        if isinstance(executions, list) and executions:
            for ex in executions:
                if not isinstance(ex, dict):
                    continue
                qty = _first_number(ex, "quantity", "shares")
                px = _first_number(ex, "price", "effective_price")
                if not qty or not px or qty <= 0 or px <= 0:
                    continue
                out.append(
                    Fill(
                        symbol=request.symbol,
                        side=request.side,
                        quantity=qty,
                        price=px,
                        ts=datetime.now(UTC),
                        order_id=order_id,
                        idempotency_key=request.idempotency_key,
                        intended_price=request.limit_price,
                    )
                )
            return tuple(out)

        qty = _first_number(payload, "cumulative_quantity", "filled_quantity")
        px = _first_number(payload, "average_price", "executed_price")
        if qty and px and qty > 0 and px > 0:
            out.append(
                Fill(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=qty,
                    price=px,
                    ts=datetime.now(UTC),
                    order_id=order_id,
                    idempotency_key=request.idempotency_key,
                    intended_price=request.limit_price,
                )
            )
        return tuple(out)

    async def get_positions(self) -> dict[str, float]:
        await self.resolve_account()
        result = await self.adapter.call("listPositions", self._account_args())
        payload = _extract_payload(result)
        rows = payload.get("positions") or payload.get("results") or payload.get("data") or []
        out: dict[str, float] = {}
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = row.get("symbol") or row.get("ticker")
                qty = _first_number(row, "quantity", "shares", "position")
                if symbol and qty:
                    out[str(symbol)] = out.get(str(symbol), 0.0) + qty
        return out

    async def get_account_equity(self) -> float:
        await self.resolve_account()
        result = await self.adapter.call("getPortfolio", self._account_args())
        payload = _extract_payload(result)

        equity = _find_number(payload, EQUITY_KEYS)
        if equity is not None and equity > 0:
            return equity

        # Report the actual structure rather than just the top-level keys. Naming
        # the paths that exist is what turns this from a guessing game into a
        # one-line fix, and `['data', 'guide']` alone said nothing useful.
        shape = "\n  ".join(describe_shape(payload)[:40]) or "(empty response)"
        raise RuntimeError(
            "Account info returned no recognizable equity field. Refusing to "
            "proceed with unknown equity: every risk limit is a fraction of it.\n"
            f"  Looked for: {', '.join(EQUITY_KEYS[:6])}...\n"
            f"  Response shape:\n  {shape}"
        )

    async def cancel_all(self) -> int:
        """Best-effort cancel of open orders. Used by the kill switch."""
        if not self.adapter.has("cancelOrder"):
            log.warning("mcp_broker.no_cancel_capability")
            return 0
        await self.resolve_account()
        try:
            result = await self.adapter.call("listOrders", self._account_args())
        except Exception as exc:
            log.error("mcp_broker.cancel_all_list_failed", error=str(exc))
            return 0
        payload = _extract_payload(result)
        rows = payload.get("orders") or payload.get("results") or []
        cancelled = 0
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            state = str(row.get("state") or row.get("status") or "").lower()
            if state not in {"queued", "confirmed", "partially_filled", "new", "pending"}:
                continue
            oid = row.get("id") or row.get("order_id")
            if not oid:
                continue
            try:
                await self.adapter.call("cancelOrder", {"order_id": str(oid)})
                cancelled += 1
            except Exception as exc:
                log.error("mcp_broker.cancel_failed", order_id=str(oid), error=str(exc))
        log.info("mcp_broker.cancel_all", cancelled=cancelled)
        return cancelled
