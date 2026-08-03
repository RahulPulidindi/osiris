"""Journal → human-readable activity feed.

The journal is optimized for audit: append-only, low-level, one record per state
transition. That is the right shape for reconstructing what happened and the wrong
shape for *reading* what happened. `order_placed` followed by `fill` followed by
`kernel_veto` are three rows about two decisions.

This module collapses those into actions a person can read, each carrying its own
reason. The design rule: **an action and its justification belong on the same
row.** Splitting them is what forces the reader to correlate by ID across a log,
which is precisely the work a dashboard should be doing for them.

Nothing here computes anything new. If a reason is absent from the journal it is
absent here too, rather than being inferred -- a plausible-sounding reason the
agent did not actually record would be a fabrication, and this feed is the thing
an operator uses to decide whether to trust the system.
"""

from __future__ import annotations

from osiris.api.schemas import ActivityOut
from osiris.execution.journal import EventType, Journal

# Veto codes rendered in plain language. The raw enum is precise but assumes the
# reader knows the risk model; an operator scanning a feed does not.
VETO_EXPLANATIONS: dict[str, str] = {
    "kill_switch": "kill switch is engaged",
    "breaker_tripped": "a circuit breaker is tripped",
    "macro_blackout": "inside a macro event blackout",
    "earnings_blackout": "too close to earnings",
    "missing_invalidation": "no exit condition was stated",
    "notional_cap": "order larger than the per-order cap (positions scale in over several days)",
    "symbol_weight_cap": "would exceed the single-name limit",
    "sector_weight_cap": "would exceed the sector limit",
    "sector_deviation": "would drift too far from the benchmark's sector mix",
    "beta_budget": "would push portfolio beta over budget",
    "position_floor": "would drop the book below its diversification floor",
    "adv_participation": "too large relative to the stock's daily volume",
    "spread_too_wide": "spread too wide to trade cheaply",
    "not_tradable": "not currently tradable",
    "stale_data": "quote was too old to trust",
    "order_budget": "daily order budget already spent",
    "duplicate_order": "duplicate of an order already sent",
    "insufficient_buying_power": "not enough buying power",
    "unsettled_funds": "funds have not settled yet",
    "review_not_run": "broker simulation had not run",
    "review_rejected": "broker rejected the simulation",
}

# Why an order existed. These come from the rebalance planner's `reason` field.
REASON_LABELS: dict[str, str] = {
    "rank_entry": "entered the top ranks",
    "rank_exit": "dropped out of the top ranks",
    "risk_exit": "hit its stop",
    "invalidation_exit": "its thesis was invalidated",
    "rebalance": "rebalancing toward target weight",
    "trim": "trimming back to target weight",
    "add": "adding toward target weight",
}


def _money(value: float) -> str:
    return f"${value:,.0f}"


def explain_veto(codes: list[str]) -> str:
    if not codes:
        return "blocked by the risk kernel"
    return "; ".join(VETO_EXPLANATIONS.get(c, c.replace("_", " ")) for c in codes)


def label_reason(reason: str) -> str:
    return REASON_LABELS.get(reason, reason.replace("_", " "))


def build_activity(journal: Journal, *, limit: int = 200) -> list[ActivityOut]:
    """Project the journal into a readable feed, newest first.

    Fills are the unit of "what it did", not `order_placed`: an order that was
    submitted and never filled changed nothing about the book, and showing it as
    a trade would overstate activity. The placement record is still the source of
    the *thesis*, so the two are joined on idempotency key.
    """
    out: list[ActivityOut] = []

    # Thesis and reason live on the placement; fills carry neither. Index the
    # placements so each fill can recover why the agent wanted it.
    context: dict[str, dict] = {}
    for rec in journal.iter_events(event=EventType.ORDER_PLACED):
        key = rec.payload.get("idempotency_key") or rec.payload.get("order_id")
        if key:
            context[str(key)] = rec.payload

    # Intents carry the thesis too, and unlike placements they exist for orders
    # the kernel BLOCKED. Without this a veto row can say what was stopped but not
    # what the agent was trying to do -- and on a day when everything is blocked,
    # that is the entire feed.
    intents: dict[str, dict] = {}
    for rec in journal.iter_events(event=EventType.INTENT_EMITTED):
        symbol = str(rec.payload.get("symbol", ""))
        if symbol:
            intents[symbol] = rec.payload

    for rec in journal.iter_events():
        payload = rec.payload
        symbol = str(payload.get("symbol", "") or "")

        if rec.event is EventType.FILL:
            side = str(payload.get("side", ""))
            qty = float(payload.get("quantity") or 0.0)
            price = float(payload.get("price") or 0.0)
            notional = qty * price
            src = context.get(str(payload.get("order_id", ""))) or {}
            # Fall back to matching on symbol+side when the key is absent, so a
            # thesis is not silently dropped.
            if not src:
                src = next(
                    (
                        p
                        for p in context.values()
                        if p.get("symbol") == symbol and p.get("side") == side
                    ),
                    {},
                )
            is_buy = side == "buy"
            # A buy's justification is its thesis; a sell's is the condition that
            # fired. Showing the thesis on an exit reads as an argument FOR
            # holding, right next to the decision to sell -- actively misleading.
            detail = str(
                (src.get("thesis") if is_buy else src.get("invalidation")) or ""
            )
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="bought" if is_buy else "sold",
                    symbol=symbol,
                    headline=f"{'Bought' if is_buy else 'Sold'} {symbol} — {_money(notional)}",
                    reason=label_reason(str(src.get("reason", ""))),
                    detail=detail,
                    notional_usd=notional,
                    quantity=qty,
                    price=price,
                    correlation_id=rec.correlation_id,
                )
            )

        elif rec.event is EventType.KERNEL_VETO:
            codes = [str(c) for c in payload.get("vetoes", [])]
            notes = [str(n) for n in payload.get("notes", [])]
            side = str(payload.get("side", ""))
            want = "buy" if side == "buy" else "sell"
            # The kernel's own note is more specific than the code, and the intent
            # supplies what the agent was actually trying to do.
            detail = "; ".join(notes)
            thesis = str(intents.get(symbol, {}).get("thesis", "") or "")
            if thesis:
                detail = f"{detail} · wanted it because: {thesis}" if detail else thesis
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="blocked",
                    symbol=symbol,
                    headline=f"Blocked a {want} of {symbol}",
                    reason=explain_veto(codes),
                    detail=detail,
                    notional_usd=payload.get("notional_usd"),
                    correlation_id=rec.correlation_id,
                )
            )

        elif rec.event is EventType.BREAKER_TRIPPED:
            reasons = [str(r) for r in payload.get("reasons", [])]
            detail = str(payload.get("detail", "") or "")
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="halted",
                    headline="Halted new trading",
                    reason="; ".join(reasons) or str(payload.get("cause", "")),
                    detail=detail
                    or "Existing positions are still managed; only new entries stop.",
                    correlation_id=rec.correlation_id,
                )
            )

        elif rec.event is EventType.RECONCILIATION_BREAK:
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="halted",
                    headline="Ledger disagreed with the broker",
                    reason="trading halted until a human resolves it",
                    detail="; ".join(str(d) for d in payload.get("divergences", [])),
                    correlation_id=rec.correlation_id,
                )
            )

        elif rec.event is EventType.KILL_SWITCH:
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="halted",
                    headline="Kill switch engaged",
                    reason=str(payload.get("reason", "")),
                    correlation_id=rec.correlation_id,
                )
            )

        elif rec.event is EventType.CYCLE_END and not payload.get("skipped"):
            fills = int(payload.get("fills") or 0)
            vetoed = int(payload.get("vetoed") or 0)
            equity = payload.get("equity")
            out.append(
                ActivityOut(
                    seq=rec.seq,
                    ts=rec.ts,
                    kind="note",
                    headline=(
                        f"Session complete — {fills} trade(s), {vetoed} blocked"
                    ),
                    reason=str(payload.get("as_of", "")),
                    detail=f"equity {_money(float(equity))}" if equity else "",
                    correlation_id=rec.correlation_id,
                )
            )

    out.sort(key=lambda a: a.seq, reverse=True)
    return out[:limit]
