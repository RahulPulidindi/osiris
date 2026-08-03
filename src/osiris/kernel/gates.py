"""Individual pre-trade gates. Each is a pure predicate returning vetoes.

Every gate fails closed: missing data is a veto, not a pass. Acting on absent
information is the failure mode that turns a bug into a loss.
"""

from __future__ import annotations

from osiris.config import RiskLimits
from osiris.kernel.state import KernelState
from osiris.types import OrderIntent, VetoCode

Veto = tuple[VetoCode, str]


def gate_kill_switch(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """Kill switch blocks new risk. Exits are always permitted.

    A halted agent that cannot close positions is worse than one that never
    traded: it holds losers with no way out.
    """
    if state.kill_switch_engaged and not intent.is_exit:
        return [(VetoCode.KILL_SWITCH, "kill switch engaged")]
    return []


def gate_breakers(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """Tripped breakers stop new risk, never risk management."""
    if state.breakers.is_tripped and not intent.is_exit:
        reasons = "; ".join(state.breakers.reasons) or "breaker tripped"
        return [(VetoCode.BREAKER_TRIPPED, reasons)]
    return []


def gate_macro_blackout(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """No new entries into a scheduled macro event (CPI, FOMC, NFP)."""
    if state.macro_blackout and not intent.is_exit:
        return [
            (
                VetoCode.MACRO_BLACKOUT,
                state.macro_blackout_reason or "macro blackout window",
            )
        ]
    return []


def gate_earnings_blackout(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """No new position into an imminent earnings report.

    Holding a concentrated name through earnings is an uncompensated coin flip
    whose variance swamps the daily alpha the strategy is trying to harvest.
    """
    if intent.is_exit:
        return []
    if state.is_in_earnings_blackout(intent.symbol, limits.earnings_blackout_hours):
        report = state.next_earnings.get(intent.symbol)
        return [
            (
                VetoCode.EARNINGS_BLACKOUT,
                f"earnings at {report.isoformat() if report else 'unknown'} within "
                f"{limits.earnings_blackout_hours}h",
            )
        ]
    return []


def gate_invalidation(intent: OrderIntent) -> list[Veto]:
    """Entries must carry a falsifiable invalidation condition.

    A position with no defined exit is how a single name eats a portfolio: the
    model rationalizes holding as the thesis quietly breaks.
    """
    if intent.is_exit:
        return []
    if not intent.invalidation.strip():
        return [
            (
                VetoCode.MISSING_INVALIDATION,
                "entry has no falsifiable invalidation condition",
            )
        ]
    return []


def gate_tradability(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """Unknown tradability is a veto. Fails closed."""
    tradable = state.tradable.get(intent.symbol)
    if tradable is None:
        return [(VetoCode.NOT_TRADABLE, f"{intent.symbol} tradability unknown")]
    if not tradable:
        return [(VetoCode.NOT_TRADABLE, f"{intent.symbol} not tradable")]
    return []


def gate_staleness(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Refuse to act on stale quotes. Missing quote is also a veto."""
    quote = state.quotes.get(intent.symbol)
    if quote is None:
        return [(VetoCode.STALE_DATA, f"no quote for {intent.symbol}")]
    age = quote.age_seconds(state.now)
    if age > limits.quote_staleness_seconds:
        return [
            (
                VetoCode.STALE_DATA,
                f"quote {age:.0f}s old exceeds {limits.quote_staleness_seconds}s",
            )
        ]
    return []


def gate_spread(state: KernelState, intent: OrderIntent, limits: RiskLimits) -> list[Veto]:
    """Wide spreads eat the alpha. This is the gate that protects the edge.

    Applied to entries only: a risk exit must not be blocked because the market
    got ugly, which is exactly when exits matter most.
    """
    if intent.is_exit:
        return []
    quote = state.quotes.get(intent.symbol)
    if quote is None:
        return [(VetoCode.STALE_DATA, f"no quote for {intent.symbol}")]
    if quote.spread_bps > limits.max_spread_bps:
        return [
            (
                VetoCode.SPREAD_TOO_WIDE,
                f"spread {quote.spread_bps:.1f}bps exceeds {limits.max_spread_bps:.1f}bps",
            )
        ]
    return []


def gate_order_budget(state: KernelState, limits: RiskLimits) -> list[Veto]:
    """Hard cap on daily orders. Blocks runaway loops regardless of cause."""
    if state.orders_placed_today >= limits.daily_order_budget:
        return [
            (
                VetoCode.ORDER_BUDGET,
                f"{state.orders_placed_today} orders placed, budget "
                f"{limits.daily_order_budget}",
            )
        ]
    return []


def gate_idempotency(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """The primary duplicate-order defense."""
    key = intent.idempotency_key
    if key in state.used_idempotency_keys:
        return [(VetoCode.DUPLICATE_ORDER, f"idempotency key {key[:12]} already used")]
    return []


def gate_review_ran(state: KernelState, intent: OrderIntent) -> list[Veto]:
    """review_* must have run clean for this exact intent before place_*."""
    if intent.idempotency_key not in state.reviewed_keys:
        return [(VetoCode.REVIEW_NOT_RUN, "pre-trade review has not passed")]
    return []
