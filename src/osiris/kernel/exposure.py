"""Portfolio-level exposure gates.

These are the constraints that stop "maximize risk-adjusted return" from being
silently satisfied by loading beta or concentrating into one theme.
"""

from __future__ import annotations

from osiris.config import AccountType, RiskLimits
from osiris.kernel.state import KernelState
from osiris.types import OrderIntent, VetoCode

Veto = tuple[VetoCode, str]


def gate_notional_cap(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Per-trade notional ceiling as a fraction of equity.

    Exits are exempt, and that exemption is load-bearing rather than a
    convenience. The cap bounds how much NEW risk one order may add; an exit
    removes risk. Applying it to sells would trap capital: positions may grow to
    `max_symbol_weight` (10%) while this cap is `max_trade_notional_pct` (2%), so
    any position above 2% of equity could never be closed in a single order. A
    stop that cannot fill is not a stop, and the position that most needs exiting
    is precisely the one that has grown largest.

    Exit size is bounded by the position itself: the broker rejects selling more
    than is held, and the ledger clamps at zero.
    """
    equity = state.portfolio.equity
    if equity <= 0:
        return [(VetoCode.NOTIONAL_CAP, "zero equity")]
    if intent.is_exit:
        return []
    cap = equity * limits.max_trade_notional_pct
    if intent.notional_usd > cap:
        return [
            (
                VetoCode.NOTIONAL_CAP,
                f"${intent.notional_usd:,.0f} exceeds per-trade cap ${cap:,.0f}",
            )
        ]
    return []


def gate_symbol_weight(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Post-trade symbol weight ceiling. Concentration is where alpha dies."""
    if intent.is_exit:
        return []
    equity = state.portfolio.equity
    if equity <= 0:
        return [(VetoCode.SYMBOL_WEIGHT_CAP, "zero equity")]
    projected = state.portfolio.weight_of(intent.symbol) + intent.notional_usd / equity
    if projected > limits.max_symbol_weight:
        return [
            (
                VetoCode.SYMBOL_WEIGHT_CAP,
                f"{intent.symbol} would reach {projected:.1%}, cap "
                f"{limits.max_symbol_weight:.1%}",
            )
        ]
    return []


def gate_sector_weight(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Absolute sector ceiling."""
    if intent.is_exit:
        return []
    equity = state.portfolio.equity
    if equity <= 0:
        return [(VetoCode.SECTOR_WEIGHT_CAP, "zero equity")]
    sector = state.sector_of(intent.symbol)
    current = state.portfolio.sector_weights().get(sector, 0.0)
    projected = current + intent.notional_usd / equity
    if projected > limits.max_sector_weight:
        return [
            (
                VetoCode.SECTOR_WEIGHT_CAP,
                f"sector {sector} would reach {projected:.1%}, cap "
                f"{limits.max_sector_weight:.1%}",
            )
        ]
    return []


def gate_sector_deviation(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Cap deviation from benchmark sector weights.

    Without this, a thematic run turns a 20-name book into a single
    undiversified sector bet that still *looks* diversified by position count.
    """
    if intent.is_exit:
        return []
    benchmark = state.benchmark_sector_weights
    if not benchmark:
        return []  # no benchmark supplied; absolute cap still applies
    equity = state.portfolio.equity
    if equity <= 0:
        return [(VetoCode.SECTOR_DEVIATION, "zero equity")]

    sector = state.sector_of(intent.symbol)
    current = state.portfolio.sector_weights().get(sector, 0.0)
    projected = current + intent.notional_usd / equity
    target = benchmark.get(sector, 0.0)
    deviation = projected - target
    if deviation > limits.max_sector_deviation:
        return [
            (
                VetoCode.SECTOR_DEVIATION,
                f"sector {sector} overweight by {deviation:.1%} vs benchmark "
                f"{target:.1%}, max {limits.max_sector_deviation:.1%}",
            )
        ]
    return []


def gate_beta_budget(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Cap portfolio beta.

    This is the gate that prevents the objective from being satisfied by
    leverage in disguise. Both cited research systems ran beta below 1.0, which
    is the evidence that *selection* was doing the work rather than market
    exposure. Without this gate, a rising market makes any long book look skilled.
    """
    if intent.is_exit:
        return []
    equity = state.portfolio.equity
    if equity <= 0:
        return [(VetoCode.BETA_BUDGET, "zero equity")]

    current_beta_contrib = state.portfolio.portfolio_beta()
    added = state.beta_of(intent.symbol) * (intent.notional_usd / equity)
    projected = current_beta_contrib + added
    if projected > limits.max_portfolio_beta:
        return [
            (
                VetoCode.BETA_BUDGET,
                f"portfolio beta would reach {projected:.2f}, cap "
                f"{limits.max_portfolio_beta:.2f}",
            )
        ]
    return []


def gate_position_floor(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Block exits that would drop the book below the diversification floor.

    Breadth is the edge. An exit that leaves 14 names is not risk management,
    it is drift toward concentration. Risk exits bypass this via the caller.
    """
    if not intent.is_exit:
        return []
    remaining = state.portfolio.position_count - 1
    if remaining < limits.min_position_count:
        return [
            (
                VetoCode.POSITION_FLOOR,
                f"exit would leave {remaining} names, floor "
                f"{limits.min_position_count}",
            )
        ]
    return []


def gate_adv_participation(
    state: KernelState, intent: OrderIntent, limits: RiskLimits
) -> list[Veto]:
    """Cap order size against average daily dollar volume. Fails closed."""
    adv = state.adv.get(intent.symbol)
    if adv is None or adv <= 0:
        return [(VetoCode.ADV_PARTICIPATION, f"no ADV data for {intent.symbol}")]
    cap = adv * limits.max_adv_participation
    if intent.notional_usd > cap:
        return [
            (
                VetoCode.ADV_PARTICIPATION,
                f"${intent.notional_usd:,.0f} exceeds "
                f"{limits.max_adv_participation:.1%} of ADV (${cap:,.0f})",
            )
        ]
    return []


def gate_buying_power(
    state: KernelState, intent: OrderIntent, account_type: AccountType
) -> list[Veto]:
    """Buying power, plus settlement rules for cash accounts.

    In a cash account, reusing unsettled proceeds causes good-faith violations,
    which restrict the account. The kernel refuses rather than risking that.
    """
    if intent.is_exit:
        return []
    if intent.notional_usd > state.portfolio.buying_power:
        return [
            (
                VetoCode.INSUFFICIENT_BUYING_POWER,
                f"${intent.notional_usd:,.0f} exceeds buying power "
                f"${state.portfolio.buying_power:,.0f}",
            )
        ]
    if account_type is AccountType.CASH:
        settled = state.portfolio.buying_power - state.unsettled_cash
        if intent.notional_usd > settled:
            return [
                (
                    VetoCode.UNSETTLED_FUNDS,
                    f"cash account: ${intent.notional_usd:,.0f} exceeds settled "
                    f"${settled:,.0f} (good-faith violation risk)",
                )
            ]
    return []
