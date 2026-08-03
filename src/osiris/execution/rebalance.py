"""Target-book diffing. Where autonomous exits actually happen.

Under a ranking engine, closing a position is a set-difference operation:

    to_buy  = target - held
    to_sell = held - target

A stock that drops out of the top-N is sold. No opinion required, and critically
**the model gets no vote on whether to keep a loser.**

That asymmetry is the point. Discretionary exits are where LLM trading agents
fail most reliably: the model forms a thesis, the position moves against it, and
the model rationalizes holding -- "thesis intact, this is just noise." That is the
standard path to one position eating a portfolio. Mechanical ranking exits delete
the failure mode, because the exit decision is arithmetic rather than judgment.

Three exit paths exist, in strict priority order:
  1. risk_exit          -- kernel-forced; checked continuously, not just at 9:30
  2. invalidation_exit  -- the stated falsifiable condition fired
  3. rank_exit          -- fell out of the target book
"""

from __future__ import annotations

from dataclasses import dataclass, field

from osiris.cognition.schemas import TargetHolding
from osiris.kernel.sizing import clamp_to_adv, vol_target_notional
from osiris.logging import get_logger
from osiris.types import OrderIntent, OrderKind, Portfolio, Side

log = get_logger(__name__)

# Do not churn a position for a trivial weight change: the cost is certain and
# the benefit is noise. Trades below this fraction of equity are skipped.
MIN_REBALANCE_WEIGHT = 0.0025


@dataclass(frozen=True)
class ExitSignal:
    """A non-rank reason to close. Takes priority over the ranking."""

    symbol: str
    reason: str  # risk_exit | invalidation_exit
    detail: str = ""


@dataclass
class RebalancePlan:
    entries: list[OrderIntent] = field(default_factory=list)
    exits: list[OrderIntent] = field(default_factory=list)
    trims: list[OrderIntent] = field(default_factory=list)
    adds: list[OrderIntent] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def all_intents(self) -> list[OrderIntent]:
        """Exits first: they free buying power and reduce risk before adding."""
        return [*self.exits, *self.trims, *self.entries, *self.adds]

    @property
    def turnover_notional(self) -> float:
        return sum(i.notional_usd for i in self.all_intents)

    def summary(self) -> str:
        return (
            f"{len(self.exits)} exits, {len(self.trims)} trims, "
            f"{len(self.entries)} entries, {len(self.adds)} adds "
            f"(${self.turnover_notional:,.0f} turnover)"
        )


def diff_book(held: set[str], target: set[str]) -> tuple[set[str], set[str], set[str]]:
    """The core set operation. Returns (to_buy, to_sell, to_hold)."""
    return target - held, held - target, held & target


def build_rebalance_plan(
    portfolio: Portfolio,
    targets: list[TargetHolding],
    *,
    prices: dict[str, float],
    exit_signals: list[ExitSignal] | None = None,
    vols: dict[str, float] | None = None,
    adv: dict[str, float] | None = None,
    theses: dict[str, str] | None = None,
    invalidations: dict[str, str] | None = None,
    max_symbol_weight: float = 0.10,
    max_adv_participation: float = 0.01,
    max_trade_notional_pct: float = 0.02,
    target_position_count: int = 20,
    correlation_id: str = "",
) -> RebalancePlan:
    """Turn a target book into intents. Emits intents only -- never orders.

    Sizing is volatility-targeted rather than equal-dollar, so each name
    contributes comparable risk instead of comparable notional.
    """
    plan = RebalancePlan()
    equity = portfolio.equity
    if equity <= 0:
        plan.skipped["*"] = "zero equity"
        return plan

    vols = vols or {}
    adv = adv or {}
    theses = theses or {}
    invalidations = invalidations or {}
    signals = {s.symbol: s for s in (exit_signals or [])}

    held_weights = {p.symbol: p.market_value / equity for p in portfolio.positions}
    held = {s for s, w in held_weights.items() if w > 1e-9}
    target_map = {t.symbol: t for t in targets}
    target_set = set(target_map)

    # --- Forced exits first. These override the ranking entirely. ---
    for symbol, signal in signals.items():
        if symbol not in held:
            continue
        notional = held_weights[symbol] * equity
        plan.exits.append(
            OrderIntent(
                symbol=symbol,
                side=Side.SELL,
                notional_usd=notional,
                kind=OrderKind.MARKET,  # risk exits do not haggle over price
                thesis=theses.get(symbol, ""),
                invalidation=invalidations.get(symbol, signal.detail or signal.reason),
                reason=signal.reason,
                correlation_id=correlation_id,
            )
        )

    # A forced-exit name is excluded from BOTH sides of the diff. Removing it
    # only from `held` would place it in `to_buy` and re-enter the position in
    # the same cycle -- selling a stop and instantly buying it back, paying the
    # spread twice to end up exactly where the stop said not to be.
    forced = set(signals) & held
    to_buy, to_sell, to_hold = diff_book(held - forced, target_set - forced)

    # --- Rank exits: fell out of the book. ---
    for symbol in sorted(to_sell):
        notional = held_weights[symbol] * equity
        if notional <= 0:
            continue
        plan.exits.append(
            OrderIntent(
                symbol=symbol,
                side=Side.SELL,
                notional_usd=notional,
                kind=OrderKind.LIMIT,
                limit_price=prices.get(symbol),
                thesis=theses.get(symbol, ""),
                invalidation="dropped out of target ranking",
                reason="rank_exit",
                correlation_id=correlation_id,
            )
        )

    def sized(symbol: str, desired_weight: float) -> float:
        """Target position size: vol-targeted, clamped to symbol cap and ADV.

        This is the size the position should REACH, not the size of one order.
        """
        vol = vols.get(symbol, 0.0)
        notional = vol_target_notional(
            equity,
            symbol_vol=vol,
            position_count=max(1, target_position_count),
            max_weight=max_symbol_weight,
        )
        if notional <= 0:
            notional = equity * min(desired_weight, max_symbol_weight)
        # Respect the PM's relative conviction while honoring the vol target.
        notional = min(notional, equity * min(desired_weight, max_symbol_weight))
        if adv.get(symbol):
            notional = clamp_to_adv(notional, adv[symbol], max_adv_participation)
        return notional

    def per_order(notional: float) -> float:
        """Clamp ONE order to the per-trade cap.

        The per-trade cap (~2% of equity) and the symbol cap (~10%) are different
        limits serving different purposes: the first bounds how much risk a single
        order can add, the second bounds total concentration. A 10% target
        therefore has to be built across several sessions rather than in one
        order.

        Sizing to the full target in one order and letting the kernel veto it
        would be strictly worse than scaling in: the kernel is a backstop, and a
        planner that routinely proposes non-compliant orders makes the veto log
        useless for spotting real problems.
        """
        return min(notional, equity * max_trade_notional_pct)

    # --- New entries. ---
    for symbol in sorted(to_buy):
        holding = target_map[symbol]
        if prices.get(symbol) is None:
            plan.skipped[symbol] = "no price"
            continue
        # Scale in: the first order takes the position up to the per-trade cap,
        # and later sessions close the remaining gap toward the target weight.
        notional = per_order(sized(symbol, holding.target_weight))
        if notional / equity < MIN_REBALANCE_WEIGHT:
            plan.skipped[symbol] = f"sized below floor ({notional / equity:.4%})"
            continue
        plan.entries.append(
            OrderIntent(
                symbol=symbol,
                side=Side.BUY,
                notional_usd=notional,
                kind=OrderKind.LIMIT,
                limit_price=prices.get(symbol),
                thesis=theses.get(symbol, holding.rationale),
                invalidation=invalidations.get(symbol, ""),
                reason="rank_entry",
                correlation_id=correlation_id,
            )
        )

    # --- Rebalance existing holdings toward target weight. ---
    for symbol in sorted(to_hold):
        holding = target_map[symbol]
        price = prices.get(symbol)
        if price is None:
            plan.skipped[symbol] = "no price"
            continue
        current_notional = held_weights[symbol] * equity
        desired_notional = sized(symbol, holding.target_weight)
        delta = desired_notional - current_notional
        if abs(delta) / equity < MIN_REBALANCE_WEIGHT:
            continue
        # Adds are clamped per order for the same reason entries are. Trims are
        # not: reducing exposure is never the risk the per-trade cap guards.
        order_notional = per_order(delta) if delta > 0 else abs(delta)
        intent = OrderIntent(
            symbol=symbol,
            side=Side.BUY if delta > 0 else Side.SELL,
            notional_usd=order_notional,
            kind=OrderKind.LIMIT,
            limit_price=price,
            thesis=theses.get(symbol, holding.rationale),
            invalidation=invalidations.get(symbol, "rebalance toward target weight"),
            reason="rebalance",
            correlation_id=correlation_id,
        )
        (plan.adds if delta > 0 else plan.trims).append(intent)

    log.info(
        "rebalance.planned",
        summary=plan.summary(),
        held=len(held),
        target=len(target_set),
    )
    return plan


def detect_invalidation_exits(
    portfolio: Portfolio,
    *,
    entry_prices: dict[str, float],
    prices: dict[str, float],
    stop_loss_pct: float = 0.15,
) -> list[ExitSignal]:
    """Mechanical stop check, evaluated continuously rather than at rebalance.

    A stop that is only checked once a day at the open is not a stop. This is the
    deterministic floor beneath whatever the model believes.
    """
    out: list[ExitSignal] = []
    for pos in portfolio.positions:
        entry = entry_prices.get(pos.symbol)
        now = prices.get(pos.symbol)
        if not entry or not now or entry <= 0:
            continue
        drawdown = (now - entry) / entry
        if drawdown <= -abs(stop_loss_pct):
            out.append(
                ExitSignal(
                    symbol=pos.symbol,
                    reason="risk_exit",
                    detail=f"position drawdown {drawdown:.1%} breached stop {-stop_loss_pct:.0%}",
                )
            )
    return out
