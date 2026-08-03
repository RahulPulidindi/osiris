"""The Risk Kernel. Deterministic, pure, and the only thing that can say yes.

Design contract:
  - No LLM anywhere in this module.
  - No network I/O at decision time.
  - Pure function of (intent, state, limits) so every verdict is reproducible.
  - Fails closed: absent data is a veto.

The model proposes; the kernel disposes. A prompt injection that fully captures
the cognition plane still cannot produce a non-compliant order, because the
kernel never reads model output as instructions.
"""

from __future__ import annotations

from collections.abc import Sequence

from osiris.config import AccountType, RiskLimits
from osiris.kernel import exposure, gates
from osiris.kernel.state import KernelState
from osiris.logging import get_logger
from osiris.types import KernelDecision, OrderIntent, VetoCode

log = get_logger(__name__)

# Reasons that mark an intent as risk management. These bypass the
# diversification floor: honoring a stop matters more than holding 15 names.
RISK_EXIT_REASONS = frozenset({"risk_exit", "invalidation_exit", "breaker_flatten"})


class RiskKernel:
    """Evaluates intents against deterministic limits."""

    def __init__(
        self,
        limits: RiskLimits,
        account_type: AccountType = AccountType.UNKNOWN,
    ) -> None:
        self.limits = limits
        self.account_type = account_type

    def evaluate(self, intent: OrderIntent, state: KernelState) -> KernelDecision:
        """Return a decision. Collects ALL vetoes, not just the first.

        Reporting every violation makes the dashboard and journal diagnostic
        rather than a guessing game.
        """
        vetoes: list[tuple[VetoCode, str]] = []
        is_risk_exit = intent.reason in RISK_EXIT_REASONS

        # --- Session-level gates ---
        vetoes += gates.gate_kill_switch(state, intent)
        vetoes += gates.gate_breakers(state, intent)
        vetoes += gates.gate_macro_blackout(state, intent)
        vetoes += gates.gate_order_budget(state, self.limits)
        vetoes += gates.gate_idempotency(state, intent)

        # --- Data quality gates ---
        vetoes += gates.gate_tradability(state, intent)
        vetoes += gates.gate_staleness(state, intent, self.limits)
        vetoes += gates.gate_spread(state, intent, self.limits)

        # --- Intent quality gates ---
        vetoes += gates.gate_invalidation(intent)
        vetoes += gates.gate_earnings_blackout(state, intent, self.limits)

        # --- Exposure gates ---
        vetoes += exposure.gate_notional_cap(state, intent, self.limits)
        vetoes += exposure.gate_symbol_weight(state, intent, self.limits)
        vetoes += exposure.gate_sector_weight(state, intent, self.limits)
        vetoes += exposure.gate_sector_deviation(state, intent, self.limits)
        vetoes += exposure.gate_beta_budget(state, intent, self.limits)
        vetoes += exposure.gate_buying_power(state, intent, self.account_type)

        # A genuine risk exit must not be trapped by gates that exist to shape
        # NEW risk. The diversification floor and the ADV participation cap
        # both fail closed -- correct for entries, but a stop that cannot fire
        # because volume DATA is missing leaves the agent holding a falling
        # position, which is the exact failure stops exist to prevent. The
        # exit's size is already bounded by the position itself.
        if not is_risk_exit:
            vetoes += exposure.gate_adv_participation(state, intent, self.limits)
            vetoes += exposure.gate_position_floor(state, intent, self.limits)

        codes = tuple(dict.fromkeys(code for code, _ in vetoes))
        notes = tuple(note for _, note in vetoes)
        approved = len(codes) == 0

        if not approved:
            log.info(
                "kernel.veto",
                symbol=intent.symbol,
                side=intent.side.value,
                reason=intent.reason,
                codes=[c.value for c in codes],
                notes=list(notes),
            )
        return KernelDecision(
            approved=approved, intent=intent, vetoes=codes, notes=notes
        )

    def evaluate_before_place(
        self, intent: OrderIntent, state: KernelState
    ) -> KernelDecision:
        """Final gate immediately before the broker call.

        Adds the mandatory-review check. Separate from `evaluate` because review
        happens between the two, so this cannot be folded into one pass.
        """
        decision = self.evaluate(intent, state)
        review_vetoes = gates.gate_review_ran(state, intent)
        if not review_vetoes:
            return decision

        codes = tuple(dict.fromkeys((*decision.vetoes, *(c for c, _ in review_vetoes))))
        notes = (*decision.notes, *(n for _, n in review_vetoes))
        return KernelDecision(approved=False, intent=intent, vetoes=codes, notes=notes)

    def evaluate_batch(
        self, intents: Sequence[OrderIntent], state: KernelState
    ) -> list[KernelDecision]:
        """Evaluate a rebalance batch, accumulating approved exposure.

        Evaluating each intent against the *initial* state independently would
        let twenty individually-compliant buys breach every portfolio cap
        collectively. Exits are processed first so they free capacity for entries.
        """
        ordered = sorted(intents, key=lambda i: (not i.is_exit, i.symbol))
        decisions: list[KernelDecision] = []
        working = state

        for intent in ordered:
            decision = self.evaluate(intent, working)
            decisions.append(decision)
            if decision.approved:
                working = self._project(working, intent)
        return decisions

    def _project(self, state: KernelState, intent: OrderIntent) -> KernelState:
        """Apply an approved intent to state so the next evaluation sees it."""
        from dataclasses import replace as dc_replace

        from osiris.types import Portfolio, Position

        equity = state.portfolio.equity
        positions = list(state.portfolio.positions)
        delta = intent.notional_usd if not intent.is_exit else -intent.notional_usd

        idx = next(
            (i for i, p in enumerate(positions) if p.symbol == intent.symbol), None
        )
        if idx is None:
            if not intent.is_exit:
                positions.append(
                    Position(
                        symbol=intent.symbol,
                        quantity=1.0,  # placeholder; reconciliation is authoritative
                        cost_basis=intent.notional_usd,
                        market_value=intent.notional_usd,
                        sector=state.sector_of(intent.symbol),
                        beta=state.beta_of(intent.symbol),
                    )
                )
        else:
            p = positions[idx]
            new_mv = max(0.0, p.market_value + delta)
            positions[idx] = Position(
                symbol=p.symbol,
                quantity=p.quantity if new_mv > 0 else 0.0,
                cost_basis=p.cost_basis,
                market_value=new_mv,
                sector=p.sector,
                beta=p.beta,
            )

        buying_power = state.portfolio.buying_power - (
            intent.notional_usd if not intent.is_exit else 0.0
        )
        projected_portfolio = Portfolio(
            equity=equity,
            cash=state.portfolio.cash,
            buying_power=max(0.0, buying_power),
            positions=tuple(p for p in positions if p.market_value > 0),
            as_of=state.portfolio.as_of,
        )
        return dc_replace(
            state,
            portfolio=projected_portfolio,
            orders_placed_today=state.orders_placed_today + 1,
            used_idempotency_keys=state.used_idempotency_keys | {intent.idempotency_key},
        )
