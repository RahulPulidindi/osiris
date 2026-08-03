"""The order pipeline: kernel -> review -> kernel -> place -> journal -> ledger.

This is the ONLY path from an intent to the venue. Structural invariants:

  1. The kernel runs twice. Once on the batch, then again immediately before
     placement (`evaluate_before_place`), because state changes between the two --
     a breaker can trip mid-cycle, and the second call is what catches it.

  2. Review is mandatory and its result feeds the kernel. `review_*` running is
     itself a kernel gate (`REVIEW_NOT_RUN`), so an unsimulated order cannot
     reach the venue even if this module has a bug.

  3. Journal before ledger. An order is recorded as placed BEFORE the ledger is
     updated, so a crash between the two leaves evidence rather than a silent
     phantom position.

  4. Ambiguity halts. If placement outcome is unknown, we stop the cycle instead
     of guessing. Both guesses corrupt the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from osiris.execution.broker import (
    Broker,
    OrderRejected,
    OrderRequest,
    PlaceResult,
)
from osiris.execution.journal import EventType, Journal
from osiris.execution.ledger import Ledger
from osiris.execution.mcp_broker import AmbiguousOrderState
from osiris.kernel.kernel import RiskKernel
from osiris.kernel.state import KernelState
from osiris.logging import get_logger
from osiris.types import Fill, KernelDecision, OrderIntent, VetoCode

log = get_logger(__name__)


@dataclass
class ExecutionReport:
    """Outcome of one execution pass. Every intent is accounted for."""

    placed: list[PlaceResult] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    vetoed: list[KernelDecision] = field(default_factory=list)
    rejected: list[tuple[OrderIntent, str]] = field(default_factory=list)
    halted_reason: str | None = None

    @property
    def filled_notional(self) -> float:
        return sum(f.quantity * f.price for f in self.fills)

    @property
    def realized_slippage_bps(self) -> float | None:
        """Realized vs intended price. The earliest signal of edge decay."""
        vals = [f.slippage_bps for f in self.fills if f.slippage_bps is not None]
        return sum(vals) / len(vals) if vals else None

    def summary(self) -> str:
        parts = [
            f"{len(self.fills)} fills (${self.filled_notional:,.0f})",
            f"{len(self.vetoed)} vetoed",
            f"{len(self.rejected)} rejected",
        ]
        slip = self.realized_slippage_bps
        if slip is not None:
            parts.append(f"slippage {slip:+.1f}bps")
        if self.halted_reason:
            parts.append(f"HALTED: {self.halted_reason}")
        return ", ".join(parts)


class Executor:
    """Drives intents through the kernel to the venue.

    Holds the broker, the kernel, the journal, and the ledger. Nothing else in
    the system is permitted a broker reference: the cognition plane emits intents
    and never learns whether a venue exists.
    """

    def __init__(
        self,
        broker: Broker,
        kernel: RiskKernel,
        journal: Journal,
        ledger: Ledger,
        *,
        armed: bool = False,
    ) -> None:
        self.broker = broker
        self.kernel = kernel
        self.journal = journal
        self.ledger = ledger
        # `armed` guards the real venue. Paper mode does not require arming.
        self.armed = armed

    async def execute(
        self,
        intents: list[OrderIntent],
        state: KernelState,
        *,
        correlation_id: str = "",
    ) -> ExecutionReport:
        """Run a batch. Exits are evaluated first so they free capacity."""
        report = ExecutionReport()
        if not intents:
            return report

        # Batch evaluation accumulates approved exposure, so twenty individually
        # compliant buys cannot collectively breach a portfolio cap.
        decisions = self.kernel.evaluate_batch(intents, state)
        working = state

        for decision in decisions:
            intent = decision.intent

            if not decision.approved:
                report.vetoed.append(decision)
                self.journal.append(
                    EventType.KERNEL_VETO,
                    {
                        "symbol": intent.symbol,
                        "side": intent.side.value,
                        "notional_usd": round(intent.notional_usd, 2),
                        "reason": intent.reason,
                        "vetoes": [v.value for v in decision.vetoes],
                        "notes": list(decision.notes),
                    },
                    correlation_id=correlation_id or intent.correlation_id,
                )
                continue

            try:
                working = await self._place_one(intent, working, report, correlation_id)
            except AmbiguousOrderState as exc:
                # Unknown venue state. Stop the cycle; a human must reconcile.
                report.halted_reason = str(exc)
                self.journal.append(
                    EventType.ERROR,
                    {"kind": "ambiguous_order_state", "symbol": intent.symbol, "error": str(exc)},
                    correlation_id=correlation_id,
                )
                log.error("executor.halted", symbol=intent.symbol, error=str(exc))
                break

        log.info("executor.batch_complete", summary=report.summary())
        return report

    async def _place_one(
        self,
        intent: OrderIntent,
        state: KernelState,
        report: ExecutionReport,
        correlation_id: str,
    ) -> KernelState:
        """Review, re-gate, place, journal, then book. In that order."""
        cid = correlation_id or intent.correlation_id
        key = intent.idempotency_key
        request = OrderRequest(
            symbol=intent.symbol,
            side=intent.side,
            notional_usd=intent.notional_usd,
            kind=intent.kind,
            limit_price=intent.limit_price,
            idempotency_key=key,
            correlation_id=cid,
        )

        # --- 1. Mandatory simulation. ---
        review = await self.broker.review(request)
        if not review.accepted:
            report.rejected.append((intent, f"review: {review.message}"))
            self.journal.append(
                EventType.REVIEW_REJECTED,
                {"symbol": intent.symbol, "message": review.message},
                correlation_id=cid,
            )
            return state

        state = state.with_review_passed(key)
        self.journal.append(
            EventType.REVIEW_PASSED,
            {
                "symbol": intent.symbol,
                "estimated_price": review.estimated_price,
                "estimated_quantity": review.estimated_quantity,
            },
            correlation_id=cid,
        )

        # --- 2. Final gate. State has changed since the batch evaluation. ---
        final = self.kernel.evaluate_before_place(intent, state)
        if not final.approved:
            report.vetoed.append(final)
            self.journal.append(
                EventType.KERNEL_VETO,
                {
                    "symbol": intent.symbol,
                    "stage": "pre_place",
                    "vetoes": [v.value for v in final.vetoes],
                    "notes": list(final.notes),
                },
                correlation_id=cid,
            )
            return state

        if not self.armed and self.broker.name != "paper":
            report.rejected.append((intent, "live path not armed"))
            self.journal.append(
                EventType.ORDER_FAILED,
                {"symbol": intent.symbol, "reason": "live path not armed"},
                correlation_id=cid,
            )
            return state

        # --- 3. Place. ---
        try:
            result = await self.broker.place(request)
        except OrderRejected as exc:
            report.rejected.append((intent, str(exc)))
            self.journal.append(
                EventType.ORDER_FAILED,
                {"symbol": intent.symbol, "error": str(exc)},
                correlation_id=cid,
            )
            return state

        if not result.accepted:
            report.rejected.append((intent, result.message))
            self.journal.append(
                EventType.ORDER_FAILED,
                {"symbol": intent.symbol, "message": result.message, "order_id": result.order_id},
                correlation_id=cid,
            )
            return state

        # --- 4. Journal BEFORE the ledger, so a crash leaves evidence. ---
        report.placed.append(result)
        self.journal.append(
            EventType.ORDER_PLACED,
            {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "notional_usd": round(intent.notional_usd, 2),
                "reason": intent.reason,
                "order_id": result.order_id,
                "idempotency_key": key,
                "thesis": intent.thesis,
                "invalidation": intent.invalidation,
            },
            correlation_id=cid,
        )

        # --- 5. Book confirmed fills only. A submitted-unfilled order books none. ---
        for fill in result.fills:
            if self.ledger.apply_fill(fill):
                report.fills.append(fill)
                self.journal.append(
                    EventType.FILL,
                    {
                        "symbol": fill.symbol,
                        "side": fill.side.value,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "order_id": fill.order_id,
                        "slippage_bps": fill.slippage_bps,
                    },
                    correlation_id=cid,
                )

        return state.with_order_placed(key)

    async def reconcile(
        self,
        state: KernelState,
        *,
        prices: dict[str, float],
        correlation_id: str = "",
    ) -> tuple[KernelState, bool]:
        """Compare ledger to broker. Divergence trips a breaker.

        Returns (state, clean). A break halts NEW risk but never blocks exits:
        a bot that freezes while holding losers is worse than one that never traded.
        """
        broker_positions = await self.broker.get_positions()
        broker_equity = await self.broker.get_account_equity()
        result = self.ledger.reconcile(
            broker_positions, broker_equity=broker_equity, prices=prices
        )

        payload = {
            "clean": result.clean,
            "ledger_equity": round(result.ledger_equity, 2),
            "broker_equity": round(broker_equity, 2),
            "divergences": [d.describe() for d in result.divergences],
        }
        if result.clean:
            self.journal.append(EventType.RECONCILIATION, payload, correlation_id=correlation_id)
            return state, True

        self.journal.append(
            EventType.RECONCILIATION_BREAK, payload, correlation_id=correlation_id
        )
        tripped = state.breakers.trip(
            VetoCode.BREAKER_TRIPPED, f"reconciliation break: {result.describe()}"
        )
        self.journal.append(
            EventType.BREAKER_TRIPPED,
            {"cause": "reconciliation", "detail": result.describe()},
            correlation_id=correlation_id,
        )
        from dataclasses import replace as dc_replace

        return dc_replace(state, breakers=tripped), False
