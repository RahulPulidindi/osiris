"""The daily loop. Runs unattended, start to finish.

  wake -> kill-switch -> reconcile -> breakers -> universe -> regime -> rank
       -> diff target book -> kernel -> execute -> verify -> attribute -> sleep

Two ordering decisions are load-bearing:

**Reconcile before deciding.** Ranking against a portfolio you do not actually
own produces a correct plan for a fictional account. Reconciliation is step one
for the same reason a bank counts the vault before lending.

**Exits survive a halt.** A tripped breaker or engaged kill switch stops new
entries, but risk exits and rank exits still run. A bot that freezes while
holding losers is worse than one that never traded.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import numpy as np

from osiris.cognition.funnel import FunnelTrace, RankingFunnel
from osiris.cognition.schemas import TargetHolding
from osiris.config import RiskLimits, Settings
from osiris.data.macro import MacroCalendar, is_trading_session
from osiris.data.regime import RegimeState, classify_regime
from osiris.execution.broker import Broker
from osiris.execution.executor import ExecutionReport, Executor
from osiris.execution.journal import EventType, Journal
from osiris.execution.killswitch import KillSwitch
from osiris.execution.ledger import DailyPnL, Ledger
from osiris.execution.rebalance import (
    ExitSignal,
    build_rebalance_plan,
    detect_invalidation_exits,
)
from osiris.kernel.kernel import RiskKernel
from osiris.kernel.state import KernelState, evaluate_breakers
from osiris.logging import get_logger
from osiris.types import Quote

log = get_logger(__name__)


@dataclass
class MarketSnapshot:
    """Everything the cycle needs about the world. Assembled by the caller.

    Passed in rather than fetched here so the loop is testable offline and the
    same code path runs in backtest, paper, and live.
    """

    as_of: date
    universe: list[str]
    closes: dict[str, np.ndarray] = field(default_factory=dict)
    quotes: dict[str, Quote] = field(default_factory=dict)
    benchmark_closes: np.ndarray | None = None
    adv: dict[str, float] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)
    betas: dict[str, float] = field(default_factory=dict)
    benchmark_sector_weights: dict[str, float] = field(default_factory=dict)
    tradable: dict[str, bool] = field(default_factory=dict)
    next_earnings: dict[str, datetime] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)

    def prices(self) -> dict[str, float]:
        """Mid-quote where available, else the last close."""
        out: dict[str, float] = {}
        for symbol, quote in self.quotes.items():
            out[symbol] = quote.mid
        for symbol, closes in self.closes.items():
            if symbol not in out and closes.size:
                out[symbol] = float(closes[-1])
        return out


@dataclass
class CycleResult:
    correlation_id: str
    as_of: date
    ran: bool
    reason: str = ""
    regime: RegimeState | None = None
    trace: FunnelTrace | None = None
    targets: list[TargetHolding] = field(default_factory=list)
    report: ExecutionReport | None = None
    reconciled_clean: bool = True
    equity: float = 0.0
    halted: bool = False
    # The ranked names with their theses. Carried on the result because the
    # reasoning is the product here, not a side effect: a dashboard that shows
    # trades without the argument behind them is not auditable.
    scores: list = field(default_factory=list)
    # Set when ranking failed. The cycle still completes -- exits must run -- but
    # "research broke" and "no opportunities today" are opposite situations and
    # must not produce the same summary.
    ranking_error: str | None = None

    def summary(self) -> str:
        if not self.ran:
            return f"cycle skipped: {self.reason}"
        parts = [f"equity ${self.equity:,.2f}"]
        if self.ranking_error:
            parts.append(f"RANKING FAILED: {self.ranking_error[:80]}")
        if self.regime:
            parts.append(f"regime {self.regime.regime.value}")
        if self.trace:
            parts.append(f"funnel {self.trace.summary()}")
        if self.report:
            parts.append(self.report.summary())
        if self.halted:
            parts.append("HALTED (exits only)")
        return " | ".join(parts)


class DailyLoop:
    """One trading cycle, end to end, with no human in the path."""

    def __init__(
        self,
        *,
        settings: Settings,
        limits: RiskLimits,
        broker: Broker,
        kernel: RiskKernel,
        journal: Journal,
        ledger: Ledger,
        funnel: RankingFunnel | None = None,
        macro: MacroCalendar | None = None,
        killswitch: KillSwitch | None = None,
        pnl: DailyPnL | None = None,
    ) -> None:
        self.settings = settings
        self.limits = limits
        self.broker = broker
        self.kernel = kernel
        self.journal = journal
        self.ledger = ledger
        self.funnel = funnel
        self.macro = macro or MacroCalendar()
        self.killswitch = killswitch or KillSwitch()
        self.pnl = pnl or DailyPnL()
        self.executor = Executor(
            broker, kernel, journal, ledger, armed=settings.live_armed
        )
        self.breakers = None

    async def run_cycle(
        self,
        snapshot: MarketSnapshot,
        *,
        target_override: list[TargetHolding] | None = None,
        theses_override: dict[str, str] | None = None,
        invalidations_override: dict[str, str] | None = None,
    ) -> CycleResult:
        cid = uuid.uuid4().hex[:16]
        prices = snapshot.prices()

        self.journal.append(
            EventType.CYCLE_START,
            {"as_of": snapshot.as_of.isoformat(), "universe": len(snapshot.universe)},
            correlation_id=cid,
        )

        if not is_trading_session(snapshot.as_of):
            # Name the actual reason. "not a trading session" left the operator to
            # work out whether it was a weekend, a holiday, or after hours.
            from osiris.data.macro import describe_session

            return self._skip(cid, snapshot, describe_session())

        # --- 1. Reconcile FIRST. Deciding against a fictional book is worse
        #        than not trading. ---
        state = self._build_state(snapshot, prices)
        state, clean = await self.executor.reconcile(state, prices=prices, correlation_id=cid)

        # --- 2. Roll the day's marks and evaluate breakers. ---
        equity = self.ledger.equity(prices)
        # Roll on every NEW session, not merely the first one.
        #
        # The previous form was `if day_start_equity <= 0`, which meant a funded
        # account never rolled at all: `day_start_equity` stayed pinned to the
        # opening balance forever. Two consequences, both silent --
        #   1. "Day P&L" reported profit-since-inception, so the dashboard's daily
        #      number was wrong by the entire lifetime return of the account.
        #   2. `consecutive_losses` is incremented by `roll_day`, so the
        #      consecutive-loss breaker could never fire. A circuit breaker that
        #      cannot trip is strictly worse than no breaker, because it reads as
        #      a passing check.
        if self.pnl.is_new_session(snapshot.as_of):
            self.pnl.roll_day(equity, as_of=snapshot.as_of)
        self.pnl.mark(equity)
        state = self._apply_pnl(state, snapshot, prices)

        breakers = evaluate_breakers(
            state,
            daily_loss_halt_pct=self.limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=self.limits.max_drawdown_halt_pct,
            consecutive_loss_halt=self.limits.consecutive_loss_halt,
            ledger_divergence=not clean,
        )
        if breakers.is_tripped and not state.breakers.is_tripped:
            self.journal.append(
                EventType.BREAKER_TRIPPED,
                {"reasons": list(breakers.reasons)},
                correlation_id=cid,
            )
        from dataclasses import replace as dc_replace

        state = dc_replace(state, breakers=breakers)

        ks = self.killswitch.check()
        if ks.engaged:
            self.journal.append(
                EventType.KILL_SWITCH, {"reason": ks.reason}, correlation_id=cid
            )
            state = dc_replace(state, kill_switch_engaged=True)

        halted = breakers.is_tripped or ks.engaged

        # --- 3. Regime. Gates chart vision and informs the strategist. ---
        regime = None
        if snapshot.benchmark_closes is not None and snapshot.benchmark_closes.size:
            regime = classify_regime(
                snapshot.benchmark_closes, universe_closes=snapshot.closes
            )
            self.journal.append(
                EventType.REGIME_CLASSIFIED,
                {"regime": regime.regime.value, "detail": regime.detail,
                 "breadth": round(regime.breadth, 3)},
                correlation_id=cid,
            )

        # --- 4. Rank, unless halted. A halted agent still computes exits. ---
        trace: FunnelTrace | None = None
        targets: list[TargetHolding] = list(target_override or [])
        # An intent with no falsifiable invalidation is auto-vetoed by the
        # kernel, so a caller supplying its own targets MUST also supply the
        # conditions under which each is wrong.
        theses: dict[str, str] = dict(theses_override or {})
        invalidations: dict[str, str] = dict(invalidations_override or {})
        scores: list = []
        ranking_error: str | None = None

        if target_override is None and not halted and self.funnel is not None:
            try:
                plan, trace, scores = await self.funnel.run(
                    snapshot.as_of,
                    snapshot.universe,
                    snapshot.closes,
                    regime=regime.regime.value if regime else "unknown",
                    sectors=snapshot.sectors,
                    metrics=snapshot.metrics,
                )
                # Only scale DOWN an over-allocated plan, and cap at 97% so there
                # is room for spread and drift between sizing and fill.
                #
                # Deliberately not scaled UP. The PM's first live plan summed to
                # 0.82 and its notes explained why: "thin evidence base... red team
                # reduction flags on top 4 names... prudent risk management." That
                # is a judgement about conviction, and overriding it would silently
                # discard the reasoning the whole funnel exists to produce.
                # Under-allocation is now surfaced instead of corrected.
                targets = plan.normalized(target_invested=0.97, scale_up=False)
                invested = sum(h.target_weight for h in targets)
                if invested < 0.80:
                    log.info(
                        "loop.plan_holds_cash",
                        invested_pct=round(invested, 3),
                        notes=plan.portfolio_notes[:200],
                    )
                theses = {s.symbol: s.thesis for s in scores}
                invalidations = {s.symbol: s.invalidation for s in scores}
                self.journal.append(
                    EventType.FUNNEL_TRACE,
                    {"summary": trace.summary(), "final": trace.final,
                     "llm_usd": round(trace.llm_usd, 4),
                     "research_usd": round(trace.research_usd, 4)},
                    correlation_id=cid,
                )
                # Record the REASONING alongside the weights. Without the thesis
                # and invalidation here, the journal proves what was decided but
                # not why -- and the dashboard has nothing to show for an order
                # that was blocked before it ever reached a fill record.
                by_symbol = {s.symbol: s for s in scores}
                self.journal.append(
                    EventType.PLAN_PRODUCED,
                    {
                        "holdings": [
                            {
                                "symbol": h.symbol,
                                "weight": h.target_weight,
                                "thesis": getattr(
                                    by_symbol.get(h.symbol), "thesis", ""
                                ),
                                "invalidation": getattr(
                                    by_symbol.get(h.symbol), "invalidation", ""
                                ),
                                "score": getattr(by_symbol.get(h.symbol), "score", 0.0),
                            }
                            for h in targets
                        ],
                        "notes": plan.portfolio_notes,
                    },
                    correlation_id=cid,
                )
            except Exception as exc:
                log.error("loop.ranking_failed", error=str(exc))
                self.journal.append(
                    EventType.ERROR,
                    {"stage": "ranking", "error": str(exc)},
                    correlation_id=cid,
                )
                # Fail safe: no new targets means exits still process below.
                targets = []
                # Record it on the RESULT too. Continuing is correct, but the
                # cycle previously reported "0 fills, 0 vetoed" and read as a
                # quiet market -- a failed ranking and a market with no
                # opportunities must never look the same.
                ranking_error = str(exc)

        portfolio = self.ledger.to_portfolio(prices)

        # --- 5. Exit signals are computed regardless of halt state. ---
        exit_signals: list[ExitSignal] = detect_invalidation_exits(
            portfolio,
            entry_prices=snapshot.entry_prices,
            prices=prices,
            stop_loss_pct=self.limits.stop_loss_pct,
        )

        if halted:
            # Liquidate nothing wholesale, but honor every risk exit and drop
            # all entries by targeting the current book minus forced exits.
            targets = [
                TargetHolding(symbol=p.symbol, target_weight=p.market_value / portfolio.equity)
                for p in portfolio.positions
                if portfolio.equity > 0
                and p.symbol not in {s.symbol for s in exit_signals}
            ]

        vols = {
            s: float(np.std(np.diff(c) / c[:-1], ddof=1) * np.sqrt(252))
            for s, c in snapshot.closes.items()
            if c.size > 21
        }

        rebalance = build_rebalance_plan(
            portfolio,
            targets,
            prices=prices,
            exit_signals=exit_signals,
            vols=vols,
            adv=snapshot.adv,
            theses=theses,
            invalidations=invalidations,
            max_symbol_weight=self.limits.max_symbol_weight,
            max_adv_participation=self.limits.max_adv_participation,
            max_trade_notional_pct=self.limits.max_trade_notional_pct,
            target_position_count=self.limits.target_position_count,
            correlation_id=cid,
        )

        for intent in rebalance.all_intents:
            self.journal.append(
                EventType.INTENT_EMITTED,
                {"symbol": intent.symbol, "side": intent.side.value,
                 "notional_usd": round(intent.notional_usd, 2),
                 "reason": intent.reason,
                 # Carried so a VETOED order still has its reasoning on record.
                 # Previously only `order_placed` held the thesis, so anything the
                 # kernel blocked appeared in the feed with no explanation of what
                 # the agent wanted or why.
                 "thesis": intent.thesis,
                 "invalidation": intent.invalidation},
                correlation_id=cid,
            )

        # --- 6. Execute. The kernel vetoes entries while halted. ---
        report = await self.executor.execute(
            rebalance.all_intents, state, correlation_id=cid
        )

        final_equity = self.ledger.equity(prices)
        # Record the close so the next session's P&L is measured from here. The
        # overnight gap is where nearly all of a daily-rebalanced book's move
        # happens, so this is the reference point that makes "Day P&L" real.
        self.pnl.close_session(final_equity)
        self.journal.append(
            EventType.CYCLE_END,
            {"as_of": snapshot.as_of.isoformat(),
             "equity": round(final_equity, 2),
             "fills": len(report.fills),
             "vetoed": len(report.vetoed),
             "halted": halted},
            correlation_id=cid,
        )

        result = CycleResult(
            correlation_id=cid,
            as_of=snapshot.as_of,
            ran=True,
            regime=regime,
            trace=trace,
            targets=targets,
            report=report,
            reconciled_clean=clean,
            equity=final_equity,
            halted=halted,
            scores=scores,
            ranking_error=ranking_error,
        )
        log.info("loop.cycle_complete", summary=result.summary())
        return result

    # ------------------------------------------------------------------ helpers
    def _skip(self, cid: str, snapshot: MarketSnapshot, reason: str) -> CycleResult:
        """Record a skipped session, carrying the CURRENT equity forward.

        Reporting 0.0 here would be read downstream as a total loss: a caller
        marking the equity curve would append a zero, and the derived return
        series would contain a -100% observation that never happened. A skipped
        session means "no change", not "no money".
        """
        self.journal.append(
            EventType.CYCLE_END, {"skipped": reason}, correlation_id=cid
        )
        return CycleResult(
            correlation_id=cid,
            as_of=snapshot.as_of,
            ran=False,
            reason=reason,
            equity=self.ledger.equity(snapshot.prices()),
        )

    def _build_state(
        self, snapshot: MarketSnapshot, prices: dict[str, float]
    ) -> KernelState:
        self.ledger.set_metadata(snapshot.sectors, snapshot.betas)
        blackout, blackout_reason = self.macro.is_blackout(
            datetime.now(UTC)
        )
        return KernelState(
            portfolio=self.ledger.to_portfolio(prices),
            quotes=snapshot.quotes,
            adv=snapshot.adv,
            sectors=snapshot.sectors,
            betas=snapshot.betas,
            benchmark_sector_weights=snapshot.benchmark_sector_weights,
            tradable=snapshot.tradable,
            next_earnings=snapshot.next_earnings,
            now=datetime.now(UTC),
            macro_blackout=blackout,
            macro_blackout_reason=blackout_reason,
            day_start_equity=self.pnl.day_start_equity,
            peak_equity=self.pnl.peak_equity,
            consecutive_losses=self.pnl.consecutive_losses,
        )

    def _apply_pnl(
        self, state: KernelState, snapshot: MarketSnapshot, prices: dict[str, float]
    ) -> KernelState:
        from dataclasses import replace as dc_replace

        return dc_replace(
            state,
            portfolio=self.ledger.to_portfolio(prices),
            day_start_equity=self.pnl.day_start_equity,
            peak_equity=self.pnl.peak_equity,
            consecutive_losses=self.pnl.consecutive_losses,
        )
