"""The guardian: continuous intraday risk monitoring.

The system has two loops at two speeds, on purpose:

  - The BRAIN (DailyLoop): research -> rank -> rebalance. Expensive, once per
    session. It decides what to own and where every exit level is.
  - The GUARDIAN (this): while the market is open, every minute or so, pull
    quotes for the HELD names only and enforce the exits the brain already
    committed to. No LLM, no research, no new positions -- it can only reduce
    risk, never add it.

Why this exists: a stop that is checked once a day at the open is not a stop.
A held name gapping down 30% at 11am would previously sit unmanaged until the
next morning's cycle. Signal is daily; risk is continuous.

Why it cannot overtrade: the guardian emits SELL intents exclusively, and only
when a stop or a breaker-level loss actually fires. Every intent still goes
through the full executor pipeline -- kernel, mandatory broker simulation,
kernel again -- so the guardian gets no shortcut past the safety rails. On a
quiet day it costs a few quote calls per minute and places nothing.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace as dc_replace
from datetime import UTC, datetime

from osiris.config import RiskLimits, Settings
from osiris.data.macro import is_market_open
from osiris.execution.executor import Executor
from osiris.execution.journal import EventType, Journal
from osiris.execution.killswitch import KillSwitch
from osiris.execution.ledger import DailyPnL, Ledger
from osiris.execution.rebalance import detect_invalidation_exits
from osiris.kernel.kernel import RiskKernel
from osiris.kernel.state import KernelState, evaluate_breakers
from osiris.logging import get_logger
from osiris.types import OrderIntent, OrderKind, Side

log = get_logger(__name__)


class Guardian:
    """Watches held positions between research cycles and fires their exits.

    Shares the ledger, journal, kernel, and executor with the daily loop, and
    the caller passes a lock so the two never trade at the same instant.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        limits: RiskLimits,
        executor: Executor,
        kernel: RiskKernel,
        journal: Journal,
        ledger: Ledger,
        pnl: DailyPnL,
        killswitch: KillSwitch | None = None,
        stop_loss_pct: float = 0.15,
    ) -> None:
        self.settings = settings
        self.limits = limits
        self.executor = executor
        self.kernel = kernel
        self.journal = journal
        self.ledger = ledger
        self.pnl = pnl
        self.killswitch = killswitch or KillSwitch()
        self.stop_loss_pct = stop_loss_pct
        # Fire each stop once. Without this, a stop that fails to fill (or
        # fills partially) would be re-fired every tick, burning the daily
        # order budget on one symbol. Cleared when the position is gone or a
        # new session begins.
        self._fired: set[str] = set()

    async def tick(
        self,
        quotes: dict,
        *,
        tradable: dict[str, bool] | None = None,
        now: datetime | None = None,
    ) -> list[OrderIntent]:
        """One pass: mark equity, evaluate breakers, fire breached stops.

        Returns the intents it acted on, mostly for tests. Quotes are passed in
        rather than fetched so the loop is testable offline and the caller
        controls the data source.
        """
        now = now or datetime.now(UTC)
        held = self.ledger.open_symbols
        if not held:
            self._fired.clear()
            return []

        # Forget symbols that are no longer held so a future re-entry gets a
        # working stop again.
        self._fired &= set(held)

        prices = {s: q.mid for s, q in quotes.items() if s in held}
        if not prices:
            log.warning("guardian.no_quotes", held=len(held))
            return []

        # --- Mark. This is what makes breakers real intraday. ---
        equity = self.ledger.equity(prices)
        self.pnl.mark(equity)

        portfolio = self.ledger.to_portfolio(prices)
        entry_prices = {
            s: p.avg_cost
            for s, p in self.ledger.positions.items()
            if p.quantity > 0
        }

        # --- Stops. The same detector the daily loop uses. ---
        signals = [
            s
            for s in detect_invalidation_exits(
                portfolio,
                entry_prices=entry_prices,
                prices=prices,
                stop_loss_pct=self.stop_loss_pct,
            )
            if s.symbol not in self._fired
        ]

        # --- Breakers, marked-to-market. A tripped breaker halts new entries
        #     in the NEXT research cycle; exits below still run. ---
        state = KernelState(
            portfolio=portfolio,
            quotes=quotes,
            tradable=tradable or dict.fromkeys(held, True),
            now=now,
            day_start_equity=self.pnl.day_start_equity,
            peak_equity=self.pnl.peak_equity,
            consecutive_losses=self.pnl.consecutive_losses,
            kill_switch_engaged=self.killswitch.check().engaged,
        )
        breakers = evaluate_breakers(
            state,
            daily_loss_halt_pct=self.limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=self.limits.max_drawdown_halt_pct,
            consecutive_loss_halt=self.limits.consecutive_loss_halt,
        )
        if breakers.is_tripped:
            state = dc_replace(state, breakers=breakers)

        if not signals:
            return []

        cid = f"guardian-{uuid.uuid4().hex[:12]}"
        weights = {p.symbol: p.market_value for p in portfolio.positions}
        intents: list[OrderIntent] = []
        for signal in signals:
            notional = weights.get(signal.symbol, 0.0)
            if notional <= 0:
                continue
            intents.append(
                OrderIntent(
                    symbol=signal.symbol,
                    side=Side.SELL,
                    notional_usd=notional,
                    kind=OrderKind.MARKET,  # a firing stop does not haggle
                    invalidation=signal.detail or signal.reason,
                    reason=signal.reason,
                    correlation_id=cid,
                )
            )
            self.journal.append(
                EventType.INTENT_EMITTED,
                {
                    "symbol": signal.symbol,
                    "side": "sell",
                    "notional_usd": round(notional, 2),
                    "reason": signal.reason,
                    "thesis": "",
                    "invalidation": signal.detail,
                    "source": "guardian",
                },
                correlation_id=cid,
            )

        if not intents:
            return []

        log.warning(
            "guardian.stops_firing",
            symbols=[i.symbol for i in intents],
            equity=round(equity, 2),
        )
        # Full pipeline: kernel -> review -> kernel -> place. No shortcuts.
        report = await self.executor.execute(intents, state, correlation_id=cid)
        for intent in intents:
            self._fired.add(intent.symbol)

        if report.fills:
            log.warning(
                "guardian.exits_filled",
                fills=[(f.symbol, round(f.quantity, 4)) for f in report.fills],
            )
        return intents

    def on_new_session(self) -> None:
        """Reset per-session memory. Called by the runner after the daily roll."""
        self._fired.clear()


async def run_guardian(
    guardian: Guardian,
    *,
    fetch_quotes,
    interval_seconds: int,
    trade_lock: asyncio.Lock,
    on_mark=None,
    is_open=is_market_open,
) -> None:
    """The watch loop. Runs forever; sleeps while the market is closed.

    `fetch_quotes(symbols) -> dict[str, Quote]` is injected so this loop knows
    nothing about MCP. `trade_lock` is shared with the daily cycle so the
    guardian never sells a position the brain is mid-way through rebalancing.
    `on_mark(prices, equity)` lets the runner push live marks to the dashboard.
    """
    log.info("guardian.started", interval_s=interval_seconds)
    while True:
        try:
            if not is_open():
                # Check twice a minute whether the bell has rung. Cheap, and it
                # keeps the loop simple compared to duplicating the scheduler.
                await asyncio.sleep(30)
                continue

            held = guardian.ledger.open_symbols
            if held:
                quotes = await fetch_quotes(held)
                async with trade_lock:
                    await guardian.tick(quotes)
                if on_mark is not None and quotes:
                    prices = {s: q.mid for s, q in quotes.items()}
                    on_mark(prices, guardian.ledger.equity(prices))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A failed tick must not kill the watch. Log and try again next
            # interval; the daily reconcile will catch anything it missed.
            log.error("guardian.tick_failed", error=str(exc))

        await asyncio.sleep(interval_seconds)
