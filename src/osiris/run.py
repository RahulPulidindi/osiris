"""Run Osiris against your real Robinhood account.

    python -m osiris.run --once       # one cycle, then stop
    python -m osiris.run --serve      # dashboard + scheduled daily cycles
    python -m osiris.run --dry-run    # decide and explain, place nothing

The single entry point for live operation. It wires the live MCP connection to
the same loop the paper runner used, so the code path that touches money is the
one that was tested.

`--dry-run` is the default. Placing real orders requires BOTH `OSIRIS_MODE=live`
and `OSIRIS_I_UNDERSTAND_THE_RISK=yes`, because one stray environment variable
should not be able to arm an account.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from datetime import UTC, datetime, timedelta

from osiris.api.events import Channel
from osiris.api.state import RuntimeState, build_runtime_state
from osiris.config import (
    DATA_DIR,
    AccountType,
    Mode,
    RiskLimits,
    load_risk_limits,
    load_settings,
)
from osiris.data.live import LiveMarket
from osiris.execution.loop import DailyLoop
from osiris.execution.mcp_broker import MCPBroker
from osiris.kernel.kernel import RiskKernel
from osiris.logging import configure_logging, get_logger
from osiris.mcp.session import LiveConnection
from osiris.runner.alerts import alerts_for_cycle, build_alerter, cycle_failed
from osiris.runner.schedule import describe_wait, market_status, next_wake

log = get_logger(__name__)

# Liquid large-caps used when the account exposes no scanner capability. Explicit
# rather than silent: screening the market and trading a fixed list are different
# strategies, and you should know which one is running.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "LLY",
    "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ", "NFLX", "ABBV",
    "BAC", "CRM", "CVX", "KO", "AMD", "PEP", "TMO", "WMT", "LIN", "ADBE",
    "MRK", "ACN", "MCD", "ABT", "CSCO", "IBM", "GE", "INTU", "TXN", "QCOM",
    "CAT", "VZ", "DHR", "NEE", "RTX", "SPGI", "PFE", "AMGN", "HON", "UNP",
    "LOW", "COP", "BKNG", "MS", "GS", "BLK", "ISRG", "NOW", "AMAT", "SBUX",
]


def build_funnel(settings, *, target_count: int = 20):
    """The LLM research funnel, or None when no API key is configured.

    Returning None rather than raising is deliberate: without a key the agent
    should still reconcile, manage exits, and report -- it just cannot form new
    opinions. Refusing to start would leave existing positions unmanaged, which
    is worse than running read-only.
    """
    import os

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        log.warning(
            "run.no_llm_key",
            detail="OPENROUTER_API_KEY unset: no new research, exits still managed",
        )
        return None

    from osiris.cognition.funnel import RankingFunnel, StageBudget
    from osiris.cognition.llm import LLMClient
    from osiris.cognition.roles import CognitionPipeline
    from osiris.data.research import ResearchClient

    llm = LLMClient(key, daily_usd_ceiling=settings.llm_daily_usd_ceiling)

    # Exa supplies the evidence for stages 2-3. Without it the models reason from
    # price history alone, which is a materially weaker input than filings and
    # primary sources -- so its absence is logged rather than passing silently.
    exa_key = os.environ.get("EXA_API_KEY", "").strip()
    research = ResearchClient(exa_key) if exa_key else None
    if research is None:
        log.warning(
            "run.no_research_key",
            detail="EXA_API_KEY unset: theses will rest on price history alone",
        )

    return RankingFunnel(
        CognitionPipeline(llm),
        research_client=research,
        # Widths scale to the book the KERNEL will allow. Researching 40 names for
        # a 5-name book is not just wasted spend: the strategist and red team emit
        # one record per candidate, so an oversized funnel truncates them and the
        # cycle produces nothing.
        budget=StageBudget.for_book(
            target_count, prerank_width=settings.funnel_prerank_width
        ),
    )


class LiveAgent:
    """Connects to Robinhood and runs the autonomous cycle.

    Owns the connection lifecycle so a single object can be started, run many
    cycles, and shut down cleanly.
    """

    def __init__(
        self,
        *,
        dry_run: bool = True,
        unrestricted: bool = False,
        force_session: bool = False,
    ) -> None:
        self.settings = load_settings()
        self.unrestricted = unrestricted
        self.force_session = force_session
        # Limits are finalized in start(), once real equity is known: whether the
        # defaults are even feasible depends on account size.
        self.limits = (
            RiskLimits.unrestricted() if unrestricted else load_risk_limits()
        )
        if unrestricted:
            log.warning(
                "run.unrestricted_limits",
                detail="position and loss caps disabled by configuration",
            )
        # Arming requires two independent affirmations AND the absence of
        # --dry-run. Any one of the three withheld means nothing is placed.
        self.armed = self.settings.live_armed and not dry_run
        self.dry_run = not self.armed
        self.conn: LiveConnection | None = None
        self.state: RuntimeState | None = None
        self.market: LiveMarket | None = None
        self.loop: DailyLoop | None = None
        self.alerter = build_alerter()
        self.guardian = None
        # Serializes the daily cycle against the guardian's exit ticks. Both
        # mutate the same ledger through the same executor; interleaving them
        # could sell a position mid-rebalance.
        self.trade_lock = asyncio.Lock()

    async def start(self) -> None:
        """Open the connection and assemble the trading system."""
        self.conn = LiveConnection(
            endpoint=self.settings.mcp_endpoint,
            # dry_run on the adapter blocks WRITE tools at the transport layer,
            # so an unarmed run cannot place an order even given a bug upstream.
            dry_run=self.dry_run,
        )
        await self.conn.open(interactive=False)

        missing = self.conn.require_capabilities(
            "getPortfolio", "listPositions", "getQuotes", "reviewOrder"
        )
        if missing:
            raise RuntimeError(
                f"Account is missing required capabilities: {missing}. "
                "Run `python -m osiris.connect` for the full report."
            )

        broker = MCPBroker(self.conn.adapter)

        # Robinhood requires `account_number` on portfolio and position reads.
        # Without it every read fails validation, so resolve it before anything
        # depends on it rather than letting each call fail separately.
        if await broker.resolve_account() is None:
            raise RuntimeError(
                "Could not determine your account number. Portfolio and position "
                "reads require it. Run `python -m osiris.connect` to see the "
                "account listing."
            )

        equity = await broker.get_account_equity()
        log.info("run.account_read", equity=round(equity, 2))

        # Scale limits to the real balance. The defaults target a five-figure
        # account; at a few hundred dollars they are infeasible rather than
        # cautious, and the agent would veto nearly every order it proposed --
        # indistinguishable from being broken.
        if not self.unrestricted:
            scaled = RiskLimits.for_equity(equity)
            if scaled.target_position_count != self.limits.target_position_count:
                log.warning(
                    "run.limits_scaled_for_account_size",
                    equity=round(equity, 2),
                    target_positions=scaled.target_position_count,
                    per_order_pct=scaled.max_trade_notional_pct,
                    detail="small account: book concentrates so orders can clear",
                )
            self.limits = scaled

        settings = self.settings.model_copy(
            update={
                "account_equity_usd": equity,
                # The account is real, so treat sizing as margin-capable only if
                # Phase 0 established it. Unknown stays unknown.
                "account_type": self.settings.account_type,
            }
        )

        self.state = build_runtime_state(
            settings=settings,
            limits=self.limits,
            journal_path=DATA_DIR / "journal-live.jsonl",
            broker=broker,
        )
        # Seed the ledger from the venue: the broker is truth, and starting from
        # an empty ledger would report every existing holding as a divergence.
        await self._seed_ledger(broker, equity)

        self.market = LiveMarket(self.conn.adapter)
        self.loop = DailyLoop(
            settings=settings,
            limits=self.limits,
            broker=broker,
            kernel=RiskKernel(self.limits, settings.account_type),
            journal=self.state.journal,
            ledger=self.state.ledger,
            funnel=build_funnel(
                settings, target_count=self.limits.target_position_count
            ),
            killswitch=self.state.killswitch,
            pnl=self.state.pnl,
        )

        # The guardian shares the loop's executor and kernel so its exits run
        # the identical pipeline. It only ever sells; it cannot open positions.
        from osiris.execution.guardian import Guardian

        self.guardian = Guardian(
            settings=settings,
            limits=self.limits,
            executor=self.loop.executor,
            kernel=self.loop.kernel,
            journal=self.state.journal,
            ledger=self.state.ledger,
            pnl=self.state.pnl,
            killswitch=self.state.killswitch,
            stop_loss_pct=self.limits.stop_loss_pct,
        )

    async def _seed_ledger(self, broker: MCPBroker, equity: float) -> None:
        """Adopt the venue's current positions as the ledger's opening state.

        Without this, reconciliation on the first cycle reports every real holding
        as an `unrecorded_position` and trips a breaker immediately -- which looks
        like a bug but is the ledger correctly noticing it knows nothing.
        """
        positions = await broker.get_positions()
        if not positions:
            log.info("run.ledger_seeded", positions=0, note="flat account")
            return

        from osiris.execution.ledger import LedgerPosition

        quotes = await self.market.quotes(list(positions)) if self.market else {}
        for symbol, qty in positions.items():
            price = quotes[symbol].last if symbol in quotes else 0.0
            self.state.ledger.positions[symbol] = LedgerPosition(
                symbol=symbol,
                quantity=qty,
                # Cost basis is unknown from a positions read. Marking at current
                # price makes unrealized P&L start at zero rather than inventing
                # a gain; realized P&L from before Osiris is not ours to claim.
                cost_basis_total=qty * price,
                opened_at=datetime.now(UTC),
            )
        held_value = sum(
            p.quantity * (quotes[s].last if s in quotes else 0.0)
            for s, p in self.state.ledger.positions.items()
        )
        self.state.ledger.cash = max(0.0, equity - held_value)
        log.info(
            "run.ledger_seeded",
            positions=len(positions),
            cash=round(self.state.ledger.cash, 2),
        )

    async def run_cycle(self):
        """One full autonomous cycle against the live account."""
        if not (self.loop and self.market and self.state):
            raise RuntimeError("start() must run before run_cycle()")

        held = self.state.ledger.open_symbols
        universe = await self.market.universe(fallback=FALLBACK_UNIVERSE)
        snapshot = await self.market.snapshot(universe=universe, held=held)
        # A new session began: let the guardian re-arm stops it fired yesterday.
        if self.guardian is not None and self.loop.pnl.is_new_session(snapshot.as_of):
            self.guardian.on_new_session()

        if self.force_session and not snapshot.universe == []:
            # Stamp the snapshot with the most recent real trading day so the loop
            # will proceed. Only ever combined with dry-run: pretending the market
            # is open while it is shut would place orders against stale quotes,
            # which the spread and staleness gates exist to prevent.
            from osiris.data.macro import is_trading_session, session_date

            probe = session_date()
            while not is_trading_session(probe):
                probe -= timedelta(days=1)
            log.warning(
                "run.forced_session",
                as_of=probe.isoformat(),
                detail="market is closed; quotes are stale, research path only",
            )
            snapshot.as_of = probe

        if not snapshot.universe:
            log.error("run.no_tradable_symbols")
            self.state.publish(
                Channel.ERROR,
                {"error": "No symbols resolved a quote and price history."},
            )
            return None

        # Entry prices drive the mechanical stop check.
        snapshot.entry_prices = {
            s: p.avg_cost
            for s, p in self.state.ledger.positions.items()
            if p.quantity > 0
        }

        async with self.trade_lock:
            result = await self.loop.run_cycle(snapshot)
        self._publish(result, snapshot)

        for alert in alerts_for_cycle(result):
            self.alerter.send(alert)
        return result

    def _publish(self, result, snapshot) -> None:
        """Push cycle output into the caches the dashboard reads."""
        st = self.state
        prices = snapshot.prices()
        st.mark_prices(prices)
        st.sectors.update(snapshot.sectors)
        st.betas.update(snapshot.betas)
        st.benchmark_sector_weights = dict(snapshot.benchmark_sector_weights)
        st.closes = snapshot.closes
        st.ledger.set_metadata(snapshot.sectors, snapshot.betas)
        st.breakers = getattr(result, "breakers", st.breakers)

        if result.scores:
            scores = result.scores
            st.ranking = [
                {
                    "symbol": s.symbol,
                    "rank": i + 1,
                    "score": s.score,
                    "conviction": s.conviction,
                    # `StrategistScore` has no `stage`: reaching the strategist IS
                    # stage 3. Reading a non-existent attribute crashed the run
                    # AFTER a complete, successful cycle -- the work was done and
                    # then thrown away at the display layer.
                    "stage": 3,
                    "thesis": s.thesis,
                    "invalidation": s.invalidation,
                    "target_weight": 0.0,
                    # The field is `citations`, not `sources`.
                    "sources": list(s.citations),
                }
                for i, s in enumerate(scores)
            ]
            st.theses = {s.symbol: s.thesis for s in scores}
            st.invalidations = {s.symbol: s.invalidation for s in scores}

        if result.report:
            st.realized_slippage_bps.extend(
                f.slippage_bps
                for f in result.report.fills
                if f.slippage_bps is not None
            )
            for fill in result.report.fills:
                st.publish(
                    Channel.FILL,
                    {
                        "symbol": fill.symbol,
                        "side": fill.side.value,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "order_id": fill.order_id,
                    },
                )
            for decision in result.report.vetoed:
                st.publish(
                    Channel.VETO,
                    {
                        "symbol": decision.intent.symbol,
                        "vetoes": [v.value for v in decision.vetoes],
                    },
                )

        benchmark = (
            float(snapshot.benchmark_closes[-1])
            if snapshot.benchmark_closes is not None
            and snapshot.benchmark_closes.size
            else None
        )
        st.record_equity(result.equity, benchmark, as_of=result.as_of)
        st.publish(
            Channel.CYCLE,
            {
                "as_of": result.as_of.isoformat(),
                "summary": result.summary(),
                "halted": result.halted,
                "equity": result.equity,
            },
        )

    async def reconnect(self) -> None:
        """Re-establish the MCP session, preserving journal and ledger state.

        Required for an always-on process. A streamable-HTTP session does not
        survive indefinitely -- tokens refresh, sockets drop, the server restarts
        -- and without this the agent would run until the first disconnect and
        then log the same error every session forever.

        Deliberately reuses the existing `RuntimeState`: the journal is
        append-only and the ledger's beliefs are reconciled against the broker at
        the start of every cycle, so rebuilding them would discard history for no
        gain and re-seed positions that are already known.
        """
        log.info("run.reconnect_starting")
        try:
            if self.conn is not None:
                await self.conn.close()
        except Exception as exc:
            # Closing a broken connection often fails. That is expected and must
            # not prevent opening a new one.
            log.debug("run.reconnect_close_failed", error=str(exc))

        self.conn = LiveConnection(
            endpoint=self.settings.mcp_endpoint, dry_run=self.dry_run
        )
        await self.conn.open(interactive=False)

        broker = MCPBroker(self.conn.adapter)
        if await broker.resolve_account() is None:
            raise RuntimeError("reconnected but could not resolve the account")

        # Rebind the broker everywhere it is held. Missing one of these would
        # leave a component talking to the dead session.
        self.state.broker = broker
        self.loop.broker = broker
        self.loop.executor.broker = broker
        self.market = LiveMarket(self.conn.adapter)
        log.info("run.reconnected")

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()


def _banner(agent: LiveAgent) -> None:
    mode = "LIVE — REAL ORDERS" if agent.armed else "DRY RUN — nothing placed"
    equity = agent.state.settings.account_equity_usd if agent.state else 0.0
    limits = agent.limits
    print("\n" + "=" * 64)
    print(f"  OSIRIS  ·  {mode}")
    print("=" * 64)
    print(f"  account value ...... ${equity:,.2f}")
    print(f"  account type ....... {agent.settings.account_type.value}")
    print(f"  target book ........ {limits.target_position_count} names "
          f"(~${equity / max(1, limits.target_position_count):,.2f} each)")
    # Dollar figures alongside percentages: "2% per order" is abstract, "$7.32"
    # tells you immediately whether the agent can buy anything at all.
    print(f"  per-order cap ...... {limits.max_trade_notional_pct:.0%} "
          f"(${equity * limits.max_trade_notional_pct:,.2f})")
    print(f"  single-name cap .... {limits.max_symbol_weight:.0%} "
          f"(${equity * limits.max_symbol_weight:,.2f})")
    print(f"  daily loss halt .... {limits.daily_loss_halt_pct:.0%} "
          f"(${equity * limits.daily_loss_halt_pct:,.2f})")
    if agent.limits.max_symbol_weight >= 1.0:
        print("\n  WARNING: position caps are disabled. One name may become the")
        print("  entire account, and no loss level will halt trading.")
    if not agent.armed:
        # Spell out WHICH affirmation is missing. "not armed" sent me looking at
        # the wrong setting more than once, and an operator who believes the agent
        # is trading when it is only observing has the worst of both worlds:
        # no fills, and no idea why.
        print("\n  NOT ARMED — this process will research and explain, but will")
        print("  place no orders. Missing:")
        if agent.settings.mode is not Mode.LIVE:
            print("    · OSIRIS_MODE=live                    (currently: "
                  f"{agent.settings.mode.value})")
        if agent.settings.i_understand_the_risk.lower() != "yes":
            print("    · OSIRIS_I_UNDERSTAND_THE_RISK=yes    (currently: "
                  f"{agent.settings.i_understand_the_risk})")
        if agent.dry_run and agent.settings.live_armed:
            print("    · remove --dry-run from the command")
    print("=" * 64 + "\n")


def _wrap(text: str, width: int = 76, indent: str = " " * 16) -> str:
    """Wrap prose for the terminal instead of truncating it."""
    import textwrap

    lines = textwrap.wrap(text.strip(), width=width)
    return ("\n" + indent).join(lines) if lines else ""


async def run_once(agent: LiveAgent) -> int:
    await agent.start()
    _banner(agent)
    try:
        result = await agent.run_cycle()
        if result is None:
            print("Cycle produced no result (no tradable symbols).\n")
            return 1

        if not result.ran:
            # A skip is correct behavior, not a failure. Say so explicitly and
            # explain how to exercise the decision path anyway, since otherwise
            # the only way to test is to wait for the opening bell.
            print(f"\nMarket closed — no cycle run.\n  {result.reason}\n")
            print("Data fetching worked; the agent simply will not trade a closed")
            print("market. To exercise research and sizing right now:\n")
            print("  python -m osiris.run --once --dry-run --force-session\n")
            return 0

        print(f"\n{result.summary()}\n")
        if result.report:
            for fill in result.report.fills:
                print(f"  {fill.side.value.upper():<5} {fill.symbol:<6} "
                      f"{fill.quantity:>10.4f} @ ${fill.price:,.2f}")
            for decision in result.report.vetoed:
                codes = ", ".join(v.value for v in decision.vetoes)
                print(f"  BLOCK {decision.intent.symbol:<6} {codes}")

            # Explain the veto that --force-session makes inevitable. Weekend
            # quotes have enormous or absent spreads, so the spread gate blocks
            # everything -- correct behavior that reads like a broken gate.
            spread_blocked = [
                d
                for d in result.report.vetoed
                if any(v.value == "spread_too_wide" for v in d.vetoes)
            ]
            if spread_blocked and agent.force_session:
                print(
                    f"\n  Note: {len(spread_blocked)} order(s) blocked on spread.\n"
                    "  Expected with --force-session: a closed market has no live\n"
                    "  bid/ask, so measured spreads are hundreds of bps. The gate is\n"
                    "  working. Run during market hours for tradable spreads."
                )

        # The research result is the point of a dry run, so show it in full. The
        # reasoning IS the product here; a thesis clipped mid-word cannot be
        # evaluated, which defeats the purpose of a supervised run.
        if result.targets:
            invested = sum(h.target_weight for h in result.targets)
            print(f"\n  Target book ({invested:.0%} invested):")
            scores = {s.symbol: s for s in result.scores}
            for holding in result.targets:
                dollars = result.equity * holding.target_weight
                print(
                    f"\n    {holding.symbol:<6} {holding.target_weight:>6.1%}"
                    f"  (${dollars:,.2f})"
                )
                score = scores.get(holding.symbol)
                if score is None:
                    continue
                for label, text in (
                    ("why", score.thesis),
                    ("sells if", score.invalidation),
                ):
                    if text:
                        print(f"      {label}: {_wrap(text)}")
        # Say where to see this in the dashboard. `--once` runs a cycle and exits;
        # it never starts the API, so a dashboard left open at :5173 has nothing to
        # connect to and shows the onboarding screen -- which reads as "the agent
        # did not run" when in fact it ran and finished.
        print(
            "  This run is recorded in the journal. To view it in the dashboard:\n"
            "    python -m osiris.run --serve      (API + scheduled cycles)\n"
        )
        return 0
    finally:
        await agent.close()


async def run_serve(agent: LiveAgent, *, host: str, port: int, interval: int) -> int:
    """Serve the dashboard and run cycles on an interval."""
    import uvicorn

    from osiris.api import create_app
    from osiris.api.app import set_state

    await agent.start()
    _banner(agent)

    # State the schedule at startup. Without this, "always on" looks identical to
    # "hung": the process sits silent for hours and there is no way to tell
    # whether it is waiting for the bell or has stopped working.
    status = market_status()
    print("  SCHEDULE")
    print(f"    market time ...... {status['market_time'][:19].replace('T', ' ')} ET")
    print(f"    trading today .... {'yes' if status['is_trading_day'] else 'no'}")
    print(f"    next cycle ....... in {status['next_wake_in']}  ({status['reason']})")
    if interval > 0:
        print(f"    then every ....... {interval} minutes while open")
    else:
        print("    cadence .......... once per session, shortly after the open")
    print(f"    monitoring ....... held positions every "
          f"{agent.limits.watch_interval_seconds}s while open "
          f"(stop-loss {agent.limits.stop_loss_pct:.0%})")
    print()

    set_state(agent.state)

    async def cycles() -> None:
        """Wake on the market's schedule, not on a fixed timer.

        A plain `sleep(interval)` loop is wrong twice over: it spends LLM budget
        researching a closed market, and it DRIFTS -- a cycle at 09:31 puts the
        next at 10:31, so after the first day the agent never trades the open
        again. This sleeps until the next moment worth waking for.
        """
        consecutive_failures = 0
        # The session this loop last COMPLETED a cycle for. With interval=0 the
        # schedule answers "wake now" for the entire trading window, so without
        # this the loop re-ran the full research funnel every ~60s all session:
        # dozens of LLM passes and rebalance attempts per day on a strategy
        # designed for one. One cycle per session means exactly that.
        last_session_run: str = ""
        while True:
            from osiris.data.macro import session_date

            wake = next_wake(interval_minutes=interval)
            if (
                interval <= 0
                and wake.should_trade
                and session_date().isoformat() == last_session_run
            ):
                # Already traded this session; nothing to do until tomorrow.
                await asyncio.sleep(300)
                continue
            wait = wake.seconds_from()
            if wait > 1:
                log.info(
                    "run.sleeping",
                    until=wake.at.strftime("%a %H:%M %Z"),
                    duration=describe_wait(wait),
                    reason=wake.reason,
                )
                # Sleep in bounded slices so a config change or shutdown is not
                # blocked behind a 60-hour weekend sleep.
                while wait > 0:
                    await asyncio.sleep(min(wait, 300))
                    wait = wake.seconds_from()

            try:
                result = await agent.run_cycle()
                consecutive_failures = 0
                # Mark the session done only when the cycle actually RAN -- a
                # skip (market closed on arrival) must not eat the day's cycle.
                if result is not None and result.ran:
                    last_session_run = result.as_of.isoformat()
            except Exception as exc:
                # A failed cycle must not kill the server: the dashboard is how
                # an operator sees that cycles are failing.
                consecutive_failures += 1
                log.error(
                    "run.cycle_failed",
                    error=str(exc),
                    consecutive=consecutive_failures,
                )
                agent.state.publish(Channel.ERROR, {"error": str(exc)})
                agent.alerter.send(cycle_failed(str(exc), consecutive_failures))

                # Repeated failures usually mean a dropped MCP session, which no
                # amount of retrying the cycle will fix. Reconnect rather than
                # spinning against a dead socket for the rest of the day.
                if consecutive_failures >= 3:
                    log.warning("run.reconnecting", after_failures=consecutive_failures)
                    try:
                        await agent.reconnect()
                        consecutive_failures = 0
                    except Exception as reconnect_error:
                        log.error("run.reconnect_failed", error=str(reconnect_error))
                        await asyncio.sleep(300)

            # One cycle per session unless an intraday cadence is set: this
            # strategy ranks daily, so hourly runs multiply cost and turnover
            # without adding signal.
            if interval <= 0:
                await asyncio.sleep(60)

    # --- The guardian: continuous risk monitoring between research cycles. ---
    # Every `watch_interval_seconds` while the market is open it pulls quotes
    # for the HELD names only, marks equity (so the dashboard ticks live and
    # breakers see intraday losses), and fires any stop the daily cycle set.
    from osiris.execution.guardian import run_guardian

    async def fetch_held_quotes(symbols: list[str]):
        return await agent.market.quotes(symbols)

    def on_mark(prices: dict[str, float], equity: float) -> None:
        st = agent.state
        st.mark_prices(prices)
        st.pnl.mark(equity)
        day_start = st.pnl.day_start_equity
        peak = st.pnl.peak_equity
        st.publish(
            Channel.PNL,
            {
                "equity": equity,
                "daily_pnl": equity - day_start if day_start > 0 else 0.0,
                "daily_pnl_pct": (
                    (equity - day_start) / day_start if day_start > 0 else 0.0
                ),
                "peak_equity": peak,
                "drawdown_pct": (
                    max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
                ),
            },
        )

    watch = asyncio.create_task(
        run_guardian(
            agent.guardian,
            fetch_quotes=fetch_held_quotes,
            interval_seconds=agent.limits.watch_interval_seconds,
            trade_lock=agent.trade_lock,
            on_mark=on_mark,
        )
    )

    task = asyncio.create_task(cycles())
    config = uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    print(f"  dashboard: http://{host}:{port}\n")
    try:
        await uvicorn.Server(config).serve()
    finally:
        for t in (task, watch):
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await agent.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Osiris on your Robinhood account.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="run one cycle and exit")
    group.add_argument("--serve", action="store_true", help="dashboard + scheduled cycles")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and explain without placing orders (default unless armed)",
    )
    parser.add_argument(
        "--unrestricted",
        action="store_true",
        help="disable position and loss caps (simulation and kill switch remain)",
    )
    parser.add_argument(
        "--force-session",
        action="store_true",
        help=(
            "run the decision path while the market is closed, for testing. "
            "Requires --dry-run: quotes are stale when the market is shut."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        metavar="MINUTES",
        help=(
            "minutes between cycles while the market is open. Default 0 = one "
            "cycle per session, shortly after the open, which is the right "
            "cadence for a daily-ranked book."
        ),
    )
    args = parser.parse_args(argv)

    # Unbuffered stdout. An always-on process is normally run with its output
    # redirected to a file or a supervisor, and Python buffers stdout when it is
    # not a terminal -- so the startup banner and schedule would sit in a buffer
    # for hours while the log lines (stderr) appeared immediately. That makes a
    # correctly-sleeping agent look like one that never printed anything.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):  # pragma: no cover - non-standard stdout
        pass

    configure_logging()
    settings = load_settings()

    if settings.mode is Mode.LIVE and settings.account_type is AccountType.UNKNOWN:
        print(
            "\nRefusing live mode with an unknown account type.\n"
            "Cash vs margin governs settlement and sizing; trading a cash account\n"
            "as margin produces good-faith violations that surface weeks later.\n"
            "Set OSIRIS_ACCOUNT_TYPE=cash|margin.\n"
        )
        return 1

    if args.force_session and not args.dry_run:
        print(
            "\n--force-session requires --dry-run.\n"
            "Overriding the market calendar while armed would place real orders\n"
            "against stale quotes from a closed market.\n"
        )
        return 1

    agent = LiveAgent(
        dry_run=args.dry_run,
        unrestricted=args.unrestricted,
        force_session=args.force_session,
    )

    try:
        if args.serve:
            return asyncio.run(
                run_serve(
                    agent, host=args.host, port=args.port, interval=args.interval
                )
            )
        return asyncio.run(run_once(agent))
    except KeyboardInterrupt:
        print("\nStopped.\n")
        return 0
    except Exception as exc:
        print(f"\nFAILED: {exc}\n")
        if "credentials" in str(exc).lower() or "connect" in str(exc).lower():
            print("Run `python -m osiris.connect` first to authorize.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
