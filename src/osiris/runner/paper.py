"""Paper-trading runner.

Drives the real DailyLoop through the real kernel, executor, ledger, and journal
against a synthetic market. The only substitution is the broker and the data
source; **every risk decision runs the production code path.** If paper and live
took different paths, paper would prove nothing about live.

The ranker here is a deterministic quant model rather than the LLM funnel. That is
deliberate for the offline runner: it makes the mechanical half of the system --
diffing, sizing, gating, reconciliation, accounting -- verifiable without spending
tokens or requiring credentials. The LLM funnel plugs into the same seam via
`DailyLoop(funnel=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from osiris.api.events import Channel
from osiris.api.state import RuntimeState
from osiris.cognition.funnel import prerank
from osiris.cognition.schemas import TargetHolding
from osiris.data.indicators import momentum, volatility_annualized
from osiris.execution.broker import PaperBroker
from osiris.execution.loop import CycleResult, DailyLoop, MarketSnapshot
from osiris.kernel.kernel import RiskKernel
from osiris.logging import get_logger
from osiris.runner.synthetic import SyntheticMarket

log = get_logger(__name__)


@dataclass
class QuantRanker:
    """Deterministic cross-sectional ranker used for offline paper runs.

    Scores on risk-adjusted momentum. Chosen because it is transparent, cheap, and
    -- importantly -- **it is a baseline the LLM must beat.** A ranking engine that
    cannot outperform a two-line momentum screen has not earned its token bill,
    and the evaluation gates are what surface that.
    """

    target_count: int = 20
    max_weight: float = 0.10

    def rank(
        self, closes: dict[str, np.ndarray], sectors: dict[str, str]
    ) -> tuple[list[TargetHolding], list[dict]]:
        scored: list[tuple[str, float, float]] = []
        for symbol, series in closes.items():
            if series.size < 130:
                continue
            mom = momentum(series, lookback=126)
            vol = volatility_annualized(series, window=20)
            if vol <= 1e-6:
                continue
            # Risk-adjusted: raw momentum systematically over-weights the most
            # volatile names, which is how a "momentum" book becomes a vol book.
            score = float(mom / vol)
            scored.append((symbol, score, vol))

        scored.sort(key=lambda t: -t[1])
        chosen = scored[: self.target_count]

        rows: list[dict] = []
        for i, (symbol, score, vol) in enumerate(scored):
            rows.append(
                {
                    "symbol": symbol,
                    "rank": i + 1,
                    # Clamp into the schema's -5..5 band without distorting order.
                    "score": float(np.clip(score, -5.0, 5.0)),
                    "conviction": float(np.clip(abs(score) / 3.0, 0.0, 1.0)),
                    "stage": 4 if i < self.target_count else 1,
                    "thesis": (
                        f"risk-adjusted 6m momentum {score:+.2f} "
                        f"(annualized vol {vol:.0%})"
                    ),
                    "invalidation": (
                        "exits the top-20 on the next rebalance, or realized vol "
                        f"exceeds {vol * 1.75:.0%}"
                    ),
                    "target_weight": 0.0,
                    "sources": [],
                }
            )

        if not chosen:
            return [], rows

        # Inverse-vol weighting inside the book, capped per name.
        inv = {s: 1.0 / max(v, 0.05) for s, _, v in chosen}
        total = sum(inv.values())
        holdings: list[TargetHolding] = []
        by_symbol = {r["symbol"]: r for r in rows}
        for symbol, _score, _vol in chosen:
            weight = min(self.max_weight, inv[symbol] / total)
            holdings.append(
                TargetHolding(
                    symbol=symbol,
                    target_weight=weight,
                    rationale=by_symbol[symbol]["thesis"],
                )
            )
            by_symbol[symbol]["target_weight"] = weight

        return holdings, rows


@dataclass
class PaperRunner:
    """Runs N sessions against a synthetic market, updating live API state."""

    state: RuntimeState
    market: SyntheticMarket
    ranker: QuantRanker = field(default_factory=QuantRanker)
    warmup_days: int = 140
    # Off by default: a 200-session replay would emit hundreds of notifications.
    # Enabled for the live/scheduled runner, where each cycle is a real day.
    alerter: object | None = None

    def __post_init__(self) -> None:
        broker = self.state.broker
        if not isinstance(broker, PaperBroker):
            raise TypeError("PaperRunner requires a PaperBroker")
        self.broker = broker
        self.loop = DailyLoop(
            settings=self.state.settings,
            limits=self.state.limits,
            broker=broker,
            kernel=RiskKernel(self.state.limits, self.state.settings.account_type),
            journal=self.state.journal,
            ledger=self.state.ledger,
            killswitch=self.state.killswitch,
            pnl=self.state.pnl,
        )

    async def run_session(self, day_index: int) -> CycleResult:
        """One trading session, driven through the production loop."""
        snapshot = self.market.snapshot(day_index)
        prices = self.market.prices_on(day_index)

        # The broker's view of the market must be updated before orders price.
        self.broker.set_quotes(snapshot.quotes)
        self.broker.set_adv(snapshot.adv)

        targets, rows = self.ranker.rank(snapshot.closes, snapshot.sectors)

        # Entry prices feed the mechanical stop check on later sessions.
        snapshot.entry_prices = {
            symbol: pos.avg_cost
            for symbol, pos in self.state.ledger.positions.items()
            if pos.quantity > 0
        }

        result = await self.loop.run_cycle(
            snapshot,
            target_override=targets,
            theses_override={r["symbol"]: r["thesis"] for r in rows},
            # Required: the kernel auto-vetoes any intent without a falsifiable
            # invalidation condition.
            invalidations_override={r["symbol"]: r["invalidation"] for r in rows},
        )

        self._publish(result, snapshot, prices, rows)
        self._alert(result)
        return result

    def _alert(self, result: CycleResult) -> None:
        """Emit alerts for this cycle, if an alerter is attached.

        Wrapped defensively: the whole premise of the alerting layer is that a
        notification failure can never affect trading, and that guarantee has to
        hold at the call site too, not just inside each sink.
        """
        if self.alerter is None:
            return
        try:
            from osiris.runner.alerts import alerts_for_cycle

            for alert in alerts_for_cycle(result):
                self.alerter.send(alert)
        except Exception as exc:
            log.warning("paper.alerting_failed", error=str(exc))

    def _publish(
        self,
        result: CycleResult,
        snapshot: MarketSnapshot,
        prices: dict[str, float],
        rows: list[dict],
    ) -> None:
        """Push cycle output into the API caches and the SSE stream."""
        st = self.state
        st.mark_prices(prices)
        st.sectors.update(snapshot.sectors)
        st.betas.update(snapshot.betas)
        st.benchmark_sector_weights = dict(snapshot.benchmark_sector_weights)
        st.closes = snapshot.closes
        st.ranking = rows
        st.theses = {r["symbol"]: r["thesis"] for r in rows}
        st.invalidations = {r["symbol"]: r["invalidation"] for r in rows}
        st.ledger.set_metadata(snapshot.sectors, snapshot.betas)
        st.breakers = getattr(result, "breakers", st.breakers)

        preranked = prerank(snapshot.closes, width=st.settings.funnel_prerank_width)
        st.funnel_stages = [
            {"stage": 0, "count": len(snapshot.universe), "cost_usd": 0.0},
            {"stage": 1, "count": len(preranked), "cost_usd": 0.0},
            {"stage": 2, "count": min(40, len(preranked)), "cost_usd": 0.0},
            {"stage": 3, "count": min(40, len(preranked)), "cost_usd": 0.0},
            {"stage": 4, "count": len(result.targets), "cost_usd": 0.0},
        ]

        if result.report:
            st.realized_slippage_bps.extend(
                f.slippage_bps for f in result.report.fills if f.slippage_bps is not None
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

        benchmark = float(snapshot.benchmark_closes[-1]) if snapshot.benchmark_closes is not None else None
        # Pass the SIMULATED date: without it every session collapses onto today
        # and the return series stays empty.
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

    async def run(self, sessions: int) -> list[CycleResult]:
        """Run `sessions` consecutive sessions after the indicator warmup."""
        results: list[CycleResult] = []
        start = max(self.warmup_days, 130)
        end = min(start + sessions, self.market.n_days)

        for day_index in range(start, end):
            # Skip weekends: the loop refuses non-trading sessions, and a runner
            # that silently produced zero-fill cycles would look like a bug.
            if self.market.dates[day_index].weekday() >= 5:
                continue
            results.append(await self.run_session(day_index))

        log.info(
            "paper.run_complete",
            sessions=len(results),
            final_equity=round(results[-1].equity, 2) if results else 0.0,
        )
        return results
