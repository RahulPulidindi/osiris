"""The 5-stage ranking funnel.

The LLM does not deeply read 1000 names: that costs roughly $210/month in search
alone plus hours of latency. A cascade spends expensive tokens only where they
change the ranking.

  Stage 0: universe (~1000)   PIT membership + liquidity floor.  Free, no LLM.
  Stage 1: pre-rank (~150)    Local factors from OHLCV.          Free, no LLM.
  Stage 2: triage (~40)       Cheap model + Exa fast search.
  Stage 3: deep (~40)         Exa contents + charts + roles.
  Stage 4: construct (20)     PM assembles the book.

FUNNEL LEAKAGE is the risk this introduces. The published alpha came from
BREADTH. If Stage 1 discards a name the deep pass would have ranked top-20, the
funnel destroys the signal being paid for. So Stage 1 ranks on cheap
breadth-preserving features and cuts conservatively -- never on a thesis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from osiris.cognition.roles import CognitionPipeline
from osiris.cognition.schemas import ChartRead, PortfolioPlan, StrategistScore
from osiris.data.indicators import (
    distance_from_high,
    momentum,
    rsi,
    volatility_annualized,
)
from osiris.logging import get_logger

log = get_logger(__name__)


@dataclass
class StageBudget:
    """Per-stage width and spend caps so a bug cannot run away."""

    prerank_width: int = 150
    deep_width: int = 40
    final_count: int = 20

    @classmethod
    def for_book(cls, final_count: int, *, prerank_width: int = 150) -> StageBudget:
        """Widths scaled to the book the kernel will actually allow.

        The defaults assume a 20-name book. Researching 40 names to fill 5 is not
        merely wasteful -- it overflows every downstream stage, because the
        strategist and red team each emit one record per candidate. That produced a
        cascade of truncations at successive stages, each looking like a new bug.

        4x the target keeps real selection breadth (a 5-name book chooses from 20)
        while keeping every response inside its budget.
        """
        deep = max(10, min(40, final_count * 4))
        return cls(
            prerank_width=prerank_width,
            deep_width=deep,
            final_count=final_count,
        )
    max_llm_usd: float = 10.0
    max_research_usd: float = 3.0


@dataclass
class FunnelTrace:
    """Per-stage record. Feeds the funnel-fidelity gate and the dashboard."""

    as_of: date
    universe: list[str] = field(default_factory=list)
    preranked: list[str] = field(default_factory=list)
    triaged: list[str] = field(default_factory=list)
    deep_researched: list[str] = field(default_factory=list)
    final: list[str] = field(default_factory=list)
    stage_of: dict[str, int] = field(default_factory=dict)
    llm_usd: float = 0.0
    research_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.universe)} -> {len(self.preranked)} -> {len(self.triaged)} "
            f"-> {len(self.deep_researched)} -> {len(self.final)} "
            f"(${self.llm_usd + self.research_usd:.2f})"
        )


def prerank(
    closes_by_symbol: dict[str, np.ndarray],
    *,
    width: int,
) -> list[str]:
    """Stage 1: cheap, breadth-preserving pre-rank. No LLM, no thesis.

    Deliberately NOT a momentum filter. A naive momentum screen drops exactly the
    "good news not yet in the price" names the research found the model was good
    at spotting -- which would delete the edge before cognition ever sees it.

    Instead this scores on a blend that keeps candidates from several regimes:
    trend participation, mean-reversion candidates, and low-volatility quality.
    """
    scored: list[tuple[str, float]] = []
    for symbol, closes in closes_by_symbol.items():
        if closes.size < 60:
            continue
        mom = momentum(closes, lookback=min(126, closes.size - 1))
        vol = volatility_annualized(closes)
        rsi_vals = rsi(closes)
        rsi_now = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0
        dist_high = distance_from_high(closes)

        # Three independent reasons to look closer. Summing them keeps the
        # candidate pool heterogeneous rather than one factor's top slice.
        trend_component = np.tanh(mom * 3.0)
        reversion_component = np.tanh((40.0 - rsi_now) / 20.0) if rsi_now < 40 else 0.0
        quality_component = 1.0 / (1.0 + max(vol, 0.05) * 3.0)
        near_high_component = np.tanh((dist_high + 0.15) * 5.0)

        score = (
            0.35 * trend_component
            + 0.25 * reversion_component
            + 0.20 * quality_component
            + 0.20 * near_high_component
        )
        scored.append((symbol, float(score)))

    scored.sort(key=lambda kv: -kv[1])
    return [s for s, _ in scored[:width]]


class RankingFunnel:
    """Runs the cascade and records a trace."""

    def __init__(
        self,
        pipeline: CognitionPipeline,
        research_client=None,
        budget: StageBudget | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.research = research_client
        self.budget = budget or StageBudget()

    async def run(
        self,
        as_of: date,
        universe: list[str],
        closes_by_symbol: dict[str, np.ndarray],
        *,
        regime: str = "unknown",
        sectors: dict[str, str] | None = None,
        chart_reads: dict[str, ChartRead] | None = None,
        metrics: dict[str, dict] | None = None,
    ) -> tuple[PortfolioPlan, FunnelTrace, list[StrategistScore]]:
        trace = FunnelTrace(as_of=as_of, universe=list(universe))
        for s in universe:
            trace.stage_of[s] = 0

        # --- Stage 1: pre-rank (free) ---
        preranked = prerank(
            {s: c for s, c in closes_by_symbol.items() if s in set(universe)},
            width=self.budget.prerank_width,
        )
        trace.preranked = preranked
        for s in preranked:
            trace.stage_of[s] = 1
        log.info("funnel.prerank", kept=len(preranked), from_universe=len(universe))

        if not preranked:
            trace.notes.append("pre-rank produced no candidates")
            return PortfolioPlan(), trace, []

        # --- Stage 2: cheap triage ---
        candidates = {
            s: (metrics or {}).get(s, {"momentum": round(momentum(closes_by_symbol[s]), 4)})
            for s in preranked
            if s in closes_by_symbol
        }
        try:
            triage_results = await self.pipeline.triage(candidates)
        except Exception as exc:
            log.warning("funnel.triage.failed", error=str(exc))
            triage_results = []

        if triage_results:
            ranked = sorted(triage_results, key=lambda r: -r.interest)
            triaged = [r.symbol for r in ranked[: self.budget.deep_width]]
        else:
            # Fail forward on the quant pre-rank rather than halting the cycle.
            triaged = preranked[: self.budget.deep_width]
            trace.notes.append("triage unavailable; fell back to quant pre-rank")

        trace.triaged = triaged
        for s in triaged:
            trace.stage_of[s] = 2

        # --- Stage 3: deep research ---
        notes = await self._deep_research(triaged, as_of, metrics or {})
        trace.deep_researched = [n.symbol for n in notes]
        for n in notes:
            trace.stage_of[n.symbol] = 3

        if not notes:
            trace.notes.append("no analyst notes produced")
            return PortfolioPlan(), trace, []

        scores = await self.pipeline.score(notes, regime=regime, chart_reads=chart_reads)
        reviews = await self.pipeline.red_team(scores)

        # --- Stage 4: construction ---
        plan = await self.pipeline.construct(
            scores,
            reviews,
            target_count=self.budget.final_count,
            sectors=sectors,
        )
        trace.final = [h.symbol for h in plan.holdings]
        for s in trace.final:
            trace.stage_of[s] = 4

        trace.llm_usd = self.pipeline.llm.ledger.spent_usd
        if self.research is not None:
            trace.research_usd = self.research.costs.usd
        log.info("funnel.complete", summary=trace.summary())
        return plan, trace, scores

    async def _deep_research(
        self, symbols: list[str], as_of: date, metrics: dict[str, dict]
    ) -> list:
        """Fetch documents and produce analyst notes, bounded by concurrency."""

        async def one(symbol: str):
            docs = []
            if self.research is not None and self.research.enabled:
                try:
                    docs = await self.research.research_symbol(
                        symbol, as_of=as_of, deep=True
                    )
                except Exception as exc:
                    log.warning("funnel.research.failed", symbol=symbol, error=str(exc))
            try:
                return await self.pipeline.analyze(symbol, docs, metrics.get(symbol))
            except Exception as exc:
                log.warning("funnel.analyze.failed", symbol=symbol, error=str(exc))
                return None

        results = await asyncio.gather(*(one(s) for s in symbols))
        return [r for r in results if r is not None]


def measure_funnel_fidelity(
    funnel_final: list[str], full_width_final: list[str]
) -> float:
    """Overlap between the funnel's output and a full-width deep pass.

    Run monthly. Low fidelity means the funnel is discarding names the model
    would have chosen -- the funnel is the bug, not the model.
    """
    if not full_width_final:
        return 0.0
    a, b = set(funnel_final), set(full_width_final)
    return len(a & b) / len(b)
