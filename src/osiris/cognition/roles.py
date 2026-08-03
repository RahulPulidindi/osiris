"""The four cognition roles plus triage and vision.

Pipeline: Analyst -> Strategist -> RedTeam -> PM. The Red Team is not decoration:
a single model asked to find trades will find trades, so the bear case is argued
by a different model family to decorrelate errors.
"""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from osiris.cognition.llm import LLMClient, Role
from osiris.cognition.prompts import SYSTEM_PROMPTS
from osiris.cognition.schemas import (
    AnalystNote,
    ChartRead,
    PortfolioPlan,
    PostmortemReport,
    RedTeamReview,
    StrategistScore,
    TriageResult,
)
from osiris.data.sanitize import build_data_block
from osiris.logging import get_logger
from osiris.types import Provenanced

log = get_logger(__name__)

# Batch ceilings for the roles whose OUTPUT scales with candidate count.
#
# The strategist and red team each emit one record per candidate, so response size
# grows with the input list. Without batching, widening the funnel silently
# truncates these stages -- which is what produced a cascade of failures at
# successive stages, each looking like an unrelated bug.
STRATEGIST_BATCH = 12
# Smaller than the strategist's: the red team writes a full bear case per name AND
# runs on a reasoning model, so its tokens-per-record is several times higher.
RED_TEAM_BATCH = 5

# Only names scoring above this are sent for adversarial review; below it they are
# not book candidates anyway. Defined once because the PM must use the SAME
# threshold to decide which names require a review -- if the two disagree, either
# unreviewed names slip into the book or the book empties for no reason.
RED_TEAM_THRESHOLD = 1.0


def _trim(value: object, limit: int = 600) -> str:
    """Bound one field's contribution to a shared prompt.

    Truncation is marked rather than silent, so the model can see that evidence
    was elided instead of treating a cut-off sentence as the complete picture.
    """
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " …[trimmed]"


def _looks_truncated(text: str) -> bool:
    """Heuristic: did the model run out of token budget mid-structure?

    Worth distinguishing because truncation and malformed output produce the same
    JSONDecodeError but need opposite fixes -- a larger budget versus a better
    prompt. Every `analyst.parse_failed` in the first live run was truncation, and
    the error message pointed at neither.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Balanced JSON ends on a closing brace or bracket.
    if stripped[-1] in "}]":
        return False
    return stripped.count("{") > stripped.count("}") or stripped.count(
        "["
    ) > stripped.count("]")


class CognitionPipeline:
    """Orchestrates the roles. Never touches a broker."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ------------------------------------------------------------- triage
    async def triage(
        self, candidates: dict[str, dict], *, batch_size: int = 40
    ) -> list[TriageResult]:
        """Stage 2: cheap screen over ~150 names in batches."""
        symbols = list(candidates)
        batches = [
            symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
        ]
        results = await asyncio.gather(
            *(self._triage_batch(b, candidates) for b in batches),
            return_exceptions=True,
        )
        out: list[TriageResult] = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
            else:
                log.warning("triage.batch.failed", error=str(r))
        return out

    async def _triage_batch(
        self, symbols: list[str], candidates: dict[str, dict]
    ) -> list[TriageResult]:
        lines = [
            f"{s}: " + ", ".join(f"{k}={v}" for k, v in candidates[s].items())
            for s in symbols
        ]
        user = "Screen these candidates:\n" + "\n".join(lines)
        # A 40-symbol batch cannot emit 40 JSON records within 1,500 tokens; the
        # response was cut off mid-array and the whole batch was discarded.
        resp = await self.llm.complete(
            Role.TRIAGE, SYSTEM_PROMPTS["triage"], user, max_tokens=6_000
        )
        payload = resp.parse_json()
        raw = payload.get("results", []) if isinstance(payload, dict) else payload
        return _parse_list(raw, TriageResult, "triage")

    # ------------------------------------------------------------ analyst
    async def analyze(
        self, symbol: str, docs: list[Provenanced], metrics: dict | None = None
    ) -> AnalystNote | None:
        """Summarize evidence for one name. Documents are wrapped as data."""
        user = (
            f"Symbol: {symbol}\n"
            f"Quantitative context: {metrics or {}}\n\n"
            f"{build_data_block(docs)}\n\n"
            f"Summarize the evidence for {symbol} as specified."
        )
        # 4,000 not 1,500. At 1,500 the model ran out of budget mid-JSON and every
        # response failed to parse with a delimiter error around char 4,000 -- which
        # reads like a malformed-output problem rather than a truncation one. 17 of
        # 20 analyses were silently discarded that way, so the strategist scored on
        # a third of the evidence it should have had.
        resp = await self.llm.complete(
            Role.ANALYST, SYSTEM_PROMPTS["analyst"], user, max_tokens=4_000
        )
        try:
            note = AnalystNote.model_validate(resp.parse_json())
        except (ValidationError, ValueError) as exc:
            log.warning(
                "analyst.parse_failed",
                symbol=symbol,
                error=str(exc),
                # Distinguish truncation from genuinely bad JSON. Without this the
                # two are indistinguishable in the logs, and they need opposite
                # fixes: more budget vs a better prompt.
                truncated=_looks_truncated(resp.content),
                chars=len(resp.content),
            )
            return None
        if note.was_targeted:
            log.warning(
                "analyst.manipulation_detected",
                symbol=symbol,
                signals=note.manipulation_signals,
            )
        return note

    # --------------------------------------------------------- strategist
    async def score(
        self,
        notes: list[AnalystNote],
        *,
        regime: str = "unknown",
        chart_reads: dict[str, ChartRead] | None = None,
    ) -> list[StrategistScore]:
        """Cross-sectional scoring. Chart reads are supplied as one feature.

        Batched above a threshold. Scoring is cross-sectional by design, so batching
        does cost some comparative context -- but a batch of 12 still supports real
        relative judgement, whereas one oversized call gets truncated and the tail
        of the list is simply never scored. Partial context beats absent records.
        """
        if not notes:
            return []

        if len(notes) > STRATEGIST_BATCH:
            batches = [
                notes[i : i + STRATEGIST_BATCH]
                for i in range(0, len(notes), STRATEGIST_BATCH)
            ]
            results = await asyncio.gather(
                *(
                    self.score(b, regime=regime, chart_reads=chart_reads)
                    for b in batches
                ),
                return_exceptions=True,
            )
            out: list[StrategistScore] = []
            for r in results:
                if isinstance(r, list):
                    out.extend(r)
                else:
                    log.error("strategist.batch_failed", error=str(r))
            return out

        blocks = []
        for n in notes:
            chart = (chart_reads or {}).get(n.symbol)
            chart_line = (
                f"\n  chart (one feature, statistically recalibrated): "
                f"trend={chart.trend}, persistence={chart.trend_persistence}, "
                f"confidence={chart.confidence:.2f}"
                if chart and chart.is_trustworthy
                else ""
            )
            # Cap each note's contribution. Analyst notes vary hugely in length,
            # and one verbose name would otherwise consume the shared prompt
            # budget that the other nineteen need -- a scoring bias determined by
            # prose length rather than by evidence.
            blocks.append(
                f"{n.symbol}:\n"
                f"  facts: {_trim(n.facts)}\n"
                f"  positive: {_trim(n.positive_evidence)}\n"
                f"  negative: {_trim(n.negative_evidence)}\n"
                f"  evidence_quality: {n.evidence_quality}{chart_line}"
            )
        user = (
            f"Market regime: {regime}\n\n"
            f"Score these candidates cross-sectionally:\n\n"
            + "\n\n".join(blocks)
        )
        # 8,000: this call emits one record per candidate, so its output scales
        # with the deep-research width rather than being fixed. At 4,000 it
        # truncated at ~11,900 chars on 20 names. Truncation is now salvaged
        # rather than fatal, but a budget that fits the whole response is still
        # better than relying on recovery.
        resp = await self.llm.complete(
            Role.STRATEGIST, SYSTEM_PROMPTS["strategist"], user, max_tokens=8_000
        )
        payload = resp.parse_json()
        raw = payload.get("scores", []) if isinstance(payload, dict) else payload
        return _parse_list(raw, StrategistScore, "strategist")

    # ----------------------------------------------------------- red team
    async def red_team(self, scores: list[StrategistScore]) -> list[RedTeamReview]:
        """Adversarial review. Runs on a different model family by design.

        Batched for the same reason as the strategist: output scales with candidate
        count, so a single call silently truncates once the list grows. A red team
        whose reviews get discarded is worse than none -- the vetoes vanish while
        the theses survive, so the system looks reviewed and is not.
        """
        candidates = [s for s in scores if s.score > RED_TEAM_THRESHOLD]
        if not candidates:
            return []

        batches = [
            candidates[i : i + RED_TEAM_BATCH]
            for i in range(0, len(candidates), RED_TEAM_BATCH)
        ]
        results = await asyncio.gather(
            *(self._red_team_batch(b) for b in batches), return_exceptions=True
        )

        reviews: list[RedTeamReview] = []
        for r in results:
            if isinstance(r, list):
                reviews.extend(r)
            else:
                # A lost batch means those names went UNREVIEWED. Surfaced loudly
                # because the safe response is to treat them as unvetted, not to
                # assume the silence was approval.
                log.error("red_team.batch_failed", error=str(r))

        unreviewed = {s.symbol for s in candidates} - {r.symbol for r in reviews}
        if unreviewed:
            log.warning("red_team.unreviewed", symbols=sorted(unreviewed))

        vetoed = [r.symbol for r in reviews if r.is_veto]
        if vetoed:
            log.info("red_team.vetoed", symbols=vetoed)
        return reviews

    async def _red_team_batch(
        self, candidates: list[StrategistScore]
    ) -> list[RedTeamReview]:
        blocks = [
            f"{s.symbol}: score={s.score:.2f}, conviction={s.conviction:.2f}\n"
            f"  thesis: {_trim(s.thesis, 400)}\n"
            f"  invalidation: {_trim(s.invalidation, 300)}"
            for s in candidates
        ]
        user = "Attack each of these theses:\n\n" + "\n\n".join(blocks)
        # 16,000. The red team runs on a REASONING model (gpt-5), which spends
        # tokens thinking before it emits anything -- so its usable output budget is
        # far smaller than the ceiling suggests. At 6,000 it produced 4 of 9 reviews
        # and truncated, and the unreviewed names then formed the whole book.
        resp = await self.llm.complete(
            Role.RED_TEAM, SYSTEM_PROMPTS["red_team"], user, max_tokens=16_000
        )
        payload = resp.parse_json()
        raw = payload.get("reviews", []) if isinstance(payload, dict) else payload
        return _parse_list(raw, RedTeamReview, "red_team")

    # ------------------------------------------------------------------ pm
    async def construct(
        self,
        scores: list[StrategistScore],
        reviews: list[RedTeamReview],
        *,
        target_count: int = 20,
        sectors: dict[str, str] | None = None,
    ) -> PortfolioPlan:
        """Assemble the final book, honoring red-team vetoes.

        Fails CLOSED on missing reviews. A name the red team never assessed is
        excluded, not admitted.

        This is the difference between "no objection was raised" and "no objection
        exists". On the first full live run the red team's response truncated after
        4 of 9 reviews, and the 5 unreviewed names became the entire book -- every
        position in it had bypassed adversarial review while the log reported
        vetoes working normally. The review is load-bearing precisely because a
        single model asked to find trades will find trades.
        """
        vetoed = {r.symbol for r in reviews if r.is_veto}
        reduce = {r.symbol for r in reviews if r.verdict == "reduce"}
        reviewed = {r.symbol for r in reviews}

        # Only names that were actually reviewed are candidates. Scores at or below
        # the red team's threshold were never sent for review, so requiring one
        # would exclude the entire book; those are handled by the score filter.
        skipped = [
            s.symbol
            for s in scores
            if s.score > RED_TEAM_THRESHOLD and s.symbol not in reviewed
        ]
        if skipped:
            log.warning(
                "pm.excluded_unreviewed",
                symbols=sorted(skipped),
                detail="red team never assessed these; excluded rather than assumed safe",
            )

        eligible = [
            s
            for s in scores
            if s.symbol not in vetoed
            and s.score > 0
            and not (s.score > RED_TEAM_THRESHOLD and s.symbol not in reviewed)
        ]
        if not eligible:
            return PortfolioPlan(
                portfolio_notes=(
                    "no eligible candidates after review"
                    + (f"; {len(skipped)} excluded as unreviewed" if skipped else "")
                )
            )

        blocks = [
            f"{s.symbol}: score={s.score:.2f}, conviction={s.conviction:.2f}, "
            f"sector={(sectors or {}).get(s.symbol, 'Unknown')}"
            + (" [RED TEAM: reduce]" if s.symbol in reduce else "")
            for s in sorted(eligible, key=lambda x: -x.score)
        ]
        user = (
            f"Target position count: {target_count}\n"
            f"Vetoed by red team (do not include): {sorted(vetoed) or 'none'}\n\n"
            f"Candidates:\n" + "\n".join(blocks)
        )
        resp = await self.llm.complete(
            Role.PM, SYSTEM_PROMPTS["pm"], user, max_tokens=3_000
        )
        try:
            plan = PortfolioPlan.model_validate(resp.parse_json())
        except (ValidationError, ValueError) as exc:
            log.warning("pm.parse_failed", error=str(exc))
            return PortfolioPlan(portfolio_notes=f"parse failure: {exc}")

        # Defense in depth: strip vetoed names even if the PM ignored the rule.
        kept = [h for h in plan.holdings if h.symbol not in vetoed]
        if len(kept) != len(plan.holdings):
            log.warning(
                "pm.veto_violation",
                attempted=[h.symbol for h in plan.holdings if h.symbol in vetoed],
            )
        return PortfolioPlan(
            holdings=kept, excluded=plan.excluded, portfolio_notes=plan.portfolio_notes
        )

    # -------------------------------------------------------------- vision
    async def read_charts(
        self, symbol: str, image_data_urls: list[str]
    ) -> ChartRead | None:
        """Read rendered charts. Caller must gate this on regime."""
        if not image_data_urls:
            return None
        resp = await self.llm.complete(
            Role.VISION,
            SYSTEM_PROMPTS["vision"],
            f"Read these charts for {symbol} (daily and weekly).",
            images=image_data_urls,
            max_tokens=800,
        )
        try:
            return ChartRead.model_validate(resp.parse_json())
        except (ValidationError, ValueError) as exc:
            log.warning("vision.parse_failed", symbol=symbol, error=str(exc))
            return None

    # ---------------------------------------------------------- postmortem
    async def postmortem(self, trade_summaries: list[dict]) -> PostmortemReport:
        """Nightly review. Separates process quality from outcome."""
        if not trade_summaries:
            return PostmortemReport()
        user = "Review these closed trades:\n" + "\n".join(
            str(t) for t in trade_summaries[:60]
        )
        resp = await self.llm.complete(
            Role.POSTMORTEM, SYSTEM_PROMPTS["postmortem"], user, max_tokens=2_500
        )
        try:
            return PostmortemReport.model_validate(resp.parse_json())
        except (ValidationError, ValueError) as exc:
            log.warning("postmortem.parse_failed", error=str(exc))
            return PostmortemReport()


def _parse_list(raw, model, label: str) -> list:
    """Validate each item independently; drop the bad ones loudly.

    One malformed entry must not discard an entire batch of good work.
    """
    out = []
    if not isinstance(raw, list):
        log.warning(f"{label}.unexpected_shape", got=type(raw).__name__)
        return out
    for item in raw:
        try:
            out.append(model.model_validate(item))
        except (ValidationError, ValueError) as exc:
            log.warning(f"{label}.item_rejected", item=str(item)[:120], error=str(exc))
    return out
