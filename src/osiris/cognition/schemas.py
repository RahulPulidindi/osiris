"""Strict schemas for LLM output.

Parse, do not validate: malformed model output is REJECTED rather than repaired.
Silently coercing a bad response is how a nonsense score becomes a real position.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalystNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    facts: list[str] = Field(default_factory=list)
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    evidence_quality: str = "thin"
    manipulation_signals: list[str] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)

    @property
    def was_targeted(self) -> bool:
        """True if the analyst detected an attempt to instruct it."""
        return len(self.manipulation_signals) > 0


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    interest: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class StrategistScore(BaseModel):
    """A cross-sectional score. Requires falsifiability above a threshold."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    score: float = Field(ge=-5.0, le=5.0)
    conviction: float = Field(ge=0.0, le=1.0, default=0.5)
    thesis: str = ""
    invalidation: str = ""
    horizon_days: int = Field(default=21, ge=1, le=365)
    citations: list[int] = Field(default_factory=list)

    @field_validator("invalidation")
    @classmethod
    def _must_be_specific(cls, v: str) -> str:
        """Reject vague conditions. Unfalsifiable is worse than absent."""
        vague = {
            "sentiment worsens",
            "things change",
            "market turns",
            "it goes down",
            "thesis breaks",
            "conditions deteriorate",
            "n/a",
            "none",
            "tbd",
        }
        if v.strip().lower() in vague:
            raise ValueError(f"invalidation condition is not falsifiable: {v!r}")
        return v.strip()

    @property
    def is_actionable(self) -> bool:
        """A meaningful score requires a real invalidation condition."""
        return abs(self.score) > 1.0 and bool(self.invalidation.strip())


class RedTeamReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    verdict: str = "pass"
    bear_case: str = ""
    unfounded_assumptions: list[str] = Field(default_factory=list)
    confidence_in_veto: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("verdict")
    @classmethod
    def _known_verdict(cls, v: str) -> str:
        allowed = {"pass", "reduce", "veto"}
        cleaned = v.strip().lower()
        if cleaned not in allowed:
            raise ValueError(f"verdict must be one of {allowed}, got {v!r}")
        return cleaned

    @property
    def is_veto(self) -> bool:
        return self.verdict == "veto"


class ChartRead(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    trend: str = "sideways"
    trend_persistence: str = "choppy"
    structure: str = ""
    raw_score: float = Field(default=0.0, ge=-5.0, le=5.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def is_trustworthy(self) -> bool:
        """Chart reads are only reliable in persistent trends."""
        return self.trend_persistence == "persistent" and self.confidence >= 0.5


class TargetHolding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    target_weight: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class PortfolioPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    holdings: list[TargetHolding] = Field(default_factory=list)
    excluded: list[dict] = Field(default_factory=list)
    portfolio_notes: str = ""

    @property
    def total_weight(self) -> float:
        return sum(h.target_weight for h in self.holdings)

    def normalized(
        self, *, target_invested: float = 1.0, scale_up: bool = False
    ) -> list[TargetHolding]:
        """Scale weights so the book targets at most `target_invested` of equity.

        Scaling DOWN is always safe: an over-allocated plan cannot be executed.

        Scaling UP is opt-in and off by default, because a plan that holds cash may
        be doing so on purpose. The first live plan summed to 0.82 and said why in
        its notes -- thin evidence, red-team reduction flags. Forcing it to 0.97
        would have overridden a deliberate conviction judgement and discarded
        exactly the reasoning the funnel exists to produce.

        The caller decides which case applies, and logs when a plan holds cash so
        the choice is visible rather than silent.
        """
        total = self.total_weight
        if total <= 0:
            return self.holdings
        if total < target_invested and not scale_up:
            return self.holdings
        # Leave a plan alone when it is already close to the target: rescaling by
        # a couple of percent adds churn for no benefit.
        if abs(total - target_invested) < 0.02:
            return self.holdings

        scale = target_invested / total
        return [
            TargetHolding(
                symbol=h.symbol,
                target_weight=h.target_weight * scale,
                rationale=h.rationale,
            )
            for h in self.holdings
        ]


class Lesson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lesson: str
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str = "selection"


class PostmortemReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lessons: list[Lesson] = Field(default_factory=list)
    process_errors: list[str] = Field(default_factory=list)
    luck_acknowledgments: list[str] = Field(default_factory=list)
