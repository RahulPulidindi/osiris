"""API response models. Strict at the boundary, like every other edge.

Numbers are returned as numbers, never pre-formatted strings. Formatting is a
presentation concern, and a backend that returns "1,234.56" makes the value
unusable for charting and sorting.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    mode: str
    armed: bool
    account_type: str
    broker: str
    kill_switch_engaged: bool
    breakers_tripped: list[str] = Field(default_factory=list)
    subscribers: int = 0
    version: str = "0.1.0"


class ActivityOut(BaseModel):
    """One thing the agent DID, in plain language.

    A projection of the journal rather than a new record. The journal stores
    low-level events (`order_placed`, `kernel_veto`) which are the right shape for
    an audit trail and the wrong shape for a human asking "what happened?". This
    collapses them into actions carrying their own reason, so the answer to
    "why did it buy that?" is on the same row as the buy.
    """

    seq: int
    ts: datetime
    kind: str          # bought | sold | blocked | halted | reconciled | note
    symbol: str = ""
    headline: str      # "Bought AAPL — $2,000"
    reason: str = ""   # why the agent chose this
    detail: str = ""   # thesis, veto explanation, or exit trigger
    notional_usd: float | None = None
    quantity: float | None = None
    price: float | None = None
    correlation_id: str = ""


class PreflightCheckOut(BaseModel):
    name: str
    passed: bool
    severity: str
    detail: str


class PreflightOut(BaseModel):
    """Go-live readiness. `armed` here means *cleared to arm*, not armed."""

    armed: bool
    evaluated_at: datetime
    checks: list[PreflightCheckOut] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)


class PositionOut(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    last_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    sector: str
    beta: float
    thesis: str = ""
    invalidation: str = ""
    opened_at: datetime | None = None


class PortfolioOut(BaseModel):
    equity: float
    cash: float
    buying_power: float
    gross_exposure: float
    net_exposure_pct: float
    position_count: int
    portfolio_beta: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    drawdown_pct: float
    peak_equity: float
    positions: list[PositionOut] = Field(default_factory=list)
    sector_weights: dict[str, float] = Field(default_factory=dict)
    as_of: datetime


class RankingRow(BaseModel):
    """One row of the ranking table. `stage` drives the funnel-depth column."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    rank: int | None = None
    score: float = 0.0
    conviction: float = 0.0
    stage: int = 0
    sector: str = "Unknown"
    last_price: float = 0.0
    change_pct: float = 0.0
    target_weight: float = 0.0
    held_weight: float = 0.0
    thesis: str = ""
    invalidation: str = ""
    red_team_verdict: str = ""
    sparkline: list[float] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class EquityPoint(BaseModel):
    date: str
    equity: float
    benchmark: float | None = None
    drawdown: float = 0.0


class BreakerOut(BaseModel):
    name: str
    tripped: bool
    value: float
    threshold: float
    detail: str = ""

    @property
    def headroom(self) -> float:
        return self.threshold - self.value


class FactorExposureOut(BaseModel):
    """Answers: am I running a strategy, or long the market with extra steps?"""

    alpha_annualized: float
    alpha_t_stat: float
    market_beta: float
    loadings: dict[str, float] = Field(default_factory=dict)
    r_squared: float
    n_periods: int
    verdict: str
    is_significant: bool


class SectorDeviationOut(BaseModel):
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    deviation: float
    within_band: bool


class AttributionOut(BaseModel):
    selection: float
    allocation: float
    interaction: float
    total_excess: float
    selection_share: float
    verdict: str
    by_sector: dict[str, dict[str, float]] = Field(default_factory=dict)
    realized_slippage_bps: float = 0.0
    modeled_slippage_bps: float = 0.0
    slippage_excess_bps: float = 0.0
    slippage_degrading: bool = False


class GateOut(BaseModel):
    name: str
    passed: bool
    statistic: float
    detail: str


class EvaluationOut(BaseModel):
    gates: list[GateOut] = Field(default_factory=list)
    all_passed: bool = False
    verdict: str = ""
    sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    after_tax_return: float = 0.0
    cagr: float = 0.0
    win_rate: float = 0.0
    monte_carlo_percentile: float = 0.0
    monte_carlo_p_value: float = 1.0
    null_distribution: list[float] = Field(default_factory=list)
    observed_return: float = 0.0
    walk_forward: list[dict] = Field(default_factory=list)
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    funnel_fidelity: float = 0.0
    cost_sensitivity: dict[str, float] = Field(default_factory=dict)


class JournalEntryOut(BaseModel):
    seq: int
    ts: datetime
    event: str
    correlation_id: str = ""
    payload: dict = Field(default_factory=dict)


class FunnelStageOut(BaseModel):
    stage: int
    name: str
    count: int
    cost_usd: float = 0.0
    description: str = ""


class CycleOut(BaseModel):
    correlation_id: str
    as_of: str
    ran: bool
    reason: str = ""
    regime: str = ""
    regime_detail: str = ""
    equity: float = 0.0
    halted: bool = False
    reconciled_clean: bool = True
    fills: int = 0
    vetoed: int = 0
    funnel: list[FunnelStageOut] = Field(default_factory=list)
    summary: str = ""


class KillSwitchRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class KillSwitchReleaseRequest(BaseModel):
    """Release requires a named acknowledgement. Automation defeats the purpose."""

    acknowledged_by: str = Field(min_length=1, max_length=120)
