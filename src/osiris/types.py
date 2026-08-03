"""Core domain types.

These are the contracts between planes. The cognition plane emits OrderIntent;
it never touches an order. The kernel returns KernelDecision; it is the only
thing permitted to say yes.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderKind(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class Regime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    HIGH_VOL = "high_vol"


class Trust(str, Enum):
    """Provenance. Anything not INTERNAL is treated as adversarial data."""

    INTERNAL = "internal"           # our own computation
    BROKER = "broker"               # Robinhood MCP
    VENDOR = "vendor"               # paid API, structured
    UNTRUSTED_EXTERNAL = "untrusted_external"  # web text; injection surface


class Provenanced(BaseModel):
    """Wrapper for every piece of external data entering cognition.

    Content is carried verbatim in a data field and must never be interpolated
    into an instruction position in a prompt.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    trust: Trust
    fetched_at: datetime
    content: str
    url: str | None = None
    published_at: datetime | None = None


class Bar(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("high")
    @classmethod
    def _high_is_high(cls, v: float, info: Any) -> float:
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError(f"high {v} < low {low}")
        return v


class Quote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last: float = Field(gt=0)
    ts: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return ((self.ask - self.bid) / mid) * 10_000.0 if mid > 0 else float("inf")

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        ts = self.ts if self.ts.tzinfo else self.ts.replace(tzinfo=UTC)
        return (now - ts).total_seconds()


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: float
    cost_basis: float = Field(ge=0)
    market_value: float = Field(ge=0)
    sector: str = "Unknown"
    beta: float = 1.0


class Portfolio(BaseModel):
    model_config = ConfigDict(frozen=True)

    equity: float = Field(ge=0)
    cash: float
    buying_power: float
    positions: tuple[Position, ...] = ()
    as_of: datetime

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)

    @property
    def position_count(self) -> int:
        return sum(1 for p in self.positions if abs(p.quantity) > 1e-9)

    def weight_of(self, symbol: str) -> float:
        if self.equity <= 0:
            return 0.0
        mv = sum(p.market_value for p in self.positions if p.symbol == symbol)
        return mv / self.equity

    def sector_weights(self) -> dict[str, float]:
        if self.equity <= 0:
            return {}
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.sector] = out.get(p.sector, 0.0) + p.market_value / self.equity
        return out

    def portfolio_beta(self) -> float:
        if self.equity <= 0:
            return 0.0
        return sum(p.beta * (p.market_value / self.equity) for p in self.positions)


class Score(BaseModel):
    """A ranked name with an auditable reason. Produced by cognition."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    score: float = Field(ge=-5.0, le=5.0)
    conviction: float = Field(ge=0.0, le=1.0, default=0.5)
    thesis: str = ""
    invalidation: str = ""
    stage: int = 0
    sources: tuple[str, ...] = ()
    chart_read: str | None = None
    chart_calibrated_score: float | None = None


class OrderIntent(BaseModel):
    """What cognition is permitted to emit. Not an order.

    An intent without a falsifiable invalidation condition is auto-vetoed: a
    position with no defined exit is how one name eats a portfolio.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    notional_usd: float = Field(gt=0)
    kind: OrderKind = OrderKind.LIMIT
    limit_price: float | None = Field(default=None, gt=0)
    thesis: str = ""
    invalidation: str = ""
    reason: str = ""  # rank_entry | rank_exit | risk_exit | invalidation_exit
    correlation_id: str = ""

    @property
    def idempotency_key(self) -> str:
        """Stable per (symbol, side, notional, day, reason).

        Prevents the same logical order being placed twice across a retry or a
        duplicated cycle.
        """
        # Market date, not UTC date. After 8pm ET the UTC day has already rolled
        # over, so a UTC-based key would change mid-session and the same logical
        # order would present two different keys -- silently disabling the
        # duplicate protection this property exists to provide.
        from osiris.data.macro import session_date

        day = session_date().isoformat()
        raw = f"{day}|{self.symbol}|{self.side.value}|{self.notional_usd:.2f}|{self.reason}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @property
    def is_exit(self) -> bool:
        return self.side is Side.SELL


class VetoCode(str, Enum):
    KILL_SWITCH = "kill_switch"
    BREAKER_TRIPPED = "breaker_tripped"
    MACRO_BLACKOUT = "macro_blackout"
    EARNINGS_BLACKOUT = "earnings_blackout"
    MISSING_INVALIDATION = "missing_invalidation"
    NOTIONAL_CAP = "notional_cap"
    SYMBOL_WEIGHT_CAP = "symbol_weight_cap"
    SECTOR_WEIGHT_CAP = "sector_weight_cap"
    SECTOR_DEVIATION = "sector_deviation"
    BETA_BUDGET = "beta_budget"
    POSITION_FLOOR = "position_floor"
    ADV_PARTICIPATION = "adv_participation"
    SPREAD_TOO_WIDE = "spread_too_wide"
    NOT_TRADABLE = "not_tradable"
    STALE_DATA = "stale_data"
    ORDER_BUDGET = "order_budget"
    DUPLICATE_ORDER = "duplicate_order"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    UNSETTLED_FUNDS = "unsettled_funds"
    REVIEW_NOT_RUN = "review_not_run"
    REVIEW_REJECTED = "review_rejected"


class KernelDecision(BaseModel):
    """The kernel's verdict. Only `approved` intents may reach the broker."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    intent: OrderIntent
    vetoes: tuple[VetoCode, ...] = ()
    notes: tuple[str, ...] = ()
    adjusted_notional_usd: float | None = None

    @property
    def effective_notional(self) -> float:
        return (
            self.adjusted_notional_usd
            if self.adjusted_notional_usd is not None
            else self.intent.notional_usd
        )


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime
    order_id: str
    idempotency_key: str = ""
    intended_price: float | None = None

    @property
    def slippage_bps(self) -> float | None:
        if self.intended_price is None or self.intended_price <= 0:
            return None
        signed = (
            (self.price - self.intended_price)
            if self.side is Side.BUY
            else (self.intended_price - self.price)
        )
        return (signed / self.intended_price) * 10_000.0
