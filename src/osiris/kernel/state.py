"""Kernel state: the facts the kernel needs to decide, and breaker status.

Deliberately a plain frozen dataclass rather than a live service. The kernel is
a pure function of (intent, state) so that every decision is reproducible and
testable, and so no I/O can hang the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

from osiris.types import Portfolio, Quote, VetoCode


@dataclass(frozen=True)
class BreakerState:
    """Circuit breakers. Tripped breakers halt NEW risk, never exits."""

    tripped: tuple[VetoCode, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def is_tripped(self) -> bool:
        return len(self.tripped) > 0

    def trip(self, code: VetoCode, reason: str) -> BreakerState:
        if code in self.tripped:
            return self
        return BreakerState(
            tripped=(*self.tripped, code), reasons=(*self.reasons, reason)
        )

    def reset(self) -> BreakerState:
        """Human-initiated only. A fuse that resets itself is not a fuse."""
        return BreakerState()


@dataclass(frozen=True)
class KernelState:
    """Everything the kernel needs. No network calls at decision time."""

    portfolio: Portfolio
    quotes: dict[str, Quote] = field(default_factory=dict)

    # Risk inputs
    adv: dict[str, float] = field(default_factory=dict)          # avg daily $ volume
    sectors: dict[str, str] = field(default_factory=dict)
    betas: dict[str, float] = field(default_factory=dict)
    benchmark_sector_weights: dict[str, float] = field(default_factory=dict)
    tradable: dict[str, bool] = field(default_factory=dict)
    next_earnings: dict[str, datetime] = field(default_factory=dict)

    # Session / accounting
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    breakers: BreakerState = field(default_factory=BreakerState)
    kill_switch_engaged: bool = False
    macro_blackout: bool = False
    macro_blackout_reason: str = ""

    orders_placed_today: int = 0
    used_idempotency_keys: frozenset[str] = frozenset()
    unsettled_cash: float = 0.0

    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0

    # review_* must have run clean for this key before place_* is allowed
    reviewed_keys: frozenset[str] = frozenset()

    def with_order_placed(self, idempotency_key: str) -> KernelState:
        return replace(
            self,
            orders_placed_today=self.orders_placed_today + 1,
            used_idempotency_keys=self.used_idempotency_keys | {idempotency_key},
        )

    def with_review_passed(self, idempotency_key: str) -> KernelState:
        return replace(self, reviewed_keys=self.reviewed_keys | {idempotency_key})

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.portfolio.equity - self.day_start_equity) / self.day_start_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.portfolio.equity) / self.peak_equity)

    def sector_of(self, symbol: str) -> str:
        return self.sectors.get(symbol, "Unknown")

    def beta_of(self, symbol: str) -> float:
        return self.betas.get(symbol, 1.0)

    def is_in_earnings_blackout(self, symbol: str, hours: int) -> bool:
        report = self.next_earnings.get(symbol)
        if report is None or hours <= 0:
            return False
        report = report if report.tzinfo else report.replace(tzinfo=UTC)
        delta_hours = (report - self.now).total_seconds() / 3600.0
        return 0 <= delta_hours <= hours


def evaluate_breakers(
    state: KernelState,
    *,
    daily_loss_halt_pct: float,
    max_drawdown_halt_pct: float,
    consecutive_loss_halt: int,
    ledger_divergence: bool = False,
    schema_drift: bool = False,
) -> BreakerState:
    """Pure breaker evaluation. Returns the state that SHOULD hold."""
    b = state.breakers

    if state.daily_pnl_pct <= -abs(daily_loss_halt_pct):
        b = b.trip(
            VetoCode.BREAKER_TRIPPED,
            f"daily loss {state.daily_pnl_pct:.2%} breached {daily_loss_halt_pct:.2%}",
        )
    if state.drawdown_pct >= abs(max_drawdown_halt_pct):
        b = b.trip(
            VetoCode.BREAKER_TRIPPED,
            f"drawdown {state.drawdown_pct:.2%} breached {max_drawdown_halt_pct:.2%}",
        )
    if state.consecutive_losses >= consecutive_loss_halt:
        b = b.trip(
            VetoCode.BREAKER_TRIPPED,
            f"{state.consecutive_losses} consecutive losses",
        )
    if ledger_divergence:
        b = b.trip(VetoCode.BREAKER_TRIPPED, "broker/ledger divergence")
    if schema_drift:
        b = b.trip(VetoCode.BREAKER_TRIPPED, "MCP schema drift")
    return b


def is_trading_day(d: date) -> bool:
    """Weekday check. Market holidays are handled by the macro calendar."""
    return d.weekday() < 5
