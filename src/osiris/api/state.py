"""Shared runtime state between the trading loop and the API.

Holds the live objects (ledger, journal, kill switch) plus derived caches the
dashboard reads. Deliberately a single object rather than module globals so tests
can construct an isolated instance.

The API is **read-mostly**. The only writes it permits are the kill switch and a
manual breaker reset, both of which are human-safety controls. It cannot place an
order, change a limit, or resume a halted system automatically -- a dashboard that
can arm trading is an attack surface pointed at the account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import numpy as np

from osiris.api.events import BUS, Channel, EventBus
from osiris.config import DATA_DIR, RiskLimits, Settings, load_risk_limits, load_settings
from osiris.execution.broker import Broker, PaperBroker
from osiris.execution.journal import Journal
from osiris.execution.killswitch import KillSwitch
from osiris.execution.ledger import DailyPnL, Ledger
from osiris.kernel.state import BreakerState
from osiris.logging import get_logger

log = get_logger(__name__)


@dataclass
class RuntimeState:
    """Everything the API needs to answer a question about the system."""

    settings: Settings
    limits: RiskLimits
    ledger: Ledger
    journal: Journal
    killswitch: KillSwitch
    pnl: DailyPnL
    broker: Broker
    bus: EventBus = field(default=BUS)

    # Caches populated by the loop; the API never computes these itself.
    prices: dict[str, float] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)
    betas: dict[str, float] = field(default_factory=dict)
    benchmark_sector_weights: dict[str, float] = field(default_factory=dict)
    closes: dict[str, np.ndarray] = field(default_factory=dict)
    theses: dict[str, str] = field(default_factory=dict)
    invalidations: dict[str, str] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)

    ranking: list[dict] = field(default_factory=list)
    equity_history: list[dict] = field(default_factory=list)
    benchmark_history: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    benchmark_returns: list[float] = field(default_factory=list)
    realized_slippage_bps: list[float] = field(default_factory=list)
    funnel_stages: list[dict] = field(default_factory=list)
    last_cycle: dict | None = None
    evaluation: dict | None = None
    breakers: BreakerState = field(default_factory=BreakerState)

    @property
    def armed(self) -> bool:
        return self.settings.live_armed

    def publish(self, channel: Channel, data: dict) -> None:
        self.bus.publish(channel, data)

    # ------------------------------------------------------------------- marks
    def mark_prices(self, prices: dict[str, float]) -> None:
        self.prices.update(prices)

    def record_equity(
        self,
        equity: float,
        benchmark: float | None = None,
        *,
        as_of: date | None = None,
    ) -> None:
        """Append a daily mark and derive the return series the gates consume.

        `as_of` is explicit rather than wall-clock so a backtest or paper replay
        records one point per SIMULATED session. Defaulting to "today" would
        collapse every simulated session onto a single date, silently producing an
        empty return series -- and every downstream gate would then be computed on
        no data while appearing to run.
        """
        if equity <= 0:
            # Refuse to record a non-positive mark. Real equity does not hit zero
            # on a long-only book, so this is a missing-price or skipped-session
            # artifact -- and recording it would inject a fabricated -100% return
            # that corrupts Sharpe, drawdown, and every gate downstream.
            log.warning("state.rejected_nonpositive_equity_mark", equity=equity)
            return

        today = (as_of or datetime.now(UTC).date()).isoformat()
        prev = self.equity_history[-1] if self.equity_history else None

        if prev and prev["date"] == today:
            prev["equity"] = equity
            if benchmark is not None:
                prev["benchmark"] = benchmark
        else:
            self.equity_history.append(
                {"date": today, "equity": equity, "benchmark": benchmark}
            )
            if prev and prev["equity"] > 0:
                self.daily_returns.append(equity / prev["equity"] - 1.0)
                if benchmark is not None and prev.get("benchmark"):
                    self.benchmark_returns.append(benchmark / prev["benchmark"] - 1.0)

        self.pnl.mark(equity)
        self.publish(
            Channel.PNL,
            {
                "equity": equity,
                "daily_pnl": equity - self.pnl.day_start_equity,
                "daily_pnl_pct": (
                    (equity - self.pnl.day_start_equity) / self.pnl.day_start_equity
                    if self.pnl.day_start_equity > 0
                    else 0.0
                ),
                "peak_equity": self.pnl.peak_equity,
                "drawdown_pct": (
                    max(0.0, (self.pnl.peak_equity - equity) / self.pnl.peak_equity)
                    if self.pnl.peak_equity > 0
                    else 0.0
                ),
            },
        )

    def drawdown_series(self) -> list[float]:
        if not self.equity_history:
            return []
        curve = np.array([p["equity"] for p in self.equity_history], dtype=float)
        peak = np.maximum.accumulate(curve)
        return list((curve - peak) / np.where(peak > 0, peak, 1.0))


def build_runtime_state(
    *,
    settings: Settings | None = None,
    limits: RiskLimits | None = None,
    journal_path=None,
    broker: Broker | None = None,
) -> RuntimeState:
    """Assemble runtime state. Paper broker unless a live one is supplied."""
    settings = settings or load_settings()
    limits = limits or load_risk_limits()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ZERO, not a friendly default. This previously fell back to $100,000, which
    # meant a fresh unconnected install displayed six figures of money that does
    # not exist. Every downstream percentage is a fraction of equity, so an
    # invented balance silently produces invented weights, invented exposure, and
    # invented risk headroom -- and it all looks plausible.
    #
    # An empty account reads as empty. The real balance arrives from the broker.
    starting_cash = settings.account_equity_usd
    return RuntimeState(
        settings=settings,
        limits=limits,
        ledger=Ledger(starting_cash=starting_cash),
        journal=Journal(journal_path or DATA_DIR / "journal.jsonl"),
        killswitch=KillSwitch(),
        pnl=DailyPnL(day_start_equity=starting_cash, peak_equity=starting_cash),
        broker=broker or PaperBroker(starting_cash=starting_cash),
    )
