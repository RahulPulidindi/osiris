"""Shared fixtures. Builders keep tests readable and intent explicit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from osiris.config import AccountType, RiskLimits
from osiris.kernel.kernel import RiskKernel
from osiris.kernel.state import BreakerState, KernelState
from osiris.types import (
    OrderIntent,
    Portfolio,
    Position,
    Quote,
    Side,
)

NOW = datetime(2026, 7, 31, 14, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_from_operator_env(monkeypatch):
    """Tests must pass regardless of what the operator's `.env` says.

    Once the account is armed for real trading, `.env` reads `OSIRIS_MODE=live`
    -- and any test that constructs `Settings()` without explicit kwargs would
    then hit the live-mode validators and fail. Environment variables outrank
    the env file in pydantic-settings, and explicit constructor kwargs outrank
    both, so pinning these here isolates every test while leaving tests that
    deliberately build live settings untouched.
    """
    monkeypatch.setenv("OSIRIS_MODE", "paper")
    monkeypatch.setenv("OSIRIS_I_UNDERSTAND_THE_RISK", "no")
    monkeypatch.setenv("OSIRIS_ACCOUNT_EQUITY_USD", "0")

SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Healthcare",
    "PG": "Staples",
    "CAT": "Industrials",
}


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        max_trade_notional_pct=0.02,
        max_symbol_weight=0.10,
        max_sector_weight=0.25,
        max_sector_deviation=0.10,
        max_portfolio_beta=1.15,
        min_position_count=15,
        target_position_count=20,
        max_adv_participation=0.01,
        max_spread_bps=25.0,
        daily_order_budget=60,
        quote_staleness_seconds=300,
        earnings_blackout_hours=48,
        daily_loss_halt_pct=0.03,
        max_drawdown_halt_pct=0.10,
        consecutive_loss_halt=5,
    )


def make_quote(
    symbol: str, price: float = 100.0, spread_bps: float = 5.0, age_s: float = 0.0
) -> Quote:
    half = price * (spread_bps / 2) / 10_000
    return Quote(
        symbol=symbol,
        bid=price - half,
        ask=price + half,
        last=price,
        ts=NOW - timedelta(seconds=age_s),
    )


def make_position(
    symbol: str, market_value: float, sector: str | None = None, beta: float = 1.0
) -> Position:
    return Position(
        symbol=symbol,
        quantity=market_value / 100.0,
        cost_basis=market_value,
        market_value=market_value,
        sector=sector or SECTORS.get(symbol, "Unknown"),
        beta=beta,
    )


def make_portfolio(
    equity: float = 100_000.0,
    positions: tuple[Position, ...] = (),
    buying_power: float | None = None,
) -> Portfolio:
    invested = sum(p.market_value for p in positions)
    cash = max(0.0, equity - invested)
    return Portfolio(
        equity=equity,
        cash=cash,
        buying_power=buying_power if buying_power is not None else cash,
        positions=positions,
        as_of=NOW,
    )


def diversified_positions(count: int = 20, equity: float = 100_000.0) -> tuple[Position, ...]:
    """A realistic equal-weight book spread across sectors."""
    symbols = list(SECTORS)
    per = equity * 0.045  # ~4.5% each, under the 10% cap
    out = []
    for i in range(count):
        sym = f"{symbols[i % len(symbols)]}{'' if i < len(symbols) else i}"
        sector = SECTORS.get(symbols[i % len(symbols)], "Unknown")
        out.append(make_position(sym, per, sector=sector))
    return tuple(out)


def make_state(
    portfolio: Portfolio | None = None,
    *,
    symbols: tuple[str, ...] = ("AAPL",),
    spread_bps: float = 5.0,
    quote_age_s: float = 0.0,
    adv: float = 500_000_000.0,
    tradable: bool = True,
    betas: dict[str, float] | None = None,
    benchmark: dict[str, float] | None = None,
    earnings: dict[str, datetime] | None = None,
    reviewed: frozenset[str] = frozenset(),
    **kwargs,
) -> KernelState:
    portfolio = portfolio or make_portfolio()
    all_symbols = set(symbols) | {p.symbol for p in portfolio.positions}
    return KernelState(
        portfolio=portfolio,
        quotes={s: make_quote(s, spread_bps=spread_bps, age_s=quote_age_s) for s in all_symbols},
        adv={s: adv for s in all_symbols},
        sectors={s: SECTORS.get(s.rstrip("0123456789"), "Unknown") for s in all_symbols},
        betas=betas or {s: 1.0 for s in all_symbols},
        benchmark_sector_weights=benchmark or {},
        tradable={s: tradable for s in all_symbols},
        next_earnings=earnings or {},
        now=NOW,
        day_start_equity=portfolio.equity,
        peak_equity=portfolio.equity,
        reviewed_keys=reviewed,
        **kwargs,
    )


def make_intent(
    symbol: str = "AAPL",
    side: Side = Side.BUY,
    notional: float = 1_500.0,
    reason: str = "rank_entry",
    invalidation: str = "close below 200DMA",
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        notional_usd=notional,
        reason=reason,
        invalidation=invalidation if side is Side.BUY else "",
        thesis="test thesis",
    )


@pytest.fixture
def kernel(limits: RiskLimits) -> RiskKernel:
    return RiskKernel(limits, account_type=AccountType.MARGIN)


@pytest.fixture
def cash_kernel(limits: RiskLimits) -> RiskKernel:
    return RiskKernel(limits, account_type=AccountType.CASH)
