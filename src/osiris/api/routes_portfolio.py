"""Portfolio, exposure, and breaker-status routes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from osiris.api.schemas import (
    BreakerOut,
    EquityPoint,
    PortfolioOut,
    PositionOut,
    SectorDeviationOut,
)
from osiris.api.state import RuntimeState
from osiris.eval.attribution import sector_deviations

router = APIRouter(prefix="/api", tags=["portfolio"])


def _get_state() -> RuntimeState:
    from osiris.api.app import get_state

    return get_state()


@router.get("/portfolio", response_model=PortfolioOut)
async def portfolio(state: RuntimeState = Depends(_get_state)) -> PortfolioOut:
    prices = state.prices
    pf = state.ledger.to_portfolio(prices)
    equity = pf.equity

    positions: list[PositionOut] = []
    for pos in pf.positions:
        lp = state.ledger.positions.get(pos.symbol)
        last = prices.get(pos.symbol, lp.avg_cost if lp else 0.0)
        avg_cost = lp.avg_cost if lp else 0.0
        unrealized = pos.market_value - pos.cost_basis
        positions.append(
            PositionOut(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_cost=avg_cost,
                last_price=last,
                market_value=pos.market_value,
                weight=pos.market_value / equity if equity > 0 else 0.0,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=(unrealized / pos.cost_basis) if pos.cost_basis > 0 else 0.0,
                realized_pnl=lp.realized_pnl if lp else 0.0,
                sector=pos.sector,
                beta=pos.beta,
                thesis=state.theses.get(pos.symbol, ""),
                invalidation=state.invalidations.get(pos.symbol, ""),
                opened_at=lp.opened_at if lp else None,
            )
        )

    day_start = state.pnl.day_start_equity
    peak = state.pnl.peak_equity
    return PortfolioOut(
        equity=equity,
        cash=pf.cash,
        buying_power=pf.buying_power,
        gross_exposure=pf.gross_exposure,
        net_exposure_pct=pf.gross_exposure / equity if equity > 0 else 0.0,
        position_count=pf.position_count,
        portfolio_beta=pf.portfolio_beta(),
        realized_pnl=state.ledger.realized_pnl,
        unrealized_pnl=state.ledger.unrealized_pnl(prices),
        daily_pnl=equity - day_start if day_start > 0 else 0.0,
        daily_pnl_pct=(equity - day_start) / day_start if day_start > 0 else 0.0,
        drawdown_pct=max(0.0, (peak - equity) / peak) if peak > 0 else 0.0,
        peak_equity=peak,
        positions=sorted(positions, key=lambda p: -p.market_value),
        sector_weights=pf.sector_weights(),
        as_of=pf.as_of or datetime.now(UTC),
    )


@router.get("/equity-curve", response_model=list[EquityPoint])
async def equity_curve(state: RuntimeState = Depends(_get_state)) -> list[EquityPoint]:
    """Equity vs benchmark, with the drawdown series for the baseline chart."""
    drawdowns = state.drawdown_series()
    return [
        EquityPoint(
            date=point["date"],
            equity=point["equity"],
            benchmark=point.get("benchmark"),
            drawdown=drawdowns[i] if i < len(drawdowns) else 0.0,
        )
        for i, point in enumerate(state.equity_history)
    ]


@router.get("/breakers", response_model=list[BreakerOut])
async def breakers(state: RuntimeState = Depends(_get_state)) -> list[BreakerOut]:
    """Live breaker status with headroom, not just a tripped flag.

    Headroom is the useful number: "3% from a halt" is actionable, "not tripped"
    is not.
    """
    limits = state.limits
    prices = state.prices
    equity = state.ledger.equity(prices)
    day_start = state.pnl.day_start_equity
    peak = state.pnl.peak_equity

    daily_loss = -((equity - day_start) / day_start) if day_start > 0 else 0.0
    drawdown = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
    tripped = {c.value for c in state.breakers.tripped}

    return [
        BreakerOut(
            name="daily_loss",
            tripped=daily_loss >= limits.daily_loss_halt_pct,
            value=max(0.0, daily_loss),
            threshold=limits.daily_loss_halt_pct,
            detail=f"day P&L {(equity - day_start):+,.2f}",
        ),
        BreakerOut(
            name="max_drawdown",
            tripped=drawdown >= limits.max_drawdown_halt_pct,
            value=drawdown,
            threshold=limits.max_drawdown_halt_pct,
            detail=f"peak {peak:,.2f}",
        ),
        BreakerOut(
            name="consecutive_losses",
            tripped=state.pnl.consecutive_losses >= limits.consecutive_loss_halt,
            value=float(state.pnl.consecutive_losses),
            threshold=float(limits.consecutive_loss_halt),
            detail=f"{state.pnl.consecutive_losses} in a row",
        ),
        BreakerOut(
            name="portfolio_beta",
            tripped=False,
            value=state.ledger.to_portfolio(prices).portfolio_beta(),
            threshold=limits.max_portfolio_beta,
            detail="beta budget prevents leverage in disguise",
        ),
        BreakerOut(
            name="ledger_divergence",
            tripped="breaker_tripped" in tripped,
            value=1.0 if tripped else 0.0,
            threshold=1.0,
            detail="; ".join(state.breakers.reasons) or "reconciled",
        ),
    ]


@router.get("/sector-deviation", response_model=list[SectorDeviationOut])
async def sector_deviation(
    state: RuntimeState = Depends(_get_state),
) -> list[SectorDeviationOut]:
    """Active sector bets against the benchmark, with band compliance."""
    pf = state.ledger.to_portfolio(state.prices)
    deviations = sector_deviations(pf.sector_weights(), state.benchmark_sector_weights)
    band = state.limits.max_sector_deviation
    return [
        SectorDeviationOut(
            sector=d.sector,
            portfolio_weight=d.portfolio_weight,
            benchmark_weight=d.benchmark_weight,
            deviation=d.deviation,
            within_band=abs(d.deviation) <= band,
        )
        for d in deviations
    ]
