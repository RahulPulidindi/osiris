"""Ranking, funnel, journal, attribution, and evaluation routes.

The journal route exists to make decisions auditable. Showing **vetoes alongside
fills** is the point: a kernel silently blocking everything looks identical to a
quiet market from the outside, so a journal that only records fills cannot
distinguish "no opportunities today" from "the system is broken."
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from osiris.api.activity import build_activity
from osiris.api.schemas import (
    ActivityOut,
    AttributionOut,
    EvaluationOut,
    FactorExposureOut,
    FunnelStageOut,
    JournalEntryOut,
    RankingRow,
)
from osiris.api.state import RuntimeState
from osiris.eval.attribution import compute_factor_exposure, slippage_report
from osiris.execution.journal import EventType

router = APIRouter(prefix="/api", tags=["research"])

FUNNEL_STAGE_NAMES = {
    0: ("Universe", "PIT membership + liquidity floor. Free, no LLM."),
    1: ("Pre-rank", "Local factors from OHLCV. Free, no LLM."),
    2: ("Triage", "Cheap model + fast search over ~150 names."),
    3: ("Deep research", "Full contents, charts, and the four roles."),
    4: ("Construction", "PM assembles the final book."),
}


def _get_state() -> RuntimeState:
    from osiris.api.app import get_state

    return get_state()


@router.get("/ranking", response_model=list[RankingRow])
async def ranking(
    limit: int = Query(1000, ge=1, le=5000),
    state: RuntimeState = Depends(_get_state),
) -> list[RankingRow]:
    """The ranked universe. Virtualized client-side, so the full list is returned."""
    pf = state.ledger.to_portfolio(state.prices)
    equity = pf.equity
    held = {p.symbol: p.market_value / equity if equity > 0 else 0.0 for p in pf.positions}

    rows: list[RankingRow] = []
    for i, raw in enumerate(state.ranking[:limit]):
        symbol = raw.get("symbol", "")
        closes = state.closes.get(symbol)
        sparkline = [float(c) for c in closes[-30:]] if closes is not None and len(closes) else []
        change = 0.0
        if closes is not None and len(closes) >= 2 and closes[-2]:
            change = float(closes[-1] / closes[-2] - 1.0)
        rows.append(
            RankingRow(
                symbol=symbol,
                rank=raw.get("rank", i + 1),
                score=raw.get("score", 0.0),
                conviction=raw.get("conviction", 0.0),
                stage=raw.get("stage", 0),
                sector=state.sectors.get(symbol, "Unknown"),
                last_price=state.prices.get(symbol, 0.0),
                change_pct=change,
                target_weight=raw.get("target_weight", 0.0),
                held_weight=held.get(symbol, 0.0),
                thesis=raw.get("thesis", ""),
                invalidation=raw.get("invalidation", ""),
                red_team_verdict=raw.get("red_team_verdict", ""),
                sparkline=sparkline,
                sources=raw.get("sources", []),
            )
        )
    return rows


@router.get("/funnel", response_model=list[FunnelStageOut])
async def funnel(state: RuntimeState = Depends(_get_state)) -> list[FunnelStageOut]:
    """Per-stage funnel widths and spend. Surfaces where breadth is being lost."""
    out: list[FunnelStageOut] = []
    for stage in state.funnel_stages:
        idx = stage.get("stage", 0)
        name, description = FUNNEL_STAGE_NAMES.get(idx, (f"Stage {idx}", ""))
        out.append(
            FunnelStageOut(
                stage=idx,
                name=name,
                count=stage.get("count", 0),
                cost_usd=stage.get("cost_usd", 0.0),
                description=description,
            )
        )
    return out


@router.get("/journal", response_model=list[JournalEntryOut])
async def journal(
    limit: int = Query(300, ge=1, le=5000),
    event: str | None = Query(None, description="Filter by event type"),
    state: RuntimeState = Depends(_get_state),
) -> list[JournalEntryOut]:
    """Append-only journal, newest last. Includes vetoes, not only fills."""
    kind: EventType | None = None
    if event:
        try:
            kind = EventType(event)
        except ValueError:
            kind = None

    records = state.journal.read(event=kind, limit=limit)
    return [
        JournalEntryOut(
            seq=r.seq,
            ts=r.ts,
            event=r.event.value,
            correlation_id=r.correlation_id,
            payload=r.payload,
        )
        for r in records
    ]


@router.get("/activity", response_model=list[ActivityOut])
async def activity(
    limit: int = Query(200, ge=1, le=2000),
    kind: str | None = Query(None, description="bought|sold|blocked|halted|note"),
    state: RuntimeState = Depends(_get_state),
) -> list[ActivityOut]:
    """What the agent did, newest first, each row carrying its own reason.

    A projection of the journal, not a second source of truth. `/journal` remains
    the raw audit trail; this is the same information shaped for a human.
    """
    rows = build_activity(state.journal, limit=limit if kind is None else 2000)
    if kind:
        wanted = {k.strip() for k in kind.split(",") if k.strip()}
        rows = [r for r in rows if r.kind in wanted][:limit]
    return rows


@router.get("/journal/veto-summary")
async def veto_summary(state: RuntimeState = Depends(_get_state)) -> dict[str, int]:
    """Veto codes by frequency. First place to look when nothing is trading."""
    return state.journal.veto_summary()


@router.get("/factor-exposure", response_model=FactorExposureOut)
async def factor_exposure(state: RuntimeState = Depends(_get_state)) -> FactorExposureOut:
    """Live regression on the benchmark. Separates selection from beta."""
    returns = state.daily_returns
    bench = state.benchmark_returns
    n = min(len(returns), len(bench))

    if n < 3:
        return FactorExposureOut(
            alpha_annualized=0.0,
            alpha_t_stat=0.0,
            market_beta=0.0,
            loadings={},
            r_squared=0.0,
            n_periods=n,
            verdict="insufficient history to distinguish alpha from noise",
            is_significant=False,
        )

    exposure = compute_factor_exposure(returns[-n:], {"market": bench[-n:]})
    return FactorExposureOut(
        alpha_annualized=exposure.alpha_annualized,
        alpha_t_stat=exposure.alpha_t_stat,
        market_beta=exposure.market_beta,
        loadings=exposure.loadings,
        r_squared=exposure.r_squared,
        n_periods=exposure.n_periods,
        verdict=exposure.verdict,
        is_significant=exposure.is_alpha_significant,
    )


@router.get("/attribution", response_model=AttributionOut)
async def attribution(state: RuntimeState = Depends(_get_state)) -> AttributionOut:
    """Brinson decomposition plus realized-vs-modeled slippage."""
    from osiris.eval.attribution import brinson_attribution

    pf = state.ledger.to_portfolio(state.prices)
    portfolio_weights = pf.sector_weights()
    benchmark_weights = state.benchmark_sector_weights

    # Per-sector realized returns derived from the position book.
    sector_returns: dict[str, float] = {}
    for pos in pf.positions:
        lp = state.ledger.positions.get(pos.symbol)
        if not lp or lp.cost_basis_total <= 0:
            continue
        r = (pos.market_value - pos.cost_basis) / pos.cost_basis
        sector_returns.setdefault(pos.sector, 0.0)
        sector_returns[pos.sector] += r * (
            pos.market_value / max(1e-9, pf.equity * portfolio_weights.get(pos.sector, 1.0))
        )

    bench_sector_returns = dict.fromkeys(
        set(portfolio_weights) | set(benchmark_weights),
        state.benchmark_returns[-1] if state.benchmark_returns else 0.0,
    )
    result = brinson_attribution(
        portfolio_weights, sector_returns, benchmark_weights, bench_sector_returns
    )

    modeled = 2.0  # half-spread assumption from the backtest cost model
    slip = slippage_report(state.realized_slippage_bps, modeled)
    return AttributionOut(
        selection=result.selection,
        allocation=result.allocation,
        interaction=result.interaction,
        total_excess=result.total_excess,
        selection_share=result.selection_share,
        verdict=result.verdict,
        by_sector=result.by_sector,
        realized_slippage_bps=slip.realized_bps,
        modeled_slippage_bps=slip.modeled_bps,
        slippage_excess_bps=slip.excess_bps,
        slippage_degrading=slip.is_degrading,
    )


@router.get("/evaluation", response_model=EvaluationOut)
async def evaluation(state: RuntimeState = Depends(_get_state)) -> EvaluationOut:
    """The four gates plus performance stats.

    Returns the cached result written by the evaluation runner rather than
    recomputing: a 10,000-trial Monte Carlo is not an HTTP-request-time operation.
    """
    if state.evaluation:
        return EvaluationOut.model_validate(state.evaluation)

    from osiris.eval.metrics import deflated_sharpe, summarize

    returns = state.daily_returns
    if not returns:
        return EvaluationOut(verdict="no return history yet")

    perf = summarize(returns)
    return EvaluationOut(
        sharpe=perf.sharpe,
        deflated_sharpe=deflated_sharpe(perf.sharpe, n_trials=1, n_periods=perf.n_periods),
        sortino=perf.sortino if perf.sortino != float("inf") else 0.0,
        max_drawdown=perf.max_drawdown,
        total_return=perf.total_return,
        cagr=perf.cagr,
        win_rate=perf.win_rate,
        verdict="gates not yet run; run the evaluation harness",
    )
