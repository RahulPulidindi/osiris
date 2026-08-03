"""Go-live gate: `python -m osiris.runner.gate`.

Runs the full Phase 4/5 validation in one command and exits non-zero unless
every blocking check passes. That exit code is the point -- it makes the gate
usable from CI and from a supervisor script, so "are we clear to trade?" has a
machine-checkable answer rather than a human reading a dashboard.

The paper run happens *inside* this command rather than being read from a
previous run's artifacts. Reading artifacts would let a stale, favorable journal
clear the gate, which is the same class of error as reusing a backtest that was
fit on the test set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from osiris.config import DATA_DIR, AccountType, RiskLimits, Settings, load_settings
from osiris.eval.backtest import Backtester, CostModel
from osiris.execution.broker import PaperBroker
from osiris.logging import configure_logging, get_logger
from osiris.runner.alerts import build_alerter, preflight_blocked
from osiris.runner.backtest_gate import seed
from osiris.runner.preflight import run_preflight
from osiris.runner.synthetic import SyntheticMarket

log = get_logger(__name__)


def _backtest_returns(market: SyntheticMarket, *, start: int, n: int) -> list[float]:
    """Run the event-driven backtester over the same window as the paper run.

    Uses the same momentum ranking the paper runner uses, so a divergence
    between the two is attributable to execution mechanics (costs, sizing,
    gating) rather than to a different strategy. Comparing two different
    strategies would make the divergence check meaningless.
    """
    from osiris.data.indicators import momentum, volatility_annualized

    prices = {
        s: {market.dates[i]: float(market.closes[s][i]) for i in range(market.n_days)}
        for s in market.symbols
    }
    date_to_index = {d: i for i, d in enumerate(market.dates)}

    def targets(as_of):
        i = date_to_index[as_of]
        scored = []
        for symbol in market.symbols:
            series = market.closes[symbol][: i + 1]
            if series.size < 130:
                continue
            vol = volatility_annualized(series, window=20)
            if vol <= 1e-6:
                continue
            scored.append((symbol, momentum(series, lookback=126) / vol))
        scored.sort(key=lambda t: -t[1])
        return [s for s, _ in scored[:20]]

    rebalance_dates = [
        market.dates[i]
        for i in range(start, min(start + n, market.n_days))
        if market.dates[i].weekday() < 5
    ]
    result = Backtester(
        prices, adv=dict(market.adv), cost_model=CostModel()
    ).run(rebalance_dates, targets, n_holdings=20)
    return result.returns


async def run_gate(args) -> int:
    settings = Settings(
        mode=load_settings().mode,
        account_type=(
            load_settings().account_type
            if load_settings().account_type is not AccountType.UNKNOWN
            else AccountType.MARGIN
        ),
        account_equity_usd=load_settings().account_equity_usd or 100_000.0,
    )

    journal_path = DATA_DIR / "journal-gate.jsonl"
    if journal_path.exists():
        journal_path.unlink()

    from osiris.api.state import build_runtime_state

    state = build_runtime_state(
        settings=settings,
        limits=RiskLimits(),
        journal_path=journal_path,
        broker=PaperBroker(starting_cash=settings.account_equity_usd or 100_000.0),
    )

    # --- Phase 4: paper trade, then the four evaluation gates. ---
    log.info("gate.paper_run_starting", sessions=args.sessions)
    await seed(state, sessions=args.sessions, seed_value=args.seed)

    # --- Phase 4 gate: paper must not diverge from backtest. ---
    backtest = None
    if not args.skip_backtest:
        market = SyntheticMarket(n_symbols=120, n_days=320, seed=args.seed)
        try:
            backtest = _backtest_returns(
                market, start=140, n=min(args.sessions, market.n_days - 140)
            )
        except Exception as exc:
            log.warning("gate.backtest_failed", error=str(exc))

    # --- Phase 5: arming checks. ---
    report = run_preflight(state, backtest_returns=backtest)

    print()
    print("=" * 72)
    print("OSIRIS GO-LIVE GATE")
    print("=" * 72)
    print(report.describe())
    print("=" * 72)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))

    if not report.armed and args.alert:
        build_alerter().send(
            preflight_blocked([c.name for c in report.blocking_failures])
        )

    return 0 if report.armed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Osiris go-live gate. Exits non-zero unless cleared."
    )
    parser.add_argument("--sessions", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Also emit JSON.")
    parser.add_argument("--alert", action="store_true", help="Alert if blocked.")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the paper-vs-backtest divergence comparison.",
    )
    args = parser.parse_args()

    configure_logging()
    sys.exit(asyncio.run(run_gate(args)))


if __name__ == "__main__":
    main()
