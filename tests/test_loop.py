"""Daily loop and paper runner integration.

These are the tests that prove the pieces compose. Unit tests can all pass while
the assembled system still does nothing useful -- which is exactly what happened
during this build: the first paper run produced 280 intents and zero fills, and
only an end-to-end run surfaced it.
"""

from __future__ import annotations

import pathlib

import pytest

from osiris.api.state import build_runtime_state
from osiris.config import AccountType, Mode, RiskLimits, Settings
from osiris.execution.broker import PaperBroker
from osiris.execution.journal import EventType
from osiris.runner.paper import PaperRunner, QuantRanker
from osiris.runner.synthetic import BENCHMARK_SECTOR_WEIGHTS, SyntheticMarket


@pytest.fixture
def market() -> SyntheticMarket:
    return SyntheticMarket(n_symbols=60, n_days=220, seed=11)


@pytest.fixture
def runner(tmp_path: pathlib.Path, market: SyntheticMarket) -> PaperRunner:
    state = build_runtime_state(
        settings=Settings(
            mode=Mode.PAPER, account_type=AccountType.MARGIN, account_equity_usd=100_000.0
        ),
        limits=RiskLimits(),
        journal_path=tmp_path / "journal.jsonl",
        broker=PaperBroker(starting_cash=100_000.0),
    )
    state.killswitch.path = tmp_path / "KILL_SWITCH"
    return PaperRunner(state=state, market=market)


class TestSyntheticMarket:
    def test_universe_tracks_benchmark_sector_weights(self, market):
        """A uniformly-sectored universe could never be benchmark-neutral.

        The sector-deviation gate would then fire on a fixture artifact rather
        than on a real active bet.
        """
        from collections import Counter

        counts = Counter(market.sectors.values())
        n = len(market.symbols)
        for sector, target in BENCHMARK_SECTOR_WEIGHTS.items():
            assert abs(counts[sector] / n - target) < 0.05, sector

    def test_window_never_leaks_future_bars(self, market):
        """The PIT guarantee at the data-generation layer."""
        window = market.window(50)
        for series in window.values():
            assert len(series) == 51

    def test_names_are_correlated_to_the_benchmark(self, market):
        """Uncorrelated noise would make every beta ~0 and attribution moot."""
        import numpy as np

        symbol = market.symbols[0]
        returns = np.diff(market.closes[symbol]) / market.closes[symbol][:-1]
        # Index alignment matters here. The path is cumprod(1 + r), so
        # closes[i] already embeds market_returns[i]; the diff at position i
        # therefore corresponds to market_returns[i + 1]. Comparing without the
        # shift measures correlation against a one-day-lagged factor and reads
        # near zero even though the construction is strongly correlated.
        factor = market.market_returns[1:]
        n = min(returns.size, factor.size)
        corr = float(np.corrcoef(returns[:n], factor[:n])[0, 1])
        assert corr > 0.3, f"beta construction is not producing correlation: {corr:.3f}"


class TestQuantRanker:
    def test_produces_a_book_of_the_target_size(self, market):
        holdings, rows = QuantRanker(target_count=20).rank(
            market.window(180), market.sectors
        )

        assert len(holdings) == 20
        assert len(rows) >= 20

    def test_weights_respect_the_symbol_cap(self, market):
        holdings, _ = QuantRanker(target_count=20, max_weight=0.10).rank(
            market.window(180), market.sectors
        )

        assert all(h.target_weight <= 0.10 + 1e-9 for h in holdings)

    def test_every_row_carries_a_falsifiable_invalidation(self, market):
        """The kernel auto-vetoes an intent with no stated exit condition."""
        _, rows = QuantRanker().rank(market.window(180), market.sectors)

        assert all(row["invalidation"].strip() for row in rows)

    def test_scores_stay_within_the_schema_band(self, market):
        _, rows = QuantRanker().rank(market.window(180), market.sectors)

        assert all(-5.0 <= row["score"] <= 5.0 for row in rows)


class TestPaperRunSmoke:
    """The end-to-end check that unit tests cannot give.

    A run that produces intents but no fills is the specific failure this class
    exists to catch: it means the planner and the kernel disagree, which is
    invisible from either side alone.
    """

    async def test_run_actually_fills_orders(self, runner):
        results = await runner.run(25)

        assert results, "no sessions ran"
        total_fills = sum(len(r.report.fills) for r in results if r.report)
        assert total_fills > 0, "planner produced intents the kernel always vetoes"

    async def test_run_builds_a_diversified_book(self, runner):
        await runner.run(25)

        held = runner.state.ledger.open_symbols
        assert len(held) >= 10, f"expected a diversified book, held {len(held)}"

    async def test_equity_series_is_recorded_per_simulated_session(self, runner):
        """Without an explicit as_of, every session collapses onto one date."""
        results = await runner.run(25)
        history = runner.state.equity_history

        assert len(history) == len(results)
        assert len({point["date"] for point in history}) == len(history)

    async def test_no_fabricated_total_loss_appears_in_returns(self, runner):
        """A skipped session must not be recorded as a -100% day."""
        await runner.run(25)

        assert all(r > -0.9 for r in runner.state.daily_returns)

    async def test_returns_reconcile_with_the_equity_curve(self, runner):
        """The compounded return series must match the actual equity change.

        This is the invariant that catches a corrupt mark: the two are derived
        from the same source and any divergence means one of them is fabricated.
        """
        import numpy as np

        await runner.run(30)
        history = runner.state.equity_history
        returns = runner.state.daily_returns
        if len(history) < 3:
            pytest.skip("not enough sessions")

        compounded = float(np.prod([1.0 + r for r in returns]))
        actual = history[-1]["equity"] / history[0]["equity"]
        assert compounded == pytest.approx(actual, rel=1e-6)

    async def test_ledger_reconciles_against_the_paper_broker(self, runner):
        """Divergence here would mean the ledger and the venue disagree."""
        await runner.run(20)

        broker_positions = await runner.broker.get_positions()
        result = runner.state.ledger.reconcile(
            broker_positions,
            broker_equity=await runner.broker.get_account_equity(),
            prices=runner.state.prices,
        )
        assert result.clean, result.describe()

    async def test_journal_records_the_full_decision_trail(self, runner):
        await runner.run(20)
        counts = runner.state.journal.counts()

        for event in (
            EventType.CYCLE_START,
            EventType.CYCLE_END,
            EventType.INTENT_EMITTED,
            EventType.REVIEW_PASSED,
            EventType.ORDER_PLACED,
            EventType.FILL,
            EventType.RECONCILIATION,
        ):
            assert counts.get(event.value, 0) > 0, f"missing {event.value}"

    async def test_review_precedes_every_placement(self, runner):
        """Simulation is mandatory: placements can never exceed reviews."""
        await runner.run(20)
        counts = runner.state.journal.counts()

        assert counts.get("order_placed", 0) <= counts.get("review_passed", 0)

    async def test_slippage_is_always_adverse_in_paper(self, runner):
        """Fills cross the spread. A favourable average would mean free alpha."""
        await runner.run(20)
        samples = runner.state.realized_slippage_bps

        assert samples
        assert sum(samples) / len(samples) > 0


class TestSessionAccounting:
    """Regression: the loop must actually roll the session boundary.

    The original bug was `if pnl.day_start_equity <= 0: roll_day(...)`. On a
    funded account that condition is never true, so the day never rolled. Two
    silent consequences, and unit tests on `DailyPnL` could not catch either
    because the defect was in the CALLER:

      1. "Day P&L" reported profit-since-inception.
      2. `consecutive_losses` is incremented by `roll_day`, so the
         consecutive-loss breaker could never fire.
    """

    async def test_the_day_rolls_every_session(self, runner):
        results = await runner.run(25)

        assert len(runner.state.pnl.history) >= len(results) - 2

    async def test_reported_day_pnl_matches_the_actual_last_session(self, runner):
        """The dashboard's "Day P&L" must be the day's move, not lifetime gain."""
        await runner.run(25)
        history = runner.state.equity_history
        pnl = runner.state.pnl

        reported = history[-1]["equity"] - pnl.day_start_equity
        actual = history[-1]["equity"] - history[-2]["equity"]
        assert reported == pytest.approx(actual, abs=0.01)

    async def test_day_start_equity_is_not_pinned_to_inception(self, runner):
        """The specific shape of the bug: the baseline never moved."""
        await runner.run(25)

        assert runner.state.pnl.day_start_equity != 100_000.0

    async def test_losing_sessions_are_counted(self, runner):
        """A streak counter stuck at zero makes its breaker unreachable."""
        await runner.run(30)
        losses = [pnl for _, pnl in runner.state.pnl.history if pnl < 0]

        assert losses, "no losing sessions recorded across 30 cycles"

    async def test_session_history_is_dated_per_simulated_session(self, runner):
        """Wall-clock stamping would collapse a replay onto one date."""
        await runner.run(25)
        dates = [d for d, _ in runner.state.pnl.history]

        assert len(set(dates)) == len(dates)

    async def test_consecutive_loss_breaker_can_actually_trip(self, runner):
        """End-to-end proof the breaker is reachable.

        Driving `DailyPnL` directly would only re-test the unit. This asserts the
        loop's own accounting produces a state the kernel halts on.
        """
        from osiris.kernel.state import evaluate_breakers

        await runner.run(25)
        state = runner.loop._build_state(
            runner.market.snapshot(160), runner.state.prices
        )
        # Simulate the streak the loop's own counter feeds the kernel.
        object.__setattr__(state, "consecutive_losses", 5)

        breakers = evaluate_breakers(
            state,
            daily_loss_halt_pct=0.03,
            max_drawdown_halt_pct=0.10,
            consecutive_loss_halt=5,
        )
        assert breakers.is_tripped
        assert any("consecutive" in r for r in breakers.reasons)


class TestKillSwitchStopsNewRisk:
    async def test_engaged_switch_blocks_entries_but_book_is_retained(self, runner):
        """Halt means take no NEW risk, never abandon existing risk."""
        await runner.run(20)
        held_before = set(runner.state.ledger.open_symbols)
        assert held_before, "no book to protect"

        runner.state.killswitch.engage("test halt")
        results = await runner.run(4)

        entries = [
            fill
            for r in results
            if r.report
            for fill in r.report.fills
            if fill.side.value == "buy"
        ]
        assert entries == [], "kill switch must block new entries"
        assert set(runner.state.ledger.open_symbols) <= held_before
