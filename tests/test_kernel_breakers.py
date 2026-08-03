"""Circuit breakers, batch evaluation, and sizing.

The batch tests matter most: twenty individually-compliant buys can collectively
breach every portfolio cap if each is evaluated against the initial state.
"""

from __future__ import annotations

import pytest

from osiris.kernel.sizing import (
    atr_from_bars,
    clamp_to_adv,
    fractional_kelly_cap,
    realized_vol,
    vol_target_notional,
)
from osiris.kernel.state import BreakerState, evaluate_breakers
from osiris.types import Side, VetoCode
from tests.conftest import (
    diversified_positions,
    make_intent,
    make_portfolio,
    make_position,
    make_state,
)


class TestBreakerEvaluation:
    def test_daily_loss_trips(self, limits) -> None:
        state = make_state(make_portfolio(equity=96_000.0))
        object.__setattr__(state, "day_start_equity", 100_000.0)
        b = evaluate_breakers(
            state,
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
        )
        assert b.is_tripped

    def test_small_loss_does_not_trip(self, limits) -> None:
        state = make_state(make_portfolio(equity=99_000.0))
        object.__setattr__(state, "day_start_equity", 100_000.0)
        b = evaluate_breakers(
            state,
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
        )
        assert not b.is_tripped

    def test_drawdown_trips(self, limits) -> None:
        state = make_state(make_portfolio(equity=85_000.0))
        object.__setattr__(state, "peak_equity", 100_000.0)
        object.__setattr__(state, "day_start_equity", 85_000.0)
        b = evaluate_breakers(
            state,
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
        )
        assert b.is_tripped

    def test_consecutive_losses_trip(self, limits) -> None:
        state = make_state(consecutive_losses=5)
        b = evaluate_breakers(
            state,
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
        )
        assert b.is_tripped

    def test_ledger_divergence_trips(self, limits) -> None:
        """Broker state is truth. Divergence means stop until reconciled."""
        b = evaluate_breakers(
            make_state(),
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
            ledger_divergence=True,
        )
        assert b.is_tripped
        assert any("divergence" in r for r in b.reasons)

    def test_schema_drift_trips(self, limits) -> None:
        b = evaluate_breakers(
            make_state(),
            daily_loss_halt_pct=limits.daily_loss_halt_pct,
            max_drawdown_halt_pct=limits.max_drawdown_halt_pct,
            consecutive_loss_halt=limits.consecutive_loss_halt,
            schema_drift=True,
        )
        assert b.is_tripped

    def test_reset_is_explicit(self) -> None:
        """A fuse that resets itself is not a fuse."""
        b = BreakerState().trip(VetoCode.BREAKER_TRIPPED, "loss")
        assert b.is_tripped
        assert not b.reset().is_tripped

    def test_trip_is_idempotent(self) -> None:
        b = BreakerState().trip(VetoCode.BREAKER_TRIPPED, "a")
        assert b.trip(VetoCode.BREAKER_TRIPPED, "b") is b


class TestBatchEvaluation:
    def test_batch_accumulates_exposure(self, kernel) -> None:
        """Ten 2k buys in one tech name must not all pass a 10% cap.

        Each is individually within the per-trade cap; collectively they would
        breach symbol and sector limits. This is why batch state must accumulate.
        """
        state = make_state(
            make_portfolio(equity=100_000.0, positions=diversified_positions(20)),
            symbols=("AAPL",),
        )
        intents = [
            make_intent("AAPL", notional=2_000, reason=f"rank_entry_{i}")
            for i in range(10)
        ]
        decisions = kernel.evaluate_batch(intents, state)
        approved = sum(1 for d in decisions if d.approved)
        assert approved < 10, "batch must not approve unbounded same-symbol exposure"

    def test_exits_processed_before_entries(self, kernel) -> None:
        """Exits free capacity, so they must be evaluated first."""
        book = diversified_positions(20)
        state = make_state(make_portfolio(equity=100_000.0, positions=book), symbols=("PG",))
        intents = [
            make_intent("PG", notional=1_500, reason="rank_entry"),
            make_intent(book[0].symbol, Side.SELL, reason="rank_exit"),
        ]
        decisions = kernel.evaluate_batch(intents, state)
        assert decisions[0].intent.is_exit, "exit should sort first"

    def test_batch_respects_order_budget(self, kernel, limits) -> None:
        state = make_state(
            make_portfolio(equity=1_000_000.0, positions=diversified_positions(20)),
            symbols=tuple(f"S{i}" for i in range(80)),
            orders_placed_today=limits.daily_order_budget - 2,
        )
        intents = [
            make_intent(f"S{i}", notional=1_000, reason=f"e{i}") for i in range(10)
        ]
        decisions = kernel.evaluate_batch(intents, state)
        approved = sum(1 for d in decisions if d.approved)
        assert approved <= 2, "batch must honor the remaining order budget"

    def test_all_vetoes_reported(self, kernel) -> None:
        """Diagnostics: report every violation, not just the first."""
        state = make_state(spread_bps=500.0, quote_age_s=9999, tradable=False)
        d = kernel.evaluate(make_intent(notional=999_999, invalidation=""), state)
        assert len(d.vetoes) >= 4, f"expected many vetoes, got {d.vetoes}"


class TestSizing:
    def test_high_vol_gets_smaller_allocation(self) -> None:
        """Equal-dollar sizing lets the noisiest name dominate outcomes."""
        low = vol_target_notional(100_000, symbol_vol=0.15, position_count=20)
        high = vol_target_notional(100_000, symbol_vol=0.80, position_count=20)
        assert high < low

    def test_respects_max_weight(self) -> None:
        n = vol_target_notional(100_000, symbol_vol=0.01, position_count=20, max_weight=0.10)
        assert n <= 100_000 * 0.10 + 1e-6

    def test_zero_vol_falls_back_to_equal_weight(self) -> None:
        n = vol_target_notional(100_000, symbol_vol=0.0, position_count=20)
        assert n == pytest.approx(5_000.0)

    def test_zero_equity_is_zero(self) -> None:
        assert vol_target_notional(0, symbol_vol=0.2, position_count=20) == 0.0

    def test_kelly_is_fractional(self) -> None:
        """Full Kelly tolerates ruinous drawdowns; quarter-Kelly is the bound."""
        full = fractional_kelly_cap(0.6, 2.0, fraction=1.0)
        quarter = fractional_kelly_cap(0.6, 2.0, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25)

    def test_kelly_negative_edge_is_zero(self) -> None:
        assert fractional_kelly_cap(0.30, 1.0) == 0.0

    def test_kelly_rejects_degenerate_inputs(self) -> None:
        assert fractional_kelly_cap(0.0, 2.0) == 0.0
        assert fractional_kelly_cap(1.0, 2.0) == 0.0
        assert fractional_kelly_cap(0.6, 0.0) == 0.0

    def test_adv_clamp(self) -> None:
        assert clamp_to_adv(10_000, adv_usd=200_000, max_participation=0.01) == 2_000
        assert clamp_to_adv(1_000, adv_usd=200_000, max_participation=0.01) == 1_000

    def test_adv_clamp_zero_adv_is_zero(self) -> None:
        assert clamp_to_adv(10_000, adv_usd=0, max_participation=0.01) == 0.0

    def test_atr_positive(self) -> None:
        highs = [10.5, 11.0, 10.8, 11.2]
        lows = [9.8, 10.2, 10.1, 10.5]
        closes = [10.0, 10.7, 10.4, 11.0]
        assert atr_from_bars(highs, lows, closes) > 0

    def test_atr_insufficient_data(self) -> None:
        assert atr_from_bars([1.0], [1.0], [1.0]) == 0.0

    def test_realized_vol_annualizes(self) -> None:
        returns = [0.01, -0.01, 0.02, -0.015, 0.005]
        daily = realized_vol(returns, annualize=False)
        annual = realized_vol(returns, annualize=True)
        assert annual > daily

    def test_realized_vol_insufficient_data(self) -> None:
        assert realized_vol([0.01]) == 0.0
