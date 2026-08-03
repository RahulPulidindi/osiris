"""Data plane: indicators, universe construction, regime, macro calendar."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from osiris.data.indicators import (
    average_dollar_volume,
    bollinger,
    distance_from_high,
    ema,
    macd,
    momentum,
    rsi,
    sma,
    volatility_annualized,
)
from osiris.data.macro import (
    EventKind,
    MacroCalendar,
    MacroEvent,
    is_market_holiday,
    is_trading_session,
    next_trading_day,
)
from osiris.data.regime import classify_regime
from osiris.data.universe import LiquidityFloor, UniverseBuilder
from osiris.eval.pit import MembershipSpell, PITUniverse
from osiris.types import Regime

RNG = np.random.default_rng(11)


def trending_series(n: int = 300, drift: float = 0.001) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + drift + RNG.normal(0, 0.004, n))


class TestIndicators:
    def test_sma_matches_manual(self) -> None:
        v = np.array([1.0, 2, 3, 4, 5])
        out = sma(v, 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(2.0)
        assert out[4] == pytest.approx(4.0)

    def test_sma_insufficient_data(self) -> None:
        assert np.all(np.isnan(sma(np.array([1.0, 2.0]), 5)))

    def test_ema_reacts_faster_than_sma(self) -> None:
        """Measured mid-transition: after 10 flat bars the SMA fully catches up,
        so the comparison must be made while the window still straddles the jump.
        """
        v = np.concatenate([np.full(30, 100.0), np.full(3, 120.0)])
        assert ema(v, 10)[-1] > sma(v, 10)[-1]

    def test_rsi_bounded(self) -> None:
        r = rsi(trending_series(200))
        valid = r[~np.isnan(r)]
        assert valid.size > 0
        assert valid.min() >= 0.0 and valid.max() <= 100.0

    def test_rsi_high_on_uptrend(self) -> None:
        rising = np.cumsum(np.full(60, 1.0)) + 100.0
        r = rsi(rising)
        assert r[-1] > 70, "monotonic rise should be overbought"

    def test_macd_crossover_shape(self) -> None:
        line, signal, hist = macd(trending_series(300))
        assert line.size == signal.size == hist.size
        assert np.allclose(hist[-10:], (line - signal)[-10:], equal_nan=True)

    def test_bollinger_ordering(self) -> None:
        lower, mid, upper = bollinger(trending_series(200))
        i = -1
        assert lower[i] < mid[i] < upper[i]

    def test_momentum_positive_on_uptrend(self) -> None:
        assert momentum(trending_series(300, drift=0.002), lookback=126) > 0

    def test_momentum_insufficient_history(self) -> None:
        assert momentum(np.array([100.0, 101.0]), lookback=126) == 0.0

    def test_volatility_positive(self) -> None:
        assert volatility_annualized(trending_series(100)) > 0

    def test_distance_from_high_non_positive(self) -> None:
        assert distance_from_high(trending_series(300)) <= 0.0

    def test_adv_computes(self) -> None:
        closes = np.full(30, 50.0)
        volumes = np.full(30, 1_000_000.0)
        assert average_dollar_volume(closes, volumes) == pytest.approx(50_000_000.0)


class TestLiquidityFloor:
    def test_rejects_thin_volume(self) -> None:
        ok, reason = LiquidityFloor().passes(dollar_volume=1_000_000, price=50.0)
        assert not ok and "ADV" in reason

    def test_rejects_penny_stock(self) -> None:
        ok, reason = LiquidityFloor().passes(dollar_volume=50_000_000, price=1.5)
        assert not ok and "price" in reason

    def test_rejects_wide_spread(self) -> None:
        """Spread is what actually protects the alpha."""
        ok, reason = LiquidityFloor().passes(
            dollar_volume=50_000_000, price=50.0, spread_bps=90.0
        )
        assert not ok and "spread" in reason

    def test_fails_closed_on_missing_data(self) -> None:
        assert not LiquidityFloor().passes(dollar_volume=None, price=50.0)[0]
        assert not LiquidityFloor().passes(dollar_volume=50_000_000, price=None)[0]

    def test_accepts_liquid_large_cap(self) -> None:
        ok, _ = LiquidityFloor().passes(
            dollar_volume=500_000_000,
            price=180.0,
            market_cap=2_500_000_000_000,
            spread_bps=1.6,
        )
        assert ok


class TestUniverseBuilder:
    def test_count_is_an_output(self) -> None:
        """The floor determines the count, not the reverse."""
        pit = PITUniverse(
            [MembershipSpell(f"S{i}", date(2015, 1, 1), None) for i in range(50)]
        )
        metrics = {
            f"S{i}": {
                "dollar_volume": 50_000_000 if i < 30 else 100_000,
                "price": 50.0,
            }
            for i in range(50)
        }
        snap = UniverseBuilder(pit).build(date(2026, 1, 5), metrics)
        assert snap.count == 30
        assert snap.eligibility_count == 50
        assert len(snap.rejected) == 20

    def test_refuses_to_infer_universe(self) -> None:
        """Inferring a universe would silently introduce survivorship bias."""
        with pytest.raises(ValueError, match="survivorship"):
            UniverseBuilder().build(date(2026, 1, 5), {})

    def test_explicit_eligible_list_accepted(self) -> None:
        snap = UniverseBuilder().build(
            date(2026, 1, 5),
            {"AAPL": {"dollar_volume": 5e8, "price": 180.0}},
            eligible=["AAPL"],
        )
        assert snap.symbols == ("AAPL",)

    def test_untradable_rejected(self) -> None:
        snap = UniverseBuilder().build(
            date(2026, 1, 5),
            {"HALT": {"dollar_volume": 5e8, "price": 50.0}},
            eligible=["HALT"],
            tradable={"HALT": False},
        )
        assert snap.count == 0
        assert snap.rejected["HALT"] == "not tradable"

    def test_missing_metrics_rejected(self) -> None:
        snap = UniverseBuilder().build(date(2026, 1, 5), {}, eligible=["GHOST"])
        assert snap.rejected["GHOST"] == "no metrics"


class TestRegimeClassifier:
    def test_detects_uptrend(self) -> None:
        closes = 100.0 * np.cumprod(1.0 + np.full(300, 0.0012))
        assert classify_regime(closes).regime is Regime.TREND_UP

    def test_detects_downtrend(self) -> None:
        closes = 100.0 * np.cumprod(1.0 + np.full(300, -0.0012))
        assert classify_regime(closes).regime is Regime.TREND_DOWN

    def test_detects_high_vol(self) -> None:
        closes = 100.0 * np.cumprod(1.0 + RNG.normal(0.0, 0.045, 300))
        assert classify_regime(closes).regime is Regime.HIGH_VOL

    def test_insufficient_history_is_chop(self) -> None:
        state = classify_regime(np.full(20, 100.0))
        assert state.regime is Regime.CHOP
        assert "insufficient" in state.detail

    def test_chart_vision_gated_off_in_chop(self) -> None:
        """VLMs are weak outside persistent trends, so the gate must close."""
        state = classify_regime(np.full(20, 100.0))
        assert not state.allows_chart_vision

    def test_chart_vision_allowed_in_trend(self) -> None:
        closes = 100.0 * np.cumprod(1.0 + np.full(300, 0.0012))
        assert classify_regime(closes).allows_chart_vision

    def test_breadth_computed(self) -> None:
        rising = {f"S{i}": 100.0 * np.cumprod(1.0 + np.full(250, 0.001)) for i in range(10)}
        state = classify_regime(
            100.0 * np.cumprod(1.0 + np.full(300, 0.001)), universe_closes=rising
        )
        assert state.breadth > 0.9


class TestMacroCalendar:
    def test_fomc_blackout_active(self) -> None:
        at = datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
        cal = MacroCalendar([MacroEvent(EventKind.FOMC, at)])
        blocked, reason = cal.is_blackout(at - timedelta(minutes=60))
        assert blocked and "FOMC" in reason

    def test_outside_window_clear(self) -> None:
        at = datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
        cal = MacroCalendar([MacroEvent(EventKind.FOMC, at)])
        assert not cal.is_blackout(at - timedelta(hours=8))[0]

    def test_fomc_window_wider_than_jolts(self) -> None:
        """Rate decisions reprice everything; minor prints do not."""
        at = datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
        fomc = MacroEvent(EventKind.FOMC, at).blackout_window()
        jolts = MacroEvent(EventKind.JOLTS, at).blackout_window()
        assert (fomc[1] - fomc[0]) > (jolts[1] - jolts[0])

    def test_next_event(self) -> None:
        now = datetime(2026, 7, 1, tzinfo=UTC)
        cal = MacroCalendar(
            [
                MacroEvent(EventKind.CPI, datetime(2026, 7, 15, 12, 30, tzinfo=UTC)),
                MacroEvent(EventKind.FOMC, datetime(2026, 7, 29, 18, 0, tzinfo=UTC)),
            ]
        )
        assert cal.next_event(now).kind is EventKind.CPI

    def test_holidays_are_not_sessions(self) -> None:
        assert is_market_holiday(date(2026, 12, 25))
        assert not is_trading_session(date(2026, 12, 25))

    def test_weekend_not_session(self) -> None:
        assert not is_trading_session(date(2026, 8, 1))  # Saturday

    def test_next_trading_day_skips_holiday_weekend(self) -> None:
        # Dec 25 2026 is a Friday holiday, so next session is Mon Dec 28.
        assert next_trading_day(date(2026, 12, 24)) == date(2026, 12, 28)
