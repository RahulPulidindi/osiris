"""Chart rendering tests.

The critical property is cutoff truncation: a chart that contains a bar dated
after the simulation date leaks the future into a backtest.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from osiris.data.charts import (
    ChartReadCalibrator,
    render_chart,
    render_multiscale,
)
from osiris.eval.pit import LookaheadError

RNG = np.random.default_rng(5)


def make_frame(n: int = 120, start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = 100.0 * np.cumprod(1.0 + RNG.normal(0.0005, 0.012, n))
    high = close * (1 + np.abs(RNG.normal(0, 0.004, n)))
    low = close * (1 - np.abs(RNG.normal(0, 0.004, n)))
    open_ = (high + low) / 2
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": RNG.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


class TestCutoffTruncation:
    def test_future_bars_excluded(self) -> None:
        """The anti-lookahead guarantee, and the reason not to use TradingView."""
        df = make_frame(120, start="2026-01-01")
        cutoff = date(2026, 3, 2)
        chart = render_chart(df, "TEST", cutoff)
        included = df.loc[pd.to_datetime(df.index) <= pd.Timestamp(cutoff)]
        assert chart.bar_count <= len(included)
        assert chart.as_of == cutoff

    def test_cutoff_before_all_data_raises(self) -> None:
        df = make_frame(60, start="2026-06-01")
        with pytest.raises(LookaheadError, match="No bars at or before"):
            render_chart(df, "TEST", date(2026, 1, 1))

    def test_bar_count_capped(self) -> None:
        df = make_frame(400, start="2024-01-01")
        chart = render_chart(df, "TEST", date(2026, 1, 1), max_bars=40)
        assert chart.bar_count == 40

    def test_deterministic_output(self) -> None:
        """Same inputs must give the same image so backtests are reproducible."""
        df = make_frame(100)
        a = render_chart(df, "TEST", date(2026, 4, 1))
        b = render_chart(df, "TEST", date(2026, 4, 1))
        assert a.png_bytes == b.png_bytes


class TestRendering:
    def test_produces_png(self) -> None:
        chart = render_chart(make_frame(80), "AAPL", date(2026, 4, 1))
        assert chart.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert chart.data_url.startswith("data:image/png;base64,")

    def test_missing_columns_rejected(self) -> None:
        df = make_frame(50).drop(columns=["Volume"])
        with pytest.raises(ValueError, match="missing OHLCV"):
            render_chart(df, "TEST", date(2026, 3, 1))

    def test_empty_frame_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty price frame"):
            render_chart(pd.DataFrame(), "TEST", date(2026, 3, 1))

    def test_multiscale_returns_both_timeframes(self) -> None:
        """Research evaluated MULTI-SCALE input; one timeframe loses half the signal."""
        charts = render_multiscale(make_frame(250, start="2025-06-02"), "AAPL", date(2026, 4, 1))
        assert len(charts) == 2
        assert {c.timeframe for c in charts} == {"daily", "weekly"}

    def test_short_history_still_renders(self) -> None:
        """Few bars means fewer MAs, not a crash."""
        chart = render_chart(make_frame(12), "NEW", date(2026, 2, 1))
        assert chart.bar_count > 0


class TestCalibration:
    def test_uncalibrated_reads_are_shrunk(self) -> None:
        """Raw VLM output carries directional bias, so it must not be trusted."""
        cal = ChartReadCalibrator()
        assert not cal.is_calibrated
        assert cal.calibrate(4.0) == pytest.approx(1.0)  # 4.0 * 0.25

    def test_clipping_enforced(self) -> None:
        cal = ChartReadCalibrator()
        assert cal.calibrate(99.0) == pytest.approx(1.25)
        assert cal.calibrate(-99.0) == pytest.approx(-1.25)

    def test_insufficient_observations_records_but_does_not_fit(self) -> None:
        cal = ChartReadCalibrator(min_observations=60)
        cal.fit([1.0] * 10, [0.01] * 10)
        assert not cal.is_calibrated
        assert cal.slope == 1.0

    def test_fit_recovers_positive_relationship(self) -> None:
        raw = list(np.linspace(-5, 5, 120))
        realized = [0.002 * r + float(RNG.normal(0, 0.0005)) for r in raw]
        cal = ChartReadCalibrator(min_observations=60)
        cal.fit(raw, realized)
        assert cal.is_calibrated
        assert cal.slope > 0

    def test_fit_detects_inverted_signal(self) -> None:
        """If chart reads anticorrelate with returns, calibration must flip sign."""
        raw = list(np.linspace(-5, 5, 120))
        realized = [-0.002 * r for r in raw]
        cal = ChartReadCalibrator(min_observations=60)
        cal.fit(raw, realized)
        assert cal.slope < 0

    def test_length_mismatch_rejected(self) -> None:
        cal = ChartReadCalibrator()
        with pytest.raises(ValueError, match="length mismatch"):
            cal.fit([1.0, 2.0], [0.01])
