"""Deterministic chart rendering for vision input.

Evidence is counterintuitive and worth following: an April 2026 VLM benchmark
found candlestick charts consistently outperform equivalent tabular data, with
higher IC significance and higher median IC than an XGBoost numerical baseline.

But the same research imposes three hard constraints, all reflected here:
  1. VLMs are reliable only in persistent trends, weak in chop -> regime gate.
  2. They show directional bias and ignore stated horizons -> calibration.
  3. Vision tokens are expensive -> Stage 3 only, ~40 names.

Rendered locally with mplfinance rather than TradingView, because TradingView
cannot be reliably captured at a historical cutoff and is therefore unusable in
a backtest. The cutoff truncation below is the anti-lookahead guarantee.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import date

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from osiris.eval.pit import LookaheadError
from osiris.logging import get_logger

log = get_logger(__name__)

MAX_BARS = 60  # research capped displayed candles to control input complexity


@dataclass(frozen=True)
class RenderedChart:
    symbol: str
    as_of: date
    timeframe: str
    png_bytes: bytes
    bar_count: int

    @property
    def data_url(self) -> str:
        b64 = base64.b64encode(self.png_bytes).decode()
        return f"data:image/png;base64,{b64}"


def _validate_and_truncate(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Drop every bar dated after the cutoff. The anti-lookahead guarantee."""
    if df.empty:
        raise ValueError("empty price frame")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")

    idx = pd.to_datetime(df.index)
    cutoff = pd.Timestamp(as_of)
    truncated = df.loc[idx <= cutoff]
    if truncated.empty:
        raise LookaheadError(
            f"No bars at or before {as_of}; refusing to render a chart that "
            "would contain only future data."
        )
    return truncated


def render_chart(
    df: pd.DataFrame,
    symbol: str,
    as_of: date,
    *,
    timeframe: str = "daily",
    max_bars: int = MAX_BARS,
) -> RenderedChart:
    """Render a single-timeframe candlestick chart with MAs and volume.

    Deterministic: same inputs produce the same image, so a backtest is
    reproducible and a chart read can be replayed.
    """
    import mplfinance as mpf

    data = _validate_and_truncate(df, as_of)
    if timeframe == "weekly":
        data = data.resample("W").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna()

    data = data.tail(max_bars)
    if data.empty:
        raise LookaheadError(f"no bars remain for {symbol} at {as_of}")

    # MAs only where enough history exists, else mplfinance raises.
    mavs = tuple(w for w in (5, 20, 50) if len(data) > w)

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"font.size": 9, "figure.facecolor": "white"},
        gridstyle=":",
    )
    buf = io.BytesIO()
    kwargs = dict(
        type="candle",
        volume=True,
        style=style,
        title=f"{symbol} {timeframe} through {as_of.isoformat()}",
        figsize=(9, 6),
        savefig=dict(fname=buf, dpi=100, bbox_inches="tight"),
        warn_too_much_data=len(data) + 10,
    )
    if mavs:
        kwargs["mav"] = mavs

    mpf.plot(data, **kwargs)
    plt.close("all")
    buf.seek(0)
    return RenderedChart(
        symbol=symbol,
        as_of=as_of,
        timeframe=timeframe,
        png_bytes=buf.getvalue(),
        bar_count=len(data),
    )


def render_multiscale(
    df: pd.DataFrame, symbol: str, as_of: date
) -> list[RenderedChart]:
    """Daily plus weekly.

    The research specifically evaluated MULTI-SCALE charts: longer horizons carry
    trend direction while shorter ones show inflection. A single timeframe
    discards half the signal the benchmark measured.
    """
    out: list[RenderedChart] = []
    for tf in ("daily", "weekly"):
        try:
            out.append(render_chart(df, symbol, as_of, timeframe=tf))
        except (ValueError, LookaheadError) as exc:
            log.warning("chart.render.skipped", symbol=symbol, timeframe=tf, error=str(exc))
    return out


# ------------------------------------------------------------------ calibration
@dataclass
class ChartReadCalibrator:
    """Post-hoc calibration for VLM chart reads.

    The research found raw VLM outputs carry significant directional bias and
    weak sensitivity to the requested horizon; Platt scaling was required to make
    them usable. Until enough live observations accumulate, the calibrator
    deliberately SHRINKS the signal toward neutral rather than trusting it.
    """

    slope: float = 1.0
    intercept: float = 0.0
    observations: int = 0
    min_observations: int = 60

    @property
    def is_calibrated(self) -> bool:
        return self.observations >= self.min_observations

    def calibrate(self, raw_score: float) -> float:
        """Map a raw score in [-5, 5] to a calibrated score.

        Uncalibrated reads are shrunk to a quarter weight: the chart is one
        feature among several, never a standalone signal.
        """
        clipped = max(-5.0, min(5.0, raw_score))
        if not self.is_calibrated:
            return clipped * 0.25
        return max(-5.0, min(5.0, self.slope * clipped + self.intercept))

    def fit(self, raw_scores: list[float], realized_returns: list[float]) -> None:
        """Least-squares fit of raw score against realized forward return."""
        import numpy as np

        if len(raw_scores) != len(realized_returns):
            raise ValueError("length mismatch between scores and returns")
        if len(raw_scores) < self.min_observations:
            self.observations = len(raw_scores)
            return

        x = np.asarray(raw_scores, dtype=float)
        y = np.asarray(realized_returns, dtype=float)
        # Scale realized returns into score space so slope is interpretable.
        y_scaled = y / (np.std(y) or 1.0)
        design = np.column_stack([np.ones(x.size), x])
        coef, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
        self.intercept, self.slope = float(coef[0]), float(coef[1])
        self.observations = len(raw_scores)
        log.info(
            "chart.calibrated",
            slope=round(self.slope, 4),
            intercept=round(self.intercept, 4),
            n=self.observations,
        )
