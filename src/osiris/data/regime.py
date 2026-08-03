"""Market regime classification.

Matters for two reasons. First, the same strategy that wins in a trend loses in
chop. Second, the VLM chart-reading research found vision models perform well
ONLY in persistent trends and are weak in the more common choppy conditions --
so the chart input must be gated on regime rather than trusted unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osiris.data.indicators import sma
from osiris.types import Regime


@dataclass(frozen=True)
class RegimeState:
    regime: Regime
    trend_strength: float      # -1 (down) .. +1 (up)
    realized_vol: float
    breadth: float             # fraction of universe above its 200-day SMA
    detail: str

    @property
    def allows_chart_vision(self) -> bool:
        """Chart reads are only trustworthy in persistent trends.

        Research finding: VLMs show significant directional bias and weak skill
        outside sustained trends, so this gate is not optional.
        """
        return self.regime in (Regime.TREND_UP, Regime.TREND_DOWN)


def classify_regime(
    benchmark_closes: np.ndarray,
    *,
    universe_closes: dict[str, np.ndarray] | None = None,
    high_vol_threshold: float = 0.28,
    trend_threshold: float = 0.02,
) -> RegimeState:
    """Classify from benchmark price action plus optional breadth."""
    if benchmark_closes.size < 60:
        return RegimeState(
            regime=Regime.CHOP,
            trend_strength=0.0,
            realized_vol=0.0,
            breadth=0.0,
            detail=f"insufficient history ({benchmark_closes.size} bars)",
        )

    rets = np.diff(benchmark_closes) / benchmark_closes[:-1]
    recent = rets[-20:] if rets.size >= 20 else rets
    vol = float(np.std(recent, ddof=1) * np.sqrt(252)) if recent.size > 1 else 0.0

    fast = sma(benchmark_closes, 50)
    slow = sma(benchmark_closes, 200) if benchmark_closes.size >= 200 else fast
    f, s = float(fast[-1]), float(slow[-1])
    trend = (f / s - 1.0) if (s and not np.isnan(s) and not np.isnan(f)) else 0.0

    breadth = 0.0
    if universe_closes:
        above = 0
        counted = 0
        for closes in universe_closes.values():
            if closes.size < 200:
                continue
            ma = sma(closes, 200)[-1]
            if not np.isnan(ma):
                counted += 1
                above += int(closes[-1] > ma)
        breadth = above / counted if counted else 0.0

    # High vol dominates: it changes position sizing regardless of direction.
    if vol >= high_vol_threshold:
        regime = Regime.HIGH_VOL
        detail = f"realized vol {vol:.1%} at/above {high_vol_threshold:.0%}"
    elif trend >= trend_threshold:
        regime = Regime.TREND_UP
        detail = f"50DMA {trend:+.1%} vs 200DMA, vol {vol:.1%}"
    elif trend <= -trend_threshold:
        regime = Regime.TREND_DOWN
        detail = f"50DMA {trend:+.1%} vs 200DMA, vol {vol:.1%}"
    else:
        regime = Regime.CHOP
        detail = f"50DMA within {trend_threshold:.0%} of 200DMA ({trend:+.1%})"

    return RegimeState(
        regime=regime,
        trend_strength=float(np.clip(trend / max(trend_threshold * 5, 1e-9), -1.0, 1.0)),
        realized_vol=vol,
        breadth=breadth,
        detail=detail,
    )
