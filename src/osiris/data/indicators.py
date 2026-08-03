"""Local technical indicators.

Computed locally rather than via TradingView so they are deterministic,
backtestable at a cutoff date, and free. The MCP also offers indicators, but a
local implementation can be replayed historically without a network call.
"""

from __future__ import annotations

import numpy as np


def sma(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or values.size < window:
        return np.full(values.size, np.nan)
    out = np.full(values.size, np.nan)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or values.size == 0:
        return np.full(values.size, np.nan)
    alpha = 2.0 / (window + 1.0)
    out = np.empty(values.size)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(closes: np.ndarray, window: int = 14) -> np.ndarray:
    if closes.size < window + 1:
        return np.full(closes.size, np.nan)
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = sma(gains, window)
    avg_loss = sma(losses, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
        out = 100.0 - (100.0 / (1.0 + rs))
    out[np.isnan(avg_gain)] = np.nan
    return out


def macd(
    closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema(closes, fast) - ema(closes, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(
    closes: np.ndarray, window: int = 20, n_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(closes, window)
    sd = np.full(closes.size, np.nan)
    for i in range(window - 1, closes.size):
        sd[i] = float(np.std(closes[i - window + 1 : i + 1], ddof=1))
    return mid - n_std * sd, mid, mid + n_std * sd


def momentum(closes: np.ndarray, lookback: int = 126) -> float:
    """Total return over the lookback. 126 sessions is roughly six months."""
    if closes.size <= lookback or closes[-lookback - 1] <= 0:
        return 0.0
    return float(closes[-1] / closes[-lookback - 1] - 1.0)


def volatility_annualized(closes: np.ndarray, window: int = 20) -> float:
    if closes.size < window + 1:
        return 0.0
    rets = np.diff(closes[-(window + 1) :]) / closes[-(window + 1) : -1]
    return float(np.std(rets, ddof=1) * np.sqrt(252)) if rets.size > 1 else 0.0


def distance_from_high(closes: np.ndarray, window: int = 252) -> float:
    """Fractional distance below the trailing high. Negative means below."""
    if closes.size == 0:
        return 0.0
    window_slice = closes[-window:] if closes.size >= window else closes
    high = float(np.max(window_slice))
    return float(closes[-1] / high - 1.0) if high > 0 else 0.0


def average_dollar_volume(closes: np.ndarray, volumes: np.ndarray, window: int = 20) -> float:
    if closes.size == 0 or volumes.size == 0:
        return 0.0
    n = min(window, closes.size, volumes.size)
    return float(np.mean(closes[-n:] * volumes[-n:]))
