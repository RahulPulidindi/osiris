"""Synthetic market generator for offline paper runs and UI verification.

Exists so the whole system -- loop, kernel, executor, ledger, API, dashboard --
can be exercised end-to-end with zero network access and zero credentials. That
matters more than it sounds: it means the wiring is testable before an account is
funded, which is the entire premise of the roadmap's ordering.

The generator is deterministic given a seed, and produces a correlated market
(names share a common factor) so the factor-attribution panel has something real
to decompose. An uncorrelated random walk would make every beta ~0 and the
attribution view meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from osiris.execution.loop import MarketSnapshot
from osiris.types import Quote

SECTORS = [
    "Technology",
    "Financials",
    "Healthcare",
    "Industrials",
    "Energy",
    "Staples",
    "Discretionary",
    "Utilities",
    "Materials",
    "Real Estate",
]

# Benchmark sector weights roughly resembling a large-cap index, so sector
# deviation is measured against something plausible rather than uniform.
BENCHMARK_SECTOR_WEIGHTS = {
    "Technology": 0.31,
    "Financials": 0.13,
    "Healthcare": 0.12,
    "Discretionary": 0.11,
    "Industrials": 0.09,
    "Staples": 0.06,
    "Energy": 0.04,
    "Utilities": 0.03,
    "Materials": 0.02,
    "Real Estate": 0.02,
}


@dataclass
class SyntheticMarket:
    """A correlated synthetic universe with a benchmark."""

    n_symbols: int = 120
    n_days: int = 320
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.symbols = [f"SYN{i:03d}" for i in range(self.n_symbols)]

        # One common factor plus idiosyncratic noise. Betas vary so the factor
        # regression has real dispersion to find.
        market_returns = rng.normal(0.0004, 0.010, self.n_days)
        self.betas = {s: float(np.clip(rng.normal(1.0, 0.28), 0.35, 1.9)) for s in self.symbols}

        # Assign sectors IN PROPORTION to the benchmark rather than uniformly.
        # A round-robin assignment would put ~10% of names in Materials while the
        # benchmark holds 2%, so any position there would breach the sector
        # deviation band -- the gate would fire constantly on a fixture artifact
        # and mask real concentration.
        self.sectors = _proportional_sectors(self.symbols)

        self.closes: dict[str, np.ndarray] = {}
        for symbol in self.symbols:
            beta = self.betas[symbol]
            idio = rng.normal(0.0, 0.013, self.n_days)
            # A small persistent per-name drift creates genuine cross-sectional
            # dispersion, so ranking has something to discover.
            drift = rng.normal(0.0, 0.0004)
            rets = beta * market_returns + idio + drift
            path = 100.0 * np.cumprod(1.0 + rets)
            self.closes[symbol] = path

        self.benchmark_closes = 100.0 * np.cumprod(1.0 + market_returns)
        self.market_returns = market_returns

        self.adv = {
            s: float(rng.uniform(3e7, 9e8)) for s in self.symbols
        }
        self.dates = [
            date(2025, 9, 1) + timedelta(days=i) for i in range(self.n_days)
        ]

    def window(self, day_index: int) -> dict[str, np.ndarray]:
        """Closes up to and including `day_index`. No future bars leak."""
        end = day_index + 1
        return {s: c[:end] for s, c in self.closes.items()}

    def prices_on(self, day_index: int) -> dict[str, float]:
        return {s: float(c[day_index]) for s, c in self.closes.items()}

    def quotes_on(self, day_index: int, *, spread_bps: float = 4.0) -> dict[str, Quote]:
        now = datetime.now(UTC)
        out: dict[str, Quote] = {}
        for symbol, price in self.prices_on(day_index).items():
            half = price * (spread_bps / 2.0) / 10_000.0
            out[symbol] = Quote(
                symbol=symbol,
                bid=price - half,
                ask=price + half,
                last=price,
                ts=now,
            )
        return out

    def snapshot(self, day_index: int, *, as_of: date | None = None) -> MarketSnapshot:
        """A full MarketSnapshot for one simulated session."""
        closes = self.window(day_index)
        return MarketSnapshot(
            as_of=as_of or self.dates[day_index],
            universe=list(self.symbols),
            closes=closes,
            quotes=self.quotes_on(day_index),
            benchmark_closes=self.benchmark_closes[: day_index + 1],
            adv=dict(self.adv),
            sectors=dict(self.sectors),
            betas=dict(self.betas),
            benchmark_sector_weights=dict(BENCHMARK_SECTOR_WEIGHTS),
            tradable=dict.fromkeys(self.symbols, True),
            metrics={
                s: {
                    "momentum": round(
                        float(c[-1] / c[max(0, len(c) - 126)] - 1.0), 4
                    ),
                    "price": round(float(c[-1]), 2),
                }
                for s, c in closes.items()
                if len(c) > 1
            },
        )


def _proportional_sectors(symbols: list[str]) -> dict[str, str]:
    """Assign sectors so the universe composition matches the benchmark.

    Largest-remainder apportionment. Keeps the sector-deviation gate meaningful:
    a breach then reflects a real active bet rather than a universe that could
    never have been benchmark-neutral in the first place.
    """
    n = len(symbols)
    exact = {sector: w * n for sector, w in BENCHMARK_SECTOR_WEIGHTS.items()}
    counts = {sector: int(v) for sector, v in exact.items()}

    remaining = n - sum(counts.values())
    for sector, _ in sorted(
        exact.items(), key=lambda kv: -(kv[1] - int(kv[1]))
    )[: max(0, remaining)]:
        counts[sector] += 1

    out: dict[str, str] = {}
    index = 0
    for sector, count in counts.items():
        for _ in range(count):
            if index >= n:
                break
            out[symbols[index]] = sector
            index += 1
    # Any residue from rounding goes to the largest sector.
    largest = max(BENCHMARK_SECTOR_WEIGHTS, key=lambda s: BENCHMARK_SECTOR_WEIGHTS[s])
    while index < n:
        out[symbols[index]] = largest
        index += 1
    return out
