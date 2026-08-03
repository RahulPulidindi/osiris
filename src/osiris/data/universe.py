"""Universe construction: eligibility pool, liquidity floor, per-name gate.

The count is an OUTPUT, not an input. What protects the alpha is spread, not
cardinality: the cited research universe had a median spread of 1.6 bps, which
is why transaction costs stayed under 10 percent of gross alpha. Small caps show
the strongest raw signal but their spreads eat it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from osiris.eval.pit import PITUniverse
from osiris.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LiquidityFloor:
    """Screens that define the tradable universe.

    Defaults target large, liquid names where modeled slippage stays honest.
    """

    min_dollar_volume: float = 20_000_000.0
    min_price: float = 5.0
    max_price: float = 100_000.0
    min_market_cap: float = 2_000_000_000.0
    max_spread_bps: float = 25.0

    def passes(
        self,
        *,
        dollar_volume: float | None,
        price: float | None,
        market_cap: float | None = None,
        spread_bps: float | None = None,
    ) -> tuple[bool, str]:
        """Fails closed: unknown metrics are rejected, not assumed adequate."""
        if dollar_volume is None:
            return False, "unknown dollar volume"
        if price is None:
            return False, "unknown price"
        if dollar_volume < self.min_dollar_volume:
            return False, f"ADV ${dollar_volume:,.0f} below ${self.min_dollar_volume:,.0f}"
        if not (self.min_price <= price <= self.max_price):
            return False, f"price ${price:.2f} outside [{self.min_price}, {self.max_price}]"
        if market_cap is not None and market_cap < self.min_market_cap:
            return False, f"market cap ${market_cap:,.0f} below ${self.min_market_cap:,.0f}"
        if spread_bps is not None and spread_bps > self.max_spread_bps:
            return False, f"spread {spread_bps:.1f}bps exceeds {self.max_spread_bps}bps"
        return True, "ok"

    def to_scanner_filters(self) -> dict[str, float]:
        """Translate to MCP `create_scan` filter arguments.

        Exact keys depend on `get_scanner_filter_specs` for the account, so this
        is a starting mapping to be reconciled during Phase 0.
        """
        return {
            "min_dollar_volume": self.min_dollar_volume,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "min_market_cap": self.min_market_cap,
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: date
    symbols: tuple[str, ...]
    eligibility_count: int
    rejected: dict[str, str]

    @property
    def count(self) -> int:
        return len(self.symbols)

    def summary(self) -> str:
        return (
            f"{self.count} names on {self.as_of} "
            f"(from {self.eligibility_count} eligible, "
            f"{len(self.rejected)} rejected by liquidity floor)"
        )


class UniverseBuilder:
    """Composes PIT membership with a liquidity screen.

    For live trading, today's membership is correct. For backtesting, membership
    MUST be as-of the simulated date or the result is fiction.
    """

    def __init__(
        self,
        pit: PITUniverse | None = None,
        floor: LiquidityFloor | None = None,
    ) -> None:
        self.pit = pit
        self.floor = floor or LiquidityFloor()

    def build(
        self,
        as_of: date,
        metrics: dict[str, dict[str, float | None]],
        *,
        eligible: list[str] | None = None,
        tradable: dict[str, bool] | None = None,
    ) -> UniverseSnapshot:
        """metrics: symbol -> {dollar_volume, price, market_cap, spread_bps}."""
        if eligible is None:
            if self.pit is None:
                raise ValueError(
                    "Provide `eligible` or a PITUniverse. Refusing to infer a "
                    "universe, which would silently introduce survivorship bias."
                )
            eligible = self.pit.members_on(as_of)

        tradable = tradable or {}
        kept: list[str] = []
        rejected: dict[str, str] = {}

        for symbol in eligible:
            if tradable.get(symbol) is False:
                rejected[symbol] = "not tradable"
                continue
            m = metrics.get(symbol)
            if m is None:
                rejected[symbol] = "no metrics"
                continue
            ok, reason = self.floor.passes(
                dollar_volume=m.get("dollar_volume"),
                price=m.get("price"),
                market_cap=m.get("market_cap"),
                spread_bps=m.get("spread_bps"),
            )
            (kept.append(symbol) if ok else rejected.__setitem__(symbol, reason))

        snapshot = UniverseSnapshot(
            as_of=as_of,
            symbols=tuple(sorted(kept)),
            eligibility_count=len(eligible),
            rejected=rejected,
        )
        log.info(
            "universe.built",
            as_of=as_of.isoformat(),
            count=snapshot.count,
            eligible=len(eligible),
            rejected=len(rejected),
        )
        return snapshot
