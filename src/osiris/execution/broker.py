"""Broker interface plus the paper implementation.

The paper broker implements the SAME interface as the live MCP adapter, so the
execution pipeline is identical in both modes. If paper and live take different
code paths, paper proves nothing about live.

The paper fill model is deliberately pessimistic:
  - fills cross the spread, never at the mid
  - a size-dependent impact term is added
  - orders can partially fill and can be rejected

Optimistic paper fills are the single most common reason a backtest and a live
account diverge. Assuming mid-price fills quietly invents alpha equal to half the
spread on every trade, which at 20 names a day is most of a thin edge.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from osiris.logging import get_logger
from osiris.types import Fill, OrderKind, Quote, Side

log = get_logger(__name__)


class OrderRejected(RuntimeError):
    """The broker refused the order. Never treated as a fill."""


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    notional_usd: float
    kind: OrderKind = OrderKind.LIMIT
    limit_price: float | None = None
    idempotency_key: str = ""
    correlation_id: str = ""


@dataclass(frozen=True)
class ReviewResult:
    """Outcome of the mandatory pre-trade simulation."""

    accepted: bool
    estimated_price: float | None = None
    estimated_quantity: float | None = None
    estimated_cost: float | None = None
    message: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlaceResult:
    order_id: str
    accepted: bool
    fills: tuple[Fill, ...] = ()
    message: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def filled_quantity(self) -> float:
        return sum(f.quantity for f in self.fills)

    @property
    def average_price(self) -> float | None:
        qty = self.filled_quantity
        if qty <= 0:
            return None
        return sum(f.quantity * f.price for f in self.fills) / qty


class Broker(ABC):
    """Every order goes review -> place. There is no place-only path."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def review(self, request: OrderRequest) -> ReviewResult: ...

    @abstractmethod
    async def place(self, request: OrderRequest) -> PlaceResult: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, float]: ...

    @abstractmethod
    async def get_account_equity(self) -> float: ...


@dataclass
class PaperFillModel:
    """Pessimistic-by-default fill assumptions."""

    spread_bps: float = 4.0
    impact_coefficient: float = 0.15
    partial_fill_probability: float = 0.08
    rejection_probability: float = 0.01
    min_fill_fraction: float = 0.55
    seed: int | None = 11

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def effective_price(
        self, side: Side, reference_price: float, notional: float, adv_usd: float
    ) -> float:
        """Cross the spread, then add sqrt impact. Always adverse."""
        half_spread = reference_price * (self.spread_bps / 2.0) / 10_000.0
        participation = (notional / adv_usd) if adv_usd > 0 else 0.0
        impact = reference_price * self.impact_coefficient * (participation**0.5) / 100.0
        adverse = half_spread + impact
        return reference_price + adverse if side is Side.BUY else max(0.01, reference_price - adverse)

    def fill_fraction(self) -> float:
        if self._rng.random() < self.partial_fill_probability:
            return self._rng.uniform(self.min_fill_fraction, 0.99)
        return 1.0

    def rejects(self) -> bool:
        return self._rng.random() < self.rejection_probability


class PaperBroker(Broker):
    """Simulated broker with realistic frictions and full accounting."""

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        quotes: dict[str, Quote] | None = None,
        adv: dict[str, float] | None = None,
        fill_model: PaperFillModel | None = None,
    ) -> None:
        self.cash = starting_cash
        self.quotes: dict[str, Quote] = dict(quotes or {})
        self.adv: dict[str, float] = dict(adv or {})
        self.model = fill_model or PaperFillModel()
        self.shares: dict[str, float] = {}
        self.order_seq = 0
        self._placed_keys: dict[str, PlaceResult] = {}

    @property
    def name(self) -> str:
        return "paper"

    def set_quotes(self, quotes: dict[str, Quote]) -> None:
        self.quotes.update(quotes)

    def set_adv(self, adv: dict[str, float]) -> None:
        self.adv.update(adv)

    def _reference_price(self, symbol: str, side: Side) -> float | None:
        """Use the far side of the book: buyers pay the ask, sellers hit the bid."""
        q = self.quotes.get(symbol)
        if q is None:
            return None
        return q.ask if side is Side.BUY else q.bid

    async def review(self, request: OrderRequest) -> ReviewResult:
        ref = self._reference_price(request.symbol, request.side)
        if ref is None or ref <= 0:
            return ReviewResult(False, message=f"no quote for {request.symbol}")

        price = self.model.effective_price(
            request.side, ref, request.notional_usd, self.adv.get(request.symbol, 0.0)
        )
        quantity = request.notional_usd / price if price > 0 else 0.0
        if quantity <= 0:
            return ReviewResult(False, message="computed zero quantity")

        if request.side is Side.BUY and request.notional_usd > self.cash + 1e-9:
            return ReviewResult(
                False,
                message=f"insufficient cash: need {request.notional_usd:.2f}, have {self.cash:.2f}",
            )
        if request.side is Side.SELL:
            held = self.shares.get(request.symbol, 0.0)
            if held <= 0:
                return ReviewResult(False, message=f"no position in {request.symbol} to sell")

        return ReviewResult(
            accepted=True,
            estimated_price=price,
            estimated_quantity=quantity,
            estimated_cost=quantity * price,
            message="ok",
        )

    async def place(self, request: OrderRequest) -> PlaceResult:
        # Idempotency: a retry of the same logical order returns the first result.
        if request.idempotency_key and request.idempotency_key in self._placed_keys:
            log.warning("paper.duplicate_place", key=request.idempotency_key[:16])
            return self._placed_keys[request.idempotency_key]

        review = await self.review(request)
        if not review.accepted:
            raise OrderRejected(review.message)

        self.order_seq += 1
        order_id = f"paper-{self.order_seq:06d}"

        if self.model.rejects():
            result = PlaceResult(order_id, False, message="simulated broker rejection")
            if request.idempotency_key:
                self._placed_keys[request.idempotency_key] = result
            return result

        price = review.estimated_price or 0.0
        quantity = (review.estimated_quantity or 0.0) * self.model.fill_fraction()
        if request.side is Side.SELL:
            quantity = min(quantity, self.shares.get(request.symbol, 0.0))
        if quantity <= 0:
            result = PlaceResult(order_id, False, message="zero fillable quantity")
            if request.idempotency_key:
                self._placed_keys[request.idempotency_key] = result
            return result

        notional = quantity * price
        if request.side is Side.BUY:
            self.shares[request.symbol] = self.shares.get(request.symbol, 0.0) + quantity
            self.cash -= notional
        else:
            self.shares[request.symbol] = max(
                0.0, self.shares.get(request.symbol, 0.0) - quantity
            )
            self.cash += notional
        if self.shares.get(request.symbol, 0.0) <= 1e-9:
            self.shares.pop(request.symbol, None)

        ref = self._reference_price(request.symbol, request.side)
        fill = Fill(
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            price=price,
            ts=datetime.now(UTC),
            order_id=order_id,
            idempotency_key=request.idempotency_key,
            intended_price=request.limit_price or ref,
        )
        result = PlaceResult(order_id, True, fills=(fill,), message="filled")
        if request.idempotency_key:
            self._placed_keys[request.idempotency_key] = result
        log.info(
            "paper.filled",
            symbol=request.symbol,
            side=request.side.value,
            quantity=round(quantity, 4),
            price=round(price, 4),
        )
        return result

    async def get_positions(self) -> dict[str, float]:
        return dict(self.shares)

    async def get_account_equity(self) -> float:
        held = 0.0
        for symbol, qty in self.shares.items():
            q = self.quotes.get(symbol)
            if q is not None:
                held += qty * q.mid
        return self.cash + held
