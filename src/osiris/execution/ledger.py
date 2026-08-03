"""Internal ledger and broker reconciliation.

The ledger is Osiris's own belief about what it owns. The broker is the truth.
Reconciliation compares them, and **any divergence trips a breaker** rather than
being logged and ignored.

Why this is treated as a correctness issue rather than housekeeping: MCP tools
that fail logically return `isError: true` with an HTTP 200. Code that only
catches exceptions books a rejected order as a fill. The ledger then believes it
holds a position it does not own, and every downstream sizing decision is wrong
in the same direction. Reconciliation is the only thing that catches this class of
bug, which is why the divergence threshold is tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from osiris.logging import get_logger
from osiris.types import Fill, Portfolio, Position, Side

log = get_logger(__name__)

# Share tolerance for float noise. Not a "close enough" allowance: fractional
# shares are exact multiples, so anything above this is a real break.
SHARE_EPSILON = 1e-6


@dataclass
class LedgerPosition:
    symbol: str
    quantity: float = 0.0
    cost_basis_total: float = 0.0
    realized_pnl: float = 0.0
    sector: str = "Unknown"
    beta: float = 1.0
    opened_at: datetime | None = None

    @property
    def avg_cost(self) -> float:
        return self.cost_basis_total / self.quantity if self.quantity > SHARE_EPSILON else 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.market_value(price) - self.cost_basis_total


@dataclass(frozen=True)
class Divergence:
    symbol: str
    ledger_qty: float
    broker_qty: float
    kind: str

    @property
    def delta(self) -> float:
        return self.broker_qty - self.ledger_qty

    def describe(self) -> str:
        return (
            f"{self.symbol}: ledger={self.ledger_qty:.6f} "
            f"broker={self.broker_qty:.6f} ({self.kind}, delta={self.delta:+.6f})"
        )


@dataclass(frozen=True)
class ReconciliationResult:
    as_of: datetime
    divergences: tuple[Divergence, ...]
    ledger_equity: float
    broker_equity: float
    equity_tolerance: float

    @property
    def clean(self) -> bool:
        return not self.divergences and self.equity_matches

    @property
    def equity_matches(self) -> bool:
        if self.broker_equity <= 0:
            return True
        rel = abs(self.ledger_equity - self.broker_equity) / self.broker_equity
        return rel <= self.equity_tolerance

    def describe(self) -> str:
        if self.clean:
            return f"clean: equity {self.broker_equity:,.2f} matches ledger"
        lines = [f"{len(self.divergences)} position divergence(s):"]
        lines += [f"  {d.describe()}" for d in self.divergences]
        if not self.equity_matches:
            lines.append(
                f"  EQUITY: ledger={self.ledger_equity:,.2f} "
                f"broker={self.broker_equity:,.2f}"
            )
        return "\n".join(lines)


class Ledger:
    """Osiris's own accounting. Updated only from confirmed fills."""

    def __init__(self, *, starting_cash: float = 0.0) -> None:
        self.cash = starting_cash
        self.positions: dict[str, LedgerPosition] = {}
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0
        self._seen_keys: set[str] = set()

    # ------------------------------------------------------------------ writes
    def apply_fill(self, fill: Fill) -> bool:
        """Apply a confirmed fill. Idempotent on idempotency_key.

        Returns False if this fill was already applied — a retry that returns the
        same order must not double-count.
        """
        key = fill.idempotency_key or f"{fill.order_id}:{fill.symbol}:{fill.ts.isoformat()}"
        if key in self._seen_keys:
            log.warning("ledger.duplicate_fill_ignored", symbol=fill.symbol, key=key[:16])
            return False
        self._seen_keys.add(key)
        self.fills.append(fill)

        pos = self.positions.setdefault(fill.symbol, LedgerPosition(symbol=fill.symbol))
        notional = fill.quantity * fill.price

        if fill.side is Side.BUY:
            if pos.quantity <= SHARE_EPSILON:
                pos.opened_at = fill.ts
            pos.quantity += fill.quantity
            pos.cost_basis_total += notional
            self.cash -= notional
        else:
            sell_qty = min(fill.quantity, pos.quantity)
            avg = pos.avg_cost
            realized = sell_qty * (fill.price - avg)
            pos.realized_pnl += realized
            self.realized_pnl += realized
            pos.quantity -= sell_qty
            pos.cost_basis_total = max(0.0, pos.cost_basis_total - sell_qty * avg)
            self.cash += notional
            if pos.quantity <= SHARE_EPSILON:
                pos.quantity = 0.0
                pos.cost_basis_total = 0.0
                pos.opened_at = None
        return True

    def set_metadata(self, sectors: dict[str, str], betas: dict[str, float]) -> None:
        for sym, pos in self.positions.items():
            pos.sector = sectors.get(sym, pos.sector)
            pos.beta = betas.get(sym, pos.beta)

    # ------------------------------------------------------------------- reads
    @property
    def open_symbols(self) -> list[str]:
        return sorted(s for s, p in self.positions.items() if p.quantity > SHARE_EPSILON)

    def quantity_of(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def equity(self, prices: dict[str, float]) -> float:
        held = sum(
            p.market_value(prices.get(s, p.avg_cost))
            for s, p in self.positions.items()
            if p.quantity > SHARE_EPSILON
        )
        return self.cash + held

    def to_portfolio(
        self, prices: dict[str, float], *, buying_power: float | None = None
    ) -> Portfolio:
        """Project into the immutable Portfolio the kernel consumes."""
        positions = tuple(
            Position(
                symbol=s,
                quantity=p.quantity,
                cost_basis=p.cost_basis_total,
                market_value=max(0.0, p.market_value(prices.get(s, p.avg_cost))),
                sector=p.sector,
                beta=p.beta,
            )
            for s, p in sorted(self.positions.items())
            if p.quantity > SHARE_EPSILON
        )
        equity = self.equity(prices)
        return Portfolio(
            equity=max(0.0, equity),
            cash=self.cash,
            buying_power=self.cash if buying_power is None else buying_power,
            positions=positions,
            as_of=datetime.now(UTC),
        )

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            p.unrealized_pnl(prices.get(s, p.avg_cost))
            for s, p in self.positions.items()
            if p.quantity > SHARE_EPSILON
        )

    # ---------------------------------------------------------- reconciliation
    def reconcile(
        self,
        broker_positions: dict[str, float],
        *,
        broker_equity: float = 0.0,
        prices: dict[str, float] | None = None,
        equity_tolerance: float = 0.005,
    ) -> ReconciliationResult:
        """Compare ledger to broker. Divergence is a breaker condition."""
        prices = prices or {}
        divergences: list[Divergence] = []

        ledger_qty = {s: self.quantity_of(s) for s in self.open_symbols}
        broker_qty = {s: q for s, q in broker_positions.items() if abs(q) > SHARE_EPSILON}

        for symbol in sorted(set(ledger_qty) | set(broker_qty)):
            lq = ledger_qty.get(symbol, 0.0)
            bq = broker_qty.get(symbol, 0.0)
            if abs(lq - bq) <= SHARE_EPSILON:
                continue
            if lq > 0 and bq == 0:
                kind = "phantom_position"   # we think we hold it; broker disagrees
            elif lq == 0 and bq > 0:
                kind = "unrecorded_position"  # broker holds it; we never booked it
            else:
                kind = "quantity_mismatch"
            divergences.append(Divergence(symbol, lq, bq, kind))

        result = ReconciliationResult(
            as_of=datetime.now(UTC),
            divergences=tuple(divergences),
            ledger_equity=self.equity(prices),
            broker_equity=broker_equity,
            equity_tolerance=equity_tolerance,
        )
        if result.clean:
            log.info("ledger.reconciled", positions=len(ledger_qty))
        else:
            log.error("ledger.divergence", detail=result.describe())
        return result


@dataclass
class DailyPnL:
    """Daily marks. Feeds the breakers and the dashboard.

    `session_date` is what makes "the day" a real boundary rather than an implicit
    one. Without it a caller has no way to ask "is this a new session?", and the
    only available answer -- wall-clock today -- is wrong in every replay.
    """

    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    history: list[tuple[str, float]] = field(default_factory=list)
    session_date: str = ""
    last_close_equity: float = 0.0

    def roll_day(self, equity: float, *, as_of: date | None = None) -> None:
        """Open a new session. Closes out the previous one first.

        `day_start_equity` is set to the PREVIOUS session's closing equity, not to
        equity at this moment. That distinction is the whole point: this strategy
        trades once at the open, so equity barely moves within a session and
        "P&L since the open" would measure little beyond slippage. The real daily
        move is the overnight gap, and it only appears if the reference point is
        the prior close -- which is also what a broker statement shows.

        It matters for safety, not just reporting: the daily-loss breaker is
        evaluated against this number, so anchoring it intra-session would leave
        a -5% gap down reading as roughly 0% and the breaker would never fire.

        `as_of` is explicit rather than wall-clock for the same reason
        `record_equity` takes it: during a replay every session would otherwise
        stamp "today", collapsing the history to one entry while appearing to work.
        """
        today = (as_of or datetime.now(UTC).date()).isoformat()

        # Only close out a session that was genuinely open. A seeded
        # `day_start_equity` with no `session_date` is a fresh account, and
        # recording a 0.00 P&L for a session that never traded would put a
        # fabricated entry at the head of the history.
        if self.session_date:
            prior_close = (
                self.last_close_equity if self.last_close_equity > 0 else equity
            )
            pnl = prior_close - self.day_start_equity
            # The streak that trips a breaker, counted on the session boundary.
            # This is why `roll_day` must run every session, not only the first.
            self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
            self.history.append((self.session_date, pnl))
            self.day_start_equity = prior_close
        else:
            self.day_start_equity = equity

        self.session_date = today
        self.peak_equity = max(self.peak_equity, equity)

    def close_session(self, equity: float) -> None:
        """Record the session's final equity. The next roll references it."""
        self.last_close_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def is_new_session(self, as_of: date | None = None) -> bool:
        """True when `as_of` is a different session than the one now open."""
        today = (as_of or datetime.now(UTC).date()).isoformat()
        return self.session_date != today

    def mark(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
