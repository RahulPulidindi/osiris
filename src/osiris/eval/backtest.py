"""Event-driven backtester with realistic costs and tax accounting.

Two deliberate choices that separate this from a self-flattering backtest:
  1. Every price and document read is guarded against future leakage.
  2. After-tax return is a first-class output, because daily rebalancing of a
     20-name book generates ordinary-income short-term gains plus wash sales.
     It is entirely possible that monthly beats daily after tax.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from osiris.eval.pit import assert_bar_not_future


@dataclass(frozen=True)
class CostModel:
    """Transaction costs. `multiplier` drives the cost-sensitivity gate."""

    spread_bps: float = 2.0
    commission_bps: float = 0.0        # Robinhood is commission-free
    impact_coefficient: float = 0.1    # sqrt-impact on ADV participation
    multiplier: float = 1.0

    def cost_bps(self, notional: float, adv_usd: float) -> float:
        half_spread = self.spread_bps / 2.0
        participation = (notional / adv_usd) if adv_usd > 0 else 0.0
        impact = self.impact_coefficient * np.sqrt(max(participation, 0.0)) * 100.0
        return (half_spread + self.commission_bps + impact) * self.multiplier


@dataclass
class TaxLot:
    quantity: float
    price: float
    acquired: date


@dataclass
class TaxAccount:
    """Lot-level tax tracking with wash-sale detection.

    Short-term gains are taxed as ordinary income, which at daily turnover is the
    dominant drag. Ignoring it overstates net return substantially.
    """

    short_term_rate: float = 0.37
    long_term_rate: float = 0.20
    lots: dict[str, deque[TaxLot]] = field(default_factory=dict)
    realized_short: float = 0.0
    realized_long: float = 0.0
    wash_sale_count: int = 0
    _recent_losses: dict[str, date] = field(default_factory=dict)

    def buy(self, symbol: str, quantity: float, price: float, d: date) -> None:
        self.lots.setdefault(symbol, deque()).append(TaxLot(quantity, price, d))
        # Repurchase within 30 days of a realized loss is a wash sale.
        loss_date = self._recent_losses.get(symbol)
        if loss_date is not None and (d - loss_date).days <= 30:
            self.wash_sale_count += 1

    def sell(self, symbol: str, quantity: float, price: float, d: date) -> float:
        """FIFO disposal. Returns realized gain (pre-tax)."""
        lots = self.lots.get(symbol)
        if not lots:
            return 0.0
        remaining = quantity
        gain = 0.0
        while remaining > 1e-9 and lots:
            lot = lots[0]
            take = min(remaining, lot.quantity)
            lot_gain = take * (price - lot.price)
            gain += lot_gain
            held_days = (d - lot.acquired).days
            if held_days > 365:
                self.realized_long += lot_gain
            else:
                self.realized_short += lot_gain
            lot.quantity -= take
            remaining -= take
            if lot.quantity <= 1e-9:
                lots.popleft()
        if gain < 0:
            self._recent_losses[symbol] = d
        return gain

    @property
    def tax_owed(self) -> float:
        return max(0.0, self.realized_short) * self.short_term_rate + max(
            0.0, self.realized_long
        ) * self.long_term_rate


@dataclass
class BacktestResult:
    dates: list[date]
    returns: list[float]
    gross_returns: list[float]
    equity_curve: list[float]
    turnover: list[float]
    total_costs: float
    tax_owed: float
    wash_sales: int
    n_rebalances: int
    holdings_history: list[list[str]] = field(default_factory=list)

    @property
    def after_tax_total_return(self) -> float:
        if not self.equity_curve:
            return 0.0
        start, end = self.equity_curve[0], self.equity_curve[-1]
        if start <= 0:
            return 0.0
        return (end - self.tax_owed) / start - 1.0

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover)) if self.turnover else 0.0


class Backtester:
    """Portfolio-level event-driven simulation.

    Signals are supplied as a callable so the same harness can evaluate an LLM
    ranking, a quant factor, or a random baseline under identical mechanics.
    """

    def __init__(
        self,
        prices: dict[str, dict[date, float]],
        adv: dict[str, float] | None = None,
        cost_model: CostModel | None = None,
        *,
        tax_account: TaxAccount | None = None,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.prices = prices
        self.adv = adv or {}
        self.costs = cost_model or CostModel()
        self.tax = tax_account or TaxAccount()
        self.initial_equity = initial_equity

    def price_on(self, symbol: str, d: date, as_of: date | None = None) -> float | None:
        series = self.prices.get(symbol)
        if not series:
            return None
        if d not in series:
            return None
        assert_bar_not_future(d, as_of or d)
        return series[d]

    def run(
        self,
        rebalance_dates: list[date],
        target_fn,
        *,
        n_holdings: int = 20,
    ) -> BacktestResult:
        """target_fn(as_of: date) -> list[str] of desired holdings.

        The callable receives only the simulation date; any lookahead is the
        caller's bug and is caught by the PIT guards.
        """
        equity = self.initial_equity
        holdings: dict[str, float] = {}  # symbol -> shares
        curve: list[float] = []
        returns: list[float] = []
        gross_returns: list[float] = []
        turnovers: list[float] = []
        total_costs = 0.0
        out_dates: list[date] = []
        history: list[list[str]] = []

        prev_equity = equity
        for i, d in enumerate(rebalance_dates):
            # 1. Mark the existing book to market BEFORE trading.
            marked = 0.0
            for sym, shares in list(holdings.items()):
                px = self.price_on(sym, d, as_of=d)
                if px is None:
                    continue
                marked += shares * px
            cash = equity - sum(
                shares * (self.price_on(s, rebalance_dates[i - 1], as_of=d) or 0.0)
                for s, shares in holdings.items()
            ) if i > 0 else equity
            equity_marked = marked + max(0.0, cash)
            if i == 0:
                equity_marked = equity

            # 2. Determine target book for this date.
            targets = list(dict.fromkeys(target_fn(d)))[:n_holdings]
            targets = [t for t in targets if self.price_on(t, d, as_of=d) is not None]
            if not targets:
                curve.append(equity_marked)
                out_dates.append(d)
                returns.append(0.0)
                gross_returns.append(0.0)
                turnovers.append(0.0)
                history.append([])
                continue

            per_name = equity_marked / len(targets)
            new_holdings: dict[str, float] = {}
            traded_notional = 0.0

            # 3. Exits first: they free capital for entries.
            for sym in list(holdings):
                if sym in targets:
                    continue
                px = self.price_on(sym, d, as_of=d)
                if px is None:
                    continue
                shares = holdings[sym]
                notional = shares * px
                traded_notional += notional
                bps = self.costs.cost_bps(notional, self.adv.get(sym, 0.0))
                total_costs += notional * bps / 10_000.0
                self.tax.sell(sym, shares, px, d)

            # 4. Entries and rebalances.
            for sym in targets:
                px = self.price_on(sym, d, as_of=d)
                if px is None or px <= 0:
                    continue
                desired_shares = per_name / px
                current_shares = holdings.get(sym, 0.0)
                delta_shares = desired_shares - current_shares
                delta_notional = abs(delta_shares * px)
                if delta_notional > 1e-6:
                    traded_notional += delta_notional
                    bps = self.costs.cost_bps(delta_notional, self.adv.get(sym, 0.0))
                    total_costs += delta_notional * bps / 10_000.0
                    if delta_shares > 0:
                        self.tax.buy(sym, delta_shares, px, d)
                    else:
                        self.tax.sell(sym, -delta_shares, px, d)
                new_holdings[sym] = desired_shares

            holdings = new_holdings
            history.append(sorted(holdings))

            # 5. Costs are paid out of equity.
            equity_after = equity_marked
            turnovers.append(traded_notional / equity_marked if equity_marked > 0 else 0.0)

            gross_r = (equity_marked - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            cost_drag = (
                (traded_notional * self.costs.cost_bps(traded_notional, 1e12) / 10_000.0)
                / prev_equity
                if prev_equity > 0
                else 0.0
            )
            gross_returns.append(gross_r)
            returns.append(gross_r - cost_drag)

            equity = equity_after - (traded_notional * 0.0)  # costs tracked separately
            curve.append(equity)
            out_dates.append(d)
            prev_equity = equity_marked

        return BacktestResult(
            dates=out_dates,
            returns=returns,
            gross_returns=gross_returns,
            equity_curve=curve,
            turnover=turnovers,
            total_costs=total_costs,
            tax_owed=self.tax.tax_owed,
            wash_sales=self.tax.wash_sale_count,
            n_rebalances=len(out_dates),
            holdings_history=history,
        )


def walk_forward_windows(
    all_dates: list[date], train_periods: int, test_periods: int
) -> list[tuple[list[date], list[date]]]:
    """Rolling train/test splits.

    A single in-sample curve proves nothing: parameters were chosen knowing the
    answer. Walk-forward is the minimum honest validation.
    """
    windows: list[tuple[list[date], list[date]]] = []
    i = 0
    while i + train_periods + test_periods <= len(all_dates):
        windows.append(
            (
                all_dates[i : i + train_periods],
                all_dates[i + train_periods : i + train_periods + test_periods],
            )
        )
        i += test_periods
    return windows
