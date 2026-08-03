"""Target-book diffing: the mechanical exit logic.

This is where autonomous exits live. The property being proven is that the model
gets **no vote** on whether to keep a loser -- a name that drops out of the target
book is sold as arithmetic, not judgment.
"""

from __future__ import annotations

import pytest

from osiris.cognition.schemas import TargetHolding
from osiris.execution.rebalance import (
    ExitSignal,
    build_rebalance_plan,
    detect_invalidation_exits,
    diff_book,
)
from osiris.types import Side
from tests.conftest import make_portfolio, make_position

PRICES = {s: 100.0 for s in ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ")}


def targets(*pairs: tuple[str, float]) -> list[TargetHolding]:
    return [TargetHolding(symbol=s, target_weight=w) for s, w in pairs]


class TestDiffBook:
    def test_computes_the_three_sets(self):
        to_buy, to_sell, to_hold = diff_book({"A", "B"}, {"B", "C"})

        assert to_buy == {"C"}
        assert to_sell == {"A"}
        assert to_hold == {"B"}

    def test_empty_target_sells_everything(self):
        """A cleared ranking must liquidate, not silently hold."""
        to_buy, to_sell, to_hold = diff_book({"A", "B"}, set())

        assert to_sell == {"A", "B"}
        assert to_buy == set()

    def test_identical_books_produce_no_trades(self):
        to_buy, to_sell, _ = diff_book({"A"}, {"A"})

        assert not to_buy and not to_sell


class TestMechanicalExits:
    def test_dropping_out_of_the_ranking_forces_a_sell(self):
        """The core property: no opinion required to close a position."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio, targets(("MSFT", 0.05)), prices=PRICES
        )

        exits = [i for i in plan.exits if i.symbol == "AAPL"]
        assert len(exits) == 1
        assert exits[0].side is Side.SELL
        assert exits[0].reason == "rank_exit"

    def test_exit_notional_matches_the_full_position(self):
        """A partial exit would leave an unmanaged remainder."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 7_500.0),))
        plan = build_rebalance_plan(portfolio, [], prices=PRICES)

        assert plan.exits[0].notional_usd == 7_500.0

    def test_exits_are_ordered_before_entries(self):
        """Exits free buying power and reduce risk first."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio, targets(("MSFT", 0.05)), prices=PRICES
        )

        assert plan.all_intents[0].side is Side.SELL

    def test_risk_exit_overrides_the_ranking(self):
        """A stop fires even when the name is still top-ranked."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            targets(("AAPL", 0.05)),
            prices=PRICES,
            exit_signals=[ExitSignal("AAPL", "risk_exit", "stop breached")],
        )

        assert plan.exits[0].reason == "risk_exit"
        assert not plan.entries

    def test_risk_exits_use_market_orders(self):
        """Risk exits do not haggle over price."""
        from osiris.types import OrderKind

        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            [],
            prices=PRICES,
            exit_signals=[ExitSignal("AAPL", "risk_exit", "stop")],
        )

        assert plan.exits[0].kind is OrderKind.MARKET

    def test_every_exit_carries_an_invalidation_reason(self):
        """The kernel auto-vetoes intents with no stated condition."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(portfolio, [], prices=PRICES)

        assert plan.exits[0].invalidation


class TestEntriesAndSizing:
    def test_new_target_produces_an_entry(self):
        plan = build_rebalance_plan(
            make_portfolio(), targets(("AAPL", 0.05)), prices=PRICES
        )

        assert plan.entries[0].symbol == "AAPL"
        assert plan.entries[0].side is Side.BUY
        assert plan.entries[0].reason == "rank_entry"

    def test_symbol_weight_cap_is_respected(self):
        """The PM may not request more than the kernel would allow."""
        plan = build_rebalance_plan(
            make_portfolio(),
            targets(("AAPL", 0.90)),
            prices=PRICES,
            max_symbol_weight=0.10,
        )

        assert plan.entries[0].notional_usd <= 100_000.0 * 0.10 + 1e-6

    def test_higher_volatility_gets_a_smaller_allocation(self):
        """Each name should contribute comparable RISK, not comparable dollars.

        Without this the highest-vol holding silently dominates outcomes.

        Observed on the FINAL approach to target. The per-trade cap clamps any
        larger order to 2% of equity regardless of volatility, so vol-targeting is
        only visible once the remaining gap is smaller than that cap -- which is
        also the only point at which the difference in target size matters.
        """
        held = make_portfolio(positions=(make_position("AAPL", 1_400.0),))
        calm = build_rebalance_plan(
            held, targets(("AAPL", 0.10)), prices=PRICES, vols={"AAPL": 0.15}
        )
        wild = build_rebalance_plan(
            held, targets(("AAPL", 0.10)), prices=PRICES, vols={"AAPL": 1.60}
        )

        # The volatile name's remaining gap to target is smaller than the calm
        # one's, because its risk-adjusted target size is smaller.
        assert wild.adds[0].notional_usd < calm.adds[0].notional_usd

    def test_first_order_respects_the_per_trade_cap(self):
        """A 10% target is built across sessions, not in one 10% order.

        The per-trade cap and the symbol cap are different limits: the first
        bounds how much risk one order adds, the second bounds concentration.
        """
        plan = build_rebalance_plan(
            make_portfolio(),
            targets(("AAPL", 0.10)),
            prices=PRICES,
            vols={"AAPL": 0.20},
            max_symbol_weight=0.10,
            max_trade_notional_pct=0.02,
        )

        assert plan.entries[0].notional_usd == pytest.approx(100_000.0 * 0.02)

    def test_trims_are_not_clamped_by_the_per_trade_cap(self):
        """Reducing exposure is never the risk the per-trade cap guards."""
        held = make_portfolio(positions=(make_position("AAPL", 9_000.0),))
        plan = build_rebalance_plan(
            held,
            targets(("AAPL", 0.01)),
            prices=PRICES,
            vols={"AAPL": 0.20},
            max_trade_notional_pct=0.02,
        )

        assert plan.trims[0].notional_usd > 100_000.0 * 0.02

    def test_adv_participation_clamps_size(self):
        """Keeps the strategy viable as capital grows."""
        plan = build_rebalance_plan(
            make_portfolio(),
            targets(("AAPL", 0.10)),
            prices=PRICES,
            adv={"AAPL": 100_000.0},
            max_adv_participation=0.01,
        )

        assert plan.entries[0].notional_usd <= 1_000.0 + 1e-6

    def test_missing_price_is_skipped_not_guessed(self):
        plan = build_rebalance_plan(
            make_portfolio(), targets(("GHOST", 0.05)), prices=PRICES
        )

        assert not plan.entries
        assert plan.skipped["GHOST"] == "no price"

    def test_zero_equity_produces_no_trades(self):
        from datetime import UTC, datetime

        from osiris.types import Portfolio

        empty = Portfolio(equity=0.0, cash=0.0, buying_power=0.0, as_of=datetime.now(UTC))
        plan = build_rebalance_plan(empty, targets(("AAPL", 0.05)), prices=PRICES)

        assert not plan.all_intents


class TestRebalanceChurn:
    def test_trivial_weight_drift_is_not_traded(self):
        """The cost is certain; the benefit is noise."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            targets(("AAPL", 0.0501)),
            prices=PRICES,
            vols={"AAPL": 0.0},
        )

        assert not plan.adds and not plan.trims

    def test_material_underweight_produces_an_add(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 2_000.0),))
        plan = build_rebalance_plan(
            portfolio, targets(("AAPL", 0.08)), prices=PRICES, vols={"AAPL": 0.20}
        )

        assert plan.adds and plan.adds[0].side is Side.BUY

    def test_material_overweight_produces_a_trim(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 9_000.0),))
        plan = build_rebalance_plan(
            portfolio, targets(("AAPL", 0.02)), prices=PRICES, vols={"AAPL": 0.20}
        )

        assert plan.trims and plan.trims[0].side is Side.SELL


class TestStopDetection:
    def test_breaching_the_stop_emits_a_risk_exit(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 4_000.0),))
        signals = detect_invalidation_exits(
            portfolio,
            entry_prices={"AAPL": 100.0},
            prices={"AAPL": 80.0},
            stop_loss_pct=0.15,
        )

        assert signals[0].symbol == "AAPL"
        assert signals[0].reason == "risk_exit"

    def test_position_above_the_stop_is_left_alone(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 4_000.0),))
        signals = detect_invalidation_exits(
            portfolio,
            entry_prices={"AAPL": 100.0},
            prices={"AAPL": 95.0},
            stop_loss_pct=0.15,
        )

        assert signals == []

    def test_missing_entry_price_does_not_fabricate_a_stop(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 4_000.0),))
        signals = detect_invalidation_exits(
            portfolio, entry_prices={}, prices={"AAPL": 10.0}
        )

        assert signals == []


class TestStoppedNameIsNotReentered:
    """Regression guard for a subtle and expensive bug.

    A forced-exit name must be removed from BOTH sides of the diff. Removing it
    only from the held set leaves it in the target set, so the same cycle sells
    the stop and immediately buys it back -- paying the spread twice to end up in
    exactly the position the stop said not to hold.
    """

    def test_stopped_name_is_not_re_bought(self):
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            targets(("AAPL", 0.05), ("MSFT", 0.05)),
            prices=PRICES,
            exit_signals=[ExitSignal("AAPL", "risk_exit", "stop breached")],
        )

        assert [i.symbol for i in plan.exits] == ["AAPL"]
        assert "AAPL" not in [i.symbol for i in plan.entries]
        assert "AAPL" not in [i.symbol for i in plan.adds]
        assert "AAPL" not in [i.symbol for i in plan.trims]

    def test_only_one_intent_per_stopped_symbol(self):
        """Two intents for the same symbol would race in the kernel batch."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            targets(("AAPL", 0.05)),
            prices=PRICES,
            exit_signals=[ExitSignal("AAPL", "risk_exit", "stop")],
        )

        assert sum(1 for i in plan.all_intents if i.symbol == "AAPL") == 1

    def test_other_targets_still_enter_normally(self):
        """A stop on one name must not suppress the rest of the book."""
        portfolio = make_portfolio(positions=(make_position("AAPL", 5_000.0),))
        plan = build_rebalance_plan(
            portfolio,
            targets(("AAPL", 0.05), ("MSFT", 0.05), ("JPM", 0.05)),
            prices=PRICES,
            exit_signals=[ExitSignal("AAPL", "risk_exit", "stop")],
        )

        assert {i.symbol for i in plan.entries} == {"MSFT", "JPM"}
