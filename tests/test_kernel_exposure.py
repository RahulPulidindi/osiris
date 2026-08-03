"""Portfolio-level exposure gates.

These are the gates that stop the objective from being satisfied by leverage or
concentration rather than by selection skill.
"""

from __future__ import annotations

import pytest

from osiris.config import AccountType, RiskLimits
from osiris.kernel.kernel import RiskKernel
from osiris.types import Side, VetoCode
from tests.conftest import (
    diversified_positions,
    make_intent,
    make_portfolio,
    make_position,
    make_state,
)


class TestNotionalCap:
    def test_oversized_trade_vetoed(self, kernel) -> None:
        # 2% of 100k = 2,000 cap
        d = kernel.evaluate(make_intent(notional=5_000), make_state())
        assert VetoCode.NOTIONAL_CAP in d.vetoes

    def test_at_cap_allowed(self, kernel) -> None:
        d = kernel.evaluate(make_intent(notional=2_000), make_state())
        assert VetoCode.NOTIONAL_CAP not in d.vetoes

    def test_zero_equity_vetoed(self, kernel) -> None:
        state = make_state(make_portfolio(equity=0.0, buying_power=0.0))
        d = kernel.evaluate(make_intent(), state)
        assert not d.approved


class TestSymbolWeight:
    def test_concentration_blocked(self, kernel) -> None:
        """Already at 9.5%; another 2% would breach the 10% cap."""
        positions = (make_position("AAPL", 9_500.0),) + diversified_positions(19)
        state = make_state(make_portfolio(equity=100_000.0, positions=positions))
        d = kernel.evaluate(make_intent("AAPL", notional=2_000), state)
        assert VetoCode.SYMBOL_WEIGHT_CAP in d.vetoes

    def test_room_remaining_allowed(self, kernel) -> None:
        positions = (make_position("AAPL", 2_000.0),) + diversified_positions(19)
        state = make_state(make_portfolio(equity=100_000.0, positions=positions))
        d = kernel.evaluate(make_intent("AAPL", notional=2_000), state)
        assert VetoCode.SYMBOL_WEIGHT_CAP not in d.vetoes

    def test_exit_exempt(self, kernel) -> None:
        positions = (make_position("AAPL", 20_000.0),) + diversified_positions(19)
        state = make_state(make_portfolio(equity=100_000.0, positions=positions))
        d = kernel.evaluate(make_intent("AAPL", Side.SELL, reason="rank_exit"), state)
        assert VetoCode.SYMBOL_WEIGHT_CAP not in d.vetoes


class TestSectorConcentration:
    def test_absolute_sector_cap_enforced(self, kernel) -> None:
        """Three tech names at 8% each = 24%; +2% breaches the 25% cap."""
        positions = (
            make_position("AAPL", 8_000.0, sector="Technology"),
            make_position("MSFT", 8_000.0, sector="Technology"),
            make_position("NVDA", 8_000.0, sector="Technology"),
        ) + diversified_positions(17)
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions), symbols=("AAPL",)
        )
        d = kernel.evaluate(make_intent("AAPL", notional=2_000), state)
        assert VetoCode.SECTOR_WEIGHT_CAP in d.vetoes

    def test_deviation_from_benchmark_enforced(self, kernel) -> None:
        """Benchmark tech is 5%; 20% held is a 15% overweight vs a 10% band.

        This is the gate that catches a thematic run turning a 20-name book into
        one undiversified bet that still looks diversified by position count.
        """
        positions = (
            make_position("AAPL", 10_000.0, sector="Technology"),
            make_position("MSFT", 10_000.0, sector="Technology"),
        ) + diversified_positions(18)
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            benchmark={"Technology": 0.05, "Financials": 0.15},
        )
        d = kernel.evaluate(make_intent("AAPL", notional=1_000), state)
        assert VetoCode.SECTOR_DEVIATION in d.vetoes

    def test_within_band_allowed(self, kernel) -> None:
        """Tech at 5% against a 30% benchmark weight is an underweight."""
        positions = (make_position("AAPL", 5_000.0, sector="Technology"),) + tuple(
            make_position(f"O{i}", 4_000.0, sector="Other") for i in range(19)
        )
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            symbols=("AAPL",),
            benchmark={"Technology": 0.30, "Other": 0.70},
        )
        d = kernel.evaluate(make_intent("AAPL", notional=1_000), state)
        assert VetoCode.SECTOR_DEVIATION not in d.vetoes

    def test_no_benchmark_skips_deviation_gate(self, kernel) -> None:
        state = make_state(benchmark={})
        d = kernel.evaluate(make_intent(notional=1_000), state)
        assert VetoCode.SECTOR_DEVIATION not in d.vetoes

    def test_unknown_sector_abstains_rather_than_vetoing(self, kernel) -> None:
        """Missing sector data is a data problem, not a concentration.

        Regression from live: the fundamentals feed returned no sector for JPM,
        UNH, and V. Every unclassified name landed in one "Unknown" bucket that
        the benchmark weights at 0.0, so on a small book every single buy read
        as a massive sector overweight and the agent could not open ANY
        position. Both sector gates must abstain on "Unknown" -- the single-name
        cap still bounds the position.
        """
        positions = (make_position("MYST1", 30_000.0, sector="Unknown"),)
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            symbols=("MYST2",),
            benchmark={"Technology": 0.30},
        )
        # Force the intent's symbol to resolve as Unknown.
        state.sectors.pop("MYST2", None)

        d = kernel.evaluate(make_intent("MYST2", notional=1_500), state)

        assert VetoCode.SECTOR_DEVIATION not in d.vetoes
        assert VetoCode.SECTOR_WEIGHT_CAP not in d.vetoes


class TestBetaBudget:
    def test_high_beta_book_blocked(self, kernel) -> None:
        """The most important gate: prevents leverage in disguise.

        A book of 1.8-beta names is a leveraged market bet, not stock selection.
        """
        positions = tuple(
            make_position(f"HB{i}", 5_000.0, sector="Technology", beta=1.8)
            for i in range(20)
        )
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            symbols=("NVDA",),
            betas={**{f"HB{i}": 1.8 for i in range(20)}, "NVDA": 2.0},
            benchmark={},
        )
        d = kernel.evaluate(make_intent("NVDA", notional=2_000), state)
        assert VetoCode.BETA_BUDGET in d.vetoes

    def test_low_beta_allowed(self, kernel) -> None:
        positions = tuple(
            make_position(f"LB{i}", 4_500.0, sector="Staples", beta=0.6)
            for i in range(20)
        )
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            symbols=("PG",),
            betas={**{f"LB{i}": 0.6 for i in range(20)}, "PG": 0.5},
            benchmark={},
        )
        d = kernel.evaluate(make_intent("PG", notional=2_000), state)
        assert VetoCode.BETA_BUDGET not in d.vetoes

    def test_exit_exempt_from_beta_gate(self, kernel) -> None:
        positions = tuple(
            make_position(f"HB{i}", 5_000.0, beta=1.9) for i in range(20)
        )
        state = make_state(
            make_portfolio(equity=100_000.0, positions=positions),
            betas={f"HB{i}": 1.9 for i in range(20)},
        )
        d = kernel.evaluate(make_intent("HB0", Side.SELL, reason="rank_exit"), state)
        assert VetoCode.BETA_BUDGET not in d.vetoes


class TestPositionFloor:
    def test_exit_below_floor_blocked(self, kernel) -> None:
        """Breadth is the edge; drifting to 14 names is not risk management."""
        book = diversified_positions(15)
        state = make_state(make_portfolio(positions=book))
        d = kernel.evaluate(make_intent(book[0].symbol, Side.SELL, reason="rank_exit"), state)
        assert VetoCode.POSITION_FLOOR in d.vetoes

    def test_risk_exit_bypasses_floor(self, kernel) -> None:
        """Honoring a stop matters more than holding a target count."""
        book = diversified_positions(15)
        state = make_state(make_portfolio(positions=book))
        d = kernel.evaluate(make_intent(book[0].symbol, Side.SELL, reason="risk_exit"), state)
        assert VetoCode.POSITION_FLOOR not in d.vetoes
        assert d.approved, d.notes

    def test_invalidation_exit_bypasses_floor(self, kernel) -> None:
        book = diversified_positions(15)
        state = make_state(make_portfolio(positions=book))
        d = kernel.evaluate(
            make_intent(book[0].symbol, Side.SELL, reason="invalidation_exit"), state
        )
        assert VetoCode.POSITION_FLOOR not in d.vetoes

    def test_exit_above_floor_allowed(self, kernel) -> None:
        book = diversified_positions(20)
        state = make_state(make_portfolio(positions=book))
        d = kernel.evaluate(make_intent(book[0].symbol, Side.SELL, reason="rank_exit"), state)
        assert d.approved, d.notes


class TestADVParticipation:
    def test_illiquid_name_blocked(self, kernel) -> None:
        state = make_state(adv=50_000.0)  # 1% = $500 cap
        d = kernel.evaluate(make_intent(notional=1_500), state)
        assert VetoCode.ADV_PARTICIPATION in d.vetoes

    def test_liquid_name_allowed(self, kernel) -> None:
        state = make_state(adv=500_000_000.0)
        d = kernel.evaluate(make_intent(notional=1_500), state)
        assert VetoCode.ADV_PARTICIPATION not in d.vetoes

    def test_missing_adv_fails_closed(self, kernel) -> None:
        state = make_state()
        object.__setattr__(state, "adv", {})
        d = kernel.evaluate(make_intent(), state)
        assert VetoCode.ADV_PARTICIPATION in d.vetoes


class TestBuyingPowerAndSettlement:
    def test_insufficient_buying_power_blocked(self, kernel) -> None:
        state = make_state(make_portfolio(equity=100_000.0, buying_power=500.0))
        d = kernel.evaluate(make_intent(notional=1_500), state)
        assert VetoCode.INSUFFICIENT_BUYING_POWER in d.vetoes

    def test_cash_account_blocks_unsettled_use(self, cash_kernel) -> None:
        """Good-faith violations restrict the account, so the kernel refuses."""
        state = make_state(
            make_portfolio(equity=100_000.0, buying_power=2_000.0),
            unsettled_cash=1_800.0,
        )
        d = cash_kernel.evaluate(make_intent(notional=1_500), state)
        assert VetoCode.UNSETTLED_FUNDS in d.vetoes

    def test_margin_account_permits_unsettled_use(self, kernel) -> None:
        state = make_state(
            make_portfolio(equity=100_000.0, buying_power=2_000.0),
            unsettled_cash=1_800.0,
        )
        d = kernel.evaluate(make_intent(notional=1_500), state)
        assert VetoCode.UNSETTLED_FUNDS not in d.vetoes
