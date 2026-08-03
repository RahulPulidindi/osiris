"""Typed configuration. Single source of truth for every tunable.

Risk limits live here rather than in the cognition layer on purpose: the model
must not be able to influence its own constraints.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/osiris/config.py -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("OSIRIS_DATA_DIR", REPO_ROOT / "data"))
# Overridable so a containerized deployment can put the switch on the persistent
# volume: a kill switch that vanishes when the container is replaced is not a
# kill switch.
KILL_SWITCH_PATH = Path(
    os.environ.get("OSIRIS_KILL_SWITCH_PATH", REPO_ROOT / "KILL_SWITCH")
)

# Load .env into the process environment at import time.
#
# Pydantic's `env_file` covers only the `OSIRIS_*` settings it declares. Third-
# party credentials (OPENROUTER_API_KEY, EXA_API_KEY) are read via plain
# `os.environ` and were therefore invisible: `.env` held the keys while the agent
# reported them unset and silently ran without research.
#
# Done here because `config` is imported before anything that needs a key, and
# `override=False` so a real shell variable still wins over the file.
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


_load_env()


class AccountType(str, Enum):
    CASH = "cash"
    MARGIN = "margin"
    UNKNOWN = "unknown"


class Mode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class RiskLimits(BaseSettings):
    """Deterministic bounds. Enforced by the kernel, invisible to the model.

    Every limit is settable from the environment, including to values that are
    effectively unlimited (`1.0` for a fraction, a large integer for a count).
    Bounds here are the *schema's* outer edge, not a policy recommendation.

    What is NOT configurable, by design: mandatory pre-trade simulation and the
    kill switch. Those are not caps -- they are what makes a rejected order
    distinguishable from a filled one. Disabling them does not grant freedom, it
    removes the system's ability to know what it owns.
    """

    model_config = SettingsConfigDict(env_prefix="OSIRIS_", extra="ignore")

    max_trade_notional_pct: float = Field(0.02, gt=0, le=1.0)
    max_symbol_weight: float = Field(0.10, gt=0, le=1.0)
    max_sector_weight: float = Field(0.25, gt=0, le=1.0)
    max_sector_deviation: float = Field(0.10, gt=0, le=1.0)
    max_portfolio_beta: float = Field(1.15, gt=0, le=10.0)
    min_position_count: int = Field(15, ge=1)
    target_position_count: int = Field(20, ge=1)
    max_adv_participation: float = Field(0.01, gt=0, le=1.0)
    max_spread_bps: float = Field(25.0, gt=0)
    daily_order_budget: int = Field(60, ge=1)
    quote_staleness_seconds: int = Field(300, ge=1)
    earnings_blackout_hours: int = Field(48, ge=0)

    # Circuit breakers. Set to 1.0 to disable a halt entirely.
    daily_loss_halt_pct: float = Field(0.03, gt=0, le=1.0)
    max_drawdown_halt_pct: float = Field(0.10, gt=0, le=1.0)
    consecutive_loss_halt: int = Field(5, ge=1)

    # Position stop-loss, enforced continuously by the guardian while the
    # market is open (and again by the daily cycle). Fraction below avg cost.
    stop_loss_pct: float = Field(0.15, gt=0, le=1.0)
    # Seconds between guardian ticks during market hours. 60s is deliberate:
    # stops on a daily-ranked book do not need sub-minute reaction, and each
    # tick costs one quotes call for the held names only.
    watch_interval_seconds: int = Field(60, ge=10, le=3600)

    @model_validator(mode="after")
    def _coherent(self) -> RiskLimits:
        if self.min_position_count > self.target_position_count:
            raise ValueError("min_position_count cannot exceed target_position_count")
        # A book of N names each capped at max_symbol_weight must be able to
        # reach 100% invested, else the kernel deadlocks against itself: it would
        # demand full investment while forbidding any order that achieves it.
        if self.max_symbol_weight * self.target_position_count < 1.0:
            raise ValueError(
                f"max_symbol_weight {self.max_symbol_weight} x target_position_count "
                f"{self.target_position_count} < 1.0: book can never be fully invested"
            )
        if self.daily_loss_halt_pct > self.max_drawdown_halt_pct:
            raise ValueError("daily loss halt should not exceed max drawdown halt")
        return self

    @classmethod
    def for_equity(cls, equity_usd: float, **overrides) -> RiskLimits:
        """Limits scaled so the agent can actually operate at this account size.

        The defaults assume a five-figure account. At $366 they are not
        conservative, they are *infeasible*: a 2% per-order cap is $7.32, a 20-name
        book means $18 positions, and building it takes ~50 orders against a
        60-order daily budget. The kernel would veto nearly everything and the
        result would look like a broken agent rather than a misconfigured one.

        So below a threshold the book concentrates: fewer names, larger per-order
        size. That is a genuine risk increase -- less diversification, more
        single-name exposure -- and it is the honest tradeoff. A $366 account
        cannot be both diversified across 20 names and meaningfully invested in
        any of them.

        Above the threshold the defaults are returned unchanged.
        """
        if equity_usd >= 10_000:
            return cls(**overrides)

        if equity_usd >= 2_000:
            # Room for a real book, but not 20 names.
            preset = {
                "max_trade_notional_pct": 0.10,
                "max_symbol_weight": 0.20,
                "target_position_count": 8,
                "min_position_count": 4,
                "max_sector_weight": 0.50,
                "max_sector_deviation": 0.40,
            }
        else:
            # Very small. Concentration is unavoidable: at $366 a 5-name book is
            # $73 per position, which is about the smallest that survives being
            # rebalanced at all.
            #
            # Sector limits are DISABLED here (1.0), not merely loosened. On a
            # five-name book, two names in one industry is already 40-70% "sector
            # concentration" -- the gate would veto routine picks while adding
            # nothing a 35% single-name cap and the stop-loss do not already
            # provide. Sector diversification is a statistical property of books
            # with dozens of names; pretending a $100 account has it only stops
            # the agent from trading at all.
            preset = {
                "max_trade_notional_pct": 0.25,
                "max_symbol_weight": 0.35,
                "target_position_count": 5,
                "min_position_count": 2,
                "max_sector_weight": 1.0,
                "max_sector_deviation": 1.0,
                # A tiny account rebalancing 20 times a day is pure cost.
                "daily_order_budget": 15,
            }
        preset.update(overrides)
        return cls(**preset)

    @classmethod
    def unrestricted(cls) -> RiskLimits:
        """Caps effectively removed. Concentration and drawdown are unbounded.

        Provided because it was asked for, and because a documented escape hatch
        is safer than someone editing gate code to get the same effect. Read the
        numbers before using it: a single name may become 100% of the account, and
        no loss level halts trading.

        Simulation-before-place and the kill switch still apply. They are not
        caps; they are how the ledger stays truthful.
        """
        return cls(
            max_trade_notional_pct=1.0,
            max_symbol_weight=1.0,
            max_sector_weight=1.0,
            max_sector_deviation=1.0,
            max_portfolio_beta=10.0,
            min_position_count=1,
            target_position_count=20,
            max_adv_participation=1.0,
            max_spread_bps=10_000.0,
            daily_order_budget=1_000,
            earnings_blackout_hours=0,
            daily_loss_halt_pct=1.0,
            max_drawdown_halt_pct=1.0,
            consecutive_loss_halt=10_000,
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OSIRIS_", extra="ignore"
    )

    mode: Mode = Mode.PAPER
    i_understand_the_risk: str = "no"

    account_type: AccountType = AccountType.UNKNOWN
    account_equity_usd: float = Field(0.0, ge=0)

    mcp_endpoint: str = "https://agent.robinhood.com/mcp/trading"
    llm_daily_usd_ceiling: float = Field(15.0, gt=0)

    rebalance_frequency: str = "daily"  # daily | weekly | monthly

    # Funnel stage widths
    funnel_prerank_width: int = Field(150, ge=1)
    funnel_deep_width: int = Field(40, ge=1)

    @property
    def live_armed(self) -> bool:
        """The live order path requires two independent affirmations.

        Mode alone is not enough: a stray env var should not be able to arm
        real money.
        """
        return self.mode is Mode.LIVE and self.i_understand_the_risk.lower() == "yes"

    @model_validator(mode="after")
    def _guard_live(self) -> Settings:
        if self.mode is Mode.LIVE:
            if self.account_type is AccountType.UNKNOWN:
                raise ValueError(
                    "Refusing live mode with unknown account type. Complete Phase 0 "
                    "(docs/PHASE0.md) first: cash vs margin governs sizing."
                )
            if self.account_equity_usd <= 0:
                raise ValueError("Refusing live mode with zero recorded equity")
        return self


def load_settings() -> Settings:
    return Settings()


def load_risk_limits() -> RiskLimits:
    return RiskLimits()
