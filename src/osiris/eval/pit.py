"""Point-in-time universe. The survivorship-bias guard.

Backtesting on today's index constituents makes results fiction: today's members
are by construction the survivors, and the failures were removed. The 2026
literature named this exactly -- "Survivorship Bias, Not Skill."

Every universe query must therefore be answered as of a date, never as of now.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class MembershipSpell:
    """A continuous period during which a symbol was an index member."""

    symbol: str
    start: date
    end: date | None  # None = still a member
    membership_num: int = 1

    def covers(self, d: date) -> bool:
        if d < self.start:
            return False
        return self.end is None or d <= self.end


class LookaheadError(AssertionError):
    """Raised when a backtest would use information not yet available."""


class PITUniverse:
    """Point-in-time index membership.

    Symbols may enter, exit, and re-enter, so membership is a list of spells per
    symbol rather than a single interval.
    """

    def __init__(self, spells: list[MembershipSpell]) -> None:
        self._spells = spells
        self._by_symbol: dict[str, list[MembershipSpell]] = {}
        for s in spells:
            self._by_symbol.setdefault(s.symbol, []).append(s)
        self._earliest = min((s.start for s in spells), default=None)

    @property
    def all_symbols_ever(self) -> list[str]:
        """Every symbol that was EVER a member, including the failures."""
        return sorted(self._by_symbol)

    @property
    def coverage_start(self) -> date | None:
        return self._earliest

    def members_on(self, d: date) -> list[str]:
        """Constituents as of date `d`. This is the only correct universe query."""
        if self._earliest and d < self._earliest:
            raise LookaheadError(
                f"Requested membership for {d}, but PIT coverage starts "
                f"{self._earliest}. Using today's list would inject survivorship bias."
            )
        return sorted(
            sym
            for sym, spells in self._by_symbol.items()
            if any(sp.covers(d) for sp in spells)
        )

    def was_member(self, symbol: str, d: date) -> bool:
        return any(sp.covers(d) for sp in self._by_symbol.get(symbol, []))

    def assert_no_survivorship(self, d: date) -> None:
        """Sanity check: a PIT universe must contain names now delisted.

        If every historical member is still a member today, the table is almost
        certainly today's list applied retroactively.
        """
        historical = set(self.members_on(d))
        current = set(self.members_on(max((s.end or d) for s in self._spells)))
        if historical and historical.issubset(current) and len(historical) > 50:
            raise LookaheadError(
                f"Universe on {d} is a strict subset of the latest universe. "
                "This is the signature of survivorship bias: no delisted or "
                "removed names present."
            )

    @classmethod
    def from_csv(cls, path: Path) -> PITUniverse:
        """Load `symbol,start,end[,membership_num]`. Empty end = current member."""
        spells: list[MembershipSpell] = []
        with Path(path).open(newline="") as fh:
            for row in csv.DictReader(fh):
                end_raw = (row.get("end") or "").strip()
                spells.append(
                    MembershipSpell(
                        symbol=row["symbol"].strip().upper(),
                        start=date.fromisoformat(row["start"].strip()),
                        end=date.fromisoformat(end_raw) if end_raw else None,
                        membership_num=int(row.get("membership_num") or 1),
                    )
                )
        if not spells:
            raise ValueError(f"No membership spells found in {path}")
        return cls(spells)


def assert_bar_not_future(bar_date: date, as_of: date) -> None:
    """Guard for every price read in a backtest."""
    if bar_date > as_of:
        raise LookaheadError(
            f"Bar dated {bar_date} read while simulating {as_of}: future leak."
        )


def assert_text_not_future(published, as_of: date) -> None:
    """Guard for every retrieved document in a backtest.

    Exa's startPublishedDate/endPublishedDate make this enforceable; without it a
    backtest silently reads tomorrow's news and flatters itself.
    """
    if published is None:
        return
    pub_date = published.date() if hasattr(published, "date") else published
    if pub_date > as_of:
        raise LookaheadError(
            f"Document published {pub_date} read while simulating {as_of}: future leak."
        )
