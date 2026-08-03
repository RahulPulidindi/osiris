"""Point-in-time universe tests. The survivorship-bias guard.

If these pass, a backtest cannot silently use today's winners to explain the past.
"""

from __future__ import annotations

from datetime import date

import pytest

from osiris.eval.pit import (
    LookaheadError,
    MembershipSpell,
    PITUniverse,
    assert_bar_not_future,
    assert_text_not_future,
)


def sample_universe() -> PITUniverse:
    return PITUniverse(
        [
            # Still a member
            MembershipSpell("AAPL", date(2015, 1, 1), None),
            MembershipSpell("MSFT", date(2015, 1, 1), None),
            # Removed: the kind of name survivorship bias deletes
            MembershipSpell("ENRN", date(2015, 1, 1), date(2019, 6, 30)),
            MembershipSpell("SEARS", date(2015, 1, 1), date(2018, 10, 15)),
            # Re-entry: left and came back
            MembershipSpell("TWTR", date(2015, 1, 1), date(2017, 6, 1), 1),
            MembershipSpell("TWTR", date(2019, 1, 1), date(2022, 10, 27), 2),
        ]
    )


class TestMembership:
    def test_includes_delisted_names_historically(self) -> None:
        """The whole point: failures must appear in the historical universe."""
        u = sample_universe()
        members_2016 = u.members_on(date(2016, 6, 1))
        assert "SEARS" in members_2016
        assert "ENRN" in members_2016

    def test_excludes_names_after_removal(self) -> None:
        u = sample_universe()
        assert "SEARS" not in u.members_on(date(2019, 1, 1))
        assert "ENRN" not in u.members_on(date(2020, 1, 1))

    def test_handles_reentry(self) -> None:
        u = sample_universe()
        assert "TWTR" in u.members_on(date(2016, 1, 1))   # first spell
        assert "TWTR" not in u.members_on(date(2018, 1, 1))  # gap
        assert "TWTR" in u.members_on(date(2020, 1, 1))   # second spell

    def test_all_symbols_ever_includes_failures(self) -> None:
        u = sample_universe()
        assert set(u.all_symbols_ever) >= {"AAPL", "MSFT", "ENRN", "SEARS", "TWTR"}

    def test_before_coverage_raises(self) -> None:
        """Silently returning today's list before coverage is the bug to prevent."""
        u = sample_universe()
        with pytest.raises(LookaheadError, match="survivorship"):
            u.members_on(date(2010, 1, 1))

    def test_was_member_query(self) -> None:
        u = sample_universe()
        assert u.was_member("SEARS", date(2016, 1, 1))
        assert not u.was_member("SEARS", date(2020, 1, 1))

    def test_boundary_dates_inclusive(self) -> None:
        u = sample_universe()
        assert u.was_member("SEARS", date(2018, 10, 15))  # last day
        assert not u.was_member("SEARS", date(2018, 10, 16))


class TestSurvivorshipDetection:
    def test_flags_a_retroactive_list(self) -> None:
        """A universe with no removals at all is almost certainly today's list."""
        spells = [
            MembershipSpell(f"S{i}", date(2015, 1, 1), None) for i in range(60)
        ]
        u = PITUniverse(spells)
        with pytest.raises(LookaheadError, match="survivorship bias"):
            u.assert_no_survivorship(date(2016, 1, 1))

    def test_accepts_a_real_pit_table(self) -> None:
        u = sample_universe()
        u.assert_no_survivorship(date(2016, 6, 1))  # should not raise


class TestLookaheadGuards:
    def test_future_bar_rejected(self) -> None:
        with pytest.raises(LookaheadError, match="future leak"):
            assert_bar_not_future(date(2026, 8, 5), as_of=date(2026, 8, 1))

    def test_same_day_bar_allowed(self) -> None:
        assert_bar_not_future(date(2026, 8, 1), as_of=date(2026, 8, 1))

    def test_future_document_rejected(self) -> None:
        """Exa's date bounding makes this enforceable rather than aspirational."""
        from datetime import UTC, datetime

        with pytest.raises(LookaheadError, match="future leak"):
            assert_text_not_future(
                datetime(2026, 8, 10, tzinfo=UTC), as_of=date(2026, 8, 1)
            )

    def test_none_published_date_tolerated(self) -> None:
        assert_text_not_future(None, as_of=date(2026, 8, 1))


class TestCSVLoading:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "members.csv"
        path.write_text(
            "symbol,start,end,membership_num\n"
            "AAPL,2015-01-01,,1\n"
            "SEARS,2015-01-01,2018-10-15,1\n"
        )
        u = PITUniverse.from_csv(path)
        assert "SEARS" in u.members_on(date(2016, 1, 1))
        assert "SEARS" not in u.members_on(date(2019, 1, 1))

    def test_empty_file_rejected(self, tmp_path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("symbol,start,end\n")
        with pytest.raises(ValueError, match="No membership spells"):
            PITUniverse.from_csv(path)
