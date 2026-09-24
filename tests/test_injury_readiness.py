"""
The rule that decides whether a game may be forecast yet: its FINAL injury
report must be published and in the data.

Why it matters. injury_impact is built from report_status, which teams only set
in the final report before each game. A forecast made earlier sees an almost
empty report, silently treats every team as healthy, and applies the model to
inputs the backtest never contained. These tests pin the calendar, the
pull-time rule and the slate-coverage rule with hand-built cases.

Run: pytest tests/test_injury_readiness.py -v
"""

import pandas as pd
import pytest

from src.predict import injury_readiness as ir

ET = "US/Eastern"


def utc(ts_et: str) -> pd.Timestamp:
    return pd.Timestamp(ts_et, tz=ET).tz_convert("UTC")


class TestFinalReportDue:
    @pytest.mark.parametrize(
        "kickoff_et, due_et",
        [
            ("2026-09-24 20:15", "2026-09-23 16:00"),  # Thursday -> Wednesday
            ("2026-09-27 13:00", "2026-09-25 16:00"),  # Sunday -> Friday
            ("2026-09-27 20:20", "2026-09-25 16:00"),  # Sunday night -> Friday
            ("2026-09-28 20:15", "2026-09-26 16:00"),  # Monday -> Saturday
            ("2026-12-19 16:30", "2026-12-17 16:00"),  # Saturday -> Thursday
            ("2026-09-27 09:30", "2026-09-25 16:00"),  # London morning -> Friday
        ],
    )
    def test_the_nfl_reporting_calendar(self, kickoff_et, due_et):
        assert ir.final_report_due(utc(kickoff_et)) == utc(due_et)


def games(*rows):
    """rows: (game_id, home, away, kickoff_et)."""
    return pd.DataFrame(
        [
            {
                "game_id": g,
                "season": 2026,
                "week": 3,
                "home_team": h,
                "away_team": a,
                "kickoff": utc(k),
            }
            for g, h, a, k in rows
        ]
    )


def injuries(statuses: dict[str, int], extra_rows: int = 5):
    """statuses: team -> number of rows WITH a report_status. Every team also
    gets rows without one (practice participation only)."""
    recs = []
    for team, n in statuses.items():
        for i in range(n + extra_rows):
            recs.append(
                {
                    "season": 2026,
                    "week": 3,
                    "team": team,
                    "gsis_id": f"{team}{i}",
                    "report_status": "Out" if i < n else None,
                }
            )
    return pd.DataFrame(recs)


SUNDAY = games(
    ("g1", "BUF", "LAC", "2026-09-27 13:00"),
    ("g2", "CLE", "CAR", "2026-09-27 13:00"),
    ("g3", "DET", "NYJ", "2026-09-27 13:00"),
)


class TestAssess:
    def test_pulled_before_the_report_was_due_is_not_final(self):
        """THE case of 2026 week 3: pulled Thursday, Sunday report due Friday."""
        inj = injuries({"BUF": 3, "LAC": 2, "CLE": 4, "CAR": 1, "DET": 2, "NYJ": 3})
        got = ir.assess(SUNDAY, inj, pulled_at=utc("2026-09-24 17:28"))
        assert not got["injury_report_final"].any()
        assert got["injury_note"].str.contains("pull again after it").all()

    def test_pulled_after_the_report_with_statuses_is_final(self):
        inj = injuries({"BUF": 3, "LAC": 2, "CLE": 4, "CAR": 1, "DET": 2, "NYJ": 3})
        got = ir.assess(SUNDAY, inj, pulled_at=utc("2026-09-27 07:00"))
        assert got["injury_report_final"].all()

    def test_a_report_not_yet_ingested_is_not_final(self):
        """Due time passed, but nflverse has no statuses for the slate yet --
        the calendar alone must not be trusted."""
        inj = injuries({"BUF": 0, "LAC": 0, "CLE": 0, "CAR": 0, "DET": 1, "NYJ": 0})
        got = ir.assess(SUNDAY, inj, pulled_at=utc("2026-09-25 17:00"))
        assert not got["injury_report_final"].any()
        assert got["injury_note"].str.contains("not ingested").all()

    def test_one_healthy_team_does_not_block_its_game(self):
        """2-4% of team-games legitimately designate nobody. That is a real
        'healthy', noted but not blocked."""
        inj = injuries({"BUF": 0, "LAC": 2, "CLE": 4, "CAR": 1, "DET": 2, "NYJ": 3})
        got = ir.assess(SUNDAY, inj, pulled_at=utc("2026-09-27 07:00")).set_index(
            "game_id"
        )
        assert got.loc["g1", "injury_report_final"]
        assert "BUF designated nobody" in got.loc["g1", "injury_note"]

    def test_the_thursday_game_is_ready_while_sunday_waits(self):
        """Different games in one week settle at different times."""
        g = games(
            ("thu", "GB", "ATL", "2026-09-24 20:15"),
            ("sun", "BUF", "LAC", "2026-09-27 13:00"),
        )
        inj = injuries({"GB": 6, "ATL": 2, "BUF": 0, "LAC": 0})
        got = ir.assess(g, inj, pulled_at=utc("2026-09-24 17:28")).set_index("game_id")
        assert got.loc["thu", "injury_report_final"]
        assert not got.loc["sun", "injury_report_final"]

    def test_no_pull_at_all_is_not_final(self):
        got = ir.assess(SUNDAY, injuries({"BUF": 3}), pulled_at=None)
        assert not got["injury_report_final"].any()

    def test_an_unknown_kickoff_is_not_final(self):
        g = SUNDAY.copy()
        g.loc[0, "kickoff"] = pd.NaT
        inj = injuries({"BUF": 3, "LAC": 2, "CLE": 4, "CAR": 1, "DET": 2, "NYJ": 3})
        got = ir.assess(g, inj, pulled_at=utc("2026-09-27 07:00")).set_index("game_id")
        assert not got.loc["g1", "injury_report_final"]
        assert got.loc["g2", "injury_report_final"]


def test_times_render_without_platform_specific_flags():
    assert ir.et(utc("2026-09-25 16:00")) == "Fri Sep 25 4:00pm ET"
    assert ir.et(utc("2026-09-27 09:30")) == "Sun Sep 27 9:30am ET"
