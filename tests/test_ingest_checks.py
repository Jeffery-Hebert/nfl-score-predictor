"""
The checks python -m src.ingest.pull_all runs on freshly pulled data.

nflverse publishes schedules, play-by-play, snap counts and injuries as separate
releases on different clocks. A pull that lands between them is inconsistent in
ways nothing downstream reports -- above all a scored game with no plays, whose
EPA then silently vanishes from every rolling feature. These pin each check on
hand-built frames.

Run: pytest tests/test_ingest_checks.py -v
"""

from datetime import date

import pandas as pd
import pytest

from src.ingest.pull_all import check
from src.ingest.seasons import current_season, load_seasons

NOW = pd.Timestamp("2026-09-24T18:00:00", tz="UTC")


def sched(*rows):
    """rows: (game_id, kickoff_utc, home_score, away_score)."""
    return pd.DataFrame(
        [
            {
                "game_id": g,
                "season": 2026,
                "week": 2 if hs is not None else 3,
                "gameday": pd.Timestamp(k).date().isoformat(),
                "kickoff": pd.Timestamp(k, tz="UTC"),
                "home_score": hs,
                "away_score": as_,
            }
            for g, k, hs, as_ in rows
        ]
    )


def finals(**games):
    return pd.DataFrame(
        [
            {"game_id": g, "total_home_score": h, "total_away_score": a}
            for g, (h, a) in games.items()
        ],
        columns=["game_id", "total_home_score", "total_away_score"],
    ).set_index("game_id")


INJ = pd.DataFrame({"season": [2026], "week": [3], "report_status": ["Out"]})


def test_a_clean_pull_passes():
    s = sched(
        ("g1", "2026-09-20T17:00", 24, 17), ("g3", "2026-09-27T17:00", None, None)
    )
    r = check(s, finals(g1=(24, 17)), {"g1"}, INJ, NOW)
    assert r["errors"] == [] and r["warnings"] == []
    assert r["info"]["games_cross_checked"] == 1


def test_a_scored_game_without_plays_is_an_error():
    """THE silent failure: its EPA would just be missing from every feature."""
    s = sched(("g1", "2026-09-20T17:00", 24, 17), ("g2", "2026-09-21T00:20", 20, 10))
    r = check(s, finals(g1=(24, 17)), {"g1", "g2"}, INJ, NOW)
    assert any("no play-by-play" in e and "g2" in e for e in r["errors"])


def test_a_score_disagreement_between_sources_is_an_error():
    s = sched(("g1", "2026-09-20T17:00", 24, 17))
    r = check(s, finals(g1=(24, 14)), {"g1"}, INJ, NOW)
    assert any("disagrees" in e for e in r["errors"])


def test_a_game_long_finished_without_a_score_is_an_error():
    s = sched(("g1", "2026-09-20T17:00", None, None))
    r = check(s, finals(), set(), INJ, NOW)
    assert any("no final score" in e for e in r["errors"])


def test_a_game_that_just_ended_is_not_yet_an_error():
    """nflverse posts finals within hours; a day of grace avoids false alarms."""
    s = sched(("g1", "2026-09-24T01:00", None, None))
    assert check(s, finals(), set(), INJ, NOW)["errors"] == []


def test_the_cancelled_2022_game_never_trips_it():
    s = sched(("2022_17_BUF_CIN", "2023-01-03T01:30", None, None))
    assert check(s, finals(), set(), INJ, NOW)["errors"] == []


def test_missing_snap_counts_warn_but_do_not_fail():
    s = sched(("g1", "2026-09-20T17:00", 24, 17))
    r = check(s, finals(g1=(24, 17)), set(), INJ, NOW)
    assert r["errors"] == [] and any("snap counts" in w for w in r["warnings"])


@pytest.mark.parametrize(
    "today, season",
    [
        (date(2026, 9, 24), 2026),
        (date(2026, 9, 9), 2026),  # the 2026 opener, a WEDNESDAY
        (date(2027, 1, 20), 2026),  # playoffs belong to the season before
        (date(2027, 3, 14), 2026),
        (date(2027, 3, 15), 2027),  # nflverse rollover: the new season is pulled
    ],
)
def test_the_season_rolls_over_on_march_15(today, season):
    """Not nflreadpy's rule (the Thursday after Labor Day): that would have left
    the 2026 season out of a pull made on its Wednesday opener."""
    assert current_season(today) == season


class TestLoadSeasons:
    def test_a_missing_newest_season_falls_back_one_year(self):
        calls = []

        def loader(seasons):
            calls.append(list(seasons))
            if max(seasons) >= date.today().year:
                raise ConnectionError("404: not published yet")
            return "ok"

        yr = date.today().year
        assert load_seasons(loader, [yr - 2, yr - 1, yr]) == "ok"
        assert calls == [[yr - 2, yr - 1, yr], [yr - 2, yr - 1]]

    def test_a_failure_on_an_old_season_is_not_swallowed(self):
        def loader(seasons):
            raise ConnectionError("network down")

        with pytest.raises(ConnectionError):
            load_seasons(loader, [2019, 2020])
