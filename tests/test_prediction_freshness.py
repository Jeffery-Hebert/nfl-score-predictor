"""
The staleness guard on the live prediction path.

What it protects against: predicting Week N from a table that never received
Week N-1. That failure is silent -- the script runs, the numbers look
plausible, and the model simply does not know last week happened.

What it must NOT do is fire on a table that is complete. The guard used to be a
calendar test ("refuse if the newest training game is over 21 days old") and it
failed exactly that way, on the worst possible week: every Week 1 sits ~210
days after the previous Super Bowl, so a perfectly current table looked seven
months stale and the guard would have refused to predict opening weekend.

So the question is completeness, not elapsed time, and the two ways of being
incomplete need different fixes -- re-pull versus rebuild. These tests pin all
of it down, including the offseason case that used to break.

Run: pytest tests/test_prediction_freshness.py -v
"""

import pandas as pd
import pytest

from src.predict import predict_week


def schedule(rows):
    return pd.DataFrame(rows)


def game(gid, day, scored=True):
    return {
        "game_id": gid,
        "gameday": pd.Timestamp(day),
        "home_score": 24.0 if scored else None,
        "away_score": 17.0 if scored else None,
    }


@pytest.fixture
def sched_file(tmp_path, monkeypatch):
    """Point the guard at a schedules file the test controls."""

    def write(rows):
        path = tmp_path / "schedules.parquet"
        schedule(rows).to_parquet(path, index=False)
        monkeypatch.setattr(predict_week, "SCHEDULES", path)
        return path

    return write


class TestOffseasonGap:
    def test_week_one_passes_across_a_213_day_gap(self, sched_file):
        """The regression test for the bug this guard used to have. Last game
        played in February, kickoff in September, nothing missing in between --
        this must be allowed through."""
        rows = [game("2025_22_SB", "2026-02-08")]
        sched_file(rows)
        train = pd.DataFrame(
            {"game_id": ["2025_22_SB"], "gameday": [pd.Timestamp("2026-02-08")]}
        )
        gap = predict_week.assert_training_is_current(train, pd.Timestamp("2026-09-09"))
        assert gap == 213  # reported, not enforced

    def test_gap_is_returned_not_enforced(self, sched_file):
        sched_file([game("g1", "2020-01-01")])
        train = pd.DataFrame(
            {"game_id": ["g1"], "gameday": [pd.Timestamp("2020-01-01")]}
        )
        assert (
            predict_week.assert_training_is_current(train, pd.Timestamp("2020-12-31"))
            == 365
        )


class TestFeatureBuildBehind:
    def test_missing_completed_game_is_refused(self, sched_file):
        """schedules has the result; the model table does not. The pipeline was
        never rebuilt."""
        sched_file([game("played_a", "2026-09-07"), game("played_b", "2026-09-08")])
        train = pd.DataFrame(
            {"game_id": ["played_a"], "gameday": [pd.Timestamp("2026-09-07")]}
        )
        with pytest.raises(SystemExit) as e:
            predict_week.assert_training_is_current(train, pd.Timestamp("2026-09-14"))
        msg = str(e.value)
        assert "played_b" in msg
        assert "build_all" in msg  # tells the user the RIGHT fix

    def test_unplayed_future_games_are_not_expected_in_training(self, sched_file):
        sched_file(
            [game("done", "2026-09-07"), game("future", "2026-09-20", scored=False)]
        )
        train = pd.DataFrame(
            {"game_id": ["done"], "gameday": [pd.Timestamp("2026-09-07")]}
        )
        predict_week.assert_training_is_current(train, pd.Timestamp("2026-09-14"))


class TestRawPullBehind:
    def test_a_played_game_with_no_score_is_refused(self, sched_file):
        """Nothing upstream knows the result yet because nobody re-pulled."""
        today = pd.Timestamp.now().normalize()
        sched_file(
            [
                game("old", today - pd.Timedelta(days=30)),
                game("yesterday", today - pd.Timedelta(days=2), scored=False),
            ]
        )
        train = pd.DataFrame(
            {"game_id": ["old"], "gameday": [today - pd.Timedelta(days=30)]}
        )
        with pytest.raises(SystemExit) as e:
            predict_week.assert_training_is_current(train, today + pd.Timedelta(days=4))
        msg = str(e.value)
        assert "pull_schedules" in msg  # re-pull, not rebuild

    def test_an_old_cancelled_game_does_not_trip_it_forever(self, sched_file):
        """2021 BUF@CIN was abandoned and never got a score. A guard that
        treats that as 'the pull is behind' would be permanently red."""
        today = pd.Timestamp.now().normalize()
        sched_file(
            [
                game("2021_17_BUF_CIN", "2022-01-02", scored=False),
                game("recent", today - pd.Timedelta(days=7)),
            ]
        )
        train = pd.DataFrame(
            {"game_id": ["recent"], "gameday": [today - pd.Timedelta(days=7)]}
        )
        predict_week.assert_training_is_current(train, today + pd.Timedelta(days=4))
