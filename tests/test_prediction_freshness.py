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


# ------------------------------------------------- feature readiness

# The guard above answers "is the training data complete?". It cannot answer
# "can the feature set actually be evaluated for the games we are about to
# predict?", and that is a separate way for a live prediction to be silently
# wrong.
#
# The failure mode. Every model imputes missing inputs with the TRAINING-fold
# mean (`X.fillna(model["means"])`). That is correct and leakage-safe, but it
# means a feature that is null for unplayed games does not raise -- every team
# quietly receives the same league-average value for it, the prediction still
# comes out looking like a football score, and the feature has silently stopped
# contributing. A whole feature family can drop out of a live forecast with no
# error anywhere.
#
# This is not hypothetical for the split-efficiency columns. They are built from
# a running accumulator over prior games, so an off-by-one at the end of the
# history -- or a join that only covers played games -- would populate the
# backtest perfectly and leave the upcoming week empty. The backtest would look
# fine. Only the live prediction would be broken.
#
# Added 2026-09-17 with the pass/rush split. Applies to whatever FEATURE_COLS
# holds, so a future feature family is covered without touching this file.


@pytest.mark.requires_data
class TestUpcomingWeekIsPredictable:
    @pytest.fixture(scope="class")
    def upcoming(self):
        from src.models.common import FEATURE_COLS

        table = pd.read_parquet("data/processed/model_table.parquet")
        table["gameday"] = pd.to_datetime(table["gameday"])
        unplayed = table[table["home_score"].isna()]
        if unplayed.empty:
            pytest.skip("no unplayed games in the table -- season is complete")
        nxt = unplayed.sort_values("gameday").iloc[0]
        week = unplayed[
            (unplayed["season"] == nxt["season"]) & (unplayed["week"] == nxt["week"])
        ]
        return table, week, FEATURE_COLS

    def test_every_feature_column_exists(self, upcoming):
        table, _, feature_cols = upcoming
        missing = set(feature_cols) - set(table.columns)
        assert not missing, (
            f"model_table is missing {sorted(missing)} -- FEATURE_COLS and the "
            "feature pipeline have drifted apart. Rebuild: "
            "python -m src.features.build_all"
        )

    def test_no_feature_is_null_for_the_next_unplayed_week(self, upcoming):
        """The core check. A null here does not crash -- it silently becomes the
        training mean, and the feature stops doing anything."""
        _, week, feature_cols = upcoming
        null_rate = week[feature_cols].isna().mean()
        broken = null_rate[null_rate > 0]
        assert broken.empty, (
            f"season {int(week['season'].iloc[0])} week {int(week['week'].iloc[0])} "
            f"has null features:\n{(broken * 100).round(1).to_string()}\n"
            "These would be silently replaced by the training mean, so every team "
            "gets the same value and the feature contributes nothing to the "
            "forecast. Nothing else would report an error."
        )

    def test_features_actually_vary_across_the_upcoming_matchups(self, upcoming):
        """Non-null is not enough. A column that is present but constant across
        all 16 games carries no information about who plays whom -- the same
        silent failure wearing a different disguise."""
        _, week, feature_cols = upcoming
        from src.models.common import GAME_FEATURE_COLS

        # Exempt by name, never by a variance threshold -- a threshold would
        # quietly excuse a genuinely broken team feature too.
        #   GAME_FEATURE_COLS   legitimately constant in most weeks (no neutral
        #                       site, no playoff game on the slate).
        #   prior_games_played  a season-progress counter. Within one week every
        #                       team has played the same number of games, apart
        #                       from byes, so near-constant is its correct
        #                       behaviour rather than a fault.
        exempt = set(GAME_FEATURE_COLS) | {
            f"{side}_prior_games_played" for side in ("home", "away")
        }
        per_team = [c for c in feature_cols if c not in exempt]
        constant = [c for c in per_team if week[c].nunique(dropna=False) <= 1]
        assert not constant, (
            f"these per-team features are identical for every game in "
            f"season {int(week['season'].iloc[0])} week {int(week['week'].iloc[0])}: "
            f"{constant}. They cannot be distinguishing the matchups."
        )

    def test_the_split_efficiency_family_reaches_the_upcoming_week(self, upcoming):
        """Named guard for the 2026-09-17 feature family.

        The generic checks above would catch a total failure. This one states the
        specific expectation so the failure message names the right builder, and
        so removing build_split_efficiency from the pipeline fails loudly here
        rather than as an anonymous null."""
        from src.models.common import SPLIT_FEATURE_COLS

        _, week, _ = upcoming
        sided = [f"{s}_{c}" for s in ("home", "away") for c in SPLIT_FEATURE_COLS]
        missing = [c for c in sided if c not in week.columns]
        assert not missing, (
            f"{missing} absent -- build_split_efficiency.py did not reach "
            "model_table. Check the join in build_game_features.py."
        )
        assert not week[sided].isna().any().any(), (
            "the pass/rush split is null for the upcoming week. The backtest "
            "would still pass; only the live forecast is broken."
        )
