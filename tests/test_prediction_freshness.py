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
        assert "src.ingest.pull_all" in msg  # re-pull, not rebuild

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


# WHICH games. Exactly the ones the NEXT forecast run chooses from, found with
# predict_week's own functions: the week of the next game still to kick off,
# its games that have not kicked off, and each one's injury-report status.
#
# The first version checked "the earliest week with an unplayed game", all of
# it, whatever the hour. Once the pipeline ran daily (pipeline.yml) that failed
# most days for no fault. After Thursday night, and all of Tuesday and
# Wednesday, no remaining game has its final injury report, so injury_impact is
# zero for every team -- correctly. And on Monday one game is left, so EVERY
# feature is "identical across the week's games". It first fired on the
# scheduled run of Friday 2026-09-25: away_injury_impact, 15 games, no report.

# Built from each team's injury report, which only carries game statuses once it
# is FINAL; until then zero for every team, by design. So they are judged only
# across games whose final report is in -- the only games a run forecasts -- and
# only with two or more such games: one game's two teams can both honestly list
# nobody (2-4% of teams do).
INJURY_FEATURES = {"injury_impact"}
MIN_READY_GAMES = 2


def next_slate(table, season, week, now, pulled_at=None):
    """The games of (season, week) a run at `now` would choose from, each with
    injury_report_final -- through predict_week's own selection functions."""
    from src.predict import injury_readiness

    games = table[(table["season"] == season) & (table["week"] == week)].copy()
    games["kickoff"] = games["game_id"].map(predict_week.kickoff_utc(games["game_id"]))
    pending = predict_week.select_pending(games, now)
    ready = injury_readiness.assess(
        pending,
        pd.read_parquet(predict_week.INJURIES),
        injury_readiness.injuries_pulled_at() if pulled_at is None else pulled_at,
    )
    return pending.merge(
        ready[["game_id", "injury_report_final"]], on="game_id", how="left"
    )


def indistinct_team_features(slate, feature_cols) -> list[str]:
    """Per-team features with ONE value across every team about to play.

    Compared across TEAMS (home and away pooled), not games, so a one-game slate
    still has two teams to tell apart. Exempt by name, never by a variance
    threshold -- a threshold would quietly excuse a genuinely broken feature:
      GAME_FEATURE_COLS   legitimately constant in most weeks (no neutral site,
                          no playoff game on the slate);
      prior_games_played  calendar counters, straight from the schedule. Within
      rest_days           a week most teams have played as many games, and two
                          teams that both played last Sunday have the same rest
                          -- Monday night 2026 week 3, PHI and CHI, 8 days each.
    """
    from src.models.common import GAME_FEATURE_COLS

    exempt = set(GAME_FEATURE_COLS) | {
        f"{side}_{c}"
        for side in ("home", "away")
        for c in ("prior_games_played", "rest_days")
    }
    bases = sorted({c.split("_", 1)[1] for c in feature_cols if c not in exempt})
    ready = slate[slate["injury_report_final"].fillna(False).astype(bool)]
    constant = []
    for base in bases:
        games = ready if base in INJURY_FEATURES else slate
        if base in INJURY_FEATURES and len(games) < MIN_READY_GAMES:
            continue  # not judged until reports are final for 2+ games
        teams = pd.concat([games[f"home_{base}"], games[f"away_{base}"]])
        if teams.nunique(dropna=False) <= 1:
            constant.append(base)
    return constant


class TestTheIndistinctFeatureCheck:
    """indistinct_team_features on hand-built slates, at every point in the
    week the daily pipeline runs. No data needed."""

    COLS = ["home_epa", "away_epa", "home_injury_impact", "away_injury_impact"]
    CALENDAR = ["home_rest_days", "away_rest_days"]

    @staticmethod
    def slate(epa, injury, final):
        """epa, injury: [(home, away), ...] per game; final: per game."""
        return pd.DataFrame(
            {
                "home_epa": [h for h, _ in epa],
                "away_epa": [a for _, a in epa],
                "home_injury_impact": [h for h, _ in injury],
                "away_injury_impact": [a for _, a in injury],
                "injury_report_final": final,
            }
        )

    def test_a_one_game_slate_still_has_two_teams_to_tell_apart(self):
        s = self.slate([(0.1, -0.2)], [(0.0, 0.0)], [True])  # Monday night
        assert indistinct_team_features(s, self.COLS) == []

    def test_two_teams_on_the_same_rest_are_not_a_fault(self):
        s = self.slate([(0.1, -0.2)], [(0.0, 0.0)], [False])
        s["home_rest_days"] = s["away_rest_days"] = 8  # both played last Sunday
        assert indistinct_team_features(s, self.COLS + self.CALENDAR) == []

    def test_injuries_are_not_judged_before_the_final_report(self):
        # Tuesday: no report is final, every injury_impact is zero, correctly
        s = self.slate([(0.1, -0.2), (0.3, 0.0)], [(0.0, 0.0)] * 2, [False, False])
        assert indistinct_team_features(s, self.COLS) == []

    def test_a_broken_team_feature_is_still_caught(self):
        s = self.slate([(0.1, 0.1), (0.1, 0.1)], [(0.2, 0.0)] * 2, [True, True])
        assert indistinct_team_features(s, self.COLS) == ["epa"]

    def test_injuries_missing_from_final_reports_are_caught(self):
        s = self.slate([(0.1, -0.2), (0.3, 0.0)], [(0.0, 0.0)] * 2, [True, True])
        assert indistinct_team_features(s, self.COLS) == ["injury_impact"]


@pytest.mark.requires_data
class TestUpcomingWeekIsPredictable:
    @pytest.fixture(scope="class")
    @classmethod
    def upcoming(cls):
        from src.models.common import FEATURE_COLS

        table = predict_week.load_table()
        now = pd.Timestamp.now(tz="UTC")
        picked = predict_week.pick_week(table, None, None, now=now)
        if picked is None:
            pytest.skip("no game left to kick off -- the season is over")
        return table, next_slate(table, *picked, now), FEATURE_COLS

    def test_every_feature_column_exists(self, upcoming):
        table, _, feature_cols = upcoming
        missing = set(feature_cols) - set(table.columns)
        assert not missing, (
            f"model_table is missing {sorted(missing)} -- FEATURE_COLS and the "
            "feature pipeline have drifted apart. Rebuild: "
            "python -m src.features.build_all"
        )

    def test_no_feature_is_null_for_the_games_still_to_play(self, upcoming):
        """The core check. A null here does not crash -- it silently becomes the
        training mean, and the feature stops doing anything."""
        _, slate, feature_cols = upcoming
        null_rate = slate[feature_cols].isna().mean()
        broken = null_rate[null_rate > 0]
        assert broken.empty, (
            f"season {int(slate['season'].iloc[0])} week {int(slate['week'].iloc[0])} "
            f"has null features:\n{(broken * 100).round(1).to_string()}\n"
            "These would be silently replaced by the training mean, so every team "
            "gets the same value and the feature contributes nothing to the "
            "forecast. Nothing else would report an error."
        )

    def test_team_features_tell_the_teams_apart(self, upcoming):
        """Non-null is not enough. A column that is present but identical for
        every team carries no information about who plays whom -- the same
        silent failure wearing a different disguise."""
        _, slate, feature_cols = upcoming
        constant = indistinct_team_features(slate, feature_cols)
        assert not constant, (
            f"these per-team features have one value for every team still to "
            f"play in season {int(slate['season'].iloc[0])} week "
            f"{int(slate['week'].iloc[0])}: {constant}. They cannot be "
            "distinguishing the matchups."
        )

    def test_the_split_efficiency_family_reaches_the_upcoming_week(self, upcoming):
        """Named guard for the 2026-09-17 feature family.

        The generic checks above would catch a total failure. This one states the
        specific expectation so the failure message names the right builder, and
        so removing build_split_efficiency from the pipeline fails loudly here
        rather than as an anonymous null."""
        from src.models.common import SPLIT_FEATURE_COLS

        _, slate, _ = upcoming
        sided = [f"{s}_{c}" for s in ("home", "away") for c in SPLIT_FEATURE_COLS]
        missing = [c for c in sided if c not in slate.columns]
        assert not missing, (
            f"{missing} absent -- build_split_efficiency.py did not reach "
            "model_table. Check the join in build_game_features.py."
        )
        assert not slate[sided].isna().any().any(), (
            "the pass/rush split is null for the upcoming week. The backtest "
            "would still pass; only the live forecast is broken."
        )


# ------------------------------------------------- freezing started games

# A week is now predicted several times -- Thursday afternoon for the Thursday
# night game, Sunday morning for the Sunday slate, Monday afternoon for Monday
# night -- so that every game is forecast on the freshest injury report
# available before ITS OWN kickoff.
#
# predict_week writes one parquet per week, so without a guard the Monday run
# would rewrite the Thursday game's forecast three days after it was played.
# The file would then claim to have predicted games whose results it had already
# seen: the single thing this project's records exist to rule out.


class TestFreezeStartedGames:
    @staticmethod
    def _row(game_id, kickoff, home=24.0, generated="2026-09-17T12:00:00+00:00"):
        return {
            "game_id": game_id,
            "gameday": pd.Timestamp(kickoff).tz_localize(None).normalize(),
            "kickoff": pd.Timestamp(kickoff, tz="UTC"),
            "combined_home": home,
            "combined_away": 20.0,
            "generated_at": generated,
        }

    def _write(self, tmp_path, rows):
        path = tmp_path / "2026_wk02.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_a_started_game_keeps_its_original_prediction(self, tmp_path):
        """THE property. The Thursday game was predicted at 24.0 before kickoff;
        a later run that would have said 31.0 must not overwrite it."""
        path = self._write(
            tmp_path,
            [
                self._row("THU", "2026-09-17T23:15:00", home=24.0),
                self._row("SUN", "2026-09-20T17:00:00", home=20.0),
            ],
        )
        fresh = pd.DataFrame(
            [
                self._row("THU", "2026-09-17T23:15:00", home=31.0, generated="LATER"),
                self._row("SUN", "2026-09-20T17:00:00", home=27.0, generated="LATER"),
            ]
        )
        now = pd.Timestamp("2026-09-20T11:00:00", tz="UTC")  # Sunday morning
        got = predict_week.freeze_started_games(fresh, path, now).set_index("game_id")

        assert got.loc["THU", "combined_home"] == 24.0, (
            "the Thursday game's forecast was rewritten after it kicked off -- "
            "the record now claims a prediction it did not make"
        )
        assert got.loc["THU", "generated_at"] != "LATER", (
            "generated_at was overwritten, so the file would misreport WHEN the "
            "Thursday forecast was made"
        )
        assert (
            got.loc["SUN", "combined_home"] == 27.0
        ), "the Sunday game had not kicked off and should have been refreshed"

    def test_nothing_is_frozen_before_any_kickoff(self, tmp_path):
        path = self._write(
            tmp_path, [self._row("THU", "2026-09-17T23:15:00", home=24.0)]
        )
        fresh = pd.DataFrame([self._row("THU", "2026-09-17T23:15:00", home=31.0)])
        now = pd.Timestamp("2026-09-17T18:00:00", tz="UTC")  # hours before kickoff
        got = predict_week.freeze_started_games(fresh, path, now).set_index("game_id")
        assert got.loc["THU", "combined_home"] == 31.0

    def test_every_game_survives_the_merge(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                self._row("THU", "2026-09-17T23:15:00"),
                self._row("SUN", "2026-09-20T17:00:00"),
                self._row("MON", "2026-09-22T00:15:00"),
            ],
        )
        fresh = pd.DataFrame(
            [
                self._row("THU", "2026-09-17T23:15:00"),
                self._row("SUN", "2026-09-20T17:00:00"),
                self._row("MON", "2026-09-22T00:15:00"),
            ]
        )
        now = pd.Timestamp("2026-09-20T11:00:00", tz="UTC")
        got = predict_week.freeze_started_games(fresh, path, now)
        assert set(got["game_id"]) == {"THU", "SUN", "MON"}
        assert len(got) == 3, "the merge duplicated or dropped a game"

    def test_a_first_run_with_no_existing_file_is_unchanged(self, tmp_path):
        fresh = pd.DataFrame([self._row("THU", "2026-09-17T23:15:00")])
        got = predict_week.freeze_started_games(
            fresh, tmp_path / "missing.parquet", pd.Timestamp.now(tz="UTC")
        )
        assert got.equals(fresh)

    def test_a_file_written_before_kickoff_existed_is_recovered(
        self, tmp_path, monkeypatch
    ):
        """Older records carry no kickoff column. It must be recovered from the
        schedule so the row is still frozen, not silently treated as pending and
        overwritten.

        The schedule is supplied by the test rather than read from data/raw/,
        which is gitignored and absent in CI.
        """
        sched = tmp_path / "schedules.parquet"
        pd.DataFrame(
            [
                {
                    "game_id": "THU",
                    "gameday": "2026-09-17",
                    "gametime": "19:15",  # ET -> 2026-09-17T23:15Z
                }
            ]
        ).to_parquet(sched, index=False)
        monkeypatch.setattr(predict_week, "SCHEDULES", sched)

        path = tmp_path / "2026_wk02.parquet"
        row = self._row("THU", "2026-09-17T23:15:00", home=24.0)
        del row["kickoff"]
        pd.DataFrame([row]).to_parquet(path, index=False)

        fresh = pd.DataFrame([self._row("THU", "2026-09-17T23:15:00", home=31.0)])
        got = predict_week.freeze_started_games(
            fresh, path, pd.Timestamp("2026-09-20T11:00:00", tz="UTC")
        )
        assert set(got["game_id"]) == {"THU"}, "the legacy row vanished"
        assert got.iloc[0]["combined_home"] == 24.0, (
            "a legacy row's kickoff was not recovered, so a played game was "
            "re-forecast and its original record lost"
        )


class TestTrainingCutoffFollowsPendingGames:
    """The cutoff must track the first game still to be played, not the first
    game of the week.

    Those are the same thing on a Thursday run and diverge afterwards. Pinned to
    the week, a Monday run refits on exactly what the Thursday run had -- it
    discards that same week's Thursday and Sunday results while forecasting
    Monday night. Leakage-safe, but blind in the way
    tests/test_training_completeness.py exists to forbid.
    """

    @staticmethod
    def _week():
        return pd.DataFrame(
            {
                "game_id": ["THU", "SUN1", "SUN2", "MON"],
                "gameday": pd.to_datetime(
                    ["2026-09-17", "2026-09-20", "2026-09-20", "2026-09-21"]
                ),
                "kickoff": pd.to_datetime(
                    [
                        "2026-09-18T00:15Z",
                        "2026-09-20T17:00Z",
                        "2026-09-20T17:00Z",
                        "2026-09-22T00:15Z",
                    ]
                ),
            }
        )

    @staticmethod
    def _cutoff(week, now):
        """The selection predict_week.main() performs -- through the SAME
        functions it calls. (This used to re-implement the selection inline, so
        a change to main() could not have failed it.)"""
        pending = predict_week.select_pending(week, now)
        if pending.empty:
            return None, 0
        return predict_week.training_cutoff(pending), len(pending)

    def test_thursday_run_uses_the_first_game_of_the_week(self):
        cutoff, n = self._cutoff(self._week(), pd.Timestamp("2026-09-17T18:00Z"))
        assert n == 4, "nothing has kicked off yet"
        assert cutoff == pd.Timestamp("2026-09-17")

    def test_sunday_run_can_train_on_the_thursday_result(self):
        cutoff, n = self._cutoff(self._week(), pd.Timestamp("2026-09-20T11:00Z"))
        assert n == 3, "only the Thursday game has been played"
        assert cutoff == pd.Timestamp("2026-09-20"), (
            "cutoff is still pinned to Thursday, so the Sunday run cannot train "
            "on Thursday night's result"
        )

    def test_monday_run_can_train_on_the_whole_weekend(self):
        cutoff, n = self._cutoff(self._week(), pd.Timestamp("2026-09-21T18:00Z"))
        assert n == 1, "only Monday night is left"
        assert cutoff == pd.Timestamp("2026-09-21"), (
            "cutoff is pinned to the week's first game, so the Monday run "
            "discards this week's Thursday and Sunday results"
        )

    def test_cutoff_advances_monotonically_through_the_week(self):
        week = self._week()
        cutoffs = [
            self._cutoff(week, pd.Timestamp(t))[0]
            for t in ("2026-09-17T18:00Z", "2026-09-20T11:00Z", "2026-09-21T18:00Z")
        ]
        assert cutoffs == sorted(cutoffs), f"cutoff went backwards: {cutoffs}"
        assert len(set(cutoffs)) == 3, "later runs are not seeing more history"

    def test_a_game_about_to_kick_off_is_left_as_it_is(self):
        """Its forecast might not be committed before kickoff."""
        week = self._week()
        sunday = pd.Timestamp("2026-09-20T17:00Z")
        just_before = predict_week.select_pending(
            week, sunday - pd.Timedelta(minutes=10)
        )
        earlier = predict_week.select_pending(week, sunday - pd.Timedelta(minutes=20))
        assert list(just_before["game_id"]) == ["MON"]
        assert list(earlier["game_id"]) == ["SUN1", "SUN2", "MON"]

    def test_the_default_week_moves_on_from_a_game_about_to_kick_off(self, monkeypatch):
        # Monday night is the week's last game; 20 minutes out it is still the
        # week to run, 10 minutes out it is left alone and the next week is.
        table = pd.DataFrame(
            {
                "game_id": ["MON", "NEXT_THU"],
                "season": 2026,
                "week": [2, 3],
                "gameday": pd.to_datetime(["2026-09-21", "2026-09-24"]),
                "home_score": [float("nan")] * 2,
            }
        )
        kick = pd.Series(
            pd.to_datetime(["2026-09-22T00:15Z", "2026-09-25T00:15Z"]),
            index=["MON", "NEXT_THU"],
        )
        monkeypatch.setattr(predict_week, "kickoff_utc", lambda ids: kick)
        mnf = pd.Timestamp("2026-09-22T00:15Z")
        pick = lambda before: predict_week.pick_week(  # noqa: E731
            table, None, None, now=mnf - pd.Timedelta(minutes=before)
        )
        assert pick(20) == (2026, 2)
        assert pick(10) == (2026, 3)

    def test_a_fully_played_week_leaves_nothing_pending(self):
        _, n = self._cutoff(self._week(), pd.Timestamp("2026-09-30T00:00Z"))
        assert n == 0, (
            "predict_week must exit rather than write a forecast for a week that "
            "is entirely in the past -- that is a backfill, not a prediction"
        )


class TestNotReForecastGamesAreKept:
    """A run can now leave games out on purpose -- their final injury report is
    not published yet -- and an earlier forecast of such a game is still a real
    pre-kickoff record. The first merge rule kept only STARTED games, so those
    rows would have been dropped from the week's file."""

    @staticmethod
    def _row(game_id, kickoff, home=24.0, generated="2026-09-24T18:00:00+00:00"):
        return {
            "game_id": game_id,
            "gameday": pd.Timestamp(kickoff).tz_localize(None).normalize(),
            "kickoff": pd.Timestamp(kickoff, tz="UTC"),
            "combined_home": home,
            "combined_away": 20.0,
            "generated_at": generated,
        }

    def test_a_game_left_out_of_this_run_keeps_its_forecast(self, tmp_path):
        path = tmp_path / "2026_wk03.parquet"
        pd.DataFrame(
            [
                self._row("THU", "2026-09-25T00:15:00", home=24.0),
                self._row("SUN", "2026-09-27T17:00:00", home=21.0),
            ]
        ).to_parquet(path, index=False)
        # Thursday run, before the Sunday report: only THU is re-forecast.
        fresh = pd.DataFrame(
            [self._row("THU", "2026-09-25T00:15:00", home=26.0, generated="LATER")]
        )
        now = pd.Timestamp("2026-09-24T20:00:00", tz="UTC")
        got = predict_week.freeze_started_games(fresh, path, now).set_index("game_id")
        assert set(got.index) == {"THU", "SUN"}, "the Sunday forecast was dropped"
        assert got.loc["SUN", "combined_home"] == 21.0
        assert got.loc["THU", "combined_home"] == 26.0, "THU had not kicked off"

    def test_output_is_in_kickoff_order(self, tmp_path):
        path = tmp_path / "2026_wk03.parquet"
        pd.DataFrame([self._row("MON", "2026-09-29T00:15:00")]).to_parquet(
            path, index=False
        )
        fresh = pd.DataFrame([self._row("THU", "2026-09-25T00:15:00")])
        got = predict_week.freeze_started_games(
            fresh, path, pd.Timestamp("2026-09-24T20:00:00", tz="UTC")
        )
        assert list(got["game_id"]) == ["THU", "MON"]
