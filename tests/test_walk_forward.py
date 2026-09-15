"""
B1/B2: tests for the evaluation harness itself.

src/validate/walk_forward.py had no tests. Every accuracy number this project
has ever produced -- every model comparison, every closed null result in the
findings doc -- comes out of this file. A leakage bug here would silently
invalidate all of it, and nothing would catch it: the feature-level leakage
gates cannot see a harness that hands a model future rows.

The central property under test: for every fold, the training frame contains
only games played strictly BEFORE the first game of the test week.

Run: pytest tests/test_walk_forward.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.validate.walk_forward import (
    score_predictions,
    walk_forward_evaluate,
    walk_forward_evaluate_by_season,
)


def make_games(seasons=(2020, 2021, 2022), weeks_per_season=6, games_per_week=16):
    """Synthetic schedule: chronological, evenly spaced, one feature column."""
    rows = []
    day = pd.Timestamp("2020-09-06")
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            for g in range(games_per_week):
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{g}",
                        "season": season,
                        "week": week,
                        "gameday": day,
                        "home_score": 20 + g,
                        "away_score": 17 + g,
                        "feature": float(g),
                    }
                )
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=200)  # offseason
    return pd.DataFrame(rows)


class SpyModel:
    """Records exactly what the harness handed fit_fn on every fold."""

    def __init__(self):
        self.train_frames = []
        self.test_frames = []

    def fit(self, train):
        self.train_frames.append(train.copy())
        return {"mean": train["home_score"].mean()}

    def predict(self, model, test):
        self.test_frames.append(test.copy())
        n = len(test)
        return np.full(n, model["mean"]), np.full(n, model["mean"])


# ---------------------------------------------------------------- leakage


def test_train_is_strictly_before_test_in_every_fold():
    """THE core property. If this fails, every number in the project is wrong."""
    df = make_games()
    spy = SpyModel()
    walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)

    assert spy.train_frames, "harness ran no folds"
    for train, test in zip(spy.train_frames, spy.test_frames):
        assert (
            train["gameday"].max() < test["gameday"].min()
        ), "LEAKAGE: a training game is on or after the first game of the test week"


def test_train_and_test_never_share_a_game():
    df = make_games()
    spy = SpyModel()
    walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)
    for train, test in zip(spy.train_frames, spy.test_frames):
        overlap = set(train["game_id"]) & set(test["game_id"])
        assert not overlap, f"LEAKAGE: {len(overlap)} games in both train and test"


def test_a_model_cannot_see_its_own_test_week():
    """Same-week games must not train the model that predicts them -- a Thursday
    result must not inform that Sunday's prediction."""
    df = make_games()
    spy = SpyModel()
    walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)
    for train, test in zip(spy.train_frames, spy.test_frames):
        season, week = test.iloc[0]["season"], test.iloc[0]["week"]
        same_week = train[(train["season"] == season) & (train["week"] == week)]
        assert (
            same_week.empty
        ), f"LEAKAGE: season {season} week {week} trained on itself"


def test_training_set_grows_monotonically():
    """Rolling origin: each fold should train on at least as much as the last."""
    df = make_games()
    spy = SpyModel()
    walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)
    sizes = [len(t) for t in spy.train_frames]
    assert sizes == sorted(sizes), f"training set shrank between folds: {sizes}"


# ------------------------------------------------------------ fold coverage


def test_every_test_game_predicted_exactly_once():
    df = make_games()
    spy = SpyModel()
    results = walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)
    assert results["game_id"].is_unique, "a game was predicted in more than one fold"

    expected = set(df[df["season"] > df["season"].min()]["game_id"])
    assert (
        set(results["game_id"]) == expected
    ), "fold coverage does not match the test seasons"


def test_min_train_seasons_controls_the_burn_in():
    df = make_games(seasons=(2020, 2021, 2022, 2023))
    spy1, spy2 = SpyModel(), SpyModel()
    r1 = walk_forward_evaluate(df, spy1.fit, spy1.predict, min_train_seasons=1)
    r2 = walk_forward_evaluate(df, spy2.fit, spy2.predict, min_train_seasons=2)

    assert sorted(r1["season"].unique()) == [2021, 2022, 2023]
    assert sorted(r2["season"].unique()) == [2022, 2023]
    assert len(r2) < len(r1), "a longer burn-in must leave fewer test games"


def test_unplayed_games_are_dropped():
    """Rows with no final score must never reach a model."""
    df = make_games()
    df.loc[df["season"] == 2022, ["home_score", "away_score"]] = np.nan
    spy = SpyModel()
    results = walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)

    assert 2022 not in set(results["season"]), "unplayed games were scored"
    for frame in spy.train_frames + spy.test_frames:
        assert frame["home_score"].notna().all(), "a NaN-target row reached the model"


def test_folds_with_too_little_history_are_skipped_not_crashed():
    """< 50 training rows should skip the fold and keep going."""
    df = make_games(seasons=(2020, 2021), weeks_per_season=1, games_per_week=10)
    spy = SpyModel()
    results = walk_forward_evaluate(df, spy.fit, spy.predict, min_train_seasons=1)
    assert len(spy.train_frames) == 0, "expected every fold to be skipped"
    assert results.empty, "skipped folds must yield an empty frame, not a crash"
    assert list(results.columns) == [
        "game_id",
        "season",
        "week",
        "home_score",
        "away_score",
        "home_pred",
        "away_pred",
    ], "empty result must still carry the normal columns so callers can rely on them"
    assert score_predictions(results)["n_games"] == 0


# -------------------------------------------------- season-level variant


def test_by_season_variant_trains_only_on_prior_seasons():
    df = make_games(seasons=(2020, 2021, 2022))
    spy = SpyModel()
    walk_forward_evaluate_by_season(df, spy.fit, spy.predict, min_train_seasons=1)
    for train, test in zip(spy.train_frames, spy.test_frames):
        assert (
            train["season"].max() < test["season"].min()
        ), "LEAKAGE: season-level split trained on the test season"


# ------------------------------------------------------- B2: scoring


def test_score_predictions_on_known_values():
    results = pd.DataFrame(
        {
            "home_score": [20.0, 30.0],
            "away_score": [10.0, 20.0],
            "home_pred": [22.0, 27.0],  # errors -2, +3
            "away_pred": [10.0, 24.0],  # errors  0, -4
        }
    )
    m = score_predictions(results)
    assert m["home_rmse"] == pytest.approx(np.sqrt((4 + 9) / 2))
    assert m["away_rmse"] == pytest.approx(np.sqrt((0 + 16) / 2))
    assert m["home_mae"] == pytest.approx(2.5)
    assert m["away_mae"] == pytest.approx(2.0)
    assert m["n_games"] == 2


def test_score_predictions_is_zero_for_perfect_predictions():
    results = pd.DataFrame(
        {
            "home_score": [20.0, 30.0],
            "away_score": [10.0, 20.0],
            "home_pred": [20.0, 30.0],
            "away_pred": [10.0, 20.0],
        }
    )
    m = score_predictions(results)
    assert m["home_rmse"] == 0 and m["away_rmse"] == 0
    assert m["home_mae"] == 0 and m["away_mae"] == 0


def test_rmse_penalises_a_single_large_miss_more_than_mae():
    """Guards the direction of the metrics -- a swapped formula would pass a
    naive equality test but not this."""
    spread = pd.DataFrame(
        {
            "home_score": [20.0, 20.0],
            "away_score": [0.0, 0.0],
            "home_pred": [10.0, 30.0],
            "away_pred": [0.0, 0.0],
        }
    )
    concentrated = pd.DataFrame(
        {
            "home_score": [20.0, 20.0],
            "away_score": [0.0, 0.0],
            "home_pred": [0.0, 20.0],
            "away_pred": [0.0, 0.0],
        }
    )
    a, b = score_predictions(spread), score_predictions(concentrated)
    assert a["home_mae"] == b["home_mae"], "fixture should hold MAE equal"
    assert b["home_rmse"] > a["home_rmse"], "RMSE must punish the concentrated miss"
