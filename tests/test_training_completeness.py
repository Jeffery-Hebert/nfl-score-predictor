"""
Training COMPLETENESS -- the mirror image of the leakage tests.

Leakage tests ask: does the model see anything it should not? (nothing from the
future). This file asks the opposite and equally important question: does the
model see everything it SHOULD? A model that is leakage-free but silently
missing last week's games is safe and stupid -- it would predict Week 2 without
knowing Week 1 happened.

Both failures produce a model that runs cleanly and scores plausibly, so both
need to be asserted rather than assumed.

The contract, for every fold:

  no leakage        every training game kicked off strictly BEFORE the first
                    game of the week being predicted
  no gap            EVERY completed game before that kickoff is in training --
                    not a subset, not "most", all of them
  current           the newest training game is within days of the prediction,
                    so the model is never unknowingly a week behind

Run: pytest tests/test_training_completeness.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.validate.walk_forward import walk_forward_evaluate


def make_schedule(seasons=(2020, 2021, 2022), weeks=6, per_week=16):
    rows = []
    day = pd.Timestamp("2020-09-10")
    for season in seasons:
        for week in range(1, weeks + 1):
            for g in range(per_week):
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{g:02d}",
                        "season": season,
                        "week": week,
                        "gameday": day,
                        "home_score": 21 + (g % 7),
                        "away_score": 17 + (g % 5),
                        "feature": float(g),
                    }
                )
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=200)
    return pd.DataFrame(rows)


class Recorder:
    """Captures the exact train/test frames the harness hands the model."""

    def __init__(self):
        self.folds = []

    def fit(self, train):
        self._train = train.copy()
        return {"mean": train["home_score"].mean()}

    def predict(self, model, test):
        self.folds.append((self._train, test.copy()))
        n = len(test)
        return np.full(n, model["mean"]), np.full(n, model["mean"])


@pytest.fixture(scope="module")
def folds():
    df = make_schedule()
    rec = Recorder()
    walk_forward_evaluate(df, rec.fit, rec.predict, min_train_seasons=1)
    assert rec.folds, "harness produced no folds"
    return df, rec.folds


def test_no_completed_game_before_kickoff_is_missing(folds):
    """The core completeness property. Training must contain EVERY game played
    before the prediction week, not merely a leakage-free subset."""
    df, recorded = folds
    for train, test in recorded:
        cutoff = test["gameday"].min()
        should_be_there = set(df[df["gameday"] < cutoff]["game_id"])
        actually_there = set(train["game_id"])
        missing = should_be_there - actually_there
        assert not missing, (
            f"{len(missing)} completed games before {cutoff.date()} were withheld "
            f"from training, e.g. {sorted(missing)[:3]}. The model is predicting "
            f"with less history than it actually had."
        )


def test_training_is_never_a_week_behind(folds):
    """The newest training game must be recent relative to the prediction.

    Catches the specific failure of predicting Week N while the most recent
    training game is from Week N-2 -- i.e. last week's results never landed.
    """
    _, recorded = folds
    for train, test in recorded:
        cutoff = test["gameday"].min()
        newest = train["gameday"].max()
        gap_days = (cutoff - newest).days
        season, week = test.iloc[0]["season"], test.iloc[0]["week"]
        # Week 1 legitimately follows an offseason; in-season should be ~7 days.
        limit = 250 if week == 1 else 10
        assert gap_days <= limit, (
            f"season {season} week {week}: newest training game is {gap_days} days "
            f"before kickoff. The model is stale by roughly {gap_days // 7} week(s)."
        )


def test_training_grows_by_exactly_the_games_that_were_played(folds):
    """Between consecutive folds, training should gain precisely the games
    played in between -- no silent drops, no duplicates."""
    df, recorded = folds
    for (train_a, test_a), (train_b, test_b) in zip(recorded, recorded[1:]):
        ids_a, ids_b = set(train_a["game_id"]), set(train_b["game_id"])
        assert (
            ids_a <= ids_b
        ), f"{len(ids_a - ids_b)} games present in one fold vanished from the next"
        gained = ids_b - ids_a
        expected = (
            set(
                df[
                    (df["gameday"] >= train_a["gameday"].max())
                    & (df["gameday"] < test_b["gameday"].min())
                ]["game_id"]
            )
            - ids_a
        )
        assert gained == expected, (
            f"training gained {len(gained)} games but {len(expected)} were played "
            "in that interval"
        )


def test_no_training_game_is_at_or_after_kickoff(folds):
    """Completeness must not be bought with leakage."""
    _, recorded = folds
    for train, test in recorded:
        assert train["gameday"].max() < test["gameday"].min()


# ------------------------------------------------- against the real pipeline


@pytest.mark.requires_data
def test_real_pipeline_trains_on_everything_up_to_the_last_fold():
    """On the actual data, the final fold must train on every completed game
    that came before it. This is the property a live Week-N prediction depends
    on: the model must know about Week N-1."""
    df = pd.read_parquet("data/processed/model_table.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])
    played = df.dropna(subset=["home_score"]).sort_values("gameday")

    rec = Recorder()
    walk_forward_evaluate(df, rec.fit, rec.predict, min_train_seasons=2)
    train, test = rec.folds[-1]

    cutoff = test["gameday"].min()
    expected = set(played[played["gameday"] < cutoff]["game_id"])
    missing = expected - set(train["game_id"])
    assert (
        not missing
    ), f"final fold withheld {len(missing)} completed games from training"
    gap = (cutoff - train["gameday"].max()).days
    assert gap <= 250, f"final fold's newest training game is {gap} days stale"
