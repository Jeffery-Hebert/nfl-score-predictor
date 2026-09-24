"""
Correctness gates over every saved *_predictions.parquet.

Accuracy metrics only mean something if the predictions are well-formed. A
model emitting NaNs, negative scores, or a constant would still produce a
finite RMSE and sit in a comparison table looking respectable. A model whose
prediction file predates the current model_table would produce a perfectly
plausible RMSE for a feature set that no longer exists -- which is exactly what
happened here for two days.

These are parametrized over whatever prediction files are on disk, so a newly
run model is covered automatically.

Run: pytest tests/test_prediction_sanity.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

# Reads built parquet from data/, which is gitignored -- excluded from CI.
pytestmark = [pytest.mark.requires_data, pytest.mark.backtest_artifacts]

PRED_DIR = Path("data/processed")
PLAUSIBLE_MAX = 70.0
REQUIRED_COLS = {
    "game_id",
    "season",
    "week",
    "home_score",
    "away_score",
    "home_pred",
    "away_pred",
}


def _models():
    return sorted(
        p.name.replace("_predictions.parquet", "")
        for p in PRED_DIR.glob("*_predictions.parquet")
    )


def _load(name):
    return pd.read_parquet(PRED_DIR / f"{name}_predictions.parquet")


pytestmark = pytestmark + [pytest.mark.parametrize("model", _models())]


def test_has_the_required_columns(model):
    missing = REQUIRED_COLS - set(_load(model).columns)
    assert not missing, f"{model} predictions missing {missing}"


def test_no_missing_predictions(model):
    df = _load(model)
    for side in ("home", "away"):
        n = df[f"{side}_pred"].isna().sum()
        assert n == 0, f"{model} has {n} NaN {side} predictions"


def test_predictions_are_physically_plausible(model):
    """NFL scores are non-negative and realistically bounded. A model predicting
    a negative score is broken regardless of its RMSE -- plain linear regression
    is unconstrained and can do exactly this, which is why poisson_glm exists."""
    df = _load(model)
    for side in ("home", "away"):
        pred = df[f"{side}_pred"]
        assert (pred >= 0).all(), (
            f"{model} predicts {(pred < 0).sum()} negative {side} scores "
            f"(min {pred.min():.1f})"
        )
        assert (pred <= PLAUSIBLE_MAX).all(), (
            f"{model} predicts {(pred > PLAUSIBLE_MAX).sum()} {side} scores above "
            f"{PLAUSIBLE_MAX:.0f} (max {pred.max():.1f})"
        )


def test_predictions_are_not_degenerate(model):
    """A constant prediction scores a finite RMSE while carrying no signal."""
    df = _load(model)
    for side in ("home", "away"):
        assert (
            df[f"{side}_pred"].nunique() > 1
        ), f"{model} {side} predictions are constant"
        assert df[f"{side}_pred"].std() > 0.5, (
            f"{model} {side} predictions barely vary (std "
            f"{df[f'{side}_pred'].std():.3f}) -- close to predicting the mean"
        )


def test_one_row_per_game(model):
    df = _load(model)
    assert df["game_id"].is_unique, f"{model} has duplicate game_ids"


def test_scored_only_against_played_games(model):
    df = _load(model)
    assert df[["home_score", "away_score"]].notna().all().all(), (
        f"{model} contains rows with no final score -- RMSE would be computed "
        "against missing targets"
    )


def test_not_stale_against_the_current_feature_table(model):
    """The staleness class of bug: predictions computed from a table that has
    since changed are not comparable to anything current.

    Judged by CONTENT, from the provenance sidecar each backtest writes
    (src/validate/backtest_io.py): the played-game rows of every input must
    still hash the same. This used to compare file modification times, which
    failed after every rebuild even when nothing had changed -- the false alarm
    that got the check ignored. Re-run: python -m src.models.run_all --only <model>
    """
    from src.validate.backtest_io import stale_inputs

    problems = stale_inputs(model)
    assert not problems, f"{model} backtest is stale: {problems}"
