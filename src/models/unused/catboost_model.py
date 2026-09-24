"""
CatBoost: handles missing values natively.

Re-benchmarked 2026-09-24 with the same two corrections every production model
has carried since 2026-09-14, because without them the comparison was not
like-for-like: historical overtime games are down-weighted in the fit (C5), and
the scoring-environment drift measured on the most recent 285 training games is
subtracted from predictions (common.recent_residual_offset). Before this the
shelved models carried the ~+0.65 away-score bias production had fixed.

Run: python -m src.models.unused.catboost_model
"""

import pandas as pd
from catboost import CatBoostRegressor
from src.models.common import (
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

MODEL_TABLE = "data/processed/model_table.parquet"


def fit_catboost(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    params = dict(
        iterations=100,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
    )
    w = ot_sample_weight(train)  # C5: halve historical overtime games
    home_model = CatBoostRegressor(**params).fit(
        X, train["home_score"], sample_weight=w
    )
    away_model = CatBoostRegressor(**params).fit(
        X, train["away_score"], sample_weight=w
    )
    off_h, off_a = recent_residual_offset(
        train, home_model.predict(X), away_model.predict(X)
    )
    return {
        "home_model": home_model,
        "away_model": away_model,
        "off_h": off_h,
        "off_a": off_a,
    }


def predict_catboost(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS]
    return (
        model["home_model"].predict(X) - model["off_h"],
        model["away_model"].predict(X) - model["off_a"],
    )


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate(
        df, fit_catboost, predict_catboost, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("CatBoost walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "catboost", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
