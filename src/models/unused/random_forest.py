"""
Random Forest: 200 trees, scale-invariant, but sklearn's implementation
requires imputed inputs (no native NaN support), unlike the GBM libraries.

Re-benchmarked 2026-09-24 with the same two corrections every production model
has carried since 2026-09-14, because without them the comparison was not
like-for-like: historical overtime games are down-weighted in the fit (C5), and
the scoring-environment drift measured on the most recent 285 training games is
subtracted from predictions (common.recent_residual_offset). Before this the
shelved models carried the ~+0.65 away-score bias production had fixed.

Run: python -m src.models.unused.random_forest
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.common import (
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

MODEL_TABLE = "data/processed/model_table.parquet"


def _forest():
    return RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42, n_jobs=-1
    )


def fit_rf(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    w = ot_sample_weight(train)  # C5: halve historical overtime games
    home_model = _forest().fit(X_filled, train["home_score"], sample_weight=w)
    away_model = _forest().fit(X_filled, train["away_score"], sample_weight=w)
    off_h, off_a = recent_residual_offset(
        train, home_model.predict(X_filled), away_model.predict(X_filled)
    )
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "off_h": off_h,
        "off_a": off_a,
    }


def predict_rf(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    return (
        model["home_model"].predict(X) - model["off_h"],
        model["away_model"].predict(X) - model["off_a"],
    )


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate(df, fit_rf, predict_rf, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Random Forest walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "rf", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
