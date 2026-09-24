"""
MLP (neural network): small architecture given ~1,400-1,800 training rows
per fold. Requires scaling (iterative optimizer, same reason as Poisson).

Re-benchmarked 2026-09-24 with the same two corrections every production model
has carried since 2026-09-14, because without them the comparison was not
like-for-like: historical overtime games are down-weighted in the fit (C5), and
the scoring-environment drift measured on the most recent 285 training games is
subtracted from predictions (common.recent_residual_offset). Before this the
shelved models carried the ~+0.65 away-score bias production had fixed.

Run: python -m src.models.unused.mlp
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from src.models.common import (
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

MODEL_TABLE = "data/processed/model_table.parquet"


def fit_mlp(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)

    params = dict(
        hidden_layer_sizes=(16, 8),
        alpha=1.0,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    )
    w = ot_sample_weight(train)  # C5: halve historical overtime games
    home_model = MLPRegressor(**params).fit(
        X_scaled, train["home_score"], sample_weight=w
    )
    away_model = MLPRegressor(**params).fit(
        X_scaled, train["away_score"], sample_weight=w
    )
    off_h, off_a = recent_residual_offset(
        train, home_model.predict(X_scaled), away_model.predict(X_scaled)
    )
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "scaler": scaler,
        "off_h": off_h,
        "off_a": off_a,
    }


def predict_mlp(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    return (
        model["home_model"].predict(X_scaled) - model["off_h"],
        model["away_model"].predict(X_scaled) - model["off_a"],
    )


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate(df, fit_mlp, predict_mlp, min_train_seasons=2)
    metrics = score_predictions(results)
    print("MLP walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "mlp", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
