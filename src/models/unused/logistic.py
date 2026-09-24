"""
Logistic regression: predicts win probability (home team wins), then maps
to a score pair via the average score differential among similar past
predicted-probability buckets. Not a natural fit for score prediction,
but included per spec.

Re-benchmarked 2026-09-24 with production's corrections: overtime games at half
weight in the fit and the averages, and the recent-residual drift offset.

Run: python -m src.models.unused.logistic
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.common import (
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

MODEL_TABLE = "data/processed/model_table.parquet"


def fit_logistic(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)

    w = ot_sample_weight(train)  # C5: halve historical overtime games
    w_arr = np.ones(len(train)) if w is None else np.asarray(w, float)
    y = (train["home_score"] > train["away_score"]).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(X_scaled, y, sample_weight=w_arr)

    margin = (train["home_score"] - train["away_score"]).to_numpy(float)
    total = (train["home_score"] + train["away_score"]).to_numpy(float)
    won = y.to_numpy() == 1
    model = {
        "clf": clf,
        "scaler": scaler,
        "means": means,
        "avg_total": float(np.average(total, weights=w_arr)),
        "margin_win": float(np.average(margin[won], weights=w_arr[won])),
        "margin_loss": float(np.average(margin[~won], weights=w_arr[~won])),
        "off_h": 0.0,
        "off_a": 0.0,
    }
    h, a = predict_logistic(model, train)
    model["off_h"], model["off_a"] = recent_residual_offset(train, h, a)
    return model


def predict_logistic(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    win_prob = model["clf"].predict_proba(X_scaled)[:, 1]

    margin = win_prob * model["margin_win"] + (1 - win_prob) * model["margin_loss"]
    total = model["avg_total"]
    home_pred = (total + margin) / 2
    away_pred = (total - margin) / 2
    return home_pred - model["off_h"], away_pred - model["off_a"]


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate(
        df, fit_logistic, predict_logistic, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("Logistic Regression (win-prob -> score) walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "logistic", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
