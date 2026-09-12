"""
Logistic regression: predicts win probability (home team wins), then maps
to a score pair via the average score differential among similar past
predicted-probability buckets. Not a natural fit for score prediction,
but included per spec.

Run: python -m src.models.logistic
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

def fit_logistic(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)

    y = (train["home_score"] > train["away_score"]).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(X_scaled, y)

    avg_total = (train["home_score"] + train["away_score"]).mean()
    avg_margin_when_home_wins = (train["home_score"] - train["away_score"])[y == 1].mean()
    avg_margin_when_away_wins = (train["home_score"] - train["away_score"])[y == 0].mean()

    return {"clf": clf, "scaler": scaler, "means": means, "avg_total": avg_total,
            "margin_win": avg_margin_when_home_wins, "margin_loss": avg_margin_when_away_wins}

def predict_logistic(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    win_prob = model["clf"].predict_proba(X_scaled)[:, 1]

    margin = win_prob * model["margin_win"] + (1 - win_prob) * model["margin_loss"]
    total = model["avg_total"]
    home_pred = (total + margin) / 2
    away_pred = (total - margin) / 2
    return home_pred, away_pred

def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_logistic, predict_logistic, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Logistic Regression (win-prob -> score) walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/logistic_predictions.parquet", index=False)

if __name__ == "__main__":
    main()