"""
Poisson GLM: respects that scores are non-negative counts, unlike plain
linear regression which can predict negative scores.

Run: python -m src.models.poisson_glm
"""

import pandas as pd
from sklearn.linear_model import PoissonRegressor
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

from sklearn.preprocessing import StandardScaler


def fit_poisson(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)
    home_model = PoissonRegressor(max_iter=300).fit(X_scaled, train["home_score"])
    away_model = PoissonRegressor(max_iter=300).fit(X_scaled, train["away_score"])
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "scaler": scaler,
    }


def predict_poisson(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    return model["home_model"].predict(X_scaled), model["away_model"].predict(X_scaled)


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(
        df, fit_poisson, predict_poisson, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("Poisson GLM walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/poisson_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
