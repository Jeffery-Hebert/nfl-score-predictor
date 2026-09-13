"""
Linear regression baseline model: separate LinearRegression for home_score
and away_score, on the shared FEATURE_COLS. Simplest real ML model in the
ensemble -- if this can't beat the rule-based baseline, none of the fancier
models are likely to either.

Run: python -m src.models.linear
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions


def fit_linear(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    home_model = LinearRegression().fit(X_filled, train["home_score"])
    away_model = LinearRegression().fit(X_filled, train["away_score"])
    return {"home_model": home_model, "away_model": away_model, "means": means}


def predict_linear(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    return model["home_model"].predict(X), model["away_model"].predict(X)


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_linear, predict_linear, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Linear Regression walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/linear_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
