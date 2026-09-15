"""
MLP (neural network): small architecture given ~1,400-1,800 training rows
per fold. Requires scaling (iterative optimizer, same reason as Poisson).

Run: python -m src.models.unused.mlp
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions


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
    home_model = MLPRegressor(**params).fit(X_scaled, train["home_score"])
    away_model = MLPRegressor(**params).fit(X_scaled, train["away_score"])
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "scaler": scaler,
    }


def predict_mlp(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    return model["home_model"].predict(X_scaled), model["away_model"].predict(X_scaled)


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_mlp, predict_mlp, min_train_seasons=2)
    metrics = score_predictions(results)
    print("MLP walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/mlp_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
