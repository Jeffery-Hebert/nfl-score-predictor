"""
CatBoost: handles missing values natively.

Run: python -m src.models.catboost_model
"""
import pandas as pd
from catboost import CatBoostRegressor
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

def fit_catboost(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    params = dict(iterations=200, depth=4, learning_rate=0.05, random_seed=42, verbose=False)
    home_model = CatBoostRegressor(**params).fit(X, train["home_score"])
    away_model = CatBoostRegressor(**params).fit(X, train["away_score"])
    return {"home_model": home_model, "away_model": away_model}

def predict_catboost(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS]
    return model["home_model"].predict(X), model["away_model"].predict(X)

def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_catboost, predict_catboost, min_train_seasons=2)
    metrics = score_predictions(results)
    print("CatBoost walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/catboost_predictions.parquet", index=False)

if __name__ == "__main__":
    main()