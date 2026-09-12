"""
Random Forest: 200 trees, scale-invariant, but sklearn's implementation
requires imputed inputs (no native NaN support), unlike the GBM libraries.

Run: python -m src.models.random_forest
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

def fit_rf(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    home_model = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42, n_jobs=-1).fit(X_filled, train["home_score"])
    away_model = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42, n_jobs=-1).fit(X_filled, train["away_score"])
    return {"home_model": home_model, "away_model": away_model, "means": means}

def predict_rf(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    return model["home_model"].predict(X), model["away_model"].predict(X)

def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_rf, predict_rf, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Random Forest walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/rf_predictions.parquet", index=False)

if __name__ == "__main__":
    main()