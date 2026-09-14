"""
LightGBM: handles missing values natively.

Run: python -m src.models.lightgbm_model
"""

import pandas as pd
from lightgbm import LGBMRegressor
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions


def fit_lgbm(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    params = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    home_model = LGBMRegressor(**params).fit(X, train["home_score"])
    away_model = LGBMRegressor(**params).fit(X, train["away_score"])
    return {"home_model": home_model, "away_model": away_model}


def predict_lgbm(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS]
    return model["home_model"].predict(X), model["away_model"].predict(X)


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_lgbm, predict_lgbm, min_train_seasons=2)
    metrics = score_predictions(results)
    print("LightGBM walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/lgbm_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
