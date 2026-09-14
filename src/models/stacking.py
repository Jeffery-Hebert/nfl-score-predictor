"""
Stacking meta-model: ridge regression trained on out-of-fold predictions
from all 10 base models. This is the 11th and final prediction.

Note: base models were evaluated at two different retraining cadences
(weekly for most, season-level for GP/Bayesian/RNN, for compute reasons).
All target the same 1,426 test games, so merging is valid, but the
season-level models' inputs are "staler" than the weekly ones -- a known
asymmetry, not hidden.

Uses its own walk-forward split (with lighter burn-in, since the input
data is already the reduced 2021-2025 test set) to avoid the meta-model
overfitting to the base predictions it's stacking.

Run: python -m src.models.stacking
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

BASE_MODELS = [
    "baseline",
    "linear",
    "poisson",
    "rf",
    "xgb",
    "lgbm",
    "catboost",
    "logistic",
    "mlp",
    "gp",
    "bayesian",
    "rnn",
    "montecarlo",
]


def load_meta_table() -> pd.DataFrame:
    schedules = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "season", "week", "gameday", "home_score", "away_score"]
    ]
    merged = None
    for name in BASE_MODELS:
        path = f"data/processed/{name}_predictions.parquet"
        try:
            preds = pd.read_parquet(path)[["game_id", "home_pred", "away_pred"]]
        except FileNotFoundError:
            print(f"WARNING: {path} not found, skipping {name} from the stack")
            continue
        preds = preds.rename(
            columns={"home_pred": f"{name}_home_pred", "away_pred": f"{name}_away_pred"}
        )
        merged = (
            preds if merged is None else merged.merge(preds, on="game_id", how="inner")
        )

    merged = merged.merge(schedules, on="game_id", how="left")
    return merged.dropna(subset=["home_score", "away_score"])


def fit_stack(train: pd.DataFrame) -> dict:
    home_cols = [c for c in train.columns if c.endswith("_home_pred")]
    away_cols = [c for c in train.columns if c.endswith("_away_pred")]

    scaler_home = StandardScaler().fit(train[home_cols])
    scaler_away = StandardScaler().fit(train[away_cols])

    ridge_home = Ridge(alpha=1.0).fit(
        scaler_home.transform(train[home_cols]), train["home_score"]
    )
    ridge_away = Ridge(alpha=1.0).fit(
        scaler_away.transform(train[away_cols]), train["away_score"]
    )

    return {
        "ridge_home": ridge_home,
        "ridge_away": ridge_away,
        "scaler_home": scaler_home,
        "scaler_away": scaler_away,
        "home_cols": home_cols,
        "away_cols": away_cols,
    }


def predict_stack(model: dict, test: pd.DataFrame):
    X_home = model["scaler_home"].transform(test[model["home_cols"]])
    X_away = model["scaler_away"].transform(test[model["away_cols"]])
    return model["ridge_home"].predict(X_home), model["ridge_away"].predict(X_away)


def main():
    df = load_meta_table()
    print(
        f"Meta-table built: {len(df)} games, {len(BASE_MODELS)} base models attempted"
    )
    results = walk_forward_evaluate(df, fit_stack, predict_stack, min_train_seasons=1)
    metrics = score_predictions(results)
    print("\nStacking Meta-Model walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/stacking_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
