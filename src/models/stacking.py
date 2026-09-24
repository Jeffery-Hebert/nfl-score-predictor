"""
Stacking meta-model: ridge regression trained on out-of-fold predictions
from the three models that actually demonstrated real out-of-sample value
(Linear, Poisson, GP) -- the other 7 base models are retained in
src/models/unused/ for research/diagnostic purposes but excluded here
since they showed no measurable improvement over these three.

All three inputs share the same weekly retraining cadence, so there's no
cross-model staleness asymmetry to account for (unlike the earlier version
of this file, which mixed weekly and season-level base models).

Uses its own walk-forward split (lighter burn-in, since the input data is
already the reduced 2021-2025 test set) to avoid the meta-model overfitting
to the base predictions it's stacking.

Run: python -m src.models.stacking
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from src.models.common import chronological
from src.validate.backtest_io import predictions_path, save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

BASE_MODELS = ["linear", "poisson", "gp"]

# Ridge penalty for the meta-model. This was Ridge(alpha=1.0) -- regularized,
# but at a strength nobody chose. Same defect class as linear.py's complete
# absence of regularization, just less severe. alpha is now selected by
# RidgeCV from a TimeSeriesSplit INSIDE each training fold, so the test week
# never influences the choice.
ALPHAS = np.logspace(-2, 4, 25)
INNER_CV = 5


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
    # RidgeCV's TimeSeriesSplit needs oldest-first rows -- see chronological().
    train = chronological(train)
    home_cols = [c for c in train.columns if c.endswith("_home_pred")]
    away_cols = [c for c in train.columns if c.endswith("_away_pred")]

    scaler_home = StandardScaler().fit(train[home_cols])
    scaler_away = StandardScaler().fit(train[away_cols])

    def _ridge():
        return RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=INNER_CV))

    ridge_home = _ridge().fit(
        scaler_home.transform(train[home_cols]), train["home_score"]
    )
    ridge_away = _ridge().fit(
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
    inputs = [predictions_path(n) for n in BASE_MODELS if predictions_path(n).exists()]
    save_predictions(
        results, "stacking", inputs=inputs + ["data/raw/schedules.parquet"]
    )


if __name__ == "__main__":
    main()
