"""
src/experiments/test_adjusted_ratings_value.py

Tests whether opponent-adjusted EPA ratings (replacing raw off_epa_per_play/
def_epa_per_play_allowed) improve Linear/Poisson out-of-sample accuracy.

Run: python -m src.experiments.test_adjusted_ratings_value
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

BLENDED_COLS_TO_REMOVE = [
    "home_pregame_off_epa_per_play",
    "home_pregame_def_epa_per_play",
    "away_pregame_off_epa_per_play",
    "away_pregame_def_epa_per_play",
]
ADJ_STAT_COLS = ["pregame_adjusted_off_epa", "pregame_adjusted_def_epa_allowed"]
ADJ_FEATURE_COLS = [f"home_{c}" for c in ADJ_STAT_COLS] + [
    f"away_{c}" for c in ADJ_STAT_COLS
]


def build_merged_table() -> pd.DataFrame:
    model_table = pd.read_parquet("data/processed/model_table.parquet")
    ratings = pd.read_parquet("data/processed/adjusted_ratings.parquet")

    home = ratings.rename(columns={c: f"home_{c}" for c in ADJ_STAT_COLS})
    home = home.rename(columns={"team": "home_team"})
    away = ratings.rename(columns={c: f"away_{c}" for c in ADJ_STAT_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = model_table.merge(home, on=["season", "week", "home_team"], how="left")
    merged = merged.merge(away, on=["season", "week", "away_team"], how="left")
    return merged


def make_linear_fns(feature_cols):
    def fit_fn(train):
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        return {
            "home_model": LinearRegression().fit(X_filled, train["home_score"]),
            "away_model": LinearRegression().fit(X_filled, train["away_score"]),
            "means": means,
            "feature_cols": feature_cols,
        }

    def predict_fn(model, test):
        X = test[model["feature_cols"]].fillna(model["means"])
        return model["home_model"].predict(X), model["away_model"].predict(X)

    return fit_fn, predict_fn


def make_poisson_fns(feature_cols):
    def fit_fn(train):
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        scaler = StandardScaler().fit(X_filled)
        X_scaled = scaler.transform(X_filled)
        return {
            "home_model": PoissonRegressor(max_iter=300).fit(
                X_scaled, train["home_score"]
            ),
            "away_model": PoissonRegressor(max_iter=300).fit(
                X_scaled, train["away_score"]
            ),
            "means": means,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

    def predict_fn(model, test):
        X = test[model["feature_cols"]].fillna(model["means"])
        X_scaled = model["scaler"].transform(X)
        return model["home_model"].predict(X_scaled), model["away_model"].predict(
            X_scaled
        )

    return fit_fn, predict_fn


def run_model(make_fns, df, feature_cols):
    fit_fn, predict_fn = make_fns(feature_cols)
    results = walk_forward_evaluate(df, fit_fn, predict_fn, min_train_seasons=2)
    return score_predictions(results), results


def report(name, base_metrics, ext_metrics):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("BASE (raw EPA):")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("WITH ADJUSTED RATINGS:")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("DELTA (negative = improvement):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        print(f"  {metric}: {ext_metrics[metric] - base_metrics[metric]:+.4f}")


def main():
    df = build_merged_table()
    adjusted_cols = [
        c for c in FEATURE_COLS if c not in BLENDED_COLS_TO_REMOVE
    ] + ADJ_FEATURE_COLS

    for name, make_fns in [
        ("Linear Regression", make_linear_fns),
        ("Poisson GLM", make_poisson_fns),
    ]:
        base_metrics, base_results = run_model(make_fns, df, FEATURE_COLS)
        ext_metrics, ext_results = run_model(make_fns, df, adjusted_cols)
        report(name, base_metrics, ext_metrics)
        base_results.to_parquet(
            f"data/processed/adjrating_base_{name.split()[0].lower()}.parquet",
            index=False,
        )
        ext_results.to_parquet(
            f"data/processed/adjrating_ext_{name.split()[0].lower()}.parquet",
            index=False,
        )


if __name__ == "__main__":
    main()
