"""
src/experiments/test_pass_rush_split_only.py

Tests pass/rush EPA decomposition in isolation: REPLACES blended
off_epa_per_play/def_epa_per_play with off_pass_epa_per_play +
off_rush_epa_per_play (and defensive equivalents). No CPOE involved.
This avoids the collinearity flaw from the earlier combined test.

Run: python -m src.experiments.test_pass_rush_split_only
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

BLENDED_COLS_TO_REMOVE = [
    "home_pregame_off_epa_per_play", "home_pregame_def_epa_per_play",
    "away_pregame_off_epa_per_play", "away_pregame_def_epa_per_play",
]
SPLIT_STAT_COLS = [
    "pregame_off_pass_epa_per_play", "pregame_off_rush_epa_per_play",
    "pregame_def_pass_epa_per_play_allowed", "pregame_def_rush_epa_per_play_allowed",
]
SPLIT_FEATURE_COLS = [f"home_{c}" for c in SPLIT_STAT_COLS] + [f"away_{c}" for c in SPLIT_STAT_COLS]


def build_merged_table() -> pd.DataFrame:
    model_table = pd.read_parquet("data/processed/model_table.parquet")
    rolling = pd.read_parquet("data/processed/team_rolling_features.parquet")

    home = rolling[rolling["is_home"] == 1][["game_id", "team", "opponent"] + SPLIT_STAT_COLS]
    home = home.rename(columns={c: f"home_{c}" for c in SPLIT_STAT_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + SPLIT_STAT_COLS]
    away = away.rename(columns={c: f"away_{c}" for c in SPLIT_STAT_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = model_table.merge(home, on=["game_id", "home_team", "away_team"], how="left")
    merged = merged.merge(away, on=["game_id", "away_team"], how="left")
    return merged


def make_linear_fns(feature_cols):
    def fit_fn(train):
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        return {
            "home_model": LinearRegression().fit(X_filled, train["home_score"]),
            "away_model": LinearRegression().fit(X_filled, train["away_score"]),
            "means": means, "feature_cols": feature_cols,
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
            "home_model": PoissonRegressor(max_iter=300).fit(X_scaled, train["home_score"]),
            "away_model": PoissonRegressor(max_iter=300).fit(X_scaled, train["away_score"]),
            "means": means, "scaler": scaler, "feature_cols": feature_cols,
        }
    def predict_fn(model, test):
        X = test[model["feature_cols"]].fillna(model["means"])
        X_scaled = model["scaler"].transform(X)
        return model["home_model"].predict(X_scaled), model["away_model"].predict(X_scaled)
    return fit_fn, predict_fn


def run_model(make_fns, df, feature_cols):
    fit_fn, predict_fn = make_fns(feature_cols)
    results = walk_forward_evaluate(df, fit_fn, predict_fn, min_train_seasons=2)
    return score_predictions(results)


def report(name, base_metrics, ext_metrics):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("BASE (blended EPA):")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("WITH PASS/RUSH SPLIT (blend REPLACED, not added):")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("DELTA (negative = improvement):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        print(f"  {metric}: {ext_metrics[metric] - base_metrics[metric]:+.4f}")


def main():
    df = build_merged_table()
    split_feature_cols = [c for c in FEATURE_COLS if c not in BLENDED_COLS_TO_REMOVE] + SPLIT_FEATURE_COLS

    for name, make_fns in [("Linear Regression", make_linear_fns), ("Poisson GLM", make_poisson_fns)]:
        base = run_model(make_fns, df, FEATURE_COLS)
        ext = run_model(make_fns, df, split_feature_cols)
        report(name, base, ext)


if __name__ == "__main__":
    main()