"""
src/experiments/bootstrap_significance_test.py

Tests whether an RMSE delta between two feature sets is statistically
distinguishable from zero, using paired bootstrap resampling over games.
Answers: is "it got worse" real, or is it noise from a ~1,426-game test set?

Run: python -m src.experiments.bootstrap_significance_test
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate

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
    return merged.merge(away, on=["game_id", "away_team"], how="left")


def fit_fn_factory(feature_cols):
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


def bootstrap_rmse_delta(base_results, ext_results, n_boot=5000, seed=42):
    merged = base_results.merge(ext_results, on="game_id", suffixes=("_base", "_ext"))
    err_base = (merged["home_pred_base"] - merged["home_score_base"]).values
    err_ext = (merged["home_pred_ext"] - merged["home_score_ext"]).values
    n = len(merged)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rmse_base = np.sqrt((err_base[idx] ** 2).mean())
        rmse_ext = np.sqrt((err_ext[idx] ** 2).mean())
        deltas[i] = rmse_ext - rmse_base
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5])


def main():
    df = build_merged_table()
    base_cols = FEATURE_COLS
    split_cols = [c for c in FEATURE_COLS if c not in BLENDED_COLS_TO_REMOVE] + SPLIT_FEATURE_COLS

    base_fit, base_pred = fit_fn_factory(base_cols)
    ext_fit, ext_pred = fit_fn_factory(split_cols)

    base_results = walk_forward_evaluate(df, base_fit, base_pred, min_train_seasons=2)
    ext_results = walk_forward_evaluate(df, ext_fit, ext_pred, min_train_seasons=2)

    mean_delta, (ci_low, ci_high) = bootstrap_rmse_delta(base_results, ext_results)
    print(f"Home RMSE delta: {mean_delta:+.4f}")
    print(f"95% bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    if ci_low <= 0 <= ci_high:
        print("Zero falls within the CI -- this delta is NOT statistically distinguishable from noise.")
    else:
        print("Zero falls OUTSIDE the CI -- this delta is likely a real effect, not noise.")


if __name__ == "__main__":
    main()