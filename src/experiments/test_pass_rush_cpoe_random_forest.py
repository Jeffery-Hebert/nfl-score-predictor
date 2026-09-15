"""
src/experiments/test_pass_rush_cpoe_random_forest.py

Tests whether Random Forest -- which can learn interactions and thresholds,
unlike Linear/Poisson -- exploits the pass/rush split + CPOE features that
hurt Linear and Poisson. Same RF hyperparameters as src/models/unused/random_forest.py
to isolate the feature effect from any hyperparameter change.

Run: python -m src.experiments.test_pass_rush_cpoe_random_forest
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

BLENDED_COLS_TO_REMOVE = [
    "home_pregame_off_epa_per_play",
    "home_pregame_def_epa_per_play",
    "away_pregame_off_epa_per_play",
    "away_pregame_def_epa_per_play",
]
NEW_STAT_COLS = [
    "pregame_off_pass_epa_per_play",
    "pregame_off_rush_epa_per_play",
    "pregame_off_cpoe",
    "pregame_def_pass_epa_per_play_allowed",
    "pregame_def_rush_epa_per_play_allowed",
    "pregame_def_cpoe_allowed",
]
NEW_FEATURE_COLS = [f"home_{c}" for c in NEW_STAT_COLS] + [
    f"away_{c}" for c in NEW_STAT_COLS
]


def build_merged_table() -> pd.DataFrame:
    model_table = pd.read_parquet("data/processed/model_table.parquet")
    rolling = pd.read_parquet("data/processed/team_rolling_features.parquet")

    home = rolling[rolling["is_home"] == 1][
        ["game_id", "team", "opponent"] + NEW_STAT_COLS
    ]
    home = home.rename(columns={c: f"home_{c}" for c in NEW_STAT_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + NEW_STAT_COLS]
    away = away.rename(columns={c: f"away_{c}" for c in NEW_STAT_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = model_table.merge(
        home, on=["game_id", "home_team", "away_team"], how="left"
    )
    merged = merged.merge(away, on=["game_id", "away_team"], how="left")
    return merged


def make_rf_fns(feature_cols):
    def fit_fn(train):
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        home_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        ).fit(X_filled, train["home_score"])
        away_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        ).fit(X_filled, train["away_score"])
        return {
            "home_model": home_model,
            "away_model": away_model,
            "means": means,
            "feature_cols": feature_cols,
        }

    def predict_fn(model, test):
        X = test[model["feature_cols"]].fillna(model["means"])
        return model["home_model"].predict(X), model["away_model"].predict(X)

    return fit_fn, predict_fn


def run_model(make_fns, df, feature_cols):
    fit_fn, predict_fn = make_fns(feature_cols)
    results = walk_forward_evaluate(df, fit_fn, predict_fn, min_train_seasons=2)
    return score_predictions(results), results


def main():
    df = build_merged_table()
    extended_cols = [
        c for c in FEATURE_COLS if c not in BLENDED_COLS_TO_REMOVE
    ] + NEW_FEATURE_COLS

    print("Running baseline Random Forest (blended EPA, no CPOE)...")
    base_metrics, base_results = run_model(make_rf_fns, df, FEATURE_COLS)

    print("Running extended Random Forest (split + CPOE)...")
    ext_metrics, ext_results = run_model(make_rf_fns, df, extended_cols)

    print("\nBASE (blended EPA, no CPOE):")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nWITH SPLIT + CPOE:")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nDELTA (negative = improvement):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        print(f"  {metric}: {ext_metrics[metric] - base_metrics[metric]:+.4f}")

    base_results.to_parquet("data/processed/rf_base_predictions.parquet", index=False)
    ext_results.to_parquet(
        "data/processed/rf_extended_predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
