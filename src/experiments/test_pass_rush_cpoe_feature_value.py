"""
src/experiments/test_pass_rush_cpoe_feature_value.py

Phase 1 falsifiable test: does decomposing blended offensive/defensive EPA
into pass/rush components, plus adding CPOE, improve Linear Regression's
out-of-sample RMSE? Standalone -- does not modify model_table.parquet,
linear.py, or common.py.

Run: python -m src.experiments.test_pass_rush_cpoe_feature_value
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

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


def make_fit_fn(feature_cols):
    def fit_fn(train: pd.DataFrame) -> dict:
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        home_model = LinearRegression().fit(X_filled, train["home_score"])
        away_model = LinearRegression().fit(X_filled, train["away_score"])
        return {
            "home_model": home_model,
            "away_model": away_model,
            "means": means,
            "feature_cols": feature_cols,
        }

    return fit_fn


def predict_fn(model: dict, test: pd.DataFrame):
    X = test[model["feature_cols"]].fillna(model["means"])
    return model["home_model"].predict(X), model["away_model"].predict(X)


def run_comparison():
    df = build_merged_table()
    coverage = df["home_pregame_off_cpoe"].notna().mean()
    print(f"CPOE feature coverage: {coverage:.1%} of games\n")

    baseline_results = walk_forward_evaluate(
        df, make_fit_fn(FEATURE_COLS), predict_fn, min_train_seasons=2
    )
    baseline_metrics = score_predictions(baseline_results)

    extended_cols = FEATURE_COLS + NEW_FEATURE_COLS
    extended_results = walk_forward_evaluate(
        df, make_fit_fn(extended_cols), predict_fn, min_train_seasons=2
    )
    extended_metrics = score_predictions(extended_results)

    print("BASELINE:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nWITH PASS/RUSH SPLIT + CPOE:")
    for k, v in extended_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nDELTA (negative = improvement):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        print(f"  {metric}: {extended_metrics[metric] - baseline_metrics[metric]:+.4f}")


if __name__ == "__main__":
    run_comparison()
