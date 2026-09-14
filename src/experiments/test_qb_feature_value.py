"""
src/experiments/test_qb_feature_value.py

Phase 1 falsifiable test: does adding starting-QB pregame features improve
Linear Regression's out-of-sample RMSE over the existing team-level-only
FEATURE_COLS? Standalone script -- does not modify model_table.parquet,
linear.py, or common.py. If this doesn't show a meaningful improvement,
we stop here and don't touch Poisson/GP/Stacking.

Run: python -m src.experiments.test_qb_feature_value
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

QB_STAT_COLS = [
    "pregame_qb_epa_per_play",
    "pregame_qb_completion_pct",
    "pregame_qb_success_rate",
    "prior_qb_games_started",
]
QB_FEATURE_COLS = [f"home_{c}" for c in QB_STAT_COLS] + [
    f"away_{c}" for c in QB_STAT_COLS
]


def build_merged_table() -> pd.DataFrame:
    model_table = pd.read_parquet("data/processed/model_table.parquet")
    qb = pd.read_parquet("data/processed/qb_rolling_features.parquet")

    home_qb = qb[["game_id", "team"] + QB_STAT_COLS].rename(
        columns={**{c: f"home_{c}" for c in QB_STAT_COLS}, "team": "home_team"}
    )
    away_qb = qb[["game_id", "team"] + QB_STAT_COLS].rename(
        columns={**{c: f"away_{c}" for c in QB_STAT_COLS}, "team": "away_team"}
    )
    merged = model_table.merge(home_qb, on=["game_id", "home_team"], how="left")
    merged = merged.merge(away_qb, on=["game_id", "away_team"], how="left")
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
    qb_coverage = df["home_prior_qb_games_started"].notna().mean()
    print(
        f"QB feature coverage: {qb_coverage:.1%} of games have an identified home starter\n"
    )

    baseline_results = walk_forward_evaluate(
        df, make_fit_fn(FEATURE_COLS), predict_fn, min_train_seasons=2
    )
    baseline_metrics = score_predictions(baseline_results)

    qb_feature_cols = FEATURE_COLS + QB_FEATURE_COLS
    qb_results = walk_forward_evaluate(
        df, make_fit_fn(qb_feature_cols), predict_fn, min_train_seasons=2
    )
    qb_metrics = score_predictions(qb_results)

    print("BASELINE (team-level features only):")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nWITH QB FEATURES:")
    for k, v in qb_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nDELTA (negative = QB features improved RMSE):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        delta = qb_metrics[metric] - baseline_metrics[metric]
        print(f"  {metric}: {delta:+.4f}")


if __name__ == "__main__":
    run_comparison()
