"""
src/experiments/test_pass_rush_cpoe_all_models.py

Extends the pass/rush/CPOE experiment to Linear, Poisson, and (optionally)
GP. GP is expensive (~20 min per fit x 2 runs) -- controlled by RUN_GP flag.

Run: python -m src.experiments.test_pass_rush_cpoe_all_models
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

RUN_GP = True  # set True only when you're ready for the ~40+ min cost

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
KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)


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


def make_gp_fns(feature_cols):
    def fit_fn(train):
        X = train[feature_cols]
        means = X.mean()
        X_filled = X.fillna(means)
        scaler = StandardScaler().fit(X_filled)
        X_scaled = scaler.transform(X_filled)
        return {
            "home_model": GaussianProcessRegressor(
                kernel=KERNEL, normalize_y=True, random_state=42
            ).fit(X_scaled, train["home_score"]),
            "away_model": GaussianProcessRegressor(
                kernel=KERNEL, normalize_y=True, random_state=42
            ).fit(X_scaled, train["away_score"]),
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


def run_model(name, make_fns, df, feature_cols):
    fit_fn, predict_fn = make_fns(feature_cols)
    results = walk_forward_evaluate(df, fit_fn, predict_fn, min_train_seasons=2)
    return score_predictions(results)


def report(name, base_metrics, ext_metrics):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("BASE FEATURES (production FEATURE_COLS only):")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("WITH PASS/RUSH SPLIT + CPOE:")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("DELTA (negative = improvement):")
    for metric in ["home_rmse", "away_rmse", "home_mae", "away_mae"]:
        print(f"  {metric}: {ext_metrics[metric] - base_metrics[metric]:+.4f}")


def main():
    df = build_merged_table()
    extended_cols = FEATURE_COLS + NEW_FEATURE_COLS

    for name, make_fns in [
        ("Linear Regression", make_linear_fns),
        ("Poisson GLM", make_poisson_fns),
    ]:
        base = run_model(name, make_fns, df, FEATURE_COLS)
        ext = run_model(name, make_fns, df, extended_cols)
        report(name, base, ext)

    if RUN_GP:
        base = run_model("Gaussian Process", make_gp_fns, df, FEATURE_COLS)
        ext = run_model("Gaussian Process", make_gp_fns, df, extended_cols)
        report("Gaussian Process", base, ext)
    else:
        print(
            "\nGaussian Process skipped (RUN_GP=False) -- set to True to include it (~40+ min)."
        )


if __name__ == "__main__":
    main()