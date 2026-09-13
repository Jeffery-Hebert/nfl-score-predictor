"""
Bayesian hierarchical model: team-level random intercepts (partial pooling
across teams) layered on top of the same fixed-effect features used
elsewhere, plus a home-field fixed effect.

Uses ADVI (variational inference), not full MCMC sampling -- MCMC would be
too slow to refit ~145 times in a walk-forward loop on this hardware.
This is a deliberate speed/precision tradeoff, not an oversight.

Run: python -m src.models.bayesian_hierarchical
"""

import pandas as pd
import numpy as np
import pymc as pm
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate_by_season, score_predictions
from sklearn.preprocessing import StandardScaler


def _fit_one_target(train: pd.DataFrame, target_col: str, team_col: str):
    means = train[FEATURE_COLS].mean()
    X_raw = train[FEATURE_COLS].fillna(means)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    y = train[target_col].values
    teams = train[team_col].astype("category")
    team_idx = teams.cat.codes.values
    n_teams = len(teams.cat.categories)

    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 5, shape=X.shape[1])
        intercept = pm.Normal("intercept", 20, 10)
        tau = pm.HalfNormal("tau", 5)
        team_effect = pm.Normal("team_effect", 0, tau, shape=n_teams)
        sigma = pm.HalfNormal("sigma", 10)

        mu = intercept + pm.math.dot(X, beta) + team_effect[team_idx]
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        approx = pm.fit(n=3000, method="advi", progressbar=False)

    return {
        "beta": approx.mean.eval()[: X.shape[1]],
        "intercept": approx.mean.eval()[X.shape[1]],
        "team_effect": approx.mean.eval()[X.shape[1] + 2 : X.shape[1] + 2 + n_teams],
        "team_categories": list(teams.cat.categories),
        "means": means,
        "scaler": scaler,
    }


def fit_bayesian(train: pd.DataFrame) -> dict:
    home_model = _fit_one_target(train, "home_score", "home_team")
    away_model = _fit_one_target(train, "away_score", "away_team")
    return {"home_model": home_model, "away_model": away_model}


def _predict_one(model: dict, test: pd.DataFrame, team_col: str):
    X_raw = test[FEATURE_COLS].fillna(model["means"])
    X = model["scaler"].transform(X_raw)
    pred = model["intercept"] + X @ model["beta"]
    team_lookup = {t: i for i, t in enumerate(model["team_categories"])}
    for i, team in enumerate(test[team_col]):
        if team in team_lookup:
            pred[i] += model["team_effect"][team_lookup[team]]
    return pred


def predict_bayesian(model: dict, test: pd.DataFrame):
    home_pred = _predict_one(model["home_model"], test, "home_team")
    away_pred = _predict_one(model["away_model"], test, "away_team")
    return home_pred, away_pred


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    print("Running Bayesian hierarchical model -- this is slower than other")
    print("models (ADVI per fold). Expect several minutes, not seconds.")
    results = walk_forward_evaluate_by_season(
        df, fit_bayesian, predict_bayesian, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("Bayesian Hierarchical walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/bayesian_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
