"""
Bayesian hierarchical model: team-level random intercepts (partial pooling
across teams) layered on top of the same fixed-effect features used
elsewhere, plus a home-field fixed effect.

Uses ADVI (variational inference), not full MCMC sampling -- MCMC would be
too slow to refit ~145 times in a walk-forward loop on this hardware.
This is a deliberate speed/precision tradeoff, not an oversight.

----------------------------------------------------- fixed 2026-09-24

The first version never converged, and its -2.50 home / -0.71 away bias was
that non-convergence, not a property of the model. It fitted the RAW score
(mean ~23.5) starting the intercept at its prior mean of 20, and ran 3,000
iterations of ADVI's default optimiser -- far too few for the intercept to
travel 3.5 points. Home scores sat furthest from 20, hence the larger home bias.
On synthetic data with a known answer the old settings recovered team effects of
+/-3 as +/-0.6 and missed the mean by -1.8.

Three changes, each measured on that synthetic check (bias / RMSE to truth):
  - fit a CENTRED, SCALED target and translate the priors onto that scale, so
    the intercept starts at the answer instead of 3.5 points away;
  - Adam at 0.005 for 30,000 steps: -1.797 / 3.27  ->  -0.093 / 1.49, better
    than the unpooled least-squares reference (1.68), as partial pooling should
    be;
  - read posterior means BY NAME from approx.sample(). The old code sliced a
    flattened parameter vector by position, which happened to be right in this
    PyMC version but silently breaks if the ordering ever changes.

Overtime down-weighting (C5) is not applied: that would need a weighted
likelihood. The drift offset every other model carries is applied.

Run: python -m src.models.unused.bayesian_hierarchical
"""

import numpy as np
import pandas as pd
import pytensor

# PyTensor compiles through numba and caches the result under ~/.pytensor.
# Some cached file names run to ~145 characters, and an encrypted home
# directory (ecryptfs, as on the operator's machine) caps names at 143 -- the
# write fails with "File name too long" and takes the fit down with it.
# Compiling without the cache costs a few seconds per process and works on any
# filesystem.
pytensor.config.numba__cache = False

import pymc as pm  # noqa: E402  (must follow the pytensor config above)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.common import FEATURE_COLS, recent_residual_offset  # noqa: E402
from src.validate.backtest_io import save_predictions  # noqa: E402
from src.validate.walk_forward import (  # noqa: E402
    score_predictions,
    walk_forward_evaluate_by_season,
)

MODEL_TABLE = "data/processed/model_table.parquet"
ADVI_STEPS = 30_000
ADVI_LEARNING_RATE = 0.005
POSTERIOR_DRAWS = 1000
SEED = 42


def _fit_one_target(train: pd.DataFrame, target_col: str, team_col: str):
    means = train[FEATURE_COLS].mean()
    X_raw = train[FEATURE_COLS].fillna(means)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    y_raw = train[target_col].to_numpy(float)
    y_mean, y_sd = float(y_raw.mean()), float(y_raw.std())
    y = (y_raw - y_mean) / y_sd
    teams = train[team_col].astype("category")
    team_idx = teams.cat.codes.to_numpy()
    n_teams = len(teams.cat.categories)

    # The original priors, in points, translated onto the standardised scale:
    # beta ~ N(0, 5 pts), tau ~ HalfN(5 pts), sigma ~ HalfN(10 pts). The
    # intercept's prior is centred on the data, which is what centring means.
    with pm.Model():
        beta = pm.Normal("beta", 0, 5 / y_sd, shape=X.shape[1])
        intercept = pm.Normal("intercept", 0, 10 / y_sd)
        tau = pm.HalfNormal("tau", 5 / y_sd)
        team_effect = pm.Normal("team_effect", 0, tau, shape=n_teams)
        sigma = pm.HalfNormal("sigma", 10 / y_sd)

        mu = intercept + pm.math.dot(X, beta) + team_effect[team_idx]
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        approx = pm.fit(
            n=ADVI_STEPS,
            method="advi",
            obj_optimizer=pm.adam(learning_rate=ADVI_LEARNING_RATE),
            progressbar=False,
            random_seed=SEED,
        )
    post = approx.sample(POSTERIOR_DRAWS, random_seed=SEED).posterior

    def mean_of(name):
        return post[name].mean(("chain", "draw")).to_numpy()

    return {
        "beta": mean_of("beta") * y_sd,
        "intercept": float(mean_of("intercept")) * y_sd + y_mean,
        "team_effect": mean_of("team_effect") * y_sd,
        "team_categories": list(teams.cat.categories),
        "means": means,
        "scaler": scaler,
    }


def _predict_one(model: dict, test: pd.DataFrame, team_col: str):
    X_raw = test[FEATURE_COLS].fillna(model["means"])
    X = model["scaler"].transform(X_raw)
    pred = model["intercept"] + X @ model["beta"]
    team_lookup = {t: i for i, t in enumerate(model["team_categories"])}
    effects = np.array(
        [
            model["team_effect"][team_lookup[t]] if t in team_lookup else 0.0
            for t in test[team_col]
        ]
    )
    return pred + effects


def fit_bayesian(train: pd.DataFrame) -> dict:
    home_model = _fit_one_target(train, "home_score", "home_team")
    away_model = _fit_one_target(train, "away_score", "away_team")
    model = {"home_model": home_model, "away_model": away_model}
    h = _predict_one(home_model, train, "home_team")
    a = _predict_one(away_model, train, "away_team")
    model["off_h"], model["off_a"] = recent_residual_offset(train, h, a)
    return model


def predict_bayesian(model: dict, test: pd.DataFrame):
    home_pred = _predict_one(model["home_model"], test, "home_team")
    away_pred = _predict_one(model["away_model"], test, "away_team")
    return (
        home_pred - model.get("off_h", 0.0),
        away_pred - model.get("off_a", 0.0),
    )


def main():
    df = pd.read_parquet(MODEL_TABLE)
    print("Running Bayesian hierarchical model -- ADVI per fold, several minutes.")
    results = walk_forward_evaluate_by_season(
        df, fit_bayesian, predict_bayesian, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("Bayesian Hierarchical walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "bayesian", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
