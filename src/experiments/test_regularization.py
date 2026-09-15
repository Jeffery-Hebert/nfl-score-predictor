"""
src/experiments/test_regularization.py

The production "Linear Regression" model is sklearn LinearRegression -- plain
OLS with no regularization -- fitted on 18 features with roughly 1,900
training rows. test_advanced_metrics.py showed it degrading MONOTONICALLY as
features were added (base 9.4110 -> +PACE 9.4359 -> +ALL 9.5302), which is an
overfitting signature rather than evidence the features carry no information.

Poisson, whose PoissonRegressor defaults to alpha=1.0 and is therefore already
regularized, degraded far less under the same features. That is the tell.

This tests two things:
  1. Does regularizing the linear model help ON ITS OWN, with no new features?
  2. Once regularized, can it then absorb the advanced metrics?

alpha selection is leakage-safe: RidgeCV picks it with a TimeSeriesSplit
INSIDE each training fold, so the test week is never involved in the choice.
Features are standardized inside the fold for the same reason.

Active benchmark (post-C1-C8, n=1426, mean home/away RMSE):
  baseline 9.4415   linear 9.4110   poisson 9.3920

Run: python -m src.experiments.test_regularization
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.test_advanced_metrics import ALL_STATS, build_table, sided
from src.models.common import FEATURE_COLS, ot_sample_weight
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42
ALPHAS = np.logspace(-2, 4, 25)


def make_estimator(kind, alpha=None):
    if kind == "ols":
        return make_pipeline(StandardScaler(), LinearRegression())
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if kind == "ridgecv":
        # TimeSeriesSplit keeps alpha selection chronological within the fold.
        return make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))
        )
    raise ValueError(kind)


def fns(cols, kind, alpha=None):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)
        h = make_estimator(kind, alpha)
        a = make_estimator(kind, alpha)
        step = h.steps[-1][0]
        h.fit(Xf, train["home_score"], **{f"{step}__sample_weight": w})
        a.fit(Xf, train["away_score"], **{f"{step}__sample_weight": w})
        return {"h": h, "a": a, "means": means, "cols": cols}

    def predict(m, test):
        X = test[m["cols"]].fillna(m["means"])
        return m["h"].predict(X), m["a"].predict(X)

    return fit, predict


def bootstrap(base, cand):
    m = base.merge(cand, on="game_id", suffixes=("_b", "_c"))
    eb = np.stack(
        [
            (m.home_pred_b - m.home_score_b).to_numpy(float),
            (m.away_pred_b - m.away_score_b).to_numpy(float),
        ]
    )
    ec = np.stack(
        [
            (m.home_pred_c - m.home_score_c).to_numpy(float),
            (m.away_pred_c - m.away_score_c).to_numpy(float),
        ]
    )
    rng = np.random.default_rng(SEED)
    n = eb.shape[1]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        k = rng.integers(0, n, n)
        d[i] = np.sqrt((ec[:, k] ** 2).mean()) - np.sqrt((eb[:, k] ** 2).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    df = build_table()
    adv_cols = FEATURE_COLS + sided(ALL_STATS)

    variants = [
        ("OLS, base features (production)", FEATURE_COLS, "ols", None),
        ("Ridge a=1, base", FEATURE_COLS, "ridge", 1.0),
        ("Ridge a=10, base", FEATURE_COLS, "ridge", 10.0),
        ("Ridge a=100, base", FEATURE_COLS, "ridge", 100.0),
        ("RidgeCV, base", FEATURE_COLS, "ridgecv", None),
        ("OLS, base+advanced", adv_cols, "ols", None),
        ("RidgeCV, base+advanced", adv_cols, "ridgecv", None),
        ("Ridge a=100, base+advanced", adv_cols, "ridge", 100.0),
    ]

    print(
        f"{len(df)} games. Benchmark: linear(OLS) 9.4110, poisson 9.3920, baseline 9.4415\n"
    )
    results = {}
    for label, cols, kind, alpha in variants:
        fit, predict = fns(cols, kind, alpha)
        res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
        results[label] = res
        m = score_predictions(res)
        print(
            f"  {label:<34} home={m['home_rmse']:.4f} away={m['away_rmse']:.4f} "
            f"mean={(m['home_rmse']+m['away_rmse'])/2:.4f}  (n={len(cols)} feats)"
        )

    base = results["OLS, base features (production)"]
    print(f"\n  Paired bootstrap vs production OLS ({N_BOOT} resamples over games)\n")
    for label, res in results.items():
        if label.startswith("OLS, base features"):
            continue
        mean, lo, hi = bootstrap(base, res)
        v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
        print(
            f"    {label:<34} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
        )


if __name__ == "__main__":
    main()
