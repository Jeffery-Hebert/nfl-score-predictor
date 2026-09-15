"""
src/experiments/test_training_recency.py

Fixes the away-score bias by weighting TRAINING GAMES by recency.

The diagnosis. Every model over-predicts away scores by roughly +0.8 points
while home scores are nearly unbiased. That is not random error, it is drift:

    era          avg away score    home-field edge
    2019-2021        23.16             +0.74
    2024-2026        21.98             +2.12

2019 and 2020 were the empty-stadium seasons, where home-field advantage
essentially disappeared (+0.04 and +0.17). Away scoring has fallen 1.18 points
since, and home-field has nearly tripled.

The pipeline already weights FEATURES by recency (17-week half-life), but the
model FIT weights every training game equally. A 2019 game in an empty stadium
therefore influences a 2026 prediction exactly as much as last week's game.
That pulls away predictions up and the home-field estimate down -- precisely
the observed bias.

The fix is the same idea already used for features, applied to the fit: weight
each training game by how old it is relative to the most recent training game.
Leakage-safe -- the weight depends only on a training game's own date.

Sweeps the half-life, because "how fast should the league's scoring environment
be forgotten" is an empirical question, not a guess. Reports BIAS as well as
RMSE, since fixing the bias is the goal even if RMSE barely moves.

Run: python -m src.experiments.test_training_recency
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import FEATURE_COLS, ot_sample_weight
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42
ALPHAS = np.logspace(-2, 4, 25)

# None = current behaviour (every training game weighted equally).
HALFLIVES_WEEKS = [None, 104, 78, 52, 34, 17]


def recency_weight(train, halflife_weeks):
    """Weight each training game by age relative to the newest training game."""
    if halflife_weeks is None:
        return None
    days = (train["gameday"].max() - train["gameday"]).dt.days.to_numpy(float)
    return 0.5 ** (days / (halflife_weeks * 7))


def combined_weight(train, halflife_weeks):
    """Recency x the existing overtime down-weight."""
    rec = recency_weight(train, halflife_weeks)
    ot = ot_sample_weight(train)
    if rec is None:
        return ot
    return rec if ot is None else rec * ot


def linear_fns(halflife_weeks):
    def make():
        return make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))
        )

    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        w = combined_weight(train, halflife_weeks)
        h, a = make(), make()
        h.fit(Xf, train["home_score"], ridgecv__sample_weight=w)
        a.fit(Xf, train["away_score"], ridgecv__sample_weight=w)
        return {"h": h, "a": a, "means": means}

    def predict(m, test):
        X = test[FEATURE_COLS].fillna(m["means"])
        return m["h"].predict(X), m["a"].predict(X)

    return fit, predict


def poisson_fns(halflife_weeks):
    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        sc = StandardScaler().fit(Xf)
        Xs = sc.transform(Xf)
        w = combined_weight(train, halflife_weeks)
        return {
            "h": PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            ),
            "a": PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            ),
            "means": means,
            "scaler": sc,
        }

    def predict(m, test):
        Xs = m["scaler"].transform(test[FEATURE_COLS].fillna(m["means"]))
        return m["h"].predict(Xs), m["a"].predict(Xs)

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
    df = pd.read_parquet("data/processed/model_table.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])

    for name, maker in [("Linear (RidgeCV)", linear_fns), ("Poisson GLM", poisson_fns)]:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(
            f"  {'training half-life':<22}{'home':>8}{'away':>8}{'mean':>8}"
            f"{'home bias':>11}{'away bias':>11}"
        )
        print("  " + "-" * 68)
        results = {}
        for hl in HALFLIVES_WEEKS:
            label = "none (current)" if hl is None else f"{hl} weeks"
            fit, predict = maker(hl)
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            results[label] = res
            m = score_predictions(res)
            hb = (res.home_pred - res.home_score).mean()
            ab = (res.away_pred - res.away_score).mean()
            print(
                f"  {label:<22}{m['home_rmse']:>8.4f}{m['away_rmse']:>8.4f}"
                f"{(m['home_rmse']+m['away_rmse'])/2:>8.4f}{hb:>+11.3f}{ab:>+11.3f}"
            )

        base = results["none (current)"]
        print(f"\n  Paired bootstrap vs current ({N_BOOT} resamples over games)\n")
        for label, res in results.items():
            if label == "none (current)":
                continue
            mean, lo, hi = bootstrap(base, res)
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {label:<22} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
