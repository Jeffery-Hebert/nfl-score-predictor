"""
src/experiments/test_bias_correction.py

Second attempt at the away-score bias, after test_training_recency.py showed
the trade-off: down-weighting old training games reduces the bias but throws
away data, and anything shorter than about two seasons costs more accuracy
than it buys.

This is more surgical. The bias is a systematic OFFSET -- roughly +0.75 points
on away scores -- caused by the league's scoring environment drifting (away
scoring down 1.18 points since 2019-21, home-field advantage up from +0.74 to
+2.12 as empty-stadium seasons rolled out of relevance).

So instead of re-weighting every game, measure the offset and subtract it:

  1. Fit the model on the training fold as usual.
  2. Predict back onto RECENT training games.
  3. Average the residual on those games -- that is the current drift.
  4. Subtract it from the test predictions.

Leakage-safe: steps 2-4 read only training rows. The test week is never
touched. For a least-squares fit the residuals sum to zero over the WHOLE
training set, so this works precisely because it looks at a recent SUBSET,
where drift shows up.

Two weighting schemes are compared:
  last-K       plain average over the most recent K training games
  weighted     recency-weighted average over all training games, which uses
               everything but emphasises recent form

Reports bias as well as RMSE. Removing a systematic offset should improve
exact-score accuracy, which is the project's actual objective, even though the
effect on RMSE is modest: roughly bias^2 / (2 x RMSE).

Run: python -m src.experiments.test_bias_correction
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
GAMES_PER_SEASON = 285


def _residual_offset(train, pred_home, pred_away, mode, param):
    """Mean (prediction - actual) on recent training games, per side."""
    rh = pred_home - train["home_score"].to_numpy(float)
    ra = pred_away - train["away_score"].to_numpy(float)
    if mode == "lastk":
        k = min(param, len(train))
        order = np.argsort(train["gameday"].to_numpy())[-k:]
        return rh[order].mean(), ra[order].mean()
    if mode == "weighted":
        days = (train["gameday"].max() - train["gameday"]).dt.days.to_numpy(float)
        w = 0.5 ** (days / (param * 7))
        return np.average(rh, weights=w), np.average(ra, weights=w)
    raise ValueError(mode)


def make_fns(kind, correction=None):
    """correction: None, or (mode, param)."""

    def _ridge():
        return make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))
        )

    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)

        if kind == "ridge":
            h, a = _ridge(), _ridge()
            h.fit(Xf, train["home_score"], ridgecv__sample_weight=w)
            a.fit(Xf, train["away_score"], ridgecv__sample_weight=w)
            model = {"h": h, "a": a, "means": means, "scaler": None}
            ph, pa = h.predict(Xf), a.predict(Xf)
        else:
            sc = StandardScaler().fit(Xf)
            Xs = sc.transform(Xf)
            h = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            )
            a = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            )
            model = {"h": h, "a": a, "means": means, "scaler": sc}
            ph, pa = h.predict(Xs), a.predict(Xs)

        if correction is None:
            model["off_h"], model["off_a"] = 0.0, 0.0
        else:
            oh, oa = _residual_offset(train, ph, pa, *correction)
            model["off_h"], model["off_a"] = oh, oa
        return model

    def predict(m, test):
        X = test[FEATURE_COLS].fillna(m["means"])
        if m["scaler"] is not None:
            X = m["scaler"].transform(X)
        return m["h"].predict(X) - m["off_h"], m["a"].predict(X) - m["off_a"]

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


VARIANTS = [
    ("none (current)", None),
    ("last 1 season", ("lastk", GAMES_PER_SEASON)),
    ("last 2 seasons", ("lastk", 2 * GAMES_PER_SEASON)),
    ("weighted, 52wk", ("weighted", 52)),
    ("weighted, 104wk", ("weighted", 104)),
]


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])

    for name, kind in [("Linear (RidgeCV)", "ridge"), ("Poisson GLM", "poisson")]:
        print(f"\n{'=' * 76}\n{name}\n{'=' * 76}")
        print(
            f"  {'bias correction':<20}{'home':>8}{'away':>8}{'mean':>8}"
            f"{'home bias':>11}{'away bias':>11}{'margin':>9}"
        )
        print("  " + "-" * 70)
        results = {}
        for label, corr in VARIANTS:
            fit, predict = make_fns(kind, corr)
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            results[label] = res
            m = score_predictions(res)
            hb = (res.home_pred - res.home_score).mean()
            ab = (res.away_pred - res.away_score).mean()
            marg = np.sqrt(
                (
                    (
                        (res.home_pred - res.away_pred)
                        - (res.home_score - res.away_score)
                    )
                    ** 2
                ).mean()
            )
            print(
                f"  {label:<20}{m['home_rmse']:>8.4f}{m['away_rmse']:>8.4f}"
                f"{(m['home_rmse']+m['away_rmse'])/2:>8.4f}{hb:>+11.3f}{ab:>+11.3f}"
                f"{marg:>9.3f}"
            )

        base = results["none (current)"]
        print(f"\n  Paired bootstrap vs current ({N_BOOT} resamples over games)\n")
        for label, res in results.items():
            if label == "none (current)":
                continue
            mean, lo, hi = bootstrap(base, res)
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {label:<20} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
