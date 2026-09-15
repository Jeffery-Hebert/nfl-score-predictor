"""
src/experiments/test_early_season_bias.py

Third and most targeted attempt at the early-season over-prediction.

What is established. The model is well calibrated on totals overall (+0.31)
but over-predicts them by roughly +1.1 in weeks 1-3. That is why Week 2 2026
came out 2.04 points above the market line: the model had absorbed an unusual
Week 1 (49.44 points a game against 2025's 45.96) off a 16-game sample.

What has already been ruled out. test_offseason_decay.py changed HOW form
decays across the offseason -- compressing the summer, and decaying by games
rather than days. Both nudged weeks 1-3 slightly better and weeks 4+ clearly
worse, and both were significantly WORSE overall (+0.036 to +0.069, CIs
excluding zero). The decay scheme is not the lever.

What this tries instead. The existing correction in common.py measures how much
the model over-predicts on the most recent 285 training games and subtracts it.
That number is dominated by mid-season football, so it cannot see an
early-season-specific offset. This computes a SECOND offset from historical
early-season games only, and applies it when the game being predicted is itself
early-season.

Leakage-safe: both offsets come from training rows. The week number of the game
being predicted is schedule information, known long before kickoff.

The honest risk, stated up front: weeks 1-3 are a thin slice. Roughly 336 such
games exist in a full training set, so the standard error on the offset is
about 0.5 points against an effect near 1.0. This may well be too noisy to
help, which is exactly what the bootstrap is for.

Run: python -m src.experiments.test_early_season_bias
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import BIAS_CORRECTION_GAMES, FEATURE_COLS, ot_sample_weight
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

ALPHAS = np.logspace(-2, 4, 25)
N_BOOT = 5000
SEED = 42
EARLY_WEEKS = 3
MIN_EARLY_GAMES = 150  # below this the offset is too noisy to trust


def offsets(train, ph, pa, week_aware):
    """(general, early) offset pairs. early is None when unused or too thin."""
    rh = ph - train["home_score"].to_numpy(float)
    ra = pa - train["away_score"].to_numpy(float)
    order = np.argsort(train["gameday"].to_numpy())
    k = min(BIAS_CORRECTION_GAMES, len(train))
    recent = order[-k:]
    general = (float(rh[recent].mean()), float(ra[recent].mean()))

    if not week_aware:
        return general, None
    early_mask = (train["week"] <= EARLY_WEEKS).to_numpy()
    if early_mask.sum() < MIN_EARLY_GAMES:
        return general, None
    return general, (float(rh[early_mask].mean()), float(ra[early_mask].mean()))


def make_fns(kind, week_aware):
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
            sc, ph, pa = None, h.predict(Xf), a.predict(Xf)
        else:
            sc = StandardScaler().fit(Xf)
            Xs = sc.transform(Xf)
            h = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            )
            a = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            )
            ph, pa = h.predict(Xs), a.predict(Xs)
        gen, early = offsets(train, ph, pa, week_aware)
        return {
            "h": h,
            "a": a,
            "means": means,
            "scaler": sc,
            "gen": gen,
            "early": early,
        }

    def predict(m, test):
        X = test[FEATURE_COLS].fillna(m["means"])
        if m["scaler"] is not None:
            X = m["scaler"].transform(X)
        ph, pa = m["h"].predict(X), m["a"].predict(X)
        gh, ga = m["gen"]
        oh = np.full(len(test), gh)
        oa = np.full(len(test), ga)
        if m["early"] is not None:
            is_early = (test["week"] <= EARLY_WEEKS).to_numpy()
            oh[is_early] = m["early"][0]
            oa[is_early] = m["early"][1]
        return ph - oh, pa - oa

    return fit, predict


def bootstrap(base, cand, subset=None):
    m = base.merge(cand, on="game_id", suffixes=("_b", "_c"))
    if subset is not None:
        m = m[subset(m)]
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

    def mean_rmse(d):
        return (
            np.sqrt(((d.home_pred - d.home_score) ** 2).mean())
            + np.sqrt(((d.away_pred - d.away_score) ** 2).mean())
        ) / 2

    for name, kind in [("Linear (RidgeCV)", "ridge"), ("Poisson GLM", "poisson")]:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(
            f"  {'correction':<26}{'all wk':>9}{'wk 1-3':>9}{'wk 4+':>9}"
            f"{'wk1-3 total bias':>19}"
        )
        print("  " + "-" * 72)
        res = {}
        for label, wa in [
            ("general only (current)", False),
            ("+ early-season offset", True),
        ]:
            fit, predict = make_fns(kind, wa)
            r = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            res[label] = r
            early, late = r[r.week <= EARLY_WEEKS], r[r.week > EARLY_WEEKS]
            etb = (
                (early.home_pred + early.away_pred)
                - (early.home_score + early.away_score)
            ).mean()
            print(
                f"  {label:<26}{mean_rmse(r):>9.4f}{mean_rmse(early):>9.4f}"
                f"{mean_rmse(late):>9.4f}{etb:>+19.3f}"
            )

        base = res["general only (current)"]
        cand = res["+ early-season offset"]
        print(f"\n  Paired bootstrap ({N_BOOT} resamples over games)\n")
        for tag, sub in [
            ("all games", None),
            ("weeks 1-3 only", lambda m: m.week_b <= EARLY_WEEKS),
        ]:
            mean, lo, hi = bootstrap(base, cand, sub)
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {tag:<18} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
