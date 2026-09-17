"""
src/experiments/test_pass_rush_split_v2.py

Measures the SECOND pass/rush split (volume-weighted, shrunk) against the
production feature set, on the production estimators.

Why this exists even though the feature has already shipped. The operator
authorised promotion regardless of the result, so this is not a gate -- it is
the record. A feature in production with no measured effect beside it is how a
project loses track of what its numbers mean, and the whole findings doc exists
to stop that.

Three feature sets, because the decision had two parts -- whether to add the
split, and whether to keep the blend next to it:

  BASE        production before any of this: 20 columns, blended off/def EPA,
              no split.
  ADD         blend AND split together, 28 columns. This is how the split first
              shipped on 2026-09-17.
  REPLACE     split INSTEAD of blend, 24 columns. This is what production runs
              now -- the blend was removed the same day because it and the split
              describe the same efficiency at different grains (r = 0.86 pass,
              0.58 rush on offence; 0.78 / 0.45 on defence).

REPLACE is structurally what v1 did. The difference is that v1 swapped one
low-variance measurement for two raw high-variance ones, where these are
volume-weighted and shrunk.

Blended EPA is no longer in model_table.parquet, so this script joins it back in
from team_rolling_features.parquet -- it needs columns production has dropped.

All of it runs on RidgeCV and PoissonRegressor, the estimators production
actually uses. That is the material difference from v1, which was measured on
unregularized LinearRegression against a 16-column base with no injury_impact.

Paired bootstrap over games, pooled home+away, 5,000 resamples -- the project's
established significance pattern.

Run: python -m src.experiments.test_pass_rush_split_v2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import (
    FEATURE_COLS,
    SPLIT_FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42
ALPHAS = np.logspace(-2, 4, 25)
INNER_CV = 5

SIDED_SPLIT_COLS = [f"home_{c}" for c in SPLIT_FEATURE_COLS] + [
    f"away_{c}" for c in SPLIT_FEATURE_COLS
]
# Blended EPA was dropped from BASE_FEATURE_COLS when the split shipped, so it
# is no longer a column of model_table.parquet. Re-joined by load_table().
BLEND_COLS = [
    f"{side}_pregame_{unit}_epa_per_play"
    for side in ("home", "away")
    for unit in ("off", "def")
]

REPLACE_COLS = list(FEATURE_COLS)  # production today
NO_SPLIT_COLS = [c for c in FEATURE_COLS if c not in SIDED_SPLIT_COLS]
BASE_COLS = NO_SPLIT_COLS + BLEND_COLS  # production before
ADD_COLS = list(FEATURE_COLS) + BLEND_COLS  # blend and split together


def load_table() -> pd.DataFrame:
    """model_table plus the blended EPA columns production no longer carries."""
    df = pd.read_parquet("data/processed/model_table.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])

    rolling = pd.read_parquet("data/processed/team_rolling_features.parquet")
    blend = ["pregame_off_epa_per_play", "pregame_def_epa_per_play"]
    home = rolling[rolling["is_home"] == 1][["game_id", "team"] + blend].rename(
        columns={**{c: f"home_{c}" for c in blend}, "team": "home_team"}
    )
    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + blend].rename(
        columns={**{c: f"away_{c}" for c in blend}, "team": "away_team"}
    )
    df = df.merge(home, on=["game_id", "home_team"], how="left")
    return df.merge(away, on=["game_id", "away_team"], how="left")


def make_ridge_fns(cols):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)

        def _m():
            return make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=INNER_CV)),
            )

        h = _m().fit(Xf, train["home_score"], ridgecv__sample_weight=w)
        a = _m().fit(Xf, train["away_score"], ridgecv__sample_weight=w)
        oh, oa = recent_residual_offset(train, h.predict(Xf), a.predict(Xf))
        return {"h": h, "a": a, "means": means, "oh": oh, "oa": oa, "cols": cols}

    def predict(m, test):
        X = test[m["cols"]].fillna(m["means"])
        return m["h"].predict(X) - m["oh"], m["a"].predict(X) - m["oa"]

    return fit, predict


def make_poisson_fns(cols):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        sc = StandardScaler().fit(Xf)
        Xs = sc.transform(Xf)
        w = ot_sample_weight(train)
        h = PoissonRegressor(alpha=1.0, max_iter=300).fit(
            Xs, train["home_score"], sample_weight=w
        )
        a = PoissonRegressor(alpha=1.0, max_iter=300).fit(
            Xs, train["away_score"], sample_weight=w
        )
        oh, oa = recent_residual_offset(train, h.predict(Xs), a.predict(Xs))
        return {
            "h": h,
            "a": a,
            "means": means,
            "sc": sc,
            "oh": oh,
            "oa": oa,
            "cols": cols,
        }

    def predict(m, test):
        Xs = m["sc"].transform(test[m["cols"]].fillna(m["means"]))
        return m["h"].predict(Xs) - m["oh"], m["a"].predict(Xs) - m["oa"]

    return fit, predict


def make_ols_fns(cols):
    """v1's estimator, kept so the old comparison can be reproduced exactly."""

    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        return {
            "h": LinearRegression().fit(Xf, train["home_score"]),
            "a": LinearRegression().fit(Xf, train["away_score"]),
            "means": means,
            "cols": cols,
        }

    def predict(m, test):
        X = test[m["cols"]].fillna(m["means"])
        return m["h"].predict(X), m["a"].predict(X)

    return fit, predict


def run(df, make_fns, cols):
    fit, predict = make_fns(cols)
    return walk_forward_evaluate(df, fit, predict, min_train_seasons=2)


def paired_bootstrap(base: pd.DataFrame, cand: pd.DataFrame):
    """Pooled home+away RMSE delta. Negative = the candidate is better."""
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
    return float(d.mean()), float(lo), float(hi), float((d < 0).mean())


def pooled_rmse(r):
    m = score_predictions(r)
    return (m["home_rmse"] + m["away_rmse"]) / 2


def report(label, base_r, cand_r):
    b, c = pooled_rmse(base_r), pooled_rmse(cand_r)
    mean, lo, hi, p_better = paired_bootstrap(base_r, cand_r)
    verdict = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
    print(f"\n  {label}")
    print(f"    base {b:.4f}  ->  candidate {c:.4f}   ({c - b:+.4f})")
    print(
        f"    bootstrap delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"-> {verdict}"
    )
    print(f"    P(candidate better) = {p_better:.1%}")
    return {"label": label, "base": b, "cand": c, "delta": mean, "lo": lo, "hi": hi}


def main():
    df = load_table()

    print("=" * 78)
    print("PASS/RUSH SPLIT v2 -- volume-weighted and shrunk, measured after the fact")
    print("=" * 78)
    print(
        f"  BASE    {len(BASE_COLS):>3} cols  blended EPA, no split (production before)"
    )
    print(f"  ADD     {len(ADD_COLS):>3} cols  blend AND split")
    print(
        f"  REPLACE {len(REPLACE_COLS):>3} cols  split INSTEAD of blend (production now)"
    )
    print(f"  games   {df['home_score'].notna().sum()}")

    rows = []
    for name, fns in (("RidgeCV", make_ridge_fns), ("Poisson", make_poisson_fns)):
        print("\n" + "-" * 78)
        print(name)
        print("-" * 78)
        base = run(df, fns, BASE_COLS)
        add = run(df, fns, ADD_COLS)
        repl = run(df, fns, REPLACE_COLS)
        rows.append(report(f"{name}: BASE -> ADD (blend kept)", base, add))
        rows.append(report(f"{name}: BASE -> REPLACE (shipped)", base, repl))
        rows.append(report(f"{name}: ADD -> REPLACE (dropping the blend)", add, repl))

    print("\n" + "=" * 78)
    print("SUMMARY (negative delta = the candidate is better)")
    print("=" * 78)
    for r in rows:
        print(
            f"  {r['label']:46s} {r['delta']:+.4f}  "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
        )
    print(
        "\nThe split shipped to production regardless of these numbers, and the\n"
        "blend was dropped on the duplication argument rather than on a measured\n"
        "RMSE gain. Both were operator decisions. This block is the record of\n"
        "what they actually cost or bought."
    )


if __name__ == "__main__":
    main()
