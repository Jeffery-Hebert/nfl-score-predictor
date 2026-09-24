"""
src/experiments/test_injury_features.py

Tests the compressed injury signal from build_injury_features.py against the
active benchmark.

Injuries are the largest genuine information gap in this project -- the model
has never known who is actually playing. The feature-saturation finding said
the estimator had to be fixed before more information could be judged fairly,
so this runs AFTER linear.py moved from unregularized OLS to RidgeCV, and it
uses the production fit/predict functions rather than plain OLS.

Two columns per team, by design rather than by accident:
  injury_impact   positional value x prior snap share, summed over unavailable
                  players (Questionable at half weight)
  qb_out          starting quarterback Out or Doubtful

Active benchmark (n=1426, mean of home/away RMSE):
  baseline 9.4415 | linear 9.4045 | poisson 9.3920

Run: python -m src.experiments.test_injury_features
"""

import numpy as np
import pandas as pd

import src.models.linear as lin
import src.models.poisson_glm as poi
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42
INJURY_COLS = ["injury_impact", "qb_out"]


def build_table() -> pd.DataFrame:
    """model_table plus qb_out.

    injury_impact has been a production feature since 2026-09-15, so it is
    already in model_table. This used to merge it in a second time, which gave
    pandas two home_injury_impact columns, suffixed them _x/_y, and left the
    experiment with no column of the expected name (KeyError). Only qb_out is
    added now.
    """
    mt = pd.read_parquet("data/processed/model_table.parquet")
    inj = pd.read_parquet("data/processed/injury_features.parquet")
    for side in ("home", "away"):
        mt = mt.merge(
            inj[["game_id", "team", "qb_out"]].rename(
                columns={"qb_out": f"{side}_qb_out", "team": f"{side}_team"}
            ),
            on=["game_id", f"{side}_team"],
            how="left",
        )
    return mt


def sided(cols):
    return [f"home_{c}" for c in cols] + [f"away_{c}" for c in cols]


def with_features(module, cols):
    """Run a production model against a different feature list.

    The production fit/predict read FEATURE_COLS from their own module
    namespace, so swapping it there is what lets this experiment reuse the
    exact production estimator (RidgeCV / GridSearchCV Poisson) instead of
    reimplementing it and risking a mismatch.
    """

    def fit(train):
        original = module.FEATURE_COLS
        module.FEATURE_COLS = cols
        try:
            return module.fit_fn(train)
        finally:
            module.FEATURE_COLS = original

    def predict(m, test):
        original = module.FEATURE_COLS
        module.FEATURE_COLS = cols
        try:
            return module.predict_fn(m, test)
        finally:
            module.FEATURE_COLS = original

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
    print(
        f"{len(df)} games; injury coverage {df['home_injury_impact'].notna().mean():.1%}"
    )
    qb_games = int((df.home_qb_out.fillna(0) + df.away_qb_out.fillna(0) > 0).sum())
    print(f"  games with a starting QB out: {qb_games}\n")

    # FEATURE_COLS now INCLUDES injury_impact, so "base" is production without
    # it -- the comparison this experiment was written to make.
    no_injury = [c for c in FEATURE_COLS if c not in sided(["injury_impact"])]
    variants = {
        "base (no injury)": no_injury,
        "+injury_impact (production)": FEATURE_COLS,
        "+qb_out": no_injury + sided(["qb_out"]),
        "+both": FEATURE_COLS + sided(["qb_out"]),
    }

    lin.fit_fn, lin.predict_fn = lin.fit_linear, lin.predict_linear
    poi.fit_fn, poi.predict_fn = poi.fit_poisson, poi.predict_poisson

    for name, module in [("Linear (RidgeCV)", lin), ("Poisson GLM", poi)]:
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        results = {}
        for label, cols in variants.items():
            fit, predict = with_features(module, cols)
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            results[label] = res
            m = score_predictions(res)
            print(
                f"  {label:<20} home={m['home_rmse']:.4f} away={m['away_rmse']:.4f} "
                f"mean={(m['home_rmse'] + m['away_rmse']) / 2:.4f}"
            )

        base = results["base (no injury)"]
        print(f"\n  Paired bootstrap vs base ({N_BOOT} resamples over games)\n")
        for label, res in results.items():
            if label == "base (no injury)":
                continue
            mean, lo, hi = bootstrap(base, res)
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {label:<20} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
