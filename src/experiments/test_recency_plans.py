"""
src/experiments/test_recency_plans.py

Does changing HOW we age evidence help? Three questions in one harness:

  RE-SWEEP      17 weeks was chosen on 2026-09-14, before injury_impact, before
                the ridge fix and before the shrunk pass/rush splits. Shrinkage
                in particular interacts with it -- n_eff is recency-weighted, so
                a shorter half-life now automatically increases shrinkage toward
                the league mean, a self-correction that did not exist when 17
                was picked. The optimum may have moved.

  PER-STAT      one half-life currently governs eight stats whose persistence
                differs threefold. Whether that is a defect depends on WHY they
                differ, which is the next paragraph.

  TWO-TIMESCALE a single exponential cannot be both steep near the present and
                long-tailed. A mixture can. This is the only untested SHAPE;
                every previous attempt varied one rate.

WHAT THE DATA SAYS BEFORE WE START, which is worth stating so the results are
not over-read either way. Autocorrelation by lag, pooled within team-season:

    off_success_rate   lag1 0.22  lag4 0.22  lag8 0.20  lag16 0.23
    off_pass_epa       lag1 0.12  lag4 0.10  lag8 0.13  lag16 0.09

Flat. Within a season these traits barely decay at all -- the correlation is low
because a single game is a noisy measurement, not because the team changed.
Fitted per-game half-lives land between 30 and 485 GAMES, all far longer than
production's 17 weeks.

That points away from steeper recency, not toward it, and it explains the two
existing null results: tune_halflife found 4 and 8 weeks worse, and compressing
the offseason was worse. The decay that matters is at the SEASON boundary, where
rosters actually turn over -- and a calendar-day exponential already applies it
(a ~200-day summer costs 0.5^(200/119) = 0.31).

So the honest prior here is that these will not help. Running it anyway, because
"the mechanism says no" has been wrong in this project before and the cost is an
hour of compute.

Standalone: writes nothing. Control arm is production's own model_table.

Run: python -m src.experiments.test_recency_plans
     python -m src.experiments.test_recency_plans --quick   # Poisson only
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.rebuild import (
    assert_shaped_like_production,
    build_table,
    production_table,
)
from src.experiments.recency import Exponential, RecencyPlan, TwoTimescale
from src.models.common import FEATURE_COLS, ot_sample_weight, recent_residual_offset
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42
ALPHAS = np.logspace(-2, 4, 25)
W = 7.0  # days per week


def plans() -> dict[str, RecencyPlan]:
    """Candidate arms. The control is production itself, not a plan."""
    out = {}

    # --- re-sweep a single exponential, including longer than production
    for weeks in (8, 12, 17, 26, 39):
        out[f"exp{weeks}w"] = RecencyPlan(default=Exponential(weeks * W))

    # --- per-stat, ordered by the MEASURED within-season decay rather than by
    # split-half r (which conflates trait drift with measurement noise, and
    # would have ranked these backwards).
    slow, mid, fast = Exponential(34 * W), Exponential(17 * W), Exponential(10 * W)
    out["per-stat"] = RecencyPlan(
        default=mid,
        per_stat={
            "off_success_rate": slow,
            "def_success_rate_allowed": slow,
            "off_pass_epa_per_play": slow,
            "off_rush_epa_per_play": slow,
            "def_pass_epa_per_play_allowed": slow,
            "def_rush_epa_per_play_allowed": slow,
            "team_score": fast,
            "opp_score": mid,
        },
    )

    # --- two-timescale: the shape a single exponential cannot make. Fast
    # component covers "current form", slow one "underlying class".
    out["two 3w/26w@40%"] = RecencyPlan(default=TwoTimescale(3 * W, 26 * W, 0.40))
    out["two 2w/34w@30%"] = RecencyPlan(default=TwoTimescale(2 * W, 34 * W, 0.30))
    # The operator's stated shape: last 2-3 weeks dominant, long tail behind it.
    out["two 2.5w/17w@60%"] = RecencyPlan(default=TwoTimescale(2.5 * W, 17 * W, 0.60))
    return out


def ridge_fns():
    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)

        def m():
            return make_pipeline(
                StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))
            )

        h = m().fit(Xf, train["home_score"], ridgecv__sample_weight=w)
        a = m().fit(Xf, train["away_score"], ridgecv__sample_weight=w)
        oh, oa = recent_residual_offset(train, h.predict(Xf), a.predict(Xf))
        return {"h": h, "a": a, "means": means, "oh": oh, "oa": oa}

    def pred(mo, test):
        X = test[FEATURE_COLS].fillna(mo["means"])
        return mo["h"].predict(X) - mo["oh"], mo["a"].predict(X) - mo["oa"]

    return fit, pred


def poisson_fns():
    def fit(train):
        X = train[FEATURE_COLS]
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
        return {"h": h, "a": a, "means": means, "sc": sc, "oh": oh, "oa": oa}

    def pred(mo, test):
        Xs = mo["sc"].transform(test[FEATURE_COLS].fillna(mo["means"]))
        return mo["h"].predict(Xs) - mo["oh"], mo["a"].predict(Xs) - mo["oa"]

    return fit, pred


def pooled_rmse(results) -> float:
    m = score_predictions(results)
    return (m["home_rmse"] + m["away_rmse"]) / 2


def paired_bootstrap(base: pd.DataFrame, cand: pd.DataFrame):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Poisson only")
    args = ap.parse_args()

    control = production_table()
    models = [("Poisson", poisson_fns)]
    if not args.quick:
        models.append(("RidgeCV", ridge_fns))

    print("=" * 84)
    print("RECENCY PLANS -- control is production's own model_table")
    print("=" * 84)

    candidates = plans()
    tables = {}
    for name, plan in candidates.items():
        t = build_table(plan)
        assert_shaped_like_production(t, control)
        tables[name] = t
    print(f"built {len(tables)} candidate tables, all shaped like the control\n")

    for model_name, fns in models:
        fit, pred = fns()
        base = walk_forward_evaluate(control, fit, pred, min_train_seasons=2)
        b = pooled_rmse(base)
        print("\n" + "-" * 84)
        print(f"{model_name}   control (production) = {b:.4f}")
        print("-" * 84)
        print(f"  {'plan':20s}{'RMSE':>9}{'delta':>9}{'95% CI':>22}{'P(better)':>11}")
        for name, t in tables.items():
            res = walk_forward_evaluate(t, fit, pred, min_train_seasons=2)
            c = pooled_rmse(res)
            mean, lo, hi, p = paired_bootstrap(base, res)
            verdict = "" if lo <= 0 <= hi else ("  BETTER" if hi < 0 else "  WORSE")
            print(
                f"  {name:20s}{c:>9.4f}{mean:>+9.4f}"
                f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>10.1%}{verdict}"
            )

    print(
        "\nReminder on interpretation: within-season autocorrelation is flat, so a\n"
        "steeper kernel is expected to lose. A null here is a confirmation of the\n"
        "mechanism, not an absence of evidence."
    )


if __name__ == "__main__":
    main()
