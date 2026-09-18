"""
src/experiments/test_relational_features.py

Do RELATIONAL features help -- the ones that describe how two teams interact
rather than how each has been playing on its own?

EXPERIMENTAL. Writes nothing to production.

Four arms, each answering a question the current feature set structurally
cannot:

  MATCHUP    explicit offence x opposing-defence products. Ridge is additive and
             can only sum an elite passing attack and a leaky secondary; these
             let the two compound. Poisson already multiplies through its log
             link, so the falsifiable prediction is that this helps RIDGE
             materially more than POISSON. If both move equally, the terms are
             doing something other than what they claim.

  ADJUSTED   opponent-adjusted ratings, v2. A closed null result revived: v1
             adjusted a BLEND of all plays, unweighted by volume, and was tested
             as a replacement. v2 adjusts pass and rush separately, weights by
             play count, and is tested as an addition.

  CONTINUITY roster carryover, line-up stability and snap-weighted tenure. The
             only arm here carrying genuinely NEW information rather than a
             re-cut of play-by-play -- which matters, because every re-cut so
             far has produced a null (five documented).

  ALL        everything at once, to see whether they are complementary or
             redundant. This project's repeated finding is that redundant
             columns cost more than they return, so a combined arm that is worse
             than its best part is a result rather than a disappointment.

ON THE BAR. The operator's instruction stands: a feature that meaningfully
improves how the data is REPRESENTED may be worth keeping even if the RMSE
delta does not clear a bootstrap threshold, because that threshold is an
artefact of a 1,440-game sample and not a law. So this script reports
calibration slope and bias alongside RMSE, and reports P(better) rather than
only a verdict, so a small consistent gain is distinguishable from a coin flip.

Run: python -m src.experiments.test_relational_features
     python -m src.experiments.test_relational_features --quick
"""

import argparse

import numpy as np
import pandas as pd

from src.experiments.adjusted_ratings_v2 import (
    RATING_COLS,
    add_adjusted_ratings,
    build_adjusted_ratings,
)
from src.experiments.continuity import (
    CONTINUITY_COLS,
    add_continuity,
    build_continuity,
)
from src.experiments.matchup import MATCHUP_COLS, add_matchup_terms
from src.experiments.rebuild import production_table
from src.experiments.test_recency_plans import (
    paired_bootstrap,
    poisson_fns,
    pooled_rmse,
    ridge_fns,
)
from src.models.common import FEATURE_COLS, ot_sample_weight, recent_residual_offset
from src.validate.calibration import bias, calibration_slope_intercept
from src.validate.walk_forward import walk_forward_evaluate

SIDED_RATINGS = [f"{s}_{c}" for s in ("home", "away") for c in RATING_COLS]
SIDED_CONTINUITY = [f"{s}_{c}" for s in ("home", "away") for c in CONTINUITY_COLS]


def build_arms() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """(table, feature list) per arm. The control is production, untouched."""
    base = production_table()
    arms = {"control": (base, list(FEATURE_COLS))}

    m = add_matchup_terms(base)
    arms["+matchup"] = (m, list(FEATURE_COLS) + MATCHUP_COLS)

    ratings = build_adjusted_ratings()
    r = add_adjusted_ratings(base, ratings)
    arms["+adjusted"] = (r, list(FEATURE_COLS) + SIDED_RATINGS)

    cont = build_continuity()
    c = add_continuity(base, cont)
    arms["+continuity"] = (c, list(FEATURE_COLS) + SIDED_CONTINUITY)

    everything = add_continuity(
        add_adjusted_ratings(add_matchup_terms(base), ratings), cont
    )
    arms["+all"] = (
        everything,
        list(FEATURE_COLS) + MATCHUP_COLS + SIDED_RATINGS + SIDED_CONTINUITY,
    )
    return arms


def make_fns(base_fns, cols):
    """Rebind a model's fit/predict onto an arbitrary feature list.

    test_recency_plans' versions close over FEATURE_COLS; here the column set is
    what varies, so they are rebuilt with the arm's list.
    """
    from sklearn.linear_model import PoissonRegressor, RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ALPHAS = np.logspace(-2, 4, 25)

    if base_fns is ridge_fns:

        def fit(train):
            X = train[cols]
            means = X.mean()
            Xf = X.fillna(means)
            w = ot_sample_weight(train)

            def m():
                return make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5)),
                )

            h = m().fit(Xf, train["home_score"], ridgecv__sample_weight=w)
            a = m().fit(Xf, train["away_score"], ridgecv__sample_weight=w)
            oh, oa = recent_residual_offset(train, h.predict(Xf), a.predict(Xf))
            return {"h": h, "a": a, "means": means, "oh": oh, "oa": oa}

        def pred(mo, test):
            X = test[cols].fillna(mo["means"])
            return mo["h"].predict(X) - mo["oh"], mo["a"].predict(X) - mo["oa"]

    else:

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
            return {"h": h, "a": a, "means": means, "sc": sc, "oh": oh, "oa": oa}

        def pred(mo, test):
            Xs = mo["sc"].transform(test[cols].fillna(mo["means"]))
            return mo["h"].predict(Xs) - mo["oh"], mo["a"].predict(Xs) - mo["oa"]

    return fit, pred


def describe(res: pd.DataFrame) -> dict:
    h_a = res["home_score"].to_numpy(float)
    h_p = res["home_pred"].to_numpy(float)
    return {
        "rmse": pooled_rmse(res),
        "slope": calibration_slope_intercept(h_a, h_p)[0],
        "bias": bias(h_a, h_p),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Poisson only")
    args = ap.parse_args()

    arms = build_arms()
    for name, (t, cols) in arms.items():
        missing = [c for c in cols if c not in t.columns]
        assert not missing, f"{name} is missing {missing}"
    print("=" * 88)
    print("RELATIONAL FEATURES")
    print("=" * 88)
    for name, (_, cols) in arms.items():
        print(f"  {name:14s} {len(cols):3d} features")

    models = [("Poisson", poisson_fns)]
    if not args.quick:
        models.append(("RidgeCV", ridge_fns))

    for model_name, fns in models:
        control_table, control_cols = arms["control"]
        cf, cp = make_fns(fns, control_cols)
        base = walk_forward_evaluate(control_table, cf, cp, min_train_seasons=2)
        cstat = describe(base)

        print("\n" + "-" * 88)
        print(
            f"{model_name}   control = {cstat['rmse']:.4f}  "
            f"slope {cstat['slope']:.3f}  bias {cstat['bias']:+.2f}"
        )
        print("-" * 88)
        print(
            f"  {'arm':14s}{'RMSE':>9}{'delta':>9}{'95% CI':>22}"
            f"{'P(bet)':>8}{'slope':>8}{'bias':>8}"
        )
        for name, (t, cols) in arms.items():
            if name == "control":
                continue
            f, p = make_fns(fns, cols)
            res = walk_forward_evaluate(t, f, p, min_train_seasons=2)
            s = describe(res)
            mean, lo, hi, pb = paired_bootstrap(base, res)
            flag = "" if lo <= 0 <= hi else ("  BETTER" if hi < 0 else "  WORSE")
            print(
                f"  {name:14s}{s['rmse']:>9.4f}{mean:>+9.4f}"
                f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{pb:>7.0%}"
                f"{s['slope']:>8.3f}{s['bias']:>+8.2f}{flag}"
            )

    print(
        "\nRead slope alongside RMSE. 1.0 is perfect calibration; above ~1.15 means\n"
        "predictions are compressed toward the mean, which flattens every derived\n"
        "spread and total while barely moving RMSE. A feature that improves slope\n"
        "without moving RMSE has improved the model for the stated objective."
    )


if __name__ == "__main__":
    main()
