"""
src/experiments/test_drive_blend.py

Can the drive model add anything to Poisson?

EXPERIMENTAL. Writes nothing to production.

------------------------------------------------------------------- the case

The A5 finding blamed the three-model stack's failure on its members being
near-duplicates of each other. Measured error correlations, home side:

    poisson vs linear    +0.998
    poisson vs gp        +0.996
    poisson vs drive_v2  +0.980     <- built on a different mechanism

0.980 is still very high, but it is the lowest pairing in the project, and
drive_model_v2 is the only competitive model here that does not work from
per-play EPA -- it works from drive outcomes and possessions. That is the first
genuinely different mechanism available to stack.

A fixed-weight probe found pooled RMSE 9.3555 at 80/20 against Poisson's 9.3679,
and the gain held across 20-40% drive weight rather than sitting on a knife
edge. That probe chose its weight on the test set, which is why this script
exists: the weight has to be selected inside each training fold or the number is
worthless.

Two arms:

  BLEND   a scalar weight on the two predictions, swept in-fold.
  RIDGE   a proper meta-model on both, the same shape as stacking.py.

------------------------------------------------------------------- leakage

Both inputs are already walk-forward out-of-sample predictions -- each was made
by a model that saw only games before its own week. The meta-level walk-forward
then trains on meta-rows strictly before the test week, exactly as stacking.py
does. min_train_seasons=1 because the input is already the reduced test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.experiments.test_recency_plans import paired_bootstrap
from src.validate.calibration import bias, calibration_slope_intercept
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

PRED_DIR = "data/processed"
MEMBERS = ["poisson", "drivev2"]
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)  # weight on the DRIVE model
INNER_FRACTION = 0.25
ALPHAS = np.logspace(-2, 4, 25)


def load_meta() -> pd.DataFrame:
    sched = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "season", "week", "gameday", "home_score", "away_score"]
    ]
    merged = None
    for name in MEMBERS:
        p = pd.read_parquet(f"{PRED_DIR}/{name}_predictions.parquet")[
            ["game_id", "home_pred", "away_pred"]
        ].rename(columns={"home_pred": f"{name}_home", "away_pred": f"{name}_away"})
        merged = p if merged is None else merged.merge(p, on="game_id", how="inner")
    out = merged.merge(sched, on="game_id", how="left").dropna(
        subset=["home_score", "away_score"]
    )
    out["gameday"] = pd.to_datetime(out["gameday"])
    return out


# ------------------------------------------------------------------ blend arm


def _blend_rmse(df: pd.DataFrame, w: float) -> float:
    h = (1 - w) * df["poisson_home"] + w * df["drivev2_home"]
    a = (1 - w) * df["poisson_away"] + w * df["drivev2_away"]
    return float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        (h - df["home_score"]).to_numpy(float) ** 2,
                        (a - df["away_score"]).to_numpy(float) ** 2,
                    ]
                )
            )
        )
    )


def fit_blend(train: pd.DataFrame) -> dict:
    """Pick the drive weight on a held-out tail of the TRAINING fold.

    Taking the test-set optimum would be the alpha=100 error this project has
    already refused twice; the in-fold choice scores slightly worse and is the
    only defensible one.
    """
    ordered = train.sort_values("gameday")
    cut = int(len(ordered) * (1 - INNER_FRACTION))
    inner_val = ordered.iloc[cut:]
    if len(inner_val) < 40:
        return {"w": 0.25}
    errs = {w: _blend_rmse(inner_val, w) for w in WEIGHT_GRID}
    return {"w": float(min(errs, key=errs.get))}


def predict_blend(model: dict, test: pd.DataFrame):
    w = model["w"]
    return (
        ((1 - w) * test["poisson_home"] + w * test["drivev2_home"]).to_numpy(float),
        ((1 - w) * test["poisson_away"] + w * test["drivev2_away"]).to_numpy(float),
    )


# ------------------------------------------------------------------ ridge arm


def fit_ridge(train: pd.DataFrame) -> dict:
    hc = [f"{m}_home" for m in MEMBERS]
    ac = [f"{m}_away" for m in MEMBERS]
    sh = StandardScaler().fit(train[hc])
    sa = StandardScaler().fit(train[ac])

    def r():
        return RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))

    return {
        "h": r().fit(sh.transform(train[hc]), train["home_score"]),
        "a": r().fit(sa.transform(train[ac]), train["away_score"]),
        "sh": sh,
        "sa": sa,
        "hc": hc,
        "ac": ac,
    }


def predict_ridge(model: dict, test: pd.DataFrame):
    return (
        model["h"].predict(model["sh"].transform(test[model["hc"]])),
        model["a"].predict(model["sa"].transform(test[model["ac"]])),
    )


# ---------------------------------------------------------------- the control


def fit_poisson_only(train):
    return {}


def predict_poisson_only(model, test):
    return (
        test["poisson_home"].to_numpy(float),
        test["poisson_away"].to_numpy(float),
    )


def pooled(res) -> float:
    m = score_predictions(res)
    return (m["home_rmse"] + m["away_rmse"]) / 2


def describe(res) -> str:
    h_a = res["home_score"].to_numpy(float)
    h_p = res["home_pred"].to_numpy(float)
    return (
        f"slope {calibration_slope_intercept(h_a, h_p)[0]:.3f}  "
        f"bias {bias(h_a, h_p):+.2f}"
    )


def main():
    df = load_meta()
    print("=" * 78)
    print("DRIVE MODEL AS A STACK PARTNER FOR POISSON")
    print("=" * 78)
    print(f"  {len(df)} games shared by both models")

    eh = {}
    for m in MEMBERS:
        eh[m] = (df[f"{m}_home"] - df["home_score"]).to_numpy(float)
    print(
        f"  home-error correlation: {np.corrcoef(eh['poisson'], eh['drivev2'])[0, 1]:+.3f}"
    )

    base = walk_forward_evaluate(
        df, fit_poisson_only, predict_poisson_only, min_train_seasons=1
    )
    b = pooled(base)
    print(f"\n  control (poisson alone)  {b:.4f}   {describe(base)}")
    print("-" * 78)

    for name, fit, pred in (
        ("in-fold blend", fit_blend, predict_blend),
        ("ridge meta", fit_ridge, predict_ridge),
    ):
        res = walk_forward_evaluate(df, fit, pred, min_train_seasons=1)
        c = pooled(res)
        mean, lo, hi, p = paired_bootstrap(base, res)
        verdict = "" if lo <= 0 <= hi else ("  BETTER" if hi < 0 else "  WORSE")
        print(
            f"  {name:24s} {c:.4f}   delta {mean:+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]  P {p:.0%}{verdict}"
        )
        print(f"  {'':24s} {describe(res)}")

    # What the in-fold selector actually chose, so a degenerate w=0 (which would
    # make the blend a silent no-op) is visible rather than hidden in the mean.
    picks = []
    orig = fit_blend

    def spy(train):
        m = orig(train)
        picks.append(m["w"])
        return m

    walk_forward_evaluate(df, spy, predict_blend, min_train_seasons=1)
    import collections

    print(
        f"\n  drive weights chosen in-fold: {dict(sorted(collections.Counter(picks).items()))}"
    )
    print(
        "\n  For reference, the test-set optimum was 0.20 at 9.3555. An in-fold\n"
        "  number at or near that, with a weight the selector actually varies,\n"
        "  is the honest version of the same result."
    )


if __name__ == "__main__":
    main()
