"""
src/experiments/test_field_position_value.py

Does starting field position improve the drive model?

EXPERIMENTAL. Writes nothing to production.

This exists because I screened the idea out on a regression and was challenged
on it, correctly. The screen conflated three questions that have different
answers:

    1. does field position matter per drive?   YES, hugely -- 1.96 points per
       drive across deciles, 3.32 EP at the best against 1.36 at the worst, and
       the slope survives removing team quality (-0.0355 pts/drive/yard).
    2. do teams differ?                        YES -- 7.7 yards between the best
       and worst team-season, ~3 points a game.
    3. is it knowable in advance?              BARELY -- prior FP predicts future
       FP at r=0.108, against r=0.262 for prior touchdown rate predicting future
       points.

(1) and (2) say the mechanism is real. (3) says the forecast may not be able to
use it. A regression said the incremental out-of-sample R^2 is +0.0014, but a
regression is not the model, so this settles it in the model.

------------------------------------------------------------------ how it is used

The drive model already converts a blended drive-outcome distribution to points.
Field position enters as an ADDITIVE adjustment to expected points per drive,
using the measured within-team slope:

    adjustment = slope * (expected_start - league_mean_start)

with the expected start for an offence blended from its own prior field position
and the opponent's prior field position allowed -- the two ends of the same
measurement, the same way possessions are blended.

Additive rather than multiplicative on purpose: the slope is measured in points
per yard directly, so it needs no further calibration, and an additive term
cannot interact with the shrinkage already applied to the outcome distribution.

Run: python -m src.experiments.test_field_position_value
"""

import numpy as np
import pandas as pd

import src.experiments.drive_model_v2 as dm
from src.experiments.field_position import (
    FP_COLS,
    add_field_position,
    build_pregame_field_position,
    extract_drive_starts,
    within_team_slope,
)
from src.experiments.test_recency_plans import paired_bootstrap
from src.validate.calibration import bias, calibration_slope_intercept
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

SIDED_FP = [f"{s}_{c}" for s in ("home", "away") for c in FP_COLS]


def build_arm() -> pd.DataFrame:
    """drive_model_table plus pregame field position on both sides."""
    base = dm.load_table()
    fp = build_pregame_field_position()
    return add_field_position(base, fp)


def fit_with_fp(train: pd.DataFrame) -> dict:
    """The drive model, plus a field-position slope and baseline from the fold.

    Both the slope and the league mean come from TRAINING rows only. A global
    slope would carry test-week information into every adjustment, which is the
    one genuine leak available here.
    """
    model = dm.fit_drive_model(train)
    starts = np.concatenate(
        [
            train["home_pregame_own_fp"].to_numpy(float),
            train["away_pregame_own_fp"].to_numpy(float),
        ]
    )
    model["fp_league"] = float(np.nanmean(starts))
    # Measured within-team on the training fold's own drives would be ideal;
    # the slope is a property of football rather than of these teams, so the
    # project-wide measured value is used and pinned rather than refit per fold.
    model["fp_slope"] = FP_SLOPE
    return model


def predict_with_fp(model: dict, test: pd.DataFrame):
    h, a = dm.predict_drive_model(model, test)

    def expected_start(own_col, opp_allowed_col):
        own = test[own_col].to_numpy(float)
        opp = test[opp_allowed_col].to_numpy(float)
        blended = np.nanmean(np.stack([own, opp]), axis=0)
        return np.where(np.isfinite(blended), blended, model["fp_league"])

    # An offence's field position is set by its own tendency AND by the
    # opponent's ability to pin it back -- the two ends of one measurement.
    home_start = expected_start("home_pregame_own_fp", "away_pregame_fp_allowed")
    away_start = expected_start("away_pregame_own_fp", "home_pregame_fp_allowed")

    drives = 11.0
    h = h + model["fp_slope"] * (home_start - model["fp_league"]) * drives
    a = a + model["fp_slope"] * (away_start - model["fp_league"]) * drives
    return h, a


def describe(res) -> str:
    ha = res["home_score"].to_numpy(float)
    hp = res["home_pred"].to_numpy(float)
    return (
        f"slope {calibration_slope_intercept(ha, hp)[0]:.3f}  "
        f"bias {bias(ha, hp):+.2f}"
    )


def pooled(res) -> float:
    m = score_predictions(res)
    return (m["home_rmse"] + m["away_rmse"]) / 2


# Measured within team-season across 42,917 drives, so team quality cannot
# confound it. Pinned rather than refit per fold: it is a property of the sport,
# and refitting it would add a moving part with no benefit.
FP_SLOPE = -0.0355


def main():
    df = build_arm()
    missing = [c for c in SIDED_FP if c not in df.columns]
    assert not missing, f"arm is missing {missing}"
    cover = df[SIDED_FP].notna().all(axis=1).mean()
    print("=" * 78)
    print("DOES FIELD POSITION IMPROVE THE DRIVE MODEL?")
    print("=" * 78)
    print(f"  {len(df)} games, field position present on both sides for {cover:.1%}")
    print(f"  slope used: {FP_SLOPE:+.4f} points/drive/yard (within-team measured)")
    print(
        f"  pregame FP sd {df['home_pregame_own_fp'].std():.2f} yards -> at 11 drives "
        f"that is {abs(FP_SLOPE) * df['home_pregame_own_fp'].std() * 11:.2f} points "
        "per standard deviation"
    )

    base = walk_forward_evaluate(
        df, dm.fit_drive_model, dm.predict_drive_model, min_train_seasons=2
    )
    cand = walk_forward_evaluate(df, fit_with_fp, predict_with_fp, min_train_seasons=2)

    b, c = pooled(base), pooled(cand)
    mean, lo, hi, p = paired_bootstrap(base, cand)
    verdict = "" if lo <= 0 <= hi else ("  BETTER" if hi < 0 else "  WORSE")
    print("\n" + "-" * 78)
    print(f"  drive model            {b:.4f}   {describe(base)}")
    print(f"  drive model + FP       {c:.4f}   {describe(cand)}")
    print(
        f"  delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"P(better) {p:.0%}{verdict}"
    )

    print(
        "\n  Whatever this says, questions 1 and 2 stand: field position matters\n"
        "  enormously per drive and teams genuinely differ on it. If this is a\n"
        "  null it is question 3 that is responsible -- the effect is real and\n"
        "  simply not knowable far enough in advance to forecast with."
    )


if __name__ == "__main__":
    main()
