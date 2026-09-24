"""
src/experiments/drive_model_v2.py

The drive-outcome model, rebuilt. A structural, explicitly RELATIONAL predictor:
it matches one team's offensive drive-outcome distribution against the specific
defence it will face, converts that to points, and multiplies by possessions.

EXPERIMENTAL. Production is untouched. main() writes its walk-forward result to
data/processed/drivev2_predictions.parquet (with a provenance sidecar) so the
scoreboard and model report can compare it; nothing else is written.

--------------------------------------------------------------- what v1 did

`src/models/unused/monte_carlo.py` scored 9.608 against a 9.458 baseline --
significantly WORSE, one of five models measurably beaten by a no-ML rule. It
blends offence and defence rates, draws 2,000 multinomial samples of N drives,
and averages the resulting point totals.

------------------------------------------------------- the four defects, in order

  1. THE SIMULATION ESTIMATES SOMETHING IT COULD COMPUTE. `draws @ POINTS` then
     `.mean()` is a Monte Carlo estimate of E[sum of points], but for a
     multinomial that expectation is exactly

         n * (probabilities . points)

     in closed form. Sampling it 2,000 times does not approximate the answer
     better than the formula; it approximates it WORSE, adding O(1/sqrt(2000))
     of pure noise to every prediction for no information. This is the single
     biggest defect and it is arithmetic, not football: a model whose point
     estimate is a noisy version of a number it already has.

     v2 computes the expectation exactly. The sampler is kept, because a drive
     model's real advantage is that it yields a DISTRIBUTION -- useful for
     totals and spreads -- but it is no longer in the path of the point
     prediction.

  2. ARITHMETIC BLENDING IGNORES THE LEAGUE BASELINE. v1 uses
     (offence + defence_allowed) / 2. If an offence scores touchdowns on 25% of
     drives and a defence allows 25%, the answer is not 25% unless the league
     average is also 25% -- it is higher, because both are above average. The
     standard correction is the odds-ratio (log5) blend, which asks how each
     side deviates from the baseline and combines the deviations multiplicatively.
     That is also what makes this a genuinely relational model rather than an
     average of two numbers.

  3. DEFENSIVE POINTS WENT TO NOBODY. `opp_touchdown` -- a pick-six or fumble
     return, 490 of them in the data -- carries 0 points in v1's POINTS vector,
     and `safety` likewise. Those points are real and belong to the DEFENDING
     team. build_drive_stats fixed exactly this bug for its own est_points_for
     column and the simulator never picked the fix up, so v1 systematically
     under-predicts, which is visible in its -1.34 home bias.

  4. POSSESSIONS ARE SHARED AND WERE NOT. Football alternates possession, so the
     two teams' drive counts are nearly equal by construction. v1 takes each
     team's own pregame drive count independently; v2 blends a team's own rate
     with its opponent's drives-faced rate, which is the same quantity measured
     from the other end.

  5. NO HOME-FIELD ADVANTAGE, AT ALL. Found by measuring v2 rather than by
     reading v1, and it is the largest of the five. Drive-outcome rates are
     team-level averages with no home/away split, so the model has no mechanism
     to produce a home edge: it predicted a home margin of -0.06 against an
     actual +2.23. Every other model in this project learns that term from the
     data; this one structurally could not. It is now estimated from the
     training fold and applied as an explicit offset, the same way baseline.py
     does it.

     Worth recording that fixing 1-4 alone made RMSE WORSE (9.608 -> 9.756),
     because they widened a distribution that was still centred in the wrong
     place. v1's dispersion was sd 2.60 against an actual 10.04 -- so compressed
     that its missing home-field term barely showed up in the error. Correcting
     the structure without correcting the centre is how a set of genuine
     improvements produces a worse number.

------------------------------------------------------------------- leakage

Every input is a `pregame_` column from drive_model_table.parquet, already built
under the project's EWM+shift discipline. This module adds no new history
handling, so there is no new leakage surface -- only arithmetic on features that
were already safe. The league baselines used for the log5 blend are computed
from the TRAINING fold only, which is the one place a naive implementation would
leak a global average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.common import recent_residual_offset

DRIVE_TABLE = "data/processed/drive_model_table.parquet"
TEAM_GAME_STATS = "data/processed/team_game_stats.parquet"
SCHEDULES = "data/raw/schedules.parquet"

CATS = [
    "touchdown",
    "field_goal",
    "punt",
    "turnover",
    "turnover_on_downs",
    "missed_field_goal",
    "end_of_half",
    "opp_touchdown",
    "safety",
]

# Points to the team WITH the ball on that drive. A touchdown is 6 plus a
# conversion that succeeds ~94% of the time for one point, so ~6.95 rather than
# the flat 7 v1 used. Two-point attempts are folded into that average.
OFFENSE_POINTS = np.array([6.95, 3.0, 0, 0, 0, 0, 0, 0, 0])

# Points to the DEFENDING team on that drive: a returned touchdown and a safety.
# Zero in v1, which is why it under-predicted.
DEFENSE_POINTS = np.array([0, 0, 0, 0, 0, 0, 0, 6.95, 2.0])

EPS = 1e-9

# How far to trust the log5 blend against simply predicting the league baseline.
# 1.0 is pure log5; 0.0 says every matchup is average.
#
# This matters more than any other single choice here. The log5 blend AMPLIFIES
# deviations -- an above-average offence against an above-average defence lands
# beyond both -- and the inputs are noisy drive rates over ~11 possessions a
# game. Amplifying noise is exactly what sank the first pass/rush split, and
# shrinking the blend is the same correction that rescued it.
#
# Measured on the test set the optimum is ~0.6 (9.469, against 9.711 unshrunk).
# That number is deliberately NOT hardcoded: taking it would be tuning on the
# test set, which is the specific error this project's walk-forward exists to
# prevent and which it already refused once over ridge alpha=100. The weight is
# instead selected inside each training fold, which scores slightly worse and is
# the only defensible option.
SHRINK_GRID = (1.0, 0.8, 0.6, 0.4, 0.2)
INNER_VALIDATION_FRACTION = 0.25


def log5(p: np.ndarray, q: np.ndarray, league: np.ndarray) -> np.ndarray:
    """Combine an offensive rate with a defensive rate against a baseline.

    The odds-ratio form: convert each side to odds relative to the league,
    multiply, convert back.

        blended_odds = (p/(1-p)) * (q/(1-q)) / (L/(1-L))

    Two league-average sides return the league rate. An above-average offence
    against an above-average defence lands above the baseline rather than at
    their midpoint, which is the whole point and what an arithmetic mean cannot
    express.
    """
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    L = np.clip(league, EPS, 1 - EPS)
    odds = (p / (1 - p)) * (q / (1 - q)) / (L / (1 - L))
    return odds / (1.0 + odds)


def blend_distribution(
    off: np.ndarray, deff: np.ndarray, league: np.ndarray
) -> np.ndarray:
    """Per-category log5, renormalised to a proper distribution.

    Applying log5 category-by-category does not preserve the sum, so the result
    is renormalised. That is the standard practical treatment and it is exact for
    the binary case each category represents.
    """
    blended = log5(off, deff, league)
    total = blended.sum(axis=-1, keepdims=True)
    return np.divide(blended, np.where(total > 0, total, 1.0))


def _league_rates(train: pd.DataFrame, side: str) -> np.ndarray:
    """Baseline rates from the TRAINING fold only.

    Using a global average here would be the one genuine leak available in this
    module -- a whole-dataset baseline carries information from the test weeks
    into every blended probability.
    """
    cols = [f"home_pregame_{side}_{c}_rate" for c in CATS]
    r = train[cols].mean().to_numpy(float)
    r = np.nan_to_num(r, nan=0.0)
    return r / r.sum() if r.sum() > 0 else np.full(len(CATS), 1 / len(CATS))


def _select_shrink(train: pd.DataFrame, base: dict) -> float:
    """Choose the blend weight inside the training fold.

    The most recent quarter of the training games is held out, the remainder is
    used to set the baselines, and the grid is scored on the held-out part. The
    test week is never involved, so the choice cannot be tuned to it.
    """
    if len(train) < 200:
        return 0.6
    ordered = train.sort_values("gameday")
    cut = int(len(ordered) * (1 - INNER_VALIDATION_FRACTION))
    inner_train, inner_val = ordered.iloc[:cut], ordered.iloc[cut:]
    if len(inner_val) < 50:
        return 0.6

    inner_base = dict(base)
    inner_base["league_off"] = _league_rates(inner_train, "off")
    inner_base["league_def"] = _league_rates(inner_train, "def")
    inner_base["home_field"] = float(
        np.nanmean(
            (inner_train["home_score"] - inner_train["away_score"]).to_numpy(float)
        )
    )

    best_w, best_err = 0.6, np.inf
    for w in SHRINK_GRID:
        trial = dict(inner_base, shrink=w)
        h, a = predict_drive_model(trial, inner_val)
        err = np.sqrt(
            np.nanmean(
                np.concatenate(
                    [
                        (h - inner_val["home_score"].to_numpy(float)) ** 2,
                        (a - inner_val["away_score"].to_numpy(float)) ** 2,
                    ]
                )
            )
        )
        if err < best_err:
            best_w, best_err = w, err
    return best_w


def load_table() -> pd.DataFrame:
    """drive_model_table plus the three context columns it lacks.

    build_drive_game_features (production) never carried is_neutral_site,
    went_to_ot or gameday-as-datetime, so the drive model had no way to know a
    Super Bowl from a home game or an overtime result from a regulation one.
    Both are corrections C3 and C5 that the rest of the project made in
    2026-09-14 and this table missed. Joined here rather than in production,
    which is untouched.
    """
    df = pd.read_parquet(DRIVE_TABLE)
    df["gameday"] = pd.to_datetime(df["gameday"])

    sched = pd.read_parquet(SCHEDULES)[["game_id", "location", "game_type", "overtime"]]
    sched["is_neutral_site"] = (sched["location"] == "Neutral").astype(int)
    sched["went_to_ot"] = sched["overtime"].fillna(0).astype(int)
    return df.merge(
        sched[["game_id", "is_neutral_site", "went_to_ot"]], on="game_id", how="left"
    )


def _home_field(train: pd.DataFrame) -> float:
    """The home edge, estimated the way baseline.py estimates it.

    Two corrections the drive model was missing, both already settled elsewhere
    in this project:

      C3  neutral-site games (London, Mexico, Munich, Super Bowl) have no home
          team. Including them drags the estimate toward zero AND the estimate
          then gets applied to them. Excluded here, and predict() gives them no
          adjustment.
      C5  overtime inflates the final margin in a way no pregame quantity
          predicts, so those rows are halved rather than dropped.
    """
    rows = train
    if "is_neutral_site" in train.columns:
        non_neutral = train[train["is_neutral_site"] == 0]
        if not non_neutral.empty:
            rows = non_neutral
    margin = (rows["home_score"] - rows["away_score"]).to_numpy(float)
    if "went_to_ot" in rows.columns:
        w = np.where(rows["went_to_ot"].to_numpy() == 1, 0.5, 1.0)
        return float(np.average(margin, weights=w))
    return float(np.nanmean(margin))


def fit_drive_model(train: pd.DataFrame) -> dict:
    """Baselines, cold-start fallbacks, the home-field offset, the blend weight
    and the drift correction -- all from the training fold only."""
    base = {
        "league_off": _league_rates(train, "off"),
        "league_def": _league_rates(train, "def"),
        "fallback_drives": float(
            np.nanmean(train["home_pregame_n_drives"].to_numpy(float))
        ),
        # Drive-outcome rates carry no home/away information whatsoever, so
        # without this the model cannot express a home edge at all. Split evenly
        # across the two sides so the predicted TOTAL is unaffected and only the
        # margin moves.
        "home_field": _home_field(train),
        "shrink": None,  # filled below, needs the rest of the dict to score
        "off_h": 0.0,
        "off_a": 0.0,
    }
    return _finish_fit(train, base)


def _finish_fit(train: pd.DataFrame, base: dict) -> dict:
    base["shrink"] = _select_shrink(train, base)
    # Scoring-environment drift, the same correction every other production
    # model applies and this one never did. Measured at an uncorrected +1.10
    # away bias -- the project already validated this mechanism taking the other
    # models from +0.78 to +0.06. Computed AFTER shrink, because it has to
    # measure the residuals of the model as finally configured.
    h, a = predict_drive_model(base, train)
    base["off_h"], base["off_a"] = recent_residual_offset(train, h, a)
    return base


def _side_matrix(df: pd.DataFrame, side: str, unit: str) -> np.ndarray:
    """A writable copy -- callers patch cold-start rows in place, and pandas can
    hand back a read-only view."""
    return df[[f"{side}_pregame_{unit}_{c}_rate" for c in CATS]].to_numpy(float).copy()


def _expected_points(
    off_rates: np.ndarray,
    def_rates: np.ndarray,
    league_off: np.ndarray,
    n_drives: np.ndarray,
    shrink: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact expected points per side, given blended drive outcomes.

    Returns (points the offence scores, points it CONCEDES on its own drives).
    The second is the pick-six/safety channel that v1 dropped entirely.
    """
    p = blend_distribution(off_rates, def_rates, league_off)
    if shrink is not None and shrink < 1.0:
        p = shrink * p + (1.0 - shrink) * league_off
    off_pts = n_drives * (p @ OFFENSE_POINTS)
    conceded = n_drives * (p @ DEFENSE_POINTS)
    return off_pts, conceded


def predict_drive_model(model: dict, test: pd.DataFrame):
    """Vectorised over the whole test frame -- v1 looped with .iterrows()."""
    lo, ld = model["league_off"], model["league_def"]

    home_off = _side_matrix(test, "home", "off")
    home_def = _side_matrix(test, "home", "def")
    away_off = _side_matrix(test, "away", "off")
    away_def = _side_matrix(test, "away", "def")

    # Cold start: fall back to the league baseline rather than to NaN.
    for m, fb in ((home_off, lo), (away_off, lo), (home_def, ld), (away_def, ld)):
        bad = ~np.isfinite(m).all(axis=1)
        m[bad] = fb

    # Possessions are shared: a team's own drive count and its opponent's
    # drives-faced are two measurements of the same quantity.
    hd = test["home_pregame_n_drives"].to_numpy(float)
    hf = test["away_pregame_n_drives_faced"].to_numpy(float)
    ad = test["away_pregame_n_drives"].to_numpy(float)
    af = test["home_pregame_n_drives_faced"].to_numpy(float)
    fb = model["fallback_drives"]

    # nanmean over an all-NaN column warns and returns NaN. The NaN is handled
    # below by the fallback, but the warning is noise on every fold, so the
    # all-missing case is answered directly instead.
    def _blend(a, b):
        stacked = np.stack([a, b])
        both_missing = np.isnan(stacked).all(axis=0)
        out = np.full(stacked.shape[1], np.nan)
        ok = ~both_missing
        if ok.any():
            out[ok] = np.nanmean(stacked[:, ok], axis=0)
        return out

    home_drives = _blend(hd, hf)
    away_drives = _blend(ad, af)
    home_drives = np.where(np.isfinite(home_drives), home_drives, fb)
    away_drives = np.where(np.isfinite(away_drives), away_drives, fb)

    w = model.get("shrink")
    h_off, h_conceded = _expected_points(home_off, away_def, lo, home_drives, w)
    a_off, a_conceded = _expected_points(away_off, home_def, lo, away_drives, w)

    # A team's score is what its offence produces plus what its DEFENCE takes
    # back on the opponent's drives.
    home = h_off + a_conceded
    away = a_off + h_conceded

    # ...and then the home edge, which nothing above can produce. C3: a
    # neutral-site game has no home team and gets no adjustment.
    adj = model.get("home_field", 0.0) / 2.0
    if "is_neutral_site" in test.columns:
        adj = adj * (1 - test["is_neutral_site"].to_numpy(float))
    return (
        home + adj - model.get("off_h", 0.0),
        away - adj - model.get("off_a", 0.0),
    )


def simulate_distribution(
    model: dict, test: pd.DataFrame, n_trials: int = 4000, seed: int = 42
):
    """Sampled score distributions, for uncertainty rather than for the mean.

    Kept because a drive model's genuine advantage over a regression is that it
    produces a full distribution -- directly useful for totals and spreads. It is
    deliberately NOT used for the point prediction, which has a closed form.
    """
    rng = np.random.default_rng(seed)
    lo, ld = model["league_off"], model["league_def"]
    h = blend_distribution(
        _side_matrix(test, "home", "off"), _side_matrix(test, "away", "def"), lo
    )
    a = blend_distribution(
        _side_matrix(test, "away", "off"), _side_matrix(test, "home", "def"), lo
    )
    hd = np.rint(
        np.nan_to_num(test["home_pregame_n_drives"].to_numpy(float), nan=11)
    ).astype(int)
    ad = np.rint(
        np.nan_to_num(test["away_pregame_n_drives"].to_numpy(float), nan=11)
    ).astype(int)

    n = len(test)
    home = np.zeros((n, n_trials))
    away = np.zeros((n, n_trials))
    for i in range(n):
        hp = np.nan_to_num(h[i], nan=0.0)
        ap = np.nan_to_num(a[i], nan=0.0)
        if hp.sum() <= 0 or ap.sum() <= 0:
            continue
        hdraw = rng.multinomial(max(1, hd[i]), hp / hp.sum(), size=n_trials)
        adraw = rng.multinomial(max(1, ad[i]), ap / ap.sum(), size=n_trials)
        home[i] = hdraw @ OFFENSE_POINTS + adraw @ DEFENSE_POINTS
        away[i] = adraw @ OFFENSE_POINTS + hdraw @ DEFENSE_POINTS
    return home, away


def main():
    from src.validate.walk_forward import score_predictions, walk_forward_evaluate

    df = load_table()
    res = walk_forward_evaluate(
        df, fit_drive_model, predict_drive_model, min_train_seasons=2
    )
    m = score_predictions(res)
    print("\nDrive model v2 walk-forward:")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  mean RMSE: {(m['home_rmse'] + m['away_rmse']) / 2:.4f}")
    print(
        "\n  for reference: monte_carlo v1 = 9.608, baseline = 9.458, poisson = 9.367"
    )
    from src.validate.backtest_io import save_predictions

    save_predictions(res, "drivev2", inputs=[DRIVE_TABLE, SCHEDULES])


if __name__ == "__main__":
    main()
