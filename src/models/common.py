"""
Shared feature list and leakage-safe imputation for all sklearn-style models.
Every model imports FEATURE_COLS from here rather than redefining it, so all
models train on an identical, consistent feature set.

BASE_FEATURE_COLS is the single source of truth. It is the per-TEAM column
naming produced by build_rolling_features.py; FEATURE_COLS is the per-GAME
form, the same stats prefixed home_/away_ once build_game_features.py has
joined the two sides together.

These used to be two hand-maintained lists in two files (here and
build_game_features.py) that had to be kept in sync by hand. Deriving both
from one list removes that class of drift; tests/test_feature_cols.py holds
the contract.
"""

# NOTE (2026-09-17): pregame_off_epa_per_play and pregame_def_epa_per_play were
# REMOVED from this list when the pass/rush split shipped. They were the blended
# all-plays EPA rates, and the split measures the same efficiency at a finer
# grain -- carrying both put two descriptions of one quantity in front of the
# model. Blended offensive EPA correlates 0.86 with the shrunk pass split and
# 0.58 with the rush split; the defensive blend, 0.78 and 0.45.
#
# This is a deliberate reversal of how the split first shipped, which kept the
# blend alongside on the grounds that it is the lower-variance measurement. The
# operator's call was that the duplication is the bigger problem. The splits are
# still volume-weighted and shrunk, so the noise argument that originally
# motivated keeping the blend is handled inside build_split_efficiency.py rather
# than by carrying a second copy of the statistic.
#
# Success rate is NOT split and stays blended -- it is the most persistent thing
# the project measures (split-half r = 0.695) and nothing about it is duplicated
# by the EPA columns.
BASE_FEATURE_COLS = [
    "pregame_team_score",
    "pregame_opp_score",
    "pregame_off_success_rate",
    "pregame_def_success_rate_allowed",
    "rest_days",
    "prior_games_played",
]

# Game-level context that is not per-team, so it is not home_/away_ prefixed.
# Both are known before kickoff (schedules.location / schedules.game_type).
#   C3 -- neutral-site games have no true home team but were modelled as
#         ordinary home games (50 games: London, Mexico, Munich, Super Bowl).
#   C4 -- playoff games were mixed into the regular season unmarked (89 games).
GAME_FEATURE_COLS = ["is_neutral_site", "is_playoff"]

# Per-team availability signal from src/features/build_injury_features.py.
# Sided like BASE_FEATURE_COLS but sourced from injury_features.parquet rather
# than team_rolling_features.parquet, hence a separate list.
#
# injury_impact = sum over unavailable players of positional value x prior snap
# share. It is the first feature in this project's history to clear the
# promotion gate on both Linear and Poisson (-0.0355 and -0.0388, CIs excluding
# zero).
#
# qb_out was tested alongside it and deliberately NOT adopted: it is largely
# redundant (a starting QB out already dominates injury_impact via weight 1.00
# x ~1.0 snap share) and adding it DILUTED the result on both models
# (-0.0388 -> -0.0312 on Poisson). Fewer, stronger features win here.
INJURY_FEATURE_COLS = ["injury_impact"]

# Pass/rush efficiency split, offense and defense, from
# src/features/build_split_efficiency.py. Sided like BASE_FEATURE_COLS but
# sourced from split_efficiency.parquet, hence a separate list.
#
# This is the SECOND attempt at this idea. The first (2026-09-14) replaced
# blended EPA with raw pass-only/rush-only averages and measured a real
# regression on Linear: +0.037 home RMSE, 95% CI [+0.0058, +0.0685]. Two things
# are different now and both matter:
#
#   1. that verdict was measured with sklearn LinearRegression -- unregularized
#      OLS, on a 16-column feature set with no injury_impact. The project
#      diagnosed that estimator as unfit for this problem the following day and
#      replaced it with RidgeCV. The old number was taken with a broken
#      instrument.
#   2. the recorded failure mechanism was that splitting halves the effective
#      play count per component and raises measurement noise. v1 did nothing
#      about that. These columns are volume-weighted per play and shrunk toward
#      a recency-weighted league prior by measured empirical-Bayes constants
#      (225-540 plays), so a thin sample is pulled toward the league instead of
#      being handed to the model at face value.
#
# These REPLACE blended EPA rather than joining it -- see the note on
# BASE_FEATURE_COLS above. That is the same shape as v1, but the noise problem
# that sank v1 is now handled by the shrinkage rather than by keeping a second,
# coarser copy of the same measurement in the feature set.
SPLIT_FEATURE_COLS = [
    "pregame_off_pass_epa_shrunk",
    "pregame_off_rush_epa_shrunk",
    "pregame_def_pass_epa_allowed_shrunk",
    "pregame_def_rush_epa_allowed_shrunk",
]

FEATURE_COLS = (
    [f"home_{c}" for c in BASE_FEATURE_COLS]
    + [f"away_{c}" for c in BASE_FEATURE_COLS]
    + [f"home_{c}" for c in INJURY_FEATURE_COLS]
    + [f"away_{c}" for c in INJURY_FEATURE_COLS]
    + [f"home_{c}" for c in SPLIT_FEATURE_COLS]
    + [f"away_{c}" for c in SPLIT_FEATURE_COLS]
    + GAME_FEATURE_COLS
)

# C5: NOT a feature. went_to_ot is 0% populated before kickoff, so using it as
# a model input would be target leakage. It exists only as a training-side
# sample weight -- see ot_sample_weight() below.
OT_TRAINING_COL = "went_to_ot"
OT_SAMPLE_WEIGHT = 0.5  # historical OT games count half when fitting


def ot_sample_weight(train):
    """Down-weight historical overtime games when fitting.

    Leakage-safe: reads the OT status of games already played, which is fully
    known at fit time, and never consults the row being predicted. An extra
    period adds scoring no pregame feature can anticipate, so those inflated
    finals otherwise pull the fitted coefficients around.

    Returns None when the column is absent, so callers degrade gracefully.
    """
    import numpy as np

    if OT_TRAINING_COL not in train.columns:
        return None
    return np.where(train[OT_TRAINING_COL] == 1, OT_SAMPLE_WEIGHT, 1.0)


# ---------------------------------------------------------------- bias drift

# The league's scoring environment moves, and the models train on every prior
# season at equal weight, so they inherit a stale picture of it:
#
#     era          avg away score    home-field edge
#     2019-2021        23.16             +0.74
#     2024-2026        21.98             +2.12
#
# 2019 and 2020 were the empty-stadium seasons, where home-field advantage
# essentially vanished (+0.04, +0.17). Carrying that forward made every model
# over-predict away scores by about +0.75 while home scores stayed unbiased.
#
# Measured at 285 games (~1 NFL season) by
# src/experiments/test_bias_correction.py:
#
#     model     away bias before -> after     margin bias before -> after
#     linear        +0.780  ->  +0.262           -0.883  ->  -0.183
#     poisson       +0.741  ->  +0.256           -0.894  ->  -0.199
#
# RMSE is unchanged either way (every variant's bootstrap CI spans zero), so
# this buys calibration rather than accuracy -- which is the point, since the
# objective is the exact score and the exact margin, not the spread.
#
# Also tested and NOT adopted: down-weighting old TRAINING games
# (test_training_recency.py). It reduces the bias too, but by discarding data:
# anything under a ~2-season half-life is measurably WORSE on RMSE, and at a
# 2-season half-life it only removes 20% of the bias. Correcting the offset
# directly is strictly better -- it keeps every training row.
BIAS_CORRECTION_GAMES = 285


def recent_residual_offset(train, pred_home, pred_away, n_games=BIAS_CORRECTION_GAMES):
    """How much the model currently over-predicts, per side, in points.

    Fits are least-squares, so residuals sum to zero over the WHOLE training
    set. This looks at a recent SUBSET, which is where drift shows up.

    Leakage-safe: reads only training rows and their own dates. The game being
    predicted is never involved.

    Returns (home_offset, away_offset) to SUBTRACT from predictions.
    """
    import numpy as np

    if "gameday" not in train.columns or len(train) == 0:
        return 0.0, 0.0
    k = min(n_games, len(train))
    recent = np.argsort(train["gameday"].to_numpy())[-k:]
    home_off = float((pred_home - train["home_score"].to_numpy(float))[recent].mean())
    away_off = float((pred_away - train["away_score"].to_numpy(float))[recent].mean())
    return home_off, away_off
