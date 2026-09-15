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

BASE_FEATURE_COLS = [
    "pregame_team_score",
    "pregame_opp_score",
    "pregame_off_epa_per_play",
    "pregame_def_epa_per_play",
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

FEATURE_COLS = (
    [f"home_{c}" for c in BASE_FEATURE_COLS]
    + [f"away_{c}" for c in BASE_FEATURE_COLS]
    + [f"home_{c}" for c in INJURY_FEATURE_COLS]
    + [f"away_{c}" for c in INJURY_FEATURE_COLS]
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
