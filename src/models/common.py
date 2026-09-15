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

FEATURE_COLS = [f"home_{c}" for c in BASE_FEATURE_COLS] + [
    f"away_{c}" for c in BASE_FEATURE_COLS
]
