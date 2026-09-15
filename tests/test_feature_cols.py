"""
B3: the feature-name contract.

BASE_FEATURE_COLS (per-team naming) and FEATURE_COLS (per-game home_/away_
naming) must stay consistent with each other and with what the feature
pipeline actually produces. These were two hand-maintained lists in two files
until 2026-09-14; this pins the contract that replaced them.

Run: pytest tests/test_feature_cols.py -v
"""

import pytest

from src.models.common import BASE_FEATURE_COLS, FEATURE_COLS


def test_feature_cols_is_base_prefixed_home_then_away():
    expected = [f"home_{c}" for c in BASE_FEATURE_COLS] + [
        f"away_{c}" for c in BASE_FEATURE_COLS
    ]
    assert FEATURE_COLS == expected


def test_home_and_away_sides_are_symmetric():
    """Every home_ feature needs its away_ twin, or the model sees one side of
    the matchup in more detail than the other."""
    home = {c[len("home_") :] for c in FEATURE_COLS if c.startswith("home_")}
    away = {c[len("away_") :] for c in FEATURE_COLS if c.startswith("away_")}
    assert home == away, f"asymmetric: {home ^ away}"


def test_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert len(BASE_FEATURE_COLS) == len(set(BASE_FEATURE_COLS))


def test_build_game_features_uses_the_shared_list():
    """It must import the list, not redeclare it -- the drift this replaced."""
    from src.features import build_game_features as bgf

    assert bgf.FEATURE_COLS is BASE_FEATURE_COLS, (
        "build_game_features.py has stopped using the shared list from "
        "src/models/common.py -- the two definitions can now drift apart again"
    )


@pytest.mark.parametrize("col", FEATURE_COLS)
def test_every_feature_is_pregame_or_a_known_non_stat(col):
    """Guards against a post-game stat entering the model feature set."""
    stripped = col.replace("home_", "", 1).replace("away_", "", 1)
    allowed_non_pregame = {"rest_days", "prior_games_played"}
    assert (
        stripped.startswith("pregame_") or stripped in allowed_non_pregame
    ), f"{col} is neither a pregame_ stat nor a known schedule-derived column"
