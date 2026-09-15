"""
Leakage-safety tests for pregame rolling features.
Uses a hand-built synthetic team history where the correct answer is known
in advance -- this is the actual leakage gate, not a sanity check on real data.

The fixture is built via tests/conftest.py::make_team_history so that adding a
column to STAT_COLS cannot silently disable these tests (see that file's
docstring -- it has happened twice).

Run: pytest tests/test_rolling_features_leakage.py -v
"""

import pandas as pd
import pytest

from src.features.build_rolling_features import add_pregame_rolling_features
from tests.conftest import make_team_history


@pytest.fixture
def synthetic_team():
    """One fake team, 4 games, weekly spacing, simple team_score sequence.

    Weeks 1-4 of 2024 are all non-finale, so finale masking is inactive here
    and these tests isolate the EWM + shift(1) leakage behavior on its own.
    Only team_score is asserted on; other stat columns are placeholders.
    """
    return make_team_history(
        gamedays=["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"],
        seasons=[2024, 2024, 2024, 2024],
        weeks=[1, 2, 3, 4],
        team_score=[10, 20, 30, 40],
    )


def test_first_game_has_no_pregame_feature(synthetic_team):
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    assert pd.isna(
        result.loc[0, "pregame_team_score"]
    ), "First game should have NaN pregame feature (no prior games exist)"


def test_pregame_feature_excludes_current_game_score(synthetic_team):
    """The pregame feature for game 4 (score=40) must not equal 40,
    and must be strictly between games 1-3's scores (10, 20, 30)."""
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    pregame_g4 = result.loc[3, "pregame_team_score"]
    assert (
        pregame_g4 != 40
    ), "Leakage: pregame feature includes the current game's own score"
    assert (
        10 <= pregame_g4 <= 30
    ), f"pregame_team_score={pregame_g4} is outside the range of prior games (10-30)"


def test_more_recent_games_weighted_higher(synthetic_team):
    """With an increasing score trend (10,20,30,40), the pregame feature for
    game 4 should be closer to 30 (most recent) than to a simple average of
    10+20+30=20."""
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    pregame_g4 = result.loc[3, "pregame_team_score"]
    simple_avg = (10 + 20 + 30) / 3
    assert (
        pregame_g4 > simple_avg
    ), "Recency weighting isn't favoring recent games over older ones"


def test_rest_days_is_passed_through_not_recomputed(synthetic_team):
    """C1 regression guard.

    rest_days used to be computed here as gameday.diff(), which reported up to
    260 "rest days" across an offseason -- a number that describes nothing in
    football. It now comes from schedules (nflverse home_rest/away_rest, max
    16) and must survive this function untouched.
    """
    synthetic_team = synthetic_team.copy()
    synthetic_team["rest_days"] = [7, 10, 4, 13]
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    assert result["rest_days"].tolist() == [7, 10, 4, 13], (
        "rest_days was modified -- it must pass through from schedules, and "
        "must never be recomputed from gameday differences"
    )


def test_prior_games_played_resets_each_season(synthetic_team):
    """C2 regression guard.

    This counter used to run globally (0 -> 152 across 2019-2026), making it a
    calendar index rather than a football quantity. It must restart per season.
    """
    df = make_team_history(
        gamedays=["2023-09-10", "2023-09-17", "2024-09-08", "2024-09-15"],
        seasons=[2023, 2023, 2024, 2024],
        weeks=[1, 2, 1, 2],
        team_score=[10, 20, 30, 40],
    )
    result = add_pregame_rolling_features(df, halflife_days=119)
    assert result["prior_games_played"].tolist() == [0, 1, 0, 1], (
        f"expected a per-season counter, got "
        f"{result['prior_games_played'].tolist()} -- looks global again"
    )
