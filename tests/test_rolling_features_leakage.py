"""
Leakage-safety tests for pregame rolling features.
Uses a hand-built synthetic team history where the correct answer is known
in advance -- this is the actual leakage gate, not a sanity check on real data.

Run: pytest tests/test_rolling_features_leakage.py -v
"""
import pandas as pd
import pytest
from src.features.build_rolling_features import add_pregame_rolling_features

@pytest.fixture
def synthetic_team():
    """One fake team, 4 games, weekly spacing, simple team_score sequence."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "team": ["FAKE"] * 4,
        "gameday": pd.to_datetime(["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"]),
        "team_score": [10, 20, 30, 40],
        "opp_score": [7, 14, 21, 28],
        "off_epa_per_play": [0.1, 0.2, 0.3, 0.4],
        "def_epa_per_play": [-0.1, -0.2, -0.3, -0.4],
        "off_success_rate": [0.4, 0.5, 0.6, 0.7],
        "def_success_rate_allowed": [0.4, 0.45, 0.5, 0.55],
    })

def test_first_game_has_no_pregame_feature(synthetic_team):
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    assert pd.isna(result.loc[0, "pregame_team_score"]), \
        "First game should have NaN pregame feature (no prior games exist)"

def test_pregame_feature_excludes_current_game_score(synthetic_team):
    """The pregame feature for game 4 (score=40) must not equal 40,
    and must be strictly between games 1-3's scores (10, 20, 30)."""
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    pregame_g4 = result.loc[3, "pregame_team_score"]
    assert pregame_g4 != 40, "Leakage: pregame feature includes the current game's own score"
    assert 10 <= pregame_g4 <= 30, \
        f"pregame_team_score={pregame_g4} is outside the range of prior games (10-30)"

def test_more_recent_games_weighted_higher(synthetic_team):
    """With an increasing score trend (10,20,30,40), the pregame feature for
    game 4 should be closer to 30 (most recent) than to a simple average of
    10+20+30=20."""
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    pregame_g4 = result.loc[3, "pregame_team_score"]
    simple_avg = (10 + 20 + 30) / 3
    assert pregame_g4 > simple_avg, \
        "Recency weighting isn't favoring recent games over older ones"

def test_rest_days_computed_correctly(synthetic_team):
    result = add_pregame_rolling_features(synthetic_team, halflife_days=119)
    assert result.loc[1, "rest_days"] == 7, "Weekly spacing should give 7 rest days"
    assert pd.isna(result.loc[0, "rest_days"]), "First game should have no rest_days value"