"""
Verifies finale-week results don't leak into the following season's rolling
average. Uses a hand-built example with a known correct answer.
"""

import pandas as pd
from src.features.build_rolling_features import (
    add_pregame_rolling_features,
    is_finale_week,
)


def test_finale_week_excluded_from_next_season_average():
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "team": ["FAKE"] * 3,
            "season": [2023, 2023, 2024],
            "week": [17, 18, 1],
            "gameday": pd.to_datetime(["2023-12-24", "2023-12-31", "2024-09-08"]),
            "team_score": [20, 45, 20],  # week 18 is a rested-starters blowout outlier
            "opp_score": [17, 10, 17],
            "off_epa_per_play": [0.1, 0.5, 0.1],
            "def_epa_per_play": [-0.1, -0.3, -0.1],
            "off_success_rate": [0.4, 0.7, 0.4],
            "def_success_rate_allowed": [0.4, 0.3, 0.4],
        }
    )
    result = add_pregame_rolling_features(df, halflife_days=119)
    week1_2024_feature = result.loc[result["week"] == 1, "pregame_team_score"].values[0]
    assert (
        week1_2024_feature != 45
    ), "Finale-week blowout leaked into next season's rolling average"
    assert (
        abs(week1_2024_feature - 20) < 5
    ), f"Expected value near 20 (week 17's score), got {week1_2024_feature}"


def test_finale_week_own_prediction_still_valid():
    """The finale week's own pregame feature (based on games before it) should be unaffected."""
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "team": ["FAKE"] * 2,
            "season": [2023, 2023],
            "week": [16, 17],
            "gameday": pd.to_datetime(["2023-12-17", "2023-12-24"]),
            "team_score": [20, 45],
            "opp_score": [17, 10],
            "off_epa_per_play": [0.1, 0.5],
            "def_epa_per_play": [-0.1, -0.3],
            "off_success_rate": [0.4, 0.7],
            "def_success_rate_allowed": [0.4, 0.3],
        }
    )
    result = add_pregame_rolling_features(df, halflife_days=119)
    week17_feature = result.loc[result["week"] == 17, "pregame_team_score"].values[0]
    assert (
        week17_feature == 20
    ), "Finale week's own pregame feature should reflect only prior (week 16) game"
