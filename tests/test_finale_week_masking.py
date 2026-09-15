"""
Verifies finale-week results don't leak into the following season's rolling
average. Uses a hand-built example with a known correct answer.

Fixtures come from tests/conftest.py::make_team_history so a future STAT_COLS
addition cannot silently disable this gate.
"""

from src.features.build_rolling_features import (
    add_pregame_rolling_features,
    is_finale_week,
)
from tests.conftest import make_team_history


def test_finale_week_excluded_from_next_season_average():
    df = make_team_history(
        gamedays=["2023-12-24", "2023-12-31", "2024-09-08"],
        seasons=[2023, 2023, 2024],
        weeks=[17, 18, 1],
        team_score=[20, 45, 20],  # week 18 is a rested-starters blowout outlier
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
    df = make_team_history(
        gamedays=["2023-12-17", "2023-12-24"],
        seasons=[2023, 2023],
        weeks=[16, 17],
        team_score=[20, 45],
    )
    result = add_pregame_rolling_features(df, halflife_days=119)
    week17_feature = result.loc[result["week"] == 17, "pregame_team_score"].values[0]
    assert (
        week17_feature == 20
    ), "Finale week's own pregame feature should reflect only prior (week 16) game"
