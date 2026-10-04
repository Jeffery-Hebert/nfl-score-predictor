"""
Verifies finale-week results don't leak into the following season's rolling
average. Uses a hand-built example with a known correct answer.

Fixtures come from tests/conftest.py::make_team_history so a future STAT_COLS
addition cannot silently disable this gate.
"""

import pytest

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


def test_decay_runs_on_the_calendar_through_a_masked_week():
    """A game N days old weighs 0.5 ** (N / halflife), masked rows or not.

    With pandas' ignore_na=True (production until 2026-10-04) the decay step
    INTO the masked finale was skipped, so a game played before it kept more
    weight against every later game than its age allows: here the week-17 game
    aged 252 days instead of 259 by the next season's week 1.
    """
    h = 119.0
    df = make_team_history(
        gamedays=["2023-12-24", "2023-12-31", "2024-09-08", "2024-09-15"],
        seasons=[2023, 2023, 2024, 2024],
        weeks=[17, 18, 1, 2],
        team_score=[10, 99, 30, 0],  # week 18 is the masked finale
    )
    result = add_pregame_rolling_features(df, halflife_days=h)
    got = result.loc[result["week"] == 2, "pregame_team_score"].values[0]
    # as of 2024 week 1: the week-17 game is 259 days old, week 1's own is 0
    w17 = 0.5 ** (259 / h)
    assert got == pytest.approx((w17 * 10 + 30) / (w17 + 1), rel=1e-12)


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
