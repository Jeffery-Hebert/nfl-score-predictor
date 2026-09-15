"""
Leakage-safety tests for pregame rolling DRIVE features.

This gate did not exist, and its absence was costly: when finale-week masking
was added in commit 2dc3f67, add_pregame_rolling_drive_features started reading
season/week columns that its caller never supplied. The stage crashed on every
run from then on, and nothing noticed -- there was no test, and the only
consumer (the shelved Monte Carlo model) was not being run either.

Mirrors tests/test_rolling_features_leakage.py, which covers the team-level
equivalent of the same EWM + shift(1) logic.

Run: pytest tests/test_drive_rolling_leakage.py -v
"""

import pandas as pd
import pytest

from src.features.build_drive_rolling_features import (
    add_pregame_rolling_drive_features,
)
from tests.conftest import make_drive_history


@pytest.fixture
def synthetic_drives():
    """Four non-finale games, weekly spacing, rising touchdown rate."""
    return make_drive_history(
        gamedays=["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"],
        seasons=[2024, 2024, 2024, 2024],
        weeks=[1, 2, 3, 4],
        off_touchdown_rate=[0.10, 0.20, 0.30, 0.40],
    )


def test_first_game_has_no_pregame_feature(synthetic_drives):
    result = add_pregame_rolling_drive_features(synthetic_drives, halflife_days=119)
    assert pd.isna(
        result.loc[0, "pregame_off_touchdown_rate"]
    ), "First game should have NaN pregame feature (no prior games exist)"


def test_pregame_feature_excludes_current_game(synthetic_drives):
    result = add_pregame_rolling_drive_features(synthetic_drives, halflife_days=119)
    pregame_g4 = result.loc[3, "pregame_off_touchdown_rate"]
    assert (
        pregame_g4 != 0.40
    ), "Leakage: pregame drive feature includes the current game's own rate"
    assert (
        0.10 <= pregame_g4 <= 0.30
    ), f"pregame_off_touchdown_rate={pregame_g4} is outside the prior-game range"


def test_finale_week_excluded_from_next_season(synthetic_drives):
    """A rested-starters finale must not contaminate next season's first game."""
    df = make_drive_history(
        gamedays=["2023-12-24", "2023-12-31", "2024-09-08"],
        seasons=[2023, 2023, 2024],
        weeks=[17, 18, 1],
        off_touchdown_rate=[0.20, 0.90, 0.20],  # week 18 is the outlier
    )
    result = add_pregame_rolling_drive_features(df, halflife_days=119)
    wk1 = result.loc[result["week"] == 1, "pregame_off_touchdown_rate"].values[0]
    assert wk1 != 0.90, "Finale-week drive rates leaked into next season"
    assert abs(wk1 - 0.20) < 0.05, f"Expected ~0.20 (week 17's rate), got {wk1}"


def test_stage_supplies_every_column_the_builder_needs():
    """Regression test for the crash itself: main() must merge in season/week,
    which drive_stats.parquet does not carry."""
    import inspect

    from src.features import build_drive_rolling_features as mod

    src = inspect.getsource(mod.main)
    for col in ("season", "week", "gameday"):
        assert (
            f'"{col}"' in src
        ), f"main() no longer selects {col!r} from schedules -- the builder needs it"
