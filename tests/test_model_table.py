"""
Validation gates for the final model-ready table.
Run: pytest tests/test_model_table.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data

DATA_PATH = Path("data/processed/model_table.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/features/build_game_features.py first"
    return pd.read_parquet(DATA_PATH)


def test_one_row_per_game(df):
    assert df["game_id"].is_unique, "Duplicate game_id rows found in model table"


def test_matches_completed_game_count(df):
    schedules = pd.read_parquet("data/raw/schedules.parquet")
    expected = len(schedules)
    assert (
        len(df) == expected
    ), f"Model table has {len(df)} rows, schedules has {expected}"


def test_no_leakage_columns_present(df):
    """Guard against accidentally carrying in-game or post-game stats."""
    forbidden_substrings = ["off_epa_per_play", "def_epa_per_play", "success_rate"]
    for col in df.columns:
        if any(s in col for s in forbidden_substrings) and not col.startswith(
            ("home_pregame_", "away_pregame_")
        ):
            pytest.fail(
                f"Column '{col}' looks like a non-pregame stat leaking into the model table"
            )


def test_targets_present_for_completed_games(df):
    completed = df.dropna(subset=["home_score"])
    assert completed["home_score"].notna().all()
    assert completed["away_score"].notna().all()


# ------------------------------------------------- B6: cold-start / missingness


def test_missing_features_are_confined_to_the_first_season(df):
    """Pregame features can only be NaN where no prior game exists -- i.e. each
    team's very first appearance, which is week 1 of the earliest season.

    A regression that NaN'd week 1 of EVERY season (for instance, resetting the
    rolling history at each season boundary) would still leave a small overall
    NaN rate and would slip past a simple threshold check. This pins the shape
    of the missingness, not just its volume.
    """
    from src.models.common import FEATURE_COLS

    first_season = df["season"].min()
    later = df[df["season"] > first_season]
    for col in FEATURE_COLS:
        if col.endswith("prior_games_played"):
            continue  # a counter, defined from game one
        rate = later[col].isna().mean()
        assert rate == 0, (
            f"{col} is {rate:.2%} NaN after the first season -- pregame features "
            "should only be missing at a team's genuine cold start"
        )


def test_cold_start_is_the_whole_first_week_and_nothing_else(df):
    from src.models.common import FEATURE_COLS

    first_season = df["season"].min()
    missing = df[df[FEATURE_COLS[0]].isna()]
    assert set(missing["season"].unique()) == {first_season}
    assert set(missing["week"].unique()) == {1}, (
        "cold-start rows appear outside week 1 of the first season -- "
        "the rolling history may be resetting mid-dataset"
    )


def test_overall_missingness_stays_small(df):
    from src.models.common import FEATURE_COLS

    rate = df[FEATURE_COLS].isna().mean().max()
    assert rate < 0.02, f"a production feature is {rate:.1%} missing -- investigate"
