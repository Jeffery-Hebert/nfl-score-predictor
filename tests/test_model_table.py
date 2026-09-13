"""
Validation gates for the final model-ready table.
Run: pytest tests/test_model_table.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

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
