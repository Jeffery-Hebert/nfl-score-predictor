"""
Validation gates for data/raw/schedules.parquet.
Run: pytest tests/test_schedules_schema.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path("data/raw/schedules.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/ingest/pull_schedules.py first"
    return pd.read_parquet(DATA_PATH)


def test_expected_columns_present(df):
    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "gameday",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_no_duplicate_game_ids(df):
    dupes = df["game_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate game_id rows"


def test_season_range(df):
    assert df["season"].min() >= 2019, "Data includes seasons before 2019 cutoff"
    assert df["season"].max() <= 2026, "Data includes unexpected future seasons"


def test_completed_games_have_valid_scores(df):
    completed = df.dropna(subset=["home_score", "away_score"])
    assert (completed["home_score"] >= 0).all(), "Negative home_score found"
    assert (completed["away_score"] >= 0).all(), "Negative away_score found"
    assert completed["home_score"].max() < 100, "Suspicious home_score outlier"
    assert completed["away_score"].max() < 100, "Suspicious away_score outlier"


def test_missingness_on_key_fields(df):
    for col in ["home_team", "away_team", "season", "week"]:
        null_pct = df[col].isna().mean()
        assert null_pct == 0, f"{col} has {null_pct:.1%} missing values — investigate"


def test_incomplete_games_are_future_or_current_week(df):
    """Games missing a score should only be upcoming games, not historical gaps."""
    incomplete = df[df["home_score"].isna()]
    if len(incomplete) > 0:
        assert (
            incomplete["season"].max() >= 2026
        ), "Found incomplete games in a season that should be fully finished"
