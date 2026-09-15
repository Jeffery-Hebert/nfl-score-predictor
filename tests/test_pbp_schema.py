"""
Validation gates for data/raw/pbp.parquet.
Run: pytest tests/test_pbp_schema.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data

DATA_PATH = Path("data/raw/pbp.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/ingest/pull_pbp.py first"
    return pd.read_parquet(DATA_PATH)


def test_expected_columns_present(df):
    required = {
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa",
        "play_type",
        "down",
        "yards_gained",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_season_range(df):
    assert df["season"].min() >= 2019
    assert df["season"].max() <= 2026


def test_epa_reasonable_range(df):
    """EPA per play is typically between -10 and 10; wider bounds catch data corruption."""
    valid_epa = df["epa"].dropna()
    assert (
        valid_epa.between(-15, 15).mean() > 0.999
    ), "More than 0.1% of EPA values are outside a plausible range"


def test_no_duplicate_plays(df):
    dupes = df.duplicated(subset=["game_id", "play_id"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate plays"


def test_game_ids_match_schedules(df):
    """Cross-check: every game_id in pbp should exist in schedules.parquet."""
    schedules = pd.read_parquet("data/raw/schedules.parquet")
    pbp_games = set(df["game_id"].unique())
    sched_games = set(schedules["game_id"].unique())
    orphaned = pbp_games - sched_games
    assert len(orphaned) == 0, f"{len(orphaned)} pbp game_ids not found in schedules"
