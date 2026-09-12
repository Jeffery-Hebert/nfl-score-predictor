"""
Validation gates for data/processed/team_game_stats.parquet.
Run: pytest tests/test_team_game_stats.py -v
"""
import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path("data/processed/team_game_stats.parquet")

@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/features/build_team_game_stats.py first"
    return pd.read_parquet(DATA_PATH)

def test_two_rows_per_completed_game(df):
    completed = df.dropna(subset=["team_score"])
    counts = completed.groupby("game_id").size()
    assert (counts == 2).all(), "Some completed games don't have exactly 2 team rows"

def test_no_team_playing_itself(df):
    assert (df["team"] != df["opponent"]).all(), "Found a row where team == opponent"

def test_home_away_are_complementary(df):
    """For each game, exactly one row should be is_home=1 and one is_home=0."""
    grouped = df.groupby("game_id")["is_home"].sum()
    completed_games = df.dropna(subset=["team_score"])["game_id"].unique()
    checked = grouped.loc[grouped.index.isin(completed_games)]
    assert (checked == 1).all(), "Some games don't have exactly one home + one away row"

def test_epa_columns_populated_for_played_games(df):
    completed = df.dropna(subset=["team_score"])
    null_pct = completed["off_epa_per_play"].isna().mean()
    assert null_pct < 0.02, f"{null_pct:.1%} of played games missing offensive EPA — investigate"