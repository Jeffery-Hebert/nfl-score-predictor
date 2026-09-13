import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path("data/processed/drive_model_table.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/features/build_drive_game_features.py first"
    return pd.read_parquet(DATA_PATH)


def test_one_row_per_game(df):
    assert df["game_id"].is_unique


def test_no_self_matchups(df):
    assert (df["home_team"] != df["away_team"]).all()
