import pandas as pd
import pytest
from pathlib import Path

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data

DATA_PATH = Path("data/processed/drive_model_table.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/features/build_drive_game_features.py first"
    return pd.read_parquet(DATA_PATH)


def test_one_row_per_game(df):
    assert df["game_id"].is_unique


def test_no_self_matchups(df):
    assert (df["home_team"] != df["away_team"]).all()


def test_every_scheduled_game_is_present(df):
    """Two gaps used to hide here: all 16 of Oakland's 2019 games were dropped
    (the schedule says OAK, play-by-play LV), and no unplayed game had a row, so
    no drive model could forecast one."""
    sched = pd.read_parquet("data/raw/schedules.parquet")
    missing = set(sched["game_id"]) - set(df["game_id"])
    assert (
        not missing
    ), f"{len(missing)} scheduled games missing, e.g. {sorted(missing)[:3]}"


def test_the_2019_raiders_are_in(df):
    lv19 = df[
        (df["season"] == 2019) & ((df["home_team"] == "LV") | (df["away_team"] == "LV"))
    ]
    assert len(lv19) == 16


def test_the_next_unplayed_week_has_drive_features(df):
    unplayed = df[df["home_score"].isna()]
    if unplayed.empty:
        pytest.skip("season complete")
    nxt = unplayed.sort_values("gameday").iloc[0]
    week = unplayed[
        (unplayed["season"] == nxt["season"]) & (unplayed["week"] == nxt["week"])
    ]
    feat = [c for c in df.columns if "pregame" in c]
    assert week[feat].notna().all().all(), "upcoming drive features are empty"
