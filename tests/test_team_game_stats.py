"""
Validation gates for data/processed/team_game_stats.parquet.
Run: pytest tests/test_team_game_stats.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data

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
    assert (
        null_pct < 0.02
    ), f"{null_pct:.1%} of played games missing offensive EPA — investigate"


def test_exactly_32_teams(df):
    n_teams = df["team"].nunique()
    assert (
        n_teams == 32
    ), f"Expected 32 NFL franchises, found {n_teams} — check for unmapped relocations/renames"


# ------------------------------------------- which plays the rates are taken on
#
# Until 2026-10-01 off_success_rate / def_success_rate_allowed (and the EPA per
# play and play counts beside them) included kickoffs, punts, extra points, field
# goals, kneels, spikes and penalty-nullified snaps -- 23% of the plays. These
# check the BUILT table against play-by-play, so a stale table, a reverted
# filter or a builder that drifts from scrimmage_snaps() all fail here.
# tests/test_team_game_stats_snaps.py pins the rule itself on synthetic plays.

RATE_COLS = {
    "off": ["off_plays", "off_epa_per_play", "off_success_rate"],
    "def": ["def_plays", "def_epa_per_play", "def_success_rate_allowed"],
}


@pytest.fixture(scope="module")
def pbp():
    return pd.read_parquet(
        "data/raw/pbp.parquet",
        columns=["game_id", "posteam", "defteam", "play_type", "epa", "success"],
    )


def _rates(plays: pd.DataFrame, side: str, prefix: str) -> pd.DataFrame:
    g = plays.groupby(["game_id", side])
    out = pd.DataFrame(
        {
            f"{prefix}_plays": g["epa"].count(),
            f"{prefix}_epa_per_play": g["epa"].mean(),
            RATE_COLS[prefix][2]: g["success"].mean(),
        }
    )
    return out.rename_axis(["game_id", "team"])


def test_rates_are_measured_on_scrimmage_snaps_only(df, pbp):
    from src.features.build_team_game_stats import scrimmage_snaps

    built = df.set_index(["game_id", "team"])
    for side, prefix in (("posteam", "off"), ("defteam", "def")):
        want = _rates(scrimmage_snaps(pbp, side), side, prefix)
        got = built.loc[want.index, RATE_COLS[prefix]]
        diff = (got - want[RATE_COLS[prefix]]).abs().max()
        assert (diff < 1e-12).all(), (
            f"team_game_stats {prefix} rates differ from scrimmage snaps in "
            f"play-by-play by {diff.to_dict()} -- stale table or a changed "
            "filter; rebuild: python -m src.features.build_all"
        )


def test_no_special_teams_or_clock_plays_leak_in(df, pbp):
    """The built rates must differ from the old every-play definition almost
    everywhere -- otherwise the fix never reached the table."""
    plays = pbp[pbp["play_type"].notna() & pbp["posteam"].notna()]
    old = plays.groupby(["game_id", "posteam"])["success"].mean()
    old.index.names = ["game_id", "team"]
    built = df.set_index(["game_id", "team"])["off_success_rate"]
    both = pd.concat([built.rename("built"), old.rename("old")], axis=1).dropna()
    assert len(both) > 3000
    same = (both["built"] - both["old"]).abs() < 1e-12
    assert same.mean() < 0.05, f"{same.mean():.1%} of team-games still match"


def test_offence_and_defence_are_two_views_of_the_same_snaps(df):
    """A team's offensive rates in a game ARE its opponent's defensive ones."""
    played = df.dropna(subset=["off_success_rate", "def_success_rate_allowed"])
    off = played.set_index(["game_id", "team"])[RATE_COLS["off"]]
    deff = played.set_index(["game_id", "opponent"])[RATE_COLS["def"]]
    deff.index.names = ["game_id", "team"]
    j = off.join(deff, how="inner")
    assert len(j) == len(off), "a team-game has no opposing defensive row"
    for o, d in zip(RATE_COLS["off"], RATE_COLS["def"]):
        assert (j[o] - j[d]).abs().max() < 1e-12, f"{o} != opponent's {d}"


def test_snap_counts_and_rates_are_plausible(df):
    played = df.dropna(subset=["team_score", "off_plays"])
    # Scrimmage snaps per team-game ran 30-95 across 2019-2026; every play with a
    # play_type ran 47-119. Bounds sit outside the first and catch the second.
    assert played["off_plays"].between(25, 105).all()
    assert played["off_success_rate"].between(0, 1).all()
    assert 0.40 < played["off_success_rate"].mean() < 0.47
