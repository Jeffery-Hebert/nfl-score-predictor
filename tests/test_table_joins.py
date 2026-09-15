"""
B4/B5: join and merge correctness, checked against the real built artifacts.

The home/away assembly in build_game_features.py and build_drive_game_features.py
is a self-join on game_id followed by a rename. If the sides were ever swapped,
every model would train on mirrored matchups and nothing else in the suite would
notice -- the table would still have the right shape, the right row count, and
plausible-looking values.

These tests go back to the source table and confirm each game's home_* values
really are the home team's.

Run: pytest tests/test_table_joins.py -v
Rebuild first: python -m src.features.build_all
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.common import BASE_FEATURE_COLS

# Reads built parquet from data/, which is gitignored -- excluded from CI.
pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def model_table():
    p = Path("data/processed/model_table.parquet")
    assert p.exists(), "Run python -m src.features.build_all first"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def rolling():
    p = Path("data/processed/team_rolling_features.parquet")
    assert p.exists(), "Run python -m src.features.build_all first"
    return pd.read_parquet(p)


# ------------------------------------------------------- B4: home/away join


def test_home_features_belong_to_the_home_team(model_table, rolling):
    """The orientation test. Looks each game's home team up in the source
    table and demands the model table's home_* values match."""
    src = rolling.set_index(["game_id", "team"])
    sample = model_table.dropna(subset=["home_pregame_team_score"]).sample(
        n=min(300, len(model_table)), random_state=42
    )

    mismatches = []
    for _, row in sample.iterrows():
        key = (row["game_id"], row["home_team"])
        if key not in src.index:
            mismatches.append(f"{key} missing from team_rolling_features")
            continue
        truth = src.loc[key]
        for base in BASE_FEATURE_COLS:
            got, want = row[f"home_{base}"], truth[base]
            if pd.isna(got) and pd.isna(want):
                continue
            if not np.isclose(got, want, equal_nan=True):
                mismatches.append(f"{key} {base}: table={got} source={want}")
    assert not mismatches, f"home-side join is wrong: {mismatches[:5]}"


def test_away_features_belong_to_the_away_team(model_table, rolling):
    src = rolling.set_index(["game_id", "team"])
    sample = model_table.dropna(subset=["away_pregame_team_score"]).sample(
        n=min(300, len(model_table)), random_state=42
    )

    mismatches = []
    for _, row in sample.iterrows():
        key = (row["game_id"], row["away_team"])
        if key not in src.index:
            mismatches.append(f"{key} missing from team_rolling_features")
            continue
        truth = src.loc[key]
        for base in BASE_FEATURE_COLS:
            got, want = row[f"away_{base}"], truth[base]
            if pd.isna(got) and pd.isna(want):
                continue
            if not np.isclose(got, want, equal_nan=True):
                mismatches.append(f"{key} {base}: table={got} source={want}")
    assert not mismatches, f"away-side join is wrong: {mismatches[:5]}"


def test_home_and_away_sides_are_not_identical(model_table):
    """A self-join bug could attach the same team to both sides."""
    played = model_table.dropna(subset=["home_pregame_team_score"])
    same = (
        played["home_pregame_team_score"] == played["away_pregame_team_score"]
    ).mean()
    assert same < 0.05, (
        f"{same:.1%} of games have identical home/away features -- the self-join "
        "is probably attaching one team to both sides"
    )


def test_home_team_matches_schedules(model_table):
    """Orientation against the source of truth for who was actually home."""
    sched = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "home_team", "away_team"]
    ].replace({"OAK": "LV"})
    merged = model_table[["game_id", "home_team", "away_team"]].merge(
        sched, on="game_id", suffixes=("_table", "_sched")
    )
    flipped = merged[merged["home_team_table"] != merged["home_team_sched"]]
    assert flipped.empty, f"{len(flipped)} games have home/away reversed vs schedules"


def test_drive_model_table_orientation():
    p = Path("data/processed/drive_model_table.parquet")
    assert p.exists(), "Run python -m src.features.build_all first"
    drive = pd.read_parquet(p)
    sched = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "home_team"]
    ].replace({"OAK": "LV"})
    merged = drive[["game_id", "home_team"]].merge(
        sched, on="game_id", suffixes=("_table", "_sched")
    )
    flipped = merged[merged["home_team_table"] != merged["home_team_sched"]]
    assert flipped.empty, f"{len(flipped)} drive-table games have sides reversed"


# --------------------------------------------- B5: adjusted-ratings week keys


def test_adjusted_ratings_are_refit_each_week():
    """If the merge key were off by a week, ratings would be identical across
    adjacent weeks. They must actually change."""
    r = pd.read_parquet("data/processed/adjusted_ratings.parquet")
    weeks = r[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    a, b = weeks.iloc[len(weeks) // 2], weeks.iloc[len(weeks) // 2 + 1]
    ra = r[(r.season == a.season) & (r.week == a.week)].set_index("team")
    rb = r[(r.season == b.season) & (r.week == b.week)].set_index("team")
    common = ra.index.intersection(rb.index)
    assert len(common) > 0
    identical = np.allclose(
        ra.loc[common, "pregame_adjusted_off_epa"],
        rb.loc[common, "pregame_adjusted_off_epa"],
    )
    assert not identical, "adjacent weeks have identical ratings -- not being refit"


def test_adjusted_ratings_match_a_refit_on_strictly_prior_games():
    """End-to-end: recompute one week's ratings from scratch using only games
    before that week's first kickoff, and demand the stored table matches.
    This is what catches an off-by-one-week merge, where a week's ratings would
    silently include that week's own results."""
    from src.features.build_adjusted_ratings import fit_ratings, load_config

    ratings = pd.read_parquet("data/processed/adjusted_ratings.parquet")
    team_games = pd.read_parquet("data/processed/team_game_stats.parquet")
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    halflife_days = load_config()["training"]["recency_half_life_weeks"] * 7
    teams = sorted(team_games["team"].unique())

    target = ratings[(ratings.season == 2024) & (ratings.week == 10)]
    assert not target.empty, "expected 2024 week 10 in the ratings table"

    cutoff = team_games[
        (team_games.season == 2024) & (team_games.week == 10)
    ].gameday.min()
    train = team_games[team_games.gameday < cutoff]
    off, _ = fit_ratings(train, teams, halflife_days, cutoff)

    stored = target.set_index("team")["pregame_adjusted_off_epa"]
    for team in teams:
        assert np.isclose(stored[team], off[team], atol=1e-9), (
            f"{team}: stored rating {stored[team]} != refit-on-prior-games "
            f"{off[team]} -- the week key may be off by one"
        )
