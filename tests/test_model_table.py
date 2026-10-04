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
        if col.endswith("injury_impact"):
            continue  # missing by design where no report exists -- next test
        rate = later[col].isna().mean()
        assert rate == 0, (
            f"{col} is {rate:.2%} NaN after the first season -- pregame features "
            "should only be missing at a team's genuine cold start"
        )


def test_injury_impact_is_missing_only_where_no_pregame_report_exists(df):
    """The one feature that may be missing after the first season, by design:
    a PLAYED game whose team has no injury report published before kickoff in
    the data -- unknown, not "healthy" (build_injury_features.unmeasurable).
    Checked against the raw reports rather than the builder's output, so a
    failed join cannot pass itself off as a missing report."""
    from src.features.build_injury_features import TEAM_CODE_MAP
    from src.schedule import kickoff_utc

    sched = pd.read_parquet("data/raw/schedules.parquet")
    sched["kickoff"] = kickoff_utc(sched)
    inj = pd.read_parquet(
        "data/raw/injuries.parquet", columns=["season", "week", "team", "date_modified"]
    )
    inj["team"] = inj["team"].replace(TEAM_CODE_MAP)
    inj["season"] = inj["season"].astype(int)
    inj["modified"] = pd.to_datetime(inj["date_modified"], utc=True, errors="coerce")

    later = df[df["season"] > df["season"].min()]
    played = later["home_score"].notna()
    for side in ("home", "away"):
        miss = later[later[f"{side}_injury_impact"].isna()]
        assert miss["home_score"].notna().all(), (
            f"{side}_injury_impact missing for a game not yet played -- it must "
            "stay 0 until the final report (predict_week waits for it)"
        )
        teams = miss[["game_id", "season", "week", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        rows = teams.merge(sched[["game_id", "kickoff"]], on="game_id").merge(
            inj, on=["season", "week", "team"], how="inner"
        )
        pregame = rows[rows["modified"].isna() | (rows["modified"] < rows["kickoff"])]
        assert pregame.empty, (
            f"{side}_injury_impact is missing for {pregame['game_id'].nunique()} "
            f"games that DO have a pre-game report, e.g. "
            f"{sorted(pregame['game_id'].unique())[:4]} -- a join is failing"
        )
    rate = later[["home_injury_impact", "away_injury_impact"]].isna().to_numpy()
    assert rate.sum() / (2 * played.sum()) < 0.01, "over 1% of team-games unmeasured"


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
