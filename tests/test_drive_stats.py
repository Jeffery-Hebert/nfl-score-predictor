"""
Validation gates for data/processed/drive_stats.parquet.
Run: pytest tests/test_drive_stats.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data

DATA_PATH = Path("data/processed/drive_stats.parquet")


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run src/features/build_drive_stats.py first"
    return pd.read_parquet(DATA_PATH)


def test_rates_sum_close_to_one(df):
    rate_cols = [c for c in df.columns if c.startswith("off_") and c.endswith("_rate")]
    rate_sum = df[rate_cols].sum(axis=1)
    assert (
        rate_sum.between(0.98, 1.02)
    ).mean() > 0.99, "Offensive outcome rates don't sum to ~1 for most rows"


def test_reasonable_drive_counts(df):
    assert (
        df["n_drives"].between(3, 25).mean() > 0.98
    ), "Unusual number of drives per game found -- investigate"


def test_points_scored_correlates_with_reality(df):
    """Approximated points (7/3/0 mapping) should correlate strongly with
    actual game scores, even though it won't match exactly due to the
    2-point-conversion/safety/opp-TD approximations."""
    schedules = pd.read_parquet("data/raw/schedules.parquet")
    home = schedules[["game_id", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "actual_score"}
    )
    away = schedules[["game_id", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "actual_score"}
    )
    actual = pd.concat([home, away])
    merged = df.merge(actual, on=["game_id", "team"], how="inner").dropna(
        subset=["actual_score"]
    )
    corr = merged["points_scored"].corr(merged["actual_score"])
    assert (
        corr > 0.9
    ), f"Approximated points only correlate {corr:.2f} with actual scores -- too low"
