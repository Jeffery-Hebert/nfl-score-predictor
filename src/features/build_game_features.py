"""
Assembles the final model-ready table: one row per game, with home and away
teams' pregame rolling features side-by-side, and the target scores attached.

Run: python -m src.features.build_game_features
Output: data/processed/model_table.parquet
"""

import pandas as pd
from pathlib import Path

# Single source of truth -- this list used to be duplicated here by hand and
# had to be kept in sync with src/models/common.py. See that module's docstring.
from src.models.common import BASE_FEATURE_COLS as FEATURE_COLS


def main():
    rolling = pd.read_parquet("data/processed/team_rolling_features.parquet")

    home = rolling[rolling["is_home"] == 1][
        ["game_id", "team", "opponent"] + FEATURE_COLS
    ]
    home = home.rename(columns={c: f"home_{c}" for c in FEATURE_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + FEATURE_COLS]
    away = away.rename(columns={c: f"away_{c}" for c in FEATURE_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = home.merge(away, on=["game_id", "away_team"], how="inner")

    schedules = pd.read_parquet("data/raw/schedules.parquet")
    final = merged.merge(
        schedules[["game_id", "season", "week", "gameday", "home_score", "away_score"]],
        on="game_id",
        how="left",
    )

    out_path = Path("data/processed/model_table.parquet")
    final.to_parquet(out_path, index=False)

    print(f"Built model table: {len(final)} games, {len(final.columns)} columns")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
