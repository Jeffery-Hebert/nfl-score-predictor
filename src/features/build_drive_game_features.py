"""
Joins home/away pregame drive-outcome-rate features into one row per game,
same pattern as build_game_features.py but for drive-level data.

Run: python src/features/build_drive_game_features.py
Output: data/processed/drive_model_table.parquet
"""

import pandas as pd
from pathlib import Path

from src.features.build_team_game_stats import normalize_team_codes

CATS = [
    "touchdown",
    "field_goal",
    "punt",
    "turnover",
    "turnover_on_downs",
    "missed_field_goal",
    "end_of_half",
    "opp_touchdown",
    "safety",
]
FEATURE_COLS = (
    [f"pregame_off_{c}_rate" for c in CATS]
    + [f"pregame_def_{c}_rate" for c in CATS]
    + ["pregame_n_drives", "pregame_n_drives_faced"]
)


def main():
    rolling = pd.read_parquet("data/processed/team_drive_rolling_features.parquet")

    home = rolling[["game_id", "team"] + FEATURE_COLS].rename(
        columns={"team": "home_team"}
    )
    home = home.rename(columns={c: f"home_{c}" for c in FEATURE_COLS})

    away = rolling[["game_id", "team"] + FEATURE_COLS].rename(
        columns={"team": "away_team"}
    )
    away = away.rename(columns={c: f"away_{c}" for c in FEATURE_COLS})

    merged = home.merge(away, on="game_id", how="inner")
    merged = merged[merged["home_team"] != merged["away_team"]]  # drop self-joins

    schedules = pd.read_parquet("data/raw/schedules.parquet")
    # Play-by-play already codes the 2019 Raiders as LV; the schedule still says
    # OAK. Merging on team names without normalising silently dropped all 16 of
    # their 2019 games from this table.
    schedules = normalize_team_codes(schedules.copy(), ["home_team", "away_team"])
    final = merged.merge(
        schedules[
            [
                "game_id",
                "home_team",
                "away_team",
                "season",
                "week",
                "gameday",
                "home_score",
                "away_score",
            ]
        ],
        on=["game_id", "home_team", "away_team"],
        how="inner",
    )

    out_path = Path("data/processed/drive_model_table.parquet")
    final.to_parquet(out_path, index=False)
    print(f"Built drive model table: {len(final)} games")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
