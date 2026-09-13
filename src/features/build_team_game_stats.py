"""
Aggregates play-by-play data into one row per team per game (offense +
defense efficiency). This is the building block for rolling/recency-weighted
features used by every downstream model.

Run: python src/features/build_team_game_stats.py
Output: data/processed/team_game_stats.parquet
"""

import pandas as pd
from pathlib import Path

TEAM_CODE_MAP = {
    "OAK": "LV",  # Raiders: Oakland (through 2019) -> Las Vegas (2020+)
}


def normalize_team_codes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].replace(TEAM_CODE_MAP)
    return df


def build_offense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[pbp["play_type"].notna() & pbp["posteam"].notna()]
    off = (
        plays.groupby(["game_id", "posteam"])
        .agg(
            off_plays=("epa", "count"),
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return off


def build_defense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[pbp["play_type"].notna() & pbp["defteam"].notna()]
    deff = (
        plays.groupby(["game_id", "defteam"])
        .agg(
            def_plays=("epa", "count"),
            def_epa_per_play=("epa", "mean"),
            def_success_rate_allowed=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return deff


def build_team_game_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from the home/away perspective."""
    schedules = normalize_team_codes(schedules, ["home_team", "away_team"])
    home = schedules.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["team_score"] = home["home_score"]
    home["opp_score"] = home["away_score"]
    home["is_home"] = 1

    away = schedules.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["team_score"] = away["away_score"]
    away["opp_score"] = away["home_score"]
    away["is_home"] = 0

    keep_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "team",
        "opponent",
        "team_score",
        "opp_score",
        "is_home",
    ]
    return pd.concat([home[keep_cols], away[keep_cols]], ignore_index=True)


def main():
    schedules = pd.read_parquet("data/raw/schedules.parquet")
    pbp = pd.read_parquet("data/raw/pbp.parquet")

    team_games = build_team_game_rows(schedules)
    off_stats = build_offense_stats(pbp)
    def_stats = build_defense_stats(pbp)

    result = team_games.merge(off_stats, on=["game_id", "team"], how="left")
    result = result.merge(def_stats, on=["game_id", "team"], how="left")

    out_path = Path("data/processed/team_game_stats.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    print(f"Built {len(result)} team-game rows ({len(result)//2} games)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
