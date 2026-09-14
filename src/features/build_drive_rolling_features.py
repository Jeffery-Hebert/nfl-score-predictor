"""
Builds pre-game, recency-weighted rolling drive-outcome-rate features per
team, using the same leakage-safe EWM+shift pattern as build_rolling_features.py.
Foundation for the Monte Carlo drive simulator.

Run: python src/features/build_drive_rolling_features.py
Output: data/processed/team_drive_rolling_features.parquet
"""

import pandas as pd
from pathlib import Path
import yaml

DRIVE_STAT_COLS = [
    "off_touchdown_rate",
    "off_field_goal_rate",
    "off_punt_rate",
    "off_turnover_rate",
    "off_turnover_on_downs_rate",
    "off_missed_field_goal_rate",
    "off_end_of_half_rate",
    "off_opp_touchdown_rate",
    "off_safety_rate",
    "def_touchdown_rate",
    "def_field_goal_rate",
    "def_punt_rate",
    "def_turnover_rate",
    "def_turnover_on_downs_rate",
    "def_missed_field_goal_rate",
    "def_end_of_half_rate",
    "def_opp_touchdown_rate",
    "def_safety_rate",
    "n_drives",
    "n_drives_faced",
]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def is_finale_week(season: int, week: int) -> bool:
    """Regular-season finale: week 17 for 2019-2020 (16-game era), week 18 for 2021+ (17-game era)."""
    return week == (17 if season in (2019, 2020) else 18)


def add_pregame_rolling_drive_features(
    group: pd.DataFrame, halflife_days: float
) -> pd.DataFrame:
    group = group.sort_values("gameday").reset_index(drop=True)
    finale_mask = group.apply(lambda r: is_finale_week(r["season"], r["week"]), axis=1)

    for col in DRIVE_STAT_COLS:
        masked_series = group[col].where(~finale_mask)
        ewm = masked_series.ewm(
            halflife=pd.Timedelta(days=halflife_days),
            times=group["gameday"],
            ignore_na=True,
        ).mean()
        group[f"pregame_{col}"] = ewm.shift(1)
    return group


def main():
    cfg = load_config()
    halflife_days = cfg["training"]["recency_half_life_weeks"] * 7

    drive_stats = pd.read_parquet("data/processed/drive_stats.parquet")
    schedules = pd.read_parquet("data/raw/schedules.parquet")[["game_id", "gameday"]]
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    df = drive_stats.merge(schedules, on="game_id", how="inner")

    pieces = []
    for team, group in df.groupby("team"):
        piece = add_pregame_rolling_drive_features(group.copy(), halflife_days)
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)

    out_path = Path("data/processed/team_drive_rolling_features.parquet")
    result.to_parquet(out_path, index=False)
    print(
        f"Built drive rolling features for {result['team'].nunique()} teams, {len(result)} rows"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
