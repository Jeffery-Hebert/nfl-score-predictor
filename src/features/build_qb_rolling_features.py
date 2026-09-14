"""
src/features/build_qb_rolling_features.py

Identifies each team-game's starting QB from play-by-play (most pass attempts),
then builds leakage-safe pregame QB rolling stats using the identical
EWM+shift(1) pattern already validated for team-level features.

Run: python src/features/build_qb_rolling_features.py
Output: data/processed/qb_rolling_features.parquet
"""

import pandas as pd
from pathlib import Path
import yaml

QB_STAT_COLS = ["qb_epa_per_play", "qb_completion_pct", "qb_success_rate"]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def identify_starters(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the QB with the most pass attempts for that team."""
    pass_plays = pbp[pbp["play_type"] == "pass"].dropna(subset=["passer_id"])
    attempts = (
        pass_plays.groupby(["game_id", "posteam", "passer_id", "passer_player_name"])
        .size()
        .reset_index(name="attempts")
    )
    starters = attempts.sort_values("attempts", ascending=False).drop_duplicates(
        subset=["game_id", "posteam"]
    )
    return starters.rename(columns={"posteam": "team"})[
        ["game_id", "team", "passer_id", "passer_player_name"]
    ]


def build_qb_game_stats(pbp: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    pass_plays = pbp[pbp["play_type"] == "pass"].dropna(subset=["passer_id"])
    qb_game = (
        pass_plays.groupby(["game_id", "posteam", "passer_id"])
        .agg(
            qb_epa_per_play=("epa", "mean"),
            qb_completion_pct=("complete_pass", "mean"),
            qb_success_rate=("success", "mean"),
            qb_attempts=("epa", "size"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return qb_game.merge(starters, on=["game_id", "team", "passer_id"], how="inner")


def is_finale_week(season: int, week: int) -> bool:
    return week == (17 if season in (2019, 2020) else 18)


def add_pregame_qb_features(group: pd.DataFrame, halflife_days: float) -> pd.DataFrame:
    group = group.sort_values("gameday").reset_index(drop=True)
    finale_mask = group.apply(lambda r: is_finale_week(r["season"], r["week"]), axis=1)

    for col in QB_STAT_COLS:
        masked_series = group[col].where(~finale_mask)
        ewm = masked_series.ewm(
            halflife=pd.Timedelta(days=halflife_days),
            times=group["gameday"],
            ignore_na=True,
        ).mean()
        group[f"pregame_{col}"] = ewm.shift(1)
    group["prior_qb_games_started"] = range(len(group))
    return group


def main():
    cfg = load_config()
    halflife_days = cfg["training"]["recency_half_life_weeks"] * 7

    pbp = pd.read_parquet("data/raw/pbp.parquet")
    schedules = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "season", "week", "gameday"]
    ]
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    starters = identify_starters(pbp)
    qb_game_stats = build_qb_game_stats(pbp, starters)
    df = qb_game_stats.merge(schedules, on="game_id", how="inner")

    pieces = []
    for passer_id, group in df.groupby("passer_id"):
        pieces.append(add_pregame_qb_features(group.copy(), halflife_days))
    result = pd.concat(pieces, ignore_index=True)

    out_path = Path("data/processed/qb_rolling_features.parquet")
    result.to_parquet(out_path, index=False)
    print(
        f"Built QB rolling features for {result['passer_id'].nunique()} QBs, {len(result)} rows"
    )


if __name__ == "__main__":
    main()
