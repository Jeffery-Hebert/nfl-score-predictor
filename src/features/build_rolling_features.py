"""
Builds pre-game, recency-weighted rolling features per team, using ONLY
games strictly before the current game (time-aware exponential weighting).

Leakage rule: the feature value attached to game N for a team must be
computable using nothing but that team's games 1..N-1.

Run: python src/features/build_rolling_features.py
Output: data/processed/team_rolling_features.parquet
"""

import pandas as pd
from pathlib import Path
import yaml

STAT_COLS = [
    "team_score",
    "opp_score",
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate_allowed",
]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def is_finale_week(season: int, week: int) -> bool:
    """Regular-season finale: week 17 for 2019-2020 (16-game era), week 18 for 2021+ (17-game era)."""
    return week == (17 if season in (2019, 2020) else 18)


def add_pregame_rolling_features(
    group: pd.DataFrame, halflife_days: float
) -> pd.DataFrame:
    group = group.sort_values("gameday").reset_index(drop=True)
    finale_mask = group.apply(lambda r: is_finale_week(r["season"], r["week"]), axis=1)

    for col in STAT_COLS:
        masked_series = group[col].where(~finale_mask)  # finale-week values become NaN
        ewm = masked_series.ewm(
            halflife=pd.Timedelta(days=halflife_days),
            times=group["gameday"],
            ignore_na=True,
        ).mean()
        group[f"pregame_{col}"] = ewm.shift(1)
    group["rest_days"] = group["gameday"].diff().dt.days
    group["prior_games_played"] = range(len(group))
    return group


def main():
    cfg = load_config()
    halflife_weeks = cfg["training"]["recency_half_life_weeks"]
    halflife_days = halflife_weeks * 7

    df = pd.read_parquet("data/processed/team_game_stats.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])

    pieces = []
    for team, group in df.groupby("team"):
        piece = add_pregame_rolling_features(group.copy(), halflife_days)
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)

    out_path = Path("data/processed/team_rolling_features.parquet")
    result.to_parquet(out_path, index=False)

    print(
        f"Built rolling features for {result['team'].nunique()} teams, "
        f"{len(result)} team-game rows"
    )
    print(f"Halflife: {halflife_weeks} weeks ({halflife_days} days)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
