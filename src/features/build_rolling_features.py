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


def add_pregame_rolling_features(
    group: pd.DataFrame, halflife_days: float
) -> pd.DataFrame:
    group = group.sort_values("gameday").reset_index(drop=True)
    for col in STAT_COLS:
        ewm = (
            group[col]
            .ewm(halflife=pd.Timedelta(days=halflife_days), times=group["gameday"])
            .mean()
        )
        # Shift by 1: game N's feature = EWM computed through game N-1 only
        group[f"pregame_{col}"] = ewm.shift(1)
    group["rest_days"] = group["gameday"].diff().dt.days
    group["prior_games_played"] = range(
        len(group)
    )  # 0 for first game, 1 for second, etc.
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
