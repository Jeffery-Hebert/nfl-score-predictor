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
from src.models.common import INJURY_FEATURE_COLS, SPLIT_FEATURE_COLS


def assemble(
    rolling: pd.DataFrame,
    inj: pd.DataFrame,
    split: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """The model table from its four inputs -- no file I/O.

    Shared with the experiments that rebuild a feature table in memory
    (tune_halflife.py, test_offseason_decay.py). They used to copy this
    assembly by hand, froze at whatever the feature list was that day, and
    crashed with a KeyError once injury and split features joined production.
    """

    # C3/C4: game-level context, identical on both rows of a game, so it is
    # taken from the home side once rather than prefixed home_/away_.
    home = rolling[rolling["is_home"] == 1][
        ["game_id", "team", "opponent", "is_neutral_site", "is_playoff"] + FEATURE_COLS
    ]
    home = home.rename(columns={c: f"home_{c}" for c in FEATURE_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + FEATURE_COLS]
    away = away.rename(columns={c: f"away_{c}" for c in FEATURE_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = home.merge(away, on=["game_id", "away_team"], how="inner")

    # Pregame availability. Keyed (game_id, team) like the rolling features but
    # built from the official injury report -- see build_injury_features.py.
    inj_home = inj[["game_id", "team"] + INJURY_FEATURE_COLS].rename(
        columns={
            **{c: f"home_{c}" for c in INJURY_FEATURE_COLS},
            "team": "home_team",
        }
    )
    inj_away = inj[["game_id", "team"] + INJURY_FEATURE_COLS].rename(
        columns={
            **{c: f"away_{c}" for c in INJURY_FEATURE_COLS},
            "team": "away_team",
        }
    )
    merged = merged.merge(inj_home, on=["game_id", "home_team"], how="left")
    merged = merged.merge(inj_away, on=["game_id", "away_team"], how="left")

    # Pass/rush efficiency split. Same grain and same join shape as the injury
    # features above; separate table because the estimator is different (volume
    # weighted and shrunk, not a plain EWM) -- see build_split_efficiency.py.
    split_home = split[["game_id", "team"] + SPLIT_FEATURE_COLS].rename(
        columns={
            **{c: f"home_{c}" for c in SPLIT_FEATURE_COLS},
            "team": "home_team",
        }
    )
    split_away = split[["game_id", "team"] + SPLIT_FEATURE_COLS].rename(
        columns={
            **{c: f"away_{c}" for c in SPLIT_FEATURE_COLS},
            "team": "away_team",
        }
    )
    merged = merged.merge(split_home, on=["game_id", "home_team"], how="left")
    merged = merged.merge(split_away, on=["game_id", "away_team"], how="left")

    # C5: went_to_ot is a TRAINING-side column, never a feature. It is 0%
    # populated before kickoff, so using it as an input would be target leakage
    # (OT games average 53.9 total points against 45.3). Models use it only to
    # down-weight historical games whose scores an extra period inflated --
    # the same reasoning as finale-week masking.
    sched_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_score",
        "away_score",
        "overtime",
    ]
    final = merged.merge(schedules[sched_cols], on="game_id", how="left")
    final["went_to_ot"] = final["overtime"].fillna(0).astype(int)
    final = final.drop(columns=["overtime"])

    return final


def main():
    final = assemble(
        pd.read_parquet("data/processed/team_rolling_features.parquet"),
        pd.read_parquet("data/processed/injury_features.parquet"),
        pd.read_parquet("data/processed/split_efficiency.parquet"),
        pd.read_parquet("data/raw/schedules.parquet"),
    )
    out_path = Path("data/processed/model_table.parquet")
    final.to_parquet(out_path, index=False)

    print(f"Built model table: {len(final)} games, {len(final.columns)} columns")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
