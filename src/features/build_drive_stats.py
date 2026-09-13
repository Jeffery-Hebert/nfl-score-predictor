"""
Aggregates play-by-play into one row per team per game with drive-outcome
rates (offense: this team's own drives; defense: opponent drives against
this team). Foundation for the Monte Carlo drive simulator.

Approximation note: Touchdown=7, Field goal=3, all other outcomes=0.
"Opp touchdown" and "Safety" (each <1% of drives) are treated as 0 points
for this team's own drive tally, rather than modeling the point swing they'd
actually give the opponent -- a deliberate simplification given their rarity.

Run: python src/features/build_drive_stats.py
Output: data/processed/drive_stats.parquet
"""

import pandas as pd
from pathlib import Path

POINTS_MAP = {"Touchdown": 7, "Field goal": 3}
OUTCOME_CATS = [
    "Touchdown",
    "Field goal",
    "Punt",
    "Turnover",
    "Turnover on downs",
    "Missed field goal",
    "End of half",
    "Opp touchdown",
    "Safety",
]


def dedupe_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    drives = (
        pbp.dropna(subset=["drive", "posteam", "defteam"])
        .groupby(["game_id", "posteam", "defteam", "drive"])
        .agg(fixed_drive_result=("fixed_drive_result", "first"))
        .reset_index()
    )
    drives["points"] = drives["fixed_drive_result"].map(POINTS_MAP).fillna(0)
    return drives


def build_offense_drive_stats(drives: pd.DataFrame) -> pd.DataFrame:
    agg = {"n_drives": ("drive", "count"), "points_scored": ("points", "sum")}
    for cat in OUTCOME_CATS:
        agg[f"off_{cat.lower().replace(' ', '_')}_count"] = (
            "fixed_drive_result",
            lambda x, c=cat: (x == c).sum(),
        )
    off = drives.groupby(["game_id", "posteam"]).agg(**agg).reset_index()
    off = off.rename(columns={"posteam": "team"})
    for cat in OUTCOME_CATS:
        col = f"off_{cat.lower().replace(' ', '_')}_count"
        off[f"off_{cat.lower().replace(' ', '_')}_rate"] = off[col] / off["n_drives"]
    return off


def build_defense_drive_stats(drives: pd.DataFrame) -> pd.DataFrame:
    agg = {"n_drives_faced": ("drive", "count"), "points_allowed": ("points", "sum")}
    for cat in OUTCOME_CATS:
        agg[f"def_{cat.lower().replace(' ', '_')}_count"] = (
            "fixed_drive_result",
            lambda x, c=cat: (x == c).sum(),
        )
    deff = drives.groupby(["game_id", "defteam"]).agg(**agg).reset_index()
    deff = deff.rename(columns={"defteam": "team"})
    for cat in OUTCOME_CATS:
        col = f"def_{cat.lower().replace(' ', '_')}_count"
        deff[f"def_{cat.lower().replace(' ', '_')}_rate"] = (
            deff[col] / deff["n_drives_faced"]
        )
    return deff


def main():
    pbp = pd.read_parquet("data/raw/pbp.parquet")
    drives = dedupe_drives(pbp)

    off = build_offense_drive_stats(drives)
    deff = build_defense_drive_stats(drives)
    result = off.merge(deff, on=["game_id", "team"], how="outer")

    out_path = Path("data/processed/drive_stats.parquet")
    result.to_parquet(out_path, index=False)
    print(f"Built drive stats for {len(result)} team-game rows")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
