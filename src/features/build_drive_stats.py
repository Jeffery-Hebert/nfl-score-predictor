"""
Aggregates play-by-play into one row per team per game with drive-outcome
rates (offense: this team's own drives; defense: opponent drives against
this team). Foundation for the Monte Carlo drive simulator.

Point attribution: Touchdown=7 and Field goal=3 to the team with the ball.
"Opp touchdown" (a defensive return TD, 490 drives) is 7 points to the
DEFENSE, and "Safety" (109 drives) is 2 points to the defense. Those used to
map to 0 for both sides, so 3,648 real points were assigned to nobody and
points_scored under-counted by 0.66 per team-game. est_points_for now adds a
team's defensive scoring to its offensive scoring; points_scored is kept
offense-only so the existing drive-rate features are unchanged.

Still approximate: 2-point conversions, extra-point misses and return TDs on
kicks are not modelled. Touchdowns are flat 7.

Run: python src/features/build_drive_stats.py
Output: data/processed/drive_stats.parquet
"""

import pandas as pd
from pathlib import Path

# Points to the team with the ball on that drive.
POINTS_MAP = {"Touchdown": 7, "Field goal": 3}
# Points to the DEFENDING team on that drive -- return TDs and safeties.
DEFENSE_POINTS_MAP = {"Opp touchdown": 7, "Safety": 2}
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
    drives["def_points"] = (
        drives["fixed_drive_result"].map(DEFENSE_POINTS_MAP).fillna(0)
    )
    return drives


def build_offense_drive_stats(drives: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "n_drives": ("drive", "count"),
        "points_scored": ("points", "sum"),
        # points this team's own offense handed the opponent (pick-six, safety)
        "points_given_up": ("def_points", "sum"),
    }
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
    agg = {
        "n_drives_faced": ("drive", "count"),
        "points_allowed": ("points", "sum"),
        # points this team's DEFENSE scored on the opponent's drives
        "def_points_scored": ("def_points", "sum"),
    }
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

    # Full estimated points: offensive scoring plus the team's own defensive
    # scoring. points_scored stays offense-only so drive-RATE features are
    # unchanged; est_points_for is the corrected total.
    result["est_points_for"] = result["points_scored"].fillna(0) + result[
        "def_points_scored"
    ].fillna(0)
    result["est_points_against"] = result["points_allowed"].fillna(0) + result[
        "points_given_up"
    ].fillna(0)

    out_path = Path("data/processed/drive_stats.parquet")
    result.to_parquet(out_path, index=False)
    print(f"Built drive stats for {len(result)} team-game rows")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
