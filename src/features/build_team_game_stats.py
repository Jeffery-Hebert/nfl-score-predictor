"""
Aggregates play-by-play data into one row per team per game (offense +
defense efficiency), split by pass/rush, plus CPOE. Building block for
rolling/recency-weighted features used by every downstream model.

Play COUNTS are emitted next to every per-play mean (off_pass_plays,
off_rush_plays, def_pass_plays, def_rush_plays). A mean without its
denominator cannot be re-weighted or shrunk downstream: averaging eleven
games' rush-EPA means treats a 9-carry game and a 38-carry game as equal
evidence, which they are not. build_split_efficiency.py needs the counts to
weight by volume and to size its shrinkage, so they are carried here rather
than recomputed from play-by-play a second time.

Every per-play rate here -- success rate, EPA per play, and their counts -- is
measured on SNAPS FROM SCRIMMAGE only: see scrimmage_snaps() for which plays
those are and why special teams are left out.

Run: python src/features/build_team_game_stats.py
Output: data/processed/team_game_stats.parquet
"""

import pandas as pd
from pathlib import Path

TEAM_CODE_MAP = {"OAK": "LV"}

# nflverse play_type values that are snaps from scrimmage: a sack is a "pass"
# and a quarterback scramble a "run", exactly as the pass/rush split counts them.
SCRIMMAGE_PLAY_TYPES = ("pass", "run")


def normalize_team_codes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].replace(TEAM_CODE_MAP)
    return df


def scrimmage_snaps(pbp: pd.DataFrame, side: str) -> pd.DataFrame:
    """The plays that measure an offence (side="posteam") or a defence
    (side="defteam"): snaps from scrimmage that counted.

    Until 2026-10-01 these stats were taken over EVERY row with a play_type, so
    23% of the plays behind "offensive success rate" were not offensive snaps:
    kickoffs (credited to the receiving team), punts, extra points, field goals,
    kneel-downs, spikes and penalty-nullified plays. Extra points succeed 94% of
    the time and field goals 85%, one per touchdown and per stalled drive in
    range, so the rate partly counted SCORING -- beside a points feature the
    model already has -- and docked teams for kneeling out a win.

    Left out, deliberately:
      special teams   kickoff, punt, extra_point, field_goal -- a different unit
      clock plays     qb_kneel, qb_spike -- not attempts to gain anything
      no_play         snaps wiped out by a penalty. nflfastR's own convention
                      (pass == 1 | rush == 1) keeps the ~11,000 of these that
                      were dropbacks or runs; it was measured as an arm of
                      src/experiments/test_success_rate_definition.py and was
                      no better than leaving them out, which is also what the
                      pass/rush split below has always done.
    """
    return pbp[pbp["play_type"].isin(SCRIMMAGE_PLAY_TYPES) & pbp[side].notna()]


def build_offense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = scrimmage_snaps(pbp, "posteam")
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

    pass_plays = plays[plays["play_type"] == "pass"]
    pass_stats = (
        pass_plays.groupby(["game_id", "posteam"])
        .agg(
            off_pass_epa_per_play=("epa", "mean"),
            off_pass_plays=("epa", "count"),
            off_cpoe=("cpoe", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    rush_plays = plays[plays["play_type"] == "run"]
    rush_stats = (
        rush_plays.groupby(["game_id", "posteam"])
        .agg(
            off_rush_epa_per_play=("epa", "mean"),
            off_rush_plays=("epa", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    off = off.merge(pass_stats, on=["game_id", "team"], how="left")
    off = off.merge(rush_stats, on=["game_id", "team"], how="left")
    return off


def build_defense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = scrimmage_snaps(pbp, "defteam")
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

    pass_plays = plays[plays["play_type"] == "pass"]
    pass_stats = (
        pass_plays.groupby(["game_id", "defteam"])
        .agg(
            def_pass_epa_per_play_allowed=("epa", "mean"),
            def_pass_plays=("epa", "count"),
            def_cpoe_allowed=("cpoe", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    rush_plays = plays[plays["play_type"] == "run"]
    rush_stats = (
        rush_plays.groupby(["game_id", "defteam"])
        .agg(
            def_rush_epa_per_play_allowed=("epa", "mean"),
            def_rush_plays=("epa", "count"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    deff = deff.merge(pass_stats, on=["game_id", "team"], how="left")
    deff = deff.merge(rush_stats, on=["game_id", "team"], how="left")
    return deff


def build_team_game_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    schedules = normalize_team_codes(schedules, ["home_team", "away_team"])
    # C3: neutral-site games (London/Mexico/Munich/Super Bowl) have no true home
    # team, but were modelled as ordinary home games.
    # C4: playoff games were mixed into the regular season with no marker.
    # Both are known before kickoff.
    schedules = schedules.copy()
    schedules["is_neutral_site"] = (schedules["location"] == "Neutral").astype(int)
    schedules["is_playoff"] = (schedules["game_type"] != "REG").astype(int)
    home = schedules.copy()
    home["team"], home["opponent"] = home["home_team"], home["away_team"]
    home["team_score"], home["opp_score"] = home["home_score"], home["away_score"]
    home["is_home"] = 1
    # C1: nflverse ships correct rest days (max 16, correct across season
    # boundaries). The pipeline used to recompute it as gameday.diff(), which
    # reported up to 260 "rest days" across an offseason -- a number that does
    # not describe rest in any football sense.
    home["rest_days"] = home["home_rest"]

    away = schedules.copy()
    away["team"], away["opponent"] = away["away_team"], away["home_team"]
    away["team_score"], away["opp_score"] = away["away_score"], away["home_score"]
    away["is_home"] = 0
    away["rest_days"] = away["away_rest"]

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
        "rest_days",
        "is_neutral_site",
        "is_playoff",
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
    splits = [
        "off_pass_plays",
        "off_rush_plays",
        "def_pass_plays",
        "def_rush_plays",
    ]
    print(
        "Per-split play counts (for build_split_efficiency.py): "
        + ", ".join(f"{c} mean {result[c].mean():.1f}" for c in splits)
    )


if __name__ == "__main__":
    main()
