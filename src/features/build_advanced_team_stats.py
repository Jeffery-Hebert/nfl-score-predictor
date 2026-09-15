"""
src/features/build_advanced_team_stats.py

Advanced per-team-game metrics that the production feature set is missing,
built from play-by-play already on disk. Four families, each with a football
reason rather than a "throw it at the model" reason:

PACE / VOLUME -- the structural gap.
    Points = efficiency x possessions. The production features carry EPA per
    PLAY and success RATE, and no measure of volume at all. Two offenses with
    identical EPA/play but 11 vs 13 drives do not score the same. Pace also
    feeds the opponent: a fast offense hands the other team more possessions,
    which is why it matters for predicting a score rather than a margin.

COMPETITIVE-SITUATION EPA -- noise reduction.
    Late-blowout snaps are prevent defense against backups. They are still
    plays, so they dilute a season-long EPA average with football that tells
    you nothing about a competitive matchup. Recomputed over plays with win
    probability between 0.20 and 0.80.

    IMPORTANT: uses nflfastR's `wp`, which is modelled from game state (score,
    time, field position, timeouts). It deliberately does NOT use `vegas_wp`
    or `vegas_wpa`, which incorporate the betting line -- no market-derived
    quantity may touch this model.

TURNOVER LUCK -- regression to the mean.
    Fumble RECOVERY is close to a coin flip and barely persists year to year,
    while fumbles FORCED does persist. A team whose EPA was inflated by
    recovering most of the loose balls will regress. Recording forced vs
    recovered separately lets a model weight the skill part and discount the
    luck part, instead of banking both as ability.

SPECIAL TEAMS -- absent entirely from the feature set.
    Field goals are points. A team that scores efficiently but cannot kick
    converts drives into 3s and 0s rather than 3s and 7s.

Run: python -m src.features.build_advanced_team_stats
Output: data/processed/advanced_team_stats.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Competitive-situation window. Outside this, the game is effectively decided
# and play-calling stops being about maximising expected points.
WP_LOW, WP_HIGH = 0.20, 0.80

# Market-derived columns that must never be read here. Listed explicitly so the
# boundary is visible and testable rather than a matter of remembering.
FORBIDDEN_MARKET_COLS = [
    "vegas_wp",
    "vegas_wpa",
    "vegas_home_wp",
    "vegas_home_wpa",
    "spread_line",
    "total_line",
    "away_moneyline",
    "home_moneyline",
]

TEAM_CODE_MAP = {"OAK": "LV"}


def _real_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp[pbp["play_type"].notna() & pbp["posteam"].notna()]


def build_pace_volume(pbp: pd.DataFrame) -> pd.DataFrame:
    """Possessions and tempo -- the volume half of points scored."""
    plays = _real_plays(pbp)
    off = (
        plays.groupby(["game_id", "posteam"])
        .agg(
            off_plays=("play_id", "count"),
            off_drives=("drive", "nunique"),
            off_no_huddle_rate=("no_huddle", "mean"),
            off_dropback_rate=("qb_dropback", "mean"),
            off_pass_oe=("pass_oe", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off["off_plays_per_drive"] = off["off_plays"] / off["off_drives"].replace(0, np.nan)

    # Seconds of game clock consumed per offensive play: raw tempo.
    tempo = plays.dropna(subset=["game_seconds_remaining"]).sort_values(
        ["game_id", "posteam", "game_seconds_remaining"], ascending=[True, True, False]
    )
    tempo["elapsed"] = -tempo.groupby(["game_id", "posteam"])[
        "game_seconds_remaining"
    ].diff()
    sec = (
        tempo[tempo["elapsed"].between(0, 60)]
        .groupby(["game_id", "posteam"])["elapsed"]
        .mean()
        .reset_index(name="off_sec_per_play")
        .rename(columns={"posteam": "team"})
    )
    return off.merge(sec, on=["game_id", "team"], how="left")


def build_competitive_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """EPA restricted to plays where the game is still in doubt."""
    plays = _real_plays(pbp)
    comp = plays[plays["wp"].between(WP_LOW, WP_HIGH)]

    off = (
        comp.groupby(["game_id", "posteam"])
        .agg(
            off_epa_competitive=("epa", "mean"),
            off_success_competitive=("success", "mean"),
            off_competitive_plays=("play_id", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    deff = (
        comp[comp["defteam"].notna()]
        .groupby(["game_id", "defteam"])
        .agg(
            def_epa_competitive=("epa", "mean"),
            def_success_competitive=("success", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return off.merge(deff, on=["game_id", "team"], how="outer")


def build_turnover_luck(pbp: pd.DataFrame) -> pd.DataFrame:
    """Separate the persistent part of turnovers from the coin-flip part."""
    plays = _real_plays(pbp)
    off = (
        plays.groupby(["game_id", "posteam"])
        .agg(
            off_fumbles=("fumble", "sum"),
            off_fumbles_lost=("fumble_lost", "sum"),
            off_interceptions=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    # Share of own fumbles retained. League-average is ~50% and it does not
    # persist, so distance from 0.5 is mostly luck due to regress.
    off["off_fumble_retain_rate"] = np.where(
        off["off_fumbles"] > 0,
        1 - off["off_fumbles_lost"] / off["off_fumbles"].replace(0, np.nan),
        np.nan,
    )
    deff = (
        plays[plays["defteam"].notna()]
        .groupby(["game_id", "defteam"])
        .agg(
            def_fumbles_forced=("fumble", "sum"),
            def_fumbles_recovered=("fumble_lost", "sum"),
            def_interceptions=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return off.merge(deff, on=["game_id", "team"], how="outer")


def build_special_teams(pbp: pd.DataFrame) -> pd.DataFrame:
    """Kicking converts drives into points, or fails to."""
    fg = pbp[pbp["field_goal_result"].notna() & pbp["posteam"].notna()]
    st = (
        fg.groupby(["game_id", "posteam"])
        .agg(
            fg_attempts=("field_goal_result", "size"),
            fg_made=("field_goal_result", lambda s: (s == "made").sum()),
            fg_avg_distance=("kick_distance", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    st["fg_pct"] = st["fg_made"] / st["fg_attempts"].replace(0, np.nan)
    return st


def main():
    pbp = pd.read_parquet("data/raw/pbp.parquet")

    leaked = [c for c in FORBIDDEN_MARKET_COLS if c in pbp.columns]
    print(f"Market-derived columns present in pbp and deliberately NOT read: {leaked}")

    pieces = [
        build_pace_volume(pbp),
        build_competitive_epa(pbp),
        build_turnover_luck(pbp),
        build_special_teams(pbp),
    ]
    result = pieces[0]
    for p in pieces[1:]:
        result = result.merge(p, on=["game_id", "team"], how="outer")

    result["team"] = result["team"].replace(TEAM_CODE_MAP)

    out = Path("data/processed/advanced_team_stats.parquet")
    result.to_parquet(out, index=False)
    print(f"Built {len(result)} team-game rows, {len(result.columns)} columns")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
