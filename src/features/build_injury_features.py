"""
src/features/build_injury_features.py

Turns official injury reports into a COMPRESSED pregame availability signal --
two columns per team, not a dozen.

Why compressed. test_advanced_metrics.py established that this model is
feature-saturated: at ~1,900 training rows, adding raw features degrades it
even under regularization. So the goal is not "expose the injury data" but
"reduce it to the smallest number of columns that carry the football".

  injury_impact   sum over unavailable players of
                      positional_value x prior_snap_share
                  Questionable counts at half weight, since Questionable
                  players frequently play. The key idea: a starting left
                  tackle being out and a fourth-string linebacker being out
                  are not the same event, and snap share is how you tell them
                  apart. A plain count of injured players cannot.

  qb_out          the starting quarterback is Out or Doubtful. Kept separate
                  rather than folded into the sum because losing a QB is a
                  different KIND of event, not merely the highest-weighted
                  position.

LEAKAGE SAFETY -- two guards, both of which needed a correction:

  1. Rows whose `date_modified` is at or after kickoff are dropped (22 rows
     across 2019-2026). NOTE: nflverse stopped populating date_modified from
     2025 onward, so ~6,250 rows carry no timestamp. Those are KEPT and
     reported separately. A first version dropped them along with genuinely
     late rows, silently discarding the two most recent seasons entirely.
     They remain keyed to a season/week, and a week's injury report is by
     construction published before that week's games -- a weaker but still
     pre-game guarantee.

  2. Snap share is read from a player's history via an AS-OF join at kickoff.

     The obvious implementation is wrong. A first version joined snap share on
     (game_id, gsis_id) -- but a player ruled Out has no snap row for that
     game, because he did not play. The join therefore failed for exactly the
     players the feature exists to measure: 0 of 289 QB-out rows matched and
     qb_out came out identically zero. The as-of join asks the correct
     question instead: what was this player's role in the games BEFORE this
     one?

Positional values follow the standard positional-value ordering in football
analytics: quarterback dominates, then the premium positions that are hardest
to replace (tackle, edge, corner, receiver), then interior and off-ball roles,
with running back low as the most replaceable starting position. These are
deliberately coarse judgement calls, not fitted parameters.

Run: python -m src.features.build_injury_features
Output: data/processed/injury_features.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

UNAVAILABLE = {"Out", "Doubtful"}
QUESTIONABLE_WEIGHT = 0.5

POSITION_VALUE = {
    "QB": 1.00,
    "T": 0.40,
    "OT": 0.40,
    "LT": 0.40,
    "RT": 0.40,
    "EDGE": 0.38,
    "DE": 0.38,
    "OLB": 0.34,
    "CB": 0.34,
    "WR": 0.32,
    "DT": 0.28,
    "NT": 0.24,
    "S": 0.26,
    "SS": 0.26,
    "FS": 0.26,
    "TE": 0.24,
    "G": 0.22,
    "OG": 0.22,
    "C": 0.22,
    "OL": 0.22,
    "LB": 0.22,
    "ILB": 0.22,
    "MLB": 0.22,
    "DB": 0.26,
    "DL": 0.28,
    "RB": 0.16,
    "FB": 0.08,
    "K": 0.10,
    "P": 0.06,
    "LS": 0.04,
}
DEFAULT_POSITION_VALUE = 0.20
QB_STARTER_SNAP_THRESHOLD = 0.50


def kickoff_times(schedules: pd.DataFrame) -> pd.DataFrame:
    """UTC kickoff per game, for the leakage filter and the as-of join. The
    construction is the shared one in src/schedule.py."""
    from src.schedule import kickoff_utc

    s = schedules.copy()
    s["kickoff"] = kickoff_utc(s)
    return s


def snap_history(snaps, schedules, players) -> pd.DataFrame:
    """Per-player running snap share, dated, for an as-of lookup.

    One row per player-game with `share_to_date`: mean role share over that
    player's games up to and including that game. Callers as-of join at a
    strictly later kickoff, so the game being predicted is never included.
    """
    s = snaps.copy()
    s["role_share"] = s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    s = s.merge(schedules[["game_id", "kickoff"]], on="game_id", how="left")
    s = s.dropna(subset=["kickoff"]).sort_values(["pfr_player_id", "kickoff"])
    s["share_to_date"] = s.groupby("pfr_player_id")["role_share"].transform(
        lambda x: x.expanding().mean()
    )
    s = s.merge(
        players[["gsis_id", "pfr_id"]].rename(columns={"pfr_id": "pfr_player_id"}),
        on="pfr_player_id",
        how="left",
    )
    return (
        s.dropna(subset=["gsis_id"])[["gsis_id", "kickoff", "share_to_date"]]
        .sort_values("kickoff")
        .reset_index(drop=True)
    )


def attach_prior_share(injury_rows, history) -> pd.DataFrame:
    """As-of join: the player's snap share from the most recent game they
    actually played BEFORE this kickoff."""
    left = injury_rows.sort_values("kickoff").reset_index(drop=True)
    joined = pd.merge_asof(
        left,
        history,
        on="kickoff",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=False,
    )
    return joined.rename(columns={"share_to_date": "prior_share"})


def main():
    injuries = pd.read_parquet("data/raw/injuries.parquet")
    snaps = pd.read_parquet("data/raw/snap_counts.parquet")
    players = pd.read_parquet("data/raw/player_ids.parquet")
    schedules = kickoff_times(pd.read_parquet("data/raw/schedules.parquet"))

    team_games = pd.concat(
        [
            schedules[["game_id", "season", "week", "home_team", "kickoff"]].rename(
                columns={"home_team": "team"}
            ),
            schedules[["game_id", "season", "week", "away_team", "kickoff"]].rename(
                columns={"away_team": "team"}
            ),
        ],
        ignore_index=True,
    )
    team_games["team"] = team_games["team"].replace({"OAK": "LV"})

    inj = injuries.copy()
    inj["team"] = inj["team"].replace({"OAK": "LV"})
    inj["date_modified"] = pd.to_datetime(
        inj["date_modified"], utc=True, errors="coerce"
    )
    merged = inj.merge(team_games, on=["season", "week", "team"], how="inner")

    # GUARD 1
    no_ts = merged["date_modified"].isna()
    late = merged["date_modified"] >= merged["kickoff"]
    drop_mask = late & ~no_ts
    print(
        f"Leakage filter: dropped {int(drop_mask.sum())} rows timestamped at/after "
        f"kickoff; kept {int(no_ts.sum())} rows with no timestamp (week-keyed only)"
    )
    merged = merged[~drop_mask]

    # GUARD 2
    history = snap_history(snaps, schedules, players)
    merged = attach_prior_share(merged, history)
    matched = merged["prior_share"].notna().mean()
    print(f"As-of snap-share match rate: {matched:.1%}")
    # No prior snaps (rookie, new signing) -> 0 impact. Conservative: it
    # understates a rookie starter rather than inventing a role for a player
    # never observed on the field.
    merged["prior_share"] = merged["prior_share"].fillna(0.0)

    merged["pos_value"] = (
        merged["position"].map(POSITION_VALUE).fillna(DEFAULT_POSITION_VALUE)
    )
    status_weight = np.where(
        merged["report_status"].isin(UNAVAILABLE),
        1.0,
        np.where(merged["report_status"] == "Questionable", QUESTIONABLE_WEIGHT, 0.0),
    )
    merged["impact"] = merged["pos_value"] * merged["prior_share"] * status_weight
    merged["is_qb_out"] = (
        (merged["position"] == "QB")
        & merged["report_status"].isin(UNAVAILABLE)
        & (merged["prior_share"] >= QB_STARTER_SNAP_THRESHOLD)
    )

    agg = (
        merged.groupby(["game_id", "team"])
        .agg(injury_impact=("impact", "sum"), qb_out=("is_qb_out", "max"))
        .reset_index()
    )
    agg["qb_out"] = agg["qb_out"].astype(float)

    result = team_games[["game_id", "season", "week", "team"]].merge(
        agg, on=["game_id", "team"], how="left"
    )
    result[["injury_impact", "qb_out"]] = result[["injury_impact", "qb_out"]].fillna(
        0.0
    )

    out = Path("data/processed/injury_features.parquet")
    result.to_parquet(out, index=False)
    print(f"Built {len(result)} team-game rows")
    print(
        f"  injury_impact: mean {result.injury_impact.mean():.3f}, "
        f"max {result.injury_impact.max():.3f}"
    )
    print(
        f"  qb_out: {int(result.qb_out.sum())} team-games ({result.qb_out.mean():.1%})"
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
