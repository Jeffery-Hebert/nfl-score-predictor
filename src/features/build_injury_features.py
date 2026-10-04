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

UNMEASURABLE IS MISSING, NOT ZERO -- fixed 2026-10-04:

  A played team-game used to get injury_impact = 0.0 -- "nobody of note was
  out" -- in two situations where the data says nothing of the kind:

    - No report published before kickoff is in the data for that team that
      week. nflverse has none at all for the 14 team-games of the 2023
      postseason after the wild card round and for a handful of other
      team-weeks; for Kansas City's rescheduled 2020 week 6 game, every row is
      timestamped after kickoff, so guard 1 rightly drops them all.
    - The team has no earlier game in the snap counts, so no player's role can
      be read: every team's first game in the data (2019 week 1).

  Those are now left MISSING (NaN), which the models replace with the training
  mean -- "unknown", not "healthy". See unmeasurable(). A game not yet played
  is different: until its final report is out it has no report rows by design,
  it keeps 0.0, and predict_week holds it back until the report is in
  (src/predict/injury_readiness.py).

ONE ROW PER PLAYER PER GAME. nflverse carries a few players twice in one week
(2024 week 15: Cade Stover and Tyler Conklin, each listed Questionable and then
Out on game day), and summing both rows counted them 1.5 times. The latest
report stands.

Positional values follow the standard positional-value ordering in football
analytics: quarterback dominates, then the premium positions that are hardest
to replace (tackle, edge, corner, receiver), then interior and off-ball roles,
with running back low as the most replaceable starting position. These are
deliberately coarse judgement calls, not fitted parameters.

They are keyed on the positions nflverse's injury reports actually use, which
are coarse: there is no EDGE or OLB, so a 3-4 outside linebacker who rushes the
passer is "LB" and carries the linebacker value, not the edge value DE gets.
The table used to list fourteen codes that never occur (EDGE, OLB, OT, LT, RT,
NT, SS, FS, OG, OL, ILB, MLB, DB, DL), which read as if those roles were valued
separately; they were never used, so removing them changed no value. Kickers,
punters and long snappers keep small values but barely register: a player's
role is his share of OFFENSIVE or DEFENSIVE snaps (snap_history), and theirs is
at most a few percent from fakes -- special-teams snaps are not counted.

Run: python -m src.features.build_injury_features
Output: data/processed/injury_features.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Snap counts and injury reports carry the code of the season (OAK in 2019);
# team-games carry today's.
TEAM_CODE_MAP = {"OAK": "LV"}

UNAVAILABLE = {"Out", "Doubtful"}
QUESTIONABLE_WEIGHT = 0.5

POSITION_VALUE = {
    "QB": 1.00,
    "T": 0.40,
    "DE": 0.38,
    "CB": 0.34,
    "WR": 0.32,
    "DT": 0.28,
    "S": 0.26,
    "TE": 0.24,
    "G": 0.22,
    "C": 0.22,
    "LB": 0.22,
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
    """As-of join: the player's role as it stood BEFORE this kickoff -- the
    share_to_date of the most recent game he played, i.e. his mean snap share
    over every game he has played in the data up to then (snap_history). It is a
    career-to-date average, not his share in that last game alone. (This
    docstring said "snap share from the most recent game" until 2026-10-04,
    which is not what the value is.)"""
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


def unmeasurable(team_games, reported, snaps) -> np.ndarray:
    """Played team-games whose injury_impact the data cannot measure.

    team_games  game_id, team, kickoff, played (has a final score)
    reported    game_id, team: team-games with at least one pre-game report row
    snaps       team, kickoff: one row per player-game of snap counts, team
                codes as nflverse publishes them (OAK is normalised here, or a
                relocated team's history would look empty)

    True where the game has been played AND either no report for that team is
    in the data, or the team has no game in the snap history before this
    kickoff (its first game in the data -- no player's role is known yet).
    """
    key = pd.MultiIndex.from_frame(team_games[["game_id", "team"]])
    has_report = key.isin(pd.MultiIndex.from_frame(reported[["game_id", "team"]]))
    snap_team = snaps["team"].replace(TEAM_CODE_MAP)
    first_snap_game = snaps["kickoff"].groupby(snap_team).min()
    has_history = (
        team_games["kickoff"] > team_games["team"].map(first_snap_game)
    ).to_numpy()
    return team_games["played"].to_numpy(bool) & ~(has_report & has_history)


def latest_report_per_player(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team, gsis_id): the latest-modified report, in a
    fixed order so the choice never depends on how the file arrived."""
    ordered = rows.sort_values(
        ["game_id", "team", "gsis_id", "date_modified"],
        na_position="first",
        kind="mergesort",
    )
    return ordered.drop_duplicates(["game_id", "team", "gsis_id"], keep="last")


def team_totals(rows: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: summed impact, and whether a starting QB is out.

    Summed in a FIXED order. groupby().sum() adds a team's players in the order
    the rows arrive, and float addition is not associative, so a pull that
    merely reorders the injury file -- new rows for the current week -- moved
    every historical team-game's total in its last bit. Harmless to a forecast,
    but the content fingerprints then read 73 played games as changed
    (2026-09-27) and every backtest as stale.
    """
    ordered = rows.sort_values(
        ["game_id", "team", "gsis_id", "impact"], kind="mergesort"
    )
    agg = (
        ordered.groupby(["game_id", "team"], sort=True)
        .agg(injury_impact=("impact", "sum"), qb_out=("is_qb_out", "max"))
        .reset_index()
    )
    agg["qb_out"] = agg["qb_out"].astype(float)
    return agg


def main():
    injuries = pd.read_parquet("data/raw/injuries.parquet")
    snaps = pd.read_parquet("data/raw/snap_counts.parquet")
    players = pd.read_parquet("data/raw/player_ids.parquet")
    schedules = kickoff_times(pd.read_parquet("data/raw/schedules.parquet"))

    cols = ["game_id", "season", "week", "kickoff", "home_score"]
    team_games = pd.concat(
        [
            schedules[cols + ["home_team"]].rename(columns={"home_team": "team"}),
            schedules[cols + ["away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    team_games["played"] = team_games.pop("home_score").notna()
    team_games["team"] = team_games["team"].replace(TEAM_CODE_MAP)

    inj = injuries.copy()
    inj["team"] = inj["team"].replace(TEAM_CODE_MAP)
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
    reported = merged[["game_id", "team"]].drop_duplicates()
    n_rows = len(merged)
    merged = latest_report_per_player(merged)
    print(
        f"Duplicate player-weeks collapsed to the latest report: {n_rows - len(merged)}"
    )

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

    agg = team_totals(merged)

    result = team_games[["game_id", "season", "week", "team"]].merge(
        agg, on=["game_id", "team"], how="left"
    )
    result[["injury_impact", "qb_out"]] = result[["injury_impact", "qb_out"]].fillna(
        0.0
    )

    # Played team-games the data cannot measure are MISSING, not healthy.
    snaps_dated = snaps[["game_id", "team"]].merge(
        schedules[["game_id", "kickoff"]], on="game_id", how="left"
    )
    gap = unmeasurable(team_games, reported, snaps_dated.dropna(subset=["kickoff"]))
    assert (result["game_id"].to_numpy() == team_games["game_id"].to_numpy()).all()
    assert (result["team"].to_numpy() == team_games["team"].to_numpy()).all()
    result.loc[gap, ["injury_impact", "qb_out"]] = np.nan
    print(
        f"Unmeasurable played team-games left missing (not 0): {int(gap.sum())} "
        f"of {int(team_games['played'].sum())}"
    )

    out = Path("data/processed/injury_features.parquet")
    result.to_parquet(out, index=False)
    print(f"Built {len(result)} team-game rows")
    print(
        f"  injury_impact: mean {result.injury_impact.mean():.3f} (missing excluded), "
        f"max {result.injury_impact.max():.3f}"
    )
    print(
        f"  qb_out: {int(result.qb_out.sum())} team-games ({result.qb_out.mean():.1%})"
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
