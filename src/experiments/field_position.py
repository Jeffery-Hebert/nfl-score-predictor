"""
src/experiments/field_position.py

Starting field position: extraction, the expected-points curve, and pregame
features.

EXPERIMENTAL. Writes nothing; production is untouched.

--------------------------------------------------------------------- why

`drive_start_yard_line` is 98.7% populated in play-by-play and no model in this
project reads it. The drive model in particular converts drive OUTCOMES to
points while ignoring where those drives began, which is the single largest
determinant of whether a drive scores.

I initially screened this out on a persistence argument and reported it as
"~0.2 points of signal". That number was wrong and the reasoning conflated three
separate questions. They are separated here and each is measured:

  1. DOES FIELD POSITION MATTER PER DRIVE?  Enormously, and it is not close.
     Expected points by starting position, 42,917 drives:

         opp 1-20    4.70 EP    55% touchdown rate
         opp 21-35   3.77
         opp 36-50   2.68
         own 49-35   2.38
         own 34-20   1.89
         own 19-1    1.46 EP    17% touchdown rate

     A 3.23-point spread across those fixed-width buckets -- though note the top
     one holds only 730 of 42,917 drives. Across DECILES of the actual
     distribution the spread is 1.96 points per drive, which is the more
     representative figure and the one the tests assert. Measured WITHIN
     team-season, so team quality cannot confound it, the slope is -0.0355
     points per drive per yard.

  2. DO TEAMS DIFFER?  Yes, meaningfully. Team-season averages run from starting
     at their own 33.3 (BUF 2024) to their own 25.6 (CAR 2023) -- a 7.7-yard
     spread, worth about 3 points a game at the within-team slope.

  3. IS IT KNOWABLE IN ADVANCE?  This is where it collapses, and it is the only
     one of the three that matters for a forecast.

         prior FP -> future FP        r = +0.108
         prior FP -> future points    r = -0.112
         prior TD rate -> future pts  r = +0.262

     Field position barely predicts itself. It is produced by opponent punting,
     turnovers, touchbacks and penalties -- largely luck and opponent, not a team
     trait. Incremental R^2 over prior touchdown rate, out of sample: +0.0014,
     against +0.0308 for the same comparison measured contemporaneously.

So (1) and (2) are large and real, and (3) is why it still may not help. This
module exists so the question is settled by the model rather than by a
regression, and so the three claims stay separated if anyone revisits it.

--------------------------------------------------------------------- extraction

The trap, which produces a plausible and entirely wrong curve: take the first
play of each drive and you pick up KICKOFFS at the 35, which piles half of all
drives into one bucket. The first SCRIMMAGE play is the drive's real start, and
`down` being populated is what distinguishes one.

Verified against nflverse's own `drive_start_yard_line` string, 100% agreement.
tests/test_field_position.py asserts that agreement on real data rather than
trusting it.

Units: `yardline_100` is yards to the OPPONENT's end zone, so LOWER IS BETTER --
own 25 is 75, opponent 10 is 10. Kept in that orientation because it is what
play-by-play uses; flipping it here while leaving the name alone is how sign
bugs are born.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PBP = "data/raw/pbp.parquet"
SCHEDULES = "data/raw/schedules.parquet"

# Points to the team with the ball. Matches drive_model_v2.OFFENSE_POINTS.
DRIVE_POINTS = {"Touchdown": 6.95, "Field goal": 3.0}

FP_COLS = ["pregame_own_fp", "pregame_fp_allowed"]

HALFLIFE_DAYS = 17 * 7


def extract_drive_starts(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per drive: who had it, where it started, how it ended.

    Takes the first SCRIMMAGE play, identified by `down` being populated.
    Kickoffs, punts and extra points carry no down, and including them puts the
    drive start at the kicking spot rather than where the offence took over.
    """
    if pbp is None:
        pbp = pd.read_parquet(
            PBP,
            columns=[
                "game_id",
                "play_id",
                "drive",
                "posteam",
                "defteam",
                "down",
                "yardline_100",
                "fixed_drive_result",
            ],
        )
    scrim = pbp[
        pbp["down"].notna()
        & pbp["yardline_100"].notna()
        & pbp["posteam"].notna()
        & pbp["drive"].notna()
    ].sort_values("play_id")

    out = (
        scrim.groupby(["game_id", "posteam", "defteam", "drive"])
        .agg(
            start_100=("yardline_100", "first"),
            result=("fixed_drive_result", "first"),
        )
        .reset_index()
    )
    out["points"] = out["result"].map(DRIVE_POINTS).fillna(0.0)
    return out


def expected_points_curve(drives: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Mean points by starting-position bucket.

    Descriptive, for validating that the extraction is sane: the curve must be
    monotone in field position. A non-monotone curve means the extraction is
    picking up the wrong play.
    """
    d = drives.dropna(subset=["start_100"]).copy()
    d["bucket"] = pd.qcut(d["start_100"], n_bins, duplicates="drop")
    return (
        d.groupby("bucket", observed=True)
        .agg(
            drives=("points", "size"),
            mean_start=("start_100", "mean"),
            ep=("points", "mean"),
            td_rate=("result", lambda x: (x == "Touchdown").mean()),
        )
        .reset_index(drop=True)
    )


def within_team_slope(drives: pd.DataFrame, seasons: pd.DataFrame) -> float:
    """Points per drive per yard of field position, team quality removed.

    The across-team slope is confounded -- good teams have both good field
    position and good offences, so regressing one on the other credits field
    position with the whole of team quality. De-meaning by team-season leaves
    only the within-team effect, which is the causal one.
    """
    d = drives.merge(seasons, on="game_id", how="left")
    d["key"] = d["season"].astype(str) + d["posteam"]
    fp = d["start_100"] - d.groupby("key")["start_100"].transform("mean")
    pts = d["points"] - d.groupby("key")["points"].transform("mean")
    ok = fp.notna() & pts.notna()
    return float(np.polyfit(fp[ok], pts[ok], 1)[0])


def team_game_field_position(drives: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: where this team started, and where it let opponents start.

    `fp_allowed` is the defensive/special-teams side of the same coin -- a team
    that punts well and does not turn the ball over hands its opponent worse
    field position.
    """
    own = (
        drives.groupby(["game_id", "posteam"])
        .agg(own_fp=("start_100", "mean"), own_drives=("points", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    allowed = (
        drives.groupby(["game_id", "defteam"])
        .agg(fp_allowed=("start_100", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return own.merge(allowed, on=["game_id", "team"], how="outer")


def build_pregame_field_position(
    drives: pd.DataFrame | None = None, halflife_days: float = HALFLIFE_DAYS
) -> pd.DataFrame:
    """Pregame field-position features, one row per (game_id, team).

    Recency-weighted over STRICTLY PRIOR games, the same EWM + shift(1) pattern
    as every other rolling feature in the project. Week 1 of a team's first
    season has no prior games and gets NaN rather than a fabricated default.
    """
    if drives is None:
        drives = extract_drive_starts()
    sched = pd.read_parquet(SCHEDULES)[["game_id", "season", "gameday"]]
    sched["gameday"] = pd.to_datetime(sched["gameday"])

    tg = team_game_field_position(drives).merge(sched, on="game_id", how="inner")
    tg = tg.sort_values("gameday").reset_index(drop=True)

    pieces = []
    for _, g in tg.groupby("team"):
        g = g.sort_values("gameday").reset_index(drop=True)
        times = pd.to_datetime(g["gameday"])
        for src, dst in (
            ("own_fp", "pregame_own_fp"),
            ("fp_allowed", "pregame_fp_allowed"),
        ):
            g[dst] = (
                g[src]
                .ewm(halflife=pd.Timedelta(days=halflife_days), times=times)
                .mean()
                .shift(1)
            )
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    return out[["game_id", "team", "season", "gameday"] + FP_COLS]


def add_field_position(df: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """Join pregame field position onto a game table as home_/away_ columns."""
    out = df.copy()
    for side in ("home", "away"):
        cols = fp[["game_id", "team"] + FP_COLS].rename(
            columns={
                **{c: f"{side}_{c}" for c in FP_COLS},
                "team": f"{side}_team",
            }
        )
        out = out.merge(cols, on=["game_id", f"{side}_team"], how="left")
    return out


def main():
    drives = extract_drive_starts()
    sched = pd.read_parquet(SCHEDULES)[["game_id", "season"]]
    print(f"Extracted {len(drives)} drives\n")

    print("EXPECTED POINTS BY STARTING FIELD POSITION (must be monotone):")
    curve = expected_points_curve(drives)
    for _, r in curve.iterrows():
        own = 100 - r["mean_start"]
        print(
            f"   start ~own {own:5.1f}   {r['drives']:6.0f} drives   "
            f"{r['ep']:.3f} EP   {r['td_rate']:.1%} TD"
        )
    monotone = curve["ep"].is_monotonic_decreasing
    print(f"   monotone in yards-to-goal: {monotone}")
    print(f"   spread: {curve.ep.max() - curve.ep.min():.2f} points per drive")

    slope = within_team_slope(drives, sched)
    print(
        f"\nWithin-team slope: {slope:+.4f} points/drive/yard "
        f"({abs(slope) * 10:.2f} per 10 yards)"
    )

    fp = build_pregame_field_position(drives)
    print(f"\nPregame features: {len(fp)} team-game rows")
    for c in FP_COLS:
        v = fp[c].dropna()
        print(
            f"   {c:20s} mean {v.mean():.2f}  sd {v.std():.2f}  null {fp[c].isna().mean():.1%}"
        )


if __name__ == "__main__":
    main()
