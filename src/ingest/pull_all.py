"""
Pull every raw dataset from nflverse, then check that what arrived is complete,
current and self-consistent before anything is built on it.

Why the checks. nflverse publishes schedules, play-by-play, snap counts and
injuries as separate releases that update on different clocks. A pull that
lands between them is internally inconsistent in ways that raise no error: a
game can have a final score in the schedule and no plays yet, and then its
EPA is simply missing -- the rolling features skip it without complaint and the
next week's forecast is built as if the game never happened. These checks turn
that silent gap into a failure with the fix attached.

  HARD failures (exit 1 -- do not build on this data):
    - a game that kicked off over a day ago has no final score
    - a scored game has no play-by-play
    - play-by-play's final score disagrees with the schedule's
  WARNINGS (reported, recorded, not fatal):
    - a scored game this season has no snap counts (injury weights lag a game)
    - the upcoming week has no injury-report rows yet

Everything is recorded in data/raw/_pull_manifest.json with pull times and
hashes (see src/ingest/manifest.py).

Run: python -m src.ingest.pull_all
     python -m src.ingest.pull_all --check-only    # validate what is on disk
"""

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from src.ingest import manifest

# Games that were scheduled, kicked off and will never have a final score.
KNOWN_UNFINISHED = {"2022_17_BUF_CIN"}  # suspended after Damar Hamlin's collapse
GRACE = pd.Timedelta(days=1)  # nflverse posts finals within hours; allow a day


def check(
    sched: pd.DataFrame,
    pbp_finals: pd.DataFrame,
    snap_game_ids: set,
    injuries: pd.DataFrame,
    now: pd.Timestamp,
) -> dict:
    """The checks themselves, on frames -- so they can be tested without files.

    sched        schedules with a UTC `kickoff` column
    pbp_finals   index game_id; total_home_score / total_away_score at the end
    """
    scored = sched[sched["home_score"].notna()]
    errors, warnings, info = [], [], {}

    # 1. every game that has been played has a result
    overdue = sched[
        sched["home_score"].isna()
        & (sched["kickoff"] < now - GRACE)
        & ~sched["game_id"].isin(KNOWN_UNFINISHED)
    ]
    if len(overdue):
        errors.append(
            f"{len(overdue)} games kicked off over a day ago with no final score "
            f"(e.g. {', '.join(overdue['game_id'].head(4))}) -- nflverse schedules "
            "have not updated; pull again later"
        )

    # 2. + 3. play-by-play covers every scored game and agrees on the score
    no_plays = sorted(set(scored["game_id"]) - set(pbp_finals.index))
    if no_plays:
        errors.append(
            f"{len(no_plays)} scored games have no play-by-play "
            f"(e.g. {', '.join(no_plays[:4])}) -- their EPA would silently vanish "
            "from every rolling feature; pull again when nflverse catches up"
        )
    both = scored.set_index("game_id").join(pbp_finals, how="inner")
    mismatch = both[
        (both["total_home_score"] != both["home_score"])
        | (both["total_away_score"] != both["away_score"])
    ]
    if len(mismatch):
        errors.append(
            f"{len(mismatch)} games where play-by-play's final score disagrees "
            f"with the schedule (e.g. {', '.join(mismatch.index[:4])})"
        )
    info["games_cross_checked"] = int(len(both))

    # 4. snap counts for this season's scored games
    season = int(sched["season"].max())
    this_season = scored[scored["season"] == season]
    no_snaps = sorted(set(this_season["game_id"]) - set(snap_game_ids))
    if no_snaps:
        warnings.append(
            f"{len(no_snaps)} {season} games have no snap counts yet "
            f"(e.g. {', '.join(no_snaps[:4])}) -- injury weights will use a "
            "player's role from before those games"
        )

    # 5. the upcoming week's injury report
    upcoming = sched[sched["home_score"].isna() & (sched["kickoff"] > now)]
    if len(upcoming):
        nxt = upcoming.sort_values("kickoff").iloc[0]
        wk = injuries[
            (injuries["season"] == nxt["season"]) & (injuries["week"] == nxt["week"])
        ]
        info["next_week"] = f"{int(nxt['season'])} week {int(nxt['week'])}"
        info["next_week_injury_rows"] = int(len(wk))
        info["next_week_injury_statuses"] = int(wk["report_status"].notna().sum())
        if not len(wk):
            warnings.append(
                f"no injury-report rows yet for {info['next_week']} -- normal early "
                "in the week; the final statuses arrive shortly before each game"
            )

    info["latest_final"] = str(scored["gameday"].max()) if len(scored) else None
    return {"errors": errors, "warnings": warnings, "info": info}


def validate(now: pd.Timestamp | None = None) -> dict:
    """Load what is on disk, run check(), and record the result."""
    from src.schedule import kickoff_utc

    now = now or pd.Timestamp.now(tz="UTC")
    sched = pd.read_parquet("data/raw/schedules.parquet")
    sched["kickoff"] = kickoff_utc(sched)
    pbp = pd.read_parquet(
        "data/raw/pbp.parquet",
        columns=["game_id", "total_home_score", "total_away_score"],
    )
    finals = pbp.groupby("game_id")[["total_home_score", "total_away_score"]].max()
    snaps = pd.read_parquet("data/raw/snap_counts.parquet", columns=["game_id"])
    inj = pd.read_parquet(
        "data/raw/injuries.parquet", columns=["season", "week", "report_status"]
    )
    result = check(sched, finals, set(snaps["game_id"]), inj, now)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    manifest.record_validation(result)
    return result


def report(result: dict) -> int:
    info = result["info"]
    print(f"\nLatest final score: {info.get('latest_final')}")
    print(
        f"Final scores cross-checked against play-by-play: {info.get('games_cross_checked')}"
    )
    if "next_week" in info:
        print(
            f"Next up: {info['next_week']} -- {info['next_week_injury_rows']} injury "
            f"rows, {info['next_week_injury_statuses']} with a game status"
        )
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    for e in result["errors"]:
        print(f"ERROR: {e}")
    if result["errors"]:
        print("\nRaw data is NOT safe to build on. Nothing downstream was changed.")
        return 1
    print("\nRaw data complete and consistent.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.check_only:
        from src.ingest import pull_injuries, pull_pbp, pull_schedules

        for mod in (pull_schedules, pull_injuries, pull_pbp):
            print(f"--- {mod.__name__.rsplit('.', 1)[-1]}")
            mod.main()
    return report(validate())


if __name__ == "__main__":
    sys.exit(main())
