"""
Which seasons to pull. One definition, used by every ingest script.

SEASON_END used to be hard-coded as 2026 in three separate files, with a comment
asking whoever touched it next to "extend each year" -- and the schema tests
asserted `season <= 2026`, so they would have failed the first time anyone did.

The newest season is included from MARCH 15, when nflverse rolls its rosters
over and the new schedule is on its way. That is deliberately earlier than
nflreadpy's get_current_season(), which only flips on the Thursday after Labor
Day: the 2026 season opened on a WEDNESDAY (Sep 9), so that rule would have left
the new season out of the very pull meant to forecast its first game. Before
the new season's data exists, pulls fall back one season (load_seasons below).
"""

from datetime import date

SEASON_START = 2019
ROLLOVER = (3, 15)  # month, day


def current_season(today: date | None = None) -> int:
    """The newest season whose data may exist: this year from March 15 on."""
    today = today or date.today()
    return today.year if (today.month, today.day) >= ROLLOVER else today.year - 1


def seasons(today: date | None = None) -> list[int]:
    return list(range(SEASON_START, current_season(today) + 1))


# How long into its season the newest season's data may still be missing. Snap
# counts and play-by-play arrive after the games, so the days around the opener
# can legitimately ask for a file nflverse has not published yet; a week later
# they cannot.
UNDERWAY_GRACE_DAYS = 7
SCHEDULES = "data/raw/schedules.parquet"


def season_underway(season: int, now=None, schedules_path=SCHEDULES) -> bool:
    """True once `season`'s first game kicked off more than UNDERWAY_GRACE_DAYS
    ago, by the schedule on disk (pull_all pulls schedules first). No schedule,
    or none of that season in it: False -- nothing says the season has begun."""
    from pathlib import Path

    import pandas as pd

    from src.schedule import kickoff_utc

    if not Path(schedules_path).exists():
        return False
    sched = pd.read_parquet(schedules_path)
    sched = sched[sched["season"] == season]
    if sched.empty:
        return False
    first = kickoff_utc(sched).min()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return bool(
        pd.notna(first) and now - first > pd.Timedelta(days=UNDERWAY_GRACE_DAYS)
    )


def load_seasons(loader, seasons_wanted: list[int], underway: bool | None = None):
    """Call an nflreadpy loader, tolerating a newest season with no data yet.

    Between March 15 and a season's first games, nflverse has a schedule for
    the new season but no play-by-play, injuries or snap counts, and a loader
    asked for it raises. That is expected, so the pull retries without the
    newest season -- saying so -- rather than failing the whole weekly run.

    Only while that season has not begun. Until 2026-10-04 this fell back on ANY
    error whenever the newest season was this calendar year -- so in October a
    network failure loading the 2026 snap counts would quietly have written a
    file with no 2026 rows, which pull_all only WARNS about, and every injury
    weight would have been read from last season. Once the newest season has
    been underway for a week (season_underway; `underway` overrides, for tests),
    a failure loading it raises like any other.
    """
    try:
        return loader(seasons=seasons_wanted)
    except Exception as err:
        newest = seasons_wanted[-1]
        if len(seasons_wanted) < 2 or newest < date.today().year:
            raise
        if underway is None:
            underway = season_underway(newest)
        if underway:
            raise
        print(
            f"NOTE: no data published for {newest} yet ({type(err).__name__}); "
            f"pulling {seasons_wanted[0]}-{seasons_wanted[-2]}"
        )
        return loader(seasons=seasons_wanted[:-1])
