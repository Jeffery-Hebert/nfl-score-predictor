"""
Kickoff instants from the nflverse schedule -- one implementation.

This construction (gameday + gametime, read as US/Eastern, converted to UTC)
used to be copied into build_injury_features.py, predict_week.py and
build_report.py, with a comment in the last one calling three copies two too
many. It lives here, in a module that imports nothing heavier than pandas, so
the one-second report builder can use it without pulling in every model.
"""

import pandas as pd

DEFAULT_KICKOFF_ET = "13:00"  # a missing gametime is treated as a 1pm Sunday slot


def kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    """UTC kickoff for each row of a schedules frame, aligned to its index.

    nflverse `gametime` is Eastern local time. DST-ambiguous or non-existent
    local times (none occur at NFL kickoff hours) become NaT rather than a
    guess.
    """
    ts = pd.to_datetime(
        schedules["gameday"].astype(str)
        + " "
        + schedules["gametime"].fillna(DEFAULT_KICKOFF_ET),
        errors="coerce",
    )
    return ts.dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")


def kickoff_by_game(schedules: pd.DataFrame, game_ids=None) -> pd.Series:
    """Kickoff instant indexed by game_id (optionally restricted to game_ids)."""
    s = (
        schedules
        if game_ids is None
        else schedules[schedules["game_id"].isin(list(game_ids))]
    )
    return pd.Series(kickoff_utc(s).to_numpy(), index=s["game_id"].to_numpy())


def games_near(
    schedules: pd.DataFrame, now: pd.Timestamp, days_before=2, days_after=9
) -> pd.DataFrame:
    """Games kicking off in [now - days_before, now + days_after].

    The scheduled pipeline's "is there anything to do?": a result from the last
    two days to grade and re-benchmark on, or a game in the next nine to
    forecast. Empty from a week after the Super Bowl until the week before the
    next season opens -- nine days covers every gap inside a season, including
    the one before the Super Bowl."""
    k = kickoff_utc(schedules)
    lo = now - pd.Timedelta(days=days_before)
    hi = now + pd.Timedelta(days=days_after)
    return schedules[(k >= lo) & (k <= hi)]
