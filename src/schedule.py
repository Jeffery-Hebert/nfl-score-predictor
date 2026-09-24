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
