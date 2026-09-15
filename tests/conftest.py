"""
Shared builders for the synthetic histories used by the leakage gates.

Why this file exists: the leakage tests assert on a hand-built team history
where the correct answer is known in advance. Those fixtures used to hardcode
their own column list, which had to stay in sync with STAT_COLS in
src/features/build_rolling_features.py. Twice it didn't:

  2dc3f67  added season/week (finale-week masking)     -> KeyError: 'season'
  81c48d9  added 6 pass/rush/CPOE columns to STAT_COLS -> KeyError: 'off_pass_epa_per_play'

Both times all six leakage tests errored out and stayed red across several
commits -- the project's most important safety net was silently disabled.

The fix: build fixtures FROM STAT_COLS rather than alongside it. Pass the
columns a test actually asserts on; every other required column is auto-filled
with a placeholder. Adding a new stat to STAT_COLS can no longer break these
tests, and a test that needs the new column will still say so explicitly.
"""

import pandas as pd

from src.features.build_rolling_features import STAT_COLS


def make_team_history(gamedays, seasons, weeks, team="FAKE", **asserted_stats):
    """One fake team's game history, with every column the rolling-feature
    builder requires present.

    gamedays:        list of date strings, chronological, one per game
    seasons/weeks:   list per game -- drives is_finale_week() masking
    asserted_stats:  the stat columns this test actually makes assertions
                     about, e.g. team_score=[10, 20, 30, 40]

    Any column in STAT_COLS not passed explicitly is filled with a simple
    ascending placeholder. Placeholders exist only to satisfy the builder;
    no test should assert on them.
    """
    n = len(gamedays)
    for name, values in asserted_stats.items():
        if name not in STAT_COLS and name != "rest_days":
            raise ValueError(
                f"{name!r} is not in STAT_COLS -- check the name, or add it to "
                f"src/features/build_rolling_features.py first"
            )
        if len(values) != n:
            raise ValueError(f"{name!r} has {len(values)} values, expected {n}")

    df = pd.DataFrame(
        {
            "game_id": [f"g{i + 1}" for i in range(n)],
            "team": [team] * n,
            "season": seasons,
            "week": weeks,
            "gameday": pd.to_datetime(gamedays),
            # C1: rest_days now arrives from schedules via build_team_game_stats
            # and is passed through, not recomputed. 7 = an ordinary week.
            "rest_days": asserted_stats.pop("rest_days", [7] * n),
        }
    )
    for col in STAT_COLS:
        df[col] = asserted_stats.get(col, [0.1 * (i + 1) for i in range(n)])
    return df


def make_drive_history(gamedays, seasons, weeks, team="FAKE", **asserted_stats):
    """One fake team's drive-rate history, for the drive-level leakage gate.

    Same contract as make_team_history but built from DRIVE_STAT_COLS, and it
    includes the season/week columns that add_pregame_rolling_drive_features
    needs for finale masking -- whose absence from the real pipeline made that
    stage crash on every run between commits 2dc3f67 and its repair.
    """
    from src.features.build_drive_rolling_features import DRIVE_STAT_COLS

    n = len(gamedays)
    for name, values in asserted_stats.items():
        if name not in DRIVE_STAT_COLS:
            raise ValueError(f"{name!r} is not in DRIVE_STAT_COLS")
        if len(values) != n:
            raise ValueError(f"{name!r} has {len(values)} values, expected {n}")

    df = pd.DataFrame(
        {
            "game_id": [f"g{i + 1}" for i in range(n)],
            "team": [team] * n,
            "season": seasons,
            "week": weeks,
            "gameday": pd.to_datetime(gamedays),
        }
    )
    for col in DRIVE_STAT_COLS:
        df[col] = asserted_stats.get(col, [0.1 * (i + 1) for i in range(n)])
    return df
