"""
tests/test_adjusted_ratings_leakage.py

A week's opponent-adjusted ratings must be solved from games strictly before
that week's first kickoff -- never the week's own games or later ones.

This used to fit a hand-filtered frame and then assert that its own filter had
kept two rows, so it could not fail whatever the builder did. It now runs the
builder's real loop (build_ratings) and checks the property by perturbation:
make the LAST week's results absurd, and no rating at any cutoff up to and
including that week may move -- a leak would drag them.

Run: pytest tests/test_adjusted_ratings_leakage.py -v
"""

import numpy as np
import pandas as pd

from src.features.build_adjusted_ratings import build_ratings, fit_ratings

TEAMS = [f"T{i}" for i in range(8)]


def league(n_weeks=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2024-09-08")
    for w in range(1, n_weeks + 1):
        order = list(TEAMS)
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            for team, opp, home in ((a, b, 1), (b, a, 0)):
                rows.append(
                    {
                        "season": 2024,
                        "week": w,
                        "gameday": day,
                        "team": team,
                        "opponent": opp,
                        "is_home": home,
                        "off_epa_per_play": rng.normal(0, 0.1),
                    }
                )
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_no_week_sees_its_own_or_later_results():
    base = league()
    spiked = base.copy()
    last = spiked["week"] == spiked["week"].max()
    spiked.loc[last, "off_epa_per_play"] = 25.0  # impossible, and unmissable
    a = build_ratings(base, halflife_days=119, min_rows=20)
    b = build_ratings(spiked, halflife_days=119, min_rows=20)
    assert (
        len(a) and a["week"].max() == base["week"].max()
    ), "no rating at the last week"
    pd.testing.assert_frame_equal(a, b)


def test_a_week_does_see_every_earlier_week():
    """The mirror property: results from BEFORE the cutoff must move it."""
    base = league()
    spiked = base.copy()
    spiked.loc[spiked["week"] == 3, "off_epa_per_play"] = 25.0
    a = build_ratings(base, halflife_days=119, min_rows=20).set_index(["week", "team"])
    b = build_ratings(spiked, halflife_days=119, min_rows=20).set_index(
        ["week", "team"]
    )
    col = "pregame_adjusted_off_epa"
    assert np.allclose(
        a.loc[a.index.get_level_values(0) <= 3, col],
        b.loc[b.index.get_level_values(0) <= 3, col],
    )
    assert not np.allclose(
        a.loc[a.index.get_level_values(0) == 4, col],
        b.loc[b.index.get_level_values(0) == 4, col],
    )


def test_fit_ratings_returns_every_team():
    g = league(n_weeks=4)
    off, deff = fit_ratings(g, TEAMS, 119, g["gameday"].max() + pd.Timedelta(days=1))
    assert set(off) == set(deff) == set(TEAMS)
