"""
tests/test_adjusted_ratings_leakage.py
Confirms a given week's ratings are computed using only games strictly
before that week -- never the week's own games or future ones.

pytest tests/test_adjusted_ratings_leakage.py -v
"""

import pandas as pd
from src.features.build_adjusted_ratings import fit_ratings


def test_ratings_exclude_current_and_future_games():
    teams = ["AAA", "BBB", "CCC"]
    games = pd.DataFrame(
        {
            "team": ["AAA", "BBB", "AAA"],
            "opponent": ["BBB", "AAA", "CCC"],
            "is_home": [1, 0, 1],
            "off_epa_per_play": [0.30, -0.10, 0.05],
            "gameday": pd.to_datetime(["2023-09-10", "2023-09-10", "2023-09-17"]),
        }
    )
    cutoff = pd.Timestamp("2023-09-17")
    train_games = games[games["gameday"] < cutoff]
    off_ratings, def_ratings = fit_ratings(
        train_games, teams, halflife_days=119, cutoff_date=cutoff
    )

    assert (
        len(train_games) == 2
    ), "Week 2's game leaked into the training set used for its own cutoff"
    assert "AAA" in off_ratings and "CCC" in off_ratings
