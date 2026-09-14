"""
tests/test_qb_rolling_features.py
Confirms QB pregame features use only that QB's prior starts, and that a
QB's OWN game stats never leak into that same game's pregame feature.
"""

import pandas as pd
from src.features.build_qb_rolling_features import add_pregame_qb_features


def test_qb_feature_excludes_own_game():
    df = pd.DataFrame(
        {
            "passer_id": ["QB1"] * 3,
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "gameday": pd.to_datetime(["2023-09-10", "2023-09-17", "2023-09-24"]),
            "qb_epa_per_play": [0.05, 0.40, 0.10],
            "qb_completion_pct": [0.6, 0.9, 0.6],
            "qb_success_rate": [0.4, 0.8, 0.4],
        }
    )
    result = add_pregame_qb_features(df, halflife_days=119)
    week1_feature = result.loc[result["week"] == 1, "pregame_qb_epa_per_play"].values[0]
    assert pd.isna(
        week1_feature
    ), "QB's first game should have no prior data (NaN expected)"
    week2_feature = result.loc[result["week"] == 2, "pregame_qb_epa_per_play"].values[0]
    assert (
        week2_feature == 0.05
    ), "Week 2 feature must equal week 1's actual result, not week 2's own"
