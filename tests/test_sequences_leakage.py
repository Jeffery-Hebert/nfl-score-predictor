"""
Leakage-safety test for RNN sequence construction.
Run: pytest tests/test_sequences_leakage.py -v
"""

import pandas as pd
from src.features.build_sequences import build_team_sequences


def test_first_game_sequence_is_all_zero():
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "team": ["FAKE", "FAKE"],
            "gameday": pd.to_datetime(["2024-09-01", "2024-09-08"]),
            "team_score": [10, 20],
            "opp_score": [7, 14],
            "off_epa_per_play": [0.1, 0.2],
            "def_epa_per_play": [-0.1, -0.2],
            "off_success_rate": [0.4, 0.5],
            "def_success_rate_allowed": [0.4, 0.45],
        }
    )
    sequences = build_team_sequences(df)
    padded, mask = sequences[("g1", "FAKE")]
    assert mask.sum() == 0, "First game should have an all-zero (fully masked) sequence"


def test_second_game_sequence_excludes_own_score():
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "team": ["FAKE", "FAKE"],
            "gameday": pd.to_datetime(["2024-09-01", "2024-09-08"]),
            "team_score": [10, 20],
            "opp_score": [7, 14],
            "off_epa_per_play": [0.1, 0.2],
            "def_epa_per_play": [-0.1, -0.2],
            "off_success_rate": [0.4, 0.5],
            "def_success_rate_allowed": [0.4, 0.45],
        }
    )
    sequences = build_team_sequences(df)
    padded, mask = sequences[("g2", "FAKE")]
    assert (
        mask.sum() == 1
    ), "Second game should have exactly one real prior game in its sequence"
    assert (
        padded[-1][0] == 10
    ), "The one real entry should be game 1's score (10), not game 2's (20)"
