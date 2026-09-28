"""
The published-forecast report (src/validate/live_report.py): closing-line
value's sign conventions and the per-model summary, on hand-built rows.

Run: pytest tests/test_live_report.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.validate.live_report import spread_clv, summarize, total_clv


class TestClosingLineValue:
    def test_getting_more_points_than_the_close_is_positive(self):
        # ATL +5.5 when forecast; GB closed -4.5 (spread_line 4.5 = home margin)
        assert spread_clv("ATL", 5.5, "GB", 4.5) == 1.0

    def test_the_favourite_side(self):
        # GB -5.5 when forecast (line -5.5 from GB's side); closed GB -6.5
        assert spread_clv("GB", -5.5, "GB", 6.5) == 1.0
        # closed GB -4.5: laying 5.5 was a point worse than the close
        assert spread_clv("GB", -5.5, "GB", 4.5) == -1.0

    def test_an_away_favourite(self):
        # CAR -2.5 when forecast; the home spread closed at -3.5 (CAR by 3.5)
        assert spread_clv("CAR", -2.5, "CLE", -3.5) == 1.0

    @pytest.mark.parametrize(
        "side, at, close, want",
        [
            ("Over", 43.5, 42.5, -1.0),
            ("Over", 43.5, 45.0, 1.5),
            ("Under", 43.5, 42.5, 1.0),
        ],
    )
    def test_totals(self, side, at, close, want):
        assert total_clv(side, at, close) == want

    def test_no_movement_is_exactly_zero(self):
        assert spread_clv("ATL", 5.5, "GB", 5.5) == 0.0
        assert total_clv("Under", 41.0, 41.0) == 0.0


def test_summarize_counts_what_it_says():
    rows = pd.DataFrame(
        {
            "pre_kickoff": [True, True, False],
            "home_pred": [24.6, 20.0, 27.0],
            "away_pred": [21.0, 23.0, 20.0],
            "home_score": [14.0, 17.0, 24.0],
            "away_score": [35.0, 20.0, 21.0],
            "line_spread": [5.5, -1.5, 3.0],
            "line_total": [43.5, 41.0, 45.0],
            "close_spread": [4.5, -2.5, 3.0],
            "close_total": [42.5, 41.0, 45.0],
            "spread_result": ["win", "loss", "push"],
            "total_result": ["win", "loss", "loss"],
            "spread_result_close": ["win", "loss", "push"],
            "total_result_close": ["win", "loss", "loss"],
            "spread_clv": [1.0, -1.0, np.nan],
            "total_clv": [-1.0, 0.0, np.nan],
        }
    )
    s = summarize(rows)
    assert s["games"] == 3 and s["pre_kickoff"] == 2
    assert s["ats"] == "1-1-1" and s["ats_pct"] == 0.5
    assert s["ou"] == "1-2" and s["ou_pct"] == pytest.approx(1 / 3)
    assert s["spread_vs_close"] == "1/0/1" and s["total_vs_close"] == "0/1/1"
    assert s["spread_clv"] == 0.0 and s["total_clv"] == -0.5
    # winners: GB lost (pred GB), CHI/home lost 17-20 (pred away), home won (pred home)
    assert s["winners"] == "2-1"
