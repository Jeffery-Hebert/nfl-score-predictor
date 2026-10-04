"""
src/validate/betting.py: records, ROI and the against-the-spread interval.

Run: pytest tests/test_betting.py -v
"""

import numpy as np
import pandas as pd

from src.validate.betting import BREAK_EVEN, ats, boot_rate_ci, totals


def games(margins, spreads, preds, totals_=None, lines=None, weeks=None):
    n = len(margins)
    return pd.DataFrame(
        {
            "true_margin": margins,
            "spread_line": spreads,
            "pred_margin": preds,
            "true_total": totals_ or [40] * n,
            "total_line": lines or [44.5] * n,
            "pred_total": [45.0] * n,
            "season": 2025,
            "week": weeks or list(range(1, n + 1)),
        }
    )


def test_a_push_is_neither_a_win_nor_a_loss():
    m = games([3, 7, -2], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0])
    r = ats(m)
    assert (r["wins"], r["losses"], r["pushes"], r["n"]) == (1, 1, 1, 2)


def test_roi_at_minus_110():
    m = games([7, -7], [3.0, 3.0], [5.0, 5.0])  # one win, one loss
    assert ats(m)["roi"] == (100 / 110 - 1) / 2
    assert BREAK_EVEN == 110 / 210


def test_totals_pick_the_side_of_the_line():
    m = games([0, 0], [0.0, 0.0], [0.0, 0.0], totals_=[50, 40], lines=[44.5, 44.5])
    assert totals(m)["wins"] == 1  # predicted 45 > 44.5: Over; 50 won, 40 lost


def test_the_interval_resamples_weeks_not_games():
    """Ten weeks of ten games, each week all won or all lost: the honest
    interval (weeks) is far wider than one that treats 100 games as
    independent."""
    hit = np.repeat([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], 10).astype(bool)
    blocks = np.repeat(np.arange(10), 10)
    picked = pd.Series(hit)
    landed = pd.Series(np.ones(100, bool))
    push = pd.Series(np.zeros(100, bool))
    lo, hi = boot_rate_ci(picked, landed, push, blocks)
    assert lo < 0.35 and hi > 0.85, (lo, hi)
    naive_lo, naive_hi = boot_rate_ci(picked, landed, push, np.arange(100))
    assert hi - lo > 1.5 * (naive_hi - naive_lo)
