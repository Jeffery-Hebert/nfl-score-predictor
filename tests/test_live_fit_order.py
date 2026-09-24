"""
Row-order invariance for every estimator that picks a hyperparameter with
TimeSeriesSplit.

The bug this pins. RidgeCV(cv=TimeSeriesSplit(5)) treats row position as time.
The walk-forward harness sorted its rows by date before fitting, but the live
path in predict_week.py handed over model_table.parquet as stored -- grouped by
HOME TEAM. On the real 2026 Week 2 fit that made RidgeCV pick alpha 562 where the
same games in date order pick 100, and Linear's live forecasts moved by up to
0.7 points. The model the backtest validated was not the model that predicted.

The fix sorts inside the fit functions (src/models/common.py::chronological),
so these tests hand them deliberately scrambled rows and demand the exact
answer the chronological rows give.

Run: pytest tests/test_live_fit_order.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models import linear
from src.models.common import FEATURE_COLS, chronological
from src.models.stacking import fit_stack, predict_stack

TEAMS = [f"T{i:02d}" for i in range(32)]


def synthetic_table(n_weeks=40, per_week=16, seed=2) -> pd.DataFrame:
    """A model table shaped like the real problem: weak signal, noise with a
    ~9-point sd, and a scoring environment that DRIFTS over time. Under those
    conditions folds cut by row position see different data from folds cut by
    time, which is what made the real bug change alpha.

    seed=2 is pinned because it is a seed where the two orders demonstrably
    pick different alphas (1778 vs 3162); the guard test below fails loudly if
    a library change ever makes this fixture order-insensitive.
    """
    rng = np.random.default_rng(seed)
    n = n_weeks * per_week
    X = rng.normal(size=(n, len(FEATURE_COLS)))
    beta = rng.normal(scale=0.15, size=len(FEATURE_COLS))
    week_index = np.repeat(np.arange(n_weeks), per_week)
    drift = 0.05 * week_index  # scoring environment moves over time
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["game_id"] = [f"g{i:05d}" for i in range(n)]
    df["gameday"] = pd.Timestamp("2021-09-12") + pd.to_timedelta(week_index * 7, "D")
    df["home_team"] = rng.choice(TEAMS, n)
    df["home_score"] = 22 + X @ beta + drift + rng.normal(0, 9, n)
    df["away_score"] = 20 - 0.5 * (X @ beta) + drift + rng.normal(0, 9, n)
    df["went_to_ot"] = (rng.random(n) < 0.05).astype(int)
    return df


def team_ordered(df: pd.DataFrame) -> pd.DataFrame:
    """The order model_table.parquet is actually stored in."""
    return df.sort_values(["home_team", "gameday"], kind="mergesort")


@pytest.fixture(scope="module")
def table():
    return synthetic_table()


def test_chronological_is_a_total_order():
    df = pd.DataFrame(
        {
            "game_id": ["c", "a", "b", "d"],
            "gameday": pd.to_datetime(
                ["2024-09-08", "2024-09-08", "2024-09-08", "2024-09-05"]
            ),
        }
    )
    got = chronological(df)["game_id"].tolist()
    assert got == ["d", "a", "b", "c"], "same-day games must fall back to game_id"
    # ...and it must not depend on the order it was handed.
    assert chronological(df.iloc[::-1])["game_id"].tolist() == got


def test_the_fixture_is_order_sensitive_without_the_fix(table):
    """Guards against a vacuous pass: fed straight to the estimator, the two
    orders must disagree, or the invariance test below proves nothing."""
    Xc = chronological(table)
    Xt = team_ordered(table)
    a_c = linear._make_model().fit(Xc[FEATURE_COLS], Xc["home_score"])
    a_t = linear._make_model().fit(Xt[FEATURE_COLS], Xt["home_score"])
    assert a_c[-1].alpha_ != a_t[-1].alpha_, (
        "the fixture no longer reproduces the bug -- strengthen the drift so "
        "position-based folds differ from time-based ones"
    )


def test_linear_fit_is_invariant_to_row_order(table):
    """THE property. Team-ordered, shuffled and chronological inputs must give
    the identical fitted model and identical forecasts."""
    test = table.sample(40, random_state=1)
    ref = linear.fit_linear(chronological(table))
    ref_h, ref_a = linear.predict_linear(ref, test)
    for label, frame in (
        ("team-ordered", team_ordered(table)),
        ("shuffled", table.sample(frac=1.0, random_state=7)),
    ):
        m = linear.fit_linear(frame)
        assert m["home_model"][-1].alpha_ == ref["home_model"][-1].alpha_, label
        assert m["away_model"][-1].alpha_ == ref["away_model"][-1].alpha_, label
        h, a = linear.predict_linear(m, test)
        np.testing.assert_allclose(h, ref_h, rtol=1e-10, err_msg=label)
        np.testing.assert_allclose(a, ref_a, rtol=1e-10, err_msg=label)


def test_linear_drift_offset_is_invariant_to_row_order(table):
    """The drift correction reads the most recent 285 games, so it is also a
    function of order if anything upstream is."""
    ref = linear.fit_linear(chronological(table))
    m = linear.fit_linear(team_ordered(table))
    assert m["off_h"] == pytest.approx(ref["off_h"], rel=1e-10)
    assert m["off_a"] == pytest.approx(ref["off_a"], rel=1e-10)


def test_stack_fit_is_invariant_to_row_order(table):
    rng = np.random.default_rng(3)
    meta = table[["game_id", "gameday", "home_team", "home_score", "away_score"]].copy()
    for name in ("m1", "m2", "m3"):
        meta[f"{name}_home_pred"] = meta["home_score"] + rng.normal(0, 8, len(meta))
        meta[f"{name}_away_pred"] = meta["away_score"] + rng.normal(0, 8, len(meta))
    test = meta.sample(30, random_state=2)
    ref_h, ref_a = predict_stack(fit_stack(chronological(meta)), test)
    h, a = predict_stack(fit_stack(team_ordered(meta)), test)
    np.testing.assert_allclose(h, ref_h, rtol=1e-10)
    np.testing.assert_allclose(a, ref_a, rtol=1e-10)
