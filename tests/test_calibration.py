"""
B7/B9: tests for the calibration and promotion-gate helpers.

Synthetic data with known answers -- a calibration metric that is silently
wrong is worse than none, because it would be used to justify shipping a model.

Run: pytest tests/test_calibration.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.validate.calibration import (
    bias,
    calibration_slope_intercept,
    implied_interval_coverage,
    promotion_report,
    reliability_table,
    rmse,
)


@pytest.fixture
def truth():
    rng = np.random.default_rng(42)
    return rng.normal(23, 10, 4000)


# ------------------------------------------------------------------- bias


def test_bias_sign_convention():
    actual = np.array([20.0, 20.0, 20.0])
    assert bias(actual, np.array([22.0, 22.0, 22.0])) == pytest.approx(2.0)
    assert bias(actual, np.array([18.0, 18.0, 18.0])) == pytest.approx(-2.0)
    assert bias(actual, actual) == 0.0


def test_rmse_matches_hand_calculation():
    actual = np.array([10.0, 20.0])
    pred = np.array([13.0, 16.0])  # errors +3, -4
    assert rmse(actual, pred) == pytest.approx(np.sqrt((9 + 16) / 2))


# ------------------------------------------------------ calibration slope


def test_perfect_predictions_give_slope_one(truth):
    slope, intercept = calibration_slope_intercept(truth, truth)
    assert slope == pytest.approx(1.0, abs=1e-6)
    assert intercept == pytest.approx(0.0, abs=1e-6)


def test_compressed_predictions_give_slope_above_one(truth):
    """Predictions shrunk toward the mean -- the failure mode that flattens
    every derived spread while barely moving RMSE."""
    pred = truth.mean() + 0.5 * (truth - truth.mean())
    slope, _ = calibration_slope_intercept(truth, pred)
    assert slope > 1.5, f"compression not detected (slope={slope:.3f})"


def test_over_dispersed_predictions_give_slope_below_one(truth):
    pred = truth.mean() + 2.0 * (truth - truth.mean())
    slope, _ = calibration_slope_intercept(truth, pred)
    assert slope < 0.75, f"over-dispersion not detected (slope={slope:.3f})"


def test_a_constant_bias_moves_the_intercept_not_the_slope(truth):
    slope, intercept = calibration_slope_intercept(truth, truth + 5.0)
    assert slope == pytest.approx(1.0, abs=1e-6)
    assert intercept == pytest.approx(-5.0, abs=1e-6)


def test_slope_is_blind_to_pure_noise_but_rmse_is_not(truth):
    """Calibration and accuracy are different things -- an unbiased, noisy
    model is well calibrated and still inaccurate. Documents the limitation."""
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 8, len(truth))
    slope, _ = calibration_slope_intercept(truth, noisy)
    assert 0.4 < slope < 1.0
    assert rmse(truth, noisy) > 7


# -------------------------------------------------------------- coverage


def test_normal_residuals_cover_close_to_nominal():
    rng = np.random.default_rng(7)
    actual = rng.normal(23, 10, 20000)
    pred = actual + rng.normal(0, 9, 20000)
    cov = implied_interval_coverage(actual, pred)
    for level, got in cov.items():
        assert abs(got - level) < 0.03, f"{level:.0%} interval covered {got:.1%}"


def test_heavy_tails_break_nominal_coverage():
    """Heavy-tailed residuals inflate sigma, so a naive normal interval
    OVER-covers the middle. (This test used to be called
    ..._under_cover_... -- the opposite of what it checks.)"""
    rng = np.random.default_rng(11)
    actual = rng.normal(23, 10, 20000)
    pred = actual + rng.standard_t(df=2, size=20000) * 3
    cov = implied_interval_coverage(actual, pred)
    assert abs(cov[0.50] - 0.50) > 0.05, "heavy tails should show up as mis-coverage"


# ------------------------------------------------------------ reliability


def test_reliability_table_reports_no_gap_for_a_perfect_model(truth):
    tbl = reliability_table(truth, truth, n_bins=5)
    assert len(tbl) == 5
    assert tbl["gap"].abs().max() < 1e-6
    assert tbl["n"].sum() == len(truth)


def test_reliability_table_localises_a_one_sided_error(truth):
    """Inflate only the low end; the low bin should show a positive gap and the
    high bin should not."""
    pred = truth.copy()
    low = pred < np.quantile(pred, 0.2)
    pred[low] += 6.0
    tbl = reliability_table(truth, pred, n_bins=5)
    assert tbl.iloc[0]["gap"] > 1.0
    assert abs(tbl.iloc[-1]["gap"]) < 1.0


# --------------------------------------------------------- B9 promotion


def _frame(n, home_pred, away_pred, home_score=21.0, away_score=20.0):
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "home_score": np.full(n, home_score),
            "away_score": np.full(n, away_score),
            "home_pred": np.full(n, home_pred, dtype=float),
            "away_pred": np.full(n, away_pred, dtype=float),
        }
    )


def test_promotion_flags_better_rmse_bought_with_worse_bias():
    """The B9 case: a candidate wins on RMSE while becoming materially more
    biased. The gate must refuse to call that an improvement."""
    rng = np.random.default_rng(3)
    n = 500
    base = _frame(n, 21.0, 20.0)
    base["home_pred"] = 21.0 + rng.normal(0, 6, n)  # unbiased, noisy
    cand = base.copy()
    cand["home_pred"] = 21.0 + 1.0 + rng.normal(0, 3, n)  # biased, tighter

    rep = promotion_report(base, cand, "cand")
    assert rep["home_rmse_delta"] < 0, "fixture should improve RMSE"
    assert rep["home_absbias_delta"] > 0.25, "fixture should worsen bias"
    assert any("do not promote on RMSE alone" in v for v in rep["verdicts"])


def test_promotion_accepts_a_clean_improvement():
    rng = np.random.default_rng(4)
    n = 500
    base = _frame(n, 21.0, 20.0)
    base["home_pred"] = 21.0 + rng.normal(0, 6, n)
    cand = base.copy()
    cand["home_pred"] = 21.0 + rng.normal(0, 2, n)  # tighter, still unbiased

    rep = promotion_report(base, cand, "cand")
    assert rep["home_rmse_delta"] < 0
    assert any("bias not worse" in v for v in rep["verdicts"])
    assert not any("do not promote" in v for v in rep["verdicts"])


def test_promotion_reports_no_improvement_when_there_is_none():
    n = 200
    base = _frame(n, 21.0, 20.0)
    cand = _frame(n, 25.0, 24.0)  # strictly worse
    rep = promotion_report(base, cand, "cand")
    assert rep["home_rmse_delta"] > 0
    assert any("no RMSE improvement" in v for v in rep["verdicts"])
