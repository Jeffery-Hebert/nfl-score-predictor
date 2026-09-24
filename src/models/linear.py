"""
Regularized linear model: separate ridge regression for home_score and
away_score on the shared FEATURE_COLS. Simplest real ML model in the ensemble
-- if this can't beat the rule-based baseline, none of the fancier models are
likely to either.

Why ridge and not ordinary least squares. This model used to be sklearn
LinearRegression, which applies NO regularization, on 18 correlated features
with roughly 1,900 training rows per fold. That is the wrong estimator for the
shape of this problem, and it showed: src/experiments/test_advanced_metrics.py
found accuracy degrading monotonically as features were added (9.4110 -> 9.5302
with 54 features), while Poisson -- whose PoissonRegressor defaults to
alpha=1.0 and is therefore already regularized -- barely moved under the same
features. A paired bootstrap confirmed OLS-with-more-features was genuinely
worse, not noise (+0.1193, 95% CI [+0.0585, +0.1800]).

Why RidgeCV and not a hand-picked alpha. In testing, a fixed alpha=100 scored
best of everything tried (9.3971). Adopting that number would be tuning a
hyperparameter on the test set -- precisely the error this project's
walk-forward discipline exists to prevent. RidgeCV instead selects alpha from
a TimeSeriesSplit INSIDE each training fold, so the test week never influences
the choice. It scores slightly worse than the test-set-optimal alpha and is
the only honest option.

Scaling happens inside the pipeline, fitted per fold, so no test-fold statistic
leaks into training.

Run: python -m src.models.linear
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models.common import (
    FEATURE_COLS,
    chronological,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

MODEL_TABLE = "data/processed/model_table.parquet"

ALPHAS = np.logspace(-2, 4, 25)
INNER_CV = 5  # chronological splits within the training fold


def _make_model():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=INNER_CV)),
    )


def fit_linear(train: pd.DataFrame) -> dict:
    # RidgeCV's TimeSeriesSplit needs oldest-first rows -- see chronological().
    train = chronological(train)
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    w = ot_sample_weight(train)  # C5: halve historical overtime games
    home_model = _make_model().fit(
        X_filled, train["home_score"], ridgecv__sample_weight=w
    )
    away_model = _make_model().fit(
        X_filled, train["away_score"], ridgecv__sample_weight=w
    )
    # Correct the scoring-environment drift -- see recent_residual_offset.
    off_h, off_a = recent_residual_offset(
        train, home_model.predict(X_filled), away_model.predict(X_filled)
    )
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "off_h": off_h,
        "off_a": off_a,
    }


def predict_linear(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    return (
        model["home_model"].predict(X) - model["off_h"],
        model["away_model"].predict(X) - model["off_a"],
    )


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate(df, fit_linear, predict_linear, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Linear Regression walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "linear", inputs=[MODEL_TABLE])


if __name__ == "__main__":
    main()
