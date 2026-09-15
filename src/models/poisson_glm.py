"""
Poisson GLM: respects that scores are non-negative counts, unlike plain
linear regression which can predict negative scores.

Regularization note. PoissonRegressor's alpha defaults to 1.0, so this model
has always been regularized -- which is why it degraded far less than the old
unregularized linear model when features were added, and why it had been
quietly outscoring it.

alpha=1.0 was inherited rather than chosen, so it was tested. In-fold
selection (GridSearchCV over a TimeSeriesSplit inside each training fold, the
same leakage-safe pattern that helped linear.py) was measured and is WORSE
here: mean RMSE 9.4175 against 9.3920 for the default, at 3.5x the runtime
(78s vs 22s). The inner folds are small enough that the selected alpha is
noisy, and the adaptivity costs more than it buys.

So the default stands -- but now on evidence rather than by accident. This is
the opposite outcome to linear.py, where the problem was a complete ABSENCE of
regularization rather than an unexamined amount of it, and where in-fold
selection genuinely helped (9.4110 -> 9.4045). Do not "fix" this one by
copying that pattern over; it has been tried.

Run: python -m src.models.poisson_glm
"""

import pandas as pd
from sklearn.linear_model import PoissonRegressor
from src.models.common import (
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

from sklearn.preprocessing import StandardScaler

# Validated, not inherited -- see the module docstring. In-fold selection was
# measured at 9.4175 mean RMSE against 9.3920 here.
POISSON_ALPHA = 1.0


def fit_poisson(train: pd.DataFrame) -> dict:
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)
    w = ot_sample_weight(train)  # C5: halve historical overtime games
    home_model = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=300).fit(
        X_scaled, train["home_score"], sample_weight=w
    )
    away_model = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=300).fit(
        X_scaled, train["away_score"], sample_weight=w
    )
    off_h, off_a = recent_residual_offset(
        train, home_model.predict(X_scaled), away_model.predict(X_scaled)
    )
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "scaler": scaler,
        "off_h": off_h,
        "off_a": off_a,
    }


def predict_poisson(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    return (
        model["home_model"].predict(X_scaled) - model["off_h"],
        model["away_model"].predict(X_scaled) - model["off_a"],
    )


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(
        df, fit_poisson, predict_poisson, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("Poisson GLM walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/poisson_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
