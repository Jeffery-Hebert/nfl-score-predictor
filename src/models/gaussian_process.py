"""
Gaussian Process regression: non-linear, with genuine predictive uncertainty
(not just a point estimate). Cost grows ~cubically with training set size,
so check timing before running the full walk-forward.

Run: python -m src.models.gaussian_process
"""

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from src.models.common import FEATURE_COLS
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)


def fit_gp(train: pd.DataFrame) -> dict:
    # C5 note: sklearn's GaussianProcessRegressor.fit() takes no sample_weight,
    # so the overtime down-weighting applied in linear.py and poisson_glm.py
    # cannot be applied here. Left unweighted deliberately rather than dropping
    # rows, which would change the model rather than re-weight it.
    X = train[FEATURE_COLS]
    means = X.mean()
    X_filled = X.fillna(means)
    scaler = StandardScaler().fit(X_filled)
    X_scaled = scaler.transform(X_filled)

    home_model = GaussianProcessRegressor(
        kernel=KERNEL, normalize_y=True, random_state=42
    ).fit(X_scaled, train["home_score"])
    away_model = GaussianProcessRegressor(
        kernel=KERNEL, normalize_y=True, random_state=42
    ).fit(X_scaled, train["away_score"])
    return {
        "home_model": home_model,
        "away_model": away_model,
        "means": means,
        "scaler": scaler,
    }


def predict_gp(model: dict, test: pd.DataFrame):
    X = test[FEATURE_COLS].fillna(model["means"])
    X_scaled = model["scaler"].transform(X)
    return model["home_model"].predict(X_scaled), model["away_model"].predict(X_scaled)


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_gp, predict_gp, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Gaussian Process walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/gp_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
