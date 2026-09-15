"""
src/experiments/bootstrap_rf_significance_test.py

Tests whether the RF split+CPOE improvement is statistically real or noise,
using the already-saved rf_base_predictions.parquet and rf_extended_predictions.parquet.

Run: python -m src.experiments.bootstrap_rf_significance_test
"""

import numpy as np
import pandas as pd


def bootstrap_rmse_delta(
    base_results, ext_results, score_col_prefix, n_boot=5000, seed=42
):
    merged = base_results.merge(ext_results, on="game_id", suffixes=("_base", "_ext"))
    err_base = (
        merged[f"{score_col_prefix}_pred_base"]
        - merged[f"{score_col_prefix}_score_base"]
    ).values
    err_ext = (
        merged[f"{score_col_prefix}_pred_ext"] - merged[f"{score_col_prefix}_score_ext"]
    ).values
    n = len(merged)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rmse_base = np.sqrt((err_base[idx] ** 2).mean())
        rmse_ext = np.sqrt((err_ext[idx] ** 2).mean())
        deltas[i] = rmse_ext - rmse_base
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5])


def main():
    base_results = pd.read_parquet("data/processed/rf_base_predictions.parquet")
    ext_results = pd.read_parquet("data/processed/rf_extended_predictions.parquet")

    for prefix in ["home", "away"]:
        mean_delta, (ci_low, ci_high) = bootstrap_rmse_delta(
            base_results, ext_results, prefix
        )
        print(f"{prefix.upper()} RMSE delta: {mean_delta:+.4f}")
        print(f"  95% bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
        if ci_low <= 0 <= ci_high:
            print(
                "  Zero falls within CI -- NOT statistically distinguishable from noise.\n"
            )
        else:
            print("  Zero falls outside CI -- likely a real effect.\n")


if __name__ == "__main__":
    main()
