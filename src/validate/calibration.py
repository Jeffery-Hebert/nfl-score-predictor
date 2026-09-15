"""
B7/B9: calibration, dispersion and the bias-aware promotion gate.

Two gaps this closes.

B7 -- the project has only ever measured RMSE/MAE. A model can carry the same
RMSE as another while being systematically wrong in a way that matters
downstream: if predicted scores are compressed toward the mean, every derived
spread and total inherits that compression, and the model will look fine on
RMSE while being useless for betting decisions. Calibration slope catches that
and RMSE cannot.

B9 -- baseline.py says every model "MUST be beaten out-of-sample" before it is
trusted, but the check has always been RMSE alone. A model can win on RMSE
while being meaningfully more biased. promotion_report() reports both and
refuses to call a model better when it trades bias for RMSE.

Note on intervals: these models emit point predictions only. GP computes a
genuine predictive sigma and gaussian_process.py discards it (the harness
contract is (home_pred, away_pred)). So implied_interval_coverage() below
assumes a normal with sigma = residual RMSE -- it measures whether residuals
are dispersed the way a naive interval would assume, NOT true predictive
calibration. Labelled as such wherever it prints.

Run: python -m src.validate.calibration                  # all saved models
     python -m src.validate.calibration poisson linear
     python -m src.validate.calibration --against baseline
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRED_DIR = Path("data/processed")
SIDES = ("home", "away")


def load_predictions(name: str) -> pd.DataFrame:
    path = PRED_DIR / f"{name}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run that model first")
    return pd.read_parquet(path)


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean signed error. Positive = the model predicts too high."""
    return float(np.mean(pred - actual))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def calibration_slope_intercept(actual: np.ndarray, pred: np.ndarray):
    """Regress actual on predicted. Perfect calibration is slope 1, intercept 0.

    slope < 1 means predictions are over-dispersed (too extreme);
    slope > 1 means they are compressed toward the mean and the model is
    under-confident -- the failure mode that quietly kills derived spreads.
    """
    slope, intercept = np.polyfit(pred, actual, 1)
    return float(slope), float(intercept)


def reliability_table(actual: np.ndarray, pred: np.ndarray, n_bins: int = 5):
    """Mean predicted vs mean actual within prediction quantile bins."""
    df = pd.DataFrame({"actual": actual, "pred": pred})
    df["bin"] = pd.qcut(df["pred"], n_bins, duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("actual", "size"),
            mean_pred=("pred", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index(drop=True)
    )
    out["gap"] = out["mean_pred"] - out["mean_actual"]
    return out


def implied_interval_coverage(
    actual: np.ndarray, pred: np.ndarray, levels=(0.50, 0.80, 0.95)
):
    """Fraction of actuals inside pred +/- z*sigma, with sigma = residual RMSE.

    NOT true predictive calibration -- these models emit no per-prediction
    sigma. It answers: if you built naive normal intervals from this model's
    overall error spread, would they cover what they claim? Big shortfalls mean
    heavy tails, and that naive intervals would be overconfident.
    """
    from scipy.stats import norm

    resid = actual - pred
    sigma = float(np.sqrt(np.mean(resid**2)))
    return {
        level: float(np.mean(np.abs(resid) <= norm.ppf(0.5 + level / 2) * sigma))
        for level in levels
    }


def calibration_report(df: pd.DataFrame, label: str):
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    for side in SIDES:
        actual = df[f"{side}_score"].to_numpy(dtype=float)
        pred = df[f"{side}_pred"].to_numpy(dtype=float)
        slope, intercept = calibration_slope_intercept(actual, pred)
        cov = implied_interval_coverage(actual, pred)

        print(f"\n-- {side} score --")
        print(f"  rmse {rmse(actual, pred):.3f}   bias {bias(actual, pred):+.3f}")
        print(
            f"  calibration slope {slope:.3f} (1.0 = perfect), intercept {intercept:+.2f}"
        )
        if slope > 1.15:
            print(
                "    -> predictions are COMPRESSED toward the mean (under-confident);"
            )
            print("       derived spreads/totals will be systematically too flat")
        elif slope < 0.85:
            print("    -> predictions are OVER-dispersed (too extreme)")
        print(
            "  implied interval coverage (naive normal, not a real predictive interval):"
        )
        for level, got in cov.items():
            print(f"    {level:.0%} interval covers {got:.1%}  ({got - level:+.1%})")
        print("  reliability by predicted-score bin:")
        print(reliability_table(actual, pred).to_string(index=False))


def promotion_report(
    baseline: pd.DataFrame, candidate: pd.DataFrame, name: str
) -> dict:
    """B9: RMSE and bias together. A model that wins on RMSE while getting
    materially more biased has not earned promotion."""
    merged = baseline.merge(candidate, on="game_id", suffixes=("_base", "_cand"))
    out = {"model": name, "n_games": len(merged)}
    verdicts = []
    for side in SIDES:
        actual = merged[f"{side}_score_base"].to_numpy(dtype=float)
        pb = merged[f"{side}_pred_base"].to_numpy(dtype=float)
        pc = merged[f"{side}_pred_cand"].to_numpy(dtype=float)

        d_rmse = rmse(actual, pc) - rmse(actual, pb)
        d_absbias = abs(bias(actual, pc)) - abs(bias(actual, pb))
        out[f"{side}_rmse_delta"] = d_rmse
        out[f"{side}_absbias_delta"] = d_absbias

        if d_rmse < 0 and d_absbias > 0.25:
            verdicts.append(
                f"{side}: better RMSE ({d_rmse:+.4f}) but MORE biased "
                f"({d_absbias:+.3f}) -- do not promote on RMSE alone"
            )
        elif d_rmse < 0:
            verdicts.append(f"{side}: better RMSE ({d_rmse:+.4f}), bias not worse")
        else:
            verdicts.append(f"{side}: no RMSE improvement ({d_rmse:+.4f})")
    out["verdicts"] = verdicts
    return out


def main(argv=None):
    available = sorted(
        p.name.replace("_predictions.parquet", "")
        for p in PRED_DIR.glob("*_predictions.parquet")
    )
    ap = argparse.ArgumentParser(
        description="Calibration and bias-aware promotion checks."
    )
    ap.add_argument(
        "models", nargs="*", help=f"available: {', '.join(available) or 'none'}"
    )
    ap.add_argument(
        "--against",
        default=None,
        help="also run the bias-aware promotion check against this model (e.g. baseline)",
    )
    args = ap.parse_args(argv)

    targets = args.models or available
    if not targets:
        print("No predictions found in data/processed -- run a model first.")
        return 1

    for name in targets:
        try:
            calibration_report(load_predictions(name), name)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1

    if args.against:
        base = load_predictions(args.against)
        print(
            f"\n\n{'=' * 66}\nB9 PROMOTION GATE vs {args.against} (RMSE *and* bias)\n{'=' * 66}"
        )
        for name in targets:
            if name == args.against:
                continue
            rep = promotion_report(base, load_predictions(name), name)
            print(f"\n  {name}  (n={rep['n_games']})")
            for v in rep["verdicts"]:
                print(f"    {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
