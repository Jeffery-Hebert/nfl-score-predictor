"""
Residual/error analysis: bias, consistency, home/away error correlation,
and week-of-season slicing. Answers: is the model consistently off (easy
to correct) or inconsistently off (hard to trust)? Does accuracy differ
between early-season and late-season games?

Run: python -m src.validate.error_analysis                  # every saved model
     python -m src.validate.error_analysis linear poisson  # named models
     python -m src.validate.error_analysis --file data/processed/rf_base_predictions.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PRED_DIR = Path("data/processed")


def discover_models() -> dict[str, str]:
    """Every *_predictions.parquet on disk, newest first."""
    found = sorted(
        PRED_DIR.glob("*_predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {p.name.replace("_predictions.parquet", ""): str(p) for p in found}


def analyze(preds_path: str, label: str):
    df = pd.read_parquet(preds_path)
    df["home_error"] = df["home_pred"] - df["home_score"]
    df["away_error"] = df["away_pred"] - df["away_score"]
    df["margin_actual"] = df["home_score"] - df["away_score"]
    df["margin_pred"] = df["home_pred"] - df["away_pred"]
    df["margin_error"] = df["margin_pred"] - df["margin_actual"]

    print(f"\n{'='*60}\n{label}\n{'='*60}")

    print("\n-- Bias check (mean signed error; 0 = no systematic bias) --")
    print(f"  Home bias: {df['home_error'].mean():+.2f}")
    print(f"  Away bias: {df['away_error'].mean():+.2f}")

    print("\n-- Consistency (std dev of error; lower = more consistent) --")
    print(f"  Home error std dev: {df['home_error'].std():.2f}")
    print(f"  Away error std dev: {df['away_error'].std():.2f}")
    print(
        f"  Margin error RMSE (direct, not approximated): {np.sqrt((df['margin_error']**2).mean()):.2f}"
    )

    print("\n-- Home/away error correlation --")
    corr = df["home_error"].corr(df["away_error"])
    print(
        f"  Correlation: {corr:.3f}  (near 0 = independent misses; near +/-1 = errors move together)"
    )

    print("\n-- Error by week-of-season bucket --")
    df["week_bucket"] = pd.cut(
        df["week"],
        bins=[0, 3, 8, 13, 22],
        labels=["Weeks 1-3", "Weeks 4-8", "Weeks 9-13", "Weeks 14+"],
    )
    bucket_stats = df.groupby("week_bucket").apply(
        lambda g: pd.Series(
            {
                "n_games": len(g),
                "home_rmse": np.sqrt((g["home_error"] ** 2).mean()),
                "away_rmse": np.sqrt((g["away_error"] ** 2).mean()),
                "home_bias": g["home_error"].mean(),
                "away_bias": g["away_error"].mean(),
            }
        ),
        include_groups=False,
    )
    print(bucket_stats.round(2))

    print("\n-- Worst 10 individual misses (by combined absolute error) --")
    df["total_abs_error"] = df["home_error"].abs() + df["away_error"].abs()
    worst = df.nlargest(10, "total_abs_error")[
        [
            "game_id",
            "season",
            "week",
            "home_score",
            "home_pred",
            "away_score",
            "away_pred",
        ]
    ]
    print(worst.to_string(index=False))


def main(argv=None):
    available = discover_models()
    ap = argparse.ArgumentParser(
        description="Residual/error analysis for saved predictions."
    )
    ap.add_argument(
        "models",
        nargs="*",
        help=f"model names to analyze (default: all). Available: {', '.join(sorted(available)) or 'none found'}",
    )
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        help="analyze an explicit prediction parquet path (repeatable)",
    )
    ap.add_argument(
        "--list", action="store_true", help="list available models and exit"
    )
    args = ap.parse_args(argv)

    if args.list:
        for name, path in available.items():
            print(f"  {name:<16} {path}")
        return 0

    targets = []
    for name in args.models:
        if name not in available:
            print(
                f"ERROR: no predictions for {name!r}. Available: {', '.join(sorted(available)) or 'none'}"
            )
            return 1
        targets.append((available[name], name))
    for path in args.file:
        if not Path(path).exists():
            print(f"ERROR: {path} does not exist")
            return 1
        targets.append((path, Path(path).stem))
    if not targets:
        if not available:
            print(
                "No *_predictions.parquet found in data/processed -- run a model first."
            )
            return 1
        targets = [(p, n) for n, p in available.items()]

    for path, label in targets:
        analyze(path, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
