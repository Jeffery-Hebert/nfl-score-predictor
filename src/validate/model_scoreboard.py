"""
One ranked view of every model that has saved predictions, with the accuracy
metrics that matter downstream and the correctness checks that say whether a
number can be trusted at all.

Why it exists: model results lived in separate parquet files written at
different times against different feature versions, and comparing them meant
eyeballing whatever a script last printed. That is how a stale baseline ended
up being compared against fresh models for two days.

Reports per model:
  accuracy    home/away RMSE and MAE, plus margin and total RMSE -- margin and
              total are what any spread or over/under decision actually uses,
              and a model can look fine on score RMSE while being worse on both
  calibration slope and bias (see calibration.py -- RMSE cannot see compression)
  correctness NaNs, negative or implausible scores, degenerate constant output,
              game coverage, and staleness against model_table.parquet
  vs baseline  paired bootstrap, resampling whole WEEKS (games in one week share
              a fitted model and a scoring environment, so resampling single
              games -- what this used to do -- gives intervals that are too
              narrow)

Run: python -m src.validate.model_scoreboard
     python -m src.validate.model_scoreboard --against baseline --common-games
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.validate.calibration import bias, calibration_slope_intercept, rmse

PRED_DIR = Path("data/processed")
PLAUSIBLE_MAX = 70.0
N_BOOT = 5000
SEED = 42


def discover() -> dict[str, Path]:
    """Registered models with a saved backtest. Unregistered leftovers in
    data/processed/ (old one-off experiment outputs) are ignored rather than
    ranked alongside current models."""
    from src.models import registry

    return {
        n: PRED_DIR / f"{n}_predictions.parquet"
        for n in registry.names()
        if (PRED_DIR / f"{n}_predictions.parquet").exists()
    }


def correctness_checks(name: str, path: Path, df: pd.DataFrame) -> list[str]:
    """Anything that makes this model's numbers untrustworthy, regardless of RMSE."""
    problems = []
    for side in ("home", "away"):
        pred = df[f"{side}_pred"]
        if pred.isna().any():
            problems.append(f"{pred.isna().sum()} NaN {side} predictions")
        if (pred < 0).any():
            problems.append(f"{(pred < 0).sum()} negative {side} predictions")
        if (pred > PLAUSIBLE_MAX).any():
            problems.append(
                f"{(pred > PLAUSIBLE_MAX).sum()} {side} predictions > {PLAUSIBLE_MAX:.0f}"
            )
        if pred.nunique() <= 1:
            problems.append(f"{side} predictions are constant (degenerate model)")
    if df["game_id"].duplicated().any():
        problems.append(f"{df['game_id'].duplicated().sum()} duplicate game_ids")
    if df[["home_score", "away_score"]].isna().any().any():
        problems.append("scored against rows with no final score")
    # Staleness by CONTENT: the played rows this backtest read must still hash
    # the same (src/validate/backtest_io.py). This used to compare file mtimes,
    # which cried wolf after every rebuild.
    from src.validate.backtest_io import stale_inputs

    for issue in stale_inputs(name):
        problems.append(f"STALE: {issue}")
    return problems


def metrics(df: pd.DataFrame) -> dict:
    h_a, a_a = df["home_score"].to_numpy(float), df["away_score"].to_numpy(float)
    h_p, a_p = df["home_pred"].to_numpy(float), df["away_pred"].to_numpy(float)
    return {
        "n": len(df),
        "home_rmse": rmse(h_a, h_p),
        "away_rmse": rmse(a_a, a_p),
        "mean_rmse": (rmse(h_a, h_p) + rmse(a_a, a_p)) / 2,
        "home_mae": float(np.mean(np.abs(h_p - h_a))),
        "away_mae": float(np.mean(np.abs(a_p - a_a))),
        "margin_rmse": rmse(h_a - a_a, h_p - a_p),
        "total_rmse": rmse(h_a + a_a, h_p + a_p),
        "home_bias": bias(h_a, h_p),
        "away_bias": bias(a_a, a_p),
        "home_slope": calibration_slope_intercept(h_a, h_p)[0],
    }


def bootstrap_vs(base: pd.DataFrame, cand: pd.DataFrame):
    """Paired bootstrap, resampling whole weeks, pooled home+away errors.
    Returns (delta, lo, hi); negative delta = the candidate is better."""
    from src.validate.model_report import BlockBootstrap

    m = base.merge(cand, on=["game_id", "season", "week"], suffixes=("_b", "_c"))
    if m.empty:
        return None
    eb = np.concatenate(
        [
            (m.home_pred_b - m.home_score_b).to_numpy(float),
            (m.away_pred_b - m.away_score_b).to_numpy(float),
        ]
    )
    ec = np.concatenate(
        [
            (m.home_pred_c - m.home_score_c).to_numpy(float),
            (m.away_pred_c - m.away_score_c).to_numpy(float),
        ]
    )
    block = (m["season"] * 100 + m["week"]).to_numpy()
    d, lo, hi, _ = BlockBootstrap(
        np.concatenate([block, block]), N_BOOT, SEED
    ).rmse_delta(ec, eb)
    return d, lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="default: every saved model")
    ap.add_argument(
        "--against",
        default="baseline",
        help="significance reference (default: baseline)",
    )
    ap.add_argument(
        "--common-games",
        action="store_true",
        help="restrict every model to the games ALL of them share",
    )
    args = ap.parse_args(argv)

    found = discover()
    names = args.models or sorted(found)
    missing = [n for n in names if n not in found]
    if missing:
        print(f"ERROR: no predictions for {missing}. Available: {sorted(found)}")
        return 1

    frames = {n: pd.read_parquet(found[n]) for n in names}

    if args.common_games:
        shared = set.intersection(*(set(f["game_id"]) for f in frames.values()))
        frames = {
            n: f[f["game_id"].isin(shared)]
            .sort_values("game_id")
            .reset_index(drop=True)
            for n, f in frames.items()
        }
        print(
            f"Restricted to the {len(shared)} games common to all {len(frames)} models.\n"
        )

    rows = []
    issues = {}
    for n, f in frames.items():
        rows.append({"model": n, **metrics(f)})
        p = correctness_checks(n, found[n], f)
        if p:
            issues[n] = p

    table = pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)
    print(
        f"{'model':<14}{'n':>6}{'home':>8}{'away':>8}{'mean':>8}{'margin':>9}{'total':>8}"
        f"{'h_bias':>8}{'slope':>7}  flags"
    )
    print("-" * 90)
    for _, r in table.iterrows():
        flag = (
            "OK"
            if r["model"] not in issues
            else f"!! {len(issues[r['model']])} issue(s)"
        )
        print(
            f"{r['model']:<14}{r['n']:>6.0f}{r['home_rmse']:>8.3f}{r['away_rmse']:>8.3f}"
            f"{r['mean_rmse']:>8.3f}{r['margin_rmse']:>9.3f}{r['total_rmse']:>8.3f}"
            f"{r['home_bias']:>+8.2f}{r['home_slope']:>7.3f}  {flag}"
        )

    if issues:
        print("\nCORRECTNESS ISSUES -- these numbers are not comparable:")
        for n, probs in issues.items():
            for p in probs:
                print(f"  {n:<14} {p}")

    ref = args.against
    if ref in frames:
        print(
            f"\nSignificance vs {ref} (paired bootstrap resampling WEEKS, pooled home+away)"
        )
        print("  negative delta = better than the reference\n")
        for n in table["model"]:
            if n == ref:
                continue
            res = bootstrap_vs(frames[ref], frames[n])
            if res is None:
                print(f"  {n:<14} no overlapping games with {ref}")
                continue
            mean, lo, hi = res
            verdict = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"  {n:<14} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
