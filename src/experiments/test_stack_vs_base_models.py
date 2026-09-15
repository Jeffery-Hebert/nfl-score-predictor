"""
src/experiments/test_stack_vs_base_models.py

A5: does the Ridge stack actually earn its place, and do the base models clear
the project's own promotion gate ("must beat the rule-based baseline")?

Two comparisons that were previously impossible to make correctly:

1. stacking.py uses min_train_seasons=1 on a meta-table that already spans
   2021-2026, so it tests on 2022-2026 (1141 games) while every base model
   reports on 2021-2026 (1426). Comparing the headline numbers compares
   different test sets. This script scores everything on the common set.

2. Deltas here are all under ~0.08 RMSE, well inside the project's "needs a
   bootstrap" band. Uses a paired bootstrap that resamples GAMES (not
   individual scores), so a resampled game contributes both its home and away
   error and within-game correlation is preserved.

Reads only saved prediction files -- run the models first:
    python -m src.models.baseline
    python -m src.models.linear
    python -m src.models.poisson_glm
    python -m src.models.gaussian_process   # ~26 min
    python -m src.models.stacking

Run: python -m src.experiments.test_stack_vs_base_models
"""

import numpy as np
import pandas as pd

MODELS = ["baseline", "linear", "poisson", "gp", "stacking"]
N_BOOT = 10000
SEED = 42


def load(name: str, game_ids: set | None = None) -> pd.DataFrame:
    df = pd.read_parquet(f"data/processed/{name}_predictions.parquet")
    if game_ids is not None:
        df = df[df["game_id"].isin(game_ids)]
    return df.sort_values("game_id").reset_index(drop=True)


def errors(df: pd.DataFrame):
    return (
        (df["home_pred"] - df["home_score"]).values,
        (df["away_pred"] - df["away_score"]).values,
    )


def pooled_rmse(df: pd.DataFrame) -> float:
    h, a = errors(df)
    return float(np.sqrt((np.concatenate([h, a]) ** 2).mean()))


def bootstrap_by_game(base: pd.DataFrame, challenger: pd.DataFrame):
    """Paired bootstrap over games. Returns (mean delta, (lo, hi), P(better)).
    delta = challenger RMSE - base RMSE, so negative means challenger is better."""
    hb, ab = errors(base)
    hc, ac = errors(challenger)
    rng = np.random.default_rng(SEED)
    n = len(hb)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        k = rng.integers(0, n, n)
        r_base = np.sqrt((np.concatenate([hb[k], ab[k]]) ** 2).mean())
        r_chal = np.sqrt((np.concatenate([hc[k], ac[k]]) ** 2).mean())
        deltas[i] = r_chal - r_base
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5]), float((deltas < 0).mean())


def verdict(lo: float, hi: float) -> str:
    if lo <= 0 <= hi:
        return "NOT distinguishable from noise"
    return "BETTER (real)" if hi < 0 else "WORSE (real)"


def compare(label: str, base_name: str, base: pd.DataFrame, others: dict):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  delta = challenger - {base_name}; NEGATIVE means challenger is better\n")
    for name, df in others.items():
        mean, (lo, hi), p_better = bootstrap_by_game(base, df)
        print(
            f"  {name:<10} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"P(better)={p_better:5.1%}  -> {verdict(lo, hi)}"
        )


def main():
    full = {m: load(m) for m in MODELS if m != "stacking"}
    print(f"=== Full test set, n={len(full['baseline'])} games ===")
    for name, df in sorted(full.items(), key=lambda kv: pooled_rmse(kv[1])):
        print(f"  {name:<10} pooled_rmse={pooled_rmse(df):.4f}")

    compare(
        "PROMOTION GATE: does each model beat the rule-based baseline?",
        "baseline",
        full["baseline"],
        {k: v for k, v in full.items() if k != "baseline"},
    )

    stack = load("stacking")
    ids = set(stack["game_id"])
    common = {m: load(m, ids) for m in MODELS if m != "stacking"}
    common["stacking"] = stack

    print(f"\n\n=== Common test set, n={len(ids)} games (stacking's window) ===")
    for name, df in sorted(common.items(), key=lambda kv: pooled_rmse(kv[1])):
        print(f"  {name:<10} pooled_rmse={pooled_rmse(df):.4f}")

    best_base = min(["linear", "poisson", "gp"], key=lambda m: pooled_rmse(common[m]))
    print(f"\n  best single base model on this set: {best_base}")

    compare(
        f"Does the stack beat its own best base model ({best_base})?",
        best_base,
        common[best_base],
        {"stacking": common["stacking"]},
    )
    compare(
        "Does the stack beat the rule-based baseline?",
        "baseline",
        common["baseline"],
        {"stacking": common["stacking"]},
    )


if __name__ == "__main__":
    main()
