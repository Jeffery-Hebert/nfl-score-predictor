"""
src/experiments/tune_ridge_alpha.py

C8: build_adjusted_ratings.py carried `RIDGE_ALPHA = 5.0  # starting default,
not yet validated`. The ratings it produces were used to decide a documented
null result, on a regularization strength nobody had checked.

Evaluates alpha on the ridge's OWN out-of-sample task rather than a downstream
model: at each week cutoff, fit team offense/defense ratings on strictly prior
games, then predict that week's actual off_epa_per_play. That is exactly what
the ratings claim to capture, so it is the right criterion for how hard to
regularize -- and it is independent of whichever model later consumes them.

Run: python -m src.experiments.tune_ridge_alpha
"""

import numpy as np
import pandas as pd

from src.features.build_adjusted_ratings import (
    RIDGE_ALPHA,
    build_design_matrix,
    fit_ratings,
    load_config,
)

ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
MIN_TRAIN_GAMES = 100


def main():
    halflife_days = load_config()["training"]["recency_half_life_weeks"] * 7
    tg = pd.read_parquet("data/processed/team_game_stats.parquet")
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    teams = sorted(tg["team"].unique())

    cutoffs = (
        tg.groupby(["season", "week"])["gameday"]
        .min()
        .reset_index()
        .sort_values("gameday")
    )

    print(f"Evaluating RIDGE_ALPHA on the ratings model's own out-of-sample task:")
    print(
        f"predicting each week's actual off_epa_per_play from strictly prior games.\n"
    )
    print(f"{'alpha':>8}{'out-of-sample RMSE':>22}{'n weeks':>10}")
    print("-" * 42)

    results = []
    for alpha in ALPHAS:
        errs = []
        n_weeks = 0
        for _, row in cutoffs.iterrows():
            cutoff = row["gameday"]
            train = tg[tg["gameday"] < cutoff]
            if len(train) < MIN_TRAIN_GAMES:
                continue
            test = tg[(tg["season"] == row["season"]) & (tg["week"] == row["week"])]
            test = test.dropna(subset=["off_epa_per_play"])
            if test.empty:
                continue

            off, deff = fit_ratings(train, teams, halflife_days, cutoff, alpha=alpha)
            # Reconstruct the model's prediction for each test row.
            intercept = train["off_epa_per_play"].mean()
            pred = (
                test["team"].map(off).to_numpy(float)
                + test["opponent"].map(deff).to_numpy(float)
                + intercept
            )
            errs.append(pred - test["off_epa_per_play"].to_numpy(float))
            n_weeks += 1

        all_err = np.concatenate(errs)
        rmse = float(np.sqrt((all_err**2).mean()))
        results.append({"alpha": alpha, "rmse": rmse, "n_weeks": n_weeks})
        print(f"{alpha:>8}{rmse:>22.6f}{n_weeks:>10}")

    table = pd.DataFrame(results)
    best = table.loc[table["rmse"].idxmin()]
    cur = table[table["alpha"] == RIDGE_ALPHA]["rmse"].iloc[0]
    print(f"\nBest alpha: {best['alpha']}  (RMSE {best['rmse']:.6f})")
    print(f"Current RIDGE_ALPHA = {RIDGE_ALPHA}  (RMSE {cur:.6f})")
    print(f"delta = {best['rmse'] - cur:+.6f}")
    if best["alpha"] != RIDGE_ALPHA:
        print(f"\n-> Set RIDGE_ALPHA = {best['alpha']} in build_adjusted_ratings.py")
    else:
        print("\n-> Current value is already optimal on this criterion.")


if __name__ == "__main__":
    main()
