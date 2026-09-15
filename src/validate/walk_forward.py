"""
Generic walk-forward (rolling-origin) evaluation harness.
For each test week, trains only on games strictly before that week,
predicts that week, and records results. This is the one evaluation
method every model in this project must be scored with.

Usage: import walk_forward_evaluate() from model-specific scripts.
"""

import time
import pandas as pd
import numpy as np


def walk_forward_evaluate(
    df: pd.DataFrame, fit_fn, predict_fn, min_train_seasons: int = 2
) -> pd.DataFrame:
    """
    df: model_table with columns season, week, gameday, home_score, away_score, features
    fit_fn(train_df) -> fitted model object (or params dict)
    predict_fn(model, test_df) -> (home_pred, away_pred) arrays
    min_train_seasons: burn-in period before folds start (rolling features need history)
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("gameday")

    seasons = sorted(df["season"].unique())
    test_seasons = seasons[min_train_seasons:]

    # Cheap upfront count (just unique season/week pairs, no model fitting) so
    # progress prints can show "fold N/total" -- not a test run, just counting rows.
    total_folds = sum(df.loc[df["season"] == s, "week"].nunique() for s in test_seasons)

    results = []
    fold_num = 0
    start_time = time.time()

    for season in test_seasons:
        weeks = sorted(df.loc[df["season"] == season, "week"].unique())
        for week in weeks:
            fold_num += 1
            cutoff = df[(df["season"] == season) & (df["week"] == week)][
                "gameday"
            ].min()
            train = df[df["gameday"] < cutoff]
            test = df[(df["season"] == season) & (df["week"] == week)]

            if len(train) < 50 or len(test) == 0:
                elapsed = time.time() - start_time
                print(
                    f"  [{elapsed:7.1f}s] fold {fold_num}/{total_folds} "
                    f"(season {season} week {week}) SKIPPED (insufficient train/test rows)"
                )
                continue

            model = fit_fn(train)
            home_pred, away_pred = predict_fn(model, test)

            fold_result = test[
                ["game_id", "season", "week", "home_score", "away_score"]
            ].copy()
            fold_result["home_pred"] = home_pred
            fold_result["away_pred"] = away_pred
            results.append(fold_result)

            elapsed = time.time() - start_time
            print(
                f"  [{elapsed:7.1f}s] fold {fold_num}/{total_folds} "
                f"(season {season} week {week}) done -- {len(train)} train rows"
            )

    total_elapsed = time.time() - start_time
    print(
        f"  Walk-forward complete: {total_folds} folds in {total_elapsed:.1f}s "
        f"({total_elapsed/max(total_folds,1):.2f}s/fold avg)"
    )

    return pd.concat(results, ignore_index=True)


def walk_forward_evaluate_by_season(
    df: pd.DataFrame, fit_fn, predict_fn, min_train_seasons: int = 2
) -> pd.DataFrame:
    """
    Coarser-grained walk-forward: refits once per SEASON instead of once per
    week. Still fully leakage-safe (trains only on seasons strictly before
    the test season) -- just a cheaper validation granularity, appropriate
    for computationally expensive models where weekly refitting is
    impractical.
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[min_train_seasons:]

    results = []
    start_time = time.time()
    for i, season in enumerate(test_seasons, 1):
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if len(train) < 50 or len(test) == 0:
            continue

        model = fit_fn(train)
        home_pred, away_pred = predict_fn(model, test)

        fold_result = test[
            ["game_id", "season", "week", "home_score", "away_score"]
        ].copy()
        fold_result["home_pred"] = home_pred
        fold_result["away_pred"] = away_pred
        results.append(fold_result)

        elapsed = time.time() - start_time
        print(f"  [{elapsed:7.1f}s] season {season} ({i}/{len(test_seasons)}) done")

    return pd.concat(results, ignore_index=True)


def score_predictions(results: pd.DataFrame) -> dict:
    home_err = results["home_score"] - results["home_pred"]
    away_err = results["away_score"] - results["away_pred"]
    return {
        "home_rmse": np.sqrt((home_err**2).mean()),
        "away_rmse": np.sqrt((away_err**2).mean()),
        "home_mae": home_err.abs().mean(),
        "away_mae": away_err.abs().mean(),
        "n_games": len(results),
    }
