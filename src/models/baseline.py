"""
Baseline model: no ML, just a matchup-adjusted blend of each team's own
recency-weighted scoring rate and the opponent's recency-weighted rate
allowed, plus a home-field adjustment learned from the training fold.

This MUST be beaten out-of-sample by every other model before that model
is trusted. If a model can't beat this, it's not adding real signal.

Run: python src/models/baseline.py
"""

import pandas as pd
from pathlib import Path
from src.models.common import ot_sample_weight
from src.validate.walk_forward import walk_forward_evaluate, score_predictions


def fit_baseline(train: pd.DataFrame) -> dict:
    """Learn the home-field edge from the training fold.

    C3: neutral-site games (London, Mexico, Munich, Super Bowl) have no home
    team, so including them drags the estimate toward zero and, worse, the
    adjustment then gets applied TO them at prediction time. They are excluded
    from the estimate and receive no adjustment below.

    C5: historical overtime games are down-weighted -- an extra period inflates
    the final margin in a way no pregame quantity predicts.
    """
    import numpy as np

    fit_rows = train
    if "is_neutral_site" in train.columns:
        fit_rows = train[train["is_neutral_site"] == 0]
        if fit_rows.empty:
            fit_rows = train

    margin = fit_rows["home_score"] - fit_rows["away_score"]
    w = ot_sample_weight(fit_rows)
    home_field_adj = float(
        np.average(margin, weights=w) if w is not None else margin.mean()
    )
    return {"home_field_adj": home_field_adj}


def predict_baseline(model: dict, test: pd.DataFrame):
    # C3: a neutral-site game gets no home-field adjustment.
    if "is_neutral_site" in test.columns:
        adj = model["home_field_adj"] * (1 - test["is_neutral_site"])
    else:
        adj = model["home_field_adj"]
    home_pred = (
        0.5 * test["home_pregame_team_score"]
        + 0.5 * test["away_pregame_opp_score"]
        + adj / 2
    )
    away_pred = (
        0.5 * test["away_pregame_team_score"]
        + 0.5 * test["home_pregame_opp_score"]
        - adj / 2
    )
    home_pred = home_pred.fillna(test["home_pregame_team_score"].mean())
    away_pred = away_pred.fillna(test["away_pregame_team_score"].mean())
    return home_pred, away_pred


def fit_league_average(train: pd.DataFrame) -> dict:
    return {"avg": train["home_score"].mean() * 0.5 + train["away_score"].mean() * 0.5}


def predict_league_average(model: dict, test: pd.DataFrame):
    n = len(test)
    return pd.Series([model["avg"]] * n, index=test.index), pd.Series(
        [model["avg"]] * n, index=test.index
    )


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(
        df, fit_baseline, predict_baseline, min_train_seasons=2
    )
    metrics = score_predictions(results)

    print("Baseline walk-forward results (2021-2025, trained on strictly-prior games):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = Path("data/processed/baseline_predictions.parquet")
    results.to_parquet(out_path, index=False)
    print(f"Saved fold-by-fold predictions to {out_path}")

    floor_results = walk_forward_evaluate(
        df, fit_league_average, predict_league_average, min_train_seasons=2
    )
    floor_metrics = score_predictions(floor_results)
    print("\nTrivial floor (always predict league-average score):")
    for k, v in floor_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
