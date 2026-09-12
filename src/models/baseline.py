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
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

def fit_baseline(train: pd.DataFrame) -> dict:
    home_field_adj = (train["home_score"] - train["away_score"]).mean()
    return {"home_field_adj": home_field_adj}

def predict_baseline(model: dict, test: pd.DataFrame):
    adj = model["home_field_adj"]
    home_pred = 0.5 * test["home_pregame_team_score"] + 0.5 * test["away_pregame_opp_score"] + adj / 2
    away_pred = 0.5 * test["away_pregame_team_score"] + 0.5 * test["home_pregame_opp_score"] - adj / 2
    home_pred = home_pred.fillna(test["home_pregame_team_score"].mean())
    away_pred = away_pred.fillna(test["away_pregame_team_score"].mean())
    return home_pred, away_pred

def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate(df, fit_baseline, predict_baseline, min_train_seasons=2)
    metrics = score_predictions(results)

    print("Baseline walk-forward results (2021-2025, trained on strictly-prior games):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = Path("data/processed/baseline_predictions.parquet")
    results.to_parquet(out_path, index=False)
    print(f"Saved fold-by-fold predictions to {out_path}")

if __name__ == "__main__":
    main()