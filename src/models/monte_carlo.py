"""
Monte Carlo drive simulation: for each game, blends the home team's
offensive drive-outcome rates with the away team's defensive (allowed)
rates -- and vice versa -- then simulates many trials of that many drives
per team, summing points per trial. Prediction is the mean simulated score.

This is the agreed substitute for literal RL: there's no sequential
agent-environment-reward structure in pregame score prediction, but this
achieves the same practical goal (a simulated score distribution).

Run: python -m src.models.monte_carlo
"""
import numpy as np
import pandas as pd
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

CATS = ["touchdown", "field_goal", "punt", "turnover", "turnover_on_downs",
        "missed_field_goal", "end_of_half", "opp_touchdown", "safety"]
POINTS = np.array([7, 3, 0, 0, 0, 0, 0, 0, 0])
N_TRIALS = 2000
RNG = np.random.default_rng(42)

def fit_montecarlo(train: pd.DataFrame) -> dict:
    """No real model fitting -- pregame features are already leakage-safe
    per-row. Only computes train-fold fallback values for cold-start rows."""
    off_cols = [f"home_pregame_off_{c}_rate" for c in CATS]
    def_cols = [f"home_pregame_def_{c}_rate" for c in CATS]
    fallback_off = train[off_cols].mean().values
    fallback_def = train[def_cols].mean().values
    fallback_off = fallback_off / fallback_off.sum()
    fallback_def = fallback_def / fallback_def.sum()
    fallback_drives = train["home_pregame_n_drives"].mean()
    return {"fallback_off": fallback_off, "fallback_def": fallback_def, "fallback_drives": fallback_drives}

def _simulate_side(off_rates: np.ndarray, def_rates_allowed: np.ndarray, n_drives: float) -> float:
    blended = (off_rates + def_rates_allowed) / 2
    blended = blended / blended.sum()
    n = max(1, round(n_drives))
    draws = RNG.multinomial(n, blended, size=N_TRIALS)
    totals = draws @ POINTS
    return totals.mean()

def predict_montecarlo(model: dict, test: pd.DataFrame):
    home_preds, away_preds = [], []
    for _, row in test.iterrows():
        home_off = row[[f"home_pregame_off_{c}_rate" for c in CATS]].values.astype(float)
        away_def = row[[f"away_pregame_def_{c}_rate" for c in CATS]].values.astype(float)
        away_off = row[[f"away_pregame_off_{c}_rate" for c in CATS]].values.astype(float)
        home_def = row[[f"home_pregame_def_{c}_rate" for c in CATS]].values.astype(float)

        if np.isnan(home_off).any(): home_off = model["fallback_off"]
        if np.isnan(away_def).any(): away_def = model["fallback_def"]
        if np.isnan(away_off).any(): away_off = model["fallback_off"]
        if np.isnan(home_def).any(): home_def = model["fallback_def"]

        n_drives_home = row["home_pregame_n_drives"]
        n_drives_away = row["away_pregame_n_drives"]
        n_drives_home = model["fallback_drives"] if pd.isna(n_drives_home) else n_drives_home
        n_drives_away = model["fallback_drives"] if pd.isna(n_drives_away) else n_drives_away

        home_preds.append(_simulate_side(home_off, away_def, n_drives_home))
        away_preds.append(_simulate_side(away_off, home_def, n_drives_away))
    return np.array(home_preds), np.array(away_preds)

def main():
    df = pd.read_parquet("data/processed/drive_model_table.parquet")
    results = walk_forward_evaluate(df, fit_montecarlo, predict_montecarlo, min_train_seasons=2)
    metrics = score_predictions(results)
    print("Monte Carlo Simulation walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/montecarlo_predictions.parquet", index=False)

if __name__ == "__main__":
    main()