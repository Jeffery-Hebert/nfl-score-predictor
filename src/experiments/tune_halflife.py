"""
src/experiments/tune_halflife.py

C7: config.yaml has carried `recency_half_life_weeks: 17  # ~1 season; will be
tuned empirically on Day 3` since the project started. It was never tuned. 17
is the single most-used hyperparameter in the system -- it governs every
rolling feature, the drive features, the QB features and the adjusted-ratings
sample weights -- and it has never had a number behind it.

This sweeps it and reports out-of-sample accuracy per value, so the config
carries a measured choice instead of a guess. Evaluated on baseline, Linear and
Poisson only.

Rebuilds the rolling features in memory for each candidate; writes nothing to
data/processed.

Run: python -m src.experiments.tune_halflife
"""

import pandas as pd

from src.features.build_game_features import assemble
from src.features.build_rolling_features import add_pregame_rolling_features
from src.features.build_split_efficiency import build_split_efficiency
from src.models.baseline import fit_baseline, predict_baseline
from src.models.linear import fit_linear, predict_linear
from src.models.poisson_glm import fit_poisson, predict_poisson
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

HALFLIFE_WEEKS = [4, 8, 12, 17, 26, 39, 52]


def build_model_table(
    team_games: pd.DataFrame, schedules: pd.DataFrame, halflife_days: float
):
    """The full production feature table rebuilt at a candidate half-life.

    Re-pointed 2026-09-24 at the production assembler: this used to rebuild only
    the base rolling columns, and crashed with a KeyError once injury_impact and
    the pass/rush split joined FEATURE_COLS. The half-life now varies EVERY
    recency-weighted feature it governs in production -- the rolling stats and
    the split efficiencies -- while injury_impact (not recency-weighted) is read
    as built.
    """
    rolling = pd.concat(
        [
            add_pregame_rolling_features(g.copy(), halflife_days)
            for _, g in team_games.groupby("team")
        ],
        ignore_index=True,
    )
    split = build_split_efficiency(team_games, halflife_days)
    injuries = pd.read_parquet("data/processed/injury_features.parquet")
    return assemble(rolling, injuries, split, schedules)


def linear_fns():
    """The production estimator (RidgeCV), not the unregularised OLS this
    script originally used."""
    return fit_linear, predict_linear


def poisson_fns():
    return fit_poisson, predict_poisson


def main():
    team_games = pd.read_parquet("data/processed/team_game_stats.parquet")
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    schedules = pd.read_parquet("data/raw/schedules.parquet")

    print(f"Sweeping recency_half_life_weeks over {HALFLIFE_WEEKS}")
    print("Evaluated on baseline, Linear and Poisson. Lower mean RMSE is better.\n")
    print(
        f"{'halflife':>9}{'baseline':>11}{'linear':>10}{'poisson':>10}   (mean of home/away RMSE)"
    )
    print("-" * 56)

    rows = []
    for weeks in HALFLIFE_WEEKS:
        df = build_model_table(team_games, schedules, weeks * 7)
        out = {"halflife_weeks": weeks}
        for name, fns in [
            ("baseline", None),
            ("linear", linear_fns),
            ("poisson", poisson_fns),
        ]:
            if name == "baseline":
                fit, predict = fit_baseline, predict_baseline
            else:
                fit, predict = fns()
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            m = score_predictions(res)
            out[name] = (m["home_rmse"] + m["away_rmse"]) / 2
        rows.append(out)
        print(
            f"{weeks:>9}{out['baseline']:>11.4f}{out['linear']:>10.4f}{out['poisson']:>10.4f}"
        )

    table = pd.DataFrame(rows)
    print("\nBest halflife by model:")
    for model in ["baseline", "linear", "poisson"]:
        best = table.loc[table[model].idxmin()]
        cur = table[table.halflife_weeks == 17][model].iloc[0]
        print(
            f"  {model:<9} {int(best['halflife_weeks']):>3} weeks "
            f"({best[model]:.4f})   vs 17 weeks ({cur:.4f})   delta={best[model]-cur:+.4f}"
        )

    table["combined"] = table[["linear", "poisson"]].mean(axis=1)
    best = table.loc[table["combined"].idxmin()]
    print(f"\nBest on Linear+Poisson combined: {int(best['halflife_weeks'])} weeks")
    print("Set config.yaml training.recency_half_life_weeks to that value.")


if __name__ == "__main__":
    main()
