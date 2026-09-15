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

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.features.build_rolling_features import add_pregame_rolling_features
from src.models.baseline import fit_baseline, predict_baseline
from src.models.common import BASE_FEATURE_COLS, FEATURE_COLS, ot_sample_weight
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

HALFLIFE_WEEKS = [4, 8, 12, 17, 26, 39, 52]


def build_model_table(
    team_games: pd.DataFrame, schedules: pd.DataFrame, halflife_days: float
):
    """In-memory equivalent of build_rolling_features -> build_game_features."""
    pieces = [
        add_pregame_rolling_features(g.copy(), halflife_days)
        for _, g in team_games.groupby("team")
    ]
    rolling = pd.concat(pieces, ignore_index=True)

    home = rolling[rolling["is_home"] == 1][
        ["game_id", "team", "opponent", "is_neutral_site", "is_playoff"]
        + BASE_FEATURE_COLS
    ].rename(columns={c: f"home_{c}" for c in BASE_FEATURE_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][
        ["game_id", "team"] + BASE_FEATURE_COLS
    ].rename(columns={c: f"away_{c}" for c in BASE_FEATURE_COLS})
    away = away.rename(columns={"team": "away_team"})

    merged = home.merge(away, on=["game_id", "away_team"], how="inner")
    final = merged.merge(
        schedules[
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "home_score",
                "away_score",
                "overtime",
            ]
        ],
        on="game_id",
        how="left",
    )
    final["went_to_ot"] = final["overtime"].fillna(0).astype(int)
    return final.drop(columns=["overtime"])


def linear_fns():
    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        return {
            "h": LinearRegression().fit(
                X.fillna(means),
                train["home_score"],
                sample_weight=ot_sample_weight(train),
            ),
            "a": LinearRegression().fit(
                X.fillna(means),
                train["away_score"],
                sample_weight=ot_sample_weight(train),
            ),
            "means": means,
        }

    def predict(m, test):
        X = test[FEATURE_COLS].fillna(m["means"])
        return m["h"].predict(X), m["a"].predict(X)

    return fit, predict


def poisson_fns():
    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        sc = StandardScaler().fit(Xf)
        Xs = sc.transform(Xf)
        w = ot_sample_weight(train)
        return {
            "h": PoissonRegressor(max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            ),
            "a": PoissonRegressor(max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            ),
            "means": means,
            "scaler": sc,
        }

    def predict(m, test):
        Xs = m["scaler"].transform(test[FEATURE_COLS].fillna(m["means"]))
        return m["h"].predict(Xs), m["a"].predict(Xs)

    return fit, predict


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
