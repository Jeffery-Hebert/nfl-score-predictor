"""
src/features/build_adjusted_ratings.py

Computes opponent-adjusted offensive/defensive EPA ratings via joint ridge
regression across the league, refit at each week using only strictly prior
games. Fixes two problems in the raw team-level EWM features: (1) they
ignore opponent strength entirely, and (2) additive models (Linear, Poisson)
can't learn matchup interactions from separate offense/defense numbers alone.

Same family of technique as Simple Rating System / Massey ratings: solve
for every team's offense and defense rating jointly, so a team's rating
reflects performance relative to the opponents actually faced.

Run: python src/features/build_adjusted_ratings
Output: data/processed/adjusted_ratings.parquet
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
import yaml

# Regularization strength for the joint offense/defense ridge.
# Validated 2026-09-14 by src/experiments/tune_ridge_alpha.py -- see that
# script for the sweep. Previously an unvalidated starting guess.
RIDGE_ALPHA = 5.0


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_design_matrix(games: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    off_dummies = pd.get_dummies(games["team"], prefix="off").astype(float)
    off_dummies = off_dummies.reindex(
        columns=[f"off_{t}" for t in teams], fill_value=0.0
    )

    def_dummies = pd.get_dummies(games["opponent"], prefix="def").astype(float)
    def_dummies = def_dummies.reindex(
        columns=[f"def_{t}" for t in teams], fill_value=0.0
    )

    home_col = games[["is_home"]].astype(float).reset_index(drop=True)
    X = pd.concat(
        [
            off_dummies.reset_index(drop=True),
            def_dummies.reset_index(drop=True),
            home_col,
        ],
        axis=1,
    )
    return X


def fit_ratings(
    train_games: pd.DataFrame,
    teams: list[str],
    halflife_days: float,
    cutoff_date,
    alpha: float = RIDGE_ALPHA,
):
    X = build_design_matrix(train_games, teams)
    y = train_games["off_epa_per_play"].reset_index(drop=True)

    age_days = (cutoff_date - train_games["gameday"]).dt.days.reset_index(drop=True)
    weights = 0.5 ** (age_days / halflife_days)

    valid = y.notna()
    model = Ridge(alpha=alpha)
    model.fit(X[valid], y[valid], sample_weight=weights[valid])

    off_ratings = {t: model.coef_[X.columns.get_loc(f"off_{t}")] for t in teams}
    def_ratings = {t: model.coef_[X.columns.get_loc(f"def_{t}")] for t in teams}
    return off_ratings, def_ratings


def main():
    cfg = load_config()
    halflife_days = cfg["training"]["recency_half_life_weeks"] * 7

    team_games = pd.read_parquet("data/processed/team_game_stats.parquet")
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    teams = sorted(team_games["team"].unique())

    week_cutoffs = (
        team_games[["season", "week"]]
        .drop_duplicates()
        .merge(
            team_games.groupby(["season", "week"])["gameday"].min().reset_index(),
            on=["season", "week"],
        )
        .sort_values("gameday")
    )

    rows = []
    for _, cutoff_row in week_cutoffs.iterrows():
        cutoff_date = cutoff_row["gameday"]
        train_games = team_games[team_games["gameday"] < cutoff_date]
        if len(train_games) < 100:
            continue

        off_ratings, def_ratings = fit_ratings(
            train_games, teams, halflife_days, cutoff_date
        )
        for t in teams:
            rows.append(
                {
                    "season": cutoff_row["season"],
                    "week": cutoff_row["week"],
                    "team": t,
                    "pregame_adjusted_off_epa": off_ratings[t],
                    "pregame_adjusted_def_epa_allowed": def_ratings[t],
                }
            )

    result = pd.DataFrame(rows)
    out_path = Path("data/processed/adjusted_ratings.parquet")
    result.to_parquet(out_path, index=False)
    print(
        f"Built adjusted ratings: {len(teams)} teams x "
        f"{result[['season', 'week']].drop_duplicates().shape[0]} week-cutoffs"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
