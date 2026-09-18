"""
src/experiments/adjusted_ratings_v2.py

Opponent-adjusted efficiency ratings, rebuilt.

EXPERIMENTAL. Writes nothing; production is untouched.

--------------------------------------------------------------- what v1 did

`src/features/build_adjusted_ratings.py` solves, at every week cutoff, a single
joint ridge over prior games:

    off_epa_per_play ~ off_team + def_opponent + is_home

recency-weighted, alpha=5.0. It is a Massey/SRS-style simultaneous solve and the
technique is sound. It was tested on Linear, Poisson, RF and XGBoost and found
NO confirmed improvement on any of them, so it is a closed null result and not
part of build_all.

------------------------------------------------------- why it is worth reviving

Four things are different now, and the first is the one that matters.

  1. IT ADJUSTED A BLEND. v1 solved for one number per team per side, built from
     `off_epa_per_play` -- all plays pooled. That throws away the distinction the
     whole adjustment exists to capture. A defence that is stout against the run
     and porous against the pass has ONE rating in v1, and a pass-heavy offence
     facing it is adjusted by the average of two things, only one of which it
     will meet. Since 2026-09-17 production carries pass and rush separately, so
     the ratings can too: four per team instead of two.

  2. EVERY GAME COUNTED EQUALLY. The response was a per-game mean and rows were
     unweighted, so a game in which a team threw 15 times contributed as much
     evidence about its passing as one in which it threw 50. Weighting each row
     by its play count is both obviously right and the same correction that
     turned the pass/rush split from a regression into a small gain.

  3. ONE ALPHA FOR EVERYONE. Ridge shrinks all team effects toward zero at the
     same rate regardless of how much evidence stands behind each. Weighting by
     plays fixes most of this implicitly -- a team with few observed plays gets
     less leverage and is pulled harder toward the league mean by the same
     penalty -- which is a cleaner mechanism than a per-team alpha would be.

  4. IT WAS TESTED AS A REPLACEMENT. v1 substituted its ratings for raw EPA.
     Here they are ADDITIONAL, so the question becomes "is the opponent
     adjustment worth anything beyond the shrunk raw rates" rather than "is
     adjusted better than raw".

------------------------------------------------------------------- leakage

Identical to v1 and to the rest of the project: at each week cutoff the fit sees
only games that kicked off STRICTLY BEFORE the first game of that week, and a
team's rating for week W is the one solved at W's cutoff. The interesting part
is that a rating is a LEAGUE-WIDE quantity -- it depends on every other team's
games too -- so the cutoff has to be global rather than per-team. That is what
makes it a different leakage shape from the rolling features, and what
tests/test_adjusted_ratings_v2.py checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TEAM_GAME_STATS = "data/processed/team_game_stats.parquet"

# Validated for v1 by src/experiments/tune_ridge_alpha.py. Carried over rather
# than re-tuned: the design matrix is the same shape and re-tuning on this
# project's test set is the error walk-forward exists to prevent.
RIDGE_ALPHA = 5.0
HALFLIFE_DAYS = 17 * 7

# unit -> (per-play rate column, play-count column)
UNITS = {
    "pass": ("off_pass_epa_per_play", "off_pass_plays"),
    "rush": ("off_rush_epa_per_play", "off_rush_plays"),
}

RATING_COLS = [
    "adj_off_pass",
    "adj_off_rush",
    "adj_def_pass_allowed",
    "adj_def_rush_allowed",
]

# A cutoff with less than this much prior evidence cannot identify 64 team
# effects. v1 used 100 team-games; the same floor applies per unit here.
MIN_TRAIN_ROWS = 100


def _design(games: pd.DataFrame, teams: list[str]) -> np.ndarray:
    """[off dummies | def dummies | home] as a dense float array.

    Built with numpy indexing rather than get_dummies so that a team absent from
    a given window still gets its column -- otherwise the coefficient vector
    changes shape between cutoffs and the ratings stop being comparable across
    weeks, which is the kind of bug that produces a plausible-looking table of
    nonsense.
    """
    index = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)
    X = np.zeros((n, 2 * k + 1))
    X[np.arange(n), [index[t] for t in games["team"]]] = 1.0
    X[np.arange(n), [k + index[t] for t in games["opponent"]]] = 1.0
    X[:, -1] = games["is_home"].to_numpy(float)
    return X


def fit_unit_ratings(
    train: pd.DataFrame,
    teams: list[str],
    rate_col: str,
    count_col: str,
    cutoff_date,
    halflife_days: float = HALFLIFE_DAYS,
    alpha: float = RIDGE_ALPHA,
    min_rows: int = MIN_TRAIN_ROWS,
):
    """Solve one unit (pass or rush) jointly across the league.

    Returns (offense_ratings, defense_allowed_ratings), both dicts keyed by team.

    The response is the per-play rate and the weight is recency x PLAY COUNT, so
    the fit is effectively over plays rather than over games. That is the v2
    correction: a 50-attempt game is fifty plays of evidence, not one game of it.
    """
    y = train[rate_col].to_numpy(float)
    plays = np.nan_to_num(train[count_col].to_numpy(float), nan=0.0)
    age = (cutoff_date - train["gameday"]).dt.days.to_numpy(float)
    w = (0.5 ** (age / halflife_days)) * plays

    valid = np.isfinite(y) & (w > 0)
    if valid.sum() < min_rows:
        return None, None

    X = _design(train, teams)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X[valid], y[valid], sample_weight=w[valid])

    k = len(teams)
    off = {t: float(model.coef_[i]) for i, t in enumerate(teams)}
    # A defensive coefficient is "EPA the opponent gained against this team", so
    # higher is WORSE. Left in that orientation deliberately: it matches
    # def_*_epa_allowed elsewhere in the project, and flipping the sign here
    # while leaving the name alone is how sign bugs are born.
    deff = {t: float(model.coef_[k + i]) for i, t in enumerate(teams)}
    return off, deff


def build_adjusted_ratings(team_games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, week, team) with four opponent-adjusted ratings."""
    tg = pd.read_parquet(TEAM_GAME_STATS) if team_games is None else team_games.copy()
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    teams = sorted(tg["team"].dropna().unique())

    cutoffs = (
        tg.groupby(["season", "week"])["gameday"]
        .min()
        .reset_index()
        .sort_values("gameday")
    )

    rows = []
    for _, c in cutoffs.iterrows():
        # STRICTLY before the first kickoff of this week, league-wide.
        train = tg[tg["gameday"] < c["gameday"]]
        if len(train) < MIN_TRAIN_ROWS:
            continue

        solved = {}
        for unit, (rate_col, count_col) in UNITS.items():
            off, deff = fit_unit_ratings(
                train, teams, rate_col, count_col, c["gameday"]
            )
            if off is None:
                break
            solved[unit] = (off, deff)
        if len(solved) != len(UNITS):
            continue

        for t in teams:
            rows.append(
                {
                    "season": int(c["season"]),
                    "week": int(c["week"]),
                    "team": t,
                    "adj_off_pass": solved["pass"][0][t],
                    "adj_off_rush": solved["rush"][0][t],
                    "adj_def_pass_allowed": solved["pass"][1][t],
                    "adj_def_rush_allowed": solved["rush"][1][t],
                }
            )
    return pd.DataFrame(rows)


def add_adjusted_ratings(df: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Join ratings onto a game table as home_/away_ columns."""
    out = df.copy()
    for side in ("home", "away"):
        cols = ratings.rename(
            columns={
                **{c: f"{side}_{c}" for c in RATING_COLS},
                "team": f"{side}_team",
            }
        )
        out = out.merge(cols, on=["season", "week", f"{side}_team"], how="left")
    return out


def main():
    r = build_adjusted_ratings()
    wk = r[["season", "week"]].drop_duplicates()
    print(
        f"Built {len(r)} rows: {r['team'].nunique()} teams x {len(wk)} week-cutoffs\n"
    )
    for c in RATING_COLS:
        v = r[c]
        print(
            f"  {c:22s} mean {v.mean():+.4f}  sd {v.std():.4f}  "
            f"range [{v.min():+.3f}, {v.max():+.3f}]"
        )
    latest = r[(r.season == r.season.max())]
    latest = latest[latest.week == latest.week.max()]
    print(
        f"\nTop 5 adjusted pass offences, {int(latest.season.iloc[0])} "
        f"wk{int(latest.week.iloc[0])} (sanity check -- these should be "
        f"recognisable):"
    )
    for _, x in latest.nlargest(5, "adj_off_pass").iterrows():
        print(f"    {x['team']:4s} {x['adj_off_pass']:+.4f}")
    print("Best 5 adjusted pass defences (most negative = fewest EPA allowed):")
    for _, x in latest.nsmallest(5, "adj_def_pass_allowed").iterrows():
        print(f"    {x['team']:4s} {x['adj_def_pass_allowed']:+.4f}")


if __name__ == "__main__":
    main()
