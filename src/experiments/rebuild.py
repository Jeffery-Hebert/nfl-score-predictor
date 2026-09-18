"""
src/experiments/rebuild.py

Builds a model table in memory under an arbitrary RecencyPlan, so recency
schemes can be compared without touching data/processed/.

EXPERIMENTAL. Writes nothing. Production's pipeline is untouched.

Same idea as tune_halflife.py, which rebuilds rolling features per candidate
half-life, extended in two ways it could not do:

  - the split-efficiency features are rebuilt too, volume-weighted and shrunk,
    so a recency change is applied to the WHOLE feature set rather than to the
    half of it that predates the split;
  - the kernel is pluggable, so per-stat rates and two-timescale mixtures are
    expressible rather than just a different scalar half-life.

THE CONTROL ARM IS PRODUCTION ITSELF. `production_table()` reads
model_table.parquet off disk rather than reconstructing it. An earlier version
of this used the kernel machinery with an Exponential as its own control, which
is wrong: production's finale masking uses pandas' ignore_na=True convention,
which differs from a clean weighted mean by ~0.012 on 88% of rows. Small, but a
control arm has to be the thing itself, or every delta measured against it
silently includes a reimplementation difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.recency import (
    RecencyPlan,
    prior_weighted_mean,
    prior_weighted_rate,
    shrink_to_prior,
)
from src.features.build_rolling_features import is_finale_week
from src.features.build_split_efficiency import SHRINKAGE_PLAYS, SPLITS
from src.models.common import (
    BASE_FEATURE_COLS,
    GAME_FEATURE_COLS,
    INJURY_FEATURE_COLS,
    SPLIT_FEATURE_COLS,
)

TEAM_GAME_STATS = "data/processed/team_game_stats.parquet"
INJURY_FEATURES = "data/processed/injury_features.parquet"
MODEL_TABLE = "data/processed/model_table.parquet"
SCHEDULES = "data/raw/schedules.parquet"

# The per-team stats BASE_FEATURE_COLS is built from, minus the two schedule
# columns that are not rolled at all.
ROLLED_STATS = [
    c[len("pregame_") :] for c in BASE_FEATURE_COLS if c.startswith("pregame_")
]
SCHEDULE_COLS = [c for c in BASE_FEATURE_COLS if not c.startswith("pregame_")]


def production_table() -> pd.DataFrame:
    """The control arm: exactly what production built, read off disk."""
    df = pd.read_parquet(MODEL_TABLE)
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def _team_rolling(team_games: pd.DataFrame, plan: RecencyPlan) -> pd.DataFrame:
    """Per-team pregame means for the plain rolled stats, under `plan`."""
    pieces = []
    for _, g in team_games.groupby("team"):
        g = g.sort_values("gameday").reset_index(drop=True)
        include = ~g.apply(
            lambda r: is_finale_week(r["season"], r["week"]), axis=1
        ).to_numpy()
        times = g["gameday"].to_numpy()
        out = g[
            ["game_id", "season", "week", "gameday", "team", "opponent", "is_home"]
            + [c for c in ("is_neutral_site", "is_playoff", "rest_days") if c in g]
        ].copy()
        for stat in ROLLED_STATS:
            out[f"pregame_{stat}"] = prior_weighted_mean(
                g[stat].to_numpy(float),
                times,
                plan.kernel_for(stat),
                include=include,
                anchor=plan.anchor,
            )
        out["prior_games_played"] = g.groupby("season").cumcount().to_numpy()
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _split_efficiency(team_games: pd.DataFrame, plan: RecencyPlan) -> pd.DataFrame:
    """Volume-weighted, shrunk pass/rush splits under `plan`.

    Mirrors build_split_efficiency: a league prior accumulated across every
    team, then per-team rates shrunk toward it by the measured constants. The
    only thing varying is the kernel.
    """
    df = team_games.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
    times = df["gameday"].to_numpy()
    include = ~df.apply(
        lambda r: is_finale_week(r["season"], r["week"]), axis=1
    ).to_numpy()
    teams = df["team"].to_numpy()

    out = df[["game_id", "team"]].copy()
    for key, (rate_col, count_col, out_col) in SPLITS.items():
        kernel = plan.kernel_for(rate_col)
        plays = np.nan_to_num(df[count_col].to_numpy(float), nan=0.0)
        epa_sum = np.nan_to_num(df[rate_col].to_numpy(float), nan=0.0) * plays

        league, _ = prior_weighted_rate(
            epa_sum, plays, times, kernel, include=include, anchor=plan.anchor
        )

        team_rate = np.full(len(df), np.nan)
        n_eff = np.zeros(len(df))
        for t in np.unique(teams):
            idx = np.flatnonzero(teams == t)
            r, n = prior_weighted_rate(
                epa_sum[idx],
                plays[idx],
                times[idx],
                kernel,
                include=include[idx],
                anchor=plan.anchor,
            )
            team_rate[idx] = r
            n_eff[idx] = n

        out[out_col] = shrink_to_prior(team_rate, n_eff, league, SHRINKAGE_PLAYS[key])
    return out


def build_table(plan: RecencyPlan) -> pd.DataFrame:
    """A full model table under `plan`, shaped exactly like production's.

    Same columns, same dtypes, same row count -- so it is a drop-in for anything
    that takes model_table.parquet, and a candidate arm differs from the control
    only in how its features were weighted.
    """
    tg = pd.read_parquet(TEAM_GAME_STATS)
    tg["gameday"] = pd.to_datetime(tg["gameday"])

    rolling = _team_rolling(tg, plan)
    splits = _split_efficiency(tg, plan)
    rolling = rolling.merge(splits, on=["game_id", "team"], how="left")

    sided = (
        [c for c in BASE_FEATURE_COLS if c != "prior_games_played"]
        + ["prior_games_played"]
        + SPLIT_FEATURE_COLS
    )

    home = rolling[rolling["is_home"] == 1][
        ["game_id", "team", "opponent", "is_neutral_site", "is_playoff"] + sided
    ].rename(columns={c: f"home_{c}" for c in sided})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})

    away = rolling[rolling["is_home"] == 0][["game_id", "team"] + sided].rename(
        columns={c: f"away_{c}" for c in sided}
    )
    away = away.rename(columns={"team": "away_team"})

    merged = home.merge(away, on=["game_id", "away_team"], how="inner")

    inj = pd.read_parquet(INJURY_FEATURES)
    for side in ("home", "away"):
        cols = inj[["game_id", "team"] + INJURY_FEATURE_COLS].rename(
            columns={
                **{c: f"{side}_{c}" for c in INJURY_FEATURE_COLS},
                "team": f"{side}_team",
            }
        )
        merged = merged.merge(cols, on=["game_id", f"{side}_team"], how="left")

    sched = pd.read_parquet(SCHEDULES)[
        ["game_id", "season", "week", "gameday", "home_score", "away_score", "overtime"]
    ]
    final = merged.merge(sched, on="game_id", how="left")
    final["went_to_ot"] = final["overtime"].fillna(0).astype(int)
    final = final.drop(columns=["overtime"])
    final["gameday"] = pd.to_datetime(final["gameday"])
    return final


def assert_shaped_like_production(candidate: pd.DataFrame, control: pd.DataFrame):
    """Fail loudly if a rebuilt table is not comparable to the control.

    A candidate with fewer rows, or a missing feature, would still produce a
    perfectly plausible RMSE -- against a different set of games. That is the
    class of bug this whole project keeps finding, so it is asserted rather than
    assumed.
    """
    from src.models.common import FEATURE_COLS

    missing = set(FEATURE_COLS) - set(candidate.columns)
    if missing:
        raise AssertionError(f"rebuilt table is missing features: {sorted(missing)}")
    if len(candidate) != len(control):
        raise AssertionError(
            f"rebuilt table has {len(candidate)} rows, control has {len(control)} -- "
            "the two would be scored on different games"
        )
    same_games = set(candidate["game_id"]) == set(control["game_id"])
    if not same_games:
        raise AssertionError("rebuilt table covers a different set of games")
    for col in ("home_score", "away_score"):
        a = candidate.set_index("game_id")[col].sort_index()
        b = control.set_index("game_id")[col].sort_index()
        if not a.equals(b):
            raise AssertionError(f"{col} differs between candidate and control")
