"""
src/experiments/test_success_rate_definition.py

Which plays should a team's SUCCESS RATE be measured on?

EXPERIMENTAL. Each arm's walk-forward predictions are cached under
data/processed/experiments/success_rate/ so the slow Gaussian Process arms can
run as separate processes; nothing else is written. The winning arm SHIPPED on
2026-10-01 -- see "results" below.

--------------------------------------------------------------------- the defect

build_team_game_stats.py computed off_success_rate and def_success_rate_allowed
over EVERY play with a play_type -- kickoffs, punts, extra points, field goals,
kneel-downs, spikes and pre-snap penalties included. 23% of the plays behind the
two success-rate features (four of the 24 model inputs) were not offensive or
defensive snaps at all, and the special-teams ones are anything but neutral:

    play_type     share of rows   success rate
    extra_point        2.9%          94.4%     <- one per touchdown
    field_goal         2.4%          84.7%     <- most drives that stall in range
    kickoff            6.3%          36.1%     <- credited to the RECEIVING team
    punt               4.8%          41.4%
    qb_kneel           1.0%           0.5%     <- teams that are winning
    no_play            5.7%          39.1%     <- penalties, incl. pre-snap

So "offensive success rate" partly counted touchdowns (extra points) and field
goals -- a scoring proxy wearing an efficiency label, beside a points feature the
model already has -- and docked teams for running out the clock.

--------------------------------------------------------------------- the arms

  control        every play with a play_type: production as it was.
  scrimmage      play_type is "pass" or "run": the snaps that counted. Sacks are
                 passes and scrambles are runs, as nflfastR labels them. The same
                 filter the pass/rush EPA split already uses.
  dropback_rush  nflfastR's own convention, pass == 1 | rush == 1: the scrimmage
                 snaps PLUS the ~11,000 dropbacks and runs wiped out by a penalty
                 (offensive holding, defensive holding, illegal contact...). Those
                 are real outcomes of a snap and arguably part of a unit's
                 quality; pre-snap fouls (false starts, delay of game) and special
                 teams are still out.

Both fixes apply to offense AND defense. Every other feature is unchanged, and
the arm that IS production must rebuild production's model_table EXACTLY before
any number is believed (asserted below) -- a control arm that is not the thing
itself makes every delta measured against it meaningless (see rebuild.py).

--------------------------------------------------------------------- results

Run 2026-10-01 on 1,472 games (2021 wk1 - 2026 wk3). RMSE per team score,
pooled; delta against control with a 95% week-block bootstrap interval:

    model       control   scrimmage                       dropback_rush
    Ridge        9.3838   9.3746  -0.0092 [-0.023,+0.005]   9.3775  -0.0063
    Poisson      9.3730   9.3665  -0.0065 [-0.022,+0.009]   9.3691  -0.0039
    GP           9.3802   9.3698  -0.0104 [-0.025,+0.005]   9.3729  -0.0072
    composite    9.3711   9.3634  -0.0078 [-0.021,+0.006]   9.3664  -0.0048
                          P(composite better) 87%            76%

The composite is the published forecast. On it, scrimmage-only also moves margin
RMSE 13.039 -> 13.016, leaves total RMSE alone (13.464 -> 13.464), cuts the mean
bias 0.188 -> 0.123 and nudges the calibration slope 0.987 -> 0.990. By season:
2021 +0.026, 2022 -0.017, 2023 -0.006, 2024 -0.029, 2025 -0.020, 2026 +0.017
(47 games).

WHETHER to fix was never this experiment's question: the old rate measured
something other than its name, and broken logic is fixed whatever it does to
accuracy (README section 1; SKILL.md, "Correctness Before Accuracy"). The
experiment chose WHICH correct definition, by a rule fixed before the GP arms
finished: the composite first, then whether every member moves the same way,
then simplicity. scrimmage wins on all three and SHIPPED
(build_team_game_stats.scrimmage_snaps). No arm clears the bootstrap gate; none
had to. Penalty-nullified snaps (dropback_rush) were worse than leaving them out
on every model.

Two facts behind the result. The old rate WAS partly a scoring count: the part
of it that scrimmage success does not explain correlates 0.22 with the team's
points in that game. And it was the more persistent number for that reason --
odd vs even games within a team-season, r = 0.52 for every play against 0.41
for scrimmage snaps -- persistence borrowed from the scoring it double-counted.

--------------------------------------------------------------------- the test

Walk-forward with the production fit/predict functions (src/models/registry.py),
min_train_seasons=2, so 2021 onward is scored exactly as the backtests score it.
Paired bootstrap resampling WEEKS (src/validate/model_report.BlockBootstrap),
pooled home+away errors -- the project's standard. Ridge and Poisson are cheap;
the GP is ~20 minutes an arm and is what makes the published composite, so it is
run as separate processes:

    python -m src.experiments.test_success_rate_definition                # ridge + poisson, all arms
    python -m src.experiments.test_success_rate_definition --models gp --arms control &
    python -m src.experiments.test_success_rate_definition --models gp --arms scrimmage &
    python -m src.experiments.test_success_rate_definition --models gp --arms dropback_rush &
    python -m src.experiments.test_success_rate_definition --report       # everything cached
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_game_features import assemble
from src.features.build_rolling_features import add_pregame_rolling_features
from src.models import registry
from src.models.common import FEATURE_COLS
from src.validate.model_report import BlockBootstrap
from src.validate.walk_forward import walk_forward_evaluate

PBP = "data/raw/pbp.parquet"
SCHEDULES = "data/raw/schedules.parquet"
TEAM_GAME_STATS = "data/processed/team_game_stats.parquet"
MODEL_TABLE = "data/processed/model_table.parquet"
OUT = Path("data/processed/experiments/success_rate")

SR_COLS = ["off_success_rate", "def_success_rate_allowed"]
MODELS = ["linear", "poisson", "gp"]
N_BOOT = 5000
SEED = 42


def is_play(p: pd.DataFrame) -> pd.Series:
    """Rows that are plays at all -- production's filter before any fix."""
    return p["play_type"].notna()


ARMS = {
    "control": is_play,
    "scrimmage": lambda p: p["play_type"].isin(["pass", "run"]),
    "dropback_rush": lambda p: is_play(p) & ((p["pass"] == 1) | (p["rush"] == 1)),
}


def success_rates(pbp: pd.DataFrame, arm: str) -> pd.DataFrame:
    """off_success_rate / def_success_rate_allowed per (game_id, team) under an
    arm's play filter. Same grouping as build_team_game_stats.py."""
    keep = pbp[ARMS[arm](pbp)]
    off = (
        keep[keep["posteam"].notna()]
        .groupby(["game_id", "posteam"])["success"]
        .mean()
        .rename("off_success_rate")
        .rename_axis(["game_id", "team"])
    )
    deff = (
        keep[keep["defteam"].notna()]
        .groupby(["game_id", "defteam"])["success"]
        .mean()
        .rename("def_success_rate_allowed")
        .rename_axis(["game_id", "team"])
    )
    return pd.concat([off, deff], axis=1).reset_index()


def rolling_for(team_games: pd.DataFrame, halflife_days: float) -> pd.DataFrame:
    """build_rolling_features.main(), in memory."""
    pieces = [
        add_pregame_rolling_features(g.copy(), halflife_days)
        for _, g in team_games.groupby("team")
    ]
    return pd.concat(pieces, ignore_index=True)


def build_tables(arms) -> dict[str, pd.DataFrame]:
    import yaml

    with open("config.yaml") as f:
        halflife_days = yaml.safe_load(f)["training"]["recency_half_life_weeks"] * 7
    pbp = pd.read_parquet(
        PBP,
        columns=[
            "game_id",
            "posteam",
            "defteam",
            "play_type",
            "pass",
            "rush",
            "success",
        ],
    )
    tgs = pd.read_parquet(TEAM_GAME_STATS)
    tgs["gameday"] = pd.to_datetime(tgs["gameday"])
    inj = pd.read_parquet("data/processed/injury_features.parquet")
    split = pd.read_parquet("data/processed/split_efficiency.parquet")
    sched = pd.read_parquet(SCHEDULES)

    tables, production_arm = {}, None
    for arm in arms:
        sr = success_rates(pbp, arm)
        # Which arm is production? The one whose filter reproduces the success
        # rates production stored. ("control" until the 2026-10-01 fix shipped,
        # "scrimmage" since.) That arm must then rebuild model_table exactly.
        stored = tgs[["game_id", "team"] + SR_COLS].merge(
            sr, on=["game_id", "team"], how="left", suffixes=("", "_re")
        )
        if all(
            (
                np.isclose(stored[c], stored[f"{c}_re"], rtol=0, atol=1e-12)
                | (stored[c].isna() & stored[f"{c}_re"].isna())
            ).all()
            for c in SR_COLS
        ):
            assert production_arm is None, "two arms reproduce production"
            production_arm = arm
        tg = tgs.drop(columns=SR_COLS).merge(sr, on=["game_id", "team"], how="left")
        tg = tg[tgs.columns]  # production column order
        table = assemble(rolling_for(tg, halflife_days), inj, split, sched)
        table["gameday"] = pd.to_datetime(table["gameday"])
        tables[arm] = table

    if production_arm is not None:
        prod = pd.read_parquet(MODEL_TABLE)
        prod["gameday"] = pd.to_datetime(prod["gameday"])
        key = ["game_id"]
        a = tables[production_arm].sort_values(key).reset_index(drop=True)
        b = prod.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False)
        print(f"{production_arm} arm rebuilds production model_table exactly")
    elif set(arms) == set(ARMS):
        raise AssertionError("no arm reproduces production -- the harness is broken")
    return tables


def pred_path(arm: str, model: str) -> Path:
    return OUT / f"{arm}__{model}.parquet"


def run_arms(arms, models) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = build_tables(arms)
    for arm, table in tables.items():
        for model in models:
            fit, predict = registry.load_fns(model)
            print(f"\n== {arm} / {model}", flush=True)
            res = walk_forward_evaluate(table, fit, predict, min_train_seasons=2)
            res.to_parquet(pred_path(arm, model), index=False)


def _pooled(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    e = np.concatenate(
        [
            (df.home_pred - df.home_score).to_numpy(float),
            (df.away_pred - df.away_score).to_numpy(float),
        ]
    )
    b = (df["season"].astype(int) * 100 + df["week"].astype(int)).to_numpy()
    return e, np.concatenate([b, b])


def _rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def load_cached() -> dict[tuple[str, str], pd.DataFrame]:
    found = {}
    for arm in ARMS:
        for model in MODELS:
            p = pred_path(arm, model)
            if p.exists():
                found[(arm, model)] = (
                    pd.read_parquet(p).sort_values("game_id").reset_index(drop=True)
                )
    # The published forecast: the equal-weight composite of all three, per arm.
    for arm in ARMS:
        parts = [found.get((arm, m)) for m in MODELS]
        if all(p is not None for p in parts):
            comp = parts[0][
                ["game_id", "season", "week", "home_score", "away_score"]
            ].copy()
            for side in ("home", "away"):
                comp[f"{side}_pred"] = np.mean(
                    [p[f"{side}_pred"].to_numpy() for p in parts], axis=0
                )
            found[(arm, "composite")] = comp
    return found


def report() -> pd.DataFrame:
    found = load_cached()
    rows = []
    for (arm, model), df in sorted(found.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = found.get(("control", model))
        e, blocks = _pooled(df)
        margin_e = (df.home_pred - df.away_pred) - (df.home_score - df.away_score)
        total_e = (df.home_pred + df.away_pred) - (df.home_score + df.away_score)
        slope = np.polyfit(
            np.concatenate([df.home_pred, df.away_pred]),
            np.concatenate([df.home_score, df.away_score]),
            1,
        )[0]
        row = {
            "model": model,
            "arm": arm,
            "games": len(df),
            "rmse": _rmse(e),
            "margin_rmse": _rmse(margin_e),
            "total_rmse": _rmse(total_e),
            "bias": float(e.mean()),
            "calib_slope": float(slope),
        }
        if base is not None and arm != "control":
            assert (base["game_id"].to_numpy() == df["game_id"].to_numpy()).all()
            eb, _ = _pooled(base)
            d, lo, hi, p = BlockBootstrap(blocks, N_BOOT, SEED).rmse_delta(e, eb)
            row.update({"vs_control": d, "lo": lo, "hi": hi, "p_better": p})
            row["verdict"] = (
                "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            )
            per = []
            for season, g in df.groupby("season"):
                gb = base[base["season"] == season]
                per.append(
                    f"{season}:{_rmse(_pooled(g)[0]) - _rmse(_pooled(gb)[0]):+.3f}"
                )
            row["by_season"] = " ".join(per)
        rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(out.round(4).to_string(index=False))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--models", nargs="+", default=["linear", "poisson"], choices=MODELS
    )
    ap.add_argument("--report", action="store_true", help="only report cached arms")
    args = ap.parse_args(argv)
    if not args.report:
        run_arms(args.arms, args.models)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
