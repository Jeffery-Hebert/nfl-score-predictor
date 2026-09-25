"""
src/predict/predict_week.py

Produces score predictions for games that have NOT been played yet. This is the
piece that turns a backtesting project into something that can answer "what
will happen on Sunday".

How it differs from the model scripts. Those run walk-forward: refit for every
historical week and score the result. This fits ONCE on every completed game
and predicts forward, with the same fit/predict functions -- and, since
2026-09-24, with its training rows in the same kickoff order the backtest uses.
Before that the live fit received model_table.parquet in its stored order
(grouped by home team), RidgeCV's time-series folds became team blocks, and the
live Linear model was not the one the backtest had validated (up to 0.7 points
apart). src/models/common.py::chronological, and tests/test_live_fit_order.py.

WHICH models. config.yaml `live.models`, plus the rule-based baseline as a
reference, plus the composite (`live.composite`): an equal-weight average of
its members computed from their UNROUNDED forecasts. Names are checked against
src/models/registry.py before anything is fitted.

WHEN. A game is only forecast once its FINAL injury report is in the data
(src/predict/injury_readiness.py): the Wednesday report for a Thursday game,
Friday's for Sunday, Saturday's for Monday. Earlier, injury_impact -- the
model's one leading indicator -- is near zero for everybody and the forecast is
built on inputs the backtest never saw. So it runs per slate, re-pulling data
each time: .github/workflows/pipeline.yml runs python -m src.weekly daily, and
each run forecasts whatever has just become ready. --allow-unsettled-injuries
overrides this, and every forecast records whether its report was final.

Training window. Every completed game in model_table.parquet before the first
game being forecast. The completeness guard below refuses to run if a finished
game is missing from the table.

Records. One parquet per week in data/predictions/, tracked in git. A game that
has kicked off is never re-forecast; a game not forecast in this run keeps the
forecast it already has. Each row carries its own generated_at, the models and
composite members that produced it, and the code version. A run that
reproduces the record's forecasts exactly leaves the file untouched, so
re-running (the scheduled pipeline runs daily) never makes a forecast look
younger than it is or commits an identical record.

Market lines. spread_line and total_line are printed ALONGSIDE the predictions
and are NEVER inputs. They are converted into the score pair the market
implies, so the two can be read on the same scale:

    implied home = (total + spread) / 2
    implied away = (total - spread) / 2

Rounding. Stored and displayed to one decimal place. A tenth of a point is
already far finer than the model can resolve -- typical error is over nine
points.

Run: python -m src.predict.predict_week                  # week of the next kickoff
     python -m src.predict.predict_week --season 2026 --week 3
     python -m src.predict.predict_week --dry-run        # plan only: what is ready
     python -m src.predict.predict_week --html           # + rebuild the ledger page
     python -m src.predict.predict_week --allow-unsettled-injuries
Output: data/predictions/<season>_wk<week>.parquet
        data/predictions/index.html   (with --html; see build_report.py)
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import live_settings
from src.models import registry
from src.models.common import chronological
from src.predict import injury_readiness
from src.provenance import git_state
from src.schedule import kickoff_by_game

MODEL_TABLE = Path("data/processed/model_table.parquet")
SCHEDULES = Path("data/raw/schedules.parquet")
INJURIES = Path("data/raw/injuries.parquet")

# One decimal place. Typical error is over nine points, so anything finer is
# noise wearing the costume of precision.
DP = 1
OUT_DIR = Path("data/predictions")
REFERENCE_MODELS = ["baseline"]  # always fitted, never averaged


def load_table() -> pd.DataFrame:
    if not MODEL_TABLE.exists():
        sys.exit(f"ERROR: {MODEL_TABLE} missing. Run: python -m src.features.build_all")
    df = pd.read_parquet(MODEL_TABLE)
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def pick_week(df: pd.DataFrame, season, week, now=None):
    """The week to forecast: as given, or else the week of the next game that
    has not kicked off. None when no game is left to kick off (the offseason).

    Not "the earliest week with an unplayed game", which the first version
    used: a game that has kicked off but has no score yet -- Monday night's, on
    Tuesday morning, before the result is published -- pinned the default to a
    week with nothing left to forecast, and the run failed."""
    if (season is None) != (week is None):
        sys.exit("ERROR: give both --season and --week, or neither.")
    if season is not None:
        return int(season), int(week)
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    unplayed = df[df["home_score"].isna()]
    if unplayed.empty:
        return None
    kickoff = pd.to_datetime(
        unplayed["game_id"].map(kickoff_utc(unplayed["game_id"])), utc=True
    )
    today_et = now.tz_convert("US/Eastern").tz_localize(None).normalize()
    # A game with no known kickoff counts as upcoming only from its date on.
    ahead = (kickoff > now) | (kickoff.isna() & (unplayed["gameday"] >= today_et))
    upcoming = unplayed[ahead]
    if upcoming.empty:
        return None
    row = upcoming.sort_values("gameday").iloc[0]
    return int(row["season"]), int(row["week"])


def select_pending(target: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Games of the week that have not kicked off. A game with no known kickoff
    is treated as pending -- refusing it would be worse than forecasting it."""
    return target[target["kickoff"].isna() | (target["kickoff"] > now)]


def training_cutoff(forecast: pd.DataFrame) -> pd.Timestamp:
    """Train on games strictly before the first game being FORECAST -- not the
    first game of the week. A Sunday run can use Thursday's result, a Monday
    run the whole weekend's; pinned to the week, both would throw them away."""
    return forecast["gameday"].min()


def assert_training_is_current(train: pd.DataFrame, kickoff: pd.Timestamp) -> int:
    """Refuse to predict from stale data.

    The failure this prevents is quiet and expensive: predicting Week N from a
    table that never received Week N-1, which produces confident, wrong numbers
    and no error message.

    This used to be a calendar test -- refuse if the newest training game was
    more than 21 days old. That was wrong, and wrong in a way that would have
    bitten on the single most important week of the year: every Week 1 sits
    roughly 210 days after the previous Super Bowl, so a fully up-to-date table
    looks 200 days stale and the guard would have blocked a live Week 1
    prediction outright.

    Staleness is a completeness question, not an elapsed-time one, so ask it
    that way. Two things can go wrong, and they need different fixes:

      1. the raw pull is behind    -- schedules has no score for a game that
                                      has already been played; re-pull.
      2. the feature build is behind -- schedules has the score but the model
                                      table does not; rebuild.
    """
    sched = pd.read_parquet(SCHEDULES)
    sched["gameday"] = pd.to_datetime(sched["gameday"])

    # (2) results that exist upstream but never reached the model table.
    completed_before = sched[(sched["gameday"] < kickoff) & sched["home_score"].notna()]
    missing = set(completed_before["game_id"]) - set(train["game_id"])
    if missing:
        sample = ", ".join(sorted(missing)[:4])
        sys.exit(
            f"ERROR: {len(missing)} completed games are in schedules but not in the "
            f"training table (e.g. {sample}).\n"
            "The model would be predicting without results it could have had. Rebuild:\n"
            "  python -m src.features.build_all"
        )

    # (1) games that have already kicked off and still have no score. Scoped to
    # a recent window: a game cancelled years ago (2022 BUF@CIN) never gets a
    # score and must not trip this forever.
    horizon = min(kickoff, pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    recent = sched[sched["gameday"].between(horizon - pd.Timedelta(days=45), horizon)]
    unscored = recent[recent["home_score"].isna()]
    if len(unscored):
        sys.exit(
            f"ERROR: {len(unscored)} games kicked off on or before "
            f"{horizon.date()} and still have no final score.\n"
            "The raw data is behind. Refresh, then rebuild:\n"
            "  python -m src.ingest.pull_all\n"
            "  python -m src.features.build_all"
        )

    return (kickoff - train["gameday"].max()).days


def kickoff_utc(game_ids) -> pd.Series:
    """Actual kickoff instant per game, indexed by game_id (see src/schedule.py)."""
    return kickoff_by_game(pd.read_parquet(SCHEDULES), game_ids)


def freeze_started_games(fresh: pd.DataFrame, path: Path, now) -> pd.DataFrame:
    """Merge this run's forecasts into the week's existing record.

    Two rules, both about never losing or rewriting a pre-kickoff forecast:

      1. a game that has KICKED OFF keeps the forecast it had, verbatim,
         including its original generated_at -- rewriting it afterwards would
         leave a file claiming to have predicted a game whose result it had
         already seen, the one thing these records exist to rule out;
      2. a game NOT forecast in this run keeps the forecast it had. A run can
         now deliberately skip games whose final injury report is not out yet,
         and an earlier forecast of such a game is still a genuine pre-kickoff
         record. (The first version kept only rule 1, so it would have dropped
         those rows.)

    Only games in `fresh` that have not started are replaced.
    """
    if not path.exists():
        return fresh

    prior = pd.read_parquet(path)
    if "kickoff" not in prior.columns:
        # Written before this column existed; recover it rather than discarding
        # the record.
        prior["kickoff"] = prior["game_id"].map(kickoff_utc(prior["game_id"]))
    prior["kickoff"] = pd.to_datetime(prior["kickoff"], utc=True)

    started = prior["kickoff"].notna() & (prior["kickoff"] <= now)
    not_refreshed = ~prior["game_id"].isin(fresh["game_id"])
    keep = prior[started | not_refreshed]
    if keep.empty:
        return fresh

    kept = fresh[~fresh["game_id"].isin(keep["game_id"])]
    n_started = int((started & prior["game_id"].isin(keep["game_id"])).sum())
    print(
        f"  preserving {n_started} already-started and "
        f"{len(keep) - n_started} not-re-forecast game(s) from the existing file; "
        f"writing {len(kept)} new forecast(s)"
    )
    merged = pd.concat([keep, kept], ignore_index=True)
    # Union the columns so a schema change between runs cannot drop a record.
    return merged.reindex(
        columns=list(dict.fromkeys(list(fresh.columns) + list(prior.columns)))
    ).sort_values(["kickoff", "game_id"], na_position="last")


# Columns that differ between runs without the forecast differing: when the
# run happened, from which commit, and the betting line at that moment (which
# is reference only, never an input).
RUN_METADATA = [
    "generated_at",
    "code_version",
    "spread_line",
    "total_line",
    "market_home",
    "market_away",
]


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted by game, datetimes at one resolution: so two records compare by
    what they say, not by how pandas happened to store it."""
    df = df.sort_values("game_id").reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
        elif pd.api.types.is_datetime64_dtype(df[c]):
            df[c] = df[c].astype("datetime64[ns]")
    return df


def unchanged(path: Path, record: pd.DataFrame) -> bool:
    """True when `record` holds exactly the forecasts already on disk, apart
    from RUN_METADATA.

    Rewriting the file then would only move generated_at later -- making each
    forecast look younger than it is -- and the scheduled pipeline would commit
    an identical record several times a week. Compared as STORED (through a
    parquet round trip), so an in-memory dtype is not mistaken for a change."""
    if not path.exists():
        return False
    buf = io.BytesIO()
    record.to_parquet(buf, index=False)
    new = pd.read_parquet(io.BytesIO(buf.getvalue()))
    old = pd.read_parquet(path)
    if set(new.columns) != set(old.columns) or len(new) != len(old):
        return False
    cols = [c for c in new.columns if c not in RUN_METADATA]
    try:
        pd.testing.assert_frame_equal(
            _canonical(new[cols]),
            _canonical(old[cols]),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def market_reference(game_ids) -> pd.DataFrame:
    """Closing spread/total and the score pair they imply. Reference only."""
    s = pd.read_parquet(SCHEDULES)
    s = s[s["game_id"].isin(game_ids)][
        ["game_id", "gametime", "spread_line", "total_line"]
    ].copy()
    # spread_line is the home margin the market expects (positive = home favoured)
    s["market_home"] = ((s["total_line"] + s["spread_line"]) / 2).round(1)
    s["market_away"] = ((s["total_line"] - s["spread_line"]) / 2).round(1)
    return s


def rows_for(table: pd.DataFrame, game_ids) -> pd.DataFrame:
    """The table's rows for exactly these games, in exactly this order -- a
    model predicting from a different table must still line up row for row."""
    idx = table.set_index("game_id")
    missing = [g for g in game_ids if g not in idx.index]
    if missing:
        raise KeyError(f"{len(missing)} games absent from the table: {missing[:3]}")
    return idx.loc[list(game_ids)].reset_index()


def training_frame(table: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Completed games strictly before `cutoff`, in kickoff order."""
    return chronological(
        table[table["gameday"] < cutoff].dropna(subset=["home_score", "away_score"])
    )


def fit_and_predict(model_names, tables, trains, cutoff, forecast_ids, skip=()):
    """Fit each model on its table's completed games before `cutoff` and
    forecast `forecast_ids`. Returns {name: (home, away)} UNROUNDED. `tables`
    and `trains` are caches keyed by table kind, filled in as needed."""
    raw = {}
    for name in model_names:
        if name in skip:
            continue
        spec = registry.get(name)
        if spec.table not in tables:
            tables[spec.table] = registry.load_table(spec.table)
        table = tables[spec.table]
        if spec.table not in trains:
            trains[spec.table] = training_frame(table, cutoff)
        fit_fn, predict_fn = registry.load_fns(name)
        print(f"  fitting {name}...", flush=True)
        model = fit_fn(trains[spec.table])
        h, a = predict_fn(model, rows_for(table, forecast_ids))
        raw[name] = (np.asarray(h, dtype=float), np.asarray(a, dtype=float))
    return raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--json", type=str, default=None, help="also write JSON here")
    ap.add_argument(
        "--html",
        action="store_true",
        help="also rebuild data/predictions/index.html (every week, graded)",
    )
    ap.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="live models to leave out of this run (e.g. gp, which is the slow one)",
    )
    ap.add_argument("--skip-gp", action="store_true", help="same as --skip gp")
    ap.add_argument(
        "--allow-unsettled-injuries",
        action="store_true",
        help="also forecast games whose FINAL injury report is not in the data "
        "yet; each such row is recorded with injury_report_final = False",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="show what is ready; fit nothing"
    )
    args = ap.parse_args(argv)
    skip = set(args.skip) | ({"gp"} if args.skip_gp else set())

    live = live_settings()
    comp = live["composite"]
    model_names = REFERENCE_MODELS + [
        m for m in live["models"] if m not in REFERENCE_MODELS
    ]
    if skip & set(comp["members"]):
        print(
            f"NOTE: skipping {sorted(skip & set(comp['members']))} leaves the "
            f"composite with {len(set(comp['members']) - skip)} of its members"
        )

    df = load_table()
    picked = pick_week(df, args.season, args.week)
    if picked is None:
        print(
            "No game in the table has yet to kick off: the season is over, or the "
            "schedule pull is out of date (python -m src.ingest.pull_all). "
            "Nothing to forecast."
        )
        return 0
    season, week = picked
    target = df[(df["season"] == season) & (df["week"] == week)].copy()
    if target.empty:
        sys.exit(f"ERROR: no games found for season {season} week {week}.")

    now = pd.Timestamp.now(tz="UTC")
    target["kickoff"] = target["game_id"].map(kickoff_utc(target["game_id"]))
    pending = select_pending(target, now)
    if pending.empty:
        sys.exit(
            f"ERROR: every game in season {season} week {week} has already "
            "kicked off. There is nothing left to forecast, and writing one now "
            "would be a backfill, not a prediction.\n"
            "Grade what is already on file instead:\n"
            "  python -m src.predict.build_report"
        )

    if not INJURIES.exists():
        sys.exit(
            f"ERROR: {INJURIES} missing -- injury readiness cannot be judged.\n"
            "Pull it first:  python -m src.ingest.pull_all"
        )
    ready = injury_readiness.assess(
        pending, pd.read_parquet(INJURIES), injury_readiness.injuries_pulled_at()
    )
    pending = pending.merge(ready, on="game_id", how="left")
    settled = pending[pending["injury_report_final"]]
    unsettled = pending[~pending["injury_report_final"]]

    print(f"Season {season}, week {week}: {len(pending)} game(s) not yet kicked off")
    for _, r in pending.sort_values("kickoff").iterrows():
        mark = "ready  " if r["injury_report_final"] else "WAITING"
        print(f"  {mark} {r['away_team']:>3} @ {r['home_team']:<3}  {r['injury_note']}")

    forecast = pending if args.allow_unsettled_injuries else settled
    if len(unsettled) and not args.allow_unsettled_injuries:
        print(
            f"\n{len(unsettled)} game(s) held back until their final injury report "
            "is in the data. Re-run after it is published (python -m src.weekly "
            "pulls and predicts in one step), or pass --allow-unsettled-injuries "
            "to forecast them now on an incomplete report."
        )
    if forecast.empty:
        print("\nNothing to forecast yet. No file written.")
        return 0
    if args.dry_run:
        print(f"\n--dry-run: would forecast {len(forecast)} game(s). Nothing written.")
        return 0

    cutoff = training_cutoff(forecast)
    train = training_frame(df, cutoff)
    gap = assert_training_is_current(train, cutoff)  # before any fitting
    raw = fit_and_predict(
        model_names,
        {"model": df},
        {"model": train},
        cutoff,
        list(forecast["game_id"]),
        skip,
    )

    print(
        f"\nForecasting {len(forecast)} game(s); trained on {len(train)} completed "
        f"games through {train['gameday'].max().date()} ({gap} days before the first "
        "forecast kickoff)"
    )
    out = forecast[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "injury_report_final",
            "injury_report_due",
            "injury_note",
            "home_status_rows",
            "away_status_rows",
        ]
    ].copy()
    for name, (h, a) in raw.items():
        out[f"{name}_home"] = np.round(h, DP)
        out[f"{name}_away"] = np.round(a, DP)

    # The composite is averaged BEFORE rounding. Averaging the rounded columns
    # (the first version) stacked three rounding errors into it.
    members = [m for m in comp["members"] if m in raw]
    if members:
        out[f"{comp['name']}_home"] = np.round(
            np.mean([raw[m][0] for m in members], 0), DP
        )
        out[f"{comp['name']}_away"] = np.round(
            np.mean([raw[m][1] for m in members], 0), DP
        )

    for name in list(raw) + ([comp["name"]] if members else []):
        out[f"{name}_margin"] = (out[f"{name}_home"] - out[f"{name}_away"]).round(DP)
        out[f"{name}_total"] = (out[f"{name}_home"] + out[f"{name}_away"]).round(DP)

    out = out.merge(market_reference(out["game_id"]), on="game_id", how="left")
    out["kickoff"] = out["game_id"].map(kickoff_utc(out["game_id"]))
    # Per-GAME provenance: a week is written in several runs, so each row says
    # when its own forecast was made, by which models, and from which code.
    git = git_state()
    out["generated_at"] = now.isoformat()
    out["trained_through"] = str(train["gameday"].max().date())
    out["n_training_games"] = len(train)
    out["models"] = ",".join(raw)
    out["composite_members"] = ",".join(members)
    out["code_version"] = git["sha"][:12] + ("+dirty" if git["dirty"] else "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{season}_wk{week:02d}.parquet"
    out = freeze_started_games(out, path, now)
    out = out.sort_values(["kickoff", "game_id"], na_position="last")
    if unchanged(path, out):
        out = pd.read_parquet(path)
        print(
            "\nEvery forecast matches the record already on file; "
            f"{path} left as it is."
        )
    else:
        out.to_parquet(path, index=False)

    shown = [comp["name"]] + [m for m in live["models"] if m in raw]
    print(
        f"\n{'KICKOFF (ET)':<19}{'MATCHUP':<13}"
        + "".join(f"{m.upper():>12}" for m in shown)
        + f"{'MARKET':>12}"
    )
    print("-" * (32 + 12 * (len(shown) + 1)))
    for _, r in out.iterrows():
        when = injury_readiness.et(r["kickoff"])[:-3] if pd.notna(r["kickoff"]) else "?"
        cells = ""
        for m in shown:
            hk, ak = f"{m}_home", f"{m}_away"
            cells += (
                f"{r[ak]:.1f}-{r[hk]:.1f}".rjust(12)
                if hk in r and pd.notna(r.get(hk))
                else f"{'--':>12}"
            )
        mk = (
            f"{r['market_away']:.1f}-{r['market_home']:.1f}".rjust(12)
            if pd.notna(r.get("market_home"))
            else f"{'no line':>12}"
        )
        final = r.get("injury_report_final")
        flag = "  *provisional" if pd.notna(final) and not bool(final) else ""
        print(
            f"{when:<19}{r['away_team'] + ' @ ' + r['home_team']:<13}{cells}{mk}{flag}"
        )

    print("\nScores shown as away-home. Market column is the CLOSING line's implied")
    print("score and is reference only -- it is never a model input.")
    if (~out["injury_report_final"].fillna(True).astype(bool)).any():
        print(
            "*provisional: forecast before the game's final injury report was in the data."
        )
    print(f"Record: {path}")

    if args.html:
        # Deliberately NOT a page for this week alone. The report is rebuilt
        # from every saved prediction, so the new week joins the ledger next to
        # the ones already graded rather than replacing them.
        from src.predict import build_report

        print()
        build_report.summarize(build_report.render())

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                json.loads(out.to_json(orient="records", date_format="iso")), indent=2
            )
            + "\n"
        )
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
