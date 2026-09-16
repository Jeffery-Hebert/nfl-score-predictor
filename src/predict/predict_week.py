"""
src/predict/predict_week.py

Produces score predictions for games that have NOT been played yet. This is the
piece that turns a backtesting project into something that can answer "what
will happen on Sunday".

How it differs from the model scripts. Those run walk-forward: refit for every
historical week and score the result. This fits ONCE on every completed game
and predicts forward. Same fit/predict functions, so a prediction here is
produced by exactly the code the backtest measured -- no reimplementation that
could drift from what was validated.

Training window. Every completed game in model_table.parquet, which after
`build_all` means everything through last week. tests/test_training_completeness.py
asserts the harness never silently withholds recent games; this script asserts
the same thing directly at runtime and refuses to run on stale data.

Market lines. spread_line and total_line are printed ALONGSIDE the predictions
and are NEVER inputs. They are also converted into the score pair the market
implies, so the two can be read on the same scale:

    implied home = (total + spread) / 2
    implied away = (total - spread) / 2

nflverse ships the CLOSING line, so a Wednesday prediction is being shown
against a number that will keep moving until kickoff.

Rounding. Every predicted score is rounded to one decimal place before it is
stored or displayed. A tenth of a point is already far finer than the model can
actually resolve -- typical error is over nine points -- so the extra digits
were noise dressed as precision.

Run: python -m src.predict.predict_week                 # next unplayed week
     python -m src.predict.predict_week --season 2026 --week 2
     python -m src.predict.predict_week --week 2 --html   # + rebuild the page
     python -m src.predict.predict_week --week 2 --json out.json
Output: data/predictions/<season>_wk<week>.parquet
        data/predictions/index.html   (with --html; see build_report.py)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import fit_baseline, predict_baseline
from src.models.gaussian_process import fit_gp, predict_gp
from src.models.linear import fit_linear, predict_linear
from src.models.poisson_glm import fit_poisson, predict_poisson

MODEL_TABLE = Path("data/processed/model_table.parquet")
SCHEDULES = Path("data/raw/schedules.parquet")

# One decimal place. Typical error is over nine points, so anything finer is
# noise wearing the costume of precision.
DP = 1
OUT_DIR = Path("data/predictions")

# Everything but the stack, which is built from these.
MODELS = {
    "baseline": (fit_baseline, predict_baseline),
    "linear": (fit_linear, predict_linear),
    "poisson": (fit_poisson, predict_poisson),
    "gp": (fit_gp, predict_gp),
}
# Weights for the combined model. Equal, deliberately: the walk-forward stack
# fits ridge weights on out-of-fold predictions, which do not exist for a game
# that has not been played. An equal average of the three is the honest
# stand-in and is what the stack's weights land near anyway.
STACK_MEMBERS = ["linear", "poisson", "gp"]


def load_table() -> pd.DataFrame:
    if not MODEL_TABLE.exists():
        sys.exit(f"ERROR: {MODEL_TABLE} missing. Run: python -m src.features.build_all")
    df = pd.read_parquet(MODEL_TABLE)
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def pick_week(df: pd.DataFrame, season, week):
    """Default to the earliest unplayed week."""
    upcoming = df[df["home_score"].isna()].sort_values("gameday")
    if upcoming.empty:
        sys.exit("ERROR: no unplayed games in the table. Re-pull schedules.")
    if season is None or week is None:
        row = upcoming.iloc[0]
        return int(row["season"]), int(row["week"])
    return int(season), int(week)


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
    # a recent window: a game cancelled years ago (2021 BUF@CIN) never gets a
    # score and must not trip this forever.
    horizon = min(kickoff, pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    recent = sched[sched["gameday"].between(horizon - pd.Timedelta(days=45), horizon)]
    unscored = recent[recent["home_score"].isna()]
    if len(unscored):
        sys.exit(
            f"ERROR: {len(unscored)} games kicked off on or before "
            f"{horizon.date()} and still have no final score.\n"
            "The raw data is behind. Refresh, then rebuild:\n"
            "  python src/ingest/pull_schedules.py\n"
            "  python src/ingest/pull_pbp.py\n"
            "  python src/ingest/pull_injuries.py\n"
            "  python -m src.features.build_all"
        )

    return (kickoff - train["gameday"].max()).days


def market_reference(game_ids) -> pd.DataFrame:
    """Closing spread/total and the score pair they imply. Reference only."""
    s = pd.read_parquet("data/raw/schedules.parquet")
    s = s[s["game_id"].isin(game_ids)][
        ["game_id", "gametime", "spread_line", "total_line"]
    ].copy()
    # spread_line is the home margin the market expects (positive = home favoured)
    s["market_home"] = ((s["total_line"] + s["spread_line"]) / 2).round(1)
    s["market_away"] = ((s["total_line"] - s["spread_line"]) / 2).round(1)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--json", type=str, default=None, help="also write JSON here")
    ap.add_argument(
        "--html",
        action="store_true",
        help="also rebuild data/predictions/index.html (every week, graded)",
    )
    ap.add_argument(
        "--skip-gp",
        action="store_true",
        help="omit the Gaussian Process (it takes ~25s to fit once)",
    )
    args = ap.parse_args()

    df = load_table()
    season, week = pick_week(df, args.season, args.week)

    target = df[(df["season"] == season) & (df["week"] == week)].copy()
    if target.empty:
        sys.exit(f"ERROR: no games found for season {season} week {week}.")
    kickoff = target["gameday"].min()

    # Strictly prior completed games -- the same rule the backtest enforces.
    train = df[df["gameday"] < kickoff].dropna(subset=["home_score", "away_score"])
    gap = assert_training_is_current(train, kickoff)

    print(f"Predicting season {season}, week {week} -- {len(target)} games")
    print(
        f"  training on {len(train)} completed games through "
        f"{train['gameday'].max().date()} ({gap} days before kickoff)"
    )
    if target["home_score"].notna().any():
        n = int(target["home_score"].notna().sum())
        print(f"  NOTE: {n} of these games already have final scores")

    out = target[
        ["game_id", "season", "week", "gameday", "home_team", "away_team"]
    ].copy()
    for name in MODELS:
        if name == "gp" and args.skip_gp:
            continue
        fit_fn, predict_fn = MODELS[name]
        print(f"  fitting {name}...", flush=True)
        model = fit_fn(train)
        h, a = predict_fn(model, target)
        out[f"{name}_home"] = np.round(h, DP)
        out[f"{name}_away"] = np.round(a, DP)

    members = [m for m in STACK_MEMBERS if f"{m}_home" in out.columns]
    if members:
        out["combined_home"] = (
            out[[f"{m}_home" for m in members]].mean(axis=1).round(DP)
        )
        out["combined_away"] = (
            out[[f"{m}_away" for m in members]].mean(axis=1).round(DP)
        )

    for name in list(MODELS) + ["combined"]:
        if f"{name}_home" in out.columns:
            out[f"{name}_margin"] = (out[f"{name}_home"] - out[f"{name}_away"]).round(
                DP
            )
            out[f"{name}_total"] = (out[f"{name}_home"] + out[f"{name}_away"]).round(DP)

    out = out.merge(market_reference(out["game_id"]), on="game_id", how="left")
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out["trained_through"] = str(train["gameday"].max().date())
    out["n_training_games"] = len(train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{season}_wk{week:02d}.parquet"
    out.to_parquet(path, index=False)

    print(
        f"\n{'MATCHUP':<22}{'COMBINED':>12}{'LINEAR':>12}{'POISSON':>12}{'GP':>12}{'MARKET':>12}"
    )
    print("-" * 82)
    for _, r in out.iterrows():
        match = f"{r['away_team']} @ {r['home_team']}"

        def pair(pre):
            hk, ak = f"{pre}_home", f"{pre}_away"
            if hk not in r or pd.isna(r[hk]):
                return f"{'--':>13}"
            return f"{r[ak]:.1f}-{r[hk]:.1f}".rjust(13)

        mk = (
            f"{r['market_away']:.1f}-{r['market_home']:.1f}".rjust(13)
            if pd.notna(r.get("market_home"))
            else f"{'no line':>13}"
        )
        print(
            f"{match:<22}{pair('combined')}{pair('linear')}{pair('poisson')}"
            f"{pair('gp')}{mk}"
        )

    print(f"\nScores shown as away-home. Market column is the CLOSING line's implied")
    print(f"score and is reference only -- it is never a model input.")
    print(f"Saved to {path}")

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


if __name__ == "__main__":
    main()
