"""
src/predict/build_report.py

Renders every saved prediction week into ONE page, defaulting to the most
recent, and grades past weeks against what actually happened.

Why this exists as its own step. The first version of this page was a static
snapshot of a single week, because it was derived from a Claude artifact --
which has no filesystem and must embed its data. That constraint does not apply
locally: the predictions are sitting in data/predictions/ and the results are in
data/raw/schedules.parquet. A local page should read both.

Two distinct jobs, which is why it is separate from predict_week.py:

    predict_week.py   produces a prediction. Slow -- it fits every model.
    build_report.py   renders predictions that already exist. Fast, and safe to
                      re-run whenever new RESULTS land, which is exactly when
                      you want last week's forecast graded without touching the
                      forecast itself.

Grading. A game with a final score gets scored against every model: points off
per side, total absolute error, margin error, and whether the model picked the
right winner. Predictions are never revised in light of the result -- the
parquet is a tamper-evident record of what was claimed beforehand, and this
only reads it.

Still one self-contained file. All weeks are embedded, so it opens over file://
in any browser with no server.

Run: python -m src.predict.build_report
     python -m src.predict.build_report --open     # and launch a browser
Output: data/predictions/index.html
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PRED_DIR = Path("data/predictions")
TEMPLATE = Path(__file__).with_name("report_template.html")
OUT = PRED_DIR / "index.html"
MODELS = ["combined", "linear", "poisson", "gp"]

# The template holds no <html>/<head> of its own so that the same file can be
# published as an artifact, where the host supplies them. Opened from disk it
# needs both: without a doctype the browser renders in quirks mode, and without
# a charset declaration a file:// page can decode the typography as mojibake.
SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head>
<body>
{body}
</body>
</html>
"""
DP = 1


def load_actuals() -> pd.DataFrame:
    s = pd.read_parquet("data/raw/schedules.parquet")
    return s[["game_id", "home_score", "away_score"]].dropna(subset=["home_score"])


def kickoff_index() -> pd.Series:
    """Kickoff instant per game_id, for ordering the page.

    Fallback only. predict_week stores a `kickoff` column on every week it
    writes, and that stored value is preferred -- it is part of the record. This
    exists for weeks written before that column did, which would otherwise have
    nothing finer than a date to sort on.

    NOTE: this is the same gameday + gametime -> US/Eastern -> UTC construction
    as predict_week.kickoff_utc() and build_injury_features.kickoff_times().
    Three copies is two too many; it is duplicated here rather than imported
    because predict_week pulls in all four models (and sklearn with them), which
    would turn this one-second script into a several-second one. Worth
    consolidating into a shared schedule helper.
    """
    s = pd.read_parquet("data/raw/schedules.parquet")
    ts = pd.to_datetime(
        s["gameday"].astype(str) + " " + s["gametime"].fillna("13:00"), errors="coerce"
    )
    ts = ts.dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    return pd.Series(ts.to_numpy(), index=s["game_id"].to_numpy())


def grade(models: dict, actual: dict) -> dict:
    """Per-model scoring for one finished game."""
    out = {}
    for name, p in models.items():
        err_home = p["home"] - actual["home"]
        err_away = p["away"] - actual["away"]
        pred_margin = p["home"] - p["away"]
        true_margin = actual["home"] - actual["away"]
        # A pick is "right" when the predicted winner matches. A true tie is
        # scored as correct only if the model also predicted a dead heat, which
        # in practice it never does -- ties are rare enough to leave literal.
        if true_margin == 0:
            correct = pred_margin == 0
        else:
            correct = (pred_margin > 0) == (true_margin > 0)
        out[name] = {
            "err_home": round(err_home, DP),
            "err_away": round(err_away, DP),
            "abs_total": round(abs(err_home) + abs(err_away), DP),
            "margin_err": round(pred_margin - true_margin, DP),
            "winner_correct": bool(correct),
        }
    return out


def week_payload(path: Path, actuals: pd.DataFrame, kickoffs: pd.Series) -> dict:
    df = pd.read_parquet(path).merge(
        actuals.rename(columns={"home_score": "act_home", "away_score": "act_away"}),
        on="game_id",
        how="left",
    )

    # Order the slate the way it is actually played. Rows arrive in whatever
    # order predict_week produced them, which is neither chronological nor
    # stable -- week 2's Thursday night game sat fourth. gameday alone cannot
    # fix it either: a dozen games share a Sunday and kick off across seven
    # hours. The stored kickoff is authoritative where present, with the
    # schedule as fallback for weeks written before that column existed.
    def _utc(values) -> pd.Series:
        """Coerce to a tz-aware series, whatever comes in.

        Both sources can be wholly unresolvable -- a week written before the
        column existed, or a schedule that knows none of these game_ids -- and an
        all-missing map comes back as float64, which cannot be cast to a
        datetime. Going via the index keeps the dtype right in every case.
        """
        return pd.to_datetime(
            pd.Series(list(values), index=df.index), utc=True, errors="coerce"
        )

    stored = (
        _utc(df["kickoff"]) if "kickoff" in df.columns else _utc([pd.NaT] * len(df))
    )
    df["_kick"] = stored.fillna(_utc(kickoffs.reindex(df["game_id"])))
    # gameday breaks ties for anything the schedule could not resolve, and
    # game_id makes the order deterministic rather than merely sorted.
    df = df.sort_values(
        ["_kick", "gameday", "game_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    games, graded = [], 0
    for _, r in df.iterrows():
        models = {}
        for m in MODELS:
            if f"{m}_home" not in df.columns or pd.isna(r[f"{m}_home"]):
                continue
            models[m] = {
                "away": round(float(r[f"{m}_away"]), DP),
                "home": round(float(r[f"{m}_home"]), DP),
            }
        g = {
            "away": r["away_team"],
            "home": r["home_team"],
            "kickoff": str(pd.Timestamp(r["gameday"]).date()),
            # Full instant when known, so the page can show a real kickoff time
            # in the reader's own timezone rather than a bare date. Null for
            # weeks predicted before the column existed; the page falls back.
            "kickoff_ts": (
                pd.Timestamp(r["_kick"]).isoformat() if pd.notna(r["_kick"]) else None
            ),
            "models": models,
            "market": (
                {
                    "spread": float(r["spread_line"]),
                    "total": float(r["total_line"]),
                    "away": round(float(r["market_away"]), DP),
                    "home": round(float(r["market_home"]), DP),
                }
                if pd.notna(r.get("spread_line"))
                else None
            ),
        }
        if pd.notna(r.get("act_home")):
            g["actual"] = {"away": float(r["act_away"]), "home": float(r["act_home"])}
            g["grade"] = grade(models, g["actual"])
            graded += 1
        games.append(g)

    summary = {}
    if graded:
        for m in MODELS:
            rows = [g["grade"][m] for g in games if "grade" in g and m in g["grade"]]
            if not rows:
                continue
            summary[m] = {
                "mae": round(sum(r["abs_total"] for r in rows) / (2 * len(rows)), 2),
                "margin_mae": round(
                    sum(abs(r["margin_err"]) for r in rows) / len(rows), 2
                ),
                "winners": sum(r["winner_correct"] for r in rows),
                "n": len(rows),
            }
    # Honesty flag. A week predicted BEFORE its first kickoff is a genuine
    # pre-registered forecast; one generated afterwards is still out of sample
    # (the fit only ever sees prior games) but nobody stopped it from being
    # regenerated until it looked good. The page says which it is rather than
    # presenting the two as equivalent.
    generated = pd.Timestamp(df["generated_at"].iloc[0])
    first_kick = pd.Timestamp(df["gameday"].min())
    if generated.tzinfo is not None:
        first_kick = first_kick.tz_localize(generated.tzinfo)
    return {
        "season": int(df["season"].iloc[0]),
        "week": int(df["week"].iloc[0]),
        "trained_through": str(df["trained_through"].iloc[0]),
        "n_training_games": int(df["n_training_games"].iloc[0]),
        "generated_at": str(df["generated_at"].iloc[0]),
        "backfilled": bool(generated > first_kick),
        "n_graded": graded,
        "summary": summary,
        "games": games,
    }


def build() -> dict:
    files = sorted(PRED_DIR.glob("*_wk*.parquet"))
    if not files:
        sys.exit(
            "ERROR: no predictions in data/predictions/.\n"
            "Make one first:  python -m src.predict.predict_week"
        )
    actuals = load_actuals()
    kickoffs = kickoff_index()
    weeks = [week_payload(f, actuals, kickoffs) for f in files]
    weeks.sort(key=lambda w: (w["season"], w["week"]))
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "weeks": weeks,
    }


def render() -> dict:
    """Write data/predictions/index.html. Returns the payload it embedded."""
    if not TEMPLATE.exists():
        sys.exit(f"ERROR: {TEMPLATE} is missing.")
    data = build()
    OUT.write_text(
        SHELL.format(
            body=TEMPLATE.read_text().replace(
                "__DATA__", json.dumps(data, separators=(",", ":"))
            )
        )
    )
    return data


def summarize(data: dict) -> None:
    total = sum(len(w["games"]) for w in data["weeks"])
    done = sum(w["n_graded"] for w in data["weeks"])
    span = f"{data['weeks'][0]['season']} wk{data['weeks'][0]['week']}"
    if len(data["weeks"]) > 1:
        span += f" → {data['weeks'][-1]['season']} wk{data['weeks'][-1]['week']}"
    print(f"Built {OUT}")
    print(f"  {len(data['weeks'])} week(s): {span}")
    print(f"  {total} games, {done} already played and graded")
    print(f"  opens at the most recent week")
    print(f"\n  firefox {OUT}     (or xdg-open / open / double-click)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true", help="launch the page when done")
    args = ap.parse_args()

    summarize(render())

    if args.open:
        for cmd in ("xdg-open", "open"):
            try:
                subprocess.run([cmd, str(OUT)], check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue


if __name__ == "__main__":
    main()
