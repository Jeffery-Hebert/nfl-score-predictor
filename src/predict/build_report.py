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

Tracked in git, next to the records, since 2026-09-27. It used to be a local,
gitignored file that only a local run rebuilt, so a `git pull` brought in new
records but left the page showing the old ones: Sunday's forecasts arrived
FINAL and the page still tagged them pre-injury-report. Three things now keep
the page and the records from disagreeing:

  - the page is a pure function of its inputs (records, results, template -- no
    wall clock), so it can be committed and only changes when they do;
  - it embeds a sha256 of every record it was built from, and verify() checks
    the page against the records game by game; render() runs verify() on its
    own output before replacing the old page;
  - tests/test_report_grading.py fails CI if the committed page and the
    committed records ever disagree, and the pipeline commits both together.

Run: python -m src.predict.build_report
     python -m src.predict.build_report --check    # is the page current? exit 1 if not
     python -m src.predict.build_report --open     # and launch a browser
Output: data/predictions/index.html
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.predict.injury_readiness import is_provisional
from src.provenance import sha256

PRED_DIR = Path("data/predictions")
SCHEDULES = Path("data/raw/schedules.parquet")
TEMPLATE = Path(__file__).with_name("report_template.html")
OUT = PRED_DIR / "index.html"
DATA_MARKER = "const DATA = "
# Which models the page shows is read from the records themselves: every
# <name>_home/<name>_away pair in a week's parquet, except these. The list used
# to be hard-coded, so changing the live models in config.yaml would have
# silently dropped the new ones from the ledger.
NOT_SHOWN = {"market", "baseline", "act"}
LABELS = {
    "combined": "Combined",
    "linear": "Linear",
    "poisson": "Poisson",
    "gp": "Gauss. Proc.",
    "rf": "Rand. Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "bayesian": "Bayesian",
    "rnn": "RNN",
    "drivev2": "Drive v2",
    "montecarlo": "Monte Carlo",
    "logistic": "Logistic",
}

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


def load_actuals(schedules=SCHEDULES) -> pd.DataFrame:
    """Final scores, for grading. Without a schedule the page is simply
    ungraded (a fresh checkout, CI) -- every forecast is still shown."""
    cols = ["game_id", "home_score", "away_score"]
    if not Path(schedules).exists():
        return pd.DataFrame(columns=cols)
    s = pd.read_parquet(schedules)
    return s[cols].dropna(subset=["home_score"])


def kickoff_index(schedules=SCHEDULES) -> pd.Series:
    """Kickoff instant per game_id, for ordering the page.

    Fallback only. predict_week stores a `kickoff` column on every week it
    writes, and that stored value is preferred -- it is part of the record. This
    exists for weeks written before that column did. The construction is the
    shared one in src/schedule.py, which imports nothing but pandas, so this
    stays a one-second script.
    """
    from src.schedule import kickoff_by_game

    if not Path(schedules).exists():
        return pd.Series(dtype="datetime64[ns, UTC]")
    return kickoff_by_game(pd.read_parquet(schedules))


def models_in(df: pd.DataFrame) -> list[str]:
    """Model prefixes with both a _home and an _away column, in column order."""
    return [
        c[: -len("_home")]
        for c in df.columns
        if c.endswith("_home")
        and f"{c[: -len('_home')]}_away" in df.columns
        and c[: -len("_home")] not in NOT_SHOWN
    ]


def display_order(found: set[str]) -> list[str]:
    """Composite first, then the live models in config order, then anything an
    older week carries that the config no longer names."""
    try:
        from src.config import live_settings

        live = live_settings()
        head = [live["composite"]["name"]] + live["models"]
    except Exception:  # a page must still render without a valid config
        head = ["combined"]
    return [m for m in head if m in found] + sorted(found - set(head))


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
    week_models = models_in(df)
    for _, r in df.iterrows():
        models = {}
        for m in week_models:
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
        # The forecast relative to the market, per model shown (the page draws
        # the composite's). Computed here, not in the page, so it is tested.
        g["vs_market"] = (
            {
                m: market_difference(
                    v["home"],
                    v["away"],
                    r["spread_line"],
                    r["total_line"],
                    r["home_team"],
                    r["away_team"],
                )
                for m, v in models.items()
            }
            if pd.notna(r.get("spread_line")) and pd.notna(r.get("total_line"))
            else None
        )
        # Honesty flag, per GAME. A forecast written before its own kickoff is a
        # genuine pre-registered prediction; one written afterwards is still out
        # of sample (the fit only ever sees prior games) but nobody stopped it
        # being regenerated until it looked good.
        #
        # This used to be decided for the whole week off the week's FIRST
        # kickoff, which condemned fifteen untouched Sunday forecasts because a
        # Thursday game had already been played. Now that each row carries its
        # own generated_at and kickoff -- a week is written three times as its
        # slates come up -- the question can be asked of each game separately,
        # which is the grain it was always about.
        made = pd.to_datetime(r.get("generated_at"), utc=True, errors="coerce")
        g["backfilled"] = bool(
            pd.notna(made) and pd.notna(r["_kick"]) and made > r["_kick"]
        )
        # Forecast before the game's FINAL injury report was in the data (only
        # possible with --allow-unsettled-injuries). Weeks written before the
        # column existed carry no flag rather than a guessed one.
        g["provisional"] = is_provisional(r.get("injury_report_final"))
        if pd.notna(r.get("act_home")):
            g["actual"] = {"away": float(r["act_away"]), "home": float(r["act_home"])}
            g["grade"] = grade(models, g["actual"])
            graded += 1
        games.append(g)

    summary = {}
    if graded:
        for m in week_models:
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
    n_backfilled = sum(g["backfilled"] for g in games)
    return {
        "models": week_models,
        "n_provisional": int(sum(g["provisional"] for g in games)),
        "season": int(df["season"].iloc[0]),
        "week": int(df["week"].iloc[0]),
        "trained_through": str(df["trained_through"].max()),
        "n_training_games": int(df["n_training_games"].max()),
        # The newest forecast in the week. Individual games carry their own.
        "generated_at": str(df["generated_at"].max()),
        "n_backfilled": int(n_backfilled),
        # Graded AND backfilled -- what the running record has to discount.
        "n_backfilled_graded": int(
            sum(g["backfilled"] and "grade" in g for g in games)
        ),
        "backfilled": bool(n_backfilled),
        "n_graded": graded,
        "summary": summary,
        "games": games,
    }


def market_difference(home, away, spread, total, home_team, away_team):
    """How far a forecast sits from the betting market, in the market's terms.

      margin  (home - away) - spread_line. nflverse's spread_line is the HOME
              team's expected margin (positive = home favoured), so a positive
              difference means the model rates the home team higher than the
              market does, a negative one the away team.
      total   (home + away) - total_line. Positive: the model expects more
              points than the market.
      side    the team the model rates higher than the market does, by
              |margin| points; None when they agree exactly.

    Rounded to one decimal, like every forecast, so float noise (3.6 - 5.5 =
    -1.9000000000000004) never reaches the page. None when there is no line.
    A difference, not an edge: against the closing line the model has been a
    coin flip (README section 9).
    """
    if any(pd.isna(x) for x in (home, away, spread, total)):
        return None
    margin = round((float(home) - float(away)) - float(spread), DP) + 0.0
    diff_total = round((float(home) + float(away)) - float(total), DP) + 0.0
    side = home_team if margin > 0 else away_team if margin < 0 else None
    return {"margin": margin, "total": diff_total, "side": side}


def records(pred_dir=PRED_DIR) -> list[Path]:
    return sorted(Path(pred_dir).glob("*_wk*.parquet"))


def build(pred_dir=PRED_DIR, schedules=SCHEDULES) -> dict:
    files = records(pred_dir)
    if not files:
        sys.exit(
            f"ERROR: no predictions in {pred_dir}/.\n"
            "Make one first:  python -m src.predict.predict_week"
        )
    actuals = load_actuals(schedules)
    kickoffs = kickoff_index(schedules)
    weeks = [week_payload(f, actuals, kickoffs) for f in files]
    weeks.sort(key=lambda w: (w["season"], w["week"]))
    order = display_order({m for w in weeks for m in w["models"]})
    composite = order[0] if order else "combined"
    graded = [g["kickoff"] for w in weeks for g in w["games"] if "actual" in g]
    return {
        # No wall clock anywhere: the same records and results always give the
        # same page, so it can live in git and changes only when they do.
        # (It used to carry built_at, which made every rebuild a new file.)
        "records": {f.name: sha256(f) for f in files},
        "forecasts_as_of": max(w["generated_at"] for w in weeks),
        "results_through": max(graded) if graded else None,
        # [key, label] in display order; the page draws one column per entry.
        "models": [[m, LABELS.get(m, m.title())] for m in order],
        "composite": composite,
        "weeks": weeks,
    }


def html_for(data: dict) -> str:
    # "</" inside a string would end the <script> early; "<\/" is the same
    # JSON string and cannot.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return SHELL.format(body=TEMPLATE.read_text().replace("__DATA__", payload))


def embedded(path) -> dict:
    """The data a page was built from, read back out of the page itself."""
    html = Path(path).read_text()
    start = html.index(DATA_MARKER) + len(DATA_MARKER)
    data, _ = json.JSONDecoder().raw_decode(html, start)
    return data


def verify(path=None, pred_dir=PRED_DIR) -> list[str]:
    """Every way the page disagrees with the records it claims to show.

    Empty means current. Checked from the RECORDS side, independently of how
    week_payload built the page: each record's fingerprint, then each game's
    forecasts and injury-report status as stored in the parquet."""
    path = Path(path) if path is not None else Path(pred_dir) / "index.html"
    if not path.exists():
        return [
            f"{path} does not exist -- build it: python -m src.predict.build_report"
        ]
    data = embedded(path)
    if "records" not in data:
        return [f"{path} predates record fingerprints -- rebuild it"]
    problems = []
    on_disk = {f.name: sha256(f) for f in records(pred_dir)}
    for name in sorted(set(on_disk) | set(data["records"])):
        if name not in data["records"]:
            problems.append(f"{name} is not on the page")
        elif name not in on_disk:
            problems.append(f"the page shows {name}, which no longer exists")
        elif data["records"][name] != on_disk[name]:
            problems.append(f"{name} has changed since the page was built")
    weeks = {(w["season"], w["week"]): w for w in data["weeks"]}
    for f in records(pred_dir):
        rec = pd.read_parquet(f)
        season, week = int(rec["season"].iloc[0]), int(rec["week"].iloc[0])
        w = weeks.get((season, week))
        if w is None:
            problems.append(f"{season} week {week} is missing from the page")
            continue
        shown = {(g["away"], g["home"]): g for g in w["games"]}
        if len(shown) != len(rec):
            problems.append(
                f"{season} week {week}: {len(rec)} games on file, "
                f"{len(shown)} on the page"
            )
        for _, r in rec.iterrows():
            label = f"{season} week {week} {r['away_team']} @ {r['home_team']}"
            g = shown.get((r["away_team"], r["home_team"]))
            if g is None:
                problems.append(f"{label}: not on the page")
                continue
            want = is_provisional(r.get("injury_report_final"))
            if g["provisional"] != want:
                problems.append(
                    f"{label}: page says "
                    f"{'provisional' if g['provisional'] else 'final'}, the "
                    f"record says {'provisional' if want else 'final'}"
                )
            for m in w["models"]:
                if f"{m}_home" not in rec.columns or pd.isna(r[f"{m}_home"]):
                    continue
                stored = [round(float(r[f"{m}_{s}"]), DP) for s in ("away", "home")]
                page = g["models"].get(m)
                if page is None or [page["away"], page["home"]] != stored:
                    problems.append(
                        f"{label}: {m} is {page} on the page, {stored} on file"
                    )
                if pd.isna(r.get("spread_line")) or pd.isna(r.get("total_line")):
                    continue
                # Recomputed here from the record, not via market_difference.
                a_pts, h_pts = stored
                margin = round(h_pts - a_pts - float(r["spread_line"]), DP)
                total = round(h_pts + a_pts - float(r["total_line"]), DP)
                vs = (g.get("vs_market") or {}).get(m) or {}
                if (vs.get("margin"), vs.get("total")) != (margin, total):
                    problems.append(
                        f"{label}: {m} vs market is {vs or None} on the page; "
                        f"the record gives margin {margin:+.1f}, total {total:+.1f}"
                    )
    return problems


def render(pred_dir=PRED_DIR, schedules=SCHEDULES) -> dict:
    """Write <pred_dir>/index.html -- only if it verifies against the records.
    Returns the payload it embedded."""
    if not TEMPLATE.exists():
        sys.exit(f"ERROR: {TEMPLATE} is missing.")
    data = build(pred_dir, schedules)
    out = Path(pred_dir) / "index.html"
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(html_for(data))
    problems = verify(tmp, pred_dir)
    if problems:
        tmp.unlink()
        raise RuntimeError(
            "the ledger page would not match the records it renders -- the old "
            "page is left in place:\n  " + "\n  ".join(problems)
        )
    tmp.replace(out)
    return data


def summarize(data: dict, out=OUT) -> None:
    total = sum(len(w["games"]) for w in data["weeks"])
    done = sum(w["n_graded"] for w in data["weeks"])
    span = f"{data['weeks'][0]['season']} wk{data['weeks'][0]['week']}"
    if len(data["weeks"]) > 1:
        span += f" → {data['weeks'][-1]['season']} wk{data['weeks'][-1]['week']}"
    print(f"Built {out} (verified against every record)")
    print(f"  {len(data['weeks'])} week(s): {span}")
    print(f"  {total} games, {done} already played and graded")
    print(f"  opens at the most recent week")
    print(f"\n  firefox {out}     (or xdg-open / open / double-click)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true", help="launch the page when done")
    ap.add_argument(
        "--check",
        action="store_true",
        help="build nothing; exit 1 if the page does not match the records",
    )
    args = ap.parse_args()

    if args.check:
        problems = verify(OUT, PRED_DIR)
        for p in problems:
            print(f"  {p}")
        if problems:
            print(f"STALE: {OUT} does not match the records. Rebuild:")
            print("  python -m src.predict.build_report")
            return 1
        print(f"{OUT} matches every record.")
        return 0

    summarize(render(PRED_DIR, SCHEDULES), OUT)

    if args.open:
        for cmd in ("xdg-open", "open"):
            try:
                subprocess.run([cmd, str(OUT)], check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
