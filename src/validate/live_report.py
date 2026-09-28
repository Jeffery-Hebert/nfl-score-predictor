"""
How the PUBLISHED forecasts have done: every record in data/predictions/,
graded against the results and against the betting market.

The walk-forward backtest (model_report) answers "how good is the method?".
This answers the narrower, harder question -- how good were the forecasts that
were actually written down before kickoff? -- which is the only record that
could have been bet on. Per model (every column pair in the records):

  accuracy      score / margin / total RMSE and MAE, winners picked
  the market    the same measures for the market's own implied score, at the
                line stored with the forecast and at the closing line
  picks         the side of the spread and of the total each forecast takes
                (build_report.market_picks -- the ledger's own rule), graded
                won / lost / push against the line it was made against, and
                against the closing line
  CLV           closing-line value, for forecasts made before kickoff: how far
                the line moved toward each pick after it was published. Beating
                the close consistently is the standard sign of an edge; noise
                averages to zero.

Games forecast after their own kickoff ("backfilled" on the ledger) are counted
in accuracy but reported separately, and have no CLV: their stored line is
already the closing one.

Run: python -m src.validate.live_report                  # every season on file
     python -m src.validate.live_report --season 2026
Output: data/processed/reports/live_report.csv (and live_picks.csv, one row per
        model and game)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.predict.build_report import (
    NOT_SHOWN,
    grade_picks,
    market_picks,
    records,
)
from src.schedule import kickoff_by_game

SCHEDULES = Path("data/raw/schedules.parquet")
REPORT_DIR = Path("data/processed/reports")
REFERENCE = {
    "baseline"
}  # fitted every week for comparison, never published as the forecast


def spread_clv(team, line_at_forecast, home_team, closing_spread) -> float:
    """Points gained on the spread by taking `team` at `line_at_forecast`
    rather than at the close. Lines are from `team`'s side (+5.5 = getting
    5.5). nflverse's closing spread_line is the HOME margin, so the home side's
    closing line is -closing_spread and the away side's +closing_spread.
    ATL +5.5, closing ATL +4.5: +1.0 -- the forecast got a better number."""
    closing = -closing_spread if team == home_team else closing_spread
    return round(line_at_forecast - closing, 1) + 0.0


def total_clv(side, total_at_forecast, closing_total) -> float:
    """Points gained on the total: an Over at 43.5 that closed 42.5 lost a
    point (the close was easier to clear); an Under gained it."""
    gained = closing_total - total_at_forecast
    return round(gained if side == "Over" else -gained, 1) + 0.0


def model_names(rec: pd.DataFrame) -> list[str]:
    return [
        c[: -len("_home")]
        for c in rec.columns
        if c.endswith("_home")
        and c[: -len("_home")] not in NOT_SHOWN - REFERENCE
        and f"{c[:-len('_home')]}_away" in rec.columns
    ]


def graded_rows(season=None) -> pd.DataFrame:
    """One row per (model, played game) across every record on file."""
    sched = pd.read_parquet(SCHEDULES)
    kick = kickoff_by_game(sched)
    close = sched.set_index("game_id")[
        ["home_score", "away_score", "spread_line", "total_line"]
    ]
    rows = []
    for f in records():
        rec = pd.read_parquet(f)
        if season is not None and int(rec["season"].iloc[0]) != season:
            continue
        for _, r in rec.iterrows():
            gid = r["game_id"]
            if gid not in close.index or pd.isna(close.at[gid, "home_score"]):
                continue  # not played yet
            c = close.loc[gid]
            made = pd.to_datetime(r.get("generated_at"), utc=True, errors="coerce")
            k = pd.to_datetime(
                r.get("kickoff") if pd.notna(r.get("kickoff")) else kick.get(gid),
                utc=True,
            )
            for m in model_names(rec):
                h, a = r[f"{m}_home"], r[f"{m}_away"]
                if pd.isna(h) or pd.isna(a):
                    continue
                at = market_picks(
                    h, a, r.get("spread_line"), r.get("total_line"),
                    r["home_team"], r["away_team"],
                )  # fmt: skip
                cl = market_picks(
                    h, a, c["spread_line"], c["total_line"],
                    r["home_team"], r["away_team"],
                )  # fmt: skip
                res_at = grade_picks(
                    at, r["home_team"], c["home_score"], c["away_score"]
                )
                res_cl = grade_picks(
                    cl, r["home_team"], c["home_score"], c["away_score"]
                )
                pre = bool(pd.notna(made) and pd.notna(k) and made <= k)
                s_clv = t_clv = np.nan
                if pre and at and at["spread"]:
                    s_clv = spread_clv(
                        at["spread"]["team"],
                        at["spread"]["line"],
                        r["home_team"],
                        c["spread_line"],
                    )
                if pre and at and at["total"]:
                    t_clv = total_clv(
                        at["total"]["side"], at["total"]["line"], c["total_line"]
                    )
                rows.append(
                    {
                        "model": m,
                        "season": int(r["season"]),
                        "week": int(r["week"]),
                        "game_id": gid,
                        "pre_kickoff": pre,
                        "home_pred": float(h),
                        "away_pred": float(a),
                        "home_score": float(c["home_score"]),
                        "away_score": float(c["away_score"]),
                        "line_spread": r.get("spread_line"),
                        "line_total": r.get("total_line"),
                        "close_spread": c["spread_line"],
                        "close_total": c["total_line"],
                        "spread_result": res_at["spread"],
                        "total_result": res_at["total"],
                        "spread_result_close": res_cl["spread"],
                        "total_result_close": res_cl["total"],
                        "spread_clv": s_clv,
                        "total_clv": t_clv,
                    }
                )
    return pd.DataFrame(rows)


def _rmse(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan


def _record(results: pd.Series) -> str:
    w, lo, p = (int((results == k).sum()) for k in ("win", "loss", "push"))
    return f"{w}-{lo}" + (f"-{p}" if p else "")


def _pct(results: pd.Series) -> float:
    decided = results[results.isin(["win", "loss"])]
    return float((decided == "win").mean()) if len(decided) else np.nan


def summarize(g: pd.DataFrame) -> dict:
    """Accuracy, the market on the same games, picks and CLV for one model."""
    he, ae = g["home_pred"] - g["home_score"], g["away_pred"] - g["away_score"]
    pm, am = g["home_pred"] - g["away_pred"], g["home_score"] - g["away_score"]
    pt, at = g["home_pred"] + g["away_pred"], g["home_score"] + g["away_score"]
    out = {
        "games": len(g),
        "pre_kickoff": int(g["pre_kickoff"].sum()),
        "score_rmse": _rmse(np.concatenate([he, ae])),
        "score_mae": float(np.mean(np.abs(np.concatenate([he, ae])))),
        "margin_rmse": _rmse(pm - am),
        "total_rmse": _rmse(pt - at),
        "total_bias": float((pt - at).mean()),
        "winners": f"{int(((pm > 0) == (am > 0))[am != 0].sum())}-"
        f"{int(((pm > 0) != (am > 0))[am != 0].sum())}",
    }
    for label, sp, tl in (
        ("line", "line_spread", "line_total"),
        ("close", "close_spread", "close_total"),
    ):
        d = g.dropna(subset=[sp, tl])
        mh, ma = (d[tl] + d[sp]) / 2, (d[tl] - d[sp]) / 2
        out[f"{label}_score_rmse"] = _rmse(
            np.concatenate([mh - d["home_score"], ma - d["away_score"]])
        )
        out[f"{label}_margin_rmse"] = _rmse(d[sp] - (d["home_score"] - d["away_score"]))
        out[f"{label}_total_rmse"] = _rmse(d[tl] - (d["home_score"] + d["away_score"]))
    out["ats"] = _record(g["spread_result"])
    out["ats_pct"] = _pct(g["spread_result"])
    out["ou"] = _record(g["total_result"])
    out["ou_pct"] = _pct(g["total_result"])
    out["ats_close"] = _record(g["spread_result_close"])
    out["ou_close"] = _record(g["total_result_close"])
    pre = g[g["pre_kickoff"]]
    for bet in ("spread", "total"):
        clv = pre[f"{bet}_clv"].dropna()
        out[f"{bet}_clv"] = float(clv.mean()) if len(clv) else np.nan
        # better / same / worse number than the close, as counts
        out[f"{bet}_vs_close"] = (
            f"{int((clv > 0).sum())}/{int((clv == 0).sum())}/{int((clv < 0).sum())}"
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args(argv)
    rows = graded_rows(args.season)
    if rows.empty:
        print("No graded forecasts on file yet.")
        return 0
    table = pd.DataFrame(
        [{"model": m, **summarize(g)} for m, g in rows.groupby("model", sort=False)]
    ).sort_values("score_rmse")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "live_report.csv", index=False)
    rows.to_csv(REPORT_DIR / "live_picks.csv", index=False)

    pd.set_option("display.width", 220)
    scope = f"{args.season} season" if args.season else "every season on file"
    print(f"PUBLISHED FORECASTS, {scope}: {rows['game_id'].nunique()} played games")
    print(
        table[
            ["model", "games", "pre_kickoff", "score_rmse", "score_mae",
             "margin_rmse", "total_rmse", "total_bias", "winners"]
        ].round(3).to_string(index=False)  # fmt: skip
    )
    print("\nTHE MARKET ON THE SAME GAMES (implied score from the line)")
    first = table.iloc[0]
    print(
        f"  line when forecast: score RMSE {first['line_score_rmse']:.3f}, margin "
        f"{first['line_margin_rmse']:.3f}, total {first['line_total_rmse']:.3f}\n"
        f"  closing line:       score RMSE {first['close_score_rmse']:.3f}, margin "
        f"{first['close_margin_rmse']:.3f}, total {first['close_total_rmse']:.3f}"
    )
    pre_rows = rows[rows["pre_kickoff"]]
    pre = pd.DataFrame(
        [{"model": m, **summarize(g)} for m, g in pre_rows.groupby("model", sort=False)]
    )
    pre.to_csv(REPORT_DIR / "live_report_pre_kickoff.csv", index=False)
    cols = ["model", "ats", "ats_pct", "ou", "ou_pct", "ats_close", "ou_close",
            "spread_clv", "spread_vs_close", "total_clv", "total_vs_close"]  # fmt: skip
    for label, t in (
        (f"ALL {rows['game_id'].nunique()} graded games", table),
        (
            f"ONLY THE {pre_rows['game_id'].nunique()} FORECAST BEFORE KICKOFF "
            "(the ones that could have been bet)",
            pre,
        ),
    ):
        if t.empty:
            continue
        print(f"\nPICKS AGAINST THE MARKET -- {label}")
        t = t.assign(
            ats_pct=(t["ats_pct"] * 100).round(1), ou_pct=(t["ou_pct"] * 100).round(1)
        )
        print(t[cols].round(2).to_string(index=False))
    print(
        "\nats / ou: won-lost-push against the line stored with the forecast, and % "
        "of decided bets; *_close: against the closing line. clv: average points the "
        "line moved toward the pick after it was published; vs_close: picks that got "
        "a better / the same / a worse number than the close. Breaking even at -110 "
        "takes 52.4%."
    )
    print(
        f"\nTables: {REPORT_DIR / 'live_report.csv'}, {REPORT_DIR / 'live_picks.csv'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
