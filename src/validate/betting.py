"""
Market-relative and economic evaluation of every saved model.

Why this is separate from model_scoreboard.py. That file answers "how close are
the predicted scores?" This one answers a different and much harder question:
"would acting on these predictions have made money?" They are not the same
question and a model can be good at the first while being worthless at the
second, which is the single most important thing to keep straight about this
project. RMSE measures distance from the truth; betting measures distance from
the MARKET, and the market is a far better forecaster than the truth is random.

Definitions used here, stated because they are easy to get subtly wrong:

  spread_line   the home margin the market expects, positive = home favoured.
                A model "picks home" against the spread when its predicted
                margin exceeds that number.
  push          the result lands exactly on the line. Not a win, not a loss --
                excluded from the denominator rather than counted as half.
  break-even    at the standard -110 price you risk 110 to win 100, so you must
                win 110/210 = 52.38% of bets just to break even. A 51% model is
                not "slightly profitable"; it is losing money steadily.
  ROI           units returned per unit risked at -110: (0.909 x W - L) / (W+L).
  edge          |predicted margin - spread|. The honest test of whether a model
                knows anything the market does not is whether its accuracy
                IMPROVES as this grows. A flat or falling curve means the
                disagreements are noise, not information.

Everything is measured on the same walk-forward predictions the scoreboard
uses, so nothing here has seen its own test data. Market lines are the CLOSING
line and were never model inputs.

Run: python -m src.validate.betting
     python -m src.validate.betting poisson gp
     python -m src.validate.betting --min-edge 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRED_DIR = Path("data/processed")
SCHEDULES = Path("data/raw/schedules.parquet")

VIG_ODDS = -110
BREAK_EVEN = 110 / 210  # 0.5238
WIN_UNITS = 100 / 110  # 0.909 per unit risked
N_BOOT = 5000
SEED = 42
EDGE_BUCKETS = [(0, 1), (1, 3), (3, 6), (6, 99)]


def discover() -> dict[str, Path]:
    return {
        p.name.replace("_predictions.parquet", ""): p
        for p in sorted(PRED_DIR.glob("*_predictions.parquet"))
    }


def load(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    s = pd.read_parquet(SCHEDULES)[["game_id", "spread_line", "total_line"]]
    m = df.merge(s, on="game_id", how="left").dropna(subset=["spread_line"])

    m["pred_margin"] = m["home_pred"] - m["away_pred"]
    m["true_margin"] = m["home_score"] - m["away_score"]
    m["pred_total"] = m["home_pred"] + m["away_pred"]
    m["true_total"] = m["home_score"] + m["away_score"]
    # Implied market score pair, so the market can be scored on the same scale.
    m["mkt_margin"] = m["spread_line"]
    m["edge_spread"] = (m["pred_margin"] - m["spread_line"]).abs()
    m["edge_total"] = (m["pred_total"] - m["total_line"]).abs()
    return m


def _record(picked_over: pd.Series, landed_over: pd.Series, push: pd.Series) -> dict:
    live = ~push
    wins = int((picked_over[live] == landed_over[live]).sum())
    n = int(live.sum())
    losses = n - wins
    rate = wins / n if n else float("nan")
    roi = (WIN_UNITS * wins - losses) / n if n else float("nan")
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": int(push.sum()),
        "rate": rate,
        "roi": roi,
    }


def ats(m: pd.DataFrame) -> dict:
    """Against the spread. Model picks home when it beats the line."""
    return _record(
        picked_over=m["pred_margin"] > m["spread_line"],
        landed_over=m["true_margin"] > m["spread_line"],
        push=m["true_margin"] == m["spread_line"],
    )


def totals(m: pd.DataFrame) -> dict:
    return _record(
        picked_over=m["pred_total"] > m["total_line"],
        landed_over=m["true_total"] > m["total_line"],
        push=m["true_total"] == m["total_line"],
    )


def straight_up(m: pd.DataFrame) -> dict:
    """Winner only. Context, not a bet -- the favourite wins ~2 games in 3, so a
    good straight-up record says nothing the line had not already said."""
    live = m["true_margin"] != 0
    wins = int(((m["pred_margin"] > 0) == (m["true_margin"] > 0))[live].sum())
    n = int(live.sum())
    return {"n": n, "wins": wins, "rate": wins / n if n else float("nan")}


def boot_rate_ci(picked, landed, push):
    """Bootstrap CI on the win rate, so 52% over 1,400 games can be told apart
    from 52% over 40."""
    live = ~push
    hit = (picked[live] == landed[live]).to_numpy()
    if not len(hit):
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(hit), (N_BOOT, len(hit)))
    rates = hit[draws].mean(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def edge_curve(m: pd.DataFrame) -> list[dict]:
    """ATS accuracy by how far the model disagrees with the line.

    This is the diagnostic that matters. If a model has real information, the
    games where it disagrees MOST with the market should be the ones it gets
    right most often. A flat curve means the disagreements are noise.
    """
    out = []
    picked = m["pred_margin"] > m["spread_line"]
    landed = m["true_margin"] > m["spread_line"]
    push = m["true_margin"] == m["spread_line"]
    for lo, hi in EDGE_BUCKETS:
        sel = (m["edge_spread"] >= lo) & (m["edge_spread"] < hi)
        r = _record(picked[sel], landed[sel], push[sel])
        out.append({"bucket": f"{lo}-{hi if hi < 99 else '+'}", **r})
    return out


def market_comparison(m: pd.DataFrame) -> dict:
    """Is the model or the closing line closer to what happened?"""

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "model_margin_rmse": rmse(m["pred_margin"], m["true_margin"]),
        "market_margin_rmse": rmse(m["mkt_margin"], m["true_margin"]),
        "model_total_rmse": rmse(m["pred_total"], m["true_total"]),
        "market_total_rmse": rmse(m["total_line"], m["true_total"]),
        # Share of games where the model landed closer than the line did.
        "model_closer_margin": float(
            np.mean(
                (m["pred_margin"] - m["true_margin"]).abs()
                < (m["mkt_margin"] - m["true_margin"]).abs()
            )
        ),
    }


def report(name: str, m: pd.DataFrame, min_edge: float) -> dict:
    if min_edge:
        m = m[m["edge_spread"] >= min_edge]
    a, t, su = ats(m), totals(m), straight_up(m)
    lo, hi = boot_rate_ci(
        m["pred_margin"] > m["spread_line"],
        m["true_margin"] > m["spread_line"],
        m["true_margin"] == m["spread_line"],
    )
    mc = market_comparison(m)
    return {
        "model": name,
        "n": a["n"],
        "ats": a["rate"],
        "ats_lo": lo,
        "ats_hi": hi,
        "ats_roi": a["roi"],
        "ou": t["rate"],
        "ou_roi": t["roi"],
        "su": su["rate"],
        **mc,
        "edges": edge_curve(m),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="default: every saved model")
    ap.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="only count games where the model disagrees with the line by at "
        "least this many points",
    )
    ap.add_argument(
        "--edge-curve",
        action="store_true",
        help="print the ATS-by-edge breakdown for each model",
    )
    args = ap.parse_args(argv)

    found = discover()
    names = args.models or sorted(found)
    missing = [n for n in names if n not in found]
    if missing:
        print(f"ERROR: no predictions for {missing}. Available: {sorted(found)}")
        return 1

    rows = [report(n, load(n, found[n]), args.min_edge) for n in names]
    rows.sort(key=lambda r: -r["ats"])

    if args.min_edge:
        print(
            f"Restricted to games where the model disagrees with the line by "
            f">= {args.min_edge} points.\n"
        )
    print(
        f"{'model':<12}{'n':>6}{'ATS':>8}{'95% CI':>18}{'ROI':>8}"
        f"{'O/U':>8}{'ROI':>8}{'SU':>7}   verdict"
    )
    print("-" * 92)
    for r in rows:
        beats = r["ats_lo"] > BREAK_EVEN
        losing = r["ats_hi"] < BREAK_EVEN
        verdict = "PROFITABLE" if beats else ("unprofitable" if losing else "coin flip")
        ci = f"[{r['ats_lo']:.1%}, {r['ats_hi']:.1%}]"
        print(
            f"{r['model']:<12}{r['n']:>6}{r['ats']:>8.1%}{ci:>18}"
            f"{r['ats_roi']:>+8.1%}{r['ou']:>8.1%}{r['ou_roi']:>+8.1%}"
            f"{r['su']:>7.1%}   {verdict}"
        )

    print(
        f"\nBreak-even at {VIG_ODDS} is {BREAK_EVEN:.2%}. Anything below that loses "
        f"money however good it looks."
    )

    print("\n\nMODEL vs THE CLOSING LINE (margin and total RMSE, lower is better)")
    print("-" * 92)
    print(
        f"{'model':<12}{'margin':>10}{'mkt':>10}{'total':>10}{'mkt':>10}"
        f"{'model closer on margin':>26}"
    )
    for r in rows:
        print(
            f"{r['model']:<12}{r['model_margin_rmse']:>10.3f}"
            f"{r['market_margin_rmse']:>10.3f}{r['model_total_rmse']:>10.3f}"
            f"{r['market_total_rmse']:>10.3f}{r['model_closer_margin']:>25.1%}"
        )

    if args.edge_curve:
        print("\n\nATS BY DISAGREEMENT WITH THE LINE")
        print("If the model knows something the market does not, accuracy should")
        print("RISE with edge. A flat curve means the disagreements are noise.")
        for r in rows:
            print(f"\n  {r['model']}")
            print(f"    {'edge':>8}{'n':>7}{'ATS':>9}{'ROI':>9}")
            for e in r["edges"]:
                if not e["n"]:
                    continue
                print(
                    f"    {e['bucket']:>8}{e['n']:>7}{e['rate']:>9.1%}"
                    f"{e['roi']:>+9.1%}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
