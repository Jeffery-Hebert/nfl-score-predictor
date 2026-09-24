"""
src/validate/model_report.py

The deep evaluation: every saved backtest, broken down by where and how it
misses. model_scoreboard.py ranks models; error_analysis.py describes one at a
time; this answers the questions neither can:

  - which models CONSISTENTLY miss in one direction, and on which side;
  - how big the misses are -- mean, median, the high end, the low end -- on the
    home score, the away score, the total and the margin;
  - whether accuracy depends on WHEN a game kicks off (early Sunday, late
    Sunday, prime time, international mornings);
  - how accuracy moves across the WEEKS of a season;
  - which TEAMS the models systematically over- or under-rate;
  - how redundant the models are with each other, how they compare with the
    closing line, and whether a smarter composite would beat equal weights.

Statistics. Every confidence interval here is a BLOCK bootstrap over weeks:
games in the same week share a fitted model, a slate of weather and one
scoring environment, so they are not independent draws, and resampling single
games (what the older scripts do) makes intervals too narrow. Error sign is
prediction minus reality throughout: positive = the model predicted too high.

Fair comparison. Breakdowns use the games EVERY included model predicted
(stacking, which starts a season later by design, is scored on its own games and
kept out of the intersection so it does not cost everyone else 2021).

Run: python -m src.validate.model_report
     python -m src.validate.model_report --models poisson gp linear composite
     python -m src.validate.model_report --boot 2000      # faster, noisier CIs
Output: terminal summary, plus data/processed/reports/model_report.html and one
        CSV per table in the same folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.validate.backtest_io import predictions_path, stale_inputs

REPORT_DIR = Path("data/processed/reports")
SCHEDULES = Path("data/raw/schedules.parquet")
N_BOOT = 5000
SEED = 42
TEAM_CODE_MAP = {"OAK": "LV"}
# Scored on its own games and kept out of the common-games intersection -- its
# walk-forward starts a season later by design (it trains on OOF predictions).
OWN_GAMES_ONLY = {"stacking"}
SIDES = ("home", "away")
TARGETS = ("home", "away", "total", "margin")


# ------------------------------------------------------------------ loading


def available_models() -> list[str]:
    from src.models import registry

    return [n for n in registry.names() if predictions_path(n).exists()]


def kickoff_slot(weekday: int, gametime: str | None) -> str:
    """Broadcast window from weekday (Mon=0 .. Sun=6) and kickoff time (ET).

    Windows are the ones the NFL schedules around, because they differ in more
    than the clock: prime-time games are single-game national broadcasts,
    Thursday games come on three or four days' rest, and international games
    kick off in the morning after transatlantic travel.
    """
    if gametime is None or (isinstance(gametime, float) and np.isnan(gametime)):
        return "unknown"
    hour = int(str(gametime).split(":")[0])
    if hour < 12:
        return "international/morning"
    if weekday == 6:  # Sunday
        if hour < 15:
            return "Sun early (1pm)"
        if hour < 19:
            return "Sun late (4pm)"
        return "Sun night"
    if weekday == 0:
        return "Mon night" if hour >= 19 else "Mon day"
    if weekday == 3:
        return "Thu night" if hour >= 19 else "Thu day (Thanksgiving)"
    if weekday == 5:
        return "Saturday"
    return "other weekday"


def load_long(models: list[str]) -> pd.DataFrame:
    """One row per (model, game) with predictions, results and schedule context."""
    sched = pd.read_parquet(SCHEDULES)
    sched = sched[
        [
            "game_id",
            "gameday",
            "gametime",
            "game_type",
            "location",
            "home_team",
            "away_team",
            "spread_line",
            "total_line",
        ]
    ].copy()
    for c in ("home_team", "away_team"):
        sched[c] = sched[c].replace(TEAM_CODE_MAP)
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched["weekday"] = sched["gameday"].dt.dayofweek
    sched["slot"] = [
        kickoff_slot(w, t) for w, t in zip(sched["weekday"], sched["gametime"])
    ]
    sched["primetime"] = sched["slot"].isin(["Thu night", "Sun night", "Mon night"])

    frames = []
    for name in models:
        p = pd.read_parquet(predictions_path(name))
        p = p[
            ["game_id", "season", "week", "home_score", "away_score"]
            + ["home_pred", "away_pred"]
        ]
        p.insert(0, "model", name)
        frames.append(p)
    df = pd.concat(frames, ignore_index=True).merge(sched, on="game_id", how="left")
    df["home_err"] = df["home_pred"] - df["home_score"]
    df["away_err"] = df["away_pred"] - df["away_score"]
    df["total_err"] = df["home_err"] + df["away_err"]
    df["margin_err"] = (df["home_pred"] - df["away_pred"]) - (
        df["home_score"] - df["away_score"]
    )
    df["block"] = df["season"].astype(int) * 100 + df["week"].astype(int)
    return df


def common_games(df: pd.DataFrame) -> set:
    core = [m for m in df["model"].unique() if m not in OWN_GAMES_ONLY]
    sets = [set(df.loc[df["model"] == m, "game_id"]) for m in core]
    return set.intersection(*sets) if sets else set()


# ------------------------------------------------------- block bootstrap core


class BlockBootstrap:
    """Resampling weeks, not games, via multinomial block counts.

    Block sums are precomputed once per statistic, so each of the thousands of
    resamples is a single matrix product rather than a Python loop.
    """

    def __init__(self, blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
        self.codes, self.index = np.unique(blocks, return_inverse=True)
        n = len(self.codes)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)

    def _sums(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.index, weights=x, minlength=len(self.codes))

    def mean_ci(self, x: np.ndarray) -> tuple[float, float]:
        s, n = self._sums(x), self._sums(np.ones_like(x))
        boot = (self.counts @ s) / (self.counts @ n)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return float(lo), float(hi)

    def rmse_ci(self, e: np.ndarray) -> tuple[float, float]:
        sq, n = self._sums(e**2), self._sums(np.ones_like(e))
        boot = np.sqrt((self.counts @ sq) / (self.counts @ n))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return float(lo), float(hi)

    def rmse_delta(self, e_cand: np.ndarray, e_ref: np.ndarray):
        """RMSE(cand) - RMSE(ref), paired: both resampled with the same weeks.
        Returns (point, lo, hi, P(cand better))."""
        n = self._sums(np.ones_like(e_cand))
        sc, sr = self._sums(e_cand**2), self._sums(e_ref**2)
        bn = self.counts @ n
        boot = np.sqrt((self.counts @ sc) / bn) - np.sqrt((self.counts @ sr) / bn)
        point = float(np.sqrt(np.mean(e_cand**2)) - np.sqrt(np.mean(e_ref**2)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return point, float(lo), float(hi), float((boot < 0).mean())


def pooled_errors(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Home and away errors stacked, with each game's block repeated -- a
    resampled week brings both scores of every game in it."""
    e = np.concatenate([g["home_err"].to_numpy(float), g["away_err"].to_numpy(float)])
    b = np.concatenate([g["block"].to_numpy(), g["block"].to_numpy()])
    return e, b


def rmse(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2)))


# ------------------------------------------------------------------ sections


def leaderboard(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    ref = "baseline" if "baseline" in df["model"].unique() else None
    for m, g in df.groupby("model", sort=False):
        e, b = pooled_errors(g)
        actual_margin = g["home_score"] - g["away_score"]
        pred_margin = g["home_pred"] - g["away_pred"]
        decided = actual_margin != 0
        slope = np.polyfit(
            np.concatenate([g["home_pred"], g["away_pred"]]),
            np.concatenate([g["home_score"], g["away_score"]]),
            1,
        )[0]
        row = {
            "model": m,
            "games": len(g),
            "rmse": rmse(e),
            "home_rmse": rmse(g["home_err"]),
            "away_rmse": rmse(g["away_err"]),
            "mae": float(np.mean(np.abs(e))),
            "margin_rmse": rmse(g["margin_err"]),
            "total_rmse": rmse(g["total_err"]),
            "home_bias": float(g["home_err"].mean()),
            "away_bias": float(g["away_err"].mean()),
            "total_bias": float(g["total_err"].mean()),
            "calib_slope": float(slope),
            "winner_pct": float(
                ((pred_margin > 0) == (actual_margin > 0))[decided].mean()
            ),
        }
        if ref and m != ref:
            r = df[df["model"] == ref].set_index("game_id").loc[g["game_id"]]
            er, _ = pooled_errors(r.reset_index())
            bb = BlockBootstrap(b, n_boot)
            d, lo, hi, p = bb.rmse_delta(e, er)
            row.update(
                {"vs_baseline": d, "vs_base_lo": lo, "vs_base_hi": hi, "p_better": p}
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    # Rank only models scored on the SAME games. Stacking's walk-forward starts
    # a season later, and 2021 happens to be a harder season to predict, so its
    # raw RMSE looks best without being comparable; it is listed unranked, and
    # model_scoreboard.py --common-games compares everything on its games.
    same = out[~out["model"].isin(OWN_GAMES_ONLY)].sort_values("rmse")
    own = out[out["model"].isin(OWN_GAMES_ONLY)]
    out = pd.concat([same, own], ignore_index=True)
    out.insert(0, "rank", list(range(1, len(same) + 1)) + [None] * len(own))
    out["rank"] = out["rank"].astype("Int64")
    return out


def miss_distribution(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """How much, and which way, each model misses -- per target."""
    rows = []
    for m, g in df.groupby("model", sort=False):
        bb = BlockBootstrap(g["block"].to_numpy(), n_boot)
        for t in TARGETS:
            e = g[f"{t}_err"].to_numpy(float)
            lo, hi = bb.mean_ci(e)
            direction = (
                "OVER-predicts" if lo > 0 else "UNDER-predicts" if hi < 0 else "-"
            )
            rows.append(
                {
                    "model": m,
                    "target": t,
                    "n": len(e),
                    "mean_err": e.mean(),
                    "bias_lo": lo,
                    "bias_hi": hi,
                    "consistent_direction": direction,
                    "median_err": float(np.median(e)),
                    "mae": float(np.mean(np.abs(e))),
                    "median_abs_err": float(np.median(np.abs(e))),
                    "rmse": rmse(e),
                    "p05_err": float(np.percentile(e, 5)),
                    "p95_err": float(np.percentile(e, 95)),
                    "worst_under": float(e.min()),
                    "worst_over": float(e.max()),
                    "pct_over": float((e > 0).mean()),
                    "pct_within_7": float((np.abs(e) <= 7).mean()),
                }
            )
    return pd.DataFrame(rows)


def bias_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signed error per season: a bias that keeps its sign every season is a
    property of the model, not of one season's luck."""
    t = (
        df.groupby(["model", "season"])[["home_err", "away_err", "total_err"]]
        .mean()
        .reset_index()
    )
    summary = []
    for m, g in t.groupby("model", sort=False):
        rec = {"model": m, "seasons": len(g)}
        for side in ("home", "away", "total"):
            s = np.sign(g[f"{side}_err"])
            overall = np.sign(df.loc[df["model"] == m, f"{side}_err"].mean())
            rec[f"{side}_same_sign_seasons"] = int((s == overall).sum())
            rec[f"{side}_overall_bias"] = float(
                df.loc[df["model"] == m, f"{side}_err"].mean()
            )
        summary.append(rec)
    return t, pd.DataFrame(summary)


def by_group(df: pd.DataFrame, key: str, n_boot: int) -> pd.DataFrame:
    """Accuracy per model within each value of `key`, with a week-blocked CI on
    the group's RMSE -- small groups (international games) get wide intervals,
    which is the point: they should not be read like a 400-game slot."""
    rows = []
    for (m, k), g in df.groupby(["model", key], sort=True):
        e, b = pooled_errors(g)
        lo, hi = BlockBootstrap(b, n_boot).rmse_ci(e)
        rows.append(
            {
                "model": m,
                key: k,
                "games": len(g),
                "rmse": rmse(e),
                "rmse_lo": lo,
                "rmse_hi": hi,
                "mae": float(np.mean(np.abs(e))),
                "home_bias": float(g["home_err"].mean()),
                "away_bias": float(g["away_err"].mean()),
                "total_bias": float(g["total_err"].mean()),
                "actual_total": float((g["home_score"] + g["away_score"]).mean()),
                "margin_rmse": rmse(g["margin_err"]),
            }
        )
    return pd.DataFrame(rows)


def team_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, game, team) from that team's point of view.

    scored_err   error on the team's OWN score  (+ = model over-rated its offence)
    allowed_err  error on its OPPONENT's score (+ = model under-rated its defence)
    margin_err   error on the team's margin    (+ = model over-rated the team)
    """
    home = pd.DataFrame(
        {
            "model": df["model"],
            "game_id": df["game_id"],
            "season": df["season"],
            "block": df["block"],
            "team": df["home_team"],
            "venue": "home",
            "scored_err": df["home_err"],
            "allowed_err": df["away_err"],
            "margin_err": df["margin_err"],
        }
    )
    away = pd.DataFrame(
        {
            "model": df["model"],
            "game_id": df["game_id"],
            "season": df["season"],
            "block": df["block"],
            "team": df["away_team"],
            "venue": "away",
            "scored_err": df["away_err"],
            "allowed_err": df["home_err"],
            "margin_err": -df["margin_err"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def by_team(df: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tl = team_long(df)
    per_model = (
        tl.groupby(["team", "model"])
        .agg(
            games=("game_id", "size"),
            scored_bias=("scored_err", "mean"),
            allowed_bias=("allowed_err", "mean"),
            margin_bias=("margin_err", "mean"),
            scored_mae=("scored_err", lambda x: np.abs(x).mean()),
            allowed_mae=("allowed_err", lambda x: np.abs(x).mean()),
            margin_rmse=("margin_err", rmse),
        )
        .reset_index()
    )
    # Consensus across models: does EVERY model lean the same way on a team?
    rows = []
    for team, g in tl.groupby("team"):
        models = g["model"].unique()
        cons = {"team": team, "games": int(g["game_id"].nunique())}
        for col in ("scored_err", "allowed_err", "margin_err"):
            per = g.groupby("model")[col].mean()
            cons[col.replace("_err", "_bias_avg")] = float(per.mean())
            cons[col.replace("_err", "_same_sign")] = int(
                (np.sign(per) == np.sign(per.mean())).sum()
            )
        # CI on the model-averaged margin bias for this team, week-blocked.
        avg = g.groupby(["game_id", "block"])["margin_err"].mean().reset_index()
        lo, hi = BlockBootstrap(avg["block"].to_numpy(), n_boot).mean_ci(
            avg["margin_err"].to_numpy(float)
        )
        cons.update(
            {"margin_bias_lo": lo, "margin_bias_hi": hi, "n_models": len(models)}
        )
        cons["verdict"] = (
            "models OVER-rate" if lo > 0 else "models UNDER-rate" if hi < 0 else "-"
        )
        rows.append(cons)
    consensus = pd.DataFrame(rows).sort_values("margin_bias_avg").reset_index(drop=True)
    return per_model, consensus


def error_correlation(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="game_id", columns="model", values="home_err")
    return wide.corr()


def market_comparison(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    d = df.dropna(subset=["spread_line", "total_line"])
    for m, g in d.groupby("model", sort=False):
        true_margin = g["home_score"] - g["away_score"]
        true_total = g["home_score"] + g["away_score"]
        pred_margin = g["home_pred"] - g["away_pred"]
        pred_total = g["home_pred"] + g["away_pred"]
        live = true_margin != g["spread_line"]
        hit = ((pred_margin > g["spread_line"]) == (true_margin > g["spread_line"]))[
            live
        ]
        lo, hi = BlockBootstrap(g.loc[live, "block"].to_numpy(), n_boot).mean_ci(
            hit.to_numpy(float)
        )
        live_t = true_total != g["total_line"]
        ou = ((pred_total > g["total_line"]) == (true_total > g["total_line"]))[live_t]
        rows.append(
            {
                "model": m,
                "games": len(g),
                "model_margin_rmse": rmse(pred_margin - true_margin),
                "market_margin_rmse": rmse(g["spread_line"] - true_margin),
                "model_total_rmse": rmse(pred_total - true_total),
                "market_total_rmse": rmse(g["total_line"] - true_total),
                "ats_pct": float(hit.mean()),
                "ats_lo": lo,
                "ats_hi": hi,
                "ou_pct": float(ou.mean()),
                "closer_than_line_on_margin": float(
                    (
                        (pred_margin - true_margin).abs()
                        < (g["spread_line"] - true_margin).abs()
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_margin_rmse").reset_index(drop=True)


def composite_alternatives(df: pd.DataFrame, members: list[str], n_boot: int):
    """Equal weights against weights fitted IN-FOLD on earlier weeks only.

    Every candidate is walk-forward: a week's weights come only from the
    out-of-sample predictions of earlier weeks, and the first season is burn-in
    for every candidate alike. Scored on identical games.
    """
    from scipy.optimize import nnls

    have = [m for m in members if m in df["model"].unique()]
    if len(have) < 2:
        return pd.DataFrame()
    wide = {}
    for side in SIDES:
        w = df[df["model"].isin(have)].pivot_table(
            index=["game_id", "season", "week", "block", f"{side}_score"],
            columns="model",
            values=f"{side}_pred",
        )
        wide[side] = w.dropna().reset_index()
    base = wide["home"][["game_id", "season", "week", "block"]]
    first = base["season"].min()
    weeks = base.loc[base["season"] > first, "block"].drop_duplicates().sort_values()

    preds = {k: [] for k in ("equal", "inv_mse", "nnls", "best_single")}
    block_ids = []
    for blk in weeks:
        rows = {}
        for side in SIDES:
            w = wide[side]
            tr, te = w[w["block"] < blk], w[w["block"] == blk]
            X, y = tr[have].to_numpy(float), tr[f"{side}_score"].to_numpy(float)
            Xt = te[have].to_numpy(float)
            mse = ((X - y[:, None]) ** 2).mean(axis=0)
            inv = (1 / mse) / (1 / mse).sum()
            coef, _ = nnls(X, y)
            coef = (
                coef / coef.sum()
                if coef.sum() > 0
                else np.full(len(have), 1 / len(have))
            )
            best = int(np.argmin(mse))
            rows[side] = {
                "equal": Xt.mean(axis=1),
                "inv_mse": Xt @ inv,
                "nnls": Xt @ coef,
                "best_single": Xt[:, best],
                "actual": te[f"{side}_score"].to_numpy(float),
                "block": te["block"].to_numpy(),
            }
        for k in preds:
            preds[k].append(
                np.concatenate(
                    [
                        rows["home"][k] - rows["home"]["actual"],
                        rows["away"][k] - rows["away"]["actual"],
                    ]
                )
            )
        block_ids.append(np.concatenate([rows["home"]["block"], rows["away"]["block"]]))
    blocks = np.concatenate(block_ids)
    errs = {k: np.concatenate(v) for k, v in preds.items()}
    bb = BlockBootstrap(blocks, n_boot)
    out = []
    for k, e in errs.items():
        d, lo, hi, p = bb.rmse_delta(e, errs["equal"])
        out.append(
            {
                "composite": k,
                "rmse": rmse(e),
                "vs_equal": d,
                "lo": lo,
                "hi": hi,
                "p_better_than_equal": p,
                "errors": len(e),
            }
        )
    return pd.DataFrame(out).sort_values("rmse").reset_index(drop=True)


# ------------------------------------------------------------------- output


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            if (
                c.endswith("pct")
                or c.startswith("pct_")
                or c
                in (
                    "p_better",
                    "p_better_than_equal",
                    "closer_than_line_on_margin",
                    "ats_lo",
                    "ats_hi",
                )
            ):
                out[c] = out[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
            else:
                out[c] = out[c].map(
                    lambda v: (
                        f"{v:+.3f}"
                        if pd.notna(v)
                        and (
                            "bias" in c
                            or "err" in c
                            or c.startswith("vs_")
                            or c in ("lo", "hi")
                        )
                        else (f"{v:.3f}" if pd.notna(v) else "")
                    )
                )
    return out


HTML_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Report</title><style>
:root{--bg:#f6f7f9;--fg:#15181e;--soft:#5b6472;--rule:#dde1e7;--card:#fff;--hi:#b06e00}
@media (prefers-color-scheme:dark){:root{--bg:#11151b;--fg:#e6eaf0;--soft:#9aa4b2;--rule:#28303b;--card:#171c24;--hi:#f0b429}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1200px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 6px;border-bottom:2px solid var(--fg);padding-bottom:4px}
p,li{color:var(--soft);max-width:80ch}.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--rule);border-radius:4px;margin:8px 0}
table{border-collapse:collapse;font:12.5px/1.35 ui-monospace,monospace;width:100%}th,td{padding:5px 9px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--card)}td:first-child,th:first-child{text-align:left}b{color:var(--hi)}
</style></head><body><main>"""


def write_html(sections: list[tuple[str, str, pd.DataFrame]], path: Path, header: str):
    parts = [HTML_HEAD, header]
    for title, blurb, table in sections:
        parts.append(f"<h2>{title}</h2><p>{blurb}</p>")
        parts.append(
            '<div class="wrap">'
            + _fmt(table).to_html(index=False, border=0, escape=True)
            + "</div>"
        )
    parts.append("</main></body></html>")
    path.write_text("\n".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    models = args.models or available_models()
    missing = [m for m in models if not predictions_path(m).exists()]
    if missing:
        print(
            f"ERROR: no saved backtest for {missing}. Run: python -m src.models.run_all"
        )
        return 1
    stale = {m: stale_inputs(m) for m in models}
    stale = {m: s for m, s in stale.items() if s}
    for m, s in stale.items():
        print(f"WARNING: {m} backtest is stale -- {s[0]}")

    full = load_long(models)
    shared = common_games(full)
    df = full[full["game_id"].isin(shared) & ~full["model"].isin(OWN_GAMES_ONLY)]
    print(
        f"{len(models)} models; {len(shared)} games shared by every model except {sorted(OWN_GAMES_ONLY)}\n"
    )

    lb = leaderboard(
        pd.concat([df, full[full["model"].isin(OWN_GAMES_ONLY)]]), args.boot
    )
    miss = miss_distribution(df, args.boot)
    season_bias, season_summary = bias_by_season(df)
    slots = by_group(df, "slot", args.boot)
    prime = by_group(
        df.assign(window=np.where(df["primetime"], "prime time", "daytime")),
        "window",
        args.boot,
    )
    weeks = by_group(df, "week", args.boot)
    team_models, team_consensus = by_team(df, args.boot)
    corr = error_correlation(df)
    market = market_comparison(df, args.boot)

    from src.config import live_settings

    members = live_settings()["composite"]["members"]
    comp = composite_alternatives(
        full[~full["model"].isin(OWN_GAMES_ONLY)], members, args.boot
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "leaderboard": lb,
        "miss_distribution": miss,
        "bias_by_season": season_bias,
        "bias_direction_summary": season_summary,
        "by_kickoff_slot": slots,
        "primetime_vs_day": prime,
        "by_week": weeks,
        "by_team_per_model": team_models,
        "by_team_consensus": team_consensus,
        "error_correlation": corr.reset_index(),
        "market_comparison": market,
        "composite_alternatives": comp,
    }
    for name, t in tables.items():
        t.to_csv(REPORT_DIR / f"{name}.csv", index=False)

    # --------------------------------------------------------- terminal summary
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = [
        "rank",
        "model",
        "games",
        "rmse",
        "home_rmse",
        "away_rmse",
        "mae",
        "margin_rmse",
        "total_rmse",
        "home_bias",
        "away_bias",
        "calib_slope",
        "winner_pct",
        "vs_baseline",
        "vs_base_lo",
        "vs_base_hi",
    ]
    print(
        "LEADERBOARD (pooled home+away RMSE; vs_baseline CI is a week-block bootstrap)"
    )
    print(lb[[c for c in show if c in lb.columns]].round(3).to_string(index=False))
    consistent = miss[miss["consistent_direction"] != "-"]
    print("\nCONSISTENT DIRECTIONAL MISSES (bias CI excludes zero)")
    print(
        consistent[
            [
                "model",
                "target",
                "mean_err",
                "bias_lo",
                "bias_hi",
                "consistent_direction",
            ]
        ]
        .round(3)
        .to_string(index=False)
        if len(consistent)
        else "  none"
    )
    top = (
        lb[~lb["model"].isin(["baseline", "stacking", "composite"])]["model"]
        .head(3)
        .tolist()
    )
    focus = top + [m for m in ("composite",) if m in df["model"].unique()]
    print(f"\nBY KICKOFF SLOT (top models {focus})")
    print(
        slots[slots["model"].isin(focus)]
        .pivot_table(index="slot", columns="model", values="rmse")
        .round(3)
        .join(
            slots[slots["model"] == focus[0]]
            .set_index("slot")[["games", "actual_total", "total_bias"]]
            .round(2)
        )
        .to_string()
    )
    print("\nTEAMS THE MODELS CONSISTENTLY MIS-RATE (margin bias CI excludes zero)")
    flagged = team_consensus[team_consensus["verdict"] != "-"]
    print(
        flagged[
            [
                "team",
                "games",
                "margin_bias_avg",
                "margin_bias_lo",
                "margin_bias_hi",
                "margin_same_sign",
                "n_models",
                "verdict",
            ]
        ]
        .round(2)
        .to_string(index=False)
        if len(flagged)
        else "  none"
    )
    if len(comp):
        print(
            "\nCOMPOSITE ALTERNATIVES (walk-forward weights; negative vs_equal = better than equal weights)"
        )
        print(comp.round(4).to_string(index=False))

    header = (
        "<h1>Model report</h1><p>Every saved walk-forward backtest, broken down by "
        f"where and how it misses. <b>{len(shared)}</b> games shared by every model "
        f"({', '.join(m for m in models if m not in OWN_GAMES_ONLY)}); stacking is "
        "scored on its own games. Error = prediction minus result (positive = "
        "predicted too high). Confidence intervals resample whole weeks.</p>"
    )
    if stale:
        header += (
            "<p><b>Stale backtests:</b> "
            + "; ".join(f"{m}: {s[0]}" for m, s in stale.items())
            + "</p>"
        )
    write_html(
        [
            (
                "Leaderboard",
                "Pooled home+away RMSE, lower is better. vs_baseline: paired, week-blocked bootstrap; negative = better than the rule-based baseline.",
                lb,
            ),
            (
                "Miss distribution",
                "Per model and target: mean (bias) with CI, median, MAE, median absolute error, RMSE, 5th/95th percentile, the single worst under- and over-prediction, share over-predicted, share within a touchdown.",
                miss,
            ),
            (
                "Bias direction across seasons",
                "How many seasons each model's bias kept the sign of its overall bias.",
                season_summary,
            ),
            ("Bias by season", "Mean signed error per season.", season_bias),
            (
                "By kickoff slot",
                "Accuracy by broadcast window (kickoff times are Eastern). actual_total is the real average total in that window.",
                slots,
            ),
            (
                "Prime time vs daytime",
                "Thursday, Sunday and Monday night games against everything else.",
                prime,
            ),
            ("By week", "Accuracy by week of season; weeks 19-22 are playoffs.", weeks),
            (
                "By team -- consensus",
                "From each team's point of view, averaged across models. margin_bias > 0 means the models rated the team too highly. same_sign counts models agreeing with the average's sign.",
                team_consensus,
            ),
            (
                "By team -- per model",
                "scored = the team's own points, allowed = its opponent's.",
                team_models,
            ),
            (
                "Error correlation (home score)",
                "How similarly the models miss. Near 1.0 = redundant.",
                corr.reset_index(),
            ),
            (
                "Against the closing line",
                "Market numbers are the closing line -- never a model input.",
                market,
            ),
            (
                "Composite alternatives",
                f"Walk-forward weighting schemes over the live composite members {members}; negative vs_equal = better than the equal-weight composite.",
                comp,
            ),
        ],
        REPORT_DIR / "model_report.html",
        header,
    )
    print(f"\nFull report: {REPORT_DIR / 'model_report.html'}  (CSV tables beside it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
