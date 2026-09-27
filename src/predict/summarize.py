"""
Plain-text views of forecast records, for the places the scheduled pipeline
reports to: the GitHub Actions run page, and the message of the commit that
records a forecast (.github/workflows/pipeline.yml).

    python -m src.predict.summarize data/predictions/2026_wk03.parquet
    python -m src.predict.summarize --commit-message data/predictions/2026_wk03.parquet
    python -m src.predict.summarize --leaderboard data/processed/reports/leaderboard.csv

Reads records only; never writes one.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import live_settings
from src.predict.injury_readiness import et, is_provisional


def _record(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df.sort_values(["kickoff", "game_id"], na_position="last")


def _pair(r, prefix: str) -> str:
    home, away = r.get(f"{prefix}_home"), r.get(f"{prefix}_away")
    if pd.isna(home) or pd.isna(away):
        return "--"
    return f"{away:.1f}-{home:.1f}"


def _is_final(r) -> bool:
    return not is_provisional(r.get("injury_report_final"))


def latest_run(df: pd.DataFrame) -> pd.DataFrame:
    """The rows written by the most recent run: every run stamps the games it
    forecast with one generated_at, and leaves the others as they were."""
    return df[df["generated_at"] == df["generated_at"].max()]


def _num(x) -> str:
    return "--" if pd.isna(x) else f"{x:.1f}"


def _fav(team, by) -> str:
    return f"{team} -{by:.1f}" if team else "PK"


def _side(team, line) -> str:
    return f"{team} PK" if line == 0 else f"{team} {line:+.1f}"


def lines_and_picks(r, comp: str):
    """(market, model line, spread pick, total pick) as text, via the ledger's
    own market_picks, so the run page and the ledger can never disagree."""
    from src.predict.build_report import market_picks

    p = market_picks(
        r.get(f"{comp}_home"),
        r.get(f"{comp}_away"),
        r.get("spread_line"),
        r.get("total_line"),
        r["home_team"],
        r["away_team"],
    )
    if p is None:
        return ("no line", "--", "--", "--")
    return (
        f"{_fav(p['market']['favorite'], p['market']['by'])} / {p['market']['total']:.1f}",
        f"{_fav(p['model']['favorite'], p['model']['by'])} / {p['model']['total']:.1f}",
        (
            f"{_side(p['spread']['team'], p['spread']['line'])} (edge {p['spread']['edge']:.1f})"
            if p["spread"]
            else "none"
        ),
        (
            f"{p['total']['side']} {p['total']['line']:.1f} (edge {p['total']['edge']:.1f})"
            if p["total"]
            else "none"
        ),
    )


def week_markdown(path) -> str:
    df = _record(path)
    comp = live_settings()["composite"]["name"]
    season, week = int(df["season"].iloc[0]), int(df["week"].iloc[0])
    lines = [
        f"### {season} week {week}: {len(df)} games on file",
        "",
        f"| kickoff | matchup | {comp} | market | model line | spread pick "
        "| total pick | injury report | forecast made |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        cells = [
            et(r["kickoff"]) if pd.notna(r["kickoff"]) else "?",
            f"{r['away_team']} @ {r['home_team']}",
            _pair(r, comp),
            *lines_and_picks(r, comp),
            "final" if _is_final(r) else "**provisional**",
            et(pd.Timestamp(r["generated_at"])),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Scores read away-home; `{comp}` is the published forecast. Market is the line "
        "when the forecast was made (GB -5.5 = GB favoured by 5.5); model line is the "
        "forecast in the same terms; the picks are the sides the forecast takes against "
        "the market line, and edge is the gap in points. The model's picks, not advice: "
        "in the backtest they have won about half the time. A provisional row was "
        "forecast before its final injury report.",
    ]
    return "\n".join(lines)


def commit_message(paths) -> str:
    """Subject + body for the commit that records this run's forecasts."""
    comp = live_settings()["composite"]
    parts, bodies = [], []
    for path in paths:
        df = _record(path)
        run = latest_run(df)
        season, week = int(df["season"].iloc[0]), int(df["week"].iloc[0])
        parts.append(f"{season} week {week}")
        when = pd.Timestamp(run["generated_at"].iloc[0])
        n_final = sum(_is_final(r) for _, r in run.iterrows())
        body = [
            f"{season} week {week}: {len(run)} game(s) forecast {when:%Y-%m-%d %H:%M} UTC, "
            f"{n_final} on their final injury report.",
            "",
        ]
        for _, r in run.iterrows():
            body.append(
                f"  {r['away_team']:>3} @ {r['home_team']:<3}  "
                f"{_pair(r, comp['name']):>11}  total {_num(r.get(comp['name'] + '_total')):>5}"
                f"   market {lines_and_picks(r, comp['name'])[0]:>14}"
                f"   picks {lines_and_picks(r, comp['name'])[2]}, "
                f"{lines_and_picks(r, comp['name'])[3]}"
                + ("" if _is_final(r) else "  (provisional)")
            )
        bodies.append("\n".join(body))
    subject = f"Forecast: {', '.join(parts)} (automated)"
    footer = (
        f"Scores away-home; {comp['name']} = mean of {', '.join(comp['members'])}.\n"
        "Written by the scheduled pipeline (.github/workflows/pipeline.yml)."
    )
    return "\n\n".join([subject] + bodies + [footer]) + "\n"


def leaderboard_markdown(csv_path, top: int = 10) -> str:
    lb = pd.read_csv(csv_path).head(top)
    lines = [
        "### Walk-forward leaderboard",
        "",
        "| rank | model | games | RMSE | vs baseline [95% CI] | winners |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in lb.iterrows():
        ci = (
            f"{r['vs_baseline']:+.3f} [{r['vs_base_lo']:+.3f}, {r['vs_base_hi']:+.3f}]"
            if pd.notna(r.get("vs_baseline"))
            else "--"
        )
        rank = "--" if pd.isna(r["rank"]) else str(int(r["rank"]))
        lines.append(
            f"| {rank} | {r['model']} | {int(r['games'])} | {r['rmse']:.3f} "
            f"| {ci} | {r['winner_pct']:.1%} |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--commit-message", action="store_true")
    ap.add_argument("--leaderboard", type=Path)
    args = ap.parse_args(argv)
    if args.leaderboard:
        print(leaderboard_markdown(args.leaderboard))
    if args.commit_message:
        if not args.paths:
            ap.error("--commit-message needs the changed record(s)")
        sys.stdout.write(commit_message(args.paths))
    elif args.paths:
        print("\n\n".join(week_markdown(p) for p in args.paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
