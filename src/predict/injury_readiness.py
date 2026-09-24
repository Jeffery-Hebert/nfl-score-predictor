"""
src/predict/injury_readiness.py

Has a game's FINAL injury report been published, and is it in the data we hold?

Why this gates live forecasts. injury_impact is the model's only confirmed
leading indicator (the first feature in the project to clear the promotion
gate), and it is built from the report_status column -- Out, Doubtful,
Questionable. Teams only assign those in the FINAL report before each game:

    Thursday game  -> Wednesday's report     Sunday game -> Friday's report
    Saturday game  -> Thursday's report      Monday game -> Saturday's report

(the final report falls two days before kickoff, except for Thursday games,
where it is the day before). Before that, the report carries practice
participation but almost no statuses: on the Thursday of 2026 week 3, 8 of 259
rows had one, all for the two Thursday-night teams.

Forecasting early is not an error anyone would see. The feature is simply near
zero, the model reads that as "fully healthy", and the prediction looks like a
football score. The backtest never had this problem -- historical rows always
carry the final report -- so an early live forecast is a model applied to
inputs it was never validated on. predict_week therefore refuses to forecast a
game whose final report is not yet in the data, unless explicitly told to.

The rule, per game:
  1. CALENDAR  the injuries pull happened after the final report was due
               (due = 4pm ET on the report day). Pulled earlier, the data
               cannot contain it, whatever it shows.
  2. DATA      most teams in that report's slate show at least one status.
               If the due time has passed but nflverse has not ingested the
               report yet, the statuses are missing across the whole slate.
A single team with no statuses is NOT disqualifying: 2-4% of team-games file a
final report with nobody designated (measured over 2019-2025), and that is a
real "healthy", not missing data. It is noted, not blocked.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INJURIES = Path("data/raw/injuries.parquet")
FINAL_REPORT_HOUR_ET = 16
# Share of the slate's teams that must show statuses before the report counts as
# ingested. Well above the ~3% of healthy-roster reports, well below a real one.
MIN_SLATE_COVERAGE = 0.5
TEAM_CODE_MAP = {"OAK": "LV"}


def et(ts: pd.Timestamp) -> str:
    """'Fri Sep 25 4:00pm ET' -- portable (no platform-specific strftime flags)."""
    t = pd.Timestamp(ts).tz_convert("US/Eastern")
    return f"{t:%a %b %d} {t.hour % 12 or 12}:{t:%M}{'am' if t.hour < 12 else 'pm'} ET"


def final_report_due(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """When the final injury report for a game is published (UTC)."""
    local = pd.Timestamp(kickoff_utc).tz_convert("US/Eastern")
    lead_days = 1 if local.dayofweek == 3 else 2  # Thursday games: day before
    report_day = (local - pd.Timedelta(days=lead_days)).normalize()
    return (report_day + pd.Timedelta(hours=FINAL_REPORT_HOUR_ET)).tz_convert("UTC")


def injuries_pulled_at() -> pd.Timestamp | None:
    """When injuries.parquet was pulled: the pull manifest's stamp while the
    file still matches the hash recorded with it, else the file's modification
    time (the pull scripts write it once, at pull time)."""
    from src.ingest.manifest import pulled_at

    stamped = pulled_at("injuries")
    if stamped is not None:
        return stamped
    if INJURIES.exists():
        return pd.Timestamp(INJURIES.stat().st_mtime, unit="s", tz="UTC")
    return None


def status_counts(injuries: pd.DataFrame) -> pd.DataFrame:
    inj = injuries.copy()
    inj["team"] = inj["team"].replace(TEAM_CODE_MAP)
    return (
        inj.groupby(["season", "week", "team"])
        .agg(
            rows=("gsis_id", "size"),
            statuses=("report_status", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )


def assess(
    games: pd.DataFrame,
    injuries: pd.DataFrame,
    pulled_at: pd.Timestamp | None,
) -> pd.DataFrame:
    """Readiness per game.

    games needs game_id, season, week, home_team, away_team, kickoff (UTC).
    Returns game_id, injury_report_due, home/away_status_rows,
    injury_report_final (bool) and injury_note (why, in words).
    """
    counts = status_counts(injuries).set_index(["season", "week", "team"])["statuses"]

    def n_status(season, week, team):
        return int(counts.get((season, week, team), 0))

    out = games[
        ["game_id", "season", "week", "home_team", "away_team", "kickoff"]
    ].copy()
    out["injury_report_due"] = [
        final_report_due(k) if pd.notna(k) else pd.NaT for k in out["kickoff"]
    ]
    out["home_status_rows"] = [
        n_status(s, w, t)
        for s, w, t in zip(out["season"], out["week"], out["home_team"])
    ]
    out["away_status_rows"] = [
        n_status(s, w, t)
        for s, w, t in zip(out["season"], out["week"], out["away_team"])
    ]

    # Slate coverage: all teams whose final report shares this due time.
    coverage = {}
    for due, g in out.groupby("injury_report_due"):
        teams = pd.concat(
            [
                g[["home_status_rows"]].rename(columns={"home_status_rows": "n"}),
                g[["away_status_rows"]].rename(columns={"away_status_rows": "n"}),
            ]
        )["n"]
        coverage[due] = float((teams > 0).mean())

    finals, notes = [], []
    for _, r in out.iterrows():
        due = r["injury_report_due"]
        if pd.isna(due):
            finals.append(False)
            notes.append("kickoff unknown -- cannot tell when its report is final")
            continue
        due_txt = et(due)
        if pulled_at is None or pulled_at < due:
            finals.append(False)
            when = "never" if pulled_at is None else et(pulled_at)
            notes.append(
                f"final report due {due_txt}; injuries pulled {when} -- pull again after it"
            )
            continue
        if coverage[due] < MIN_SLATE_COVERAGE:
            finals.append(False)
            notes.append(
                f"final report was due {due_txt} but only {coverage[due]:.0%} of that "
                "slate's teams show statuses -- nflverse has not ingested it yet; re-pull later"
            )
            continue
        finals.append(True)
        quiet = [
            t
            for t, n in (
                (r["home_team"], r["home_status_rows"]),
                (r["away_team"], r["away_status_rows"]),
            )
            if n == 0
        ]
        notes.append(
            f"final ({', '.join(quiet)} designated nobody)" if quiet else "final"
        )
    out["injury_report_final"] = finals
    out["injury_note"] = notes
    return out.drop(columns=["season", "week", "home_team", "away_team", "kickoff"])
