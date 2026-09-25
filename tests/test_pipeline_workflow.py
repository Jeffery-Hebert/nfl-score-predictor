"""
.github/workflows/pipeline.yml, checked against the code it drives.

The workflow only ever runs on GitHub, on a schedule, so a mistake in it shows
up days later as a forecast that was never made -- or one made after kickoff.
These pin every place where the YAML and the Python must agree, plus the
timing rules the schedule exists to honour. Also src/predict/summarize.py,
which writes what the workflow reports.

Run: pytest tests/test_pipeline_workflow.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import live_settings
from src.predict import summarize

WORKFLOW = Path(".github/workflows/pipeline.yml")


@pytest.fixture(scope="module")
def wf():
    w = yaml.safe_load(WORKFLOW.read_text())
    w["on"] = w.pop(True, w.get("on"))  # YAML 1.1 reads the key `on` as True
    return w


def steps(wf):
    return wf["jobs"]["pipeline"]["steps"]


def run_text(wf):
    return "\n".join(s.get("run", "") for s in steps(wf))


def crons(wf):
    return [c["cron"] for c in wf["on"]["schedule"]]


class TestTheWorkflow:
    def test_it_may_commit_and_nothing_more(self, wf):
        assert wf["permissions"] == {"contents": "write"}

    def test_runs_never_overlap(self, wf):
        assert wf["concurrency"]["cancel-in-progress"] is False

    def test_the_backtest_runs_on_the_cron_the_job_checks_for(self, wf):
        backtest = wf["jobs"]["pipeline"]["env"]["BACKTEST"]
        quoted = [c for c in crons(wf) if f"'{c}'" in backtest]
        assert len(quoted) == 1, "BACKTEST must name exactly one scheduled cron"
        assert quoted[0].split()[4] == "2", "the re-benchmark runs on Tuesday"

    def test_every_day_is_covered(self, wf):
        days = set()
        for c in crons(wf):
            for part in c.split()[4].split(","):
                lo, _, hi = part.partition("-")
                days |= set(range(int(lo), int(hi or lo) + 1))
        assert days == set(range(7))

    def test_every_run_finishes_before_the_earliest_kickoff_it_serves(self, wf):
        # London games kick off 9:30am ET = 13:30 UTC in October (the earliest
        # of the year). Sunday needs a run well before that, allowing for
        # GitHub starting schedules late and a ~15 minute pipeline.
        sunday = [
            c for c in crons(wf) if "0" in c.split()[4].replace("-", ",").split(",")
        ]
        earliest = min(int(c.split()[1]) * 60 + int(c.split()[0]) for c in sunday)
        assert earliest <= 13 * 60 + 30 - 90

    def test_it_runs_the_weekly_command_and_skips_the_offseason(self, wf):
        text = run_text(wf)
        assert "python -m src.weekly" in text and "--only-in-season" in text
        assert "set -o pipefail" in text, "tee would hide a failed pipeline"

    def test_nothing_is_written_into_the_checkout_but_forecasts(self, wf):
        # An untracked file in the working tree marks the build "dirty" and
        # stamps every automated forecast's code_version "+dirty".
        text = run_text(wf)
        assert 'tee "$RUNNER_TEMP/' in text
        assert "tee pipeline.log" not in text

    def test_the_idle_phrase_matches_what_weekly_prints(self, wf):
        phrase = "Nothing to pull, build or forecast"
        assert phrase in run_text(wf)
        assert phrase in Path("src/weekly.py").read_text()

    def test_only_forecast_records_are_committed(self, wf):
        commit = next(
            s for s in steps(wf) if s.get("name") == "Commit changed forecasts"
        )
        assert "git add data/predictions/*.parquet" in commit["run"]
        assert (
            "!cancelled()" in commit["if"]
        ), "a valid record must survive a later failure"

    def test_python_matches_ci(self, wf):
        ci = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
        version = lambda w, job: next(  # noqa: E731
            s["with"]["python-version"]
            for s in w["jobs"][job]["steps"]
            if "setup-python" in s.get("uses", "")
        )
        assert version(wf, "pipeline") == version(ci, "test")


# ------------------------------------------------------------ summaries --


@pytest.fixture
def record(tmp_path):
    comp = live_settings()["composite"]["name"]
    rows = [
        # (game, kickoff UTC, generated_at, final, home, away, total)
        ("2026_03_ATL_GB", "2026-09-25 00:15", "2026-09-24T13:40:00+00:00", True, 24.6, 21.0, 45.6),
        ("2026_03_CAR_CLE", "2026-09-27 17:00", "2026-09-26T13:41:00+00:00", True, 20.7, 24.1, 44.8),
        ("2026_03_PHI_CHI", "2026-09-29 00:15", "2026-09-26T13:41:00+00:00", False, 24.9, 22.8, np.nan),
    ]  # fmt: skip
    df = pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": 2026,
            "week": 3,
            "away_team": [r[0].split("_")[2] for r in rows],
            "home_team": [r[0].split("_")[3] for r in rows],
            "kickoff": pd.to_datetime([r[1] for r in rows], utc=True),
            "generated_at": [r[2] for r in rows],
            "injury_report_final": [r[3] for r in rows],
            f"{comp}_home": [r[4] for r in rows],
            f"{comp}_away": [r[5] for r in rows],
            f"{comp}_total": [r[6] for r in rows],
            "market_home": [24.5, 20.0, 18.5],
            "market_away": [19.0, 22.5, 23.0],
        }
    )
    path = tmp_path / "2026_wk03.parquet"
    df.to_parquet(path, index=False)
    return path


class TestSummaries:
    def test_the_week_table_has_every_game_and_marks_provisional_ones(self, record):
        md = summarize.week_markdown(record)
        rows = [
            line for line in md.splitlines() if line.startswith("| ") and "@" in line
        ]
        assert len(rows) == 3
        assert "ATL @ GB" in rows[0] and "21.0-24.6" in rows[0] and "45.6" in rows[0]
        # a missing number blanks its own cell, never the whole row
        assert "PHI @ CHI" in rows[2] and "| -- |" in rows[2]
        assert "**provisional**" in rows[2] and "provisional" not in rows[0]

    def test_the_commit_message_lists_only_this_runs_games(self, record):
        msg = summarize.commit_message([record])
        subject, body = msg.split("\n", 1)
        assert subject == "Forecast: 2026 week 3 (automated)"
        assert "2 game(s) forecast 2026-09-26 13:41 UTC, 1 on their final" in body
        assert "CAR @ CLE" in body and "PHI @ CHI" in body and "ATL @ GB" not in body
        assert "(provisional)" in body

    def test_the_leaderboard_leaves_unranked_models_unranked(self, tmp_path):
        csv = tmp_path / "leaderboard.csv"
        pd.DataFrame(
            {
                "rank": [1.0, np.nan],
                "model": ["composite", "stacking"],
                "games": [1456, 1171],
                "rmse": [9.378, 9.298],
                "vs_baseline": [-0.098, np.nan],
                "vs_base_lo": [-0.164, np.nan],
                "vs_base_hi": [-0.031, np.nan],
                "winner_pct": [0.64, 0.63],
            }
        ).to_csv(csv, index=False)
        md = summarize.leaderboard_markdown(csv)
        assert (
            "| 1 | composite | 1456 | 9.378 | -0.098 [-0.164, -0.031] | 64.0% |" in md
        )
        assert "| -- | stacking | 1171 |" in md
