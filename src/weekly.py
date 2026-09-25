"""
The weekly run, as one command.

    python -m src.weekly                        # the normal run
    python -m src.weekly --skip-pull            # data already pulled this hour
    python -m src.weekly --allow-unsettled-injuries
    python -m src.weekly --backtest             # also re-benchmark every model (slow)
    python -m src.weekly --only-in-season       # do nothing unless a game is near

Steps, each stopping the run with its own error message on failure:

  1. pull   fresh schedules, play-by-play, injuries and snap counts, and check
            they are complete and agree with each other   (src.ingest.pull_all)
  2. build  every feature table, in dependency order      (src.features.build_all)
  3. test   every unit test and leakage gate, and every check on the built
            tables: joins, completeness, next week's features populated
            (pytest -m "not backtest_artifacts")
  4. predict every game that has not kicked off AND whose final injury report
            is in the data                                 (src.predict.predict_week)
  5. ledger rebuild data/predictions/index.html with every week graded

GitHub runs this on a schedule (.github/workflows/pipeline.yml) and commits any
changed forecast back to the repository; nothing needs to be run by hand.

WHEN to run it. The injury report is only final shortly before each game --
Wednesday afternoon for a Thursday game, Friday for Sunday, Saturday for Monday
-- and step 4 holds back any game whose report is not in yet. So:

    Thursday afternoon   forecasts the Thursday game
    Sunday morning       forecasts the Sunday and Monday games

Running it at other times is safe: it forecasts only what is ready and leaves
every other game's existing forecast alone.

With --backtest it also re-runs every model's walk-forward (src.models.run_all,
~40 minutes, dominated by the Gaussian Process), checks the fresh backtests
(pytest -m backtest_artifacts) and writes the model report.

--only-in-season makes the whole run a no-op unless a game kicked off in the
last two days or kicks off in the next nine (src/schedule.py::games_near): the
scheduled pipeline runs daily all year, and most of the year there is nothing
to pull, build or forecast.
"""

import argparse
import subprocess
import sys
import time

import pandas as pd

CHECKS = "not backtest_artifacts"  # everything that does not need a backtest


def step(title: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 72}\n{title}\n  $ {' '.join(cmd)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\nSTOPPED: '{title}' failed (exit {rc}) after {time.time() - t0:.0f}s.")
        print("Nothing after this step ran. Fix the error above and re-run.")
        sys.exit(rc)
    print(f"-- {title}: ok ({time.time() - t0:.0f}s)")


def games_near_now(now=None) -> pd.DataFrame:
    """Games within the --only-in-season window, read straight from nflverse:
    one small file, so asking costs seconds when the answer is no."""
    import nflreadpy as nfl

    from src.ingest.seasons import current_season
    from src.schedule import games_near

    season = current_season()
    sched = nfl.load_schedules(seasons=[season - 1, season]).to_pandas()
    return games_near(sched, now if now is not None else pd.Timestamp.now(tz="UTC"))


def pytest_step(title: str, marker: str) -> None:
    step(
        title,
        [sys.executable, "-m", "pytest", "tests/", "-q", "-m", marker]
        + ["-p", "no:cacheprovider"],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--allow-unsettled-injuries", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--only-in-season", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args(argv)
    py = sys.executable

    if args.only_in_season:
        near = games_near_now()
        if near.empty:
            print(
                "No NFL game kicked off in the last 2 days or kicks off in the "
                "next 9. Nothing to pull, build or forecast."
            )
            return 0
        print(f"{len(near)} game(s) within 2 days back / 9 ahead: running.")

    if not args.skip_pull:
        step("1. Pull and validate raw data", [py, "-m", "src.ingest.pull_all"])
    step("2. Build features", [py, "-m", "src.features.build_all"])
    pytest_step("3. Test the code and the built data", CHECKS)
    if args.backtest:
        step("3b. Re-run every backtest", [py, "-m", "src.models.run_all"])
        pytest_step("3c. Check the fresh backtests", "backtest_artifacts")
        step("3d. Model report", [py, "-m", "src.validate.model_report"])

    predict = [py, "-m", "src.predict.predict_week"]
    if args.allow_unsettled_injuries:
        predict.append("--allow-unsettled-injuries")
    if args.season is not None:
        predict += ["--season", str(args.season)]
    if args.week is not None:
        predict += ["--week", str(args.week)]
    step("4. Forecast what is ready", predict)
    # Its own step, not predict_week --html: the ledger must be rebuilt even
    # when nothing is ready to forecast -- that is the run that grades a week.
    step("5. Rebuild the ledger", [py, "-m", "src.predict.build_report"])

    print(
        "\nDone. On GitHub the pipeline workflow commits any changed forecast. "
        "Run locally, record them before kickoff so the ledger stays "
        "pre-registered:\n  git add data/predictions/*.parquet && git commit -m "
        '"Forecast: <season> week <N>"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
