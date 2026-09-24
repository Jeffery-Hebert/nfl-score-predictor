"""
The weekly run, as one command.

    python -m src.weekly                        # the normal run
    python -m src.weekly --skip-pull            # data already pulled this hour
    python -m src.weekly --allow-unsettled-injuries
    python -m src.weekly --backtest             # also re-benchmark every model (slow)

Steps, each stopping the run with its own error message on failure:

  1. pull   fresh schedules, play-by-play, injuries and snap counts, and check
            they are complete and agree with each other   (src.ingest.pull_all)
  2. build  every feature table, in dependency order      (src.features.build_all)
  3. check  the built tables: leakage, joins, completeness, and that next
            week's features are populated
            (pytest -m "requires_data and not backtest_artifacts")
  4. predict every game that has not kicked off AND whose final injury report
            is in the data                                 (src.predict.predict_week)
  5. ledger rebuild data/predictions/index.html with every week graded

WHEN to run it. The injury report is only final shortly before each game --
Wednesday afternoon for a Thursday game, Friday for Sunday, Saturday for Monday
-- and step 4 holds back any game whose report is not in yet. So:

    Thursday afternoon   forecasts the Thursday game
    Sunday morning       forecasts the Sunday and Monday games

Running it at other times is safe: it forecasts only what is ready and leaves
every other game's existing forecast alone.

With --backtest it also re-runs every model's walk-forward (src.models.run_all,
~40 minutes, dominated by the Gaussian Process) and writes the model report.
"""

import argparse
import subprocess
import sys
import time

DATA_CHECKS = "requires_data and not backtest_artifacts"


def step(title: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 72}\n{title}\n  $ {' '.join(cmd)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\nSTOPPED: '{title}' failed (exit {rc}) after {time.time() - t0:.0f}s.")
        print("Nothing after this step ran. Fix the error above and re-run.")
        sys.exit(rc)
    print(f"-- {title}: ok ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--allow-unsettled-injuries", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args(argv)
    py = sys.executable

    if not args.skip_pull:
        step("1. Pull and validate raw data", [py, "-m", "src.ingest.pull_all"])
    step("2. Build features", [py, "-m", "src.features.build_all"])
    step(
        "3. Check the built data",
        [
            py,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-m",
            DATA_CHECKS,
            "-p",
            "no:cacheprovider",
        ],
    )
    if args.backtest:
        step("3b. Re-run every backtest", [py, "-m", "src.models.run_all"])
        step("3c. Model report", [py, "-m", "src.validate.model_report"])

    predict = [py, "-m", "src.predict.predict_week", "--html"]
    if args.allow_unsettled_injuries:
        predict.append("--allow-unsettled-injuries")
    if args.season is not None:
        predict += ["--season", str(args.season)]
    if args.week is not None:
        predict += ["--week", str(args.week)]
    step("4-5. Forecast what is ready, rebuild the ledger", predict)

    print(
        "\nDone. Record the forecasts before kickoff so the ledger stays "
        "pre-registered:\n  git add data/predictions/*.parquet && git commit -m "
        '"Forecast: <season> week <N>"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
