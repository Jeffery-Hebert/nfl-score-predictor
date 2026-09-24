"""
Runs every model's walk-forward backtest -- production, shelved, experimental
and meta -- in dependency order, in parallel, and summarises the result.

Why this exists. Re-benchmarking used to mean running sixteen scripts by hand
in the right order (the stack after its members, the composite after its), with
the Gaussian Process alone taking half an hour and every other script sitting
idle behind it. Missing one left a stale prediction file that the scoreboard
would happily compare against fresh ones.

    python -m src.models.run_all                    # everything
    python -m src.models.run_all --only linear gp   # a subset
    python -m src.models.run_all --skip gp rnn
    python -m src.models.run_all --jobs 3           # parallel processes
    python -m src.models.run_all --list             # what would run, and why

Each model runs as `python -m <module>` in its own process, output to
data/processed/logs/<model>.log. Each writes its predictions plus a provenance
sidecar (src/validate/backtest_io.py), so staleness is checked by content hash
afterwards. The longest jobs start first. Exits non-zero if anything failed.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from src.models import registry
from src.validate.backtest_io import load_meta

LOG_DIR = Path("data/processed/logs")

# Rough wall-clock seconds on a 4-core laptop. Only used to start the longest
# jobs first so the parallel run finishes as early as possible.
EXPECTED_SECONDS = {
    "gp": 1800,
    "drivechain": 900,
    "bayesian": 600,
    "mlp": 240,
    "rf": 220,
    "linear": 150,
    "rnn": 120,
    "drivev2": 120,
    "poisson": 60,
    "stacking": 60,
}


def dependencies(spec) -> tuple[str, ...]:
    if spec.name == "composite":
        from src.config import live_settings

        return tuple(live_settings()["composite"]["members"])
    return spec.depends_on


def missing_requirement(spec) -> str | None:
    if spec.name == "rnn" and importlib.util.find_spec("torch") is None:
        return (
            "torch not installed: pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
    return None


def plan(only=None, skip=None) -> list:
    names = registry.names()
    chosen = [n for n in names if (not only or n in only) and n not in (skip or [])]
    unknown = [n for n in (only or []) + (skip or []) if n not in names]
    if unknown:
        raise SystemExit(f"ERROR: unknown model(s) {unknown}. Known: {names}")
    return [registry.get(n) for n in chosen]


def run_one(spec, threads: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{spec.name}.log"
    env = dict(os.environ)
    # Parallel jobs must not oversubscribe the CPU -- except the GP, which is
    # BLAS-bound and the long pole of the whole run, so it gets half the cores.
    n = max(threads, (os.cpu_count() or 2) // 2) if spec.name == "gp" else threads
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, str(n))
    env["PYTHONUNBUFFERED"] = "1"  # so a log shows fold progress while it runs
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(
            [sys.executable, "-m", spec.module],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    return {
        "name": spec.name,
        "ok": proc.returncode == 0,
        "seconds": time.time() - t0,
        "log": str(log),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--skip", nargs="+", default=None)
    ap.add_argument(
        "--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2))
    )
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")

    specs = plan(args.only, args.skip)
    selected = {s.name for s in specs}
    if args.list:
        for s in specs:
            deps = dependencies(s)
            print(
                f"  {s.name:<11} {s.status:<12} python -m {s.module}"
                + (f"   (after {', '.join(deps)})" if deps else "")
            )
        return 0

    threads = max(1, (os.cpu_count() or 2) // args.jobs)
    pending = sorted(specs, key=lambda s: -EXPECTED_SECONDS.get(s.name, 30))
    results, done, failed, skipped = {}, set(), set(), set()
    print(f"Running {len(pending)} backtests, {args.jobs} at a time. Logs: {LOG_DIR}/")
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        running = {}
        while pending or running:
            launched = True
            while launched and len(running) < args.jobs:
                launched = False
                for spec in list(pending):
                    deps = [d for d in dependencies(spec) if d in selected]
                    if any(d in failed or d in skipped for d in deps):
                        # A failed dependency fails its dependents; a skipped one
                        # (optional requirement absent) only skips them.
                        pending.remove(spec)
                        broke = any(d in failed for d in deps)
                        (failed if broke else skipped).add(spec.name)
                        why = (
                            "a dependency failed"
                            if broke
                            else "a dependency was skipped"
                        )
                        results[spec.name] = {
                            "name": spec.name,
                            "ok": False,
                            "why": why,
                        }
                        print(f"  SKIP  {spec.name}: {why}")
                        launched = True
                        break
                    if any(d not in done for d in deps):
                        continue
                    need = missing_requirement(spec)
                    pending.remove(spec)
                    if need:
                        # An optional dependency is absent (torch for the RNN):
                        # skipped, not failed -- the run's exit code must not
                        # depend on whether this machine has torch installed.
                        skipped.add(spec.name)
                        results[spec.name] = {
                            "name": spec.name,
                            "ok": False,
                            "why": need,
                        }
                        print(f"  SKIP  {spec.name}: {need}")
                    else:
                        print(f"  start {spec.name}", flush=True)
                        running[pool.submit(run_one, spec, threads)] = spec.name
                    launched = True
                    break
            if not running:
                if pending:  # nothing runnable and nothing running: a cycle
                    raise SystemExit(f"ERROR: unresolvable dependencies: {pending}")
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                name = running.pop(fut)
                res = fut.result()
                results[name] = res
                (done if res["ok"] else failed).add(name)
                status = "done " if res["ok"] else "FAIL "
                print(f"  {status} {name} in {res['seconds']:.0f}s", flush=True)

    print(f"\n{'model':<12}{'status':<10}{'seconds':>8}{'games':>7}{'mean RMSE':>11}")
    print("-" * 50)
    for spec in specs:
        r = results.get(spec.name, {})
        meta = load_meta(spec.name) if r.get("ok") else None
        m = (meta or {}).get("metrics", {})
        mean_rmse = (
            f"{(m['home_rmse'] + m['away_rmse']) / 2:.4f}" if "home_rmse" in m else "-"
        )
        status = (
            "ok"
            if r.get("ok")
            else (
                "skipped"
                if spec.name in skipped
                else ("FAILED" if "seconds" in r else "not run")
            )
        )
        secs = f"{r['seconds']:.0f}" if "seconds" in r else "-"
        games = str(meta["n_games"]) if meta else "-"
        print(f"{spec.name:<12}{status:<10}{secs:>8}{games:>7}{mean_rmse:>11}")
        if not r.get("ok") and r.get("log"):
            print(f"{'':<12}see {r['log']}")
        elif r.get("why"):
            print(f"{'':<12}{r['why']}")
    print(
        f"\nTotal {time.time() - started:.0f}s. Next: python -m src.validate.model_report"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
