"""
Runs every feature-build stage in dependency order and records what it built.

Why this exists: the stages are independent scripts with no declared ordering,
so it was possible -- and it happened -- to rebuild an upstream table and leave
a downstream one stale. model_table.parquet once carried a timestamp 94 minutes
OLDER than the team_rolling_features.parquet it is derived from, which silently
invalidated every model comparison made against it.

This script makes the correct order executable, and writes a manifest so
staleness is detectable afterwards instead of being invisible. See
tests/test_pipeline_freshness.py, which fails if any output is older than an
input or has been modified out of band.

Run: python -m src.features.build_all
     python -m src.features.build_all --dry-run     # print the plan only
Output: every table under data/processed/, plus data/processed/_manifest.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = "config.yaml"
RAW_SCHEDULES = "data/raw/schedules.parquet"
RAW_PBP = "data/raw/pbp.parquet"
MANIFEST_PATH = Path("data/processed/_manifest.json")

# Dependency order. Verified against the read_parquet/to_parquet calls in each
# script -- if you change a script's inputs or output, update its entry here.
STAGES = [
    {
        "module": "src.features.build_team_game_stats",
        "inputs": [RAW_SCHEDULES, RAW_PBP],
        "output": "data/processed/team_game_stats.parquet",
    },
    {
        "module": "src.features.build_drive_stats",
        "inputs": [RAW_PBP],
        "output": "data/processed/drive_stats.parquet",
    },
    {
        "module": "src.features.build_qb_rolling_features",
        "inputs": [RAW_PBP, RAW_SCHEDULES, CONFIG],
        "output": "data/processed/qb_rolling_features.parquet",
    },
    {
        "module": "src.features.build_rolling_features",
        "inputs": ["data/processed/team_game_stats.parquet", CONFIG],
        "output": "data/processed/team_rolling_features.parquet",
    },
    {
        "module": "src.features.build_adjusted_ratings",
        "inputs": ["data/processed/team_game_stats.parquet", CONFIG],
        "output": "data/processed/adjusted_ratings.parquet",
    },
    {
        "module": "src.features.build_sequences",
        "inputs": ["data/processed/team_game_stats.parquet"],
        "output": "data/processed/team_sequences.npz",
    },
    {
        "module": "src.features.build_drive_rolling_features",
        "inputs": ["data/processed/drive_stats.parquet", RAW_SCHEDULES, CONFIG],
        "output": "data/processed/team_drive_rolling_features.parquet",
    },
    {
        "module": "src.features.build_injury_features",
        "inputs": [
            "data/raw/injuries.parquet",
            "data/raw/snap_counts.parquet",
            "data/raw/player_ids.parquet",
            RAW_SCHEDULES,
        ],
        "output": "data/processed/injury_features.parquet",
    },
    {
        "module": "src.features.build_game_features",
        "inputs": [
            "data/processed/team_rolling_features.parquet",
            "data/processed/injury_features.parquet",
            RAW_SCHEDULES,
        ],
        "output": "data/processed/model_table.parquet",
    },
    {
        "module": "src.features.build_drive_game_features",
        "inputs": [
            "data/processed/team_drive_rolling_features.parquet",
            RAW_SCHEDULES,
        ],
        "output": "data/processed/drive_model_table.parquet",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stamp(path: str) -> dict:
    p = Path(path)
    st = p.stat()
    return {"path": path, "bytes": st.st_size, "mtime": st.st_mtime}


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            args, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def check_inputs_exist() -> list[str]:
    """Raw inputs can't be built by this script -- fail early with the fix."""
    required = (
        RAW_SCHEDULES,
        RAW_PBP,
        CONFIG,
        "data/raw/injuries.parquet",
        "data/raw/snap_counts.parquet",
        "data/raw/player_ids.parquet",
    )
    return [p for p in required if not Path(p).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the stage order, build nothing"
    )
    args = ap.parse_args()

    if args.dry_run:
        print(f"{len(STAGES)} stages, in dependency order:\n")
        for i, s in enumerate(STAGES, 1):
            print(f"  {i}. {s['module']}")
            print(f"       inputs: {', '.join(s['inputs'])}")
            print(f"       output: {s['output']}")
        return

    missing = check_inputs_exist()
    if missing:
        print(f"ERROR: missing raw inputs: {', '.join(missing)}")
        print("Run the ingestion scripts first:")
        print("  python src/ingest/pull_schedules.py")
        print("  python src/ingest/pull_pbp.py")
        print("  python src/ingest/pull_injuries.py")
        sys.exit(1)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []

    for i, stage in enumerate(STAGES, 1):
        module, output = stage["module"], stage["output"]
        print(f"\n[{i}/{len(STAGES)}] {module}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", module], text=True)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"\nFAILED at stage {i} ({module}) after {elapsed:.1f}s.")
            print("Manifest NOT written -- data/processed/ is now in a mixed state.")
            print(f"Fix the error above, then re-run: python -m {__spec__.name}")
            sys.exit(proc.returncode)

        out = Path(output)
        if not out.exists():
            print(f"\nFAILED: {module} exited 0 but did not write {output}.")
            sys.exit(1)

        records.append(
            {
                "module": module,
                "seconds": round(elapsed, 2),
                "inputs": [file_stamp(p) for p in stage["inputs"]],
                "output": file_stamp(output) | {"sha256": sha256(out)},
            }
        )
        print(f"      done in {elapsed:.1f}s -> {output}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": round(time.time() - started, 2),
        "git": git_state(),
        "python": sys.version.split()[0],
        "stages": records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{'=' * 60}")
    print(f"All {len(STAGES)} stages rebuilt in {manifest['total_seconds']:.1f}s")
    print(f"Manifest: {MANIFEST_PATH}")
    if manifest["git"]["dirty"]:
        print("NOTE: working tree was dirty -- this build is not reproducible")
    print("Now validate:  pytest tests/ -m requires_data -v")


if __name__ == "__main__":
    main()
