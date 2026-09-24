"""
Small provenance helpers shared by the pipeline stages: content hashes and git
state. A hash says what a file CONTAINS; an mtime only says when something last
touched it. The staleness checks in this project used to compare mtimes, which
flags a table as stale after every rebuild even when its contents are identical,
and cannot tell a genuinely changed input from a re-save of the same one.
"""

import hashlib
import subprocess
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def file_record(path) -> dict:
    p = Path(path)
    return {"path": str(path), "bytes": p.stat().st_size, "sha256": sha256(p)}
