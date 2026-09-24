"""
data/raw/_pull_manifest.json: when each raw dataset was pulled, and what it held.

Why. Two decisions depend on WHEN data was pulled, not just what it contains:
whether the injury report behind a forecast was the final one (see
src/predict/injury_readiness.py), and whether a build is using the latest
results. A file's modification time answers that badly -- a copy or a restore
changes it -- so every pull records its time together with the file's hash, and
readers trust the time only while the hash still matches.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.provenance import sha256

MANIFEST = Path("data/raw/_pull_manifest.json")


def read() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"datasets": {}}


def _write(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    tmp.replace(MANIFEST)


def record_pull(dataset: str, path, df: pd.DataFrame, seasons: list[int]) -> None:
    """Stamp one dataset's pull. Called by each pull script right after writing."""
    import nflreadpy

    manifest = read()
    manifest.setdefault("datasets", {})[dataset] = {
        "path": str(path),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        # What actually arrived, not what was asked for: the newest season is
        # legitimately absent between March and its first games.
        "seasons": (
            [int(df["season"].min()), int(df["season"].max())]
            if "season" in df.columns and len(df)
            else [int(seasons[0]), int(seasons[-1])]
        ),
        "sha256": sha256(path),
        "nflreadpy": getattr(nflreadpy, "__version__", "unknown"),
    }
    _write(manifest)


def record_validation(result: dict) -> None:
    manifest = read()
    manifest["validation"] = result
    _write(manifest)


def pulled_at(dataset: str) -> pd.Timestamp | None:
    """The recorded pull time, if the file on disk is still the one recorded."""
    rec = read().get("datasets", {}).get(dataset)
    if not rec or not Path(rec["path"]).exists():
        return None
    if sha256(rec["path"]) != rec["sha256"]:
        return None  # file replaced since the pull was recorded
    return pd.Timestamp(rec["pulled_at"])
