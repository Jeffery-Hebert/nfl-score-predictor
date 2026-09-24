"""
One way to save a walk-forward result: the predictions, plus a sidecar that
records exactly which inputs produced them.

Why. Every model used to end with `results.to_parquet(...)`, and staleness was
judged by comparing file modification times against model_table.parquet. That
check fired after every rebuild even when nothing had changed, which trained
everyone -- the Sunday routine's prompt included -- to ignore it; and it could
not see a model scored against the wrong table if the timestamps happened to
line up. The sidecar stores a sha256 of every input, so "is this backtest
current?" becomes an exact question: do the hashes still match?

What is hashed. A backtest only ever reads the rows of games that have been
PLAYED (the harness drops unscored rows before any fold). Hashing whole files
would flag every backtest stale whenever next week's injury report or betting
line moved an UNPLAYED row -- the false alarm that trained everyone to ignore
the old check. So tables are fingerprinted on their played rows only; files
without a score column (and anything very large) fall back to the whole-file
hash.

Sidecars are one small JSON file per model rather than one shared manifest,
because src/models/run_all.py runs models in parallel and a shared file would
lose updates to concurrent writers.

    data/processed/<name>_predictions.parquet
    data/processed/<name>_predictions.meta.json
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.provenance import file_record, git_state

PRED_DIR = Path("data/processed")


def predictions_path(name: str) -> Path:
    return PRED_DIR / f"{name}_predictions.parquet"


def meta_path(name: str) -> Path:
    return PRED_DIR / f"{name}_predictions.meta.json"


FINGERPRINT_MAX_BYTES = 20_000_000  # play-by-play is ~130MB: whole-file hash


def played_fingerprint(path) -> str | None:
    """sha256 of the rows a backtest can actually read: games with a score.

    None when the file is not a parquet table with game_id and home_score, or
    is too large to fingerprint cheaply -- callers then compare whole-file
    hashes instead.
    """
    import hashlib

    p = Path(path)
    if p.suffix != ".parquet" or p.stat().st_size > FINGERPRINT_MAX_BYTES:
        return None
    df = pd.read_parquet(p)
    if not {"game_id", "home_score"} <= set(df.columns):
        return None
    played = df[df["home_score"].notna()]
    played = played.sort_values(["game_id"] + [c for c in ("team",) if c in played])
    h = hashlib.sha256("|".join(map(str, played.columns)).encode())
    h.update(pd.util.hash_pandas_object(played, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _input_record(path) -> dict:
    rec = file_record(path)
    rec["played_sha256"] = played_fingerprint(path)
    return rec


def save_predictions(results: pd.DataFrame, name: str, inputs) -> Path:
    """Write <name>_predictions.parquet and its provenance sidecar.

    inputs: every file the predictions were derived from. Hashed now, so a later
    rebuild of any of them is detectable by content rather than by clock.
    """
    from src.validate.walk_forward import score_predictions

    out = predictions_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)

    metrics = score_predictions(results) if len(results) else {}
    meta = {
        "model": name,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "git": git_state(),
        "output": file_record(out),
        "inputs": [_input_record(p) for p in inputs],
        "n_games": int(len(results)),
        "seasons": sorted(int(s) for s in results["season"].unique()),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    tmp = meta_path(name).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(meta_path(name))
    print(f"Saved {len(results)} predictions to {out} (+ provenance sidecar)")
    return out


def load_meta(name: str) -> dict | None:
    p = meta_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def stale_inputs(name: str) -> list[str]:
    """Inputs whose CONTENT has changed since the backtest ran. Empty = current.

    A missing sidecar or a missing input is reported as stale too: in both cases
    nobody can say what the predictions were computed from.
    """
    from src.provenance import sha256

    meta = load_meta(name)
    if meta is None:
        return [f"no provenance sidecar at {meta_path(name)}"]
    problems = []
    for rec in meta["inputs"]:
        p = Path(rec["path"])
        if not p.exists():
            problems.append(f"{p} no longer exists")
            continue
        if rec.get("played_sha256"):
            if played_fingerprint(p) != rec["played_sha256"]:
                problems.append(f"{p}: played games changed since this backtest ran")
        elif sha256(p) != rec["sha256"]:
            problems.append(f"{p} changed since this backtest ran")
    out = meta["output"]
    if Path(out["path"]).exists() and sha256(out["path"]) != out["sha256"]:
        problems.append(f"{out['path']} was modified after it was written")
    return problems


def add_fingerprints(name: str) -> list[str]:
    """Upgrade an older sidecar (whole-file hashes only) with played-row
    fingerprints -- but ONLY for inputs whose recorded whole-file hash still
    matches the file on disk, i.e. inputs provably identical to what the
    backtest read. Anything else is left as recorded (and so reads as stale).
    Returns the inputs upgraded."""
    from src.provenance import sha256

    meta = load_meta(name)
    if meta is None:
        return []
    upgraded = []
    for rec in meta["inputs"]:
        if rec.get("played_sha256") or not Path(rec["path"]).exists():
            continue
        if sha256(rec["path"]) == rec["sha256"]:
            rec["played_sha256"] = played_fingerprint(rec["path"])
            if rec["played_sha256"]:
                upgraded.append(rec["path"])
    if upgraded:
        tmp = meta_path(name).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n")
        tmp.replace(meta_path(name))
    return upgraded
