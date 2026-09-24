"""
Provenance gates for data/processed/.

These exist because of a real, silent failure: model_table.parquet once carried
a timestamp 94 minutes OLDER than the team_rolling_features.parquet it is built
from. Nothing detected it, and every model comparison made against that table
was quietly invalid -- prediction files written days apart were being compared
as though they shared a feature set.

Run: pytest tests/test_pipeline_freshness.py -v
Rebuild if these fail: python -m src.features.build_all
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.features.build_all import MANIFEST_PATH, STAGES, input_fingerprint

# Reads built parquet from data/, which is gitignored -- excluded from CI.
# Run locally after building the pipeline; see the marker note in pyproject.toml.
pytestmark = pytest.mark.requires_data


REBUILD = "Rebuild with: python -m src.features.build_all"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.fail(f"No manifest at {MANIFEST_PATH}. {REBUILD}")
    return json.loads(MANIFEST_PATH.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_covers_every_stage(manifest):
    recorded = {s["module"] for s in manifest["stages"]}
    expected = {s["module"] for s in STAGES}
    missing = expected - recorded
    assert not missing, f"Manifest is missing stages: {sorted(missing)}. {REBUILD}"


def test_every_output_exists(manifest):
    for stage in STAGES:
        assert Path(
            stage["output"]
        ).exists(), f"{stage['output']} is missing ({stage['module']}). {REBUILD}"


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: s["module"].split(".")[-1])
def test_inputs_unchanged_since_the_build(stage, manifest):
    """The exact bug this suite was written for: a downstream table left stale
    after an upstream one was rebuilt.

    Judged by CONTENT: does every input still fingerprint to what the stage
    read? This used to compare modification times, which failed after any
    re-save of identical data and any edit to config.yaml (a comment, the live
    model list) -- none of which changes a feature -- and a gate that fails for
    nothing gets ignored the day it fails for something."""
    record = next(
        (r for r in manifest["stages"] if r["module"] == stage["module"]), None
    )
    assert record, f"{stage['module']} is not in the manifest. {REBUILD}"
    recorded = {r["path"]: r.get("fingerprint") for r in record["inputs"]}
    assert set(recorded) == set(stage["inputs"]), (
        f"{stage['module']} was built before its inputs in STAGES changed "
        f"(built from {sorted(recorded)}). {REBUILD}"
    )
    for inp in stage["inputs"]:
        assert Path(inp).exists(), f"Input {inp} is missing."
        assert recorded[inp], (
            f"The manifest predates input fingerprints, so the freshness of "
            f"{stage['output']} cannot be judged. {REBUILD}"
        )
        if input_fingerprint(inp) != recorded[inp]:
            pytest.fail(
                f"STALE: {inp} has changed since {stage['output']} was built "
                f"from it. Anything derived from it is untrustworthy. {REBUILD}"
            )


def test_outputs_unmodified_since_the_build(manifest):
    """Catches an output rebuilt out of band -- by running one stage's script
    directly, bypassing the ordering in build_all.py."""
    drifted = []
    for record in manifest["stages"]:
        out = Path(record["output"]["path"])
        if not out.exists():
            drifted.append(f"{out} (missing)")
        elif _sha256(out) != record["output"]["sha256"]:
            drifted.append(f"{out} (contents changed)")
    assert not drifted, (
        "These outputs no longer match the manifest, so data/processed/ is a mix "
        f"of build runs: {drifted}. {REBUILD}"
    )


def test_build_was_from_a_clean_tree(manifest):
    """Not fatal, but a build from a dirty tree can't be reproduced from its sha."""
    if manifest["git"]["dirty"]:
        pytest.skip(
            f"Built from a dirty working tree at {manifest['git']['sha'][:8]} -- "
            "this dataset is not reproducible from that commit."
        )
    assert manifest["git"]["sha"], "Manifest recorded no git sha."
