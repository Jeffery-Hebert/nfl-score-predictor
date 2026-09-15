"""
Freshness gate for the meta-model's dependencies.

tests/test_prediction_sanity.py compares each prediction file to
model_table.parquet. That is the right reference for a base model, which reads
model_table directly -- but stacking.py reads OTHER PREDICTION FILES, so a
rebuilt base model leaves the stack stale while every other check stays green.

This caught exactly that: linear.py was switched from OLS to ridge, which
regenerated linear_predictions.parquet, and stacking_predictions.parquet stayed
15 minutes older with a fully passing suite.

Lives in its own module because test_prediction_sanity.py parametrizes every
test in it over the discovered model list, and this check is per-stack rather
than per-model.

Run: pytest tests/test_stacking_freshness.py -v
"""

from pathlib import Path

import pytest

# Reads built parquet from data/, which is gitignored -- excluded from CI.
pytestmark = pytest.mark.requires_data

PRED_DIR = Path("data/processed")


def test_stacking_is_not_older_than_its_base_models():
    from src.models.stacking import BASE_MODELS

    stack = PRED_DIR / "stacking_predictions.parquet"
    if not stack.exists():
        pytest.skip("stacking has not been run")
    stack_mtime = stack.stat().st_mtime

    stale = []
    for base in BASE_MODELS:
        p = PRED_DIR / f"{base}_predictions.parquet"
        if not p.exists():
            continue
        age_min = (p.stat().st_mtime - stack_mtime) / 60
        if age_min > 0:
            stale.append(f"{base} is {age_min:.0f} min newer than the stack")
    assert not stale, (
        f"stacking was built from out-of-date base predictions: {stale}. "
        "Re-run: python -m src.models.stacking"
    )
