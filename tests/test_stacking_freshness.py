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

import pytest

# Reads built parquet from data/, which is gitignored -- excluded from CI.
pytestmark = [pytest.mark.requires_data, pytest.mark.backtest_artifacts]


def test_stacking_is_not_older_than_its_base_models():
    """The stack's provenance sidecar records a hash of every base prediction
    file it read; a re-run base model changes that hash. (This used to compare
    modification times.)"""
    from src.validate.backtest_io import predictions_path, stale_inputs

    if not predictions_path("stacking").exists():
        pytest.skip("stacking has not been run")
    stale = [p for p in stale_inputs("stacking") if "_predictions" in p]
    assert not stale, (
        f"stacking was built from out-of-date base predictions: {stale}. "
        "Re-run: python -m src.models.run_all --only stacking"
    )
