"""
The model registry and the `live` block of config.yaml.

The config decides which models forecast real games; a mistake there must fail
loudly before anything is fitted, not drop a model silently from the ledger.
The registry must stay importable without the heavy optional dependencies
(torch, pymc) so CI and the report builder never pay for them.

Run: pytest tests/test_registry_and_config.py -v
"""

import importlib
import sys

import pytest

from src.config import live_settings
from src.models import registry


def cfg(models, members=None, name="combined", method="mean"):
    comp = {"name": name, "method": method}
    if members is not None:
        comp["members"] = members
    return {"live": {"models": models, "composite": comp}}


class TestLiveSettings:
    def test_the_repo_config_is_valid(self):
        live = live_settings()
        assert live["models"], "config.yaml names no live models"
        assert set(live["composite"]["members"]) <= set(live["models"])

    def test_unknown_model_is_refused(self):
        with pytest.raises(KeyError, match="unknown model"):
            live_settings(cfg(["poisson", "posson"]))

    def test_a_model_that_cannot_forecast_unplayed_games_is_refused(self):
        with pytest.raises(ValueError, match="cannot forecast"):
            live_settings(cfg(["poisson", "stacking"]))

    def test_composite_members_must_be_live(self):
        with pytest.raises(ValueError, match="not in live.models"):
            live_settings(cfg(["poisson"], members=["poisson", "gp"]))

    def test_duplicates_and_empty_lists_are_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            live_settings(cfg(["poisson", "poisson"]))
        with pytest.raises(ValueError, match="empty"):
            live_settings(cfg([]))

    def test_composite_name_cannot_shadow_a_model(self):
        with pytest.raises(ValueError, match="collides"):
            live_settings(cfg(["poisson", "gp"], name="gp"))

    def test_members_default_to_the_live_models(self):
        live = live_settings(cfg(["poisson", "gp"]))
        assert live["composite"]["members"] == ["poisson", "gp"]


class TestRegistry:
    def test_names_are_unique_and_prediction_files_distinct(self):
        names = registry.names()
        assert len(names) == len(set(names))

    def test_every_live_capable_model_names_its_fit_and_predict(self):
        for spec in registry.MODELS:
            if spec.live_capable:
                assert spec.fit and spec.predict, spec.name

    def test_dependencies_exist(self):
        for spec in registry.MODELS:
            for dep in spec.depends_on:
                assert dep in registry.names(), f"{spec.name} depends on {dep}"

    def test_importing_the_registry_imports_no_model(self):
        """Lazy by design: the RNN needs torch, the Bayesian model pymc."""
        for mod in (
            "src.models.unused.rnn_lstm",
            "src.models.unused.bayesian_hierarchical",
        ):
            sys.modules.pop(mod, None)
        importlib.reload(registry)
        assert "src.models.unused.rnn_lstm" not in sys.modules
        assert "src.models.unused.bayesian_hierarchical" not in sys.modules

    def test_load_fns_resolves_production_models(self):
        for name in ("baseline", "linear", "poisson", "gp"):
            fit, predict = registry.load_fns(name)
            assert callable(fit) and callable(predict)

    def test_meta_models_cannot_be_fitted_live(self):
        with pytest.raises(ValueError, match="cannot forecast"):
            registry.load_fns("stacking")
