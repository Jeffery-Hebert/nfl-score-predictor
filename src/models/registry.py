"""
Every model in the project, described once.

Why this exists. Model knowledge used to be scattered: predict_week.py
hard-coded four models, stacking.py its three members, the README its run order,
and the scoreboard discovered whatever *_predictions.parquet happened to be on
disk -- including stale files from models nobody had re-run. Adding or swapping
a live model meant editing several files in step. Now:

  - src/models/run_all.py runs every backtest from this list, in dependency order;
  - src/predict/predict_week.py fits the live models named in config.yaml, and
    refuses any name that is not here or cannot predict an unplayed game;
  - the evaluation tools label models by status (production / shelved / ...).

Nothing here imports a model module at import time. Several pull in heavy or
optional dependencies (torch for the RNN, pymc for the Bayesian model), so the
registry stays importable in CI and on machines without them; the module is
loaded only when a model is actually used.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import pandas as pd

MODEL_TABLE = "data/processed/model_table.parquet"
DRIVE_TABLE = "data/processed/drive_model_table.parquet"
SCHEDULES = "data/raw/schedules.parquet"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # stem of data/processed/<name>_predictions.parquet
    module: str  # `python -m <module>` runs its walk-forward backtest
    status: str  # production | shelved | experimental | meta
    table: str  # which table it trains on: model | drive | meta
    fit: str | None = None  # attribute names in `module`, for live fitting
    predict: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    live_capable: bool = True  # can it forecast a game that has not been played?
    extra_inputs: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


MODELS: list[ModelSpec] = [
    # ------------------------------------------------------------ production
    ModelSpec(
        "baseline",
        "src.models.baseline",
        "production",
        "model",
        "fit_baseline",
        "predict_baseline",
        note="rule-based reference every model must beat",
    ),
    ModelSpec(
        "linear",
        "src.models.linear",
        "production",
        "model",
        "fit_linear",
        "predict_linear",
    ),
    ModelSpec(
        "poisson",
        "src.models.poisson_glm",
        "production",
        "model",
        "fit_poisson",
        "predict_poisson",
    ),
    ModelSpec(
        "gp",
        "src.models.gaussian_process",
        "production",
        "model",
        "fit_gp",
        "predict_gp",
        note="~25-30 minutes for the full walk-forward",
    ),
    # ------------------------------------------------------------------ meta
    ModelSpec(
        "stacking",
        "src.models.stacking",
        "meta",
        "meta",
        depends_on=("linear", "poisson", "gp"),
        live_capable=False,
        note="ridge on out-of-fold predictions; needs OOF rows that an "
        "unplayed game cannot have",
    ),
    ModelSpec(
        "composite",
        "src.models.composite",
        "meta",
        "meta",
        depends_on=("linear", "poisson", "gp"),
        live_capable=False,
        note="equal-weight average of the members in config.yaml; predict_week "
        "computes it live from the members' own live forecasts",
    ),
    # --------------------------------------------------------------- shelved
    ModelSpec(
        "rf",
        "src.models.unused.random_forest",
        "shelved",
        "model",
        "fit_rf",
        "predict_rf",
    ),
    ModelSpec(
        "xgb",
        "src.models.unused.xgboost_model",
        "shelved",
        "model",
        "fit_xgb",
        "predict_xgb",
    ),
    ModelSpec(
        "lgbm",
        "src.models.unused.lightgbm_model",
        "shelved",
        "model",
        "fit_lgbm",
        "predict_lgbm",
    ),
    ModelSpec(
        "catboost",
        "src.models.unused.catboost_model",
        "shelved",
        "model",
        "fit_catboost",
        "predict_catboost",
    ),
    ModelSpec(
        "logistic",
        "src.models.unused.logistic",
        "shelved",
        "model",
        "fit_logistic",
        "predict_logistic",
    ),
    ModelSpec(
        "mlp",
        "src.models.unused.mlp",
        "shelved",
        "model",
        "fit_mlp",
        "predict_mlp",
    ),
    ModelSpec(
        "bayesian",
        "src.models.unused.bayesian_hierarchical",
        "shelved",
        "model",
        "fit_bayesian",
        "predict_bayesian",
        note="season-level walk-forward (ADVI per fold)",
    ),
    ModelSpec(
        "rnn",
        "src.models.unused.rnn_lstm",
        "shelved",
        "model",
        "fit_rnn",
        "predict_rnn",
        extra_inputs=("data/processed/team_sequences.npz",),
        note="needs torch (CPU wheel; see requirements.txt); season-level",
    ),
    ModelSpec(
        "montecarlo",
        "src.models.unused.monte_carlo",
        "shelved",
        "drive",
        "fit_montecarlo",
        "predict_montecarlo",
        note="v1 drive simulator, kept as the record drive_model_v2 improved on",
    ),
    # ---------------------------------------------------------- experimental
    ModelSpec(
        "drivev2",
        "src.experiments.drive_model_v2",
        "experimental",
        "drive",
        "fit_drive_model",
        "predict_drive_model",
    ),
    ModelSpec(
        "drivechain",
        "src.experiments.drive_chain",
        "experimental",
        "drive",
        live_capable=False,
        extra_inputs=("data/raw/pbp.parquet",),
        note="Markov chain over field position; fits from play-by-play tables "
        "built in its main(), so it has no standalone live fit",
    ),
]

_BY_NAME = {m.name: m for m in MODELS}


def get(name: str) -> ModelSpec:
    if name not in _BY_NAME:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(_BY_NAME)}")
    return _BY_NAME[name]


def names(status: str | None = None) -> list[str]:
    return [m.name for m in MODELS if status is None or m.status == status]


def load_fns(name: str):
    """(fit_fn, predict_fn) for a live-capable model. Imports it lazily."""
    spec = get(name)
    if not spec.live_capable or not spec.fit:
        raise ValueError(
            f"{name!r} cannot forecast an unplayed game ({spec.note or 'no fit'})"
        )
    mod = importlib.import_module(spec.module)
    return getattr(mod, spec.fit), getattr(mod, spec.predict)


def load_table(kind: str) -> pd.DataFrame:
    """The frame a model of the given table kind trains and predicts on."""
    if kind == "model":
        df = pd.read_parquet(MODEL_TABLE)
        df["gameday"] = pd.to_datetime(df["gameday"])
        return df
    if kind == "drive":
        # drive_model_table plus the neutral-site / overtime context columns it
        # lacks -- the same loader the drive models' own backtests use.
        from src.experiments.drive_model_v2 import load_table as load_drive

        return load_drive()
    raise ValueError(f"no table loader for {kind!r}")


def table_inputs(kind: str) -> list[str]:
    return {
        "model": [MODEL_TABLE],
        "drive": [DRIVE_TABLE, SCHEDULES],
    }[kind]
