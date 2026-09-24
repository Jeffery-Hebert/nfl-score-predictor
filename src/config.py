"""
The keys of config.yaml that code reads, parsed and validated in one place.

Most of config.yaml is documentation (its header says which keys are read). The
`live` block is not: it decides which models forecast real games, so a typo in
it must fail loudly here rather than silently drop a model from the ledger.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load(path=CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def live_settings(cfg: dict | None = None) -> dict:
    """{'models': [...], 'composite': {'name', 'members', 'method'}}, validated."""
    from src.models import registry

    cfg = load() if cfg is None else cfg
    live = cfg.get("live") or {}
    models = list(live.get("models") or [])
    comp = dict(live.get("composite") or {})
    comp.setdefault("name", "combined")
    comp.setdefault("members", list(models))
    comp.setdefault("method", "mean")

    if not models:
        raise ValueError("config.yaml live.models is empty -- nothing to forecast")
    if len(set(models)) != len(models):
        raise ValueError(f"config.yaml live.models has duplicates: {models}")
    for name in models:
        spec = registry.get(name)  # KeyError names the unknown model
        if not spec.live_capable:
            raise ValueError(
                f"live model {name!r} cannot forecast an unplayed game: {spec.note}"
            )
    missing = [m for m in comp["members"] if m not in models]
    if missing:
        raise ValueError(
            f"composite members {missing} are not in live.models -- a composite "
            "can only average forecasts that are actually made"
        )
    if comp["method"] != "mean":
        raise ValueError(f"unsupported composite method {comp['method']!r}")
    if comp["name"] in models or comp["name"] == "baseline":
        raise ValueError(f"composite name {comp['name']!r} collides with a model")
    return {"models": models, "composite": comp}
