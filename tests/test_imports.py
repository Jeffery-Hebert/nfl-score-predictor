"""
Every module under src/ must import cleanly, without data and without running.

Why. Three experiment scripts sat broken against the current feature set for
days (KeyErrors, a pandas _x/_y column collision) and nothing noticed, because
nothing ran them. The RNN module could not even be imported without the data
directory, because it loaded its sequence file at import time. Importing every
module on every push catches the cheapest class of rot -- a renamed function, a
removed constant, a moved file -- the moment it happens.

Modules that need an optional dependency (torch for the RNN) are skipped, not
failed, when it is absent; CI deliberately does not install torch.

Run: pytest tests/test_imports.py -v
"""

import importlib
import importlib.util
import pkgutil

import pytest

import src

OPTIONAL = {"src.models.unused.rnn_lstm": "torch"}


def all_modules():
    return sorted(
        m.name
        for m in pkgutil.walk_packages(src.__path__, prefix="src.")
        if not m.ispkg
    )


@pytest.mark.parametrize("name", all_modules())
def test_module_imports(name):
    needs = OPTIONAL.get(name)
    if needs and importlib.util.find_spec(needs) is None:
        pytest.skip(f"{name} needs {needs}, which is optional")
    importlib.import_module(name)


def test_there_are_modules_to_check():
    assert len(all_modules()) > 50
