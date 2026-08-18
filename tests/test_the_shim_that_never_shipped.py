"""The deprecation shim nothing uses, pinned while the decision is pending.

Instruction 127, finding 5: `spacr/regression_search.py` is 25 lines that
re-export :mod:`spacr.parameter_sweep` and raise a ``DeprecationWarning``, with
zero references in ``spacr/`` and zero in ``tests/``. The finding asks one
question -- "check whether any RELEASED version advertised the name. If it did,
keep one more cycle; if the rename never shipped, delete it."

ANSWERED 2026-08-18: the rename never shipped. None of the six tags in this
repository (v1.3.5, v1.3.6, v1.4.9.8, v1.4.9.9, v1.5.0.1, v1.5.0.4) carries
``spacr/regression_search.py``, and none carries ``spacr/parameter_sweep.py``
either -- both names arrived after the last release, so no installed spaCR has
ever exposed either of them and there is nothing downstream to keep working.

IT IS STILL HERE. Which modules exist is the maintainer's decision, so the
evidence is recorded and the module is left alone. What this file does in the
meantime is stop it rotting: an alias that has quietly stopped resolving is
worse than the deprecation it was written to soften, because it fails at the
importer's line with a name error about a module they were told still worked.
"""
from __future__ import annotations

import importlib
import warnings

import pytest


#: The old spellings the shim promises, and the new names behind them.
ALIASES = {
    "DEFAULT_SEARCH_SPACE": "DEFAULT_SWEEP_SPACE",
    "SearchSpace": "SweepSpace",
    "run_search": "run_sweep",
    "run_search_parallel": "run_sweep_parallel",
    "summarise_search": "summarise_sweep",
}


def _shim():
    return importlib.import_module("spacr.regression_search")


def test_importing_it_says_what_it_became():
    """A silent alias is the failure mode a deprecation shim exists to avoid:
    code keeps working and nobody is told to move."""
    import sys

    sys.modules.pop("spacr.regression_search", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _shim()
    messages = [str(record.message) for record in caught
                if issubclass(record.category, DeprecationWarning)]
    assert messages, "the shim deprecates nothing"
    assert "spacr.parameter_sweep" in messages[0], (
        "the warning does not name the module to move to")


def test_every_old_name_still_resolves_to_the_new_one():
    shim = _shim()
    sweep = importlib.import_module("spacr.parameter_sweep")
    for old, new in ALIASES.items():
        assert hasattr(shim, old), f"the shim promises {old} and lost it"
        assert getattr(shim, old) is getattr(sweep, new), (
            f"{old} no longer IS {new}; the alias points somewhere else")


def test_the_star_import_covers_the_whole_new_api():
    """`from .parameter_sweep import *` is only a promise while
    `parameter_sweep.__all__` and the shim agree. A name added to the sweep
    module reaches the old spelling with no edit here -- this asserts it."""
    shim = _shim()
    sweep = importlib.import_module("spacr.parameter_sweep")
    missing = [name for name in sweep.__all__ if not hasattr(shim, name)]
    assert not missing, f"the shim does not re-export {missing}"


def test_nothing_in_the_package_still_imports_the_old_name():
    """The finding's other half. If this ever fails, the module has a caller
    and the deletion question changes."""
    import ast
    import os

    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spacr")
    importers = []
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py") or name == "regression_search.py":
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read())
                except SyntaxError:                              # pragma: no cover
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    target = node.module
                elif isinstance(node, ast.Import):
                    target = ",".join(alias.name for alias in node.names)
                else:
                    continue
                if "regression_search" in target:
                    importers.append(
                        f"{os.path.relpath(path, root)}:{node.lineno}")
    assert not importers, (
        "spacr.regression_search has a caller now, so finding 5's answer "
        f"changed: {importers}")
