"""The ``spacr.regression_search`` shim was removed in 1.5.1.6.

Instruction 127, finding 5: the module only re-exported
:mod:`spacr.parameter_sweep` under the old names and raised a
``DeprecationWarning``. It shipped from v1.5.0.5 to v1.5.1.5; the maintainer
removed it in 1.5.1.6, and the release notes tell old
code to import :mod:`spacr.parameter_sweep` instead.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import spacr

ROOT = Path(__file__).resolve().parents[1]


def test_the_old_module_is_gone():
    """Read off the source tree, not ``importlib``: an editable install's
    finder can resolve the name from another checkout."""
    package = Path(spacr.__file__).resolve().parent
    assert not (package / "regression_search.py").exists(), (
        "spacr.regression_search was removed in 1.5.1.6; import "
        "spacr.parameter_sweep instead")
    assert not (package / "regression_search").exists()


def test_the_package_no_longer_advertises_it():
    assert "regression_search" not in dir(spacr), (
        "spacr still lists regression_search among its lazy submodules")


def test_the_replacement_module_imports():
    sweep = importlib.import_module("spacr.parameter_sweep")
    for name in ("SweepSpace", "run_sweep", "run_sweep_parallel",
                 "summarise_sweep", "DEFAULT_SWEEP_SPACE"):
        assert hasattr(sweep, name), name


def test_the_release_notes_name_the_replacement():
    notes = (ROOT / "docs/notes/release_notes_1.5.1.6.md").read_text(
        encoding="utf-8")
    assert "`spacr.regression_search` was removed" in notes
    assert "spacr.parameter_sweep" in notes
