"""Regression's masthead opens the diagnostics a run wrote.

Instruction 322. ``spacr.ml`` writes design, residual and inference
panels into the run's results folder as a regression finishes -- the
checks that would have caught 824 guides in 587 wells returning a
confident P value from a rank-deficient matrix.

Until now nothing opened them. The panels existed on disk and no
surface offered them, which is the same shape as the defect 322 was
filed for: the diagnostics were written and unreachable.

The button is not a module. There is nothing to compute and nothing to
configure -- it shows files that are already there -- which is why it
can be pressed when there are none, and why saying so is part of the
contract rather than an error path.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import regression as reg


class _Screen:
    """A host screen with a project folder and nothing else."""

    app_key = "regression"

    def __init__(self, root=""):
        self._root = str(root)


@pytest.fixture
def opener(monkeypatch, tmp_path):
    """A DiagnosticsOpener over a project at ``tmp_path``."""
    monkeypatch.setattr(reg, "project_path", lambda screen: str(tmp_path))
    return reg.DiagnosticsOpener(_Screen(tmp_path)), tmp_path


def _write_run(root, name="run1", note=None):
    """A results folder with a diagnostics directory in it."""
    folder = root / reg.RESULTS_DIRNAME / name / reg.DIAGNOSTICS_DIRNAME
    folder.mkdir(parents=True)
    (folder / "residuals.png").write_bytes(b"")
    if note:
        (folder / "residual_panels_not_available.txt").write_text(note)
    return folder


def test_it_finds_the_diagnostics_a_run_wrote(opener):
    found, root = opener
    written = _write_run(root)
    assert found._folder() == str(written)


def test_a_project_with_no_run_has_nothing_to_show(opener):
    """The button is pressable before any regression has been run.

    Returning None rather than raising is what lets `open` say so.
    """
    found, _root = opener
    assert found._folder() is None


def test_it_offers_the_newest_run_not_the_first_it_finds(opener):
    """A project accumulates results folders.

    The one the user just produced is the one they mean; showing an
    older one silently would draw panels for a fit they are not looking
    at, and nothing on the page would say which.
    """
    found, root = opener
    old = _write_run(root, "run_old")
    new = _write_run(root, "run_new")
    os.utime(old, (1, 1))
    os.utime(new, (10_000_000, 10_000_000))

    assert found._folder() == str(new)


def test_the_key_and_the_folder_name_match_what_ml_writes():
    """One spelling, shared with the writer.

    `spacr.ml._write_regression_diagnostics` joins the results folder
    with this name. Two spellings is the button looking somewhere the
    writer does not use, which fails as an empty folder rather than an
    error.
    """
    import inspect

    from spacr import ml

    source = inspect.getsource(ml._write_regression_diagnostics)
    assert f'"{reg.DIAGNOSTICS_DIRNAME}"' in source


def test_the_button_describes_itself_without_a_registry_row():
    """There is no module behind it, so `fold_description` must still
    answer -- otherwise the button carries no tooltip and takes the
    colour of finished code."""
    from spacr.qt.screens.map_barcodes import fold_description

    name, detail, stage = fold_description(reg.DIAGNOSTICS_KEY)
    assert name == "Diagnostics"
    assert detail and stage == "beta"


def test_it_is_on_the_masthead_with_the_other_folds():
    assert reg.DIAGNOSTICS_KEY in reg.FOLDED_APPS
