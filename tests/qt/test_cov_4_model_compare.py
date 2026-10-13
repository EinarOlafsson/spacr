"""Model Compare can be preloaded by another screen, and refuses to double-run.

The Model Zoo hands this screen two models and a folder rather than reaching
into its widgets, so the preload path is public API and has to apply the field
count before the folder -- loading N fields and immediately reloading with a
different N is visible as a stutter and doubles the work. The job plumbing has
the other half of the contract: a second run while one is in flight is refused
inline, and a result that cannot be applied lands in the status label rather
than raising on the GUI thread.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import model_compare as mc
from spacr.qt.screens.model_compare import ModelCompareScreen


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep this screen's manifests out of the user's real run history."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture
def fields(tmp_path):
    """A folder of three ``.npy`` fields, the shape spaCR leaves on disk."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        np.save(folder / f"field_{i}.npy",
                rng.integers(0, 4096, size=(32, 32), dtype=np.uint16))
    return str(folder)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = ModelCompareScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# -- preloading --------------------------------------------------------------

def test_another_screen_can_preload_both_models_and_the_folder(screen, fields):
    """The Model Zoo's "compare these two" is this call, not widget poking."""
    assert screen.configure(model_a="cyto3", model_b="nuclei",
                            folder=fields, n_fields=2) is True
    assert screen._panel_a.model_edit.text() == "cyto3"
    assert screen._panel_b.model_edit.text() == "nuclei"
    assert screen.source_folder() == fields


def test_the_field_count_is_applied_before_the_folder_is_loaded(screen,
                                                                fields):
    """Loading N fields and reloading with another N is work done twice."""
    screen.configure(folder=fields, n_fields=2)
    assert screen._fields_box.value() == 2
    assert len(screen._field_names) == 2


def test_preloading_without_a_folder_leaves_the_source_alone(screen):
    """An empty argument means "do not touch this control"."""
    assert screen.configure(model_a="cyto3") is True
    assert screen.source_folder() == ""
    assert screen._panel_b.model_edit.text() != "cyto3"


# -- refusing a second run ---------------------------------------------------

def test_a_second_load_while_one_is_running_is_refused_inline(screen, fields):
    """Two loads at once would interleave two folders into one field list."""
    screen._busy = True
    assert screen.set_source(fields) is False
    assert "already running" in screen.status_text()
    assert screen.source_folder() == ""


# -- job plumbing ------------------------------------------------------------

def test_the_job_body_leaves_its_result_where_the_gui_thread_looks(
        qtbot, qt_theme_applied, fields, monkeypatch):
    """The worker and the GUI thread meet at one dict, and nowhere else."""
    real_make_thread = mc.make_thread
    captured = {}

    def _capture(fn, settings, *args, **kwargs):
        thread, worker = real_make_thread(fn, settings, *args, **kwargs)
        thread.start = lambda: None
        captured["fn"] = fn
        captured["payload"] = settings
        return thread, worker

    monkeypatch.setattr(mc, "make_thread", _capture)
    widget = ModelCompareScreen(threaded=True)
    qtbot.addWidget(widget)
    assert widget.set_source(fields) is True
    assert widget.is_busy()

    captured["fn"](captured["payload"])
    assert "result" in captured["payload"]

    widget._on_job_settled(True)
    assert widget.is_busy() is False
    assert widget.source_folder() == fields


def test_a_result_that_cannot_be_applied_lands_in_the_status_label(screen):
    """Raising on the GUI thread here would take the window with it."""
    def _refuses(_result):
        raise ValueError("the fields did not decode")

    screen._pending.append(({"result": None}, _refuses, "loading fields"))
    settled = []
    screen.job_finished.connect(settled.append)
    screen._on_job_settled(True)
    assert settled == [False]
    assert "the fields did not decode" in screen.status_text()
