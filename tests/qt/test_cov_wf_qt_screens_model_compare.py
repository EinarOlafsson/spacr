"""Model Compare — the branches a normal comparison never takes.

The main suite (``tests/qt/test_model_compare_screen.py``) drives the happy
path: a folder with fields in it, two models that both segment something, a
worker that fails with a real traceback. Each case below is the *other* side
of a guard that stands between the user and a crash or a lie:

* the field spin box fires ``_reload`` on every change, including the changes
  made **before a folder was ever chosen** — re-reading ``""`` would replace
  the opening instructions with a load error for a folder nobody named;
* a comparison that came back with **no rows at all** must still fill in the
  status line and must not try to select row 0 of an empty table;
* the thread sweep sees **every** outstanding job, and dropping the references
  of one that is still running takes the interpreter down with it;
* a worker error whose traceback is **empty** must still produce a sentence —
  "Job failed:" with nothing after it tells the user nothing;
* an entry in ``matched`` that is **not a label of this mask** (0, a negative,
  or an id from the other model's mask) must colour nothing, because the one
  thing this preview promises is that teal means "has a partner".

One arc in ``_on_worker_error_text`` is deliberately left open, because no
input can close it: the *continue* side of ``if candidate.strip():``, which
would need the scan-back loop to reject a line and go on to the next one. It
never does. The loop reads ``reversed(str(tb).strip().splitlines())``, and not
one of the characters ``splitlines()`` treats as a line boundary survives
``strip()`` at the end of a string: line feed, carriage return, vertical tab,
form feed, the four ASCII separators, NEL, and the two Unicode line/paragraph
separators are all ``isspace()``. So a stripped traceback either is empty, and
the body never runs (the blank-traceback test below), or it ends in a
non-blank character, which puts a non-blank line first under ``reversed`` and
breaks on iteration one. Checked by walking every Unicode code point, not
argued from the ASCII ones.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

import spacr.model_compare as spmc
from spacr.qt.screens.model_compare import (
    COLOUR_MATCHED,
    COLOUR_UNMATCHED,
    ModelCompareScreen,
    compose_overlay,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep this screen's manifests out of the user's real run history.

    ``_run_job`` goes through :func:`spacr.qt.bridge.make_thread`, which
    journals by default; left alone a Qt suite buries the records of real
    analyses under its own debris.
    """
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture(autouse=True)
def _never_load_a_real_model(monkeypatch):
    """A Cellpose download inside this file would be a 2 GB hung CI job."""
    def _boom(*_a, **_k):
        raise AssertionError(
            "the screen tried to load a real Cellpose model — inject a "
            "segment_fn instead")

    monkeypatch.setattr(spmc, "segment_with_cellpose", _boom)
    yield


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Every error path here must land inline; a modal dialog hangs offscreen."""
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


def a_field(size: int = 48) -> np.ndarray:
    """A deterministic field: two bright blobs on a dim background."""
    image = np.full((size, size), 100.0, dtype=np.float32)
    image[4:16, 4:16] = 900.0
    image[28:40, 28:40] = 700.0
    return image


def mask_two_objects(size: int = 48) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int32)
    mask[4:16, 4:16] = 1
    mask[28:40, 28:40] = 2
    return mask


class FakeSegmenter:
    """Stands in for Cellpose: the same two objects, whatever it is asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, len(images)))
        return [mask_two_objects() for _ in images]


class _FakeThread:
    """A QThread whose running state the test decides.

    Real threads race, and these paths are about *which* of several jobs the
    bookkeeping touches — that has to be pinned exactly. Same injection idiom
    as ``tests/qt/test_cov_wf_qt_screens_db_browser.py``.
    """

    def __init__(self, running: bool):
        self._running = running
        self.calls = []

    def isRunning(self):  # noqa: N802 - QThread's spelling
        return self._running

    def quit(self):
        self.calls.append("quit")

    def wait(self, msecs):
        self.calls.append(("wait", msecs))
        return True


@pytest.fixture
def fields(tmp_path):
    """A folder of three ``.npy`` fields, the shape spaCR leaves on disk."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    for i in range(3):
        np.save(folder / f"field_{i}.npy", a_field())
    return str(folder)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — every job runs inline so assertions are exact."""
    widget = ModelCompareScreen(threaded=False)
    widget.set_segment_fn(FakeSegmenter())
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# the field count changes before there is anything to re-read
# ---------------------------------------------------------------------------

def test_the_field_count_only_reloads_once_a_folder_has_been_chosen(
        screen, fields):
    """Typing a field count first is the normal order, and must cost nothing.

    ``_reload`` is wired to ``valueChanged``, so it fires while the screen is
    still on its opening instructions. Without the folder guard the spin box
    would call ``set_source("")``, which replaces "choose a folder…" with a
    red "no fields found in " — an error about a folder the user never named,
    on a screen they have not used yet. Once a folder *is* loaded the same
    keystroke must re-read it, or the count silently disagrees with the table.
    """
    screen._fields_box.setValue(2)
    assert screen.source_folder() == ""
    assert screen.field_names() == []
    assert screen.last_error == ""
    assert "Choose a folder of fields" in screen.status_text()

    assert screen.set_source(fields) is True
    assert len(screen.field_names()) == 2

    screen._fields_box.setValue(1)
    assert screen.source_folder() == fields
    assert len(screen.field_names()) == 1
    assert "Loaded 1 field(s)" in screen.status_text()

    # The "no error yet" assertion above is worth something only if this
    # screen can produce one at all: naming a folder that is not there does.
    assert screen.set_source(fields + "_gone") is False
    assert screen.last_error.startswith("Load failed: no such folder:")


# ---------------------------------------------------------------------------
# a report with no rows in it
# ---------------------------------------------------------------------------

def test_a_comparison_with_no_rows_reports_itself_and_selects_nothing(
        screen, fields, monkeypatch):
    """Selecting row 0 of an empty table is how a screen dies on a bad plate.

    ``_apply_result`` fills the tables and then jumps to the first field so the
    previews are never blank after a run. When the backend hands back a report
    with no comparisons — a plate where every field failed its read, which is
    a normal state, not a crash — there is no row 0 to select and no mask to
    draw. The screen still has to say what happened: an empty table under a
    stale summary reads as "the two models agreed perfectly".
    """
    screen.set_source(fields)
    config_a, config_b = screen.model_configs()
    empty = spmc.ComparisonReport(model_a=config_a, model_b=config_b)
    real_compare_models = spmc.compare_models
    monkeypatch.setattr(spmc, "compare_models", lambda *a, **k: empty)

    assert screen.compare() is True
    assert screen.report() is empty
    assert screen.metric_rows() == []
    assert screen._row_table.currentRow() == -1
    assert screen.preview_captions() == ("Model A", "Model B")
    assert screen.summary_text() == "No field was compared."
    assert "Compared 0 field(s): 0 vs 0" in screen.status_text()
    assert screen.last_error == ""

    # The same call with rows in the report does select the first field, so
    # the assertions above are about the empty case and not about a screen
    # that never draws anything.
    monkeypatch.setattr(spmc, "compare_models", real_compare_models)
    assert screen.compare() is True
    assert len(screen.metric_rows()) == 3
    assert screen._row_table.currentRow() == 0
    assert screen.preview_captions()[0].startswith("A \u2014 field_0")
    assert "2 with a partner in B" in screen.preview_captions()[0]


# ---------------------------------------------------------------------------
# the sweep that retires finished jobs
# ---------------------------------------------------------------------------

def test_the_sweep_retires_the_stopped_job_and_keeps_the_running_one(
        qtbot, qt_theme_applied):
    """Releasing a running QThread's references crashes the whole process.

    ``_retire_finished_jobs`` is connected to *every* job's ``finished``, and
    it sweeps the whole list rather than naming a sender, so with two loads in
    flight it sees both. It must release only the one whose event loop has
    actually exited: a QThread garbage-collected while it is still running
    takes the interpreter down with it. Retiring too little is just as bad —
    ``active_jobs()`` never returns to zero and every wait on it times out.
    """
    widget = ModelCompareScreen(threaded=True)
    qtbot.addWidget(widget)
    alive, stopped = _FakeThread(True), _FakeThread(False)
    alive_worker, stopped_worker = object(), object()
    widget._jobs.append((stopped, stopped_worker))
    widget._jobs.append((alive, alive_worker))
    try:
        widget._retire_finished_jobs()
        assert widget.active_jobs() == 1
        assert widget._jobs == [(alive, alive_worker)]

        alive._running = False
        widget._retire_finished_jobs()
        assert widget.active_jobs() == 0
        assert widget._jobs == []
    finally:
        widget._jobs.clear()


# ---------------------------------------------------------------------------
# a worker traceback with nothing in it
# ---------------------------------------------------------------------------

def test_an_empty_worker_traceback_still_produces_a_sentence(screen, fields):
    """A status line reading "Comparison failed:" tells the user nothing.

    The worker's ``error`` signal carries whatever text the failure produced,
    and a process that died without unwinding Python — a segfaulting
    Cellpose build, a killed CUDA context — emits blank. The screen scans
    back from the end for the last line with anything on it; when there is no
    such line the status has to fall back to a name for the failure rather
    than trailing off mid-sentence. Either way the numbers from the run before
    have to go: a table of per-field ARIs left standing under "failed" reads
    as the result of the run that just failed.
    """
    screen.set_source(fields)
    assert screen.compare() is True
    assert len(screen.metric_rows()) == 3
    assert screen.report().n_fields == 3

    screen._pending.append(({}, None, "comparison"))
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n"
        "  File \"x.py\", line 1\nRuntimeError: cellpose is not installed")
    assert screen.status_text() == (
        "Comparison failed: RuntimeError: cellpose is not installed")
    assert screen.last_error == screen.status_text()
    assert screen.metric_rows() == []
    assert screen.report() is None

    # The same slot with nothing but whitespace on the wire: still a sentence.
    screen._on_worker_error_text("  \n \n\t ")
    assert screen.status_text() == "Comparison failed: unknown error"


# ---------------------------------------------------------------------------
# a matched label that is not a label of this mask
# ---------------------------------------------------------------------------

def test_a_matched_label_outside_this_mask_colours_nothing():
    """Teal has to mean "this object has a partner", or the preview lies.

    ``matched`` is a set of label ids taken from a match table, and the two
    masks are numbered independently — B's label 7 is a perfectly ordinary id
    that A may not have. Background (0) and a negative id can arrive the same
    way. Indexing the palette with any of them would either raise (7 is past
    the end) or paint the wrong object: 0 is the background row, and -3 wraps
    round to the last real label. The guard has to drop them all while still
    colouring the ids that are genuinely in range.
    """
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 2

    out = compose_overlay(None, mask, matched=[2, 7, 0, -3], alpha=0.5)

    assert out.shape == (8, 8, 3)
    matched_rgb = tuple((np.array(COLOUR_MATCHED) * 0.5).astype(np.uint8))
    unmatched_rgb = tuple((np.array(COLOUR_UNMATCHED) * 0.5).astype(np.uint8))
    assert tuple(out[5, 5]) == matched_rgb
    assert tuple(out[1, 1]) == unmatched_rgb
    assert tuple(out[0, 0]) == (0, 0, 0)
