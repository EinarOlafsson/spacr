"""Four analysis screens: how a job ends, and what a load or an open reports.

Each screen runs its work through a job plumbing that may never raise into a
Qt slot and never open a modal. Whatever fails -- the worker, the completion
handler on the GUI thread, a database read, a crop opener -- ends in exactly
one sentence on the screen's status line, with nothing half-applied.

Screens: Agreement (``_run_job``), Lineage (load, collisions, opening crops),
Model Compare (refused loads and failed results), Model Zoo (the benchmark
field folder: typed, picked, cancelled, and the field-count box).
"""
from __future__ import annotations

import os
import sqlite3
import traceback

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep these screens' manifests out of the user's real run history."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

class _InlineWorker(QObject):
    """The two signals ``_run_job`` connects, with no thread behind them."""

    error = Signal(str)
    finished = Signal(bool)


class _InlineThread(QObject):
    """A ``start()`` that runs the job where the test can see it."""

    finished = Signal()

    def __init__(self, run):
        super().__init__()
        self._run = run

    def start(self):
        self._run()

    def isRunning(self):  # noqa: N802 - QThread's spelling
        return False


def _inline_make_thread(fn, settings, *_args, **_kwargs):
    """``make_thread`` that runs the job body inline, signalling as the real one."""
    worker = _InlineWorker()

    def run():
        ok = True
        try:
            fn(settings)
        except Exception:  # noqa: BLE001
            ok = False
            worker.error.emit(traceback.format_exc())
        worker.finished.emit(ok)
        thread.finished.emit()

    thread = _InlineThread(run)
    return thread, worker


def test_an_agreement_job_that_fails_anywhere_ends_in_one_status_line(
        subtests, qtbot, monkeypatch):
    """Worker or handler failure: job_finished(False), a sentence, no report."""
    import spacr.qt.screens.agreement as mod
    from spacr.qt.screens.agreement import AgreementScreen

    monkeypatch.setattr(mod, "make_thread", _inline_make_thread)

    def explodes(_result):
        raise RuntimeError("could not paint the table")

    def boom():
        raise ValueError("the database went away")

    cases = [
        ("threaded: the handler fails", True, lambda: {"kappa": 0.5},
         explodes, "Agreement failed: could not paint the table", []),
        ("threaded: the worker raises", True, boom, None,
         "Agreement failed: ValueError: the database went away", []),
        ("inline: the handler fails", False, lambda: {"kappa": 0.5},
         explodes, "Agreement failed: could not paint the table", []),
        ("inline: the worker raises", False, boom, None,
         "Agreement failed: the database went away", []),
        ("threaded: the result reaches the handler", True,
         lambda: {"kappa": 0.5}, None, None, [{"kappa": 0.5}]),
    ]
    for case_id, threaded, job, handler, status, delivered_expected in cases:
        with subtests.test(msg=case_id):
            screen = AgreementScreen(threaded=threaded)
            qtbot.addWidget(screen)
            before = screen.status_text()
            delivered = []
            settled = []
            screen.job_finished.connect(settled.append)

            started = screen._run_job(job, handler or delivered.append)

            ok = status is None
            assert settled == [ok]
            assert started is (True if threaded else ok)
            assert delivered == delivered_expected
            assert screen.status_text() == (before if ok else status)
            assert screen.last_error == ("" if ok else status)
            assert screen.report() is None
            assert screen.is_busy() is False
            assert screen.active_jobs() == 0


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

_F1 = ("plate1", "r1", "c1", "f1")


def _lineage_frames():
    """One cell holding one nucleus, which is enough to build a tree."""
    def rows(table, label, parent):
        plate, row, column, field = _F1
        record = {"plateID": plate, "rowID": row, "columnID": column,
                  "fieldID": field, "object_label": label,
                  f"{table}_area": 100.0 + label}
        if parent is not None:
            record["cell_id"] = parent
        return pd.DataFrame([record])

    return {"cell": rows("cell", 7, None), "nucleus": rows("nucleus", 1, 7)}


def test_the_lineage_screen_reports_every_load_collision_and_open(
        subtests, qtbot, monkeypatch, tmp_path):
    """Missing databases, key collisions and openers all end in exact words."""
    from spacr import lineage as lin
    from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND,
                                           register_object_opener,
                                           unregister_object_opener)
    from spacr.qt.screens import lineage as screen_module

    def fresh():
        view = screen_module.LineageScreen(threaded=False)
        qtbot.addWidget(view)
        return view

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        for table, frame in _lineage_frames().items():
            frame.to_sql(table, connection, index=False)

    for case_id, text in [("no database named", ""),
                          ("a path that is not a file",
                           str(tmp_path / "not_here.db")),
                          ("a folder, not a file", str(tmp_path))]:
        with subtests.test(msg=f"load: {case_id}"):
            screen = fresh()
            screen._db.setText(text)
            screen.load()
            assert screen.status.text() == "Choose a measurements database first."
            assert screen.tree.topLevelItemCount() == 0

    with subtests.test(msg="load: a real database builds its tree"):
        screen = fresh()
        screen._db.setText(f"  {db}  ")
        screen.load()
        assert screen.tree.topLevelItemCount() == 1
        assert screen.status.text() == (
            "1 of 1 parent object(s) in the tree · every child has a parent")

    with subtests.test(msg="collisions: none leaves the summary quiet"):
        screen = fresh()
        screen.set_frames(_lineage_frames())
        assert screen.collision_note() == ""
        assert "name more than one object" not in screen.summary.text()

    with subtests.test(msg="collisions: a shared key is announced"), \
            monkeypatch.context() as patch:
        patch.setattr(lin, "forest_key_collisions", lambda _forest: {
            "plate1_r1_c1_f1_9": ["cell", "pathogen"],
            "plate1_r1_c1_f1_7": ["cell", "nucleus"]})
        screen = fresh()
        screen.set_frames(_lineage_frames())
        note = screen.collision_note()
        assert note == (
            "2 object key(s) name more than one object — plate1_r1_c1_f1_7 "
            "is a cell and a nucleus. Every other view treats them as one "
            "object, so opening this family will show fewer crops than it "
            "has objects.")
        assert screen.summary.text().endswith(f"\n{note}")

    key = ["plate1_r1_c1_f1_7"]
    with subtests.test(msg="open: nowhere to open points at Annotate"):
        screen = fresh()
        screen.set_frames(_lineage_frames())
        assert screen._open(key, "because") is None
        assert screen.status.text() == (
            "Open the Annotate screen first — it is what shows crops.")

    requests = []

    def open_it(request):
        requests.append((list(request.keys), request.reason))
        return list(request.keys)

    register_object_opener(DEFAULT_OPEN_KIND, open_it)
    try:
        with subtests.test(msg="open: a working opener returns its answer"):
            screen = fresh()
            screen.set_frames(_lineage_frames())
            before = screen.status.text()
            assert screen._open(key, "because") == key
            assert requests == [(key, "because")]
            assert screen.status.text() == before

        with subtests.test(msg="open: an opener that raises is a sentence"), \
                monkeypatch.context() as patch:
            screen = fresh()
            screen.set_frames(_lineage_frames())

            def refuse(_keys, **_kwargs):
                raise RuntimeError("the annotate screen has no crops for those")

            patch.setattr(screen, "open_objects", refuse)
            assert screen._open(key, "because") is None
            assert screen.status.text() == (
                "Could not open those objects: the annotate screen has no "
                "crops for those")
            assert screen.tree.topLevelItemCount() == 1
    finally:
        unregister_object_opener(DEFAULT_OPEN_KIND, open_it)


# ---------------------------------------------------------------------------
# Model Compare
# ---------------------------------------------------------------------------

def _npy_fields(folder, n=3):
    """A folder of ``n`` ``.npy`` fields, the shape spaCR leaves on disk."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        np.save(folder / f"field_{i}.npy",
                rng.integers(0, 4096, size=(32, 32), dtype=np.uint16))
    return str(folder)


def test_model_compare_refuses_a_second_job_and_words_every_failure(
        subtests, qtbot, qt_theme_applied, tmp_path):
    """Busy refusals, failed handlers and blank tracebacks all become sentences."""
    from spacr.qt.screens.model_compare import ModelCompareScreen

    fields = _npy_fields(tmp_path / "plate1")

    def two_objects(images, _config):
        masks = []
        for image in images:
            mask = np.zeros(np.shape(image)[:2], dtype=np.int32)
            mask[2:10, 2:10] = 1
            mask[16:26, 16:26] = 2
            masks.append(mask)
        return masks

    def fresh():
        widget = ModelCompareScreen(threaded=False)
        widget.set_segment_fn(two_objects)
        qtbot.addWidget(widget)
        settled = []
        widget.job_finished.connect(settled.append)
        return widget, settled

    with subtests.test(msg="a load while busy is refused"):
        screen, settled = fresh()
        screen._busy = True
        assert screen.set_source(fields) is False
        assert screen.status_text() == (
            "Another Model Compare job is already running.")
        assert screen.last_error == screen.status_text()
        assert screen.source_folder() == ""
        assert settled == []

    def refuses(_result):
        raise ValueError("the fields did not decode")

    def refuses_quietly(_result):
        raise KeyError()

    settle_cases = [
        ("a handler that raises names the operation",
         [({"result": None}, refuses, "loading fields")], True,
         "Loading fields failed: the fields did not decode"),
        ("a handler with no message is named by its class",
         [({"result": None}, refuses_quietly, "comparison")], True,
         "Comparison failed: KeyError"),
    ]
    for case_id, pending, ok, status in settle_cases:
        with subtests.test(msg=f"settled: {case_id}"):
            screen, settled = fresh()
            screen._pending.extend(pending)
            screen._busy = True
            screen._on_job_settled(ok)
            assert settled == [False]
            assert screen.is_busy() is False
            assert screen._pending == []
            assert screen.status_text() == status

    error_cases = [
        ("the last traceback line is quoted",
         "Traceback (most recent call last):\n  File \"x.py\", line 1\n"
         "RuntimeError: cellpose is not installed\n\n",
         [({}, None, "comparison")],
         "Comparison failed: RuntimeError: cellpose is not installed"),
        ("a blank traceback still makes a sentence", "  \n \n\t ",
         [({}, None, "comparison")], "Comparison failed: unknown error"),
        ("no pending job is a generic job", "OSError: gone", [],
         "Job failed: OSError: gone"),
    ]
    for case_id, text, pending, status in error_cases:
        with subtests.test(msg=f"worker error: {case_id}"):
            screen, _settled = fresh()
            assert screen.set_source(fields) is True
            assert len(screen.field_names()) == 3
            assert screen.compare() is True
            assert len(screen.metric_rows()) == 3
            assert screen.report().n_fields == 3
            screen._pending.extend(pending)
            screen._on_worker_error_text(text)
            assert screen.status_text() == status
            assert screen.last_error == status
            assert screen.metric_rows() == []
            assert screen.report() is None


# ---------------------------------------------------------------------------
# Model Zoo
# ---------------------------------------------------------------------------

def test_the_model_zoo_loads_benchmark_fields_only_from_a_real_answer(
        subtests, qtbot, monkeypatch, tmp_path):
    """Typed or picked folders load; a cancel or a count edit with no folder does not."""
    from spacr.qt.screens import model_zoo as mz
    from spacr.qt.screens.model_zoo import ModelZooScreen

    def fresh():
        widget = ModelZooScreen(threaded=False)
        qtbot.addWidget(widget)
        return widget

    def plate(name, n):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(n):
            np.save(folder / f"A01_f{i:02d}.npy",
                    np.full((32, 32), 100.0 + i, dtype=np.float32))
        return str(folder)

    def picker(patch, answer):
        asked = []

        class _Dialog:
            @staticmethod
            def getExistingDirectory(_parent, title, start):
                asked.append((title, start))
                return answer

        patch.setattr(mz, "QFileDialog", _Dialog)
        return asked

    one = plate("plate1", 1)
    three = plate("plate3", 3)

    with subtests.test(msg="a typed folder is loaded"):
        screen = fresh()
        screen._fields_edit.setText(one)
        screen._on_fields_typed()
        assert screen.fields_folder() == one
        assert screen.field_names() == ["A01_f00"]
        assert screen.status_text() == f"Loaded 1 field(s) from {one}: A01_f00."

    with subtests.test(msg="a typed folder that is not there is refused"):
        screen = fresh()
        screen._fields_edit.setText(str(tmp_path / "gone"))
        screen._on_fields_typed()
        assert screen.fields_folder() == ""
        assert screen.field_names() == []
        assert screen.status_text().startswith(
            "Could not load benchmark fields: ")

    with subtests.test(msg="a picked folder is loaded"), \
            monkeypatch.context() as patch:
        screen = fresh()
        asked = picker(patch, one)
        screen._pick_fields_folder()
        assert asked == [("Choose a folder of fields", os.getcwd())]
        assert screen.fields_folder() == one
        assert screen.field_names() == ["A01_f00"]

    with subtests.test(msg="the picker reopens on the loaded folder"), \
            monkeypatch.context() as patch:
        screen = fresh()
        assert screen.set_fields_source(one) is True
        asked = picker(patch, "")
        screen._pick_fields_folder()
        assert asked == [("Choose a folder of fields", one)]
        assert screen.fields_folder() == one
        assert screen.field_names() == ["A01_f00"]

    with subtests.test(msg="a cancelled picker loads nothing"), \
            monkeypatch.context() as patch:
        screen = fresh()
        picker(patch, "")
        before = screen.status_text()
        screen._pick_fields_folder()
        assert screen.fields_folder() == ""
        assert screen.field_names() == []
        assert screen.status_text() == before

    with subtests.test(msg="the field count relabels, and reloads a real folder"):
        screen = fresh()
        screen._fields_box.setValue(2)
        assert screen._btn_test.text() == "Test on 2 fields"
        assert screen.fields_folder() == ""
        assert screen.field_names() == []
        assert screen.set_fields_source(three) is True
        assert screen.field_names() == ["A01_f00", "A01_f01"]
        screen._fields_box.setValue(3)
        assert screen._btn_test.text() == "Test on 3 fields"
        assert screen.fields_folder() == three
        assert screen.field_names() == ["A01_f00", "A01_f01", "A01_f02"]

    with subtests.test(msg="a load while busy is refused"):
        screen = fresh()
        screen._busy = True
        assert screen.set_fields_source(one) is False
        assert screen.status_text() == "Another Model Zoo job is already running."
        assert screen.fields_folder() == ""
