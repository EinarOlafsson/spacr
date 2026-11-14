"""Annotate screen: auto-annotate preview, text reports, and retiring worker threads.

* The auto-annotate dialog's values hint and preview: the hint shows what a
  metadata column really holds, and Apply is enabled only by a preview that
  matched something under the current inputs.
* Coverage and learning-curve reports: no source asks for one, a failure
  names its exception, a success opens one window per report title.
* Thread retirement and close: every retrain/page thread is interrupted,
  drained with the close timeout, retired with ``deleteLater`` only when it
  stopped, and a thread whose C++ half is already gone never turns into a
  traceback out of the event loop.

Screens are real :class:`AnnotateScreen` widgets on a real on-disk experiment.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from spacr.qt import annotate_engine as engine
from spacr.qt.screens import annotate as annotate_mod

pytestmark = pytest.mark.qt


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A small experiment folder: 8 crops and a png_list that names them."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(8):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        p = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    db = src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    engine.ensure_annotation_column(str(db), "annotate")
    return src


@pytest.fixture
def screen(qtbot, qt_theme_applied, experiment: Path):
    """An open Annotate screen on a pinned 3x3 grid, page 0 decoded."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    scr._settings.grid_rows = 3
    scr._settings.grid_cols = 3
    scr._settings.image_size = (24, 24)
    scr._compute_grid_dims = lambda: None
    scr._rebuild_grid()
    scr._open_source(str(experiment))
    qtbot.waitUntil(lambda: len(scr._page_paths) == 8, timeout=10000)
    qtbot.waitUntil(lambda: scr._page_worker is None, timeout=10000)
    yield scr
    if scr._worker is not None:
        scr._worker.stop(wait=True)


class _Recorder:
    """Records what a message box was asked to show."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args[1:])


def _raises(exc):
    def _raise(*_args, **_kwargs):
        raise exc
    return _raise


# ---------------------------------------------------------------------------
# _AutoAnnotateDialog
# ---------------------------------------------------------------------------

def _dialog(qtbot, experiment=None):
    settings = engine.AnnotateSettings()
    if experiment is not None:
        settings.src = str(experiment)
        settings.db_path = str(experiment / "measurements" / "measurements.db")
    dlg = annotate_mod._AutoAnnotateDialog(settings)
    qtbot.addWidget(dlg)
    return dlg


def test_the_auto_annotate_values_hint_shows_what_the_column_holds(
        qtbot, qt_theme_applied, experiment, monkeypatch, subtests):
    """The hint lists up to 12 real values; failures and other sources leave it."""
    db = str(experiment / "measurements" / "measurements.db")
    first_column = engine.METADATA_COLUMNS[0]
    untouched = "untouched"
    cases = [
        ("twenty values show twelve and count the rest", True, True,
         lambda *a: [f"c{i}" for i in range(20)],
         ", ".join(f"c{i}" for i in range(12)) + " (+8 more)", True),
        ("twelve values show them all", True, True,
         lambda *a: [f"c{i}" for i in range(12)],
         ", ".join(f"c{i}" for i in range(12)), True),
        ("no values leave the hint", True, True, lambda *a: [], untouched,
         True),
        ("a column that cannot be read leaves the hint", True, True,
         _raises(sqlite3.OperationalError("no")), untouched, True),
        ("the measurement source does not read the column", False, True,
         lambda *a: ["x"], untouched, False),
        ("no open database does not read the column", True, False,
         lambda *a: ["x"], untouched, False),
    ]
    for case_id, metadata, has_db, values, hint, reads in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            dlg = _dialog(qtbot, experiment if has_db else None)
            if not metadata:
                dlg._source.setCurrentIndex(1)
            calls = []
            m.setattr(engine, "metadata_values",
                      lambda *a: calls.append(a) or values(*a))
            dlg._values.setPlaceholderText(untouched)
            dlg._apply.setEnabled(True)
            dlg._on_metadata_column()
            assert dlg._values.placeholderText() == hint
            assert calls == ([(db, first_column)] if reads else [])
            assert dlg._apply.isEnabled() is False


def test_the_auto_annotate_preview_counts_matches_before_apply_is_allowed(
        qtbot, qt_theme_applied, experiment, monkeypatch, subtests):
    """Apply follows the last preview; refusals say why and clear the matches."""
    db = str(experiment / "measurements" / "measurements.db")
    first_column = engine.METADATA_COLUMNS[0]

    with subtests.test(msg="no source open says so"):
        dlg = _dialog(qtbot)
        dlg._on_preview()
        assert dlg._preview_label.text() == "Open a source first."
        assert dlg._apply.isEnabled() is False

    with subtests.test(msg="typed values are matched once each"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(engine, "paths_by_metadata",
                  lambda *a: calls.append(a) or ["a.png", "a.png", "b.png"])
        dlg = _dialog(qtbot, experiment)
        dlg._values.setText(" c1, ,c2 ")
        dlg._value.setValue(3)
        dlg._on_preview()
        assert calls == [(db, first_column, ["c1", "c2"])]
        assert dlg.matched_paths() == ["a.png", "b.png"]
        assert dlg._apply.isEnabled() is True
        assert dlg._preview_label.text() == (
            '2 object(s) match. Apply writes class 3 to "annotate".')

    with subtests.test(msg="no typed values take every value of the column"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(engine, "metadata_values", lambda *a: ["r1", "r2"])
        m.setattr(engine, "paths_by_metadata",
                  lambda *a: calls.append(a) or [])
        dlg = _dialog(qtbot, experiment)
        dlg._on_preview()
        assert calls == [(db, first_column, ["r1", "r2"])]
        assert dlg._preview_label.text() == "Nothing matches."
        assert dlg._apply.isEnabled() is False
        assert dlg.matched_paths() == []

    with subtests.test(msg="measurements with no rules ask for one"):
        dlg = _dialog(qtbot, experiment)
        dlg._source.setCurrentIndex(1)
        dlg._rules.setPlainText("\n  \n")
        dlg._on_preview()
        assert dlg._preview_label.text() == "Add at least one rule."
        assert dlg._apply.isEnabled() is False

    with subtests.test(msg="measurement rules are parsed and matched"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(engine, "paths_by_measurements",
                  lambda *a: calls.append(a) or ["c.png"])
        dlg = _dialog(qtbot, experiment)
        dlg._source.setCurrentIndex(1)
        dlg._rules.setPlainText("cell_area >= 500\n\nnucleus_area < 2e2")
        dlg._on_preview()
        assert calls == [(db, "annotate", [
            {"column": "cell_area", "threshold": 500.0,
             "direction": "higher"},
            {"column": "nucleus_area", "threshold": 200.0,
             "direction": "lower"}])]
        assert dlg._preview_label.text() == (
            '1 object(s) match. Apply writes class 1 to "annotate".')
        assert dlg._apply.isEnabled() is True

    with subtests.test(msg="a failing rule says which and disarms a good preview"), \
            monkeypatch.context() as m:
        m.setattr(engine, "paths_by_metadata", lambda *a: ["a.png"])
        dlg = _dialog(qtbot, experiment)
        dlg._on_preview()
        assert dlg._apply.isEnabled() is True
        dlg._source.setCurrentIndex(1)
        dlg._rules.setPlainText("no_such_column > 1")
        m.setattr(engine, "paths_by_measurements", _raises(
            ValueError("no measurement table has 'no_such_column'")))
        dlg._on_preview()
        assert dlg._preview_label.text() == (
            "Could not preview: no measurement table has 'no_such_column'")
        assert dlg._apply.isEnabled() is False
        assert dlg.matched_paths() == []


# ---------------------------------------------------------------------------
# Coverage and learning-curve reports
# ---------------------------------------------------------------------------

def test_the_coverage_and_learning_curve_reports_open_refuse_or_name_failures(
        qtbot, qt_theme_applied, screen, monkeypatch, subtests):
    """No source asks for one; a failure names its exception; a report opens once."""
    from spacr import active_learning as al
    db = screen._settings.db_path

    bare = annotate_mod.AnnotateScreen()
    qtbot.addWidget(bare)
    for case_id, press, message in (
            ("coverage without a source", bare._on_coverage,
             "Open an experiment source to see annotation coverage."),
            ("learning curve without a source", bare._on_learning_curve,
             "Open an experiment source to see the learning curve.")):
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            info = _Recorder()
            m.setattr(QMessageBox, "information", staticmethod(info))
            press()
            assert info.calls == [("Open a source first", message)]

    failures = [
        ("a coverage report that cannot be built", "annotation_coverage",
         KeyError("plate"), screen._on_coverage,
         ("Coverage unavailable", "KeyError: 'plate'")),
        ("a learning curve that cannot be built", "learning_curve",
         RuntimeError("no rounds table"), screen._on_learning_curve,
         ("Learning curve unavailable", "RuntimeError: no rounds table")),
    ]
    for case_id, name, exc, press, expected in failures:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.setattr(al, name, _raises(exc))
            warn = _Recorder()
            m.setattr(QMessageBox, "warning", staticmethod(warn))
            press()
            assert warn.calls == [expected]

    with subtests.test(msg="coverage opens one window and rewrites it"), \
            monkeypatch.context() as m:
        calls = []
        bodies = iter(["plate  n\np1     40", "plate  n\np1     41"])
        m.setattr(al, "annotation_coverage",
                  lambda *a: calls.append(a) or {"n": 1})
        m.setattr(al, "format_coverage_summary",
                  lambda cov: next(bodies) if cov == {"n": 1} else "")
        screen._on_coverage()
        report = screen._reports["Annotation coverage"]
        assert calls == [(db, "annotate")]
        assert report.windowTitle() == "Annotation coverage"
        assert report._view.toPlainText() == "plate  n\np1     40"
        assert report.isVisible() is True
        screen._on_coverage()
        assert screen._reports["Annotation coverage"] is report
        assert report._view.toPlainText() == "plate  n\np1     41"

    with subtests.test(msg="the learning curve opens with its verdict"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(al, "learning_curve",
                  lambda *a: calls.append(a) or ["curve"])
        m.setattr(al, "should_stop",
                  lambda curve: "stop" if curve == ["curve"] else "go")
        m.setattr(al, "format_learning_curve",
                  lambda curve, verdict: f"round  acc\n1      0.7 {verdict}")
        screen._on_learning_curve()
        report = screen._reports["Active-learning rounds"]
        assert calls == [(db, "annotate")]
        assert report._view.toPlainText() == "round  acc\n1      0.7 stop"


# ---------------------------------------------------------------------------
# Retiring finished threads
# ---------------------------------------------------------------------------

def _record_delete(monkeypatch, worker):
    deleted = []
    monkeypatch.setattr(worker, "deleteLater", lambda: deleted.append(worker))
    return deleted


def test_a_finished_thread_is_retired_even_when_its_c_half_is_gone(
        qtbot, qt_theme_applied, monkeypatch, subtests):
    """The finished slots clear the worker, re-arm the UI and start what is queued."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    qtbot.waitUntil(lambda: scr._page_worker is None, timeout=10000)

    with subtests.test(msg="no retrain running still re-enables Retrain"):
        scr._btn_retrain.setEnabled(False)
        scr._on_retrain_finished()
        assert scr._retrain_worker is None
        assert scr._btn_retrain.isEnabled() is True

    with subtests.test(msg="a retrain whose signals were never connected is retired"), \
            monkeypatch.context() as m:
        worker = annotate_mod._RetrainWorker("db", "annotate", {})
        deleted = _record_delete(m, worker)
        scr._retrain_worker = worker
        scr._btn_retrain.setEnabled(False)
        scr._on_retrain_finished()
        assert deleted == [worker]
        assert scr._retrain_worker is None
        assert scr._btn_retrain.isEnabled() is True

    with subtests.test(msg="a retrain whose C++ half is gone is survivable"):
        worker = annotate_mod._RetrainWorker("db", "annotate", {})
        scr._retrain_worker = worker
        scr._btn_retrain.setEnabled(False)
        shiboken6.delete(worker)
        scr._on_retrain_finished()
        assert scr._retrain_worker is None
        assert scr._btn_retrain.isEnabled() is True

    with subtests.test(msg="a page thread is retired and nothing queued starts"), \
            monkeypatch.context() as m:
        started = []
        m.setattr(scr, "_start_page_worker", started.append)
        worker = annotate_mod._PageLoadWorker(1, [], lambda row: row)
        deleted = _record_delete(m, worker)
        scr._page_worker = worker
        scr._pending_page_load = None
        scr._on_page_worker_finished()
        assert (deleted, started, scr._page_worker) == ([worker], [], None)

    with subtests.test(msg="a dead page thread still starts the queued load"), \
            monkeypatch.context() as m:
        started = []
        m.setattr(scr, "_start_page_worker", started.append)
        worker = annotate_mod._PageLoadWorker(1, [], lambda row: row)
        scr._page_worker = worker
        request = (99, [], None, scr._settings)
        scr._pending_page_load = request
        shiboken6.delete(worker)
        scr._on_page_worker_finished()
        assert scr._page_worker is None
        assert scr._pending_page_load is None
        assert started == [request]

    with subtests.test(msg="a closing screen starts no queued load"), \
            monkeypatch.context() as m:
        started = []
        m.setattr(scr, "_start_page_worker", started.append)
        request = (100, [], None, scr._settings)
        scr._pending_page_load = request
        m.setattr(scr, "_closing", True)
        scr._on_page_worker_finished()
        assert started == []
        assert scr._pending_page_load == request
        scr._pending_page_load = None


# ---------------------------------------------------------------------------
# Closing over running threads
# ---------------------------------------------------------------------------

def test_closing_drains_each_thread_and_retires_only_the_ones_that_stopped(
        qtbot, qt_theme_applied, monkeypatch, subtests):
    """Interrupt, drain with the close timeout, then delete or park the thread."""
    retrain = ("retrain", "_retrain_worker",
               lambda: annotate_mod._RetrainWorker("db", "annotate", {}))
    page = ("page", "_page_worker",
            lambda: annotate_mod._PageLoadWorker(1, [], lambda row: row))
    for kind, attribute, make in (retrain, page):
        for stops in (True, False):
            case_id = f"a {kind} thread that {'stops' if stops else 'will not stop'}"
            with subtests.test(msg=case_id), monkeypatch.context() as m:
                scr = annotate_mod.AnnotateScreen()
                qtbot.addWidget(scr)
                scr._compute_grid_dims = lambda: None
                qtbot.waitUntil(lambda: scr._page_worker is None,
                                timeout=10000)
                worker = make()
                events = []
                m.setattr(worker, "requestInterruption",
                          lambda: events.append("interrupt"))
                m.setattr(worker, "deleteLater",
                          lambda: events.append("delete"))
                setattr(scr, attribute, worker)
                m.setattr(annotate_mod, "drain_thread",
                          lambda thread, timeout_ms: events.append(
                              (thread is worker, timeout_ms)) or stops)
                scr.close()
                assert events == (
                    ["interrupt", (True, annotate_mod.CLOSE_DRAIN_MS)]
                    + (["delete"] if stops else []))
                assert getattr(scr, attribute) is None
                assert scr._worker is None
                assert worker.parent() is None

    for kind, attribute, make in (retrain, page):
        cls = type(make())
        with subtests.test(msg=f"a {kind} thread whose C++ half goes mid-close"), \
                monkeypatch.context() as m:
            scr = annotate_mod.AnnotateScreen()
            qtbot.addWidget(scr)
            scr._compute_grid_dims = lambda: None
            qtbot.waitUntil(lambda: scr._page_worker is None, timeout=10000)
            setattr(scr, attribute, make())
            m.setattr(annotate_mod, "drain_thread",
                      lambda thread, timeout_ms: True)
            m.setattr(cls, "requestInterruption",
                      lambda self: shiboken6.delete(self))
            scr.close()
            assert getattr(scr, attribute) is None
