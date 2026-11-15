"""What six analysis screens say when an action cannot be carried out.

Every action here runs inside a Qt slot, so it may never raise and never open
a modal (a headless run has nobody to dismiss one); and it may never be
silent, because a button that appears to do nothing reads as a broken build.
Each case drives the real screen with its jobs inline and reads back the one
channel the screen reports through (status line, source label, summary or
provenance strip), plus what was or was not handed on.

Screens: Embeddings export, Graph Builder load and Annotate hand-off, Hit
List export and investigate, Methods & Results copy, Profiler redraw, Report
job plumbing.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_embeddings_export_refuses_or_reports_each_failure(
        subtests, qtbot, monkeypatch, tmp_path):
    """No source/folder/spec starts nothing; embed failures and success land in the status."""
    import spacr.embeddings as emb
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    crops = tmp_path / "crops"
    crops.mkdir()

    def _screen(with_source=True):
        screen = EmbeddingsScreen(threaded=False)
        qtbot.addWidget(screen)
        if with_source:
            screen._where.setCurrentIndex(1)
            screen._path.setText(str(crops))
            screen._on_path_changed()
        return screen

    def _no_dialog(*_a, **_k):
        raise AssertionError("a folder dialog was opened")

    with subtests.test(msg="no source asks for one and opens no dialog"), \
            monkeypatch.context() as patch:
        patch.setattr(QFileDialog, "getExistingDirectory",
                      staticmethod(_no_dialog))
        screen = _screen(with_source=False)
        assert screen._embed_and_export(str(tmp_path)) == ""
        assert screen._status.text() == "Choose a database or a folder first."

    with subtests.test(msg="a cancelled folder dialog starts nothing"), \
            monkeypatch.context() as patch:
        asked = []
        patch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
            lambda *a, **k: asked.append(a[1:]) or ""))
        patch.setattr(emb, "embed_store", lambda *a, **k: pytest.fail(
            "an embedding started without a folder"))
        screen = _screen()
        before = screen._status.text()
        assert screen._embed_and_export() == ""
        assert asked == [("Folder for the embeddings",)]
        assert screen._export_store.isEnabled() is True
        assert screen._status.text() == before

    with subtests.test(msg="a chosen folder is where the export goes"), \
            monkeypatch.context() as patch:
        calls = []
        patch.setattr(QFileDialog, "getExistingDirectory",
                      staticmethod(lambda *a, **k: str(tmp_path)))

        def _record(query, run, **kwargs):
            calls.append((query.limit, kwargs["output_dir"]))
            raise RuntimeError("stop")

        patch.setattr(emb, "embed_store", _record)
        screen = _screen()
        assert screen._embed_and_export() == str(tmp_path)
        assert calls == [(0, str(tmp_path))]

    with subtests.test(msg="a spec that does not build is reported"), \
            monkeypatch.context() as patch:
        screen = _screen()

        def bad():
            raise ValueError("the spec is wrong")

        patch.setattr(screen, "spec", bad)
        patch.setattr(screen._foundation, "currentData", lambda *a: "other")
        assert screen._embed_and_export(str(tmp_path)) == ""
        assert screen._status.text() == "the spec is wrong"
        assert screen._export_store.isEnabled() is True

    with subtests.test(msg="an embedding failure is reported and Export re-enabled"), \
            monkeypatch.context() as patch:
        def boom(*a, **k):
            raise RuntimeError("no encoder")

        patch.setattr(emb, "embed_store", boom)
        screen = _screen()
        assert screen._embed_and_export(str(tmp_path)) == str(tmp_path)
        assert screen._status.text() == "RuntimeError: no encoder"
        assert screen._export_store.isEnabled() is True

    with subtests.test(msg="the automatic DINO choice embeds with the DINO note"), \
            monkeypatch.context() as patch:
        screen = _screen()
        seen = []
        patch.setattr(screen._foundation, "currentData",
                      lambda *a: "dino_auto")
        patch.setattr(screen, "_dino_note", lambda: ("DINO-SPEC", ""))
        patch.setattr(screen, "spec", lambda: pytest.fail(
            "the spec was built for the automatic DINO choice"))

        def record(query, run, **k):
            seen.append(run)
            raise RuntimeError("stop")

        patch.setattr(emb, "embed_store", record)
        assert screen._embed_and_export(str(tmp_path)) == str(tmp_path)
        assert seen == ["DINO-SPEC"]

    with subtests.test(msg="a finished export says how much it wrote"), \
            monkeypatch.context() as patch:
        patch.setattr(emb, "embed_store", lambda *a, **k: {
            "record": {"n_cells": 12, "n_wells": 3}})
        screen = _screen()
        assert screen._embed_and_export(str(tmp_path)) == str(tmp_path)
        assert screen._status.text() == (
            f"Wrote 12 cell embeddings and 3 wells to {tmp_path}.")
        assert screen._export_store.isEnabled() is True


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

@pytest.fixture
def measurements():
    """Four objects across two plates, with one measured column."""
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "rowID": ["r1", "r2", "r1", "r2"],
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "object_label": [1, 2, 3, 4],
        "area": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def graph_screen(qtbot):
    """A Graph Builder on a PRIVATE link, reading inline."""
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.graph_builder import GraphBuilderScreen

    widget = GraphBuilderScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_graph_builder_load_reports_on_the_source_line(
        subtests, graph_screen, measurements, tmp_path):
    """A file that will not list says so inline; picking a table reloads the same file."""
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.to_sql("zebra", db, index=False)

    with subtests.test(msg="a file whose tables cannot be listed"):
        broken = tmp_path / "not-a-database.db"
        broken.write_bytes(b"this is not sqlite")
        graph_screen.load_path(str(broken))
        assert graph_screen._source.text() == (
            "could not read not-a-database.db: sqlite3.DatabaseError: file is not "
            "a database")
        assert graph_screen._frame is None
        assert graph_screen._table_picker.count() == 0

    with subtests.test(msg="a database loads its preferred table"):
        graph_screen.load_path(str(database))
        assert graph_screen._source.text() == (
            "measurements.db · cell · 4 rows × 6 columns")
        assert [graph_screen._table_picker.itemText(i) for i in range(
            graph_screen._table_picker.count())] == ["cell", "zebra"]
        assert graph_screen._table_picker.currentText() == "cell"

    with subtests.test(msg="picking another table reloads the same file"):
        graph_screen._on_table_picked("zebra")
        assert graph_screen._source.text() == (
            "measurements.db · zebra · 4 rows × 6 columns")
        assert graph_screen._table_picker.currentText() == "zebra"

    with subtests.test(msg="a blank pick does nothing"):
        graph_screen._on_table_picked("")
        assert graph_screen._source.text() == (
            "measurements.db · zebra · 4 rows × 6 columns")


def test_graph_builder_open_selection_routes_a_brush_or_says_why_not(
        subtests, graph_screen, measurements, monkeypatch):
    """No brush asks for one; no or failing opener is reported; a brush reaches the opener."""
    from spacr.selection import Selection

    graph_screen.set_frame(measurements)
    link = graph_screen.builder.canvas.link
    ask = "Brush a region first — nothing is selected."

    def _brush(n):
        link.set_selection(Selection(keys=measurements.index[:n],
                                     source="graph_builder"))

    with subtests.test(msg="nothing brushed asks for a brush"):
        graph_screen._open_selection()
        assert graph_screen._source.text() == ask

    with subtests.test(msg="a brush around blank space is nothing to open"):
        _brush(0)
        graph_screen._open_selection()
        assert graph_screen._source.text() == ask

    with subtests.test(msg="with nothing able to show crops the screen says so"):
        _brush(2)
        graph_screen._open_selection()
        text = graph_screen._source.text()
        assert text.startswith("could not open those objects: ")
        assert "open the Annotate screen once" not in text

    with subtests.test(msg="a brushed selection reaches the registered opener"), \
            monkeypatch.context() as patch:
        _brush(2)
        patch.setattr("spacr.qt.linked_selection.has_object_opener",
                      lambda kind: True)
        opened = []
        link.register_object_opener("annotate", opened.append)
        before = graph_screen._source.text()
        graph_screen._open_selection()
        [request] = opened
        assert [str(key) for key in request.keys] == ["0", "1"]
        assert request.reason.startswith("brushed in the Graph Builder · ")
        assert graph_screen._source.text() == before

    with subtests.test(msg="an opener that raises is reported, not propagated"), \
            monkeypatch.context() as patch:
        _brush(2)
        patch.setattr("spacr.qt.linked_selection.has_object_opener",
                      lambda kind: True)

        def _explode(request):
            raise RuntimeError("the crops are on a disconnected share")

        link.register_object_opener("annotate", _explode)
        graph_screen._open_selection()
        assert graph_screen._source.text() == (
            "could not open those objects: the crops are on a disconnected "
            "share")


# ---------------------------------------------------------------------------
# Hit List
# ---------------------------------------------------------------------------

@pytest.fixture
def hit_folder(tmp_path):
    """A results folder laid out the way ``perform_regression`` writes one."""
    genes = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in ("100", "200", "300")],
        "coefficient": [2.4, -1.8, 0.9],
        "std_err": [0.30, 0.25, 0.40],
        "p_value": [1e-6, 4e-5, 0.02],
        "condition": ["other", "other", "other"],
        "n_gene": [48, 44, 30],
    })
    rows = [("100_1", 2.2), ("100_2", 2.9), ("100_3", 1.7),
            ("200_1", -1.9), ("200_2", 1.4), ("300_1", 0.9)]
    grnas = pd.DataFrame({
        "feature": [f"fraction:grna[{g}]" for g, _ in rows],
        "grna": [g for g, _ in rows],
        "coefficient": [c for _, c in rows]})
    root = tmp_path / "results" / "pred" / "ols"
    root.mkdir(parents=True)
    genes.to_csv(root / "results_gene.csv", index=False)
    grnas.to_csv(root / "results_grna.csv", index=False)
    pd.concat([genes, grnas], ignore_index=True).to_csv(
        root / "results.csv", index=False)
    return str(root)


def _hit_screen(qtbot, folder=None):
    from spacr.qt.screens.hit_list import HitListScreen

    kwargs = {"folder": folder, "regression_type": "ols"} if folder else {}
    widget = HitListScreen(threaded=False, **kwargs)
    qtbot.addWidget(widget)
    return widget


def _save_dialog(path, asked):
    class _Dialog:
        @staticmethod
        def getSaveFileName(*args, **_kwargs):     # noqa: N802 - Qt spelling
            asked.append(args[1:])
            return str(path), ""

    return _Dialog


def test_hit_list_export_asks_where_and_writes_the_filtered_rows(
        subtests, qtbot, hit_folder, tmp_path, monkeypatch):
    """No list refuses without a dialog; a cancel writes nothing; a path gets the shown rows."""
    from spacr.qt.screens import hit_list as screen_module

    with subtests.test(msg="before a list exists it says so and opens no dialog"), \
            monkeypatch.context() as patch:
        asked = []
        patch.setattr(screen_module, "QFileDialog", _save_dialog("", asked))
        empty = _hit_screen(qtbot)
        empty._ask_and_export("csv", "Export hit list", "CSV (*.csv)")
        assert asked == []
        assert empty._summary.text() == "There is no hit list to export yet."
        assert empty._summary.property("problem") == "true"

    loaded = _hit_screen(qtbot, hit_folder)

    with subtests.test(msg="a cancelled dialog writes nothing"), \
            monkeypatch.context() as patch:
        asked = []
        patch.setattr(screen_module, "QFileDialog", _save_dialog("", asked))
        before = loaded._summary.text()
        loaded._ask_and_export("csv", "Export hit list", "CSV (*.csv)")
        assert asked == [("Export hit list", "", "CSV (*.csv)")]
        assert list(tmp_path.glob("*.csv")) == []
        assert loaded._summary.text() == before

    with subtests.test(msg="a chosen path gets the filtered rows"), \
            monkeypatch.context() as patch:
        target = tmp_path / "hits.csv"
        asked = []
        patch.setattr(screen_module, "QFileDialog", _save_dialog(target, asked))
        loaded._direction.setCurrentText("up")
        shown = len(loaded._shown)
        assert 0 < shown < len(loaded._all), "the filter has to bite"
        loaded._ask_and_export("csv", "Export hit list", "CSV (*.csv)")
        assert asked == [("Export hit list", "", "CSV (*.csv)")]
        assert len(pd.read_csv(target)) == shown
        assert "written to hits.csv" in loaded._summary.text()


def test_hit_list_investigate_emits_only_a_live_selected_row(
        subtests, qtbot, hit_folder):
    """No selection or a filtered-away gene is refused by name; a live row emits its evidence."""
    loaded = _hit_screen(qtbot, hit_folder)
    emitted = []
    loaded.investigate_requested.connect(emitted.append)
    item = loaded._table.topLevelItem(0)
    gene = str(item.data(0, Qt.UserRole))

    with subtests.test(msg="nothing selected asks for a selection"):
        loaded._table.setCurrentItem(None)
        loaded._on_investigate_selected()
        assert loaded._summary.text() == "Select one hit to investigate."
        assert loaded._summary.property("problem") == "true"
        assert emitted == []

    with subtests.test(msg="a live row emits its evidence"):
        loaded._table.setCurrentItem(item)
        loaded._on_investigate_selected()
        hit = loaded._shown.gene(gene)
        assert emitted == [{
            "folder": loaded._shown.source,
            "gene": gene,
            "effect": hit.effect,
            "guides": hit.agreeing_guides,
            "guide_agreement": hit.agreement,
            "n_guides": hit.n_guides,
            "well_support": hit.n_obs,
            "fdr": hit.q_value,
            "phenotype": "",
        }]
        assert emitted[0]["folder"] == hit_folder
        emitted.clear()

    with subtests.test(msg="a row the filters have since dropped is named"):
        loaded._table.setCurrentItem(item)
        loaded._shown = loaded._all.filter(query="no-such-gene-at-all")
        assert loaded._shown.gene(gene) is None
        loaded._on_investigate_selected()
        assert loaded._summary.text() == (
            f"The selected gene {gene!r} is no longer in the filtered list.")
        assert loaded._summary.property("problem") == "true"
        assert emitted == []

    with subtests.test(msg="a screen with no list asks for a selection"):
        empty = _hit_screen(qtbot)
        empty._on_investigate_selected()
        assert empty._summary.text() == "Select one hit to investigate."


# ---------------------------------------------------------------------------
# Methods & Results
# ---------------------------------------------------------------------------

def test_methods_export_copy_and_provenance_strip_say_what_happened(
        subtests, qtbot, monkeypatch):
    """Copy refuses empty panes, stays silent with no clipboard; the strip reports without a style."""
    from spacr import artifacts
    from spacr.qt.screens.methods_export import MethodsExportScreen

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)
    screen = MethodsExportScreen(threaded=False)
    qtbot.addWidget(screen)

    with subtests.test(msg="nothing to copy is refused"):
        assert screen.text() == ""
        screen._on_copy()
        assert screen._provenance.text() == "There is nothing to copy yet."
        assert screen._provenance.property("problem") == "true"

    screen._methods_view.setPlainText("## Methods\nWe segmented 96 fields.")

    with subtests.test(msg="no clipboard means no claim of a copy"), \
            monkeypatch.context() as patch:
        screen._set_provenance("Digest built.", problem=False)
        patch.setattr(QApplication, "clipboard", staticmethod(lambda: None))
        screen._on_copy()
        assert screen._provenance.text() == "Digest built."

    with subtests.test(msg="a clipboard gets both sections"):
        screen._on_copy()
        assert screen._provenance.text() == "Both sections copied."
        assert screen._provenance.property("problem") == "false"
        assert QApplication.clipboard().text() == screen.text()
        assert "We segmented 96 fields." in screen.text()

    with subtests.test(msg="a failure lands on a strip with no style"), \
            monkeypatch.context() as patch:
        patch.setattr(type(screen._provenance), "style", lambda self: None,
                      raising=False)
        screen._on_job_failed("the run folder holds no settings journal")
        assert screen.last_error == "the run folder holds no settings journal"
        assert screen._provenance.text() == (
            "Could not read the run: the run folder holds no settings journal")
        assert screen._provenance.property("problem") == "true"

    with subtests.test(msg="the next clean message clears the flag"):
        screen._set_provenance("Digest built.", problem=False)
        assert screen._provenance.text() == "Digest built."
        assert screen._provenance.property("problem") == "false"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

def _profiler_model():
    from spacr.profiler import FittedLinear

    return FittedLinear(params=pd.Series({
        "Intercept": 0.5,
        "fraction:grna[100_1]": 2.4,
        "fraction:grna[200_1]": -0.8,
    }))


def _profiler_design():
    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame({
        "Intercept": np.ones(n),
        "fraction:grna[100_1]": rng.uniform(0.0, 0.4, n),
        "fraction:grna[200_1]": rng.uniform(0.0, 1.0, n),
    })


def test_profiler_link_change_and_refused_sweep(subtests, qtbot, monkeypatch):
    """A link change re-reads a file or redraws a model; a refused sweep clears curve and canvas."""
    from spacr.qt.screens import profiler as screen_module

    def _screen(**kwargs):
        widget = screen_module.ProfilerScreen(threaded=False, **kwargs)
        qtbot.addWidget(widget)
        return widget

    with subtests.test(msg="no model and no file does nothing"):
        blank = _screen()
        emitted = []
        blank.profiled.connect(emitted.append)
        blank._on_link_changed()
        assert (blank.curve(), emitted) == (None, [])

    with subtests.test(msg="a model with no file is redrawn"):
        widget = _screen(model=_profiler_model(), design=_profiler_design())
        assert widget._path_edit.text().strip() == ""
        emitted = []
        widget.profiled.connect(emitted.append)
        widget._on_link_changed()
        assert len(emitted) == 1
        assert widget.curve() is not None
        assert widget.curve().variable == widget.variable()
        assert widget._canvas.curve() is widget.curve()

    with subtests.test(msg="a file path is re-read rather than redrawn"), \
            monkeypatch.context() as patch:
        widget = _screen(model=_profiler_model(), design=_profiler_design())
        loads = []
        patch.setattr(widget, "load_coefficients", loads.append)
        patch.setattr(widget, "_redraw", lambda: pytest.fail("redrawn"))
        widget._path_edit.setText("  /runs/results.csv ")
        widget._on_link_changed()
        assert loads == ["  /runs/results.csv "]

    with subtests.test(msg="a refused sweep is reported and clears the canvas"), \
            monkeypatch.context() as patch:
        widget = _screen(model=_profiler_model(), design=_profiler_design())
        assert widget.curve() is not None
        variable = widget.variable()
        message = "held value for 'Intercept' is not a number"

        def refuse(*args, **kwargs):
            raise ValueError(message)

        patch.setattr(screen_module, "profile", refuse)
        widget._redraw()
        assert widget.last_error == message
        assert widget.curve() is None
        assert widget._canvas.curve() is None
        assert widget._canvas._message == message
        assert widget._status.text() == f"Could not profile {variable}: {message}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_jobs_deliver_results_and_report_failures_inline(
        subtests, qtbot, monkeypatch):
    """The worker fills the box; a dead worker's last line, or a raising handler, lands in the status."""
    from spacr.qt.screens import report as module
    from spacr.qt.screens.report import ReportScreen

    def _screen(threaded):
        widget = ReportScreen(threaded=threaded)
        qtbot.addWidget(widget)
        return widget

    with subtests.test(msg="a dead worker reports its last line"):
        screen = _screen(False)
        screen._busy = True
        screen._on_worker_error_text(
            "Traceback (most recent call last):\n  ...\nValueError: no figures")
        assert screen.last_error == "Report failed: ValueError: no figures"
        assert screen.status_text() == "Report failed: ValueError: no figures"
        assert screen.is_busy() is False

    with subtests.test(msg="a worker that said nothing is an unknown error"):
        screen = _screen(False)
        screen._on_worker_error_text("")
        assert screen.last_error == "Report failed: unknown error"

    with subtests.test(msg="a request while busy is refused"):
        screen = _screen(False)
        screen._busy = True
        assert screen._run_job(lambda: pytest.fail("ran"), print) is False
        assert screen.last_error == "Still working on the previous request."

    with subtests.test(msg="an inline handler that raises reports its message"):
        screen = _screen(False)
        with qtbot.waitSignal(screen.job_finished, timeout=5000) as caught:
            assert screen._run_job(lambda: 1, lambda _r: (_ for _ in ()).throw(
                KeyError())) is False
        assert caught.args == [False]
        assert screen.last_error == "KeyError"

    with subtests.test(msg="the threaded body leaves its result in the box"), \
            monkeypatch.context() as patch:
        real_make_thread = module.make_thread
        captured = {}

        def _run_the_body_here(fn, settings, *args, **kwargs):
            captured["payload"] = settings
            fn(settings)
            return real_make_thread(lambda payload: None, settings,
                                    *args, **kwargs)

        patch.setattr(module, "make_thread", _run_the_body_here)
        widget = _screen(True)
        delivered = []
        with qtbot.waitSignal(widget.job_finished, timeout=20000) as caught:
            widget._run_job(lambda: "the collected report", delivered.append)
        assert caught.args == [True]
        assert captured["payload"]["result"] == "the collected report"
        assert delivered == ["the collected report"]
        qtbot.waitUntil(lambda: widget.is_busy() is False, timeout=20000)
        widget.close()

    with subtests.test(msg="a threaded handler that raises lands in the status"):
        widget = _screen(True)

        def _explode(_result):
            raise ValueError("the report could not be rendered")

        with qtbot.waitSignal(widget.job_finished, timeout=20000) as caught:
            widget._run_job(lambda: "anything", _explode)
        assert caught.args == [False]
        assert widget.last_error == "the report could not be rendered"
        assert widget.is_busy() is False
        widget.close()
