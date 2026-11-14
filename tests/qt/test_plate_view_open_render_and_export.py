"""Plate Viewer: opening a database, rendering a plate, exporting it, job errors.

Three doors open a database (file picker, run-folder picker, typed path) and
each is walked with a real choice and a cancel. Rendering is walked over its
refusals -- no table, no numeric column, an aggregation that raises, an
unreadable shared filter -- and every one lands as an exact sentence on the
status line, never a dialog, with the stale grid cleared. The export picker
writes the tidy well grid only where the user pointed. Job errors (a failing
painter, a worker traceback) end the job instead of leaving it busy.
"""
from __future__ import annotations

import sqlite3
import traceback

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr import plate_qc as pqc  # noqa: E402
from spacr.qt.screens import plate_view as pv  # noqa: E402
from spacr.qt.screens.plate_view import PlateViewScreen  # noqa: E402

pytestmark = pytest.mark.qt

DRAWN = ("96 well(s) drawn. No edge effect: the outer ring is within "
         "+0.0 % of the interior (δ = 0).")


@pytest.fixture
def measdb(tmp_path):
    """A flat 96-well ``cell`` table and a text-only ``notes`` table."""
    db_path = tmp_path / "plate1" / "measurements" / "measurements.db"
    db_path.parent.mkdir(parents=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, value REAL, note TEXT)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?)",
            [(f"plate1_r{r}_c{c}", 100.0, "text")
             for r in range(1, 9) for c in range(1, 13) for _ in range(4)])
        con.execute("CREATE TABLE notes (prc TEXT, note TEXT)")
        con.execute("INSERT INTO notes VALUES ('plate1_r1_c1', 'nothing')")
        con.commit()
    finally:
        con.close()
    return str(db_path)


@pytest.fixture
def empty_db(tmp_path):
    """A real sqlite file that holds no tables at all."""
    path = tmp_path / "empty" / "measurements.db"
    path.parent.mkdir(parents=True)
    sqlite3.connect(path).close()
    return str(path)


@pytest.fixture
def screen(qtbot):
    """A synchronous screen: jobs run inline so assertions are immediate."""
    widget = PlateViewScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _tables(screen):
    return [screen._table_combo.itemText(i)
            for i in range(screen._table_combo.count())]


def test_every_door_opens_the_same_database_and_cancel_opens_nothing(
        screen, measdb, empty_db, tmp_path, monkeypatch, subtests):
    """File picker, run folder and typed path all resolve to the database."""
    ready = "cell: 1 numeric column(s). Pick a measurement and press Render."
    with subtests.test(msg="cancelled pickers open nothing"):
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: ""))
        screen._pick_database()
        screen._pick_run_folder()
        assert screen._path_edit.text() == ""
        assert screen.current_table() == ""
        assert screen.status_text() == (
            "Choose a measurements.db, or a run folder containing "
            "measurements/measurements.db.")

    doors = [
        ("the database picker", "getOpenFileName", (measdb, ""),
         screen._pick_database),
        ("the run-folder picker", "getExistingDirectory",
         str(tmp_path / "plate1"), screen._pick_run_folder),
    ]
    for case_id, picker, answer, door in doors:
        with subtests.test(msg=case_id):
            screen.open_database(empty_db)
            monkeypatch.setattr(QFileDialog, picker,
                                staticmethod(lambda *a, _r=answer, **k: _r))
            door()
            assert screen._path_edit.text() == measdb
            assert _tables(screen) == ["cell", "notes"]
            assert screen.current_table() == "cell"
            assert screen.current_value_column() == "value"
            assert screen.status_text() == ready

    with subtests.test(msg="the typed path"):
        screen.open_database(empty_db)
        screen._path_edit.setText(measdb)
        screen._on_open_typed_path()
        assert screen._path_edit.text() == measdb
        assert screen.current_table() == "cell"
        assert screen.status_text() == ready


def test_render_refuses_in_words_and_never_leaves_a_stale_grid(
        screen, measdb, empty_db, monkeypatch, subtests):
    """Each refusal is the status sentence; a good render draws 96 wells."""
    with subtests.test(msg="a database with no tables draws nothing"):
        assert screen.open_database(empty_db) is True
        assert screen.status_text() == "measurements.db has no tables."
        assert screen.current_table() == ""
        assert screen.render_plate() is False
        assert screen.status_text() == "Pick a table first."
        assert screen._grid.has_plate() is False

    screen.open_database(measdb)
    with subtests.test(msg="a text-only table has no measurement"):
        screen._table_combo.setCurrentText("notes")
        assert screen.render_plate() is False
        assert screen.status_text() == (
            "Pick a measurement column first — this table exposed no "
            "numeric columns.")

    with subtests.test(msg="a good render draws the plate"):
        screen._table_combo.setCurrentText("cell")
        assert screen.render_plate() is True
        assert screen.status_text() == DRAWN
        assert len(screen._layout_df) == 96
        assert screen._grid.has_plate() is True

    with subtests.test(msg="an unreadable shared filter is noted"):
        class _BrokenLink:
            @property
            def filter(self):
                raise RuntimeError("the shared link is gone")

        real_link = screen._link
        screen._link = _BrokenLink()
        try:
            assert screen.render_plate() is True
        finally:
            screen._link = real_link
        assert screen.status_text() == DRAWN + " · filter ignored (RuntimeError)"
        assert len(screen._layout_df) == 96

    with subtests.test(msg="an aggregation that refuses clears the plate"):
        def refuses(*_args, **_kwargs):
            raise ValueError("no such column: plate")

        monkeypatch.setattr(pqc, "plate_layout", refuses)
        assert screen.render_plate() is True
        assert screen._layout_df is None
        assert screen.status_text() == "no such column: plate"
        assert screen._grid.has_plate() is False


def test_the_export_picker_writes_the_well_grid_only_where_pointed(
        screen, measdb, tmp_path, monkeypatch, subtests):
    """A chosen path gets the tidy grid; a cancel writes no file."""
    screen.open_database(measdb)
    screen.render_plate()
    with subtests.test(msg="a cancelled save writes nothing"):
        before = sorted(p.name for p in tmp_path.iterdir())
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        screen._pick_export_path()
        assert sorted(p.name for p in tmp_path.iterdir()) == before
        assert screen.status_text() == DRAWN

    with subtests.test(msg="the chosen path gets one row per well"):
        out = tmp_path / "wells.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        screen._pick_export_path()
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ("plateID,well,rowID,columnID,row_index,"
                            "column_index,n,value,ring,is_edge")
        assert lines[1] == "plate1,A01,r1,c1,1,1,4,100.0,0,True"
        assert len(lines) == 97
        assert screen.status_text() == f"Exported 96 well(s) → {out}"


class _InlineWorker(QObject):
    error = Signal(str)
    finished = Signal(bool)


class _InlineThread(QObject):
    finished = Signal()

    def __init__(self, run):
        super().__init__()
        self._run = run

    def start(self):
        self._run()

    def isRunning(self):  # noqa: N802 - Qt naming
        return False


@pytest.fixture
def inline_jobs(monkeypatch):
    """Run every threaded job body inline, signalling like the real thread."""
    def fake_make_thread(fn, settings, *_args, **_kwargs):
        worker = _InlineWorker()

        def run():
            ok = True
            try:
                fn(settings)
            except Exception:                       # noqa: BLE001
                ok = False
                worker.error.emit(traceback.format_exc())
            worker.finished.emit(ok)
            thread.finished.emit()

        thread = _InlineThread(run)
        return thread, worker

    monkeypatch.setattr(pv, "make_thread", fake_make_thread)


def test_a_failed_job_ends_with_its_last_line_and_frees_the_screen(
        qtbot, inline_jobs, subtests):
    """A failing painter or a worker traceback ends the job, not busy."""
    screen = PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)

    with subtests.test(msg="a painter that raises fails the job"):
        def explodes(_result):
            raise RuntimeError("could not paint the grid")

        with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
            assert screen._run_job(lambda: {"wells": 96}, explodes) is True
        assert blocker.args == [False]
        assert screen.status_text() == "could not paint the grid"
        assert screen.is_busy() is False

    with subtests.test(msg="a body that raises reports its last line"):
        def body():
            raise ValueError("the query was refused")

        with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
            screen._run_job(body, lambda _r: pytest.fail("painted a failure"))
        assert blocker.args == [False]
        assert screen.status_text() == (
            "Plate view failed: ValueError: the query was refused")
        assert screen.is_busy() is False

    cases = [
        ("a traceback is reduced to its last line",
         "Traceback (most recent call last):\n"
         "  File \"x.py\", line 1, in <module>\n"
         "ValueError: the query was refused\n",
         "Plate view failed: ValueError: the query was refused"),
        ("no text at all is an unknown error", "",
         "Plate view failed: unknown error"),
    ]
    for case_id, text, said in cases:
        with subtests.test(msg=case_id):
            screen._busy = True
            screen._on_worker_error_text(text)
            assert screen.status_text() == said
            assert screen.is_busy() is False
