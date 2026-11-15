"""What the Plate Viewer does when the plate cannot be drawn as asked.

A heatmap screen is mostly refusals. The database may hold no tables, the
table may hold no numeric column, the aggregation may reject the grouping,
the shared filter may be unreadable, and the colormap the user's
colour-vision preference names may not exist in this matplotlib. None of
those may leave the old grid on screen pretending to be the new one, and
none of them may open a modal dialog -- a QMessageBox in a headless run
hangs forever, so every one of these has to land in the status line.

The job plumbing is here for the same reason: the worker computes a plate
and hands it to a painter that runs on the GUI thread, and a failure in
either half has to end the job rather than leave the screen busy.
"""

from __future__ import annotations

import sqlite3
import traceback

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

from spacr.qt.screens import plate_view as pv  # noqa: E402
from spacr.qt.screens.plate_view import (  # noqa: E402
    DEFAULT_CMAP,
    PlateGridWidget,
    PlateViewScreen,
)

pytestmark = pytest.mark.qt


N_ROWS, N_COLS = 8, 12


def _rows():
    """A flat 96-well plate: every well the same, so nothing is an edge."""
    return [(f"plate1_r{r}_c{c}", 100.0, "text")
            for r in range(1, N_ROWS + 1)
            for c in range(1, N_COLS + 1)
            for _ in range(4)]


@pytest.mark.parametrize('fields', [None, [], 'not a field mapping'])
def test_live_plate_rejects_nonmapping_ledger_without_overwriting_committed_state(
        qtbot, tmp_path, fields):
    import json

    card = pv._WatchLivePlate()
    qtbot.addWidget(card)
    work = tmp_path / 'spacr_watch'
    work.mkdir()
    (work / 'watch_ledger.json').write_text(json.dumps({'fields': fields}), encoding="utf-8")
    card._source = str(tmp_path)
    card._wells = {'plate1_A1': 2}
    card.refresh()
    assert card._wells == {'plate1_A1': 2}
    assert not card._timer.isActive()


def test_finishing_inactive_live_plate_stops_timer_without_reading_old_source(qtbot, monkeypatch):
    card = pv._WatchLivePlate()
    qtbot.addWidget(card)
    card._timer.start()
    monkeypatch.setattr(card, 'refresh', lambda: pytest.fail('inactive card must not read disk'))
    card.finish()
    assert not card.is_active()
    assert not card._timer.isActive()


@pytest.fixture
def measdb(tmp_path):
    """A run folder whose database also holds a table with no numbers in it."""
    db_path = tmp_path / "plate1" / "measurements" / "measurements.db"
    db_path.parent.mkdir(parents=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, value REAL, note TEXT)")
        con.executemany("INSERT INTO cell VALUES (?, ?, ?)", _rows())
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
    """A synchronous screen — jobs run inline so assertions are immediate."""
    widget = PlateViewScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# the colour lookup table
# ---------------------------------------------------------------------------

def test_an_unknown_colormap_still_paints_a_readable_ramp():
    """A plate in grey beats a screen that will not paint at all."""
    lut = pv._cmap_lut("not_a_real_colormap", size=5)

    assert len(lut) == 5
    assert lut[0] == QColor(0, 0, 0)
    assert lut[-1] == QColor(255, 255, 255)
    assert all(c.red() == c.green() == c.blue() for c in lut)


def test_a_named_colormap_is_not_grey():
    """The fallback above is a fallback, not what a normal plate looks like."""
    lut = pv._cmap_lut(DEFAULT_CMAP, size=5)

    assert any(c.red() != c.blue() for c in lut)


def test_changing_the_colormap_rebuilds_the_lookup_table(qtbot):
    """The paint loop reads the table, so the table has to follow the name."""
    grid = PlateGridWidget()
    qtbot.addWidget(grid)
    before = list(grid._lut)

    grid.set_plate(None, 0.0, 1.0, "magma", n_rows=N_ROWS, n_cols=N_COLS)

    assert grid._cmap_name == "magma"
    assert list(grid._lut) != before


def test_an_empty_grid_has_no_cell_size_and_no_selection(qtbot):
    """Nothing to divide by: geometry must answer zero, not raise."""
    grid = PlateGridWidget()
    qtbot.addWidget(grid)

    assert grid.has_plate() is False
    assert grid._cell_size() == 0.0

    grid.set_plate(None, 0.0, 1.0, DEFAULT_CMAP, n_rows=N_ROWS, n_cols=N_COLS)
    grid.select(2, 3)
    assert grid.selected_well() == (2, 3)

    grid.select(None)
    assert grid.selected_well() is None


# ---------------------------------------------------------------------------
# opening
# ---------------------------------------------------------------------------


def test_a_table_change_with_no_table_selected_does_nothing(screen, empty_db):
    """The combo fires while it is being rebuilt; an empty name is not a table."""
    screen.open_database(empty_db)

    assert screen._on_table_changed() is None
    assert screen.current_value_column() == ""


# ---------------------------------------------------------------------------
# rendering failures
# ---------------------------------------------------------------------------


def test_a_colour_preference_that_cannot_be_read_falls_back_to_the_default(
        screen, monkeypatch):
    """A preference store that is unavailable costs a palette, not a plate."""
    import spacr.qt.preferences as preferences

    def boom():
        raise RuntimeError("the preference store is unavailable")

    monkeypatch.setattr(preferences, "color_blind_continuous_cmap", boom)

    assert screen._colormap_name() == DEFAULT_CMAP


# ---------------------------------------------------------------------------
# job plumbing and shutdown
# ---------------------------------------------------------------------------

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

    def isRunning(self):
        return False


@pytest.fixture
def inline_jobs(monkeypatch):
    """Run every job body inline, signalling the way the real thread does."""
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


def test_the_job_body_carries_its_result_back_to_the_painter(qtbot,
                                                             inline_jobs):
    """What the worker computed is what the GUI-thread handler is given."""
    screen = PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    delivered = []

    with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
        started = screen._run_job(lambda: {"wells": 96}, delivered.append)

    assert started is True
    assert blocker.args[0] is True
    assert delivered == [{"wells": 96}]
    assert screen.is_busy() is False


def test_closing_survives_a_link_whose_other_half_is_already_gone(
        screen, measdb, monkeypatch):
    """At interpreter teardown the shared link can go first."""
    screen.open_database(measdb)

    def gone():
        raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(screen, "unlink_selection", gone)

    screen.close()

    assert screen.isVisible() is False
