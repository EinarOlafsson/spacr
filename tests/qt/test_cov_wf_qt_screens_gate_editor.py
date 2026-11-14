"""The Gate Editor's edges: the cases where one half of a pair is missing.

Every branch here is the screen coping with a half-state that a real session
does reach -- a table the first database does not list, a settings window the
user closed, a canvas that has not drawn, a save dialog that was answered
rather than cancelled -- and in each of them the failure mode is silent: the
frame loads but the picker names another table, the scale is applied but a
stale window still shows the old one, the cutoff is recorded but never
redrawn, the clipboard keeps yesterday's picture. So each test drives BOTH
sides of the pair in one place: the state that produces the effect, and the
state that must not.

Runs unthreaded, offscreen, with every modal answered by a stub -- a real
``exec`` here has nobody to dismiss it and hangs the run.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.screens.gate_editor import GateEditorScreen  # noqa: E402
from spacr.qt.widgets.table_chip import TableChip  # noqa: E402

pytestmark = pytest.mark.qt


def _objects(plate="p1", n=8):
    """One object table with the identity spaCR's measurement tables carry."""
    return pd.DataFrame({
        "plateID": [plate] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


def _database(path, plate, n=8):
    with sqlite3.connect(str(path)) as db:
        _objects(plate, n).to_sql("cell", db, index=False)
    return str(path)


def _chip_names(layout):
    return [layout.itemAt(i).widget().name for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), TableChip)]


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def drawn(screen):
    """The screen as it is once a table has been read and plotted."""
    screen._table = "cell"
    screen.set_frame(_objects())
    screen._x.setCurrentText("area")
    screen._y.setCurrentText("intensity")
    QApplication.processEvents()
    return screen


# ---------------------------------------------------------------------------
# The settings window and the axis menu, two editors of one value
# ---------------------------------------------------------------------------

def test_a_settings_window_the_user_closed_is_not_reopened_by_the_menu(screen):
    """Choosing a scale from the axis menu must not resurrect a closed window.

    The window and the menu write the same field, and a window that cannot be
    told about the change is rebuilt rather than left showing the old scale.
    Rebuilding one the user had already dismissed would pop a settings dialog
    over the plot every time they touched the axis menu -- so the rebuild is
    owed only to a window that was actually on screen.
    """
    screen.open_settings()
    first = screen._settings_dialog
    assert first.isVisible()

    # Visible: the stale window is replaced by one showing the new scale.
    screen.set_axis_scale("x", "log")
    assert screen._settings.x_scale == "log"
    assert screen._settings_dialog is not None
    assert screen._settings_dialog is not first
    assert screen._settings_dialog.isVisible()

    # Closed: the scale is applied and no window comes back.
    screen._settings_dialog.hide()
    screen.set_axis_scale("x", "linear")

    assert screen._settings.x_scale == "linear"
    assert screen._settings_dialog is None, (
        "a window the user had closed was rebuilt behind their back")


# ---------------------------------------------------------------------------
# Cutoffs on a canvas that cannot draw
# ---------------------------------------------------------------------------

def test_a_cutoff_is_recorded_even_when_nothing_can_redraw_it(drawn,
                                                              monkeypatch):
    """A cutoff has to survive a canvas that is not ready to draw.

    The redraw is a request to the canvas, and the canvas is replaceable --
    the 3D volume view and the plain scatter are different objects, and one
    of them may not answer ``render_now`` yet. If the cutoff were only stored
    as a side effect of drawing, the axis would go back to the full range at
    the next repaint and the user's chosen window would be gone.
    """
    redraws = []
    monkeypatch.setattr(drawn.gates.canvas, "render_now",
                        lambda: redraws.append(True))

    drawn.set_axis_cutoffs("x", 20.0, 60.0)

    assert len(redraws) == 1
    assert drawn._cutoffs.get("area").is_set
    assert "area shows" in drawn.console.log.toPlainText()

    monkeypatch.setattr(drawn.gates.canvas, "render_now", None)

    assert drawn.clear_axis_cutoffs("x") is True
    assert len(redraws) == 1, "a canvas that cannot draw was asked to"
    assert not drawn._cutoffs.get("area").is_set
    assert "area follows the data again." in drawn.console.log.toPlainText()


# ---------------------------------------------------------------------------
# The clipboard
# ---------------------------------------------------------------------------

def test_a_canvas_that_grabbed_nothing_leaves_the_clipboard_alone(drawn,
                                                                  monkeypatch):
    """An empty grab must not overwrite the clipboard with a blank picture.

    ``grab`` answers with a null pixmap when the widget has no surface to
    copy -- a canvas that has not been shown, or a compositor that refused.
    Putting that on the clipboard would replace whatever the user had copied
    with an empty rectangle and then claim, in the console, that the graph
    was copied.
    """
    clipboard = QApplication.clipboard()
    clipboard.clear()

    drawn._copy_graph_to_clipboard()

    assert not clipboard.pixmap().isNull()
    log = drawn.console.log.toPlainText()
    assert log.count("Graph copied to the clipboard.") == 1

    clipboard.clear()
    monkeypatch.setattr(drawn.gates.canvas, "grab",
                        lambda *a, **k: QPixmap())

    drawn._copy_graph_to_clipboard()

    assert clipboard.pixmap().isNull(), "an empty grab was pasted anyway"
    assert drawn.console.log.toPlainText().count(
        "Graph copied to the clipboard.") == 1


# ---------------------------------------------------------------------------
# Saving the picture through the dialog
# ---------------------------------------------------------------------------

def test_the_graph_is_written_to_the_path_the_dialog_returned(drawn, tmp_path,
                                                              monkeypatch):
    """Answering the save dialog has to write the file it names.

    ``save_graph`` is called with no path from the menu and the toolbar, so
    the dialog's answer is the only destination there is. Treating an answered
    dialog like a cancelled one -- or writing to the default name beside it --
    loses the export silently: the console says nothing and no file appears
    where the user pointed.
    """
    import spacr.qt.screens.gate_editor as module
    import spacr.qt.widgets.figure_queue as queue

    picked = {"path": str(tmp_path / "picked.png")}
    written = []

    def _render(_figure, path, **_kwargs):
        written.append(path)
        open(path, "wb").close()
        return True

    monkeypatch.setattr(queue, "render_figure_to_png", _render)
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (picked["path"], "")))

    assert drawn.save_graph() == picked["path"]
    assert written == [picked["path"]]
    assert f"Saved the graph to {picked['path']}" in \
        drawn.console.log.toPlainText()

    picked["path"] = ""
    assert drawn.save_graph() == "", "a cancelled dialog still wrote a file"
    assert written == [str(tmp_path / "picked.png")]


# ---------------------------------------------------------------------------
# Rebuilding the two chip strips
# ---------------------------------------------------------------------------

def test_the_table_chip_strip_is_rebuilt_whatever_is_in_it(screen):
    """A rebuild has to clear the strip, not just the chips it recognises.

    The strip is a layout with a trailing stretch, and anything else that
    lands in front of it -- a spacer, a separator someone adds later -- is not
    a chip and has no widget to delete. Skipping such an item instead of
    dropping it would leave it in place for ever, and every reload would push
    the working set one gap further right until the names ran off the strip.
    """
    screen._tables = ["cell", "nucleus"]
    screen._rebuild_chips()

    assert _chip_names(screen._chips) == ["cell", "nucleus"]
    tidy = screen._chips.count()

    screen._chips.insertSpacing(0, 24)
    assert screen._chips.count() == tidy + 1
    screen._rebuild_chips()

    assert _chip_names(screen._chips) == ["cell", "nucleus"]
    assert screen._chips.count() == tidy, "a spacer survived the rebuild"


def test_the_database_chip_strip_is_rebuilt_whatever_is_in_it(screen,
                                                              tmp_path):
    """Same for the database strip, which keeps its label at index 0.

    The label is deliberately kept -- it names what the chips are -- so the
    sweep starts at index 1, and anything there that is not a chip has no
    widget of its own. Left behind, it would push 'plate1' out from under the
    'Databases' label it belongs to, and the chip the user clicks × on would
    no longer be the database they think they are dropping.
    """
    screen._paths = [str(tmp_path / "plate1.db"), str(tmp_path / "plate2.db")]
    screen._rebuild_database_chips()

    assert _chip_names(screen._db_chips) == ["plate1", "plate2"]
    assert screen._db_chips_label.isVisibleTo(screen)
    tidy = screen._db_chips.count()

    screen._db_chips.insertSpacing(1, 24)
    assert screen._db_chips.count() == tidy + 1
    screen._rebuild_database_chips()

    assert _chip_names(screen._db_chips) == ["plate1", "plate2"]
    assert screen._db_chips.count() == tidy, "a spacer survived the rebuild"


def test_a_renderer_that_raises_is_reported_instead_of_thrown(drawn, tmp_path,
                                                              monkeypatch):
    """A save that blew up must land in the console, not in the event loop.

    ``save_graph`` is wired to a menu item and a toolbar button, so an
    exception out of the renderer -- a font the PDF backend cannot embed, a
    directory that went read-only mid-session -- would leave a Qt slot
    raising and the user with no file and no reason. The path it returns is
    what the caller reports, so it has to be empty exactly when nothing was
    written.
    """
    import spacr.qt.widgets.figure_queue as queue

    good = str(tmp_path / "good.png")

    def _render(_figure, path, **_kwargs):
        open(path, "wb").close()
        return True

    monkeypatch.setattr(queue, "render_figure_to_png", _render)
    assert drawn.save_graph(good) == good

    def _explode(_figure, _path, **_kwargs):
        raise RuntimeError("the PDF backend went away")

    monkeypatch.setattr(queue, "render_figure_to_png", _explode)
    doomed = tmp_path / "doomed.png"

    assert drawn.save_graph(str(doomed)) == ""
    assert not doomed.exists()
    assert "Could not save the graph: the PDF backend went away" in \
        drawn.console.log.toPlainText()


def test_a_table_with_no_loaded_base_keeps_its_own_kinds_for_the_sample(
        screen, monkeypatch):
    from spacr.qt.widgets import data_filter_panel as dfp

    assert screen.formulas.frame() is None
    frame = _objects()
    displayed = frame.head(3).copy()

    screen._classify_from_the_table(frame, displayed)
    monkeypatch.setattr(dfp, "_classify_columns_uncached", lambda f: pytest.fail(
        "the sample was counted again instead of taking its table's kinds"))

    assert dfp.classify_columns(displayed) == dfp.classify_columns(frame)
    assert dfp.classify_columns(displayed)["area"] == "category"


def test_a_cluster_only_edit_skips_the_redraw_and_reaches_an_open_dialog(
        screen, monkeypatch):
    redrawn = []
    monkeypatch.setattr(screen, "apply_settings", redrawn.append)
    screen._settings_dialog = SimpleNamespace(_settings=None)

    screen._on_search_settings({"cluster_eps": 0.37, "cluster_min_samples": 9})

    assert redrawn == []
    assert screen._settings.cluster_eps == pytest.approx(0.37)
    assert screen._settings_dialog._settings is screen._settings


def test_an_edit_of_anything_but_the_clustering_numbers_is_applied_in_full(
        screen, monkeypatch):
    applied = []
    monkeypatch.setattr(screen, "apply_settings", applied.append)

    screen._on_search_settings({"max_points": 7})
    screen._on_search_settings({"max_points": 9, "cluster_eps": 0.4})
    screen._on_search_settings({})

    assert [s.max_points for s in applied] == [7, 9, screen._settings.max_points]
    assert applied[1].cluster_eps == pytest.approx(0.4)


def test_the_count_line_is_written_for_nothing_when_nothing_is_drawn(
        screen, monkeypatch):
    shown = []
    monkeypatch.setattr(screen, "_show_displayed_count", shown.append)
    assert screen._displayed_request() is None

    screen._update_displayed_count()

    assert shown == [None]


def test_a_narrowed_count_for_a_population_no_longer_drawn_is_dropped(
        drawn, monkeypatch):
    shown = []
    monkeypatch.setattr(drawn, "_show_displayed_count", shown.append)
    frame, _axes, key = drawn._displayed_request()
    drawn._displayed_cache = None
    narrowed = frame.head(2)

    drawn._displayed_count_done(frame.copy(), key, narrowed)   # another population
    drawn._displayed_count_done(frame, key + ("stale",), narrowed)  # other axes
    assert shown == []
    assert drawn._displayed_cache is None

    drawn._displayed_count_done(frame, key, narrowed)          # the current one
    assert shown == [narrowed]
    assert drawn._displayed_cache == (frame, key, narrowed)


def test_a_count_arriving_after_the_graph_was_emptied_is_dropped(screen, monkeypatch):
    shown = []
    monkeypatch.setattr(screen, "_show_displayed_count", shown.append)

    screen._displayed_count_done(_objects(), (), _objects().head(1))

    assert shown == []
    assert getattr(screen, "_displayed_cache", None) is None


def test_a_graph_that_cannot_be_pickled_is_left_for_the_caller_to_save(
        screen, tmp_path):
    figure = SimpleNamespace(lock=threading.Lock())      # locks do not pickle

    assert screen._save_graph_in_background(
        figure, tmp_path / "g.png", tmp_path / "g.png") is False

    assert getattr(screen, "_graph_runner", None) is None
    assert screen.cursor().shape() != Qt.CursorShape.BusyCursor


def test_a_background_save_that_failed_says_so_and_frees_the_cursor(
        screen, tmp_path):
    screen.setCursor(Qt.CursorShape.BusyCursor)

    screen._graph_saved(tmp_path / "g.pdf", tmp_path / "g.png", False)

    assert "Could not save the graph." in screen.console.log.toPlainText()
    assert "Saved the graph" not in screen.console.log.toPlainText()
    assert screen.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_a_background_worker_that_raised_is_reported_with_its_reason(
        screen):
    screen.setCursor(Qt.CursorShape.BusyCursor)

    screen._graph_save_failed("the PDF backend went away")

    assert ("Could not save the graph: the PDF backend went away"
            in screen.console.log.toPlainText())
    assert screen.cursor().shape() == Qt.CursorShape.ArrowCursor
