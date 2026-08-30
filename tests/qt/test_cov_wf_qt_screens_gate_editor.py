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
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap                                # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from spacr.qt.screens.gate_editor import GateEditorScreen        # noqa: E402
from spacr.qt.widgets.table_chip import TableChip                # noqa: E402

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


def _wide(n=60, seed=0):
    """Enough numeric columns for a projection to have something to do."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"m{i}": rng.normal(i, 1.0, n) for i in range(6)})
    frame["plateID"] = "p1"
    return frame


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


@pytest.fixture
def two_databases(tmp_path):
    """Two plates in two files, so the merge is not refused as a collision."""
    return [_database(tmp_path / "runA.db", "plate1"),
            _database(tmp_path / "runB.db", "plate2")]


# ---------------------------------------------------------------------------
# The table picker after a merge
# ---------------------------------------------------------------------------

def test_a_table_the_first_database_does_not_list_is_still_merged(
        screen, two_databases, monkeypatch):
    """The picker lists what the FIRST file holds; the merge reads what it was
    asked for, and the two can disagree.

    A session restored against a table the first database has since lost --
    renamed, rebuilt without it -- would otherwise put a name into a picker
    that has no such row. The gates must still be drawn on the table that was
    actually read, so what matters is that ``_table`` follows the merge and
    not the picker: gate on 'cell', export to 'cell'.
    """
    import spacr.qt.screens.gate_editor as module
    from spacr.multi_database import SOURCE_COLUMN

    # The ordinary case first: the chosen table IS one the file lists.
    screen.load_paths(two_databases, table="cell")
    assert screen._table_picker.currentText() == "cell"

    monkeypatch.setattr(module, "table_names", lambda _path: ["nucleus"])
    screen.load_paths(two_databases, table="cell")

    picker = screen._table_picker
    assert [picker.itemText(i) for i in range(picker.count())] == ["nucleus"]
    assert picker.currentText() == "nucleus", (
        "a name the picker never listed was forced into it")
    assert screen._table == "cell"
    assert _chip_names(screen._chips) == ["cell"]
    assert sorted(screen._frame[SOURCE_COLUMN].unique()) == ["runA", "runB"]
    assert len(screen._frame) == 16


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
# A projection with only one component
# ---------------------------------------------------------------------------

def test_a_projection_of_one_component_leaves_the_axis_pickers_alone(
        screen, monkeypatch):
    """Two axes need two components; one component may not half-fill them.

    The pickers are a pair -- x AND y -- and a reduction that came back with a
    single column (a method that clamped itself, a backend that returned less
    than it was asked for) has nothing to put on y. Moving x alone would plot
    the new component against the old measurement and label it a projection,
    which is a picture of two unrelated things.
    """
    import spacr.merge_tables as merge

    screen.set_frame(_wide())

    # Two components: both pickers follow, which is what makes the guard
    # below a guard rather than dead weight.
    assert screen.reduce_to_components() is None
    assert (screen._x.currentText(), screen._y.currentText()) == ("PC1", "PC2")

    screen.set_frame(_wide(seed=1))
    screen._x.setCurrentText("m0")
    screen._y.setCurrentText("m1")

    def _one_component(frame, columns, **_kwargs):
        out = pd.DataFrame({"PC1": np.arange(len(frame), dtype=float)},
                           index=frame.index)
        out.attrs["explained_variance"] = [1.0]
        return out

    monkeypatch.setattr(merge, "reduce_dimensions", _one_component)

    assert screen.reduce_to_components() is None
    assert "PC1" in screen._frame.columns
    assert (screen._x.currentText(), screen._y.currentText()) == ("m0", "m1")
    assert screen._z.currentText() != "PC1"
    assert "projected onto PC1 100%" in screen._source.text()


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

    def _render(_figure, path):
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

    def _render(_figure, path):
        open(path, "wb").close()
        return True

    monkeypatch.setattr(queue, "render_figure_to_png", _render)
    assert drawn.save_graph(good) == good

    def _explode(_figure, _path):
        raise RuntimeError("the PDF backend went away")

    monkeypatch.setattr(queue, "render_figure_to_png", _explode)
    doomed = tmp_path / "doomed.png"

    assert drawn.save_graph(str(doomed)) == ""
    assert not doomed.exists()
    assert "Could not save the graph: the PDF backend went away" in \
        drawn.console.log.toPlainText()
