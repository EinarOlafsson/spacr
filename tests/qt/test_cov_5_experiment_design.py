"""The plate map is squares you drag across, and an export that reports.

The design screen turns a table of conditions into a plate map and a CSV that
another table joins to. What matters here is that the mouse reaches the
selection code at all -- the tests around it drive the selection API directly
-- and that a row the user is halfway through typing, an export they
cancelled, and an export that failed each leave the screen saying something
true.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTableWidgetItem

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen

    widget = ExperimentDesignScreen()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()
    return widget


def _wells(screen) -> dict:
    return {(w.row, w.column): w for w in screen._well_labels}


# --------------------------------------------------------------------------- #
#  The mouse actually reaches the selection
# --------------------------------------------------------------------------- #

def test_pressing_a_square_with_the_mouse_selects_that_square(screen):
    """A real press on a well selects it.

    The selection API is driven directly everywhere else, so this is the only
    place that proves a well label finds the screen it belongs to and hands
    the gesture over -- a well in a panel inside a scroll area, several
    parents down.
    """
    well = _wells(screen)[(2, 3)]

    QTest.mousePress(well, Qt.LeftButton, Qt.NoModifier,
                     well.rect().center())
    QTest.mouseRelease(well, Qt.LeftButton, Qt.NoModifier,
                       well.rect().center())

    assert screen.selected_wells() == {(2, 3)}
    assert screen.selected_well_names() == ["B03"]


def test_dragging_across_the_squares_with_the_mouse_selects_the_block(screen):
    """Press, move, release over real widgets selects the rectangle.

    The pressed widget keeps the grab, so every move event during a drag is
    delivered to the well the gesture STARTED on. It has to translate its own
    local point back to a global one and ask which well is under it -- if it
    waited to be entered instead, a drag would select one square.
    """
    wells = _wells(screen)
    start, end = wells[(2, 3)], wells[(4, 5)]
    centre = start.rect().center()
    towards = start.mapFromGlobal(end.mapToGlobal(end.rect().center()))

    QTest.mousePress(start, Qt.LeftButton, Qt.NoModifier, centre)
    QTest.mouseMove(start, towards)
    QApplication.processEvents()
    QTest.mouseRelease(start, Qt.LeftButton, Qt.NoModifier, towards)

    assert screen.selected_wells() == {
        (r, c) for r in (2, 3, 4) for c in (3, 4, 5)}


def test_a_move_with_no_press_behind_it_selects_nothing(screen):
    """A pointer crossing the plate with no button down changes nothing.

    Mouse-move events arrive whenever the pointer is over the widget. Without
    the anchor check, moving across the plate would select wells the user
    never pressed.
    """
    screen.finish_well_drag()                 # make sure no drag is open
    before = screen.selected_wells()
    wells = _wells(screen)

    screen.drag_wells_to(
        wells[(4, 5)].mapToGlobal(wells[(4, 5)].rect().center()))

    assert screen.selected_wells() == before


# --------------------------------------------------------------------------- #
#  The conditions table
# --------------------------------------------------------------------------- #

def test_removing_the_selected_rows_takes_them_off_the_plate(screen):
    """Deleting a selected condition removes it and redraws the map.

    Removing rows back to front is the whole trick: deleting row 0 first
    renumbers everything below it, and the second delete then takes a row the
    user never selected.
    """
    names_before = [c.name for c in screen.design().conditions]
    assert len(names_before) == 3

    from PySide6.QtCore import QItemSelectionModel

    screen._table.selectRow(0)
    screen._table.selectionModel().select(
        screen._table.model().index(2, 0),
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows)
    screen._remove_row()

    remaining = [c.name for c in screen.design().conditions]
    assert remaining == [names_before[1]]


def test_a_condition_with_no_replicates_is_not_a_condition(screen):
    """A row asking for zero or fewer replicates is dropped, not drawn.

    Zero replicates occupies no wells, and a negative count would make the
    layout arithmetic run backwards over the plate. The row stays in the
    table -- the user is mid-edit -- and simply does not reach the map.
    """
    screen._table.setItem(0, 1, QTableWidgetItem("0"))
    screen._table.setItem(1, 1, QTableWidgetItem("-4"))

    names = [c.name for c in screen.design().conditions]

    assert len(names) == 1
    assert screen._table.rowCount() == 3


# --------------------------------------------------------------------------- #
#  Exporting
# --------------------------------------------------------------------------- #

def test_choosing_a_folder_writes_the_plate_map_into_it(
        screen, tmp_path, monkeypatch, qtbot):
    """The export button writes into the folder the dialog returned.

    The plate map is what joins a measurements table to the design, so the
    status line has to name the file and the folder -- a silent success
    leaves the user hunting for it.
    """
    from spacr.qt.screens import experiment_design as ed

    monkeypatch.setattr(ed.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))

    screen._on_export()
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=5000)
    QApplication.processEvents()

    assert (tmp_path / "plate_map.csv").exists()
    assert "plate_map.csv" in screen.status_text()


def test_a_cancelled_export_writes_nothing(screen, tmp_path, monkeypatch):
    """Dismissing the folder dialog starts no job and writes no file.

    An empty path is the user saying no; passing it on would write the plate
    map into the process's working directory.
    """
    from spacr.qt.screens import experiment_design as ed

    monkeypatch.setattr(ed.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    monkeypatch.chdir(tmp_path)

    screen._on_export()

    assert list(tmp_path.iterdir()) == []


def test_an_export_that_wrote_nothing_leaves_the_status_line_alone(screen):
    """A finished job with no paths reports no files.

    "Wrote  to ." is worse than silence: it claims a write and names nothing,
    and the user goes looking for a file that is not there.
    """
    before = screen.status_text()

    screen._on_exported({})
    assert screen.status_text() == before

    screen._on_exported(None)
    assert screen.status_text() == before


def test_a_failed_export_says_so_and_shows_as_an_error(screen):
    """A worker failure reaches the status line with its message.

    The write happens off the GUI thread, so an exception there has no other
    way to be seen; without this the screen looks exactly as it did before
    the button was pressed.
    """
    screen._on_job_failed("permission denied on /mnt/share")

    text = screen.status_text()
    assert "Export failed" in text
    assert "permission denied on /mnt/share" in text
