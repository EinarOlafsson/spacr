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


# --------------------------------------------------------------------------- #
#  The mouse actually reaches the selection
# --------------------------------------------------------------------------- #


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
