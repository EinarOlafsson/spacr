"""Tabulate's panel: the axis wells, and what the grid does when it cannot draw.

The wells are driven with real Qt drag payloads and real key presses, and the
grid is asked for the text it actually painted -- a table that renders nothing
would otherwise pass a test that only counted rows.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (  # noqa: E402
    QEvent, QMimeData, QPoint, QPointF, QTimer, Qt,
)
from PySide6.QtGui import (  # noqa: E402
    QDragEnterEvent, QDragMoveEvent, QDropEvent, QKeyEvent,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.graph_builder import COLUMN_MIME  # noqa: E402
from spacr.qt.widgets import pivot_builder as pb  # noqa: E402
from spacr.qt.widgets.pivot_spec import (  # noqa: E402
    MEAN, N, SD, PivotSpec, pivot,
)

pytestmark = pytest.mark.qt


def _frame(rows=6):
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2", "p2"][:rows],
        "rowID": ["r1", "r1", "r2", "r1", "r2", "r2"][:rows],
        "columnID": ["c1", "c2", "c1", "c2", "c1", "c2"][:rows],
        "gene": ["a", "b", "a", "b", "a", "b"][:rows],
        "area": [10.0, 20.0, 30.0, 34.0, 5.0, 7.0][:rows],
    })


@pytest.fixture()
def panel(qtbot):
    widget = pb.PivotPanel()
    qtbot.addWidget(widget)
    widget.set_frame(_frame())
    return widget


def _column_mime(name):
    mime = QMimeData()
    mime.setData(COLUMN_MIME, name.encode("utf-8"))
    return mime


def _drop(well, name):
    """Drop a column payload onto a well the way the shelf does.

    The QMimeData is bound to a name first: a Qt event does not own its
    payload, and a temporary one is freed the moment the expression ends,
    which segfaults the handler that then reads it.
    """
    target = well._list
    payload = _column_mime(name)
    where = QPointF(target.rect().center())
    event = QDropEvent(where, Qt.CopyAction, payload, Qt.LeftButton,
                       Qt.NoModifier)
    target.dropEvent(event)


# ---------------------------------------------------------------------------
# The wells
# ---------------------------------------------------------------------------

def test_an_axis_nobody_defines_is_refused(qtbot):
    with pytest.raises(ValueError, match="unknown pivot axis"):
        pb.DropWell("diagonal")


def test_a_drag_of_something_that_is_not_a_column_is_ignored(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    plain = QMimeData()
    plain.setText("plateID")
    where = QPointF(well._list.rect().center())

    enter = QDragEnterEvent(where.toPoint(), Qt.CopyAction, plain,
                            Qt.LeftButton, Qt.NoModifier)
    enter.accept()
    well._list.dragEnterEvent(enter)
    assert not enter.isAccepted()

    move = QDragMoveEvent(where.toPoint(), Qt.CopyAction, plain,
                          Qt.LeftButton, Qt.NoModifier)
    move.accept()
    well._list.dragMoveEvent(move)
    assert not move.isAccepted()

    well._list.dropEvent(QDropEvent(where, Qt.CopyAction, plain,
                                    Qt.LeftButton, Qt.NoModifier))
    assert well.columns() == ()


def test_a_column_drag_is_taken(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    payload = _column_mime("plateID")
    where = QPointF(well._list.rect().center()).toPoint()

    enter = QDragEnterEvent(where, Qt.CopyAction, payload, Qt.LeftButton,
                            Qt.NoModifier)
    enter.ignore()
    well._list.dragEnterEvent(enter)
    assert enter.isAccepted()

    move = QDragMoveEvent(where, Qt.CopyAction, payload, Qt.LeftButton,
                          Qt.NoModifier)
    move.ignore()
    well._list.dragMoveEvent(move)
    assert move.isAccepted()

    _drop(well, "plateID")
    assert well.columns() == ("plateID",)


def test_delete_takes_the_selected_key_off_the_axis(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    seen = []
    well.changed.connect(lambda: seen.append(well.columns()))
    _drop(well, "plateID")
    _drop(well, "rowID")
    well._list.setCurrentRow(0)
    well._list.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete,
                                       Qt.NoModifier))
    assert well.columns() == ("rowID",)
    assert seen[-1] == ("rowID",)


def test_delete_with_nothing_selected_removes_nothing(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    _drop(well, "plateID")
    well._list.setCurrentRow(-1)
    well._list.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Backspace,
                                       Qt.NoModifier))
    assert well.columns() == ("plateID",)


def test_a_key_that_is_not_delete_falls_through(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    _drop(well, "plateID")
    well._list.setCurrentRow(0)
    well._list.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A,
                                       Qt.NoModifier))
    assert well.columns() == ("plateID",)


def test_a_removal_of_a_row_that_is_gone_changes_nothing(qtbot):
    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    _drop(well, "plateID")
    seen = []
    well.changed.connect(seen.append)
    well._on_remove(5)
    well._on_remove(-1)
    assert well.columns() == ("plateID",)
    assert seen == []


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def test_an_empty_grid_has_no_result_and_no_key_columns(qtbot):
    table = pb.PivotTable()
    qtbot.addWidget(table)
    assert table.result is None
    assert table._header_offset_keys() == ()
    assert table.cell_text(0, 0) == ""


def test_a_table_too_large_to_draw_says_so_instead_of_freezing(qtbot,
                                                              monkeypatch):
    """The grid builds the shape, not a million items."""
    monkeypatch.setattr(pb, "MAX_RENDERED_CELLS", 3)
    table = pb.PivotTable()
    qtbot.addWidget(table)
    result = pivot(_frame(), PivotSpec(rows=("plateID", "rowID"),
                                       cols=("gene",), values=("area",),
                                       aggs=(N, MEAN, SD)))
    table.set_result(result)
    assert table.truncated_cells == result.n_cells
    assert table.rowCount() == 1 and table.columnCount() == 1
    assert "narrow the axes" in table.item(0, 0).text()
    assert table.horizontalHeaderItem(0).text() == "too large to draw"


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_a_whole_spec_can_be_pushed_in_at_once(panel):
    computed = []
    panel.computed.connect(computed.append)
    panel.set_spec(PivotSpec(rows=("plateID",), cols=("gene",),
                             values=("area",), aggs=(N, MEAN), quantile=0.9))
    assert panel.wells[pb.AXIS_ROWS].columns() == ("plateID",)
    assert panel.wells[pb.AXIS_COLS].columns() == ("gene",)
    assert panel.wells[pb.AXIS_VALUES].columns() == ("area",)
    assert panel._agg_boxes[MEAN].isChecked()
    assert not panel._agg_boxes[SD].isChecked()
    assert panel._quantile.value() == pytest.approx(0.9)
    # One rebuild for the whole spec, not one per well.
    assert len(computed) == 1


def test_the_well_hierarchy_preset_needs_a_table(qtbot):
    empty = pb.PivotPanel()
    qtbot.addWidget(empty)
    empty.use_well_hierarchy()
    assert empty.wells[pb.AXIS_ROWS].columns() == ()

    empty.set_frame(_frame())
    empty.use_well_hierarchy()
    assert empty.wells[pb.AXIS_ROWS].columns() == (
        "plateID", "rowID", "columnID")


def test_a_panel_with_no_table_asks_for_one(qtbot):
    empty = pb.PivotPanel()
    qtbot.addWidget(empty)
    assert empty.recompute() is None
    assert "Load a table" in empty.notice.text()
    assert empty.long_frame().empty


def test_an_empty_spec_says_what_to_drop(panel):
    assert panel.recompute() is None
    assert "Drop a column onto Rows" in panel.notice.text()


def test_a_refusal_from_the_pivot_becomes_the_notice(panel, monkeypatch):
    from spacr.qt.widgets.pivot_spec import PivotError

    def refuse(_frame, _spec):
        raise PivotError("area is not a measurement in this table")

    monkeypatch.setattr(pb, "pivot", refuse)
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    assert panel.recompute() is None
    assert panel.notice.text() == "area is not a measurement in this table"
    assert panel.result is None


def test_a_truncated_table_says_to_export_it(panel, monkeypatch):
    monkeypatch.setattr(pb, "MAX_RENDERED_CELLS", 2)
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    _drop(panel.wells[pb.AXIS_COLS], "gene")
    _drop(panel.wells[pb.AXIS_VALUES], "area")
    assert panel.recompute() is not None
    assert "export it" in panel.notice.text()


def test_dropping_while_a_spec_is_being_restored_does_not_rebuild(panel):
    panel._building = True
    panel._debounce.stop()
    panel._on_axis_changed()
    assert not panel._debounce.isActive()
    panel._building = False
    panel._on_axis_changed()
    assert panel._debounce.isActive()
    panel._debounce.stop()


def test_plotting_an_empty_table_says_why(panel):
    requested = []
    panel.plot_requested.connect(requested.append)
    panel._on_plot()
    assert requested == []
    assert "Nothing to plot yet" in panel.notice.text()


def test_plotting_a_built_table_carries_the_long_frame(panel):
    requested = []
    panel.plot_requested.connect(requested.append)
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    _drop(panel.wells[pb.AXIS_VALUES], "area")
    panel.recompute()
    panel._on_plot()
    assert len(requested) == 1
    assert not requested[0].empty


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_exporting_before_there_is_a_table_says_so(panel):
    assert panel.export_csv("/tmp/never-written.csv") is None
    assert "Nothing to export" in panel.notice.text()


def test_a_cancelled_save_dialog_writes_nothing(panel):
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    _drop(panel.wells[pb.AXIS_VALUES], "area")
    panel.recompute()

    def dismiss():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            QTimer.singleShot(5, dismiss)
            return
        dialog.reject()

    QTimer.singleShot(0, dismiss)
    assert panel.export_csv() is None


def test_a_file_that_cannot_be_written_becomes_the_notice(panel, tmp_path):
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    _drop(panel.wells[pb.AXIS_VALUES], "area")
    panel.recompute()
    assert panel.export_csv(str(tmp_path / "no such folder" / "t.csv")) is None
    assert "could not write that file" in panel.notice.text()


def test_a_written_table_is_named_in_the_notice(panel, tmp_path):
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    _drop(panel.wells[pb.AXIS_VALUES], "area")
    panel.recompute()
    target = tmp_path / "tabulate.csv"
    assert panel.export_csv(str(target)) == str(target)
    assert target.read_text().strip()
    assert f"wrote {target}" in panel.notice.text()
