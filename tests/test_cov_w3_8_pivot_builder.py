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
    QMimeData, QPointF, Qt,
)
from PySide6.QtGui import (  # noqa: E402
    QDropEvent,
)

from spacr.qt.widgets.graph_builder import COLUMN_MIME  # noqa: E402
from spacr.qt.widgets import pivot_builder as pb  # noqa: E402
from spacr.qt.widgets.pivot_spec import (  # noqa: E402
    MEAN, N, SD, PivotSpec,
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


def test_a_refusal_from_the_pivot_becomes_the_notice(panel, monkeypatch):
    from spacr.qt.widgets.pivot_spec import PivotError

    def refuse(_frame, _spec):
        raise PivotError("area is not a measurement in this table")

    monkeypatch.setattr(pb, "pivot", refuse)
    _drop(panel.wells[pb.AXIS_ROWS], "plateID")
    assert panel.recompute() is None
    assert panel.notice.text() == "area is not a measurement in this table"
    assert panel.result is None


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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_exporting_before_there_is_a_table_says_so(panel):
    assert panel.export_csv("/tmp/never-written.csv") is None
    assert "Nothing to export" in panel.notice.text()
