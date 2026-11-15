"""Tabulate: the pivot engine's announced cuts, the axis wells and the panel.

A pivot is read as a summary of everything in the frame, so every way it is
not -- a row product too large to draw, a row cap, a cell ceiling, source
rows outside the shown levels -- must be printed in its notice, and a spec
naming a column the table lacks must come back as a :class:`PivotError`
sentence.  The wells accept only column drags and drop a key on Delete; the
panel turns every refusal into its notice, clears the previous table, warns on
low-n cells, plots the long frame and exports the CSV it shows.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QMimeData, QPointF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QDragEnterEvent, QDragMoveEvent, QDropEvent, QKeyEvent,
)

from spacr.qt.widgets import pivot_builder as pb  # noqa: E402
from spacr.qt.widgets import pivot_spec as PS  # noqa: E402
from spacr.qt.widgets.graph_builder import COLUMN_MIME  # noqa: E402
from spacr.qt.widgets.pivot_spec import (  # noqa: E402
    MEAN, N, SD, PivotError, PivotSpec, pivot,
)

pytestmark = pytest.mark.qt


def _frame():
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "rowID": ["r1", "r1", "r2", "r1", "r2", "r2"],
        "columnID": ["c1", "c2", "c1", "c2", "c1", "c2"],
        "gene": ["a", "b", "a", "b", "a", "b"],
        "area": [10.0, 20.0, 30.0, 34.0, 5.0, 7.0],
    })


def _six_keys():
    return pd.DataFrame({"plateID": [f"p{i}" for i in range(6)],
                         "gene": [f"g{i}" for i in range(6)],
                         "value": np.arange(6, dtype=float)})


def _column_mime(name):
    mime = QMimeData()
    mime.setData(COLUMN_MIME, name.encode("utf-8"))
    return mime


def _drop(well, name):
    """Drop a column payload onto a well the way the shelf does (payload kept alive)."""
    payload = _column_mime(name)
    well._list.dropEvent(QDropEvent(QPointF(well._list.rect().center()),
                                    Qt.CopyAction, payload, Qt.LeftButton,
                                    Qt.NoModifier))


_FULL_GRID = ("36 row combinations would be the full grid; only the 6 that "
              "occur are shown")


def test_pivot_names_every_cut_and_refuses_unknown_keys(monkeypatch, subtests):
    """Row-product fallback, row cap, cell ceiling, empty and keyless tables give exact shapes and notices."""
    with subtests.test(msg="row_product_falls_back_to_observed"):
        monkeypatch.setattr(PS, "MAX_ROWS", 10)
        result = pivot(_six_keys(), PivotSpec(rows=("plateID", "gene"),
                                              values=("value",)))
        assert result.shape == (6, 1)
        assert result.notice == _FULL_GRID
        assert result.hidden_rows == 0
    with subtests.test(msg="observed_rows_capped_and_named"):
        monkeypatch.setattr(PS, "MAX_ROWS", 3)
        result = pivot(_six_keys(), PivotSpec(rows=("plateID", "gene"),
                                              values=("value",)))
        cut = (f"{_FULL_GRID}; row capped at 3; 3 not shown; 3 source row(s) "
               "outside the shown levels")
        assert result.shape == (3, 1)
        assert result.notice == cut
        assert result.hidden_rows == 3
        assert result.summary() == (
            "3 × 1 = 3 cells, 3 with data · n per cell 1–1 · 3 cell(s) at "
            f"n ≤ 5 · 3 source row(s) outside the shown levels · {cut}")
    monkeypatch.undo()
    with subtests.test(msg="cell_ceiling_drops_rows_not_columns"):
        monkeypatch.setattr(PS, "MAX_CELLS", 2)
        frame = pd.DataFrame({"plateID": ["p1", "p2", "p3"],
                              "gene": ["a", "a", "a"],
                              "value": [1.0, 2.0, 3.0]})
        result = pivot(frame, PivotSpec(rows=("plateID",), cols=("gene",),
                                        values=("value",)))
        assert result.shape == (2, 1)
        assert result.row_levels == (("p1",), ("p2",))
        assert result.notice == ("grid capped at 2 cells; 1 source row(s) "
                                 "outside the shown levels")
    monkeypatch.undo()
    small = pd.DataFrame({"plateID": ["p1", "p1", "p2", "p2"],
                          "gene": ["a", "b", "a", "b"],
                          "value": [1.0, 2.0, 3.0, 4.0]})
    with subtests.test(msg="missing_row_key_is_a_pivot_error"):
        with pytest.raises(PivotError) as caught:
            pivot(small, PivotSpec(rows=("missing",), values=("value",)))
        assert str(caught.value) == ("'missing' is not a column of this table; "
                                     "it has 3 columns and none of them is "
                                     "that one")
    with subtests.test(msg="no_row_keys_labels_the_single_row_all"):
        result = pivot(small, PivotSpec(cols=("gene",), values=("value",)))
        assert result.shape[0] == 1
        assert result.to_frame()["rows"].tolist() == ["all"]
    with subtests.test(msg="no_rows_has_no_n_range"):
        empty = pd.DataFrame({"plateID": pd.Series(dtype=str),
                              "value": pd.Series(dtype=float)})
        result = pivot(empty, PivotSpec(rows=("plateID",), values=("value",)))
        assert result.n_range() is None
        assert result.shape == (0, 1)
        assert result.summary() == "0 × 1 = 0 cells, 0 with data"


def test_drop_wells_take_only_column_drags_and_drop_keys_on_delete(qtbot, subtests):
    """A well accepts column payloads, ignores plain text, and Delete/Backspace remove only the selected key."""
    def make(*names):
        well = pb.DropWell(pb.AXIS_ROWS)
        qtbot.addWidget(well)
        for name in names:
            _drop(well, name)
        return well

    for case_id, payload, accepted, after in (
            ("plain_text_ignored", "text", False, ()),
            ("column_taken", "column", True, ("plateID",))):
        with subtests.test(msg=case_id):
            well = make()
            if payload == "text":
                mime = QMimeData()
                mime.setText("plateID")
            else:
                mime = _column_mime("plateID")
            where = QPointF(well._list.rect().center())
            for cls, handler in ((QDragEnterEvent, well._list.dragEnterEvent),
                                 (QDragMoveEvent, well._list.dragMoveEvent)):
                event = cls(where.toPoint(), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
                event.setAccepted(not accepted)
                handler(event)
                assert event.isAccepted() is accepted
            well._list.dropEvent(QDropEvent(where, Qt.CopyAction, mime,
                                            Qt.LeftButton, Qt.NoModifier))
            assert well.columns() == after

    for case_id, row, key, after in (
            ("delete_selected", 0, Qt.Key_Delete, ("rowID",)),
            ("backspace_with_nothing_selected", -1, Qt.Key_Backspace,
             ("plateID", "rowID")),
            ("other_key_falls_through", 0, Qt.Key_A, ("plateID", "rowID"))):
        with subtests.test(msg=case_id):
            well = make("plateID", "rowID")
            seen = []
            well.changed.connect(lambda: seen.append(well.columns()))
            well._list.setCurrentRow(row)
            well._list.keyPressEvent(QKeyEvent(QEvent.KeyPress, key,
                                               Qt.NoModifier))
            assert well.columns() == after
            assert seen == ([after] if after == ("rowID",) else [])


def test_pivot_panel_refusals_become_the_notice_and_clear_the_table(qtbot, monkeypatch, subtests):
    """No table, a PivotError and an unexpected failure each set an exact notice and leave no result."""
    def boom(_frame, _spec):
        raise ValueError("reshape of a non-numeric column")

    def refuse(_frame, _spec):
        raise PivotError("group by something that is in the table")

    cases = {
        "no_table": (None, None, "Load a table, then drop a column onto Rows."),
        "pivot_error_as_it_stands": (
            _frame(), refuse, "group by something that is in the table"),
        "unexpected_failure_named_generically": (
            _frame(), boom,
            "could not build that table: reshape of a non-numeric column"),
    }
    for case_id, (frame, fake, notice) in cases.items():
        with subtests.test(msg=case_id):
            panel = pb.PivotPanel()
            qtbot.addWidget(panel)
            if frame is not None:
                panel.set_frame(frame)
                panel.set_spec(PivotSpec(rows=("plateID",), values=("area",),
                                         aggs=(MEAN,)))
                assert panel.recompute() is not None
            shown = []
            monkeypatch.setattr(panel.table, "set_result", shown.append)
            if fake is not None:
                monkeypatch.setattr(pb, "pivot", fake)
            assert panel.recompute() is None
            assert panel.notice.text() == notice
            assert panel.result is None
            assert shown == [None]
            assert panel.long_frame().empty
            monkeypatch.undo()


def test_pivot_panel_warns_low_n_plots_and_exports_what_it_shows(qtbot, monkeypatch, tmp_path, subtests):
    """Built tables give exact notices, tooltips, long frames, CSV writes and truncation messages."""
    def make(frame, spec):
        panel = pb.PivotPanel()
        qtbot.addWidget(panel)
        panel.set_frame(frame)
        panel.set_spec(spec)
        panel.recompute()
        return panel

    by_plate = PivotSpec(rows=("plateID",), values=("area",))
    with subtests.test(msg="low_n_cells_alone_are_anecdotes"):
        uneven = pd.DataFrame({
            "plateID": ["p1"] * 8 + ["p2"] * 2, "gene": ["a"] * 8 + ["b"] * 2,
            "area": [float(v) for v in range(10, 90, 10)] + [5.0, 7.0]})
        panel = make(uneven, by_plate)
        offset = len(panel.result.row_keys)
        assert panel.table.item(0, offset).toolTip() == (
            "p1\n8 source row(s)\nn(area) = 8\nmean(area) = 45\nsd(area) = 24.49")
        assert panel.table.item(1, offset).toolTip() == (
            "p2\n2 source row(s)\nn(area) = 2\nmean(area) = 6\nsd(area) = 1.414"
            "\nn ≤ 5: read this cell as an anecdote.")
    with subtests.test(msg="plot_carries_the_long_frame"):
        panel = make(_frame(), by_plate)
        assert panel.notice.text() == ("2 × 1 = 2 cells, 2 with data · n per "
                                       "cell 3–3 · 2 cell(s) at n ≤ 5")
        requested = []
        panel.plot_requested.connect(requested.append)
        panel._on_plot()
        assert len(requested) == 1
        assert requested[0].to_dict("list") == {
            "plateID": ["p1", "p2"], "cell_rows": [3, 3],
            "value_column": ["area", "area"], "n": [3.0, 3.0],
            "mean": [20.0, pytest.approx(46.0 / 3.0)],
            "sd": [10.0, pytest.approx(16.19670748434179)]}
    with subtests.test(msg="written_csv_named_in_notice"):
        panel = make(_frame(), by_plate)
        target = tmp_path / "tabulate.csv"
        assert panel.export_csv(str(target)) == str(target)
        assert target.read_text(encoding="utf-8").strip()
        assert panel.notice.text() == f"wrote {target}"
    with subtests.test(msg="unwritable_file_becomes_the_notice"):
        panel = make(_frame(), by_plate)
        assert panel.export_csv(str(tmp_path / "no such folder" / "t.csv")) is None
        assert panel.notice.text().startswith("could not write that file: ")
    with subtests.test(msg="cancelled_dialog_writes_nothing"):
        panel = make(_frame(), by_plate)
        monkeypatch.setattr(pb.QFileDialog, "getSaveFileName",
                            lambda *_a, **_k: ("", ""))
        before = panel.notice.text()
        assert panel.export_csv() is None
        assert panel.notice.text() == before
        monkeypatch.undo()
    with subtests.test(msg="truncated_table_says_to_export"):
        monkeypatch.setattr(pb, "MAX_RENDERED_CELLS", 2)
        panel = make(_frame(), PivotSpec(rows=("plateID",), cols=("gene",),
                                         values=("area",)))
        assert panel.notice.text() == (
            "2 × 2 = 4 cells, 4 with data · n per cell 1–2 · 4 cell(s) at "
            "n ≤ 5 · shown as a message rather than a grid — export it")
        monkeypatch.undo()
    with subtests.test(msg="grid_too_large_draws_one_message_cell"):
        monkeypatch.setattr(pb, "MAX_RENDERED_CELLS", 3)
        table = pb.PivotTable()
        qtbot.addWidget(table)
        result = pivot(_frame(), PivotSpec(rows=("plateID", "rowID"),
                                           cols=("gene",), values=("area",),
                                           aggs=(N, MEAN, SD)))
        table.set_result(result)
        assert table.truncated_cells == result.n_cells == 8
        assert (table.rowCount(), table.columnCount()) == (1, 1)
        assert table.item(0, 0).text() == ("8 cells is past the 3 this grid "
                                           "draws — narrow the axes, or "
                                           "export the CSV.")
        assert table.horizontalHeaderItem(0).text() == "too large to draw"
        monkeypatch.undo()
