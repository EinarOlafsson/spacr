"""Redraws in the widgets_b group rewrite their items, never delete them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible

pytestmark = pytest.mark.qt


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    iface = QAccessible.queryAccessibleInterface(view)
    if iface is None:
        return
    for i in range(iface.childCount()):
        iface.child(i).text(QAccessible.Name)


class Seen:
    def __init__(self, view):
        self.view = view
        self.items = []

    def _current(self):
        view = self.view
        if hasattr(view, "rowCount"):
            return [view.item(r, c) for r in range(view.rowCount())
                    for c in range(view.columnCount()) if view.item(r, c) is not None]
        return [view.item(i) for i in range(view.count())]

    def take(self):
        _walk(self.view)
        for item in self._current():
            if not any(item is seen for seen in self.items):
                self.items.append(item)

    def valid(self):
        return all(shiboken6.isValid(item) for item in self.items)


def _frame(n, cols=("a", "b", "c")):
    return pd.DataFrame({c: [f"{c}{i}" if k else i for i in range(n)]
                         for k, c in enumerate(cols)})


def test_results_table_set_frame(qtbot, accessible):
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)
    seen = Seen(table.table)
    for _ in range(3):
        for n, cols in ((5, ("a", "b", "c")), (2, ("a", "b")),
                        (7, ("a", "b", "c", "d")), (0, ("a",))):
            frame = _frame(n, cols) if n else None
            assert table.set_frame(frame) == n
            assert table.table.rowCount() == n
            seen.take()
            assert seen.valid()
    table.set_frame(_frame(3))
    assert table.table.item(2, 0).text() == "2"
    assert table.table.item(1, 1).text() == "b1"


def test_pivot_well_and_table(qtbot, accessible):
    from spacr.qt.widgets import pivot_builder as pb
    from spacr.qt.widgets.pivot_spec import MEAN, N, PivotSpec, pivot

    well = pb.DropWell(pb.AXIS_ROWS)
    qtbot.addWidget(well)
    seen = Seen(well._list)
    for _ in range(3):
        for names in (("a", "b", "c"), ("d",), ("e", "f")):
            well.set_columns(names)
            assert well.columns() == names
            seen.take()
            assert seen.valid()
        well._on_dropped("z")
        assert well.columns() == ("e", "f", "z")
        well._on_remove(0)
        assert well.columns() == ("f", "z")
        seen.take()
        assert seen.valid()
        well.clear()
        assert well.columns() == () and well._list.count() == 0
        seen.take()
        assert seen.valid()

    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "rowID": ["r1", "r1", "r2", "r1", "r2", "r2"],
        "gene": ["a", "b", "a", "b", "a", "b"],
        "area": [10.0, 20.0, 30.0, 34.0, 5.0, 7.0],
    })
    table = pb.PivotTable()
    qtbot.addWidget(table)
    tseen = Seen(table)
    for _ in range(3):
        for rows, cols in ((("plateID", "rowID"), ("gene",)),
                           (("plateID",), ()), (("rowID",), ("gene",))):
            result = pivot(frame, PivotSpec(rows=rows, cols=cols,
                                            values=("area",), aggs=(N, MEAN)))
            table.set_result(result)
            assert table.rowCount() == result.shape[0]
            assert table.cell_text(0, 0) != ""
            tseen.take()
            assert tseen.valid()
        table.set_result(None)
        assert table.rowCount() == 0 and table.columnCount() == 0
        tseen.take()
        assert tseen.valid()


def test_multiselect_and_checkable_combo(qtbot, accessible):
    from spacr.qt.widgets.row_exclusion import _CheckableValueCombo
    from spacr.qt.widgets.volcano_explorer import _MultiSelect

    box = _MultiSelect("which")
    qtbot.addWidget(box)
    seen = Seen(box)
    heard = []
    box.changed.connect(lambda: heard.append(1))
    for _ in range(3):
        for options, ticked in ((["x", "y", "z"], ["y"]), (["x"], []),
                                (["p", "q"], ["q", "w"])):
            box.setValues(ticked, options)
            expect = list(options) + [t for t in ticked if t not in options]
            assert box.options() == expect
            assert list(box.values()) == [o for o in expect if o in ticked]
            assert box.count() == len(expect)
            seen.take()
            assert seen.valid()
    assert heard == []
    box.setOptions(["m", "n"])
    assert box.options()[:2] == ["m", "n"]

    combo = _CheckableValueCombo()
    qtbot.addWidget(combo)
    model = combo.model()
    kept = []
    for _ in range(3):
        for options, ticked in (([1, 2, 3], [2]), ([4], []), ([5, 6], [6, 9])):
            combo.set_options(options, ticked)
            expect = options + [t for t in ticked if t not in options]
            assert model.rowCount() == len(expect)
            assert [model.item(r).data(Qt.UserRole)
                    for r in range(model.rowCount())] == expect
            assert combo.checked_values() == [o for o in expect if o in ticked]
            for r in range(model.rowCount()):
                if not any(model.item(r) is k for k in kept):
                    kept.append(model.item(r))
            kept.extend(combo.__dict__.get("_spare_rows", []))
            assert all(shiboken6.isValid(k) for k in kept)


def test_sweep_panel_refill(qtbot, accessible):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    rng = np.random.default_rng(0)
    n = 80
    a = rng.random(n)
    cells = pd.DataFrame({
        "plateID": ["plate1"] * 40 + ["plate2"] * 40,
        "rowID": [f"r{i}" for i in range(n)],
        "columnID": ["c1"] * n,
        "real": a * 3.0 + rng.normal(0, 0.2, n),
        "noise": rng.normal(0, 1, n),
        "pred": rng.random(n),
    })
    counts = pd.DataFrame(
        [{"prc": f"plate{1 + i // 40}_r{i}_c1", "grna": g,
          "fraction": (a[i] if g == "A" else 1 - a[i])}
         for i in range(n) for g in ("A", "B")])
    panel = SweepPanel(lambda: cells, lambda: counts, threaded=False)
    qtbot.addWidget(panel)
    assert panel.start() is True
    seen = Seen(panel.table)
    for _ in range(3):
        for alpha in (1.0, 0.05, 0.001, 1.0):
            panel.alpha.setValue(alpha)
            panel._refill()
            assert panel.table.rowCount() == len(panel.rows())
            if panel.table.rowCount():
                assert panel.table.item(0, 1) is not None
            seen.take()
            assert seen.valid()


def test_feature_explorer_table(qtbot, accessible):
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    from spacr.qt.widgets.feature_rank import ExplorerSpec

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"label": ["a", "b"] * 20,
                          "f1": rng.normal(size=40), "f2": rng.normal(size=40),
                          "f3": rng.normal(size=40)})
    panel.set_frame(frame)
    seen = Seen(panel.table)
    for _ in range(3):
        for top in (3, 1, 2):
            panel.set_spec(ExplorerSpec(label="label", top=top, bins=8))
            result = panel.rank_now()
            assert result is not None
            assert panel.table.rowCount() == len(result.scores)
            assert panel.selected_feature() == result.scores[0].feature
            seen.take()
            assert seen.valid()
        panel._clear_result()
        assert panel.table.rowCount() == 0
        seen.take()
        assert seen.valid()


def test_measure_input_table_rebuilds(qtbot, accessible):
    from spacr.qt.widgets import measure_input_table as mit

    widget = mit.MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    seen = Seen(widget._grid)
    lseen = Seen(widget._left_over)
    widget._unassigned = [("/x/a.tif", "no match"), ("/x/b.tif", "no match")]
    for _ in range(3):
        for _n in range(3):
            widget.add_field()
            assert widget._grid.rowCount() == len(widget.table().rows)
            seen.take()
            lseen.take()
            assert seen.valid() and lseen.valid()
        assert widget._left_over.count() == 2
        assert widget._left_over.item(0).text() == "a.tif -- no match"
        widget._grid.selectRow(0)
        widget.remove_selected()
        assert widget._grid.rowCount() == len(widget.table().rows)
        seen.take()
        assert seen.valid()
        widget.clear()
        assert widget._grid.rowCount() == 0
        widget._unassigned = [("/x/a.tif", "no match"), ("/x/b.tif", "no match")]
        widget._rebuild()
        assert widget._left_over.count() == 2
        seen.take()
        lseen.take()
        assert seen.valid() and lseen.valid()


def test_regression_columns_and_scan_tables(qtbot, accessible):
    from spacr.qt.widgets import measurement_scan_panel as msp

    state = {"cols": ["m1", "m2", "m3"]}

    def provider():
        rng = np.random.default_rng(0)
        return pd.DataFrame({c: rng.normal(size=6) for c in state["cols"]})

    widget = msp.ColumnRegressionPanel(frame_provider=provider, threaded=False,
                                       fit=lambda s: {})
    qtbot.addWidget(widget)
    seen = Seen(widget.columns_list)
    for _ in range(3):
        for cols in (["m1", "m2", "m3"], ["m2"], ["m1", "m2", "m3", "m4"]):
            state["cols"] = cols
            assert widget.refresh() == len(cols)
            assert [widget.columns_list.item(i).text()
                    for i in range(widget.columns_list.count())] == cols
            seen.take()
            assert seen.valid()
    widget.columns_list.item(1).setSelected(True)
    state["cols"] = ["m1", "m2"]
    widget.refresh()
    assert widget.selected_columns() == ("m2",)

    panel = msp.DatabaseMergePanel(lambda: [], threaded=False)
    qtbot.addWidget(panel)
    names = {"v": ("cell", "pathogen")}
    panel.paths = lambda: ["a.db"]
    panel.describe = lambda *a, **k: None
    panel._read_off_thread = lambda key, work: names["v"]
    tseen = Seen(panel.tables_list)
    for _ in range(3):
        for n, ticked in ((("cell", "pathogen", "cytoplasm"), ("cell",)),
                          (("cell",), ()), (("cell", "pathogen"), ("pathogen",))):
            names["v"] = n
            panel._offer_tables()
            panel.set_selected_tables(ticked)
            assert panel.selected_tables() == tuple(t for t in n if t in ticked)
            panel._offer_tables()
            assert panel.selected_tables() == tuple(t for t in n if t in ticked) or not ticked
            assert [panel.tables_list.item(i).text()
                    for i in range(panel.tables_list.count())] == list(n)
            tseen.take()
            assert tseen.valid()


def test_merge_rules_table(qtbot, accessible, tmp_path):
    import sqlite3

    from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog

    path = str(tmp_path / "measurements.db")
    ident = dict(plateID=["p1"] * 3, rowID=["r1"] * 3, columnID=["c1"] * 3,
                 fieldID=["f1"] * 3)
    with sqlite3.connect(path) as connection:
        pd.DataFrame({**ident, "object_label": [1, 2, 3],
                      "area": [100., 200., 300.]}).to_sql("cell", connection, index=False)
        pd.DataFrame({**ident, "object_label": [7, 8, 9], "cell_id": [1, 1, 2],
                      "area": [2., 3., 7.],
                      "mean_intensity": [5., 7., 11.]}).to_sql("pathogen", connection, index=False)
    dialog = MergeTablesDialog(path, selected=["cell", "pathogen"], threaded=False)
    qtbot.addWidget(dialog)
    seen = Seen(dialog.rules)
    first = dialog.rules.rowCount()
    assert first > 0
    for _ in range(4):
        dialog._fill_rules()
        assert dialog.rules.rowCount() == first
        assert dialog.rules.item(0, 0).text() == "cell"
        assert dialog.rules.item(0, 1).text() == "area"
        assert [dialog.rules.item(row, 0).text() for row in range(first)] == (
            ["cell"] + ["pathogen"] * (first - 1))
        assert {dialog.rules.item(row, 1).text() for row in range(1, first)} >= {
            "area", "mean_intensity"}
        seen.take()
        assert seen.valid()


def test_model_zoo_picker_rebuild(qtbot, accessible, tmp_path, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker as mzp

    picker = mzp.ModelZooPicker(kinds=("cellpose",))
    qtbot.addWidget(picker)
    picker.folder_edit.setText(str(tmp_path / "models"))
    entries = list(picker._entries)
    assert entries
    seen = Seen(picker.table)
    for _ in range(3):
        for chunk in (entries, entries[:1], entries):
            picker._rebuild(chunk)
            assert picker.table.rowCount() == len(picker._groups)
            first = picker.table.item(0, 0).text()
            assert first == picker._groups[picker._group_of_row(0)][0]
            seen.take()
            assert seen.valid()
        picker._fill_row(0)
        picker.table.selectRow(0)
        seen.take()
        assert seen.valid()


def test_plaque_tables_and_image_list(qtbot, accessible, monkeypatch):
    from types import SimpleNamespace

    from spacr.qt.widgets import plaque_preview as ppv

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    panel = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    rows = {"v": 3}
    monkeypatch.setattr(panel, "plaque_table_rows", lambda: [
        {key: float(i) if key != "well" else f"w{i}" for key in ppv.PLAQUE_KEYS}
        for i in range(rows["v"])])
    pseen = Seen(panel._plaque_table)
    tseen = Seen(panel._table)
    edits = []
    monkeypatch.setattr(panel, "_on_table_edit", lambda item: edits.append(item))
    panel._table.itemChanged.disconnect()
    panel._table.itemChanged.connect(panel._on_table_edit)
    for _ in range(3):
        for n in (3, 1, 4):
            rows["v"] = n
            panel._fill_plaque_table()
            assert panel._plaque_table.rowCount() == n
            assert panel._plaque_table.item(0, 0).text() == "w0"
            pseen.take()
            assert pseen.valid()
        panel._clear_figure()
        assert panel._plaque_table.rowCount() == 0 and panel._table.rowCount() == 0
        pseen.take()
        tseen.take()
        assert pseen.valid() and tseen.valid()
    assert edits == []

    dialog = ppv.ContributeDialog("plaque", [], threaded=False)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_build_page", lambda path, known: None)
    kept = []
    for _ in range(3):
        for name in ("a.png", "b.png", "c.png"):
            dialog.add_image(name)
        assert dialog.image_list.count() == 3
        for _n in range(3):
            kept.append(dialog.image_list.item(dialog.image_list.currentRow()))
            dialog.remove_current()
            _walk(dialog.image_list)
        assert dialog.image_list.count() == 0
        assert all(shiboken6.isValid(k) for k in kept)
