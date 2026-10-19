"""Screens redraws keep their items (stale-wrapper sweep, group screens_d).

Model Zoo, Outliers, Power, Profiler, Project Browser, Report, Run Compare,
the cloud address chooser, Train Compare and Batch used to delete every
Python-made item in a redraw a click can trigger; with accessibility on that
segfaulted the tutorial host. Each redraw now rewrites the items it has and
parks the surplus.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication

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
    interface = QAccessible.queryAccessibleInterface(view)
    if interface is None:
        return
    for index in range(min(interface.childCount(), 80)):
        child = interface.child(index)
        if child is not None:
            child.text(QAccessible.Name)


def _list_items(listing):
    return [listing.item(r) for r in range(listing.count())]


def _cells(table):
    return [table.item(r, c) for r in range(table.rowCount())
            for c in range(table.columnCount()) if table.item(r, c)]


def _tree_items(tree):
    found = []

    def visit(item):
        found.append(item)
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))
    return found


def _alive(seen):
    assert all(shiboken6.isValid(i) for i in seen)


def test_model_zoo_listing_and_version_pick(qtbot, qt_theme_applied,
                                            accessible):
    from spacr import model_zoo as zoo
    from spacr.qt.screens.model_zoo import ModelZooScreen
    screen = ModelZooScreen()
    qtbot.addWidget(screen)
    seen = []
    for names in (["a", "b", "c"], ["x"], ["p", "q", "r", "s"], [],
                  ["a", "b"]):
        screen.set_entries([zoo.ModelEntry(key=n, name=f"{n}.CP_model",
                                           source="remote") for n in names])
        _walk(screen._table)
        QApplication.processEvents()
        seen.extend(_cells(screen._table))
        _alive(seen)
        assert screen._table.rowCount() == len(names)
        assert sorted(screen._table.item(r, 0).text()
                      for r in range(len(names))) == sorted(names)
    screen.set_entries([zoo.ModelEntry(key="m", name="m.CP_model",
                                       source="remote")])
    cell = screen._table.item(0, 0)
    for _ in range(3):
        screen._fill_row(0)
        assert screen._table.item(0, 0) is cell
    _alive(seen)


def test_outliers_fill_table_reuses_cells(qtbot, accessible):
    from PySide6.QtWidgets import QTableWidget
    from spacr.qt.screens.outliers import _fill_table
    table = QTableWidget()
    qtbot.addWidget(table)
    seen = []
    for rows, cols in ((3, 2), (1, 4), (5, 1), (0, 2), (2, 3)):
        frame = pd.DataFrame({f"c{c}": [f"v{r}{c}" for r in range(rows)]
                              for c in range(cols)})
        _fill_table(table, frame)
        _walk(table)
        seen.extend(_cells(table))
        _alive(seen)
        assert table.rowCount() == rows and table.columnCount() == cols
        assert table.horizontalHeaderItem(0).text() == "c0"
        for r in range(rows):
            for c in range(cols):
                assert table.item(r, c).text() == f"v{r}{c}"


def test_power_design_table_reuses_cells(qtbot, qt_theme_applied, accessible):
    from spacr.qt.screens.power import PowerScreen
    screen = PowerScreen()
    qtbot.addWidget(screen)
    seen = []
    for n in (3, 10, 1, 0, 4):
        designs = pd.DataFrame({
            "replicates": range(n), "wells": range(n), "fields": range(n),
            "power": [0.8] * n, "cost": [1.0] * n})
        screen._show_arrayed_designs(designs, "summary")
        _walk(screen._plan_table)
        seen.extend(_cells(screen._plan_table))
        _alive(seen)
        assert screen._plan_table.rowCount() == min(10, n)
        if n:
            assert screen._plan_table.item(0, 3).text() == "0.800"


def test_profiler_inputs_reuse_items(qtbot, tmp_path, accessible):
    from spacr.qt.screens import profiler as module
    for name, count in (("a", 3), ("b", 1), ("c", 4)):
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame({
            "feature": ["Intercept"] + [f"x{i}" for i in range(count)],
            "coefficient": [0.5] + [1.0 + i for i in range(count)],
        }).to_csv(folder / "results.csv", index=False)
    screen = module.ProfilerScreen(
        coefficients=str(tmp_path / "a" / "results.csv"), threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for name in ("a", "b", "c", "a"):
        screen.load_coefficients(str(tmp_path / name / "results.csv"))
        _walk(screen._inputs)
        seen.extend(_tree_items(screen._inputs))
        _alive(seen)
        assert screen._inputs.topLevelItemCount() == len(screen._ranked)
        assert screen._inputs.currentItem() is screen._inputs.topLevelItem(0)
        assert screen._inputs.topLevelItem(0).data(0, Qt.UserRole) \
            == screen._ranked[0].variable


def test_project_browser_roots_and_table(qtbot, tmp_path, accessible):
    from spacr.qt.screens.project_browser import ProjectBrowserScreen
    folders = []
    for i in range(3):
        folder = tmp_path / f"r{i}"
        folder.mkdir()
        folders.append(str(folder))
    screen = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for folder in folders:
        screen.add_root(folder)
        _walk(screen._root_list)
        _walk(screen._table)
        seen.extend(_list_items(screen._root_list))
        seen.extend(_cells(screen._table))
        _alive(seen)
        assert screen._root_list.count() == len(screen.roots())
    for _ in range(2):
        screen._root_list.setCurrentRow(0)
        screen.forget_selected_root()
        _walk(screen._root_list)
        _alive(seen)
        assert screen._root_list.count() == len(screen.roots())
    screen.rescan()
    _alive(seen)


def test_report_sections_reuse_items(qtbot, tmp_path, accessible):
    from spacr import report as rep
    from spacr.qt.screens.report import ReportScreen
    screen = ReportScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for count in (4, 2, 6, 0, 3):
        sections = [rep.Section(
            title=f"S{i}", key=f"k{i}",
            status=(rep.STATUS_MISSING if i % 3 == 0 else rep.STATUS_OK))
            for i in range(count)]
        screen._report = rep.Report(src=tmp_path, sections=sections,
                                    status="complete")
        screen._render_sections()
        _walk(screen._sections)
        seen.extend(_list_items(screen._sections))
        _alive(seen)
        assert screen._sections.count() == count
        for i in range(count):
            item = screen._sections.item(i)
            assert item.data(Qt.UserRole) == f"k{i}"
            assert not (item.flags() & Qt.ItemIsSelectable)
            assert ("not available" in item.text()) == (i % 3 == 0)
    screen._report = None
    screen._render_sections()
    assert screen._sections.count() == 0
    _alive(seen)


class _Block(SimpleNamespace):
    def __len__(self):
        return len(self.rows)


def test_run_compare_trees_reuse_items(qtbot, accessible):
    from spacr.qt.screens import run_compare as module
    tree = module._tree(module._SETTINGS_COLUMNS)
    qtbot.addWidget(tree)
    seen = []
    for cats, per in ((3, 4), (1, 1), (4, 2), (0, 0), (2, 5)):
        diff = SimpleNamespace(identical=False, include_same=False, categories=[
            _Block(category=f"c{c}", n_same=1, same=[],
                   rows=[SimpleNamespace(key=f"k{i}", a_val=i, b_val=i,
                                         kind="changed") for i in range(per)])
            for c in range(cats)])
        module._fill_settings(tree, SimpleNamespace(settings=diff))
        _walk(tree)
        seen.extend(_tree_items(tree))
        _alive(seen)
        assert tree.topLevelItemCount() == cats
        for c in range(cats):
            top = tree.topLevelItem(c)
            assert top.text(0) == f"c{c}" and top.childCount() == per
    module._fill_settings(tree, SimpleNamespace(settings=SimpleNamespace(
        identical=True, include_same=False, categories=[])))
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).text(0) == "No setting changed."
    assert tree.topLevelItem(0).childCount() == 0
    module._fill_settings(tree, SimpleNamespace(settings=None))
    assert tree.topLevelItemCount() == 0
    _alive(seen)


def test_cloud_chooser_listing_survives_a_double_click(qtbot, monkeypatch,
                                                       accessible):
    import threading
    from spacr.qt.screens.settings_model import _CloudBrowserDialog
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: SimpleNamespace(
        start=lambda: None))
    dialog = _CloudBrowserDialog("", True, "", None)
    qtbot.addWidget(dialog)
    seen = []
    for names in (["a", "b", "c"], ["z"], ["p", "q"], []):
        dialog._request += 1
        dialog._show_listing(dialog._request,
                             [(n, "folder", 0) for n in names], "")
        _walk(dialog.entries)
        seen.extend(_list_items(dialog.entries))
        _alive(seen)
        assert [dialog.entries.item(i).data(Qt.UserRole)
                for i in range(dialog.entries.count())] == names
    dialog.address.setText("s3://bucket")
    dialog._request += 1
    dialog._show_listing(dialog._request, [("sub", "folder", 0)], "")
    delivered = dialog.entries.item(0)
    dialog.entries.itemDoubleClicked.emit(delivered)
    assert dialog.address.text() == "s3://bucket/sub"
    assert dialog.entries.count() == 0
    assert shiboken6.isValid(delivered)
    _alive(seen)


def test_train_compare_runs_and_diff_reuse_items(qtbot, qt_theme_applied,
                                                 monkeypatch, accessible):
    from spacr import train_compare as tc
    from spacr.qt.screens.train_compare import TrainCompareScreen
    monkeypatch.setattr(tc, "available_metrics", lambda runs: [])
    screen = TrainCompareScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for n in (3, 1, 5, 0, 2):
        screen._runs = [SimpleNamespace(
            summary_line=lambda i=i: f"run {i}", run_id=f"id{i}",
            notes=["n"] if i % 2 else [], has_curves=bool(i % 2))
            for i in range(n)]
        screen._fill_runs_list()
        _walk(screen._runs_list)
        seen.extend(_list_items(screen._runs_list))
        _alive(seen)
        assert screen._runs_list.count() == n
        for i in range(n):
            item = screen._runs_list.item(i)
            assert item.data(Qt.UserRole) == f"id{i}"
            assert item.checkState() == Qt.Unchecked
            assert item.flags() & Qt.ItemIsUserCheckable
    for headers, rows in ((("a", "b"), [["1", "2"], ["3", "4"]]),
                          (("a", "b", "c"), [["1", "2", "3"]]),
                          (("a",), [["x"], ["y"], ["z"]]),
                          (("a", "b"), [])):
        screen._set_table(headers, rows)
        _walk(screen._diff_table)
        seen.extend(_cells(screen._diff_table))
        _alive(seen)
        assert screen.diff_headers() == list(headers)
        assert screen.diff_rows() == [list(r) for r in rows]
    screen._clear_diff()
    assert screen._diff_table.rowCount() == 0
    assert screen._diff_table.columnCount() == 0
    _alive(seen)


def test_batch_table_rewrites_cells_in_place(qtbot, qt_theme_applied,
                                             tmp_path, accessible):
    from spacr import batch as bt
    from spacr.qt.screens.batch import BatchScreen
    screen = BatchScreen()
    qtbot.addWidget(screen)
    seen = []
    for n in (3, 1, 4, 0, 2):
        screen._queue.jobs[:] = []
        for i in range(n):
            screen._queue.add(bt.Job(module="mask", id=f"j{i}",
                                     settings={"src": str(tmp_path)}),
                              validate=False)
        screen._refresh_table()
        _walk(screen._table)
        seen.extend(_cells(screen._table))
        _alive(seen)
        assert screen._table.rowCount() == n
        for i in range(n):
            assert screen._table.item(i, 1).text() == f"j{i}"
            assert screen._table.item(i, 0).data(Qt.UserRole) == f"j{i}"
    keep = _cells(screen._table)
    screen._refresh_table()
    assert _cells(screen._table) == keep
    _alive(seen)
