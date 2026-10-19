"""Redraws on five screens keep the items their views already have.

A redraw that deletes Python-made Qt items while assistive technology holds
wrappers of them crashed lessons 39, 42 and 44. Each test drives the user
action that redraws, with accessibility on, and checks no item seen earlier
was deleted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PySide6")
import shiboken6
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QListWidget, QTreeWidget


@pytest.fixture()
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
    for k in range(iface.childCount()):
        child = iface.child(k)
        if child is not None:
            child.text(QAccessible.Name)


def _items(view):
    if isinstance(view, QTreeWidget):
        out = []

        def visit(item):
            out.append(item)
            for k in range(item.childCount()):
                visit(item.child(k))
        for k in range(view.topLevelItemCount()):
            visit(view.topLevelItem(k))
        return out
    if isinstance(view, QListWidget):
        return [view.item(k) for k in range(view.count())]
    return [view.item(r, c) for r in range(view.rowCount())
            for c in range(view.columnCount()) if view.item(r, c)]


class _Seen:
    def __init__(self, *views):
        self.views = views
        self.seen = []

    def check(self):
        for view in self.views:
            _walk(view)
            self.seen.extend(_items(view))
        assert all(shiboken6.isValid(item) for item in self.seen)


def _hit_list(n):
    from spacr.hits import Hit, HitList
    return HitList(hits=tuple(
        Hit(gene=f"g{i}", effect=0.9 - i * 0.1, p_value=0.001, q_value=0.01,
            n_guides=3, n_agree=3, agreement=1.0, direction="up", rank=i + 1,
            flags=("f",) if i % 2 else ())
        for i in range(n)))


def test_hit_list_redraw(qtbot, accessible):
    from spacr.qt.screens.hit_list import HitListScreen
    screen = HitListScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = _Seen(screen._table)
    for n in (5, 2, 6, 0, 3):
        screen._on_hits_ready(_hit_list(n))
        assert screen._table.topLevelItemCount() == n
        for row in range(n):
            top = screen._table.topLevelItem(row)
            assert top.text(1) == f"g{row}"
            assert top.data(0, 256) == f"g{row}"
        seen.check()
    assert screen._table.currentItem() is None


def test_lineage_reload(qtbot, accessible):
    from spacr.qt.screens.lineage import LineageScreen

    def rows(table, entries):
        out = []
        for label, parent in entries:
            rec = {"plateID": "p", "rowID": "r", "columnID": "c",
                   "fieldID": "f", "object_label": label,
                   f"{table}_area": 1.0}
            if parent is not None:
                rec["cell_id"] = parent
            out.append(rec)
        return pd.DataFrame(out)

    screen = LineageScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = _Seen(screen.tree, screen.orphan_list)
    for cells, nuclei, orphan in [(4, 3, 9), (1, 1, None), (3, 2, 8),
                                  (0, 0, None)]:
        frames = {}
        if cells:
            frames["cell"] = rows("cell", [(i + 1, None) for i in range(cells)])
            entries = [(i + 1, 1 + i % cells) for i in range(nuclei)]
            if orphan:
                entries.append((50, orphan))
            frames["nucleus"] = rows("nucleus", entries)
        screen.set_frames(frames)
        assert screen.tree.topLevelItemCount() == cells
        if cells:
            assert screen.tree.topLevelItem(0).childCount() >= 1
            screen.tree.setCurrentItem(screen.tree.topLevelItem(0))
        seen.check()


def _finding(i):
    return SimpleNamespace(
        role="r", table=f"t{i}", file_label="f", orientation="fwd",
        observed_rate=0.5, expected_rate=0.1, enrichment=5.0,
        modal_offset=None, verdict="present", reason="why")


def test_barcode_findings_redraw(qtbot, accessible):
    from spacr.qt.screens.map_barcodes import BarcodeSearchPanel
    panel = BarcodeSearchPanel(threaded=False)
    qtbot.addWidget(panel)
    seen = _Seen(panel.findings)
    for n in (3, 5, 2, 0, 4):
        panel._fill_findings(SimpleNamespace(
            findings=[_finding(i) for i in range(n)]))
        assert panel.findings.rowCount() == n
        seen.check()
        panel._reset()
        assert panel.findings.rowCount() == 0
        seen.check()


def _side(**honoured):
    return SimpleNamespace(honoured_parameters=lambda: honoured,
                           ignored_parameters=lambda: {})


def _comparison(name):
    return SimpleNamespace(
        field=name, n_objects_a=1, n_objects_b=2, object_count_delta=1,
        ari=0.5, iou_matched_fraction=0.5, mean_matched_iou=0.5,
        split_events=0, merge_events=0, new_objects_b=0, missing_objects_a=0,
        qc_a=None, qc_b=None, note="n")


def test_model_compare_results_redraw(qtbot, accessible):
    from spacr.qt.screens.model_compare import ModelCompareScreen
    screen = ModelCompareScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = _Seen(screen._param_table, screen._row_table)
    for n in (3, 1, 4):
        report = SimpleNamespace(
            model_a=_side(a=1, b=2), model_b=_side(a=1, b=3),
            comparisons=[_comparison(f"f{i}") for i in range(n)])
        screen._fill_param_table(report)
        screen._fill_row_table(report)
        assert screen._row_table.rowCount() == n
        assert screen._param_table.rowCount() == 2
        assert screen._row_table.item(n - 1, 0).text() == f"f{n - 1}"
        seen.check()
        screen._clear_results()
        assert screen._row_table.rowCount() == 0
        assert screen._param_table.rowCount() == 0
        seen.check()


def test_model_explanation_click_rebuilds_distributions(qtbot, accessible):
    from spacr.qt.screens.model_explanation import ExplainCvPanel, _fill_table
    from tests.test_cov_w2_4_model_explanation import _surrogate
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    panel._loaded((_surrogate(), {}))
    seen = _Seen(panel.importance, panel.distributions, panel.metrics)
    seen.check()
    for row in (0, 1, 0, 1):
        panel.importance.selectRow(row)
        assert panel.distributions.rowCount() == (2 if row == 0 else 1)
        seen.check()
        panel.importance.clearSelection()
        panel._show_selected_feature_distribution()
        assert panel.distributions.rowCount() == 3
        seen.check()
    _fill_table(panel.metrics, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert panel.metrics.columnCount() == 2
    assert panel.metrics.horizontalHeaderItem(1).text() == "b"
    seen.check()
