"""A Lineage tree loaded from the database is written over many event-loop turns.

Rewriting a sorted 7,015-node tree took 210-230 ms in one turn and its sort
another 56-60 ms in the next (lesson 56, Build tree). A load now writes the
rows a slice at a time with the sort held, reads each row's sort key in the
turn that writes it, and sorts once at the end on those keys. What ends up
on screen -- rows, order, sort indicator, expansion -- is what the one-turn
redraw gave. Counted, not timed.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.screens import lineage as lgs  # noqa: E402
from spacr.qt.widgets import sortable_table as st  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_lineage_screen import F1, F2, _rows  # noqa: E402

pytestmark = pytest.mark.qt


def _frames(cells):
    entries = [(F1 if index % 2 else F2, index + 1, None)
               for index in range(cells)]
    nuclei = [(F1 if index % 2 else F2, 500 + index, index + 1)
              for index in range(0, cells, 3)]
    pathogens = [(F1 if index % 2 else F2, 900 + index, index + 1)
                 for index in range(0, cells, 4)]
    return {"cell": _rows("cell", entries),
            "nucleus": _rows("nucleus", nuclei),
            "pathogen": _rows("pathogen", pathogens)}


def _shape(tree):
    out = []
    for row in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(row)
        out.append((item.text(0), item.text(1), item.isExpanded(),
                    [item.child(i).text(0) for i in range(item.childCount())]))
    return out


def _screen(qtbot, frames):
    screen = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_frames(frames)
    QApplication.processEvents()
    screen.tree.header().setSortIndicator(1, Qt.DescendingOrder)
    screen.tree.topLevelItem(0).setExpanded(True)
    return screen


def test_a_load_writes_one_slice_a_turn_and_sorts_once_on_read_keys(
        qtbot, qt_theme_applied, monkeypatch):
    frames = _frames(40)
    reference = _screen(qtbot, frames)
    reference._show_built(lgs._build_lineage(frames), in_turns=False)
    st._settle_sorting(reference.tree)
    expected = _shape(reference.tree)

    screen = _screen(qtbot, frames)
    monkeypatch.setattr(lgs, "_FILL_TURN_S", 0.0)
    written = []
    real_item = screen._item

    def item(node, row, prime=False):
        if row.parent() is None:
            written.append(node.key)
        return real_item(node, row, prime)

    monkeypatch.setattr(screen, "_item", item)
    screen._show_built(lgs._build_lineage(frames), in_turns=True)

    assert len(written) == 1
    assert not screen.tree.updatesEnabled()
    assert screen.tree.header().sortIndicatorSection() == -1
    QApplication.processEvents()
    assert len(written) == 2

    key_reads = []
    real_key = st.SortableTreeItem._resolve_key

    def counted(self, *args):
        key_reads.append(self)
        return real_key(self, *args)

    qtbot.waitUntil(lambda: len(written) == 40, timeout=10000)
    monkeypatch.setattr(st.SortableTreeItem, "_resolve_key", counted)
    qtbot.waitUntil(lambda: screen.tree.updatesEnabled(), timeout=10000)

    assert key_reads == []
    header = screen.tree.header()
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (
        1, Qt.DescendingOrder)
    assert _shape(screen.tree) == expected
    assert screen.status.text() == reference.status.text()


def test_a_newer_load_stops_the_fill_before_it(qtbot, qt_theme_applied,
                                               monkeypatch):
    frames = _frames(30)
    screen = _screen(qtbot, frames)
    monkeypatch.setattr(lgs, "_FILL_TURN_S", 0.0)
    screen._show_built(lgs._build_lineage(frames), in_turns=True)
    smaller = _frames(6)
    screen.set_frames(smaller)
    assert screen.tree.updatesEnabled()
    QApplication.processEvents()
    QApplication.processEvents()
    assert screen.tree.topLevelItemCount() == 6
    assert screen.tree.updatesEnabled()
