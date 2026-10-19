"""Redraws after a user action keep their items (lessons 42 and 44 segfaults).

Track curation's "Delete selected" and Data Manager's choose-folder click
both crashed the tutorial host with SIGSEGV in a redraw that deleted every
Python-made item of a list or table (``curation_tool.py`` ``refresh`` and
``data_manager.py`` ``_clear_tables``). Reproduced offscreen on 2026-10-18
with accessibility on and wrappers of accessible cells held: the native
frame is ``Shiboken::BindingManager::releaseWrapper`` in an item's
destructor. The redraws now rewrite the items they have and park the
surplus, so no click deletes an item; these tests pin that, with
accessibility on.
"""
from __future__ import annotations

import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication

from spacr.qt.curation_tool import TrackCurationPanel


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    """Read a view's cells the way a screen reader would, holding nothing."""
    interface = QAccessible.queryAccessibleInterface(view)
    if interface is None:
        return
    for index in range(min(interface.childCount(), 60)):
        child = interface.child(index)
        if child is not None:
            child.text(QAccessible.Name)


def _items(listing):
    return [listing.item(row) for row in range(listing.count())]


def _tracks(n=12):
    rows = [(f, t, 10.0 * t + f, 5.0 + f) for t in range(n) for f in range(5)]
    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"])


def test_deleting_tracks_keeps_every_list_item_alive(qtbot, accessible):
    panel = TrackCurationPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_tracks(_tracks())
    seen = list(_items(panel.track_list))
    for step in range(8):
        listing = panel.track_list
        listing.item(step % listing.count()).setSelected(True)
        _walk(listing)
        assert panel.delete_selected()
        _walk(listing)
        _walk(panel.ledger)
        QApplication.processEvents()
        seen.extend(_items(listing))
        assert all(shiboken6.isValid(item) for item in seen)
        ids = [item.data(Qt.UserRole) for item in _items(listing)]
        assert ids == list(panel.session.track_ids)
        assert panel.track_list.selectedItems() == []
        assert panel.ledger.count() == len(panel.session.log.edits)
    texts = [item.text() for item in _items(panel.track_list)]
    assert all(text.startswith("track ") for text in texts)


def test_reopening_tracks_reuses_rows_and_restores_the_count(qtbot, accessible):
    panel = TrackCurationPanel()
    qtbot.addWidget(panel)
    panel.set_tracks(_tracks(12))
    first = _items(panel.track_list)
    panel.set_tracks(_tracks(4))
    assert panel.track_list.count() == 4
    assert _items(panel.track_list) == first[:4]
    panel.set_tracks(_tracks(10))
    assert panel.track_list.count() == 10
    assert all(shiboken6.isValid(item) for item in first)
    assert [item.data(Qt.UserRole) for item in _items(panel.track_list)] == list(range(10))


def test_choosing_project_folders_keeps_the_table_items(
        qtbot, accessible, tmp_path, monkeypatch, qt_theme_applied):
    from PySide6.QtWidgets import QFileDialog, QPushButton
    from PySide6.QtTest import QTest
    from spacr.qt.screens import data_manager as DM

    roots = []
    for k in range(3):
        root = tmp_path / f"project{k}"
        for sub in ("masks", "measurements", f"extra{k}"):
            (root / sub).mkdir(parents=True)
            for i in range(3 + k):
                (root / sub / f"f{i}.bin").write_bytes(b"x" * (10 * i + 1))
        roots.append(str(root))
    choice = [roots[0]]
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: choice[0]))
    screen = DM.DataManagerScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.show()
    button = [b for b in screen.findChildren(QPushButton)
              if b.text() == "Choose project…"][0]
    seen = []
    for step in range(9):
        choice[0] = roots[step % 3]
        _walk(screen.usage_table)
        QTest.mouseClick(button, Qt.LeftButton)
        qtbot.waitUntil(lambda: not screen._busy and not screen._jobs,
                        timeout=5000)
        table = screen.usage_table
        assert table.rowCount() > 0
        current = [table.item(r, c) for r in range(table.rowCount())
                   for c in range(table.columnCount())]
        assert all(item is not None for item in current)
        seen.extend(current)
        assert all(shiboken6.isValid(item) for item in seen)
        assert screen.project_label.text() == choice[0]
        _walk(table)
        QApplication.processEvents()
    first = screen.usage_table.item(0, 0)
    assert first.textAlignment() == int(Qt.AlignRight | Qt.AlignVCenter)
    other = screen.usage_table.item(0, 1)
    assert other.data(Qt.TextAlignmentRole) in (None, 0)
