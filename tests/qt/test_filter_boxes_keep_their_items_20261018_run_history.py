"""Typing in the run-history filter never deletes or creates table items."""
import pytest
from PySide6.QtCore import Qt

from spacr.qt.screens import run_history as history


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch, tmp_path):
    records = []
    for index, (app, status) in enumerate(
            [('measure', 'success'), ('mask', 'failed'),
             ('measure', 'failed'), ('classify', 'success')]):
        directory = tmp_path / f'run{index}'
        directory.mkdir()
        records.append(dict(
            run_id=f'run{index}', dir=directory, app_key=app, status=status,
            start_utc=f'2026-01-0{index + 1}T00:00:00+00:00',
            settings={'tag': f'alpha{index}' if index % 2 else 'beta'},
            performance={'wall_s': 10 - index}))
    monkeypatch.setattr(history, 'search_runs', lambda: list(records))
    widget = history.RunHistoryScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def _items(table):
    return {id(table.item(r, c)) for r in range(table.rowCount())
            for c in range(table.columnCount())}


def _shown(table):
    return [table.item(r, 0).data(Qt.UserRole)
            for r in range(table.rowCount()) if not table.isRowHidden(r)]


def _expected(widget):
    return {r['run_id'] for r in widget.records if widget._matches_filters(r)}


def test_typing_keeps_items_and_shows_matches(screen):
    table = screen._table
    before = _items(table)
    refs = [table.item(r, c) for r in range(table.rowCount())
            for c in range(table.columnCount())]
    for text in ('alpha', 'alpha1', 'zzz-none', '', 'beta', 'measure', ''):
        screen._search.setText(text)
        assert table.rowCount() == 4
        assert _items(table) == before
        assert set(_shown(table)) == _expected(screen)
        visible = _shown(table)
        if visible:
            first = next(r for r in range(4) if not table.isRowHidden(r))
            assert table.currentRow() == first
            assert screen._selected_record()["run_id"] == visible[0]
    assert refs
    assert set(_shown(table)) == {'run0', 'run1', 'run2', 'run3'}


def test_sorted_then_filtered(screen):
    table = screen._table
    before = _items(table)
    table.sortByColumn(3, Qt.AscendingOrder)
    screen._search.setText('alpha')
    assert _items(table) == before
    assert set(_shown(table)) == {'run1', 'run3'}
    first = next(r for r in range(4) if not table.isRowHidden(r))
    assert table.item(first, 0).data(Qt.UserRole) == _shown(table)[0]
    assert table.currentRow() == first
    screen._search.setText('nothing-matches')
    assert _shown(table) == []
    assert _items(table) == before
