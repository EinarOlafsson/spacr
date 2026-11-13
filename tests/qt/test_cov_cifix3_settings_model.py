"""The settings form's record table reads every shape of stored value and removes rows safely."""
from __future__ import annotations

import pytest

from spacr.qt.screens.settings_model import _RecordColumn, _RecordTable

pytestmark = pytest.mark.qt

COLUMNS = (_RecordColumn('name', 'Name'), _RecordColumn('size', 'Size', 'int'))


@pytest.fixture
def make(qtbot):
    def build(default=None, mapping_key=None):
        table = _RecordTable(COLUMNS, default, mapping_key=mapping_key)
        qtbot.addWidget(table)
        return table
    return build


def test_removing_with_nothing_selected_and_no_current_row_changes_nothing(make):
    table = make([{'name': 'a'}])
    table.table.clearSelection()
    table.table.setCurrentCell(-1, -1)
    changes = []
    table.value_changed.connect(lambda: changes.append(True))
    table._on_remove()
    assert table.table.rowCount() == 1
    assert changes == []
