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


@pytest.mark.parametrize('stored', [None, '', '   ', 'None'])
def test_an_empty_stored_value_shows_no_rows_and_no_problems(make, stored):
    table = make(stored)
    assert table.table.rowCount() == 0
    assert table.problems() == []
    assert table.get_value() is None


def test_a_stored_text_that_is_not_a_literal_is_listed_as_a_problem(make):
    table = make('[{unclosed')
    assert table.table.rowCount() == 0
    assert 'is not a list of records' in table.problems()[0]
    assert not table.status.isHidden()


def test_a_stored_text_holding_records_becomes_rows(make):
    table = make("[{'name': 'a', 'size': 3}]")
    assert table.table.rowCount() == 1
    assert table.get_value() == [{'name': 'a', 'size': 3}]


def test_a_single_record_and_a_mapping_without_a_mapping_key_become_one_row(make):
    assert make({'name': 'a', 'size': 1}).get_value() == [{'name': 'a', 'size': 1}]
    table = make({'x': {'size': 2}})
    assert table.table.rowCount() == 1
    assert table.get_value() == [{'x': {'size': 2}}]


def test_a_mapping_of_records_becomes_one_row_per_name_and_reads_back_as_a_mapping(make):
    table = make({'one': {'size': 1}, 'two': {'size': 2}}, mapping_key='name')
    assert table.table.rowCount() == 2
    assert table.get_value() == {'one': {'size': 1}, 'two': {'size': 2}}


def test_a_row_without_a_name_is_left_out_of_a_stored_mapping(make):
    table = make({'one': {'size': 1}}, mapping_key='name')
    table._on_add()
    table.table.item(1, 1).setText('5')
    assert table.get_value() == {'one': {'size': 1}}


def test_entries_that_are_not_records_are_left_out_with_a_note(make):
    table = make([{'name': 'a'}, 'oops', 7])
    assert table.table.rowCount() == 1
    notes = table.problems()
    assert len(notes) == 2
    assert 'Entry 2 is not a record' in notes[0]
    assert 'Entry 3 is not a record' in notes[1]


def test_a_single_value_that_is_not_a_list_is_one_entry_and_not_a_record(make):
    table = make(5)
    assert table.table.rowCount() == 0
    assert 'Entry 1 is not a record' in table.problems()[0]


def test_removing_with_nothing_selected_and_no_current_row_changes_nothing(make):
    table = make([{'name': 'a'}])
    table.table.clearSelection()
    table.table.setCurrentCell(-1, -1)
    changes = []
    table.value_changed.connect(lambda: changes.append(True))
    table._on_remove()
    assert table.table.rowCount() == 1
    assert changes == []


def test_removing_the_current_row_without_a_selection_removes_it_and_announces(make):
    table = make([{'name': 'a'}, {'name': 'b'}])
    table.table.clearSelection()
    table.table.setCurrentCell(1, 0)
    changes = []
    table.value_changed.connect(lambda: changes.append(True))
    table._on_remove()
    assert table.table.rowCount() == 1
    assert table.get_value() == [{'name': 'a'}]
    assert changes == [True]
