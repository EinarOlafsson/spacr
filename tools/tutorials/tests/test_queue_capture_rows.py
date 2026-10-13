"""Replay inputs must retain every measured value, not merely cell counts."""
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_plate_queue import verify_frozen_rows
from recruitment_data import _row_hash, _rows_hash


@pytest.fixture
def frozen(tmp_path):
    path = tmp_path / 'measurements.db'
    table_names = ('cell', 'nucleus', 'pathogen', 'cytoplasm')
    rows = [('plate1', 'r1', 'c2', 'f3', 1, 17.5),
            ('plate1', 'r1', 'c2', 'f3', 2, 28.5)]
    expected = {}
    with sqlite3.connect(path) as connection:
        for name in table_names:
            connection.execute('CREATE TABLE '+name+' (plateID TEXT, rowID TEXT, '
                               'columnID TEXT, fieldID TEXT, object_label INTEGER, value REAL)')
            connection.executemany('INSERT INTO '+name+' VALUES (?,?,?,?,?,?)', rows)
            expected[name] = {'row_count': len(rows), 'rows_sha256': _rows_hash(
                {r[:5]: _row_hash(r) for r in rows})}
    return path, expected


def test_positive_all_four_tables_match_frozen_values(frozen):
    path, expected = frozen
    assert verify_frozen_rows(path, expected) == expected


@pytest.mark.parametrize('sql', [
    'UPDATE cell SET value=999 WHERE object_label=1',
    'UPDATE nucleus SET columnID="c8" WHERE object_label=1',
    'DELETE FROM pathogen WHERE object_label=1',
    'ALTER TABLE cytoplasm ADD COLUMN invented REAL',
])
def test_a_changed_value_identity_row_or_schema_is_rejected(frozen, sql):
    path, expected = frozen
    with sqlite3.connect(path) as connection:
        connection.execute(sql)
    with pytest.raises(ValueError, match='frozen source rows changed'):
        verify_frozen_rows(path, expected)


def test_duplicate_identity_is_not_hidden_by_dictionary_collapse(frozen):
    path, expected = frozen
    with sqlite3.connect(path) as connection:
        connection.execute('INSERT INTO cell SELECT * FROM cell WHERE object_label=1')
    with pytest.raises(ValueError, match='Duplicate source object identity'):
        verify_frozen_rows(path, expected)


@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_empty_table_manifest_fails(frozen, missing):
    path, expected = frozen
    if missing:
        del expected['nucleus']
    else:
        expected.clear()
    with pytest.raises(ValueError, match='All four'):
        verify_frozen_rows(path, expected)
