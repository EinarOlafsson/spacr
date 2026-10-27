"""Explicit duplicate merging changes only working frames and never identifiers."""
import copy
import hashlib
import sqlite3

import pandas as pd
import pytest

from spacr.derived_tables import default_definition, execute
from spacr.merge_tables import (
    MergeError, _DuplicateObservations, _duplicate_merge_enabled, merge_tables,
)


def _database(tmp_path, base_duplicate=True, child_duplicate=True):
    path = tmp_path / 'measurements.db'
    identity = {'plateID': 'p1', 'rowID': 'r1', 'columnID': 'c1', 'fieldID': 'f1'}
    cells = pd.DataFrame([{**identity, 'object_label': 1, 'area': 10., 'mean_intensity': 2.,
                           'min_intensity': 2., 'max_intensity': 4.}])
    if base_duplicate:
        cells = pd.concat([cells, cells.assign(area=20., mean_intensity=4.,
                                              min_intensity=1., max_intensity=8.)], ignore_index=True)
    children = pd.DataFrame([{**identity, 'object_label': 7, 'cell_id': 1, 'area': 2., 'mean_intensity': 5.},
                            {**identity, 'object_label': 8, 'cell_id': 1, 'area': 7., 'mean_intensity': 11.}])
    if child_duplicate:
        children = pd.concat([children, children.iloc[:1].assign(area=3., mean_intensity=9.)], ignore_index=True)
    with sqlite3.connect(path) as db:
        cells.to_sql('cell', db, index=False)
        children.to_sql('nucleus', db, index=False)
    return path


def _bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_collects_both_base_and_child_repeats_and_refuses_without_mutation(tmp_path):
    path = _database(tmp_path)
    before = _bytes(path)
    with pytest.raises(_DuplicateObservations) as error:
        execute(path, default_definition(path, ['cell', 'nucleus']))
    assert error.value.counts == {'cell': 2, 'nucleus': 2}
    assert 'time' not in str(error.value).lower()
    assert _bytes(path) == before and not _duplicate_merge_enabled.get()


def test_opt_in_uses_per_measurement_aggregation_for_every_source(tmp_path):
    path = _database(tmp_path)
    before = _bytes(path)
    definition = default_definition(path, ['cell', 'nucleus'])
    definition['merge_duplicates'] = True
    frame, _ = execute(path, definition)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.cell_area == 30 and row.cell_mean_intensity == 3
    assert row.cell_min_intensity == 1 and row.cell_max_intensity == 8
    assert row.nucleus_area == 12 and row.nucleus_mean_intensity == 9
    assert row.object_label == 1 and row.nucleus_object_label == 7
    assert frame.attrs['merge_definition']['merge_duplicates'] is True
    assert _bytes(path) == before and not _duplicate_merge_enabled.get()
    with pytest.raises(_DuplicateObservations):
        execute(path, default_definition(path, ['cell', 'nucleus']))


def test_explicit_overrides_and_custom_base_overrides_are_honored(tmp_path):
    path = _database(tmp_path)
    definition = default_definition(path, ['cell', 'nucleus'])
    definition['merge_duplicates'] = True
    definition['policy']['overrides'] = {'cell.area': 'mean'}
    frame, _ = execute(path, definition)
    assert frame.cell_area.tolist() == [15.]
    definition.update(mode='custom', acknowledged=True, base_overrides={'area': 'max'})
    frame, _ = execute(path, definition)
    assert frame.cell_area.tolist() == [20.]


def test_child_repeats_are_detected_when_base_has_no_duplicates(tmp_path):
    path = _database(tmp_path, base_duplicate=False)
    before = _bytes(path)
    with pytest.raises(_DuplicateObservations) as error:
        merge_tables(path, ['cell', 'nucleus'])
    assert error.value.counts == {'nucleus': 2}
    token = _duplicate_merge_enabled.set(True)
    try:
        assert len(merge_tables(path, ['cell', 'nucleus'])) == 1
    finally:
        _duplicate_merge_enabled.reset(token)
    assert _bytes(path) == before


def test_parent_link_conflicts_and_identifier_arithmetic_never_reassign_objects(tmp_path):
    path = _database(tmp_path)
    definition = default_definition(path, ['cell', 'nucleus'])
    definition['merge_duplicates'] = True
    definition['policy']['overrides'] = {'nucleus.cell_id': 'mean'}
    before = _bytes(path)
    with pytest.raises(MergeError, match='preserve identifiers'):
        execute(path, definition)
    assert _bytes(path) == before and not _duplicate_merge_enabled.get()
    definition['policy']['overrides'] = {}
    with sqlite3.connect(path) as db:
        db.execute('UPDATE nucleus SET cell_id=2 WHERE _rowid_=3')
    before = _bytes(path)
    with pytest.raises(MergeError, match='parent link'):
        execute(path, definition)
    assert _bytes(path) == before and not _duplicate_merge_enabled.get()


def test_opt_in_does_not_silently_fix_incomplete_chosen_keys(tmp_path):
    path = _database(tmp_path, base_duplicate=False, child_duplicate=False)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO cell SELECT plateID,rowID,columnID,'f2',object_label,area,mean_intensity,min_intensity,max_intensity FROM cell")
    definition = default_definition(path, ['cell', 'nucleus'])
    definition.update(mode='custom', acknowledged=True, merge_duplicates=True, base_keys=['object_label'])
    with pytest.raises(MergeError, match='fieldID') as error:
        execute(path, definition)
    assert not isinstance(error.value, _DuplicateObservations)


def test_all_missing_duplicate_features_stay_missing_and_zero_remains_zero(tmp_path):
    path = tmp_path / 'measurements.db'
    frame = pd.DataFrame({'plateID': ['p1'] * 4, 'rowID': ['r1'] * 4,
                          'columnID': ['c1'] * 4, 'fieldID': ['f1'] * 4,
                          'object_label': [1, 1, 2, 2], 'area': [None, None, 0., 0.],
                          'flag_any': [None, None, 0., 0.], 'flag_all': [None, None, 0., 0.]})
    with sqlite3.connect(path) as db:
        frame.to_sql('cell', db, index=False)
    definition = default_definition(path, ['cell'])
    before = _bytes(path)
    with pytest.raises(_DuplicateObservations):
        execute(path, definition)
    definition['merge_duplicates'] = True
    definition['base_overrides'] = {'flag_any': 'any', 'flag_all': 'all'}
    result, _ = execute(path, definition)
    assert result.object_label.tolist() == [1, 2]
    assert pd.isna(result.cell_area.iloc[0]) and result.cell_area.iloc[1] == 0
    for column in ('cell_flag_any', 'cell_flag_all'):
        assert str(result[column].dtype) == 'boolean'
        assert pd.isna(result[column].iloc[0]) and result[column].iloc[1] == False
    assert _bytes(path) == before and not _duplicate_merge_enabled.get()


def test_truthy_text_is_not_explicit_duplicate_permission(tmp_path):
    path = _database(tmp_path)
    definition = default_definition(path, ['cell', 'nucleus'])
    definition['merge_duplicates'] = 'False'
    with pytest.raises(_DuplicateObservations):
        execute(path, definition)


def test_explicit_external_image_keys_are_not_replaced_with_partial_standard_ids(tmp_path):
    path = tmp_path / 'external.db'
    with sqlite3.connect(path) as db:
        pd.DataFrame({'image': ['a', 'b'], 'plateID': ['p1', 'p1'],
                      'object_label': [1, 1], 'reading': [10., 20.]}).to_sql('observations', db, index=False)
    definition = default_definition(path, ['observations'], base='observations')
    definition.update(mode='custom', acknowledged=True, merge_duplicates=True,
                      base_keys=['image', 'object_label'])
    frame, _ = execute(path, definition)
    assert len(frame) == 2 and frame.observations_reading.tolist() == [10., 20.]
