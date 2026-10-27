"""Incomplete chosen identities differ from repeated complete observation IDs."""
import copy
import hashlib
import sqlite3

import pandas as pd
import pytest

from spacr.derived_tables import default_definition, execute
from spacr.merge_tables import MergeError, merge_tables


def _database(tmp_path, extra=None):
    path = tmp_path / 'measurements.db'
    cell = pd.DataFrame({'plateID': ['p1', 'p1'], 'rowID': ['r1', 'r1'],
                         'columnID': ['c1', 'c1'], 'fieldID': ['f1', 'f2'],
                         'object_label': [1, 1], 'area': [100., 200.]})
    if extra:
        cell['fieldID'] = 'f1'
        cell[extra] = [1, 2] if extra.startswith('time') else ['image_a', 'image_b']
    nucleus = cell.iloc[[0, 0, 1]].reset_index(drop=True).assign(
        object_label=[7, 8, 7], cell_id=[1, 1, 1], area=[2., 3., 11.])
    with sqlite3.connect(path) as db:
        cell.to_sql('cell', db, index=False)
        nucleus.to_sql('nucleus', db, index=False)
    return path, cell


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('extra', [None, 'file_name', 'time_id', 'timeID'])
def test_default_merge_uses_only_available_identity_without_cross_image_mixing(tmp_path, extra):
    path, _ = _database(tmp_path, extra)
    before = _digest(path)
    definition = default_definition(path, ['cell', 'nucleus'])
    merged, diagnostics = execute(path, definition)
    direct = merge_tables(path, ['cell', 'nucleus'])
    assert len(merged) == len(direct) == 2
    assert merged.nucleus_area.tolist() == direct.nucleus_area.tolist() == [5., 11.]
    if extra is None:
        assert not any('time' in key.lower() for key in definition['base_keys'])
        assert not any('time' in key.lower() for key in merged.columns)
    assert diagnostics['image_provenance'] is (extra != 'file_name')
    assert _digest(path) == before


@pytest.mark.parametrize('extra, expected', [(None, 'fieldID'), ('file_name', 'file_name'),
                                            ('time_id', 'time_id')])
def test_incomplete_custom_keys_name_existing_discriminator_and_can_be_corrected(tmp_path, extra, expected):
    path, _ = _database(tmp_path, extra)
    definition = default_definition(path, ['cell', 'nucleus'])
    bad = copy.deepcopy(definition)
    bad.update(mode='custom', acknowledged=True, base_keys=['object_label'])
    before = _digest(path)
    with pytest.raises(MergeError) as error:
        execute(path, bad)
    message = str(error.value)
    assert 'Add these existing identity columns' in message and expected in message
    assert 'full available image/object identity' not in message
    if extra != 'time_id':
        assert 'time' not in message.lower()
    bad['base_keys'] = definition['base_keys']
    merged, _ = execute(path, bad)
    assert len(merged) == 2 and merged.nucleus_area.tolist() == [5., 11.]
    assert _digest(path) == before


@pytest.mark.parametrize('route', ['derived', 'direct'])
def test_repeated_complete_non_timelapse_ids_are_refused_not_aggregated(tmp_path, route):
    path, cell = _database(tmp_path)
    with sqlite3.connect(path) as db:
        cell.to_sql('cell', db, index=False, if_exists='append')
    before = _digest(path)
    with pytest.raises(MergeError) as error:
        if route == 'derived':
            execute(path, default_definition(path, ['cell', 'nucleus']))
        else:
            merge_tables(path, ['cell', 'nucleus'])
    message = str(error.value)
    assert '4 rows have duplicate observation keys' in message
    assert 'full available image/object identity' in message
    assert 'measurement-run history' in message and 'Add these existing' not in message
    assert 'time' not in message.lower()
    assert _digest(path) == before


def test_mixed_incomplete_keys_and_true_repeats_report_both(tmp_path):
    path, cell = _database(tmp_path)
    with sqlite3.connect(path) as db:
        cell.iloc[:1].to_sql('cell', db, index=False, if_exists='append')
    definition = default_definition(path, ['cell', 'nucleus'])
    definition.update(mode='custom', acknowledged=True, base_keys=['object_label'])
    with pytest.raises(MergeError) as error:
        execute(path, definition)
    assert 'fieldID' in str(error.value)
    assert 'Even with those columns, 2 rows repeat complete observation identities' in str(error.value)
