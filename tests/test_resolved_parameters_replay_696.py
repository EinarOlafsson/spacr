"""Saved provenance distinguishes native None, exact estimates and fresh runs."""

import json
import pickle

import numpy as np
import pytest

from spacr import run_journal as journal
from spacr.settings import RESOLVED_PARAMETERS_KEY as KEY


def test_runtime_mode_survives_workers_but_saved_files_always_replay(tmp_path, monkeypatch):
    """Exact floats and native None survive both snapshots without permitting refits."""
    from spacr.utils import load_settings

    settings = {'src': str(tmp_path), 'nucleus_diameter': 'auto'}
    source = journal._resolved_source(np.arange(24, dtype=np.uint16).reshape(4, 6))
    configuration = {'model': 'test-cpu', 'threshold': 0.4}
    assert journal._resolved_parameter(settings, 'plate1_A01_1.npy',
                                       'diameter:nucleus', source, configuration) == (False, None)
    journal._record_resolved_parameter(settings, 'plate1_A01_1', 'diameter:nucleus',
                                       source, configuration, 26.2450135792468)
    journal._record_resolved_parameter(settings, 'plate1_A01_2', 'diameter:nucleus',
                                       source, configuration, None)
    worker = pickle.loads(pickle.dumps(settings))
    assert not journal._resolved_parameters(worker).replay
    assert journal._resolved_parameter(worker, 'plate1_A01_3', 'diameter:nucleus',
                                       source, configuration) == (False, None)
    run = journal.Run('mask', {}, tmp_path)
    monkeypatch.setattr(journal, 'current_run', lambda: run)
    journal._publish_resolved_settings(settings)
    for restored in (json.loads((tmp_path / 'settings.json').read_text()),
                     load_settings(str(tmp_path / 'settings.csv'))):
        assert journal._resolved_parameters(restored).replay
        assert journal._resolved_parameter(restored, 'plate1_A01_1', 'diameter:nucleus',
                                           source, configuration) == (True, 26.2450135792468)
        assert journal._resolved_parameter(restored, 'plate1_A01_2', 'diameter:nucleus',
                                           source, configuration) == (True, None)
        with pytest.raises(ValueError, match='no resolved'):
            journal._resolved_parameter(restored, 'plate1_A01_3', 'diameter:nucleus',
                                         source, configuration)
        with pytest.raises(ValueError, match='differs from the saved run'):
            journal._resolved_parameter(restored, 'plate1_A01_1', 'diameter:nucleus',
                                         source, dict(configuration, threshold=0.5))
    before = (tmp_path / 'settings.json').read_bytes()
    run._settings_owner_pid = -1
    journal._publish_resolved_settings(dict(settings, nucleus_diameter=100))
    assert (tmp_path / 'settings.json').read_bytes() == before


@pytest.mark.parametrize('payload', [None, {}, {'version': 2, 'fields': {}},
                                    {'version': 1, 'fields': {'site': {'diameter': {}}}},
                                    {'version': 1, 'fields': {}, 'extra': True}])
def test_invalid_saved_records_refuse_instead_of_becoming_fresh(payload):
    """A malformed saved context never silently enables new estimation."""
    with pytest.raises(ValueError, match='unsupported format'):
        journal._resolved_parameters({KEY: payload})


def test_worker_merge_is_exact_and_conflicts_cannot_overwrite_a_record():
    """Independent worker records merge, but a conflicting value remains an error."""
    parent, worker = {}, {}
    journal._resolved_parameters(parent)
    journal._record_resolved_parameter(worker, 'field', 'diameter:cell', {}, {}, None)
    journal._merge_resolved_parameters(parent, worker[KEY])
    journal._merge_resolved_parameters(parent, worker[KEY])
    assert parent[KEY]['fields']['field']['diameter:cell']['value'] is None
    conflicting = json.loads(json.dumps(worker[KEY]))
    conflicting['fields']['field']['diameter:cell']['value'] = 26.366
    with pytest.raises(ValueError, match='different resolved'):
        journal._merge_resolved_parameters(parent, conflicting)
    assert parent[KEY]['fields']['field']['diameter:cell']['value'] is None


def test_source_identity_binds_pixel_bytes_shape_dtype_and_preserves_field_suffixes():
    """Identical values with another layout or numeric representation remain distinct."""
    values = np.arange(6, dtype=np.uint16).reshape(2, 3)
    source = journal._resolved_source(values)
    assert journal._resolved_source(values.copy()) == source
    assert journal._resolved_source(values.reshape(3, 2)) != source
    assert journal._resolved_source(values.astype(np.uint32)) != source
    assert journal._resolved_field_id('plate1_A01_1_0.npy') == 'plate1_A01_1_0'
    for invalid in (None, '', '.npy', '../field', 'folder/field', ' field'):
        with pytest.raises(ValueError, match='nonempty basenames'):
            journal._resolved_field_id(invalid)
    with pytest.raises(ValueError, match='finite JSON'):
        journal._record_resolved_parameter({}, 'field', 'diameter:cell', {}, {}, float('nan'))
