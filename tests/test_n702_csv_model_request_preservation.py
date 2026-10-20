"""Pinned request cells and nested opaque metadata survive real CSV readers."""
from copy import deepcopy
import csv
import json
import os

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    for name in ('fetch', 'catalogue', 'shared_catalogue', 'sha256_file', '_capture_model_metadata'):
        monkeypatch.setattr(zoo, name, lambda *a, **k: pytest.fail('reader performed model I/O/enrichment'))
    return {'base_model': 'origin-model', 'count': 2, 'enabled': True,
            'model_metadata': {'schema_version': 1, 'settings': {'base_model': {
                'requested_value': 'origin-model', 'key': 'private-reader-fixture',
                'version': '001', 'version_status': 'declared', 'filename': 'checkpoint.pt',
                'kind': 'cellpose', 'sha256': '1' * 64}}}}


def write_csv(path, rows, header=('Key', 'Value')):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def read(reader, path, **kwargs):
    if reader == 'cli':
        from spacr.cli import load_settings_file
        return load_settings_file(str(path))
    from spacr.utils import load_settings
    return load_settings(str(path), **kwargs)


def error_type(reader):
    if reader == 'cli':
        from spacr.cli import SettingsError
        return SettingsError
    return zoo.ModelZooError


@pytest.mark.parametrize('reader', ['cli', 'utils'])
@pytest.mark.parametrize('model_request', ['001', 'true', 'None', 'nan'])
def test_pinned_raw_request_and_opaque_version_survive(reader, model_request, settings, tmp_path):
    settings['base_model'] = model_request
    settings['model_metadata']['settings']['base_model']['requested_value'] = model_request
    path = write_csv(tmp_path / 'settings.csv', settings.items())
    original = path.read_bytes()
    restored = read(reader, path)
    assert restored['base_model'] == model_request and isinstance(restored['base_model'], str)
    assert restored['model_metadata'] == settings['model_metadata']
    assert restored['count'] == 2 and restored['enabled'] is True
    assert path.read_bytes() == original


@pytest.mark.parametrize('reader', ['cli', 'utils'])
def test_json_encoded_csv_metadata_does_not_coerce_unspecified_null_version(reader, settings, tmp_path):
    record = settings['model_metadata']['settings']['base_model']
    record['version_status'] = 'unspecified'
    record['version'] = None
    rows = [(key, json.dumps(value) if key == 'model_metadata' else value) for key, value in settings.items()]
    path = write_csv(tmp_path / 'settings.csv', rows)
    restored = read(reader, path)
    assert restored['model_metadata'] == settings['model_metadata']


@pytest.mark.parametrize('reader', ['cli', 'utils'])
@pytest.mark.parametrize('problem', ['schema_string', 'binding', 'filename', 'version_bool', 'settings_list', 'syntax'])
def test_malformed_supplied_metadata_is_not_repaired_or_accepted(reader, problem, settings, tmp_path):
    metadata = settings['model_metadata']
    if problem == 'schema_string':
        metadata['schema_version'] = '1'
    elif problem == 'binding':
        metadata['settings']['base_model']['requested_value'] = 'different-request'
    elif problem == 'filename':
        metadata['settings']['base_model']['filename'] = '.'
    elif problem == 'version_bool':
        metadata['settings']['base_model']['version'] = True
    elif problem == 'settings_list':
        metadata['settings'] = [1, 2]
    else:
        settings['model_metadata'] = '{'
    path = write_csv(tmp_path / 'settings.csv', settings.items())
    original = path.read_bytes()
    with pytest.raises(error_type(reader)):
        read(reader, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('reader', ['cli', 'utils'])
@pytest.mark.parametrize('duplicate', ['request', 'metadata'])
def test_conflicting_pinned_rows_refuse_instead_of_last_value_wins(reader, duplicate, settings, tmp_path):
    rows = list(settings.items())
    if duplicate == 'request':
        rows.insert(0, ('base_model', 'different-earlier-request'))
    else:
        earlier = deepcopy(settings['model_metadata'])
        earlier['settings']['base_model']['key'] = 'different-earlier-identity'
        rows.insert(0, ('model_metadata', earlier))
    path = write_csv(tmp_path / 'settings.csv', rows)
    original = path.read_bytes()
    with pytest.raises(error_type(reader)):
        read(reader, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('reader', ['cli', 'utils'])
def test_identical_pin_duplicates_and_unrelated_legacy_duplicates_stay_compatible(reader, settings, tmp_path):
    rows = list(settings.items()) + [('base_model', settings['base_model']), ('count', 3)]
    restored = read(reader, write_csv(tmp_path / 'settings.csv', rows))
    assert restored['count'] == 3 and restored['base_model'] == 'origin-model'
    assert restored['model_metadata'] == settings['model_metadata']


@pytest.mark.parametrize('reader', ['cli', 'utils'])
def test_unpinned_legacy_values_keep_existing_coercion(reader, tmp_path):
    restored = read(reader, write_csv(tmp_path / 'settings.csv', [('base_model', '001'), ('count', '2'), ('enabled', 'true')]))
    assert restored == {'base_model': 1, 'count': 2, 'enabled': True}


def test_utils_explicit_custom_headers_preserve_pinned_values(settings, tmp_path):
    settings['base_model'] = '001'
    settings['model_metadata']['settings']['base_model']['requested_value'] = '001'
    path = write_csv(tmp_path / 'settings.csv', settings.items(), header=('custom_key', 'custom_value'))
    restored = read('utils', path, setting_key='custom_key', setting_value='custom_value')
    assert restored['base_model'] == '001'
    assert restored['model_metadata'] == settings['model_metadata']


@pytest.mark.parametrize('reader', ['cli', 'utils'])
def test_real_pipeline_writer_to_reader_roundtrip_preserves_pins(reader, settings, tmp_path):
    from spacr.utils import save_settings
    settings['src'] = str(tmp_path)
    settings['base_model'] = '001'
    settings['model_metadata']['settings']['base_model']['requested_value'] = '001'
    original = deepcopy(settings)
    save_settings(settings, name='pipeline')
    path = tmp_path / 'settings/pipeline.csv'
    restored = read(reader, path)
    assert restored['base_model'] == '001'
    assert restored['model_metadata'] == original['model_metadata']
    assert settings == original


def test_utils_reader_never_mixes_replaced_file_pins_with_previous_settings(settings, tmp_path, monkeypatch):
    from spacr import tabular
    path = write_csv(tmp_path / 'settings.csv', settings.items())
    replacement = deepcopy(settings)
    replacement['base_model'] = 'new-origin-model'
    replacement['count'] = 3
    replacement['model_metadata']['settings']['base_model']['requested_value'] = 'new-origin-model'
    pending = write_csv(tmp_path / 'pending.csv', replacement.items())
    replacement_bytes = pending.read_bytes()
    actual_read = tabular.read_table
    calls = []

    def change_after_first_read(*args, **kwargs):
        frame = actual_read(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            os.replace(pending, path)
        return frame

    monkeypatch.setattr(tabular, 'read_table', change_after_first_read)
    try:
        restored = read('utils', path)
    except zoo.ModelZooError:
        restored = None
    if restored is not None:
        assert (restored['base_model'], restored['count']) in {('origin-model', 2), ('new-origin-model', 3)}
    assert path.read_bytes() == replacement_bytes


def test_utils_reader_refuses_pins_when_the_file_vanishes_between_reads(settings, tmp_path, monkeypatch):
    from spacr import tabular
    path = write_csv(tmp_path / 'settings.csv', settings.items())
    actual_read = tabular.read_table
    calls = []

    def delete_after_second_read(*args, **kwargs):
        frame = actual_read(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            os.remove(path)
        return frame

    monkeypatch.setattr(tabular, 'read_table', delete_after_second_read)
    with pytest.raises(zoo.ModelZooError, match='unchanged input'):
        read('utils', path)
    assert len(calls) == 2


class _UnstattableOs:
    """``os`` as :func:`spacr.utils.load_settings` sees it, for a source with no file identity."""

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def stat(path):
        raise ValueError(f'{path!r} has no local file identity')


def test_utils_reader_checks_pins_of_a_source_without_a_file_identity(settings, tmp_path, monkeypatch):
    from spacr import utils
    path = write_csv(tmp_path / 'settings.csv', settings.items())
    monkeypatch.setattr(utils, 'os', _UnstattableOs())
    restored = read('utils', path)
    assert restored['base_model'] == 'origin-model'
    assert restored['model_metadata'] == settings['model_metadata']
