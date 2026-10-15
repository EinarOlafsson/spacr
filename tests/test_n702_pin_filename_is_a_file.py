"""Recorded checkpoint filenames cannot describe a directory or invalid OS path."""
import copy
import hashlib

import pytest

from spacr import model_zoo as zoo
from spacr.cli import SettingsError, write_settings_file


@pytest.fixture
def pinned(tmp_path):
    path = tmp_path / 'actual-local-model.pt'
    data = b'private native checkpoint fixture'
    path.write_bytes(data)
    return {'base_model': str(path), 'model_metadata': {'schema_version': 1, 'settings': {
        'base_model': {'requested_value': str(path), 'key': 'private-native-fixture',
                       'version': '001', 'version_status': 'declared',
                       'sha256': hashlib.sha256(data).hexdigest(), 'kind': 'cellpose',
                       'filename': 'nested/checkpoint.pt'}}}}


@pytest.mark.parametrize('filename', ['.', 'nested/.', 'nested/', 'nested\\', 'bad\x00name.pt'])
def test_invalid_filename_refused_even_when_requested_local_bytes_match(pinned, filename):
    pinned['model_metadata']['settings']['base_model']['filename'] = filename
    original = copy.deepcopy(pinned)
    with pytest.raises(zoo.ModelZooError):
        zoo._validate_model_metadata(pinned)
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(pinned, 'base_model', kinds=('cellpose',))
    assert pinned == original


@pytest.mark.parametrize('filename', ['.', 'nested/.', 'nested/', 'nested\\', 'bad\x00name.pt'])
def test_invalid_filename_refused_before_existing_settings_outputs_are_opened(pinned, tmp_path, filename):
    pinned['model_metadata']['settings']['base_model']['filename'] = filename
    csv_path = tmp_path / 'settings.csv'
    json_path = tmp_path / 'settings.json'
    csv_path.write_bytes(b'previous CSV settings')
    json_path.write_bytes(b'previous JSON settings')
    with pytest.raises(SettingsError):
        write_settings_file(pinned, csv_path)
    assert csv_path.read_bytes() == b'previous CSV settings'
    assert json_path.read_bytes() == b'previous JSON settings'


@pytest.mark.parametrize('filename', ['checkpoint.pt', 'nested/checkpoint.pt', 'nested\\checkpoint.pt', './checkpoint.pt'])
def test_actual_relative_file_names_and_opaque_version_remain_valid(pinned, filename):
    pinned['model_metadata']['settings']['base_model']['filename'] = filename
    metadata = zoo._validate_model_metadata(pinned)
    assert metadata['settings']['base_model']['version'] == '001'
    actual = zoo._resolve_pinned_model_file(pinned, 'base_model', kinds=('cellpose',))
    assert str(actual) == pinned['base_model']
