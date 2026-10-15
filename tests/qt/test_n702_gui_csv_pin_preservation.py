"""File-menu CSV imports preserve the exact model evidence recorded by exports."""
import copy
import csv
import json

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spacr import model_zoo as zoo
from spacr.cli import SettingsError
from spacr.qt.screens.app_screen import AppScreen


def payload(model_request='origin-model'):
    return {'nucleus_model_name': model_request, 'segmentation_backend': 'cellpose',
            'n_jobs': 2, 'verbose': True, 'empty_legacy': 'None',
            'model_metadata': {'schema_version': 1, 'settings': {'nucleus_model_name': {
                'requested_value': model_request, 'key': 'private-gui-fixture',
                'version': '001', 'version_status': 'declared', 'filename': 'checkpoint.pt',
                'kind': 'cellpose', 'sha256': '1' * 64}}}}


def write_csv(path, settings, encoding='literal'):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Key', 'Value'])
        writer.writerows((name, json.dumps(value) if name == 'model_metadata' and encoding == 'json' else value)
                         for name, value in settings.items())
    return path


@pytest.fixture(autouse=True)
def no_model_io(monkeypatch):
    for name in ('fetch', 'catalogue', 'shared_catalogue', 'sha256_file', '_capture_model_metadata'):
        monkeypatch.setattr(zoo, name, lambda *a, **k: pytest.fail('CSV import performed model I/O'))


@pytest.mark.parametrize('model_request', ['001', 'true', 'None', 'nan'])
@pytest.mark.parametrize('encoding', ['literal', 'json'])
def test_native_csv_preserves_request_and_opaque_version(tmp_path, model_request, encoding):
    settings = payload(model_request)
    path = write_csv(tmp_path / 'settings.csv', settings, encoding)
    original = path.read_bytes()
    loaded = AppScreen._load_settings_csv(str(path))
    assert loaded['nucleus_model_name'] == model_request
    assert isinstance(loaded['nucleus_model_name'], str)
    assert loaded['model_metadata'] == settings['model_metadata']
    assert loaded['n_jobs'] == 2 and loaded['verbose'] is True
    assert loaded['empty_legacy'] is None
    assert path.read_bytes() == original


@pytest.mark.parametrize('problem', ['binding', 'schema_string', 'filename'])
def test_malformed_pins_refuse_before_replacing_the_real_form(qtbot, tmp_path, monkeypatch, problem):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    before = copy.deepcopy(screen._settings_model.collect())
    settings = payload()
    if problem == 'binding':
        settings['nucleus_model_name'] = 'changed'
    elif problem == 'schema_string':
        settings['model_metadata']['schema_version'] = '1'
    else:
        settings['model_metadata']['settings']['nucleus_model_name']['filename'] = 'folder/'
    path = write_csv(tmp_path / 'invalid.csv', settings)
    original = path.read_bytes()
    messages = []
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))
    screen._on_import_settings()
    assert len(messages) == 1 and 'model_metadata' in messages[0]
    assert screen._settings_model.collect() == before
    assert path.read_bytes() == original


def test_real_file_menu_import_keeps_model_evidence(qtbot, tmp_path, monkeypatch):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    settings = payload('001')
    path = write_csv(tmp_path / 'settings.csv', settings)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: pytest.fail(str(a[2])))
    screen._on_import_settings()
    loaded = screen._settings_model.collect()
    assert loaded['nucleus_model_name'] == '001'
    assert loaded['model_metadata'] == settings['model_metadata']


def test_duplicate_pinned_request_is_rejected_without_changing_input(tmp_path):
    path = write_csv(tmp_path / 'duplicates.csv', payload())
    with path.open('a', newline='') as stream:
        csv.writer(stream).writerow(['nucleus_model_name', 'other'])
    original = path.read_bytes()
    with pytest.raises(SettingsError):
        AppScreen._load_settings_csv(str(path))
    assert path.read_bytes() == original


def test_metadata_import_stays_torch_free_in_a_fresh_process(tmp_path):
    import subprocess
    import sys

    path = write_csv(tmp_path / 'settings.csv', payload('001'))
    code = """import sys
from spacr.qt.screens.app_screen import AppScreen
values = AppScreen._load_settings_csv(sys.argv[1])
assert values['model_metadata']['settings']['nucleus_model_name']['version'] == '001'
assert values['nucleus_model_name'] == '001'
assert 'spacr.utils' not in sys.modules
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
