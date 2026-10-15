"""Actual journal files retain original model request and verified effective bytes."""
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path

import pytest

from spacr import cli, model_zoo as zoo, run_journal


@pytest.mark.parametrize('cached', [False, True])
def test_actual_cli_journal_records_original_pin_and_effective_verified_model(tmp_path, monkeypatch, cached):
    payload = b'exact recorded provenance bytes'
    raw = {'key': 'recorded-provenance', 'name': 'current-model.CP_model',
           'kind': 'cellpose', 'sha256': hashlib.sha256(payload).hexdigest(), 'version': '001',
           'uri': 'https://trusted.invalid/weights'}
    request = str(tmp_path / 'old-laptop-model.CP_model')
    source = tmp_path / 'private-source'
    source.mkdir()
    settings = {'src': str(source), 'cell_model_name': request, 'segmentation_backend': 'cellpose',
                'model_metadata': {'schema_version': 1, 'settings': {
                    'cell_model_name': zoo._model_reference_from_record(request, raw)}}}
    original = deepcopy(settings)
    path = tmp_path / 'input-settings.json'
    path.write_text(json.dumps(settings))
    private_home = tmp_path / 'private-home'
    monkeypatch.setenv('SPACR_HOME', str(private_home))
    monkeypatch.setattr(run_journal, 'runs_root', lambda: tmp_path / 'runs')
    monkeypatch.setattr(zoo, '_spacr_home', lambda: private_home)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin discovered providers'))
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    target = private_home / 'models' / raw['name']
    if cached:
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
    fetched = []
    def fetch(entry, destination):
        fetched.append(entry.key)
        target.write_bytes(payload)
        return target
    monkeypatch.setattr(zoo, 'fetch', fetch)
    module = cli.Module('mask', 'native provenance fixture', 'spacr.fake:run', None, 'mask')
    monkeypatch.setattr(cli, 'resolve_module', lambda _name: module)
    monkeypatch.setattr(cli, 'module_defaults', lambda _module: {})
    monkeypatch.setattr(cli, 'setup_logging', lambda *a: logging.getLogger('n702-native-journal'))
    monkeypatch.setattr(cli, 'use_agg_if_headless', lambda: False)
    observed = []
    def pipeline(supplied):
        assert supplied == original
        current = run_journal.current_run()
        observed.append(current.dir)
        assert current.model_files['cell_model_name']['sha256'] == raw['sha256']
    monkeypatch.setattr(cli, 'import_entry', lambda _module: pipeline)
    assert cli.main(['mask', '--settings', str(path), '--no-preflight']) == cli.EXIT_OK
    assert len(observed) == 1
    directory = observed[0]
    recorded = json.loads((directory / 'settings.json').read_text())
    assert recorded == original
    assert cli.load_settings_file(str(directory / 'settings.csv')) == original
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'success'
    actual = manifest['model_files']['cell_model_name']
    assert Path(actual['path']) == target.resolve()
    assert actual['sha256'] == raw['sha256']
    assert actual['size_bytes'] == len(payload)
    assert raw['sha256'][:16] in manifest['model_hashes']['cell_model_name']
    assert fetched == ([] if cached else [raw['key']])
    assert settings == original
