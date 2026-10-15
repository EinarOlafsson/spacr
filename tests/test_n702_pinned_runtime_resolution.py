"""Recorded model bytes take precedence over current filenames and fallback rules."""
from copy import deepcopy
import hashlib

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'exact recorded checkpoint bytes'
    raw = {'key': 'synthetic-recorded', 'name': 'new-published-name.CP_model',
           'kind': 'cellpose', 'sha256': hashlib.sha256(payload).hexdigest(),
           'version': '001', 'uri': 'https://trusted.invalid/checkpoint'}
    request = str(tmp_path / 'old-laptop-name.CP_model')
    record = zoo._model_reference_from_record(request, raw)
    settings = {'cell_model_name': request, 'model_metadata': {
        'schema_version': 1, 'settings': {'cell_model_name': record}}}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin used provider discovery'))
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'home')
    return settings, raw, payload


def test_matching_supplied_bytes_work_without_catalogue_or_acquisition(pinned, monkeypatch):
    from pathlib import Path
    settings, _raw, payload = pinned
    local = Path(settings['cell_model_name'])
    local.write_bytes(payload)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('local pin fetched'))
    original = deepcopy(settings)
    assert zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('cellpose',)) == local
    assert settings == original


def test_wrong_supplied_bytes_are_preserved_and_never_replaced(pinned, monkeypatch):
    from pathlib import Path
    settings, raw, _payload = pinned
    local = Path(settings['cell_model_name'])
    local.write_bytes(b'wrong original file')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('wrong local pin fetched replacement'))
    with pytest.raises(zoo.ModelZooError) as error:
        zoo._resolve_pinned_model_file(settings, 'cell_model_name')
    assert raw['sha256'] in str(error.value)
    assert local.read_bytes() == b'wrong original file'


def test_current_catalogue_contradiction_refuses_even_matching_local_bytes(pinned):
    from pathlib import Path
    settings, raw, payload = pinned
    Path(settings['cell_model_name']).write_bytes(payload)
    raw['version'] = '1'
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name')


def test_recorded_identity_wins_over_changed_laptop_basename(pinned, monkeypatch):
    from pathlib import Path
    settings, raw, payload = pinned
    original = deepcopy(settings)
    calls = []
    def fetch(entry, destination):
        calls.append(entry)
        target = Path(destination) / entry.name
        target.write_bytes(payload)
        return target
    monkeypatch.setattr(zoo, 'fetch', fetch)
    resolved = zoo._resolve_pinned_model_file(settings, 'cell_model_name')
    assert resolved.read_bytes() == payload
    assert resolved.name == raw['name']
    assert [(entry.key, entry.version, entry.sha256) for entry in calls] == [(raw['key'], '001', raw['sha256'])]
    assert settings == original


def test_planning_returns_exact_cache_destination_without_fetching(pinned, monkeypatch):
    settings, raw, _payload = pinned
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('planning fetched weights'))
    original = deepcopy(settings)
    resolved = zoo._resolve_pinned_model_file(settings, 'cell_model_name', download=False)
    assert resolved.name == raw['name']
    assert not resolved.exists()
    assert settings == original


def test_matching_cached_bytes_reused_without_fetch(pinned, monkeypatch):
    settings, raw, payload = pinned
    target = zoo._spacr_home() / 'models' / raw['name']
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('cache pin fetched'))
    assert zoo._resolve_pinned_model_file(settings, 'cell_model_name') == target


def test_wrong_acquired_bytes_never_reach_consumer(pinned, monkeypatch):
    from pathlib import Path
    settings, _raw, _payload = pinned
    def wrong_fetch(entry, destination):
        target = Path(destination) / entry.name
        target.write_bytes(b'bad fetcher result')
        return target
    monkeypatch.setattr(zoo, 'fetch', wrong_fetch)
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name')


def test_unpinned_field_retains_existing_consumer_semantics(pinned):
    settings, _raw, _payload = pinned
    assert zoo._resolve_pinned_model_file(settings, 'absent_model') is None


def test_pin_kind_and_untrusted_missing_identity_refuse_acquisition(pinned, monkeypatch):
    settings, _raw, _payload = pinned
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('invalid pin fetched'))
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('classifier',))
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name')


@pytest.mark.parametrize('flag', ['--force', '--no-preflight'])
def test_actual_cli_refuses_wrong_pinned_bytes_before_pipeline_import(pinned, tmp_path, monkeypatch, flag):
    import json
    import logging
    from pathlib import Path
    from spacr import cli
    settings, _raw, _payload = pinned
    Path(settings['cell_model_name']).write_bytes(b'incorrect supplied bytes')
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    module = cli.Module('mask', 'fixture', 'spacr.fake:run', None, 'mask')
    monkeypatch.setattr(cli, 'resolve_module', lambda _name: module)
    monkeypatch.setattr(cli, 'module_defaults', lambda _module: {'segmentation_backend': 'cellpose'})
    monkeypatch.setattr(cli, 'setup_logging', lambda *a: logging.getLogger('n702'))
    monkeypatch.setattr(cli, '_preflight', lambda *a, **k: [])
    monkeypatch.setattr(cli, 'import_entry', lambda *a: pytest.fail('wrong bytes reached pipeline import'))
    assert cli.main(['mask', '--settings', str(path), flag]) == cli.EXIT_USAGE


def test_actual_cli_dry_run_never_acquires_pinned_bytes(pinned, tmp_path, monkeypatch):
    import json
    import logging
    from spacr import cli
    settings, _raw, _payload = pinned
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    module = cli.Module('mask', 'fixture', 'spacr.fake:run', None, 'mask')
    monkeypatch.setattr(cli, 'resolve_module', lambda _name: module)
    monkeypatch.setattr(cli, 'module_defaults', lambda _module: {'segmentation_backend': 'cellpose'})
    monkeypatch.setattr(cli, 'setup_logging', lambda *a: logging.getLogger('n702'))
    monkeypatch.setattr(cli, '_preflight', lambda *a, **k: [])
    monkeypatch.setattr(cli, 'import_entry', lambda *a: pytest.fail('dry run imported pipeline'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('dry run acquired weights'))
    assert cli.main(['mask', '--settings', str(path), '--dry-run']) == cli.EXIT_OK


def test_actual_cli_records_verified_path_but_keeps_original_settings(pinned, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import json
    import logging
    from spacr import cli, run_journal
    settings, _raw, payload = pinned
    original = deepcopy(settings)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    module = cli.Module('mask', 'fixture', 'spacr.fake:run', None, 'mask')
    events = []
    monkeypatch.setattr(cli, 'resolve_module', lambda _name: module)
    monkeypatch.setattr(cli, 'module_defaults', lambda _module: {'segmentation_backend': 'cellpose'})
    monkeypatch.setattr(cli, 'setup_logging', lambda *a: logging.getLogger('n702'))
    monkeypatch.setattr(cli, 'use_agg_if_headless', lambda: False)
    def import_entry(_module):
        events.append('import')
        return lambda supplied: events.append(('pipeline', deepcopy(supplied)))
    monkeypatch.setattr(cli, 'import_entry', import_entry)
    @contextmanager
    def open_run(_module, supplied):
        events.append(('original', deepcopy(supplied)))
        def record(name, actual):
            assert actual.read_bytes() == payload
            events.append(('record', name, actual))
        yield SimpleNamespace(dir=tmp_path, record_model=record)
    monkeypatch.setattr(run_journal, 'open_run', open_run)
    def fetch(entry, destination):
        events.append('fetch')
        actual = destination / entry.name
        actual.write_bytes(payload)
        return actual
    monkeypatch.setattr(zoo, 'fetch', fetch)
    assert cli.main(['mask', '--settings', str(path), '--no-preflight']) == cli.EXIT_OK
    assert events[0:2] == ['fetch', 'import']
    assert events[2][0] == 'original'
    assert events[3][0:2] == ('record', 'cell_model_name')
    assert events[4][0] == 'pipeline'
    for event in (events[2], events[4]):
        assert event[1]['cell_model_name'] == original['cell_model_name']
        assert event[1]['model_metadata'] == original['model_metadata']
    assert settings == original


def test_planning_preserves_wrong_cache_bytes_and_plans_an_unused_destination(pinned, monkeypatch):
    settings, raw, _payload = pinned
    target = zoo._spacr_home() / 'models' / raw['name']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old unrelated cache bytes')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('planning fetched replacement'))
    planned = zoo._resolve_pinned_model_file(settings, 'cell_model_name', download=False)
    assert planned.parent == target.parent
    assert planned != target
    assert not planned.exists()
    assert target.read_bytes() == b'old unrelated cache bytes'


def test_actual_file_mirror_acquisition_preserves_wrong_cache_bytes(pinned, tmp_path):
    settings, raw, payload = pinned
    source = tmp_path / 'private-mirror.CP_model'
    source.write_bytes(payload)
    raw['uri'] = source.as_uri()
    target = zoo._spacr_home() / 'models' / raw['name']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old unrelated cache bytes')
    original = deepcopy(settings)
    resolved = zoo._resolve_pinned_model_file(settings, 'cell_model_name')
    assert resolved != target
    assert resolved.read_bytes() == payload
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == raw['sha256']
    assert target.read_bytes() == b'old unrelated cache bytes'
    assert settings == original


def test_planning_still_refuses_wrong_explicitly_supplied_local_bytes(pinned):
    from pathlib import Path
    settings, _raw, _payload = pinned
    original = Path(settings['cell_model_name'])
    original.write_bytes(b'incorrect explicit input')
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name', download=False)
    assert original.read_bytes() == b'incorrect explicit input'


def test_recorded_identity_reuses_relocated_cache_while_offline(pinned, monkeypatch, tmp_path):
    from pathlib import Path
    settings, raw, payload = pinned
    original = deepcopy(settings)
    destination = tmp_path / 'relocated-models'
    destination.mkdir()
    expected = destination / raw['name']
    expected.write_bytes(payload)
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(destination))
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('verified relocated pin fetched'))
    assert zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('cellpose',)) == expected
    assert settings == original
    assert not (tmp_path / 'home' / 'models').exists()
