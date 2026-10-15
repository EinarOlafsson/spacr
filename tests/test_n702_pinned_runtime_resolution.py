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
