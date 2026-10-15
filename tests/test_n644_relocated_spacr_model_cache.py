"""Moving spaCR's own model cache is honored by discovery and runtime acquisition."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from spacr import model_zoo as zoo, run_journal as journal


@pytest.fixture
def cache(tmp_path, monkeypatch):
    home = tmp_path / 'private-spacr'
    home.mkdir()
    monkeypatch.setenv('SPACR_HOME', str(home))
    monkeypatch.setenv('SPACR_MODEL_CACHE', '')
    monkeypatch.setattr(journal, '_spacr_home', lambda: home)
    monkeypatch.setattr(zoo, '_spacr_home', lambda: home)
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    data = b'exact relocated model cache bytes'
    mirror = tmp_path / 'mirror.pt'
    mirror.write_bytes(data)
    raw = {'key': 'relocation-model', 'name': 'model-r7.pt', 'kind': 'classifier',
           'sha256': hashlib.sha256(data).hexdigest(), 'uri': mirror.as_uri()}
    entry = zoo._entry_from_mapping(raw)
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    return home, entry, raw, data


def test_model_cache_relocation_is_used_by_exact_runtime_resolution(cache, tmp_path, monkeypatch):
    home, entry, _raw, data = cache
    original = home / 'models'
    original.mkdir()
    (original / entry.name).write_bytes(data)
    relocated = journal._relocate_cache('models', tmp_path / 'other-drive')
    assert not original.exists()
    assert (relocated / entry.name).read_bytes() == data
    assert os.environ['SPACR_MODEL_CACHE'] == str(relocated)
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('moved cache downloaded again'))
    assert zoo._ensure_model_file('/old/laptop/' + entry.name, kinds=('classifier',)) == relocated / entry.name
    assert relocated in zoo.default_local_roots()
    assert original not in zoo.default_local_roots()
    assert journal._cache_folder('models') == relocated
    row = next(row for row in journal._disk_caches() if row['key'] == 'models')
    assert row['relocatable'] is True
    assert Path(row['path']) == relocated


def test_new_acquisition_after_relocation_lands_only_in_new_model_cache(cache, tmp_path):
    home, entry, _raw, data = cache
    relocated = journal._relocate_cache('models', tmp_path / 'other-drive')
    resolved = zoo._ensure_model_file(entry.key, kinds=('classifier',))
    assert resolved.parent == relocated
    assert resolved.read_bytes() == data
    assert not (home / 'models').exists()
    assert json.loads((home / 'cache_locations.json').read_text())['SPACR_MODEL_CACHE'] == str(relocated)


def test_configured_model_root_survives_normal_cache_location_reapplication(cache, tmp_path, monkeypatch):
    home, _entry, _raw, _data = cache
    relocated = journal._relocate_cache('models', tmp_path / 'other-drive')
    monkeypatch.setenv('SPACR_MODEL_CACHE', '')
    assert journal._apply_cache_locations() == 1
    assert zoo._model_cache_root() == relocated
    assert journal._cache_folder('models') == relocated


def test_external_model_cache_override_wins_and_is_not_relocated(cache, tmp_path, monkeypatch):
    _home, _entry, _raw, _data = cache
    external = tmp_path / 'external-cache'
    external.mkdir()
    (external / 'kept.pt').write_bytes(b'external bytes')
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(external))
    assert zoo._model_cache_root() == external
    with pytest.raises(ValueError, match='outside spaCR'):
        journal._relocate_cache('models', tmp_path / 'other-drive')
    assert (external / 'kept.pt').read_bytes() == b'external bytes'


def test_unconfigured_model_cache_keeps_existing_home_default(cache):
    home, _entry, _raw, _data = cache
    assert zoo._model_cache_root() == home / 'models'
    assert journal._cache_folder('models') == home / 'models'


def test_mitonet_default_cache_uses_same_configured_root(cache, tmp_path, monkeypatch):
    _home, _entry, raw, data = cache
    target = tmp_path / 'external-models'
    target.mkdir()
    (target / raw['name']).write_bytes(data)
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(target))
    monkeypatch.setattr(zoo, '_mitonet_record', lambda: raw)
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('moved Mitonet cache downloaded again'))
    assert zoo.mitonet_weights(download=False) == target / raw['name']
