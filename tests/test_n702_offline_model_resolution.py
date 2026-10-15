"""Offline model resolution reuses checked files without metadata or weights traffic."""
import hashlib

import pytest

from spacr import model_zoo as zoo

_DECLARED_CATALOGUE = zoo.catalogue


@pytest.fixture
def offline_model(tmp_path, monkeypatch):
    payload = b'verified offline checkpoint'
    entry = zoo.ModelEntry(key='cells_r7', name='cells_r7.CP_model', kind='cellpose',
                           uri='https://example.invalid/cells_r7.CP_model', version='7',
                           sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    monkeypatch.setattr(zoo, '_planning_model_entries', lambda: [entry])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path)
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('offline weights fetch'))
    return entry, payload


def test_offline_missing_weights_explain_exact_copy_destination(offline_model, tmp_path):
    entry, _payload = offline_model
    with pytest.raises(zoo.ModelUnreadable) as error:
        zoo._ensure_model_file('/laptop/' + entry.name)
    message = str(error.value)
    assert 'SPACR_MODEL_DOWNLOAD' in message
    assert entry.key in message
    assert entry.sha256 in message
    assert str(tmp_path / 'models' / entry.name) in message
    assert not (tmp_path / 'models').exists()


def test_offline_cached_verified_model_remains_usable(offline_model, tmp_path):
    entry, payload = offline_model
    cached = tmp_path / 'models' / entry.name
    cached.parent.mkdir()
    cached.write_bytes(payload)
    assert zoo._ensure_model_file('/laptop/' + entry.name) == cached
    assert cached.read_bytes() == payload


def test_offline_preflight_names_cache_path_without_fetching(offline_model, tmp_path):
    entry, _payload = offline_model
    assert zoo._ensure_model_file('/laptop/' + entry.name, download=False) == tmp_path / 'models' / entry.name


@pytest.mark.parametrize('value', ['off', 'FALSE', '0', ' No '])
@pytest.mark.parametrize('block', [True, False])
def test_offline_catalogue_never_starts_metadata_traffic(tmp_path, monkeypatch, value, block):
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', value)
    sentinel = zoo.ModelEntry(key='cached', name='cached', kind='cellpose')
    monkeypatch.setattr(zoo, '_SHARED_CATALOGUE_CACHE', {'fetched_at': 0, 'entries': (sentinel,)})
    monkeypatch.setattr(zoo, '_refresh_shared_catalogue_in_background',
                        lambda *args: pytest.fail('offline daemon metadata fetch'))
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('offline foreground metadata fetch'))
    assert zoo.shared_catalogue(force=True, block=block) == (sentinel,)


def test_offline_resolver_uses_declared_catalogue_without_metadata(offline_model, monkeypatch):
    entry, _payload = offline_model
    monkeypatch.delenv(zoo.CATALOGUE_ENV_VAR, raising=False)
    for name in ('stock_cellpose_entries', 'installable_backend_entries',
                 '_cellpose3_model_entries', '_prefixed_model_entries', 'bioimageio_entries'):
        monkeypatch.setattr(zoo, name, lambda: [])
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', ({
        'key': entry.key, 'name': entry.name, 'kind': entry.kind,
        'uri': entry.uri, 'version': entry.version, 'sha256': entry.sha256},))
    monkeypatch.setattr(zoo, '_SHARED_CATALOGUE_CACHE', {'fetched_at': 0, 'entries': ()})
    monkeypatch.setattr(zoo, '_refresh_shared_catalogue_in_background',
                        lambda *args: pytest.fail('offline resolver daemon fetch'))
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('offline resolver metadata fetch'))
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: _DECLARED_CATALOGUE(
        include_bundled=False, include_plugins=False))
    with pytest.raises(zoo.ModelUnreadable, match='SPACR_MODEL_DOWNLOAD'):
        zoo._ensure_model_file('/laptop/' + entry.name)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_offline_copy_instructions_use_the_selected_language(offline_model, tmp_path, monkeypatch, language):
    import importlib
    from spacr.qt import i18n
    entry, _payload = offline_model
    source = ('Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. '
              'Copy verified model {model} (sha256 {sha256}) to {path}.')
    target = importlib.import_module('spacr.qt.i18n_catalogs.' + language).UI[source]
    assert target != source
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    with pytest.raises(zoo.ModelUnreadable) as error:
        zoo._ensure_model_file('/laptop/' + entry.name)
    assert str(error.value) == target.format(
        model=entry.key, sha256=entry.sha256, path=str(tmp_path / 'models' / entry.name))
