"""A missing laptop path resolves only a unique exact zoo identity."""
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    data = b'exact version seven checkpoint'
    source = tmp_path / 'mirror' / 'cells_r7.CP_model'
    source.parent.mkdir()
    source.write_bytes(data)
    entry = zoo.ModelEntry(key='cells_r7', name=source.name, kind='cellpose',
                           uri=source.as_uri(), version='7',
                           sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'node')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    return entry, data


@pytest.mark.parametrize('requested', [
    '/laptop/private/models/cells_r7.CP_model',
    r'C:\Users\person\models\cells_r7.CP_model',
    '~/old/models/cells_r7',
])
def test_missing_laptop_path_downloads_the_exact_basename(checkpoint, requested):
    entry, data = checkpoint
    result = zoo._ensure_model_file(requested, kinds=('cellpose',))
    assert result is not None
    assert result.read_bytes() == data
    assert result.name == entry.name
    assert zoo._ensure_model_file(requested, kinds=('cellpose',)) == result


def test_missing_laptop_path_dry_run_selects_without_fetching(checkpoint, monkeypatch):
    entry, _data = checkpoint
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('dry run fetched bytes'))
    result = zoo._ensure_model_file('/laptop/' + entry.name, download=False)
    assert result.name == entry.name
    assert not result.exists()


def test_ambiguous_basename_never_substitutes_a_version(checkpoint, monkeypatch):
    entry, _data = checkpoint
    other = replace(entry, key='other_r8', version='8', sha256='f' * 64)
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry, other])
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('ambiguous identity fetched'))
    with pytest.raises(zoo.ModelZooError):
        zoo._ensure_model_file('/laptop/' + entry.name)


def test_nearby_model_name_is_never_substituted(checkpoint, monkeypatch):
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('fuzzy identity fetched'))
    assert zoo._ensure_model_file('/laptop/cells_r6.CP_model') is None
    assert zoo._ensure_model_file('/laptop/cells') is None


def test_existing_local_path_wins_without_catalogue_lookup(checkpoint, tmp_path, monkeypatch):
    local = tmp_path / 'cells_r7.CP_model'
    local.write_bytes(b'user local custom weights')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('local path queried catalogue'))
    assert zoo._ensure_model_file(local) == local
    assert local.read_bytes() == b'user local custom weights'


def test_exact_missing_path_still_verifies_downloaded_checksum(checkpoint, tmp_path):
    entry, _data = checkpoint
    Path(entry.uri.removeprefix('file://')).write_bytes(b'corrupt mirror')
    with pytest.raises(zoo.ChecksumMismatch):
        zoo._ensure_model_file('/laptop/' + entry.name)
    assert list((tmp_path / 'node' / 'models').iterdir()) == []


def test_exact_missing_path_still_refuses_the_wrong_kind(checkpoint):
    entry, _data = checkpoint
    with pytest.raises(zoo.ModelZooError, match='requires.*classifier'):
        zoo._ensure_model_file('/laptop/' + entry.name, kinds=('classifier',))


def test_missing_path_does_not_download_on_the_gui_thread(checkpoint, monkeypatch):
    entry, _data = checkpoint
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: True)
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('GUI thread fetched'))
    with pytest.raises(zoo.ModelUnreadable, match='Model Zoo'):
        zoo._ensure_model_file('/laptop/' + entry.name)


def test_explicit_remote_uri_is_not_reinterpreted_as_a_laptop_path(checkpoint, monkeypatch):
    entry, _data = checkpoint
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('URI reinterpreted'))
    assert zoo._ensure_model_file('https://unrelated.example.invalid/' + entry.name) is None
