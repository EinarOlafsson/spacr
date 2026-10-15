"""Relocation metadata may configure only the cache manager's known variables."""
import json
import os

import pytest

from spacr import run_journal as rj


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    path = tmp_path / 'private-cache-locations.json'
    monkeypatch.setattr(rj, '_cache_locations_path', lambda: path)
    for _key, _label, variable in rj._CACHE_ROWS:
        if variable:
            monkeypatch.setenv(variable, '')
    monkeypatch.setenv('SPACR_TEST_UNRELATED_FIELD', '')
    monkeypatch.delenv('SPACR_TEST_UNRELATED_FIELD')
    return path


def test_unknown_metadata_field_is_not_installed_as_an_environment_variable(metadata, tmp_path):
    values = {'SPACR_NEWS_CACHE': str(tmp_path / 'news'),
              'SPACR_TEST_UNRELATED_FIELD': str(tmp_path / 'unrelated')}
    metadata.write_text(json.dumps(values))
    original = metadata.read_bytes()
    assert rj._apply_cache_locations() == 1
    assert os.environ['SPACR_NEWS_CACHE'] == values['SPACR_NEWS_CACHE']
    assert 'SPACR_TEST_UNRELATED_FIELD' not in os.environ
    assert metadata.read_bytes() == original
    assert rj._apply_cache_locations() == 0


def test_only_unknown_metadata_is_preserved_but_has_no_runtime_effect(metadata, tmp_path):
    metadata.write_text(json.dumps({'SPACR_TEST_UNRELATED_FIELD': str(tmp_path / 'opaque')}))
    original = metadata.read_bytes()
    assert rj._apply_cache_locations() == 0
    assert 'SPACR_TEST_UNRELATED_FIELD' not in os.environ
    assert metadata.read_bytes() == original


@pytest.mark.parametrize('variable', [value for _key, _label, value in rj._CACHE_ROWS if value])
def test_each_supported_cache_variable_still_applies(metadata, tmp_path, variable):
    target = str(tmp_path / 'relocated-cache')
    metadata.write_text(json.dumps({variable: target}))
    assert rj._apply_cache_locations() == 1
    assert os.environ[variable] == target


def test_external_environment_value_wins_over_recorded_cache_location(metadata, tmp_path, monkeypatch):
    metadata.write_text(json.dumps({'TORCH_HOME': str(tmp_path / 'recorded')}))
    monkeypatch.setenv('TORCH_HOME', str(tmp_path / 'externally-selected'))
    original = metadata.read_bytes()
    assert rj._apply_cache_locations() == 0
    assert os.environ['TORCH_HOME'] == str(tmp_path / 'externally-selected')
    assert metadata.read_bytes() == original
