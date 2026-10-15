"""Legacy missing-model planning uses only bounded trusted declarations."""
from pathlib import Path

import pytest

from spacr import model_zoo as zoo


def test_missing_legacy_planning_never_enters_network_or_provider_discovery(tmp_path, monkeypatch):
    from spacr import plugins
    import requests
    import urllib.request
    raw = {'key': 'legacy-plan', 'name': 'published-plan.CP_model', 'kind': 'cellpose',
           'sha256': '7' * 64, 'uri': 'https://trusted.invalid/weights'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    def forbidden(*args, **kwargs):
        pytest.fail('planning attempted network or provider discovery')
    monkeypatch.setattr(zoo, 'catalogue', forbidden)
    monkeypatch.setattr(zoo, 'shared_catalogue', forbidden)
    monkeypatch.setattr(plugins, 'model_providers', forbidden)
    monkeypatch.setattr(plugins, 'load_object', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(requests, 'get', forbidden)
    monkeypatch.setattr(zoo, 'fetch', forbidden)
    assert zoo._ensure_model_file('/old/laptop/' + raw['name'], download=False) == (
        tmp_path / 'cache/models' / raw['name'])
    assert zoo._ensure_model_file('unknown-name', download=False) is None


def test_explicit_existing_custom_file_still_precedes_all_planning(tmp_path, monkeypatch):
    local = tmp_path / 'custom.pt'
    local.write_bytes(b'custom user file')
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('local path discovered providers'))
    assert zoo._ensure_model_file(str(local), download=False) == local
    assert local.read_bytes() == b'custom user file'
