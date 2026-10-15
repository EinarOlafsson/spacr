"""An unavailable origin user home cannot hide an exact verified model identity."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import uuid

import pytest

from spacr import model_zoo as zoo
from tests.test_n702_pinned_runtime_resolution import pinned
from tests.test_n702_missing_laptop_model_path import checkpoint

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='native POSIX named-user home expansion contract')


@pytest.fixture
def foreign(monkeypatch):
    request = f'~spacr_missing_{uuid.uuid4().hex}/origin-models/'
    with pytest.raises(RuntimeError):
        Path(request + 'checkpoint.pt').expanduser()
    monkeypatch.setenv('SPACR_MODEL_CACHE', '')
    return request


@pytest.mark.parametrize('download', [False, True])
def test_pinned_identity_survives_unavailable_original_user_home(pinned, foreign, tmp_path, monkeypatch, download):
    settings, raw, payload = pinned
    request = foreign + 'old-laptop-name.CP_model'
    settings['cell_model_name'] = request
    settings['model_metadata']['settings']['cell_model_name']['requested_value'] = request
    mirror = tmp_path / 'mirror.CP_model'
    mirror.write_bytes(payload)
    raw['uri'] = mirror.as_uri()
    original = deepcopy(settings)
    if not download:
        monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('planning acquired bytes'))
    result = zoo._resolve_pinned_model_file(settings, 'cell_model_name', download=download)
    assert result.name == raw['name']
    assert result.exists() == download
    if download:
        assert result.read_bytes() == payload
    assert settings == original


@pytest.mark.parametrize('download', [False, True])
def test_legacy_exact_basename_survives_unavailable_origin_home(checkpoint, foreign, monkeypatch, download):
    entry, payload = checkpoint
    if not download:
        monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('planning discovered providers'))
        monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('planning acquired bytes'))
    result = zoo._ensure_model_file(foreign + entry.name, download=download)
    assert result.name == entry.name
    assert result.exists() == download
    if download:
        assert result.read_bytes() == payload


def test_unavailable_origin_home_never_accepts_unknown_legacy_basename(checkpoint, foreign, monkeypatch):
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('unknown model fetched'))
    assert zoo._ensure_model_file(foreign + 'cells_r6.CP_model') is None


def test_unavailable_origin_home_never_chooses_ambiguous_legacy_identity(checkpoint, foreign, monkeypatch):
    entry, _payload = checkpoint
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry, replace(entry, key='other', sha256='f' * 64)])
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('ambiguous model fetched'))
    with pytest.raises(zoo.ModelZooError):
        zoo._ensure_model_file(foreign + entry.name)


def test_untrusted_pin_still_refuses_when_original_user_home_is_unavailable(pinned, foreign, monkeypatch):
    settings, _raw, _payload = pinned
    request = foreign + 'old.CP_model'
    settings['cell_model_name'] = request
    settings['model_metadata']['settings']['cell_model_name']['requested_value'] = request
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('unknown pin fetched'))
    with pytest.raises(zoo.ModelZooError):
        zoo._resolve_pinned_model_file(settings, 'cell_model_name')
