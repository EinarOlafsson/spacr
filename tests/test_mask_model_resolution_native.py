"""Role resolution reaches Cellpose's real constructor without cross-model reuse."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import model_zoo
from spacr import object as objects
from spacr._mask_workers import _mask_model_contract
from spacr.remote_execution import map_settings_paths
from spacr.settings import _get_object_settings, set_default_settings_preprocess_generate_masks
from spacr.utils import _resolve_cellpose_pretrained


@pytest.fixture
def native_model_loader(monkeypatch):
    from cellpose import models

    records = []

    class Network:
        def __init__(self, **kwargs):
            pass
        def to(self, device):
            return self
        def load_model(self, path, device=None):
            records.append((str(path), Path(path).read_bytes()))

    if hasattr(models, 'Transformer'):
        monkeypatch.setattr(models, 'Transformer', Network)
    else:
        monkeypatch.setattr(models, 'CPSAM', Network)
        monkeypatch.setattr(models, 'CPDINO', Network)
        monkeypatch.setattr(models, 'get_backbone', lambda path: 'sam_vitl')
    monkeypatch.setattr(models, 'assign_device', lambda **kwargs: (SimpleNamespace(type='cpu'), False))
    monkeypatch.setattr(models.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(models.torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(model_zoo, '_ensure_model_file', lambda *args, **kwargs: pytest.fail('unexpected model download'))
    for cache in ('cache_CPSAM_model_path', 'cache_model_path'):
        if hasattr(models, cache):
            monkeypatch.setattr(models, cache, lambda *args: pytest.fail('unexpected stock fallback'))
    return models.CellposeModel, records


def _settings(tmp_path, pinned=False):
    local = tmp_path / 'local'
    payload = dict(cell_channel=3, nucleus_channel=3, pathogen_channel=2, verbose=False)
    pins = {}
    for role in ('cell', 'nucleus', 'pathogen'):
        path = local / role / 'model.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('distinct-' + role + '-weights').encode())
        key = role + '_model_name'
        payload[key] = str(path)
        pins[key] = dict(requested_value=str(path), key='fixture-' + role,
                         version=None, version_status='unspecified', kind='cellpose',
                         filename='model.pth', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if pinned:
        payload['model_metadata'] = dict(schema_version=1, settings=pins)
    return local, set_default_settings_preprocess_generate_masks(payload)


def _resolve_and_construct(settings, role, constructor):
    model = _get_object_settings(role, settings)['model_name']
    key = role + '_model_name'
    if role == 'pathogen' and settings.get('pathogen_model') is not None:
        model, key = settings['pathogen_model'], 'pathogen_model'
    model = objects._pinned_object_model(settings, role, model, 'cellpose')
    resolved = _resolve_cellpose_pretrained(model, object_type=role, setting=key)
    return constructor(gpu=False, pretrained_model=resolved, use_bfloat16=False)


@pytest.mark.parametrize('pinned', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test_native_constructor_loads_each_roles_distinct_bytes_even_with_same_basename(
        tmp_path, native_model_loader, pinned, remote):
    constructor, records = native_model_loader
    local, settings = _settings(tmp_path, pinned)
    if remote:
        remote_root = tmp_path / 'remote'
        for path in local.rglob('model.pth'):
            target = remote_root / path.relative_to(local)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        settings = map_settings_paths(settings, str(local), str(remote_root))
    original = copy.deepcopy(settings)
    for role in ('cell', 'nucleus', 'pathogen'):
        prepared, signature, identity = _mask_model_contract(settings, role)
        assert len(signature) == 64
        prepared = json.loads(json.dumps(prepared))
        model = _resolve_and_construct(prepared, role, constructor)
        assert model.pretrained_model == settings[role + '_model_name'] == identity['path']
    assert [contents for _path, contents in records] == [
        b'distinct-cell-weights', b'distinct-nucleus-weights', b'distinct-pathogen-weights']
    assert len({path for path, _contents in records}) == 3
    assert settings == original


@pytest.mark.parametrize('override', [False, True])
def test_explicit_pathogen_override_is_the_only_route_to_nucleus_weights(
        tmp_path, native_model_loader, override):
    constructor, records = native_model_loader
    _local, settings = _settings(tmp_path)
    if override:
        settings['pathogen_model'] = settings['nucleus_model_name']
    model = _resolve_and_construct(settings, 'pathogen', constructor)
    expected_role = 'nucleus' if override else 'pathogen'
    assert records == [(settings[expected_role + '_model_name'],
                        ('distinct-' + expected_role + '-weights').encode())]
    assert model.pretrained_model == settings[expected_role + '_model_name']


@pytest.mark.parametrize('changed', ['setting', 'bytes'])
def test_stale_pin_is_refused_before_native_constructor_or_fallback(
        tmp_path, native_model_loader, changed):
    constructor, records = native_model_loader
    _local, settings = _settings(tmp_path, pinned=True)
    if changed == 'setting':
        settings['pathogen_model_name'] = settings['nucleus_model_name']
    else:
        Path(settings['pathogen_model_name']).write_bytes(b'different-private-weights')
    with pytest.raises(model_zoo.ModelZooError):
        _resolve_and_construct(settings, 'pathogen', constructor)
    assert records == []
