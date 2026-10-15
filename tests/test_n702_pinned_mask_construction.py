"""Pinned worker payloads retain original requests through actual Mask construction."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from spacr import _mask_workers as workers
from spacr import model_zoo as zoo
from tests.test_cellpose4_model_story import _mask_settings, sam_pipeline


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'exact pinned mask checkpoint'
    raw = {'key': 'recorded-mask', 'name': 'current-mask.CP_model', 'kind': 'cellpose',
           'sha256': hashlib.sha256(payload).hexdigest(), 'version': '001',
           'uri': 'https://trusted.invalid/weights'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin reached name discovery'))
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    calls = []
    def fetch(entry, destination):
        calls.append(entry.key)
        path = Path(destination) / entry.name
        path.write_bytes(payload)
        return path
    monkeypatch.setattr(zoo, 'fetch', fetch)
    def settings(key, kind='cellpose', prefix=''):
        raw['kind'] = kind
        requested = prefix + str(tmp_path / 'old-laptop-name.CP_model')
        return {key: requested, 'model_metadata': {'schema_version': 1, 'settings': {
            key: zoo._model_reference_from_record(requested, raw)}}}
    return settings, raw, payload, calls


@pytest.mark.parametrize('role,key', [('cell', 'cell_model_name'), ('nucleus', 'nucleus_model_name'),
                                    ('pathogen', 'pathogen_model')])
def test_worker_preparation_retains_valid_recorded_request_and_pins(pinned, role, key):
    make_settings, raw, payload, calls = pinned
    settings = make_settings(key)
    original = deepcopy(settings)
    prepared, signature, identity = workers._prepare_mask_model(settings, role)
    assert settings == original
    assert prepared[key] == original[key]
    assert prepared['model_metadata'] == original['model_metadata']
    assert zoo._validate_model_metadata(prepared) == original['model_metadata']
    assert Path(identity['path']).read_bytes() == payload
    assert identity['sha256'] == raw['sha256']
    assert len(signature) == 64
    assert calls == [raw['key']]


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino'])
def test_backend_worker_retains_prefixed_request_and_exact_signature(pinned, backend):
    make_settings, raw, payload, _calls = pinned
    settings = make_settings('cell_model_name', backend, backend + ':')
    original = deepcopy(settings)
    prepared, signature, identity = workers._resolved_backend_model(settings, 'cell')
    assert settings == original
    assert prepared['cell_model_name'] == original['cell_model_name']
    assert zoo._validate_model_metadata(prepared) == original['model_metadata']
    assert identity['backend'] == backend
    assert len(identity['sha256']) == 64
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest() == raw['sha256']
    assert Path(identity['path']).read_bytes() == payload
    assert len(signature) == 64


def test_actual_sam_constructor_reuses_worker_prepared_exact_pin(pinned, sam_pipeline, tmp_path):
    from spacr import object as objects
    make_settings, _raw, payload, calls = pinned
    src = tmp_path / 'stacks'
    src.mkdir()
    np.savez(src / 'batch.npz', data=np.ones((1, 32, 32, 2), dtype=np.float32),
             filenames=np.array(['plate1_A01_1.npy']))
    settings = _mask_settings(src, **make_settings('cell_model_name'))
    original = deepcopy(settings)
    prepared, _signature, identity = workers._prepare_mask_model(settings, 'cell')
    objects.generate_cellpose_masks_sam(str(src), prepared, 'cell', run_qc=False)
    assert Path(sam_pipeline['model'].pretrained_model).read_bytes() == payload
    assert sam_pipeline['model'].pretrained_model == identity['path']
    assert settings == original
    assert prepared['cell_model_name'] == original['cell_model_name']
    assert prepared['model_metadata'] == original['model_metadata']
    assert len(calls) == 1
    assert np.load(src / 'cell_mask_stack/plate1_A01_1.npy').max() == 1


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino'])
def test_actual_generator_preserves_backend_prefix_then_passes_verified_path(pinned, tmp_path, monkeypatch, backend):
    from spacr import object as objects
    from spacr import _segmentation_backends as sb
    make_settings, _raw, payload, _calls = pinned
    src = tmp_path / 'stacks'
    src.mkdir()
    np.savez(src / 'batch.npz', data=np.ones((1, 32, 32, 2), dtype=np.float32),
             filenames=np.array(['plate1_A01_1.npy']))
    settings = _mask_settings(src, **make_settings('cell_model_name', backend, backend + ':'))
    original_metadata = deepcopy(settings['model_metadata'])
    seen = {}
    class Stop(Exception):
        pass
    def construct(name, **kwargs):
        seen.update(backend=name, **kwargs)
        raise Stop
    monkeypatch.setattr(sb, '_load_backend', construct)
    with pytest.raises(Stop):
        objects.generate_cellpose_masks_sam(str(src), settings, 'cell', run_qc=False)
    assert seen['backend'] == backend
    assert Path(seen['model_name']).read_bytes() == payload
    assert settings['cell_model_name'] == original_metadata['settings']['cell_model_name']['requested_value']
    assert settings['model_metadata'] == original_metadata


def test_wrong_pinned_worker_bytes_refuse_before_stock_fallback(pinned):
    make_settings, _raw, _payload, calls = pinned
    settings = make_settings('cell_model_name')
    local = Path(settings['cell_model_name'])
    local.write_bytes(b'wrong supplied file')
    with pytest.raises(zoo.ModelZooError):
        workers._prepare_mask_model(settings, 'cell')
    assert local.read_bytes() == b'wrong supplied file'
    assert calls == []
