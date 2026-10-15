"""Remaining registered Cellpose constructors honor exact recorded identities."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import ops_engine, spacr_cellpose
from tests.test_spatial_submodules import xenium_slide, _mask_settings


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'exact pinned additional consumer bytes'
    raw = {'key': 'registered-model', 'name': 'current-register.CP_model',
           'kind': 'cellpose', 'sha256': hashlib.sha256(payload).hexdigest(),
           'version': '01', 'uri': 'https://trusted.invalid/weights'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin reached discovery'))
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    def fetch(entry, destination):
        actual = Path(destination) / entry.name
        actual.write_bytes(payload)
        return actual
    monkeypatch.setattr(zoo, 'fetch', fetch)
    def settings(name):
        requested = str(tmp_path / 'obsolete-laptop.CP_model')
        return {name: requested, 'model_metadata': {'schema_version': 1, 'settings': {
            name: zoo._model_reference_from_record(requested, raw)}}}
    return settings, payload


def test_actual_ops_constructor_uses_pinned_checkpoint(pinned, monkeypatch):
    from cellpose import models
    make_settings, payload = pinned
    settings = make_settings('cellpose_model')
    original = deepcopy(settings)
    calls = []
    monkeypatch.setattr(models, 'CellposeModel', lambda **kwargs: calls.append(kwargs))
    ops_engine._cellpose_model(settings, gpu=False)
    assert Path(calls[0]['pretrained_model']).read_bytes() == payload
    assert calls[0]['gpu'] is False
    assert settings == original


@pytest.mark.parametrize('name', ['model_name', 'custom_model'])
def test_actual_standalone_constructor_uses_selected_pinned_field(pinned, tmp_path, monkeypatch, name):
    make_settings, payload = pinned
    settings = make_settings(name)
    settings['src'] = str(tmp_path)
    original = deepcopy(settings['model_metadata'])
    class Stop(Exception):
        pass
    def construct(**kwargs):
        assert Path(kwargs['pretrained_model']).read_bytes() == payload
        raise Stop
    monkeypatch.setattr(spacr_cellpose.cp_models, 'CellposeModel', construct)
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    monkeypatch.setattr('spacr.accelerator.cellpose_gpu', lambda: False)
    monkeypatch.setattr('spacr.accelerator.cellpose_kwargs', lambda: {'gpu': False})
    with pytest.raises(Stop):
        spacr_cellpose.identify_masks_finetune(settings)
    assert settings[name] == original['settings'][name]['requested_value']
    assert settings['model_metadata'] == original


def test_actual_spatial_window_preserves_pin_until_constructor_boundary(pinned, tmp_path, monkeypatch):
    make_settings, payload = pinned
    source = xenium_slide(tmp_path / 'private-slide')
    settings = _mask_settings(source, tmp_path / 'private-fields', spatial_segmenter='cellpose',
                              **make_settings('spatial_cellpose_model'))
    original = deepcopy(settings)
    class Stop(Exception):
        pass
    def construct(derived, gpu):
        assert Path(derived['cellpose_model']).read_bytes() == payload
        assert gpu is False
        raise Stop
    monkeypatch.setattr(ops_engine, '_cellpose_model', construct)
    with pytest.raises(Stop):
        ops_engine.run_spatial_masks(settings)
    assert settings == original
