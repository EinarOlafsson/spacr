"""Actual desktop adapter contract; injected predictions are not model parity."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile

from tools.web_mask_generator import masks


@pytest.fixture
def case(tmp_path, monkeypatch):
    raw = (500 + np.arange(192).reshape(12, 16) * 40).astype(np.uint16)
    source = tmp_path / 'input.tif'
    tifffile.imwrite(source, raw, metadata={'axes': 'YX'})
    checkpoint = tmp_path / 'weights'
    checkpoint.write_bytes(b'trusted test bytes, not model weights')
    labels = np.zeros(raw.shape, dtype=np.uint32)
    labels[3:6, 4:8] = 12
    labels[0:2, 13:] = 70000
    vectors = np.ones((2, *raw.shape), dtype=np.float32)
    logits = np.linspace(-2, 3, raw.size, dtype=np.float32).reshape(raw.shape)
    calls = []
    class Model:
        def eval(self, **kwargs):
            calls.append(kwargs)
            return [labels.copy()], [[np.zeros((*raw.shape, 3)), vectors, logits, None]], None
    monkeypatch.setattr(masks, '_load_mask_model', lambda *args: Model())
    arguments = dict(input_file=source, checkpoint=checkpoint,
                     expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     model_version='v1', effective_settings={})
    return tmp_path / 'download.zip', arguments, raw, labels, vectors, logits, calls


def unpack(target):
    with zipfile.ZipFile(target) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.mark.parametrize('family,role', list(masks.OBJECT_ROLES.items()))
def test_all_six_routes_use_actual_desktop_pipeline(case, family, role):
    target, arguments, raw, labels, vectors, logits, calls = case
    masks.run_mask_download(target, family=family, model_id=family + '_v1', **arguments)
    members = unpack(target)
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(members['image.tif'])), raw)
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(members['mask.tif'])), labels)
    np.testing.assert_array_equal(np.load(io.BytesIO(members['flows.npy'])), vectors)
    np.testing.assert_array_equal(np.load(io.BytesIO(members['cell_probability.npy'])), logits)
    frame = pd.read_csv(io.BytesIO(members['objects.csv']))
    assert frame.label.tolist() == [12, 70000]
    assert frame.intensity_mean_channel_0.tolist() == [raw[labels == label].mean() for label in [12, 70000]]
    settings = json.loads(members['manifest.json'])['settings']
    from spacr.object_settings_table import _resolved_diameter
    assert settings['diameter_used'] == _resolved_diameter(None, settings['magnification'], role)
    assert settings['object_role'] == role
    assert str(arguments['checkpoint']) not in json.dumps(settings)
    assert len(calls) == 1
    call = calls[0]
    assert call['normalize'] is False and call['resample'] is (role != 'pathogen')
    assert call['cellprob_threshold'] == (-1 if role == 'pathogen' else 0)
    assert call['x'][0].shape == (*raw.shape, 1)
    assert call['x'][0].dtype == np.float32
    assert np.isfinite(call['x'][0]).all()
    assert call['x'][0].min() >= 0 and call['x'][0].max() <= 1


@pytest.mark.parametrize('diameter,expected', [(31.5, 31.5), ('auto', None), (0, None)])
def test_explicit_and_auto_diameters_reach_desktop_eval(case, diameter, expected):
    target, arguments, *_, calls = case
    arguments['effective_settings'] = {'diameter': diameter, 'flow_threshold': 0.8, 'CP_prob': -2.5}
    masks.run_mask_download(target, family='toxoplasma_pv', model_id='toxoplasma_pv_v1', **arguments)
    assert calls[0]['diameter'] == expected
    assert calls[0]['flow_threshold'] == 0.8 and calls[0]['cellprob_threshold'] == -2.5


def test_actual_border_filter_and_original_intensity(case):
    target, arguments, raw, labels, *rest = case
    arguments['effective_settings'] = {'remove_border_objects': True, 'remove_background': True, 'background': 400}
    masks.run_mask_download(target, family='live_cell', model_id='live_cell_v1', **arguments)
    members = unpack(target)
    actual = tifffile.imread(io.BytesIO(members['mask.tif']))
    assert set(np.unique(actual)) == {0, 1}
    np.testing.assert_array_equal(actual > 0, labels == 12)
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(members['image.tif'])), raw)


@pytest.mark.parametrize('settings', [
    {'bad': 1}, {'diameter': -1}, {'diameter': True}, {'diameter': float('inf')},
    {'magnification': 0}, {'flow_threshold': -1}, {'background': False},
    {'signal_to_noise': float('nan')}, {'remove_background': 'yes'},
    {'remove_border_objects': 1}, {'perimeter_fraction': 1.1},
    {'lower_percentile': 101}, {'object_filters': {}},
])
def test_invalid_settings_do_not_run_or_replace(case, settings):
    target, arguments, *_, calls = case
    target.write_bytes(b'previous')
    arguments['effective_settings'] = settings
    with pytest.raises((ValueError, TypeError)):
        masks.run_mask_download(target, family='live_cell', model_id='live_cell_v1', **arguments)
    assert target.read_bytes() == b'previous' and calls == []


def test_exact_checkpoint_and_family_identity(case):
    target, arguments, *_, calls = case
    arguments['expected_sha256'] = '00' * 32
    with pytest.raises(ValueError, match='SHA-256'):
        masks.run_mask_download(target, family='live_cell', model_id='live_cell_v1', **arguments)
    with pytest.raises(ValueError, match='class'):
        masks.run_mask_download(target, family='live_cell', model_id='toxoplasma_pv_v1', **arguments)
    assert calls == []


def test_multiple_input_channels_require_explicit_selection(case):
    target, arguments, raw, *_, calls = case
    tifffile.imwrite(arguments['input_file'], np.stack([raw, raw]), metadata={'axes': 'CYX'}, photometric='minisblack')
    with pytest.raises(ValueError, match='exactly one'):
        masks.run_mask_download(target, family='live_cell', model_id='live_cell_v1', **arguments)
    assert calls == []
    masks.run_mask_download(target, family='live_cell', model_id='live_cell_v1', selection={'channels': [1]}, **arguments)
    assert len(calls) == 1


def test_filter_area_floor_and_actual_filter_application(case):
    target, arguments, raw, labels, *_, calls = case
    arguments['effective_settings'] = {'object_filters': [{'property': 'area', 'min': 10, 'max': None}]}
    masks.run_mask_download(target, family='nuclei_from_cellmask', model_id='nuclei_from_cellmask_v1', **arguments)
    assert calls[0]['min_size'] == 10
    actual = tifffile.imread(io.BytesIO(unpack(target)['mask.tif']))
    np.testing.assert_array_equal(actual > 0, labels == 12)


def test_loader_uses_actual_desktop_constructor_and_digest_cache(tmp_path, monkeypatch):
    from spacr import object as desktop
    calls = []
    model = object()
    monkeypatch.setattr(desktop.cp_models, 'CellposeModel', lambda **kw: calls.append(kw) or model)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'local')
    masks._load_mask_model.cache_clear()
    assert masks._load_mask_model(checkpoint, 'a' * 64, 'cell') is model
    assert masks._load_mask_model(checkpoint, 'a' * 64, 'cell') is model
    assert len(calls) == 1
    assert calls[0]['pretrained_model'] == str(checkpoint)
    assert masks._load_mask_model(checkpoint, 'b' * 64, 'cell') is model
    assert len(calls) == 2
    masks._load_mask_model.cache_clear()
