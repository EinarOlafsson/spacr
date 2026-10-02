"""Native StarDist volumes keep their axes, labels and worker transport."""
import json
import types
import sys
from pathlib import Path

import numpy as np
import pytest

from spacr import _segmentation_backends as B
from spacr.zstack import ZStackSpec


@pytest.fixture(autouse=True)
def _isolated_normalizer(monkeypatch):
    """The real csbdeep package belongs only to the isolated backend."""
    utility = types.ModuleType('csbdeep.utils')
    utility.normalize = lambda image, *args, **kwargs: image.astype(np.float32)
    monkeypatch.setitem(sys.modules, 'csbdeep.utils', utility)


def _folder(tmp_path, **changes):
    config = dict(n_dim=3, axes='ZYXC', n_channel_in=1)
    config.update(changes)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    return str(tmp_path)


class _Network:
    def __init__(self, *args, **kwargs):
        self.config = types.SimpleNamespace(anisotropy=(2, 1, 1))
        self.calls = []

    def _guess_n_tiles(self, image):
        return (1, 1, 1)

    def predict_instances(self, image, **kwargs):
        self.calls.append((image.copy(), kwargs))
        labels = np.zeros(image.shape, np.int32)
        labels[:, 1:4, 2:6] = 8
        labels[0, 0, 0] = 9
        return (labels, {}), (np.full((2, 3, 4), .7), np.zeros((2, 3, 4, 8)))


def _adapter(tmp_path):
    return B._StarDistAdapter(_folder(tmp_path), models_module=types.SimpleNamespace(StarDist3D=_Network))


def test_native_prediction_keeps_volume_and_removes_small_objects(tmp_path):
    adapter = _adapter(tmp_path)
    image = np.arange(4*8*10*2, dtype=np.float32).reshape(4, 8, 10, 2)
    original = image.copy()
    masks, flows, _ = adapter.eval([image], do_3D=True, z_axis=0,
                                   anisotropy=2, min_size=2, normalize=False)
    assert masks[0].shape == (4, 8, 10)
    assert masks[0].max() == 1 and masks[0][0, 0, 0] == 0
    assert np.all(masks[0][:, 1:4, 2:6] == 1)
    assert flows[0][2].shape == masks[0].shape
    np.testing.assert_array_equal(image, original)
    passed, kwargs = adapter._model.calls[0]
    assert passed.ndim == 3 and kwargs['axes'] == 'ZYX'
    assert kwargs['scale'] == (1., 1., 1.)
    assert any('normalisation' in value for value in adapter.translated)


def test_reordered_channels_and_z_are_restored_and_anisotropy_is_honoured(tmp_path):
    adapter = _adapter(tmp_path)
    image = np.zeros((2, 8, 4, 10), np.float32)  # CYZX
    masks, flows, _ = adapter.eval([image], channel_axis=0, do_3D=True,
                                   z_axis=2, anisotropy=4, diameter=15)
    assert masks[0].shape == (8, 4, 10)
    assert flows[0][2].shape == masks[0].shape
    assert adapter._model.calls[0][1]['scale'] == (4., 2., 2.)


@pytest.mark.parametrize('updates, message', [
    ({'do_3D': False}, 'do_3D'),
    ({'anisotropy': None}, 'anisotropy'),
    ({'anisotropy': float('nan')}, 'anisotropy'),
    ({'z_axis': None}, 'z_axis'),
    ({'z_axis': -1}, 'must differ'),
])
def test_invalid_volume_requests_refuse_before_inference(tmp_path, updates, message):
    adapter = _adapter(tmp_path)
    kwargs = dict(do_3D=True, z_axis=0, anisotropy=2)
    kwargs.update(updates)
    with pytest.raises(ValueError, match=message):
        adapter.eval([np.zeros((4, 8, 10, 1))], **kwargs)
    assert not adapter._model.calls


@pytest.mark.parametrize('shape', [(8, 10), (1, 8, 10), (0, 8, 10)])
def test_missing_volume_is_not_segmented_as_planes(tmp_path, shape):
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError):
        adapter.eval([np.zeros(shape)], do_3D=True, z_axis=0, anisotropy=2)


@pytest.mark.parametrize('config', [dict(n_dim=4), dict(n_channel_in=2), dict(axes='YXC')])
def test_custom_model_dimension_metadata_is_validated(tmp_path, config):
    with pytest.raises(ValueError):
        B._stardist_model_ndim(_folder(tmp_path, **config))


def test_native_loader_requires_volumetric_plan_without_time(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    created = []
    monkeypatch.setattr(B, '_RemoteBackend', lambda *args, **kw:
                        created.append(kw) or types.SimpleNamespace(note='test'))
    for plan, time in [(None, None), (ZStackSpec(mode='project'), None),
                       (ZStackSpec(mode='stitch'), None),
                       (ZStackSpec(mode='volumetric'), object())]:
        with pytest.raises(ValueError, match='volumetric'):
            B._load_prefixed('stardist', folder, z_plan=plan, t_plan=time)
    assert not created
    B._load_prefixed('stardist', folder, z_plan=ZStackSpec(mode='volumetric'))
    assert created[0]['model'] == folder


def test_worker_transport_sends_one_volume_and_returns_cellpose_volume_shape(tmp_path):
    calls = []

    class Worker:
        def request(self, operation, **kwargs):
            calls.append(kwargs)
            assert operation == 'segment'
            assert len(kwargs['inputs']) == 1
            source = np.load(kwargs['inputs'][0])
            assert source.shape == (4, 8, 10, 1)
            path = Path(kwargs['outputs']) / 'mask.npy'
            np.save(path, np.ones(source.shape[:-1], np.uint16))
            return {'outputs': [{'mask': str(path), 'flows': [None]*4}]}

    model = object.__new__(B._RemoteBackend)
    model.name = 'stardist'
    model.label = 'StarDist'
    model.env = str(tmp_path)
    model.model = 'test'
    model.device = 'cpu'
    model.options = {}
    model._said = set()
    model._worker_for = lambda *args: Worker()
    volume = np.zeros((4, 8, 10, 1), np.float32)
    masks, _, _ = model.eval(volume, do_3D=True, z_axis=0, anisotropy=2)
    assert masks.shape == (4, 8, 10)
    assert calls[0]['params']['do_3D'] is True
    assert calls[0]['params']['z_axis'] == 0
    assert calls[0]['params']['anisotropy'] == 2
    batched, _, _ = model.eval([volume], do_3D=True, z_axis=0, anisotropy=2)
    assert len(batched) == 1 and batched[0].shape == masks.shape


def test_two_dimensional_model_rejects_native_request(tmp_path):
    adapter = object.__new__(B._StarDistAdapter)
    B._PrefixedAdapter.__init__(adapter)
    adapter._ndim = 2
    with pytest.raises(ValueError, match='StarDist3D'):
        adapter.eval([np.zeros((4, 8, 10))], do_3D=True)
