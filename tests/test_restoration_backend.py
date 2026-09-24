"""The isolated restoration protocol preserves coordinates and source files."""
from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from spacr import _segmentation_backends as backend


@pytest.fixture
def restoration(tmp_path, monkeypatch):
    weights = tmp_path / 'weights'
    weights.write_bytes(b'known restoration checkpoint')
    built = []

    class Cellpose3DenoiseModel:
        def __init__(self, *, model_type, device, gpu):
            self.pretrained_model = str(weights)
            self.device = device
            self.model_type = model_type
            self.gpu = gpu
            self.calls = []
            self.result = None
            built.append(self)

        def eval(self, image, batch_size=8, channels=None, channel_axis=None,
                 z_axis=None, normalize=True, rescale=None, diameter=None,
                 tile=True, do_3D=False, tile_overlap=0.1, bsize=224):
            self.calls.append((image.copy(), dict(channels=channels,
                channel_axis=channel_axis, diameter=diameter,
                normalize=normalize, batch_size=batch_size)))
            if self.result is not None:
                return self.result
            image[:] = -0.125
            return image[..., None]

    package = types.ModuleType('cellpose')
    package.denoise = types.SimpleNamespace(DenoiseModel=Cellpose3DenoiseModel)
    monkeypatch.setitem(sys.modules, 'cellpose', package)
    source = tmp_path / 'input.npy'
    image = np.arange(120, dtype=np.uint16).reshape(10, 12)
    np.save(source, image)
    request = dict(protocol=backend._PROTOCOL, id=42, op='restore',
                   model='denoise_nuclei', diameter=25.0, device='cpu',
                   input=str(source), output=str(tmp_path / 'output.npy'))
    return request, image, built, weights


def test_restoration_keeps_float_output_source_and_model_identity(restoration):
    request, original, built, weights = restoration
    cache = {}
    reply = backend._handle('cellpose3', request, cache)
    assert reply['ok'], reply
    result = np.load(reply['output'])
    assert result.shape == original.shape
    assert result.dtype == np.float32
    assert np.all(result == -0.125)
    np.testing.assert_array_equal(np.load(request['input']), original)
    record = reply['provenance']
    assert record['weights_sha256'] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert record['model'] == request['model']
    assert record['diameter_px'] == 25.0
    assert record['device'] == 'cpu'
    assert record['intensity_units'] == 'normalized model output'
    assert not built[0].gpu
    assert built[0].calls[0][1] == dict(channels=None, channel_axis=None,
        diameter=25.0, normalize=True, batch_size=1)
    assert backend._handle('cellpose3', request, cache)['ok']
    assert len(built) == 1
    assert len(built[0].calls) == 2
    assert backend._handle('cellpose3', dict(request, model='deblur_nuclei'), cache)['ok']
    assert len(built) == 2


@pytest.mark.parametrize('changes', [
    {'model': 'upsample_nuclei'}, {'model': 'missing'},
    {'diameter': 0}, {'diameter': -1}, {'diameter': float('nan')},
    {'diameter': float('inf')},
])
def test_invalid_settings_fail_before_loading_a_model(restoration, changes):
    request, _, built, _ = restoration
    reply = backend._handle('cellpose3', dict(request, **changes), {})
    assert not reply['ok']
    assert reply['error']['type'] == 'ValueError'
    assert not built


def test_other_backend_cannot_silently_restore(restoration):
    request, _, built, _ = restoration
    reply = backend._handle('dinocell', request, {})
    assert not reply['ok']
    assert 'Cellpose 3' in reply['error']['message']
    assert not built


@pytest.mark.parametrize('image', [
    np.ones((2, 3, 4)), np.empty((0, 4)), np.ones((1, 4)),
    np.full((3, 4), np.nan), np.full((3, 4), np.inf),
    np.full((3, 4), 1+2j), np.full((3, 4), 'pixels'),
])
def test_invalid_input_is_rejected_before_inference(restoration, image):
    request, _, built, _ = restoration
    np.save(request['input'], image)
    reply = backend._handle('cellpose3', request, {})
    assert not reply['ok']
    assert not built


@pytest.mark.parametrize('invalid', [
    np.ones((20, 24)), np.full((10, 12), np.nan), np.full((10, 12), np.inf),
])
def test_invalid_result_cannot_replace_a_valid_output(restoration, invalid):
    request, _, built, _ = restoration
    cache = {}
    assert backend._handle('cellpose3', request, cache)['ok']
    previous = np.load(request['output'])
    built[0].result = invalid
    reply = backend._handle('cellpose3', request, cache)
    assert not reply['ok']
    assert 'invalid values or changed image dimensions' in reply['error']['message']
    np.testing.assert_array_equal(np.load(request['output']), previous)


@pytest.fixture
def host(restoration, monkeypatch, tmp_path):
    cache = {}
    requests = []
    after_request = []

    class Worker:
        def request(self, op, *, should_cancel=None, **payload):
            requests.append((op, payload))
            reply = backend._handle('cellpose3', dict(
                payload, op=op, protocol=backend._PROTOCOL, id=1), cache)
            if not reply['ok']:
                raise backend._BackendError(reply['error']['message'])
            for callback in after_request:
                callback(op, reply)
            return reply

    monkeypatch.setattr(backend, '_backend_state', lambda *a: types.SimpleNamespace(
        state=backend._INSTALLED, in_process=False, env=str(tmp_path / 'env')))
    worker = Worker()
    factory = lambda *a: worker
    return factory, cache, requests, after_request


def test_captured_plan_is_immutable_and_returns_owned_result(restoration, host):
    factory, _, requests, _ = host
    _, source, built, _ = restoration
    original = source.copy()
    plan = backend._restoration_plan('denoise_nuclei', 25, worker_for=factory)
    assert requests[0][0] == 'restoration_model'
    assert len(built) == 1
    assert not built[0].calls
    with pytest.raises(FrozenInstanceError):
        plan.diameter = 40
    assert hash(plan) != hash(replace(plan, diameter=40))
    result, record = backend._restore_plane(source, plan, worker_for=factory)
    assert result.dtype == np.float32
    assert np.all(result == -0.125)
    assert record['weights_sha256'] == plan.weights_sha256
    np.testing.assert_array_equal(source, original)
    assert not Path(requests[-1][1]['input']).parent.exists()


def test_restarted_worker_rejects_changed_weights_before_inference(restoration, host):
    factory, cache, requests, _ = host
    _, source, built, weights = restoration
    plan = backend._restoration_plan('denoise_nuclei', 25, worker_for=factory)
    cache.clear()
    weights.write_bytes(b'a replaced checkpoint')
    with pytest.raises(backend._BackendError, match='model changed'):
        backend._restore_plane(source, plan, worker_for=factory)
    assert len(built) == 2
    assert not any(model.calls for model in built)
    assert not Path(requests[-1][1]['input']).parent.exists()


def test_cancelled_result_is_discarded_and_scratch_removed(restoration, host):
    factory, _, requests, callbacks = host
    _, source, _, _ = restoration
    plan = backend._restoration_plan('denoise_nuclei', 25, worker_for=factory)
    cancelled = []
    callbacks.append(lambda *a: cancelled.append(True))
    with pytest.raises(backend._BackendCancelled):
        backend._restore_plane(source, plan, worker_for=factory,
                              should_cancel=lambda: bool(cancelled))
    assert not Path(requests[-1][1]['input']).parent.exists()


def test_cancelled_plan_does_not_start_worker(host):
    factory, _, requests, _ = host
    with pytest.raises(backend._BackendCancelled):
        backend._restoration_plan('denoise_nuclei', 25, worker_for=factory,
                                 should_cancel=lambda: True)
    assert not requests


@pytest.mark.parametrize('corruption', ['shape', 'dtype', 'nan', 'identity'])
def test_host_rejects_invalid_worker_reply(restoration, host, corruption):
    factory, _, requests, callbacks = host
    _, source, _, _ = restoration
    plan = backend._restoration_plan('denoise_nuclei', 25, worker_for=factory)

    def corrupt(op, reply):
        if corruption == 'identity':
            reply['provenance']['weights_sha256'] = 'incorrect'
        else:
            output = np.load(reply['output'])
            if corruption == 'shape':
                output = output[:2]
            elif corruption == 'dtype':
                output = output.astype(np.uint16)
            else:
                output[:] = np.nan
            np.save(reply['output'], output)

    callbacks.append(corrupt)
    with pytest.raises(backend._BackendError):
        backend._restore_plane(source, plan, worker_for=factory)
    assert not Path(requests[-1][1]['input']).parent.exists()
