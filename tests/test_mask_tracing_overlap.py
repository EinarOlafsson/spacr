"""Public tracing overlaps without replacing Cellpose functions or escaping the governor."""

import threading
import numpy as np
import pytest
import torch

from spacr import _gpu_governor as governor
from spacr import _mask_tracing as tracing
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS
from tests.conftest import check_cellpose_eval_call
from tests.test_gpu_memory_governor import card, _oom


def test_the_next_network_call_overlaps_one_trace_and_preserves_order():
    """The second network call must enter before the first tracing call finishes."""
    started, release = threading.Event(), threading.Event()
    images = [np.full((3, 4, 1), index, np.float32) for index in range(4)]
    checked, evaluated = [], []
    active = []

    def evaluate(image):
        """Release the first trace only when the second network call enters."""
        index = int(image[0, 0, 0])
        if index == 1:
            assert started.wait(5)
            release.set()
        evaluated.append(index)
        return None, index, f'style-{index}'

    def trace(flow, shape):
        """Record serialized tracing calls while the network proceeds."""
        assert shape == (3, 4)
        active.append(flow)
        assert len(active) == 1
        if flow == 0:
            started.set()
            assert release.wait(5)
        active.pop()
        return flow + 10

    def checkpoint():
        """Count cancellation checks at image boundaries."""
        checked.append(True)

    masks, flows, styles = tracing._overlap_images(images, evaluate, trace, checkpoint)
    assert masks == [10, 11, 12, 13]
    assert flows == evaluated == [0, 1, 2, 3]
    assert styles == [f'style-{index}' for index in range(4)]
    assert len(checked) == 4 and not active


@pytest.mark.parametrize('failure', ['network', 'cancellation', 'trace'])
def test_failures_drain_the_pending_trace_before_a_governor_retry(failure):
    """No tracing thread remains active when the complete eval call fails."""
    started, release, finished = (threading.Event() for _ in range(3))
    images = [np.full((3, 4, 1), index, np.float32) for index in range(3)]
    checks = []

    def evaluate(image):
        """Fail the second network call only after the first trace starts."""
        index = int(image[0, 0, 0])
        if index == 1:
            assert started.wait(5)
            release.set()
            if failure == 'network':
                raise RuntimeError('network sentinel')
        return None, index, None

    def trace(flow, shape):
        """Always mark completion, including when tracing raises."""
        started.set()
        try:
            assert release.wait(5)
            if failure == 'trace':
                raise RuntimeError('trace sentinel')
            return flow
        finally:
            finished.set()

    def checkpoint():
        """Cancel before the second image while a trace is pending."""
        checks.append(True)
        if len(checks) == 2 and failure == 'cancellation':
            assert started.wait(5)
            release.set()
            raise RuntimeError('cancellation sentinel')

    with pytest.raises(RuntimeError, match=f'{failure} sentinel'):
        tracing._overlap_images(images, evaluate, trace, checkpoint)
    assert finished.is_set()


def test_public_cellpose_functions_remain_the_original_objects():
    """Run real public empty-flow tracing without replacing any Cellpose callable."""
    from cellpose import dynamics, models

    modules = [dynamics, models]
    original = {(module.__name__, name): value for module in modules
                for name, value in vars(module).items() if callable(value)}
    original_eval = models.CellposeModel.eval
    images = [np.zeros((12, 14, 1), np.float32) for _ in range(2)]

    def evaluate(image):
        """Return empty test flows without altering the real model API."""
        return None, (np.zeros((2, *image.shape[:2]), np.float32),
                      np.full(image.shape[:2], -1, np.float32)), None

    def trace(flow, shape):
        """Use the original public dynamics function on the CPU."""
        return dynamics.resize_and_compute_masks(
            flow[0], flow[1], min_size=0, device=torch.device('cpu'))

    def checkpoint():
        """Allow both synthetic images to complete."""
        return None

    masks, _, _ = tracing._overlap_images(images, evaluate, trace, checkpoint)
    assert len(masks) == 2 and all(not mask.any() for mask in masks)
    assert models.CellposeModel.eval is original_eval
    for module in modules:
        for (module_name, name), value in original.items():
            if module_name == module.__name__:
                assert getattr(module, name) is value


class _Model:
    """A local model stand-in, without any changes to Cellpose modules."""

    def __init__(self, device='cuda'):
        """Record the apparent device and network moves."""
        self.device = torch.device(device)
        self.gpu = device == 'cuda'
        self.net = _Network()

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        """Return the original inputs for the fallback-path assertion."""
        check_cellpose_eval_call(x, channel_axis)
        return x, {'batch_size': batch_size, 'normalize': normalize,
                   'channel_axis': channel_axis}


class _Network:
    """Record device transfers without initializing CUDA."""

    def __init__(self):
        """Start with no device transfers."""
        self.moves = []

    def to(self, device):
        """Record a requested transfer and retain this stand-in."""
        self.moves.append(str(device))
        return self


@pytest.mark.parametrize('changed', [
    {'normalize': True}, {'do_3D': True}, {'stitch_threshold': 0.5},
    {'z_axis': 0}, {'channel_axis': None}, {'compute_masks': False},
    {'diameter': [40, 50]}, {'diameter': float('nan')}, {'diameter': True},
    {'channels': [0, 0]},
])
def test_unverified_calls_keep_the_combined_public_api(monkeypatch, changed):
    """Only the measured flat-image contract can select overlap."""
    model = _Model()
    images = [np.zeros((3, 4, 1), np.float32) for _ in range(2)]
    monkeypatch.setattr(tracing, '_public_trace_api', lambda: (_Model, _Model.eval, None))
    keywords = {'normalize': False, 'channel_axis': -1}
    assert tracing._can_overlap(model, images, keywords)
    assert not tracing._can_overlap(model, images, {**keywords, **changed})


def test_cpu_and_single_image_calls_keep_their_original_return_value():
    """The ordinary eval call handles unsupported inputs and CPU fallback."""
    images = [np.zeros((3, 4, 1), np.float32)]
    keywords = {'normalize': False, 'channel_axis': -1, 'batch_size': 5}
    model = _Model('cpu')
    assert tracing._overlap_eval(model, images, **keywords) == (images, keywords)
    assert not tracing._can_overlap(_Model(), images, keywords)


def test_the_governor_retries_the_orchestration_with_the_same_images(card):
    """An OOM still shrinks tile size while retaining the complete image batch."""
    model, images, sizes = _Model(), [object(), object()], []

    def evaluate(actual_model, actual_images, **keywords):
        """Refuse the first tile size without changing models or images."""
        assert actual_model is model and actual_images is images
        sizes.append(keywords['batch_size'])
        if len(sizes) == 1:
            raise _oom()
        return 'completed'

    result = governor._governed_eval(model, images, site='trace-retry',
                                      batch_size=8, evaluate=evaluate)
    assert result == 'completed' and sizes == [8, 4]
    assert governor._counters()['oom_retries'] == 1


def test_the_orchestration_observes_cpu_fallback_and_device_restoration(card):
    """CPU fallback uses the same callback and restores the original device."""
    model, images, devices = _Model(), [object(), object()], []

    def evaluate(actual_model, actual_images, **keywords):
        """Fail only GPU execution, allowing the governed CPU call to finish."""
        assert actual_model is model and actual_images is images
        devices.append(model.device.type)
        if model.device.type == 'cuda':
            raise _oom()
        return 'cpu completed'

    result = governor._governed_eval(model, images, site='trace-cpu',
                                      batch_size=2, evaluate=evaluate)
    assert result == 'cpu completed' and devices[-1] == 'cpu'
    assert model.device.type == 'cuda' and model.gpu
    assert model.net.moves == ['cpu', 'cuda']
    assert governor._counters()['cpu_fallbacks'] == 1
