"""CPU operations and metadata boundaries of the Metal training adapter."""

import json
import subprocess
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_cpu_training_context_preserves_actual_upstream_references(monkeypatch):
    """The default CPU route must not replace Cellpose's modules or augmentation."""
    from cellpose import train
    from spacr import accelerator, metal_cellpose

    original = train.torch, train.dynamics, train.random_rotate_and_resize
    monkeypatch.setattr(accelerator, 'resolve', lambda: accelerator.Accelerator(
        kind='cpu', device='cpu', label='CPU'))
    with metal_cellpose.training_context():
        assert (train.torch, train.dynamics, train.random_rotate_and_resize) == original
    assert (train.torch, train.dynamics, train.random_rotate_and_resize) == original


def test_metadata_selected_metal_context_runs_real_cpu_helpers_and_restores_on_failure(monkeypatch):
    """Metadata routing may adapt CPU helpers without pretending to execute on Metal."""
    import torch
    from cellpose import train
    from spacr import accelerator, metal_cellpose

    original_torch, original_dynamics, original_augment = (
        train.torch, train.dynamics, train.random_rotate_and_resize)
    original_autocast = torch.autocast
    original_rng = np.random.get_state()
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 2:6] = 1
    pixels = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / 192
    original_labels, original_pixels = labels.copy(), pixels.copy()
    expected_flows = original_dynamics.labels_to_flows(
        [labels.copy()], device=torch.device('cpu'), niter=2)
    monkeypatch.setattr(accelerator, 'resolve', lambda: accelerator.Accelerator(
        kind='mps', device='mps', label='Metadata-only Metal'))
    net = torch.nn.Linear(2, 1)
    try:
        with pytest.raises(RuntimeError, match='owned cancellation'):
            with metal_cellpose.training_context(net):
                assert train.torch is not original_torch
                assert sys.modules['torch'] is original_torch
                assert torch.autocast is original_autocast
                assert train.dynamics.masks_to_flows_gpu is original_dynamics.masks_to_flows_gpu
                actual_flows = train.dynamics.labels_to_flows(
                    [labels.copy()], device=torch.device('mps'), niter=2)
                assert len(actual_flows) == len(expected_flows) == 1
                np.testing.assert_array_equal(actual_flows[0], expected_flows[0])
                assert np.isfinite(actual_flows[0]).all()
                for device in ('cpu', 'mps'):
                    with train.torch.autocast(device_type=device):
                        value = train.torch.ones(2)
                        assert value.device.type == 'cpu' and value.dtype == torch.float32
                with train.torch.autocast(device_type='cuda', enabled=False):
                    assert train.torch.ones(1).device.type == 'cpu'
                tensor_api = 'bsize' in inspect.signature(original_augment).parameters
                options = (dict(bsize=8, scale_range=0, device=torch.device('cpu')) if tensor_api
                           else dict(xy=(8, 8), scale_range=0))
                np.random.seed(7)
                expected = original_augment([pixels], **options)
                np.random.seed(7)
                actual = train.random_rotate_and_resize([pixels], **options)
                if tensor_api:
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    assert actual[0].device.type == 'cpu' and actual[1] is expected[1] is None
                else:
                    np.testing.assert_array_equal(actual[0], expected[0])
                    assert actual[0].shape == (1, 3, 8, 8) and list(actual[1]) == list(expected[1]) == []
                np.testing.assert_array_equal(actual[2], expected[2])
                raise RuntimeError('owned cancellation')
    finally:
        np.random.set_state(original_rng)
    assert (train.torch, train.dynamics, train.random_rotate_and_resize) == (
        original_torch, original_dynamics, original_augment)
    assert torch.autocast is original_autocast
    assert all(parameter.device.type == 'cpu' for parameter in net.parameters())
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(pixels, original_pixels)


@pytest.mark.parametrize('memory,expected', [('8 GB', 8 * 1024**3), ('512 MB', 512 * 1024**2)])
def test_radeon_profiler_counts_only_reported_dedicated_vram(monkeypatch, memory, expected):
    """Parse the profiler API response without running a native profiler or GPU."""
    from spacr import metal_cellpose

    response = {'SPDisplaysDataType': [
        {'sppci_model': 'Intel UHD Graphics', 'spdisplays_vram': '1536 MB'},
        {'sppci_model': 'AMD Radeon Pro 5300', 'spdisplays_vram': memory},
    ]}

    def profile(arguments, **options):
        """Return API-shaped metadata only for the actual bounded query."""
        assert arguments == ['system_profiler', 'SPDisplaysDataType', '-json']
        assert options == dict(capture_output=True, text=True, check=True, timeout=10)
        return SimpleNamespace(stdout=json.dumps(response))

    metal_cellpose._radeon_vram_bytes.cache_clear()
    try:
        with monkeypatch.context() as scope:
            scope.setattr(sys, 'platform', 'darwin')
            scope.setattr(subprocess, 'run', profile)
            assert metal_cellpose._radeon_vram_bytes() == expected
    finally:
        metal_cellpose._radeon_vram_bytes.cache_clear()


def test_radeon_profiler_missing_memory_stays_unknown(monkeypatch):
    """A card name alone cannot establish a usable dedicated-memory budget."""
    from spacr import metal_cellpose

    response = {'SPDisplaysDataType': [
        {'sppci_model': 'Intel UHD Graphics', 'spdisplays_vram': '1536 MB'},
        {'sppci_model': 'AMD Radeon Pro 5300'},
    ]}
    metal_cellpose._radeon_vram_bytes.cache_clear()
    try:
        with monkeypatch.context() as scope:
            scope.setattr(sys, 'platform', 'darwin')
            scope.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
                stdout=json.dumps(response)))
            assert metal_cellpose._radeon_vram_bytes() is None
    finally:
        metal_cellpose._radeon_vram_bytes.cache_clear()


def test_radeon_profiler_query_failure_stays_unknown(monkeypatch):
    """A failed system query cannot turn an advisory into a claimed memory budget."""
    from spacr import metal_cellpose

    def unavailable(*args, **kwargs):
        """Simulate an unavailable metadata service without starting a process."""
        raise OSError('profiler unavailable')

    metal_cellpose._radeon_vram_bytes.cache_clear()
    try:
        with monkeypatch.context() as scope:
            scope.setattr(sys, 'platform', 'darwin')
            scope.setattr(subprocess, 'run', unavailable)
            assert metal_cellpose._radeon_vram_bytes() is None
    finally:
        metal_cellpose._radeon_vram_bytes.cache_clear()


@pytest.mark.parametrize('budget,migrate', [(None, False), (0, False), (152, False), (151, True)])
def test_training_memory_policy_accounts_for_frozen_and_wide_cpu_parameters(
        monkeypatch, capsys, budget, migrate):
    """MPS metadata routes the measured CPU parameter budget without using a GPU."""
    import torch
    from spacr import metal_cellpose

    parameters = torch.nn.ParameterList([
        torch.nn.Parameter(torch.arange(3, dtype=torch.float16)),
        torch.nn.Parameter(torch.arange(2, dtype=torch.float64)),
        torch.nn.Parameter(torch.arange(5, dtype=torch.float64), requires_grad=False),
    ])
    before = [parameter.detach().clone() for parameter in parameters]
    assert sum(parameter.numel() * parameter.element_size() for parameter in parameters) == 62
    assert 3 * 4 + 2 * 8 + 5 * 8 + 3 * (3 * 4 + 2 * 8) == 152
    metadata = [SimpleNamespace(
        device=torch.device('mps'), numel=parameter.numel,
        element_size=parameter.element_size, requires_grad=parameter.requires_grad,
    ) for parameter in parameters]
    moves, cache_calls = [], []

    def move(device):
        moves.append(device)
        return parameters.to(device)

    net = SimpleNamespace(parameters=lambda: iter(metadata), to=move)
    monkeypatch.setattr(metal_cellpose, '_radeon_vram_bytes', lambda: budget)
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: cache_calls.append('empty'))
    metal_cellpose._fit_training_memory(net)
    assert moves == (['cpu'] if migrate else [])
    assert cache_calls == (['empty'] if migrate else [])
    assert ('exceeding Radeon VRAM' in capsys.readouterr().out) is migrate
    assert [parameter.requires_grad for parameter in parameters] == [True, True, False]
    for parameter, original in zip(parameters, before):
        assert parameter.device.type == 'cpu'
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.parametrize('net', [
    object(), SimpleNamespace(parameters=None), SimpleNamespace(parameters=lambda: iter(())),
])
def test_training_memory_without_parameters_does_not_query_gpu_metadata(monkeypatch, net):
    """Missing, non-callable and empty parameter providers cannot select a GPU policy."""
    from spacr import metal_cellpose

    def unexpected():
        raise AssertionError('no usable parameters must not query VRAM')

    monkeypatch.setattr(metal_cellpose, '_radeon_vram_bytes', unexpected)
    metal_cellpose._fit_training_memory(net)


def test_non_darwin_training_does_not_start_a_native_profiler(monkeypatch):
    """Unsupported platforms return unknown VRAM without invoking system_profiler."""
    from spacr import metal_cellpose

    def unexpected(*args, **kwargs):
        raise AssertionError('non-Darwin must not invoke a native profiler')

    metal_cellpose._radeon_vram_bytes.cache_clear()
    try:
        with monkeypatch.context() as scope:
            scope.setattr(sys, 'platform', 'linux')
            scope.setattr(subprocess, 'run', unexpected)
            assert metal_cellpose._radeon_vram_bytes() is None
    finally:
        metal_cellpose._radeon_vram_bytes.cache_clear()


def test_metal_augmentation_dispatch_computes_on_cpu_then_requests_original_device(monkeypatch):
    """Recording transfer metadata proves routing, without claiming native MPS execution."""
    import cellpose
    import torch
    from spacr import accelerator, metal_cellpose

    calls, transfers = [], []
    pixels = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original = pixels.clone()
    extra = np.array([3, 7])

    class TransferRequest:
        """Keep actual CPU output while recording the adapter's requested destination."""
        def __init__(self, tensor):
            self.tensor = tensor

        def to(self, device):
            transfers.append(device)
            return self.tensor

    def augment(images, device='cpu', scale=2):
        calls.append(torch.device(device))
        assert torch.device(device).type == 'cpu'
        return TransferRequest(images * scale), extra, None

    train = SimpleNamespace(torch=torch, random_rotate_and_resize=augment)
    monkeypatch.setattr(cellpose, 'train', train, raising=False)
    monkeypatch.setattr(accelerator, 'resolve', lambda: accelerator.Accelerator(
        kind='mps', device='mps', label='Metadata-only Metal'))
    with pytest.raises(RuntimeError, match='owned cancellation'):
        with metal_cellpose.training_context():
            result = train.random_rotate_and_resize(pixels, torch.device('mps'), scale=3)
            assert calls == [torch.device('cpu')]
            assert transfers == [torch.device('mps')]
            torch.testing.assert_close(result[0], pixels * 3, rtol=0, atol=0)
            assert result[0].device.type == 'cpu'
            assert result[1] is extra and result[2] is None
            assert sys.modules['torch'] is torch
            raise RuntimeError('owned cancellation')
    assert train.torch is torch and train.random_rotate_and_resize is augment
    assert not hasattr(train, 'dynamics')
    torch.testing.assert_close(pixels, original, rtol=0, atol=0)


def test_metal_context_preserves_older_training_modules_without_optional_seams(monkeypatch):
    """A minimal older module receives only the temporary Torch adapter."""
    import cellpose
    import torch
    from spacr import accelerator, metal_cellpose

    train = SimpleNamespace(torch=torch)
    monkeypatch.setattr(cellpose, 'train', train, raising=False)
    monkeypatch.setattr(accelerator, 'resolve', lambda: accelerator.Accelerator(
        kind='mps', device='mps', label='Metadata-only Metal'))
    with metal_cellpose.training_context():
        assert train.torch is not torch
        with train.torch.autocast(device_type='mps'):
            torch.testing.assert_close(train.torch.arange(3) + 2, torch.tensor([2, 3, 4]))
        assert not hasattr(train, 'random_rotate_and_resize')
        assert not hasattr(train, 'dynamics')
    assert train.torch is torch
    assert vars(train) == {'torch': torch}
