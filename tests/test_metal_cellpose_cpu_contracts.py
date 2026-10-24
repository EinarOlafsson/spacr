"""CPU operations and metadata boundaries of the Metal training adapter."""

import json
import subprocess
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
                np.random.seed(7)
                expected = original_augment([pixels], bsize=8, scale_range=0,
                                            device=torch.device('cpu'))
                np.random.seed(7)
                actual = train.random_rotate_and_resize(
                    [pixels], bsize=8, scale_range=0, device=torch.device('cpu'))
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                assert actual[0].device.type == 'cpu' and actual[1] is expected[1] is None
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
