"""A no-probe diagnostic cannot allocate through its capability decoration."""
from types import SimpleNamespace

import pytest

from spacr import accelerator as acc, doctor


@pytest.fixture
def machine(monkeypatch):
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'SPACR_DEVICE'):
        monkeypatch.delenv(key, raising=False)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('forbidden hardware access')
        raise AssertionError('unexpected initialization/allocation')
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_name=forbidden, init=forbidden, synchronize=forbidden)
    torch = SimpleNamespace(version=SimpleNamespace(cuda='12.8', hip=None), cuda=cuda,
                            zeros=forbidden)
    monkeypatch.setattr(doctor, '_import_torch', lambda: torch)
    monkeypatch.setattr(doctor, '_nvidia_driver', lambda: 'fake driver')
    monkeypatch.setattr(acc, 'resolve', forbidden)
    monkeypatch.setattr(acc, '_measure_dtypes', forbidden)
    monkeypatch.setattr(acc, '_directml', lambda: None)
    monkeypatch.setattr(acc, '_opengl_likely', lambda: False)
    return torch, calls, forbidden


@pytest.mark.parametrize('available', [True, False])
def test_no_probe_never_initializes_names_or_allocates_via_capabilities(machine, available):
    torch, calls, _ = machine
    torch.cuda.is_available = lambda: available
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    assert result.status == (doctor.PASS if available else doctor.FAIL)
    assert calls == []
    assert any('Segmentation (Cellpose)' in row for row in result.details)
    assert any('Model training' in row for row in result.details)
    assert 'Live backdrop and spaceout: CPU — CPU renderer' in result.details


@pytest.mark.parametrize('hidden', ['', '-1', '  -1  '])
@pytest.mark.parametrize('probe', [False, True])
def test_explicit_hidden_cuda_is_skipped_without_querying_cuda(machine, monkeypatch, hidden, probe):
    torch, calls, forbidden = machine
    torch.cuda.is_available = forbidden
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', hidden)
    result = doctor.check_gpu(doctor.Context(probe_gpu=probe))
    assert result.status == doctor.SKIP
    assert 'deliberately hidden' in result.message
    assert 'CUDA_VISIBLE_DEVICES' in result.message
    assert calls == []
    assert doctor.exit_code([result]) == 0


@pytest.mark.parametrize('key', ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
def test_hidden_rocm_is_not_misdiagnosed_as_missing_nvidia(machine, monkeypatch, key):
    torch, calls, forbidden = machine
    torch.version.cuda = None
    torch.version.hip = '7.0'
    torch.cuda.is_available = forbidden
    monkeypatch.setenv(key, '')
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    assert result.status == doctor.SKIP and key in result.message
    assert calls == []


def test_hiding_cuda_does_not_disable_apple_metal(machine, monkeypatch):
    torch, calls, forbidden = machine
    torch.version.cuda = None
    torch.cuda.is_available = forbidden
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(acc, '_metal_gpu_name', lambda: 'Test Apple GPU')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    assert result.status == doctor.PASS and 'Apple Metal' in result.message
    assert calls == []


def test_forced_cpu_does_not_import_or_probe_torch(machine, monkeypatch):
    _, calls, forbidden = machine
    monkeypatch.setenv('SPACR_DEVICE', 'cpu')
    monkeypatch.setattr(doctor, '_import_torch', forbidden)
    result = doctor.check_gpu(doctor.Context(probe_gpu=True))
    assert result.status == doctor.SKIP and 'SPACR_DEVICE=cpu' in result.message
    assert calls == []


def test_real_failure_is_reported_when_probing_is_requested(machine):
    torch, calls, _ = machine
    torch.cuda.is_available = lambda: False
    def initialize():
        calls.append('init')
        raise RuntimeError('incompatible driver')
    torch.cuda.init = initialize
    result = doctor.check_gpu(doctor.Context(probe_gpu=True))
    assert result.status == doctor.FAIL
    assert 'RuntimeError: incompatible driver' in result.details
    assert calls == ['init']
