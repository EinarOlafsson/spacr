"""Unavailable drivers and uncertain dtype probes must retain a usable fallback."""
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spacr import accelerator as acc


@pytest.mark.parametrize('backend,probe', [('cuda', acc._cuda_or_rocm), ('xpu', acc._xpu)])
def test_an_available_device_without_a_marketing_name_keeps_its_backend(backend, probe):
    driver = SimpleNamespace(is_available=lambda: True,
                             get_device_name=Mock(side_effect=OSError('name query failed')))
    found = probe(SimpleNamespace(**{backend: driver}))
    assert found.kind == backend and found.usable
    assert found.name == ('GPU' if backend == 'cuda' else 'Intel GPU')


@pytest.mark.parametrize('available', [False, OSError('driver unavailable')])
def test_failed_xpu_availability_probe_does_not_advertise_a_device(available):
    probe = Mock(side_effect=available) if isinstance(available, Exception) else lambda: available
    assert acc._xpu(SimpleNamespace(xpu=SimpleNamespace(is_available=probe))) is None


def test_failed_metal_availability_probe_does_not_advertise_a_device():
    backend = SimpleNamespace(is_available=Mock(side_effect=OSError('Metal unavailable')))
    assert acc._mps(SimpleNamespace(backends=SimpleNamespace(mps=backend))) is None


@pytest.mark.parametrize('result', ['unavailable', 'raises', 'working'])
def test_directml_probe_reports_only_a_working_device(monkeypatch, result):
    available = Mock(side_effect=RuntimeError('driver error')) if result == 'raises' else lambda: result == 'working'
    module = SimpleNamespace(is_available=available, device=lambda: 'privateuseone:0',
                             device_name=lambda index: 'Test Radeon')
    monkeypatch.setitem(sys.modules, 'torch_directml', module)
    found = acc._directml()
    if result != 'working':
        assert found is None
    else:
        assert found.device == 'privateuseone:0' and found.kind == 'directml'
        assert found.name == 'Test Radeon' and found.fallback
        assert not found.float64 and not found.autocast


@pytest.mark.parametrize('output', ['Chipset Model: Test Radeon\n', 'no chipset row', OSError('profiler missing')])
def test_metal_name_handles_missing_profiler_and_unrecognized_output(monkeypatch, output):
    monkeypatch.setattr(acc.platform, 'system', lambda: 'Darwin')
    profiler = Mock(side_effect=output) if isinstance(output, Exception) else Mock(return_value=SimpleNamespace(stdout=output))
    monkeypatch.setattr(subprocess, 'run', profiler)
    # Bypass only the cache, so a simulated machine cannot poison later tests.
    name = acc._metal_gpu_name.__wrapped__()
    assert name == ('Test Radeon' if isinstance(output, str) and 'Chipset' in output else 'Metal GPU')
    assert profiler.call_args.kwargs['timeout'] == 10


def test_unparseable_macos_version_keeps_the_supported_hardware_explanation(monkeypatch):
    monkeypatch.setattr(acc.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(acc.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(acc.platform, 'mac_ver', lambda: ('unknown', (), ''))
    monkeypatch.setattr(acc, '_metal_gpu_name', lambda: 'Apple GPU')
    label, note = acc._why_metal_is_unavailable()
    assert label == 'Apple GPU'
    assert 'does not offer a Metal device' in note and 'Upgrading' not in note


@pytest.mark.parametrize('error,expected', [(None, True), (RuntimeError('unsupported dtype'), False),
                                         (OSError('transient allocation failure'), True)])
def test_dtype_probe_separates_unsupported_from_inconclusive(error, expected):
    found = acc.Accelerator(kind='xpu', device='xpu', label='Intel GPU')
    allocate = Mock(side_effect=error)
    torch = SimpleNamespace(zeros=allocate, float64='double', bfloat16='brain')
    measured = acc._measure_dtypes(torch, found)
    assert measured.float64 is expected and measured.bfloat16 is expected
    assert [call.kwargs for call in allocate.call_args_list] == [
        dict(dtype='double', device='xpu'), dict(dtype='brain', device='xpu')]
    assert found.float64 and found.bfloat16  # the immutable original is unchanged


def test_a_broken_probe_does_not_hide_a_later_working_backend(monkeypatch):
    found = acc.Accelerator(kind='xpu', device='xpu', label='Intel GPU')
    monkeypatch.setattr(acc, '_cuda_or_rocm', Mock(side_effect=OSError('broken driver')))
    monkeypatch.setattr(acc, '_mps', lambda torch: None)
    monkeypatch.setattr(acc, '_xpu', lambda torch: found)
    monkeypatch.setattr(acc, '_torch', lambda: SimpleNamespace())
    monkeypatch.delenv(acc.ENV_DEVICE, raising=False)
    assert acc.inspect_torch(object()) is found
    assert acc.resolve(refresh=True) == found


def test_neural_engine_is_reported_only_for_apple_silicon(monkeypatch):
    monkeypatch.setattr(acc.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(acc.platform, 'machine', lambda: 'arm64')
    assert acc.neural_engines() == ('Apple Neural Engine (CoreML only)',)
    monkeypatch.setattr(acc.platform, 'machine', lambda: 'x86_64')
    assert acc.neural_engines() == ()


def test_xpu_cache_failure_is_reported_as_no_cleanup(monkeypatch):
    driver = SimpleNamespace(empty_cache=Mock(side_effect=RuntimeError('driver stopped')))
    torch = SimpleNamespace(xpu=driver)
    monkeypatch.setattr(acc, 'inspect_torch', lambda module: acc.Accelerator(
        kind='xpu', device='xpu', label='Intel GPU'))
    assert acc.empty_cache(torch) == ''
    driver.empty_cache.assert_called_once_with()
    driver.empty_cache.side_effect = None
    assert acc.empty_cache(torch) == 'torch.xpu.empty_cache()'


def test_missing_cellpose_does_not_break_the_metal_workaround(monkeypatch):
    monkeypatch.setattr(acc, '_FLOW_PATCH_APPLIED', False)
    monkeypatch.setitem(sys.modules, 'cellpose', None)
    acc._keep_cellpose_flows_off_metal()
    assert acc._FLOW_PATCH_APPLIED


@pytest.mark.parametrize('kind,gpu,double,brain', [
    ('cpu', False, True, True), ('rocm', True, True, True),
    ('xpu', True, False, False), ('mps', True, False, False)])
def test_device_questions_and_cellpose_arguments_agree(monkeypatch, kind, gpu, double, brain):
    device = 'cuda:0' if kind == 'rocm' else kind
    found = acc.Accelerator(kind=kind, device=device, label=kind,
                            float64=double, bfloat16=brain)
    monkeypatch.setattr(acc, '_CACHED', found)
    # Constructing a device object is exercised without importing or querying
    # any real driver. The same identity must reach the Cellpose arguments.
    torch = SimpleNamespace(device=lambda value: ('torch device', value))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    workaround = Mock()
    monkeypatch.setattr(acc, '_keep_cellpose_flows_off_metal', workaround)
    assert acc.torch_device() == ('torch device', device)
    assert acc.device_string() == device
    assert acc.is_gpu() is gpu
    assert not acc.is_cuda()
    assert acc.supports_float64() is double
    assert acc.supports_bfloat16() is brain
    kwargs = acc.cellpose_kwargs()
    assert kwargs['gpu'] is gpu
    assert kwargs['device'] == acc.torch_device()
    if gpu and not brain:
        assert kwargs['use_bfloat16'] is False
    else:
        assert 'use_bfloat16' not in kwargs
    assert workaround.call_count == int(kind == 'mps')


def test_missing_torch_is_a_safe_cpu_diagnostic_and_no_cache_cleanup(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.delenv(acc.ENV_DEVICE, raising=False)
    assert acc._torch() is None
    assert acc.empty_cache() == ''
    assert acc.describe() == 'CPU (PyTorch is not installed)'


def test_torch_handle_is_returned_without_querying_a_device(monkeypatch):
    module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'torch', module)
    assert acc._torch() is module


def test_shader_probe_failure_falls_back_to_the_cpu_renderer(monkeypatch):
    module = SimpleNamespace(gpu_is_available=Mock(side_effect=RuntimeError('no context')))
    monkeypatch.setitem(sys.modules, 'spacr.qt.widgets.fractal_travel', module)
    assert acc._opengl_likely() is False


def test_forcing_the_detected_kind_does_not_replace_measured_capabilities():
    found = acc.Accelerator(kind='xpu', device='xpu:0', label='Test GPU', bfloat16=False)
    assert acc._forced_device('xpu', found) is found
    assert acc._forced_device('xpu:0', found) is found
