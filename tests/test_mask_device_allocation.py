"""Mask workers must stay within the GPUs allocated to the parent process."""

from types import SimpleNamespace

import pytest

from spacr import accelerator


@pytest.mark.parametrize('hip,backend', [(None, 'cuda'), ('6.4', 'rocm')])
@pytest.mark.parametrize('forced', [None, 'auto'])
def test_discovery_uses_visible_ordinals_without_allocating_tensors(monkeypatch, hip, backend, forced):
    seen = []

    def properties(index):
        seen.append(index)
        return SimpleNamespace(name=f'GPU {index}', total_memory=(index + 1) * 1024)

    fake = SimpleNamespace(version=SimpleNamespace(hip=hip), cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=properties))
    monkeypatch.setattr(accelerator, '_torch', lambda: fake)
    if forced is None:
        monkeypatch.delenv(accelerator.ENV_DEVICE, raising=False)
    else:
        monkeypatch.setenv(accelerator.ENV_DEVICE, forced)
    assert accelerator._mask_devices() == (
        {'index': 0, 'name': 'GPU 0', 'memory_bytes': 1024, 'backend': backend},
        {'index': 1, 'name': 'GPU 1', 'memory_bytes': 2048, 'backend': backend})
    assert seen == [0, 1]


@pytest.mark.parametrize('forced', ['cpu', 'cuda:1', 'mps', 'xpu:0', 'off'])
def test_forced_single_device_never_probes_parallel_gpus(monkeypatch, forced):
    monkeypatch.setenv(accelerator.ENV_DEVICE, forced)
    monkeypatch.setattr(accelerator, '_torch', lambda: pytest.fail('GPU probe'))
    assert accelerator._mask_devices() == ()


@pytest.mark.parametrize('visible', ['4,2', 'GPU-aaaa,GPU-bbbb', 'MIG-aaaa,MIG-bbbb'])
def test_cuda_worker_selects_the_allocated_device_not_the_host_ordinal(visible):
    parent = {'CUDA_VISIBLE_DEVICES': visible, 'UNCHANGED': 'value'}
    child = accelerator._mask_worker_environment(1, backend='cuda', count=2,
                                                 environment=parent)
    assert child == {'CUDA_VISIBLE_DEVICES': visible.split(',')[1],
                     'UNCHANGED': 'value', 'SPACR_DEVICE': 'cuda'}
    assert parent == {'CUDA_VISIBLE_DEVICES': visible, 'UNCHANGED': 'value'}


@pytest.mark.parametrize('aliases', [
    {}, {'HIP_VISIBLE_DEVICES': '2,0'}, {'CUDA_VISIBLE_DEVICES': '2,0'},
    {'HIP_VISIBLE_DEVICES': '2,0', 'CUDA_VISIBLE_DEVICES': '2,0'}])
def test_rocm_keeps_the_outer_allocation_and_restricts_both_hip_aliases(aliases):
    parent = {'ROCR_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c', **aliases}
    child = accelerator._mask_worker_environment(1, backend='rocm', count=2,
                                                 environment=parent)
    assert child['ROCR_VISIBLE_DEVICES'] == parent['ROCR_VISIBLE_DEVICES']
    assert child['HIP_VISIBLE_DEVICES'] == child['CUDA_VISIBLE_DEVICES'] == (
        '0' if aliases else '1')
    assert child['SPACR_DEVICE'] == 'rocm'


def test_contradictory_hip_visibility_is_not_guessed():
    with pytest.raises(ValueError, match='disagree'):
        accelerator._mask_worker_environment(0, backend='rocm', count=2,
            environment={'HIP_VISIBLE_DEVICES': '0,1', 'CUDA_VISIBLE_DEVICES': '1,0'})


@pytest.mark.parametrize('index,count', [(-1, 2), (2, 2), (0, 0),
                                        (True, 2), (0.5, 2), (0, '2')])
def test_invalid_selection_is_rejected(index, count):
    with pytest.raises(ValueError):
        accelerator._mask_worker_environment(index, backend='cuda', count=count,
                                             environment={})


@pytest.mark.parametrize('visible', ['', '-1', '2'])
def test_a_changed_or_disabled_allocation_is_rejected(visible):
    with pytest.raises(ValueError, match='visibility changed'):
        accelerator._mask_worker_environment(0, backend='cuda', count=2,
            environment={'CUDA_VISIBLE_DEVICES': visible})
