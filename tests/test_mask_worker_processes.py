"""Actual spawned CPU workers exercise the mask coordinator's process protocol."""

import json
import multiprocessing
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacr._mask_workers import _mask_worker, _partition_batches, _run_mask_workers
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token


def _cpu_segmenter(src, settings, object_type, *, batch_paths, on_batch_done, run_qc):
    """Save real arrays while emulating one persistent model per child."""
    from spacr.cancellation import checkpoint

    assert not run_qc
    selected = os.environ['CUDA_VISIBLE_DEVICES']
    metadata = {'pid': os.getpid(), 'selected': selected, 'start': time.monotonic()}
    if settings.get('barrier'):
        settings['barrier'].wait(timeout=15)
    if settings.get('crash') == selected:
        os._exit(7)
    if settings.get('silent') == selected:
        os._exit(0)
    if settings.get('fail') == selected:
        raise ValueError('simulated inference failure')
    for path in batch_paths:
        checkpoint()
        for _ in range(10):
            time.sleep(0.01)
            checkpoint()
        target = Path(src) / f'{Path(path).stem}.npy'
        np.save(target, np.array([[0, 1], [1, 1]], dtype=np.uint16))
        if not settings.get('omit'):
            on_batch_done(path)
        if settings.get('duplicate'):
            on_batch_done(path)
    metadata['end'] = time.monotonic()
    (Path(src) / f'worker-{selected}.json').write_text(json.dumps(metadata))
    print(f'worker {selected} finished')


def _setup(tmp_path):
    inputs = []
    for index in range(4):
        path = tmp_path / f'batch{index}.npz'
        path.write_bytes(f'original input {index}'.encode())
        inputs.append(str(path))
    assignments = {0: inputs[:2], 1: inputs[2:]}
    environments = {0: {**os.environ, 'CUDA_VISIBLE_DEVICES': '7'},
                    1: {**os.environ, 'CUDA_VISIBLE_DEVICES': '3'}}
    return assignments, environments


def test_processes_run_in_parallel_with_disjoint_inputs_and_private_environments(tmp_path):
    assignments, environments = _setup(tmp_path)
    context = multiprocessing.get_context('spawn')
    before = dict(os.environ)
    updates = []
    result = _run_mask_workers(str(tmp_path), {'barrier': context.Barrier(2)}, 'cell',
        assignments, environments, context=context, segmenter=_cpu_segmenter,
        on_progress=updates.append)
    assert os.environ == before
    assert result['total_batches'] == len(result['completed_batches']) == 4
    assert all(row['state'] == 'success' for row in result['workers'].values())
    records = [json.loads(path.read_text()) for path in tmp_path.glob('worker-*.json')]
    assert len({row['pid'] for row in records}) == 2
    assert os.getpid() not in {row['pid'] for row in records}
    assert {row['selected'] for row in records} == {'3', '7'}
    assert max(row['start'] for row in records) < min(row['end'] for row in records)
    assert len(list(tmp_path.glob('*.npy'))) == 4
    assert len(list(tmp_path.glob('*.npz'))) == 4
    assert updates[0]['completed_batches'] == []
    assert updates[-1] == result


@pytest.mark.parametrize('failure', ['fail', 'crash', 'duplicate', 'silent', 'omit'])
def test_failed_crashed_or_misreporting_workers_stop_the_run(tmp_path, failure):
    assignments, environments = _setup(tmp_path)
    with pytest.raises(RuntimeError, match='simulated inference failure|code [07]|repeated batch|without completing'):
        _run_mask_workers(str(tmp_path), {failure: '7'}, 'cell', assignments,
                          environments, segmenter=_cpu_segmenter)
    assert len(list(tmp_path.glob('*.npz'))) == 4
    assert not [child for child in multiprocessing.active_children()
                if child.name.startswith('spacr-mask-gpu-')]


@pytest.mark.parametrize('initialized,available,count,success', [
    (False, True, 1, True), (True, True, 1, False),
    (False, False, 0, False), (False, True, 2, False)])
def test_default_worker_checks_its_binding_before_loading_the_generator(
        tmp_path, monkeypatch, initialized, available, count, success):
    from spacr import accelerator

    calls = []

    def segment(src, settings, role, **kwargs):
        calls.append((os.environ['CUDA_VISIBLE_DEVICES'], kwargs['run_qc']))
        kwargs['on_batch_done'](kwargs['batch_paths'][0])

    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: initialized, is_available=lambda: available,
        device_count=lambda: count))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setitem(sys.modules, 'spacr.object',
                        SimpleNamespace(generate_cellpose_masks_sam=segment))
    monkeypatch.setattr(accelerator, '_CACHED', None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'parent-allocation')
    messages = queue.Queue()
    _mask_worker(str(tmp_path), {}, 'cell', 0, [str(tmp_path / 'batch.npz')],
                 {'CUDA_VISIBLE_DEVICES': 'assigned-device'}, messages,
                 threading.Event())
    records = []
    while not messages.empty():
        records.append(messages.get())
    assert records[-1][0] == 'finished'
    assert records[-1][2][0] == ('success' if success else 'failed')
    assert calls == ([('assigned-device', False)] if success else [])
    if initialized:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == 'parent-allocation'


def test_cancellation_retains_completed_outputs_and_all_inputs(tmp_path):
    assignments, environments = _setup(tmp_path)
    token = CancellationToken()
    completed = set()

    def progress(state):
        completed.update(state['completed_batches'])
        if completed:
            token.cancel()

    with installed_token(token), pytest.raises(PipelineCancelled):
        _run_mask_workers(str(tmp_path), {}, 'cell', assignments, environments,
                          segmenter=_cpu_segmenter, on_progress=progress)
    assert 0 < len(completed) < 4
    for path in completed:
        assert np.load(tmp_path / f'{Path(path).stem}.npy').max() == 1
    assert len(list(tmp_path.glob('*.npz'))) == 4
    assert not [child for child in multiprocessing.active_children()
                if child.name.startswith('spacr-mask-gpu-')]


def test_bad_assignments_fail_before_starting_processes(tmp_path):
    assignments, environments = _setup(tmp_path)
    assignments[1].append(assignments[0][0])
    with pytest.raises(ValueError, match='more than once'):
        _run_mask_workers(str(tmp_path), {}, 'cell', assignments, environments)


def test_partition_assigns_every_archive_once_and_keeps_devices_explicit():
    records = [{'path': f'/batch{index}.npz', 'bytes': size}
               for index, size in enumerate([8, 4, 4, 2])]
    result = _partition_batches(records, [5, 2])
    assert result == {5: ['/batch0.npz', '/batch3.npz'],
                      2: ['/batch1.npz', '/batch2.npz']}
    assert records[0]['bytes'] == 8
    with pytest.raises(ValueError, match='unique paths'):
        _partition_batches(records + records[:1], [5, 2])
    with pytest.raises(ValueError, match='distinct'):
        _partition_batches(records, [2, 2])
