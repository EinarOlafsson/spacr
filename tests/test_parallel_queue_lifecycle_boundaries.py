"""Real owned pools preserve final errors, cooperative stop and recycling."""

import multiprocessing
from pathlib import Path
import sys
import time

import pytest

from spacr.cancellation import PipelineCancelled
from spacr.resource_log import (
    _WorkerStartGate, _parallel_chunks, _parallel_cloudpickle_map,
    _parallel_pool, _parallel_process_executor, _parallel_thread_executor,
)


def _invalid(value):
    raise ValueError('invalid scientific input')


def _slow(value):
    time.sleep(0.2)
    return value


def test_async_map_reports_original_failure_through_callback_and_result():
    errors = []
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(delay=0)) as pool:
        result = pool.map_async(_invalid, [1, 2], error_callback=errors.append)
        with pytest.raises(ValueError, match='invalid scientific input'):
            result.get(timeout=15)
        assert result.ready() and not result.successful()
        assert result.wait(timeout=0) is None
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


def test_terminating_owned_async_pool_joins_coordinator_and_reports_stop():
    from multiprocessing import TimeoutError

    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(delay=0)) as pool:
        result = pool.map_async(_slow, [1, 2])
        with pytest.raises(ValueError, match='not ready'):
            result.successful()
        with pytest.raises(TimeoutError):
            result.get(timeout=0)
        pool.terminate()
        with pytest.raises(PipelineCancelled, match='pool stopped'):
            result.get(timeout=5)
        assert result.ready() and not result._thread.is_alive()


@pytest.mark.parametrize('size', [0, -1])
def test_invalid_streaming_chunk_size_does_not_consume_scientific_inputs(size):
    seen = []

    def inputs():
        seen.append('consumed')
        yield 1

    with pytest.raises(ValueError, match='Chunksize'):
        list(_parallel_chunks(inputs(), size))
    assert seen == []


def test_failed_serial_final_attempt_keeps_error_and_finishes_other_primaries():
    seen = []

    def calculate(value):
        seen.append(value)
        if value == 0:
            if seen.count(0) == 1:
                raise MemoryError('primary exhausted')
            raise ValueError('final scientific error')
        return value

    with pytest.raises(ValueError, match='final scientific error') as error:
        _parallel_cloudpickle_map([(calculate, (value,), {}) for value in [0, 1]], workers=1)
    assert seen == [0, 1, 0]
    assert isinstance(error.value.__cause__, MemoryError)


def test_direct_executor_submission_retains_the_original_future_interface():
    with _parallel_thread_executor(1, gate=_WorkerStartGate(delay=0)) as executor:
        future = executor.submit(abs, -3)
        assert future.result(timeout=5) == 3


@pytest.mark.skipif(sys.version_info < (3, 11), reason='Python added executor recycling in 3.11')
def test_process_executor_recycling_preserves_all_ordered_results():
    with _parallel_process_executor(1, mp_context=multiprocessing.get_context('spawn'),
                                    max_tasks_per_child=1,
                                    gate=_WorkerStartGate(delay=0)) as executor:
        assert list(executor.map(abs, [-3, -4])) == [3, 4]


def test_ops_missing_reference_loads_all_declared_slots_before_skip():
    from spacr import ops_engine

    result = ops_engine._decode_field({
        'site': 1, 'cycles': [1], 'reference': 1, 'gpu': False,
        'planes': {1: [None]},
    })
    assert result['site'] == 1 and result['missing'] == [1]
    assert result['skipped'] == 'the reference cycle could not be read'
    assert result['unreadable'] == []


def test_ops_corrupt_plane_is_reported_separately_from_resource_exhaustion(tmp_path):
    from spacr import ops_engine

    path = tmp_path / 'corrupt.tif'
    path.write_bytes(b'invalid image bytes')
    unreadable = []
    assert ops_engine._read_plane((str(path), None), unreadable) is None
    assert len(unreadable) == 1 and unreadable[0][0] == str(path)
    assert 'TiffFileError' in unreadable[0][1]


def test_spacr_prefetch_loader_adapter_keeps_full_epoch():
    from spacr.io import spacrDataLoader

    loader = spacrDataLoader([3, 4], batch_size=1, num_workers=0, preload_batches=1)
    try:
        assert [batch.item() for batch in loader] == [3, 4]
    finally:
        loader.cleanup()


def _echo_with_worker_identity(value):
    import os
    return os.getpid(), value


def _record_worker_calculation(payload):
    folder, value = payload
    with (Path(folder) / f'{value}.txt').open('a') as handle:
        handle.write('calculated\n')
    return _echo_with_worker_identity(value)


def test_recycled_processing_worker_obeys_default_ten_second_start_spacing():
    started = time.monotonic()
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
                        maxtasksperchild=1) as pool:
        result = pool.map_async(_echo_with_worker_identity, [17, 23], chunksize=1)
        rows = result.get(timeout=45)
        assert [value for _pid, value in rows] == [17, 23]
        assert len({pid for pid, _value in rows}) == 2
        assert time.monotonic() - started >= 9.0


@pytest.mark.parametrize('operation', ['map_async', 'starmap_async'])
def test_failed_result_callback_is_reported_without_repeating_scientific_work(tmp_path, operation):
    received = []

    def failed_callback(values):
        received.append(values)
        raise OSError('could not record completed scientific results')

    with _parallel_pool(1, context=multiprocessing.get_context('spawn')) as pool:
        payloads = [(str(tmp_path), value) for value in [17, 23]]
        inputs = [(payload,) for payload in payloads] if operation == 'starmap_async' else payloads
        result = getattr(pool, operation)(_record_worker_calculation, inputs,
                                         callback=failed_callback)
        with pytest.raises(OSError, match='could not record completed'):
            result.get(timeout=15)
        assert result.ready() and not result.successful()
    assert len(received) == 1
    assert [value for _pid, value in received[0]] == [17, 23]
    assert [(tmp_path / f'{value}.txt').read_text() for value in [17, 23]] == ['calculated\n', 'calculated\n']


def test_unordered_processing_preserves_original_invalid_error_after_primary_results():
    with _parallel_pool(1, context=multiprocessing.get_context('spawn')) as pool:
        with pytest.raises(ValueError, match='invalid scientific input'):
            list(pool.imap_unordered(_invalid, [17, 23]))


def _cancel_inside_processing_child(path):
    with Path(path).open('a') as handle:
        handle.write('primary\n')
    raise PipelineCancelled('cancelled inside the processing child')


@pytest.mark.parametrize('operation', ['map_async', 'starmap_async', 'imap_unordered'])
def test_child_cancellation_keeps_its_original_reason_without_final_retry(tmp_path, operation):
    attempt = tmp_path / 'attempts.txt'
    with _parallel_pool(1, context=multiprocessing.get_context('spawn')) as pool:
        inputs = [(str(attempt),)] if operation == 'starmap_async' else [str(attempt)]
        with pytest.raises(PipelineCancelled, match='cancelled inside the processing child'):
            result = getattr(pool, operation)(_cancel_inside_processing_child, inputs)
            if operation == 'imap_unordered':
                list(result)
            else:
                result.get(timeout=15)
        if operation != 'imap_unordered':
            assert result.ready() and not result.successful()
    assert attempt.read_text() == 'primary\n'


def test_parent_cancellation_joins_async_coordinator_without_success_callback():
    from spacr.cancellation import CancellationToken, installed_token

    token = CancellationToken('parent requested Stop')
    completed = []
    with installed_token(token):
        with _parallel_pool(1, context=multiprocessing.get_context('spawn')) as pool:
            result = pool.map_async(_slow, [17, 23], callback=completed.append)
            token.cancel()
            with pytest.raises(PipelineCancelled, match='parent requested Stop'):
                result.get(timeout=15)
        assert result.ready() and not result._thread.is_alive()
    assert completed == []


def _send_invalid_mask_status(messages, device):
    messages.put(('finished', device, ('unknown_status', 'malformed worker report')))


class _InvalidStatusContext:
    def __init__(self):
        self.real = multiprocessing.get_context('spawn')

    def __getattr__(self, name):
        return getattr(self.real, name)

    def Process(self, *, target, args, name):
        from spacr._mask_workers import _mask_worker
        assert target is _mask_worker
        return self.real.Process(target=_send_invalid_mask_status,
                                 args=(args[6], args[3]), name=name)


def test_malformed_mask_worker_status_fails_and_preserves_original_archive(tmp_path):
    import numpy as np
    from spacr._mask_workers import _run_mask_workers

    path = tmp_path / 'batch.npz'
    np.savez(path, images=np.array([17, 23], dtype=np.uint16))
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match='Invalid worker status: unknown_status'):
        _run_mask_workers(str(tmp_path), {}, 'cell', {0: [str(path)]},
                          {0: {'CUDA_VISIBLE_DEVICES': ''}},
                          context=_InvalidStatusContext())
    assert path.read_bytes() == original
    assert not [child for child in multiprocessing.active_children()
                if child.name.startswith('spacr-mask-gpu-')]
