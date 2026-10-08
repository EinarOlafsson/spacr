"""Worker startup pacing and a distinct serial final overload pass."""
import errno
import multiprocessing
import sqlite3
import time
from pathlib import Path

import pytest

from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from spacr.resource_log import (
    _StaggeredContext, _WorkerStartGate, _iter_parallel_outcomes, _parallel_pool,
    _parallel_process_executor, _parallel_thread_executor,
)
from spacr.runctx import _DeferredOverloadRetries, _is_overload_failure


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _exit_without_result(value):
    import os
    os._exit(value)


def _invalid_field_in_child(value):
    raise ValueError('invalid scientific field')


def _slow_echo(value):
    time.sleep(0.2)
    return value


def _overload_then_invalid_in_child(folder):
    marker = Path(folder) / 'primary-error-recorded'
    if not marker.exists():
        marker.write_text('failed once')
        raise MemoryError('first overload at original calculation')
    raise ValueError('final invalid result at different calculation')


def test_processing_error_keeps_its_real_remote_worker_stack():
    import traceback
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(delay=0)) as pool:
        with pytest.raises(ValueError, match='invalid scientific field') as failure:
            pool.map(_invalid_field_in_child, [1])
    formatted = ''.join(traceback.format_exception(failure.value))
    assert 'in _invalid_field_in_child' in formatted
    assert "raise ValueError('invalid scientific field')" in formatted


def test_exhausted_retry_keeps_both_real_remote_calculation_tracebacks(tmp_path):
    import traceback
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(delay=0)) as pool:
        with pytest.raises(ValueError, match='final invalid result') as failure:
            pool.map(_overload_then_invalid_in_child, [str(tmp_path)])
    formatted = ''.join(traceback.format_exception(failure.value))
    assert "raise MemoryError('first overload at original calculation')" in formatted
    assert "raise ValueError('final invalid result at different calculation')" in formatted


@pytest.mark.parametrize('method', ['map', 'starmap', 'map_async', 'starmap_async', 'imap', 'imap_unordered', 'apply_async'])
def test_native_worker_exit_fails_instead_of_hanging_or_guessing_overload(method):
    context = multiprocessing.get_context('spawn')
    started = time.monotonic()
    with _parallel_pool(1, context=context, gate=_WorkerStartGate(delay=0)) as pool:
        values = (7,) if method == 'apply_async' else [(7,)] if 'star' in method else [7]
        with pytest.raises(RuntimeError, match='exited with code 7; no explicit overload'):
            result = getattr(pool, method)(_exit_without_result, values)
            if 'async' in method:
                result.get(timeout=15)
            elif method.startswith('imap'):
                list(result)
    assert time.monotonic() - started < 15


@pytest.mark.parametrize('operation', ['wait', 'ready'])
def test_direct_submission_wait_and_readiness_detect_native_worker_loss(operation):
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(delay=0)) as pool:
        result = pool.apply_async(_exit_without_result, (7,))
        with pytest.raises(RuntimeError, match='exited with code 7; no explicit overload'):
            if operation == 'wait':
                result.wait(timeout=10)
            else:
                deadline = time.monotonic() + 10
                while not result.ready():
                    assert time.monotonic() < deadline
                    time.sleep(0.05)


def test_direct_submission_keeps_timeouts_callbacks_and_success_reporting():
    from multiprocessing import TimeoutError

    completed = []
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(delay=0)) as pool:
        result = pool.apply_async(_slow_echo, (4,), callback=completed.append)
        with pytest.raises(TimeoutError):
            result.get(timeout=0)
        assert result.wait(timeout=0) is None
        result.wait(timeout=10)
        assert result.ready() and result.successful()
        assert result.get(timeout=0) == 4 and completed == [4]


def test_process_executor_default_keeps_the_python39_constructor_contract(monkeypatch):
    import concurrent.futures

    observed = {}
    class Python39Executor:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
            observed.update(workers=max_workers, context=mp_context,
                            initializer=initializer, initargs=initargs)
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Python39Executor)
    _parallel_process_executor(2, mp_context=multiprocessing.get_context('spawn'),
                              initializer=_echo, initargs=(4,), gate=_WorkerStartGate(delay=0))
    assert observed['workers'] == 2 and observed['context'].get_start_method() == 'spawn'
    assert observed['initializer'] is _echo and observed['initargs'] == (4,)


@pytest.mark.parametrize('method', ['imap', 'imap_unordered'])
@pytest.mark.parametrize('chunksize', [1, 3])
def test_streaming_retains_chunk_size_order_and_normal_zero_exit_recycling(method, chunksize):
    context = multiprocessing.get_context('spawn')
    with _parallel_pool(2, context=context, maxtasksperchild=1,
            gate=_WorkerStartGate(delay=0)) as pool:
        result = list(getattr(pool, method)(_echo, range(7), chunksize))
    assert result == list(range(7)) if method == 'imap' else sorted(result) == list(range(7))


@pytest.mark.parametrize('method', ['imap', 'imap_unordered'])
def test_native_loss_inside_a_multi_item_chunk_does_not_wait_forever(method):
    with _parallel_pool(1, context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(delay=0)) as pool:
        with pytest.raises(RuntimeError, match='exited with code 7'):
            list(getattr(pool, method)(_exit_without_result, [7, 7, 7], chunksize=3))


@pytest.mark.parametrize('error', [MemoryError(), sqlite3.OperationalError('database is locked'),
    sqlite3.OperationalError('database is busy'), OSError(errno.ENOMEM, 'memory'),
    OSError(errno.EMFILE, 'files'), RuntimeError('CUDA out of memory')])
def test_only_resource_exhaustion_qualifies(error):
    assert _is_overload_failure(error)


@pytest.mark.parametrize('error', [ValueError('bad image'), sqlite3.IntegrityError('constraint'),
    sqlite3.OperationalError('no such table'), OSError(errno.ENOSPC, 'disk full'),
    RuntimeError('worker crashed'), PipelineCancelled('stop'), KeyboardInterrupt()])
def test_invalid_inputs_cancellation_and_unexplained_crashes_are_not_retried(error):
    assert not _is_overload_failure(error)


def test_ten_second_start_gaps_and_existing_workers_have_no_task_delay():
    clock = Clock()
    gate = _WorkerStartGate(clock=clock.read, sleep=clock.sleep)
    starts = []
    for _ in range(3):
        gate.start(lambda: starts.append(clock.read()))
    assert starts == pytest.approx([0, 10, 20])
    clock.now += 30
    gate.start(lambda: starts.append(clock.read()))
    assert starts[-1] == 50


def test_failed_worker_start_is_still_spaced_before_next_attempt():
    clock = Clock()
    gate = _WorkerStartGate(clock=clock.read, sleep=clock.sleep)
    def fail():
        raise OSError(errno.EAGAIN, 'too many processes')
    with pytest.raises(OSError):
        gate.start(fail)
    assert gate.start(clock.read) == pytest.approx(10)


def test_pool_gate_keeps_the_run_stop_token_on_replacement_threads():
    import threading
    token = CancellationToken()
    with installed_token(token):
        gate = _WorkerStartGate(delay=0)
    token.cancel()
    outcomes = []
    def replacement():
        try:
            gate.start(lambda: outcomes.append('started'))
        except PipelineCancelled:
            outcomes.append('cancelled')
    worker = threading.Thread(target=replacement)
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and outcomes == ['cancelled']


def test_successful_prefix_streams_before_the_primary_queue_is_exhausted():
    seen = []
    def outcomes():
        for index in range(3):
            seen.append(index)
            yield index, True, index, None
    results = _iter_parallel_outcomes(outcomes(), _echo, lambda *_: None)
    assert next(results) == 0 and seen == [0]
    assert list(results) == [1, 2] and seen == [0, 1, 2]


def test_unordered_primary_successes_stream_before_the_final_retry():
    seen = []
    def outcomes():
        for index in [0, 2, 1]:
            seen.append(index)
            yield (index, False, MemoryError(), (index,)) if index == 0 else (
                index, True, index, None)
    def retry(function, arguments):
        assert seen == [0, 2, 1]
        return function(*arguments)
    results = _iter_parallel_outcomes(outcomes(), _echo, retry, ordered=False)
    assert next(results) == 2 and seen == [0, 2]
    assert list(results) == [1, 0]


def test_retry_queue_is_distinct_ordered_bounded_and_not_recursive():
    queue = _DeferredOverloadRetries()
    calls = []
    assert queue.defer('a', MemoryError('original'), lambda: calls.append('a') or 1)
    assert not queue.defer('a', MemoryError(), lambda: calls.append('duplicate'))
    assert not queue.defer('bad', ValueError(), lambda: calls.append('invalid'))
    def still_busy():
        calls.append('b')
        raise sqlite3.OperationalError('database is locked')
    original = sqlite3.OperationalError('database is busy')
    assert queue.defer('b', original, still_busy)
    assert calls == []
    results = list(queue.drain())
    assert calls == ['a', 'b']
    assert results[0] == ('a', 1, None)
    assert results[1][0] == 'b' and results[1][2].__cause__ is original
    assert list(queue.drain()) == []
    with pytest.raises(RuntimeError):
        queue.defer('b', MemoryError(), still_busy)


def _echo(value):
    return value


def _overloaded_field(arguments):
    directory, identity = arguments
    directory = Path(directory)
    started = directory / f'primary-{identity}'
    if not started.exists():
        started.write_text('primary')
        if identity == 0:
            raise sqlite3.OperationalError('database is locked')
        return identity
    assert all((directory / f'primary-{index}').exists() for index in range(4))
    (directory / 'final-retry').write_text('once')
    return identity


def test_spawned_pool_retries_after_every_primary_task_and_preserves_order(tmp_path):
    clock = Clock()
    gate = _WorkerStartGate(clock=clock.read, sleep=clock.sleep)
    with _parallel_pool(2, context=multiprocessing.get_context('spawn'), gate=gate) as pool:
        assert pool.map(_overloaded_field, [(str(tmp_path), index) for index in range(4)]) == [0, 1, 2, 3]
    assert (tmp_path / 'final-retry').read_text() == 'once'
    assert clock.now == pytest.approx(10)


def test_real_worker_process_deployments_are_ten_seconds_apart():
    starts = []
    class ObservedGate(_WorkerStartGate):
        def start(self, call):
            def deploy():
                starts.append(time.monotonic())
                return call()
            return super().start(deploy)
    with _parallel_pool(2, context=multiprocessing.get_context('spawn'), gate=ObservedGate()) as pool:
        assert pool.map(_echo, [1, 2]) == [1, 2]
    assert len(starts) == 2 and starts[1] - starts[0] >= 10


@pytest.mark.parametrize('method', ['map_async', 'starmap_async'])
def test_async_map_waits_for_final_pass_and_preserves_order(tmp_path, method):
    callbacks = []
    arguments = [(str(tmp_path), index) for index in range(4)]
    inputs = arguments if method == 'map_async' else [(item,) for item in arguments]
    with _parallel_pool(2, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(delay=0)) as pool:
        result = getattr(pool, method)(_overloaded_field, inputs,
                                      callback=callbacks.append)
        assert result.get(timeout=20) == [0, 1, 2, 3]
        assert result.ready() and result.successful()
        assert callbacks == [[0, 1, 2, 3]]
    assert (tmp_path / 'final-retry').read_text() == 'once'


def test_real_torch_loader_accepts_paced_context_and_keeps_seed_initializer():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from spacr.runctx import seed_worker

    starts = []
    class ObservedGate(_WorkerStartGate):
        def start(self, call):
            starts.append(time.monotonic())
            return super().start(call)
    context = _StaggeredContext(multiprocessing.get_context('spawn'),
                                ObservedGate(delay=0))
    dataset = TensorDataset(torch.arange(6))
    loader = DataLoader(dataset, batch_size=2, num_workers=2,
                        multiprocessing_context=context, worker_init_fn=seed_worker)
    assert torch.cat([batch[0] for batch in loader]).tolist() == list(range(6))
    assert len(starts) == 2 and loader.worker_init_fn is seed_worker


def test_cloudpickle_tasks_keep_closures_keywords_and_returned_closures():
    from spacr.resource_log import _parallel_cloudpickle_map

    offset = 7
    def compute(value, *, multiplier):
        return lambda: (value + offset) * multiplier
    values = _parallel_cloudpickle_map([
        (compute, (3,), {'multiplier': 2}),
        (compute, (4,), {'multiplier': 3})], 2)
    assert [value() for value in values] == [20, 33]


def test_failed_mask_write_preserves_original_for_its_final_retry(tmp_path, monkeypatch):
    import numpy as np
    from spacr import utils

    path = tmp_path / 'mask.npy'
    original = np.zeros((8, 8), dtype=np.uint16)
    original[1:4, 1:4] = 1
    np.save(path, original)
    def failed_save(temporary, filtered):
        np.save(temporary, np.ones((1, 1), dtype=np.uint16))
        raise OSError(errno.ENOMEM, 'write buffer exhausted')
    monkeypatch.setattr(utils, '_save_image', failed_save)
    with pytest.raises(OSError):
        utils._process_single_fov(str(path), None, None, False, 0, 0, 0, False)
    assert np.array_equal(np.load(path), original)
    assert not list(tmp_path.glob('.spacr-mask-*'))


def test_sequencing_retry_spools_large_chunks_and_runs_only_when_drained(tmp_path):
    from spacr.sequencing import _ChunkOverloadRetries

    calls, saved = [], []
    class Result:
        def get(self):
            assert calls == ['primary-1', 'primary-2']
            calls.append('final')
            return ('df', 'counts', 'qc')
    class Pool:
        def apply_async(self, function, args):
            assert args[0] == {'original': 'reads' * 1024}
            return Result()
    class Queue:
        def put(self, result):
            saved.append(result)
    retry = _ChunkOverloadRetries(Pool(), Queue(), None, str(tmp_path / 'out.h5'))
    assert retry.defer(1, {'original': 'reads' * 1024}, MemoryError('overloaded'))
    assert list(tmp_path.rglob('*.pkl')) and calls == []
    calls.extend(['primary-1', 'primary-2'])
    retry.drain()
    retry.drain()
    assert calls == ['primary-1', 'primary-2', 'final']
    assert saved == [('df', 'counts', 'qc')]
    assert not list(tmp_path.iterdir())


def test_sequencing_final_failure_does_not_drop_later_deferred_chunks(tmp_path, monkeypatch):
    from spacr import sequencing

    calls, saved, aborted = [], [], []
    class Result:
        def __init__(self, identity):
            self.identity = identity
        def get(self):
            calls.append(self.identity)
            if self.identity == 1:
                raise ValueError('final invalid chunk')
            return ('later data', 'counts', 'qc')
    class Pool:
        def apply_async(self, function, args):
            return Result(args[0]['identity'])
    class Queue:
        def put(self, result):
            saved.append(result)
    monkeypatch.setattr(sequencing, '_abort_chunk_workers',
                        lambda *args: aborted.append(True))
    retry = sequencing._ChunkOverloadRetries(Pool(), Queue(), None, str(tmp_path / 'out.h5'))
    for identity in [1, 2]:
        assert retry.defer(identity, {'identity': identity}, MemoryError('primary overloaded'))
    with pytest.raises(ValueError, match='final invalid chunk'):
        retry.drain()
    assert calls == [1, 2] and saved == [('later data', 'counts', 'qc')]
    assert aborted == [True]
    assert len(list(tmp_path.rglob('*.pkl'))) == 1
    final_error = list(tmp_path.rglob('*.final-error.txt'))
    assert len(final_error) == 1 and 'final invalid chunk' in final_error[0].read_text()


def test_thread_pool_uses_the_same_final_queue_rule(tmp_path):
    clock = Clock()
    gate = _WorkerStartGate(clock=clock.read, sleep=clock.sleep)
    with _parallel_thread_executor(2, gate=gate) as pool:
        assert list(pool.map(_overloaded_field, [(str(tmp_path), index) for index in range(4)])) == [0, 1, 2, 3]
    assert (tmp_path / 'final-retry').exists()


@pytest.mark.parametrize('method', ['imap', 'imap_unordered'])
def test_streaming_process_pool_uses_final_queue_after_all_primary_tasks(tmp_path, method):
    clock = Clock()
    with _parallel_pool(2, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(clock=clock.read, sleep=clock.sleep)) as pool:
        results = list(getattr(pool, method)(
            _overloaded_field, [(str(tmp_path), index) for index in range(4)]))
    assert sorted(results) == [0, 1, 2, 3]
    if method == 'imap':
        assert results == [0, 1, 2, 3]
    else:
        assert results[-1] == 0
    assert (tmp_path / 'final-retry').exists()


def test_spawned_process_executor_retains_order_and_the_primary_retry_boundary(tmp_path):
    clock = Clock()
    with _parallel_process_executor(2, mp_context=multiprocessing.get_context('spawn'),
            gate=_WorkerStartGate(clock=clock.read, sleep=clock.sleep)) as executor:
        assert list(executor.map(_overloaded_field,
            [(str(tmp_path), index) for index in range(4)])) == [0, 1, 2, 3]
    assert (tmp_path / 'final-retry').exists()


@pytest.mark.parametrize('second_failure', [False, True])
def test_sweep_final_retry_follows_all_primaries_keeps_one_csv_row_and_original_errors(
        tmp_path, monkeypatch, second_failure):
    from concurrent.futures import Future
    import concurrent.futures
    import pandas as pd
    from spacr import parameter_sweep as sweep

    calls, contexts = [], []
    class InlineExecutor:
        def __init__(self, max_workers, mp_context):
            contexts.append(mp_context)
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def submit(self, function, payload):
            future = Future()
            try:
                future.set_result(function(payload))
            except BaseException as error:
                future.set_exception(error)
            return future
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', InlineExecutor)
    monkeypatch.setattr(sweep, 'recommended_workers', lambda **_kw: (2, 'test'))
    monkeypatch.setattr(sweep, 'memory_is_low', lambda **_kw: False)
    def compute(payload):
        identity = payload[1]['trial_id']
        calls.append(identity)
        if identity == 1 and calls.count(1) == 1:
            folder = tmp_path / 'trial_0001'
            folder.mkdir()
            (folder / 'error.txt').write_text('original complete traceback')
            (folder / '_trial_result.json').write_text('original result')
            raise sqlite3.OperationalError('database is locked')
        if identity == 1 and second_failure:
            raise MemoryError('final failure')
        return {'trial_id': identity, 'status': 'ok', 'regression_type': 'ols'}
    monkeypatch.setattr(sweep, '_execute_trial', compute)
    space = sweep.SweepSpace(axes={'fraction_threshold': [0.1, 0.2, 0.3]},
                            fixed={'regression_type': 'ols'})
    result = sweep.run_sweep_parallel({'ram_guard': False}, tmp_path, space,
                                     n_jobs=2, progress_every=0)
    assert calls == [1, 2, 3, 1]
    assert len(contexts) == 1 and isinstance(contexts[0], _StaggeredContext)
    assert contexts[0].get_start_method() == 'spawn'
    assert result['trial_id'].tolist() == [1, 2, 3]
    assert result.iloc[0]['status'] == ('failed' if second_failure else 'ok')
    assert result.iloc[0]['primary_error_type'] == 'OperationalError'
    assert result.iloc[0]['primary_error'] == 'database is locked'
    assert bool(result.iloc[0]['overload_retry'])
    assert (tmp_path / 'trial_0001' / 'error.txt.primary').read_text() == 'original complete traceback'
    assert (tmp_path / 'trial_0001' / '_trial_result.json.primary').read_text() == 'original result'
    saved = pd.read_csv(tmp_path / 'sweep_results.csv')
    assert saved['trial_id'].tolist() == [1, 2, 3]
    assert saved.iloc[0]['status'] == result.iloc[0]['status']
    settings = sweep.settings_for_trial({}, result.iloc[0].to_dict())
    assert not set(settings) & {'overload_retry', 'primary_error', 'primary_error_type', 'primary_seconds'}


def test_contained_retry_cannot_read_the_previous_attempts_result(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import subprocess
    from spacr import parameter_sweep as sweep
    result = tmp_path / '_trial_result.json'
    result.write_text('{"status":"ok","trial_id":7}')
    monkeypatch.setattr(sweep, 'containment_available', lambda: False)
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kw: SimpleNamespace(
        returncode=137, stderr='worker killed without result'))
    row = sweep.run_trial_contained({'src': str(tmp_path)}, trial_id=7)
    assert row['status'] == 'killed'
    assert not result.exists() and not row.get('_overload', False)


@pytest.mark.parametrize('error,expected', [
    (MemoryError('buffer exhausted'), True),
    (sqlite3.OperationalError('database is locked'), True),
    (ValueError('bad model input'), False)])
def test_sweep_child_reports_explicit_overload_without_losing_original_traceback(
        tmp_path, monkeypatch, error, expected):
    import json
    from spacr import ml, sweep_child
    settings, result = tmp_path / 'settings.json', tmp_path / 'result.json'
    settings.write_text(json.dumps({'settings': {'src': str(tmp_path)}, 'trial_id': 4}))
    def fail(_settings):
        raise error
    monkeypatch.setattr(ml, 'perform_regression', fail)
    assert sweep_child.main([str(settings), str(result)]) == 0
    row = json.loads(result.read_text())
    assert row['_overload'] is expected
    assert row['status'] == 'failed' and row['error_type'] == type(error).__name__
    assert type(error).__name__ in (tmp_path / 'error.txt').read_text()
