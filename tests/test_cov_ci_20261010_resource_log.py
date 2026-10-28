"""Worker tracking, context delegation and final-retry polling in resource_log."""
from __future__ import annotations

import multiprocessing
from multiprocessing import TimeoutError

import pytest

from spacr.resource_log import (
    _ParallelPool, _StaggeredContext, _check_parallel_workers)


class _Worker:
    def __init__(self, pid, exitcode=None):
        self.pid = pid
        self.exitcode = exitcode


class _Backend:
    def __init__(self, workers=()):
        self._pool = list(workers)


def test_new_pool_workers_are_tracked_and_a_failed_one_is_reported():
    first = _Worker(1)
    backend = _Backend([first])
    tracked = []
    _check_parallel_workers(backend, tracked)
    assert tracked == [first]

    replacement = _Worker(2, exitcode=0)
    backend._pool = [replacement]
    _check_parallel_workers(backend, tracked)
    assert tracked == [first, replacement]

    first.exitcode = -9
    with pytest.raises(RuntimeError, match='Processing worker 1 exited with code -9'):
        _check_parallel_workers(backend, tracked)


def test_staggered_context_delegates_unknown_attributes_to_the_real_context():
    real = multiprocessing.get_context('spawn')
    context = _StaggeredContext(real)
    assert context._name == 'spawn'
    assert context.__dict__.get('_name') is None
    with pytest.raises(AttributeError):
        context.no_such_context_attribute


class _SlowResult:
    def __init__(self, value, timeouts):
        self.value = value
        self.timeouts = timeouts
        self.calls = []

    def get(self, timeout=None):
        self.calls.append(timeout)
        if len(self.calls) <= self.timeouts:
            raise TimeoutError
        return self.value


class _RetryBackend(_Backend):
    def __init__(self):
        super().__init__([_Worker(7)])
        self.submitted = []
        self.result = None

    def apply_async(self, function, arguments):
        self.submitted.append((function, arguments))
        self.result = _SlowResult(function(*arguments), timeouts=2)
        return self.result


def test_final_retry_keeps_polling_through_timeouts_until_the_result_arrives():
    backend = _RetryBackend()
    pool = _ParallelPool(backend)
    assert pool._retry(pow, (2, 5)) == 32
    assert backend.submitted == [(pow, (2, 5))]
    assert backend.result.calls == [0.1, 0.1, 0.1]


class _RecordedNativeCall:
    """Record requested Win32 operations without making an operating-system call."""

    def __init__(self, name, result, calls):
        self.name, self.result, self.calls = name, result, calls

    def __call__(self, *args):
        self.calls.append((self.name, args))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _JobKernel:
    def __init__(self, create=71, information=1, opened=81, assigned=1):
        self.calls = []
        for name, result in [('CreateJobObjectW', create), ('SetInformationJobObject', information),
                             ('OpenProcess', opened), ('AssignProcessToJobObject', assigned), ('CloseHandle', 1)]:
            setattr(self, name, _RecordedNativeCall(name, result, self.calls))


@pytest.mark.parametrize('create,information', [(71, 1), (0, 1), (71, 0)])
def test_windows_parent_job_requests_kill_on_close_and_cleans_failed_handles(monkeypatch, create, information):
    import ctypes
    from spacr import resource_log as rl

    kernel = _JobKernel(create=create, information=information)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 5, raising=False)
    monkeypatch.setattr(rl, '_PARENT_JOB', None)
    if not create or not information:
        with pytest.raises(OSError):
            rl._parent_job()
        assert rl._PARENT_JOB is None
        assert [args[0] for name, args in kernel.calls if name == 'CloseHandle'] == ([71] if create else [])
    else:
        assert rl._parent_job() == rl._parent_job() == 71
        assert [name for name, _args in kernel.calls] == ['CreateJobObjectW', 'SetInformationJobObject']
        job, kind, pointer, size = kernel.calls[1][1]
        assert job == 71 and kind == 9
        assert pointer._obj.BasicLimitInformation.LimitFlags == 0x2000
        assert size == ctypes.sizeof(pointer._obj)


@pytest.mark.parametrize('opened,assigned,expected', [(0, 1, False), (81, 0, False), (81, 1, True),
                                                     (81, OSError('assignment refused'), False)])
def test_windows_binding_closes_only_the_opened_process_handle(monkeypatch, opened, assigned, expected):
    import ctypes
    from types import SimpleNamespace
    from spacr import resource_log as rl

    kernel = _JobKernel(opened=opened, assigned=assigned)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(rl, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(rl, '_parent_job', lambda: 71)
    assert rl._bind_to_parent_job(1234) is expected
    assert kernel.calls[0] == ('OpenProcess', (0x0101, False, 1234))
    assert [args[0] for name, args in kernel.calls if name == 'CloseHandle'] == ([81] if opened else [])
    if opened:
        assert kernel.calls[1] == ('AssignProcessToJobObject', (71, 81))


@pytest.mark.parametrize('starts_done', [False, True])
def test_pool_close_is_idempotent_and_waits_for_pending_starts(starts_done):
    import functools
    import queue
    import threading
    from types import SimpleNamespace
    from spacr import resource_log as rl

    notifier = queue.SimpleQueue()
    pool = SimpleNamespace(_state=multiprocessing.pool.RUN, _starts_lock=threading.Lock(),
                           _closing=False, _starts_done=starts_done, _change_notifier=notifier,
                           _worker_handler=SimpleNamespace(_state=multiprocessing.pool.RUN))
    pool._hand_over_close = functools.partial(rl._PacedPool._hand_over_close, pool)
    rl._PacedPool.close(pool)
    rl._PacedPool.close(pool)
    assert pool._state == multiprocessing.pool.CLOSE and pool._closing
    assert notifier.empty() is (not starts_done)
    if starts_done:
        assert notifier.get_nowait() is None and notifier.empty()
        assert pool._worker_handler._state == multiprocessing.pool.CLOSE
    stopped = SimpleNamespace(_starts_stopped=lambda: True)
    assert rl._PacedPool._start_worker(stopped) is None
