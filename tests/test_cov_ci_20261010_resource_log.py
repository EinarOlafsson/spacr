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
