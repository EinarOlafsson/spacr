"""A paced pool runs work on its first worker while the others start.

The stock ``Pool`` constructor starts every worker before it returns, so a
pool built on a paced context waited (N-1) gate intervals before any task
ran: 70 s for eight workers at the production ten seconds. ``_PacedPool``
starts one worker in the constructor and the rest from a background starter
that still honours the gate, so the stagger that protects RAM is kept.
"""
import errno
import multiprocessing
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from spacr.cancellation import CancellationToken, installed_token
from spacr import resource_log as R
from spacr.resource_log import _PacedPool, _WorkerStartGate, _parallel_pool

GIB = 1024 ** 3


def _identity(value):
    return value, os.getpid()


def _hold(seconds):
    time.sleep(seconds)
    return os.getpid()


def _interval(value):
    begin = time.monotonic()
    time.sleep(0.3)
    return value, os.getpid(), begin, time.monotonic()


class _ObservedGate(_WorkerStartGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starts = []

    def start(self, call):
        def deploy():
            self.starts.append(time.monotonic())
            return call()
        return super().start(deploy)


def _backend(pool):
    return pool._backend


def _assert_cleanly_stopped(pool):
    backend = _backend(pool)
    assert not backend._starter.is_alive()
    for worker in backend._pool:
        assert worker._popen is not None
        assert not worker.is_alive()
    for worker in pool._workers:
        assert worker._popen is None or not worker.is_alive()


def _methods():
    return [method for method in ('spawn', 'fork', 'forkserver')
            if method in multiprocessing.get_all_start_methods()]


@pytest.mark.parametrize('workers', [1, 4])
def test_first_result_arrives_within_one_gate_interval_whatever_the_worker_count(workers):
    gate = _WorkerStartGate(4.0)
    started = time.monotonic()
    with _parallel_pool(workers, context=multiprocessing.get_context('spawn'),
                        gate=gate) as pool:
        built = time.monotonic() - started
        stream = pool.imap(_identity, range(6))
        first = next(stream)
        first_at = time.monotonic() - started
        rest = list(stream)
    assert [value for value, _pid in [first, *rest]] == list(range(6))
    assert built < 4.0 and first_at < 4.0


@pytest.mark.parametrize('method', _methods())
def test_later_workers_still_start_one_gate_interval_apart(method):
    gate = _ObservedGate(1.0)
    with _parallel_pool(3, context=multiprocessing.get_context(method),
                        gate=gate) as pool:
        pids = pool.map(_hold, [3.5] * 3, chunksize=1)
    assert len(gate.starts) == 3
    gaps = [later - earlier for earlier, later in zip(gate.starts, gate.starts[1:])]
    assert all(gap >= 0.95 for gap in gaps), gaps
    assert len(set(pids)) >= 2


def test_terminate_with_starts_pending_stops_them_and_never_meets_an_unstarted_worker():
    gate = _ObservedGate(30.0)
    started = time.monotonic()
    pool = _parallel_pool(4, context=multiprocessing.get_context('spawn'), gate=gate)
    assert pool.apply_async(_identity, (7,)).get(timeout=20)[0] == 7
    pool.terminate()
    pool.join()
    assert time.monotonic() - started < 15
    assert len(gate.starts) == 1
    _assert_cleanly_stopped(pool)


def test_with_block_exit_with_starts_pending_is_prompt():
    gate = _WorkerStartGate(30.0)
    started = time.monotonic()
    with _parallel_pool(8, context=multiprocessing.get_context('spawn'), gate=gate) as pool:
        assert [value for value, _pid in pool.map(_identity, range(4))] == list(range(4))
    assert time.monotonic() - started < 15
    _assert_cleanly_stopped(pool)


def test_close_and_join_with_starts_pending_and_no_work_left_are_prompt():
    gate = _ObservedGate(30.0)
    started = time.monotonic()
    pool = _parallel_pool(4, context=multiprocessing.get_context('spawn'), gate=gate)
    assert pool.apply_async(_identity, (3,)).get(timeout=20)[0] == 3
    pool.close()
    pool.join()
    assert time.monotonic() - started < 15
    assert len(gate.starts) == 1
    _assert_cleanly_stopped(pool)


def test_close_with_queued_work_still_starts_the_pending_workers_for_it():
    gate = _ObservedGate(1.0)
    pool = _parallel_pool(2, context=multiprocessing.get_context('spawn'), gate=gate)
    results = [pool.apply_async(_hold, (2.5,)) for _ in range(2)]
    pool.close()
    pool.join()
    pids = [result.get(timeout=1) for result in results]
    assert len(gate.starts) == 2 and len(set(pids)) == 2
    _assert_cleanly_stopped(pool)


def test_cancel_event_holds_pending_starts_and_the_pool_still_finishes():
    cancelled = threading.Event()
    gate = _ObservedGate(0.5)
    gate.idle = cancelled.is_set
    with _parallel_pool(4, context=multiprocessing.get_context('spawn'), gate=gate) as pool:
        assert pool.apply_async(_identity, (1,)).get(timeout=20)[0] == 1
        cancelled.set()
        time.sleep(0.2)
        held = len(gate.starts)
        time.sleep(1.5)
        assert len(gate.starts) == held < 4
        assert [value for value, _pid in pool.map(_identity, range(3))] == [0, 1, 2]
    _assert_cleanly_stopped(pool)


def test_run_cancellation_token_stops_pending_starts():
    token = CancellationToken()
    with installed_token(token):
        gate = _ObservedGate(0.5)
    with _parallel_pool(4, context=multiprocessing.get_context('spawn'), gate=gate) as pool:
        assert pool.apply_async(_identity, (1,)).get(timeout=20)[0] == 1
        token.cancel()
        time.sleep(0.2)
        held = len(gate.starts)
        time.sleep(1.5)
        assert len(gate.starts) == held < 4
        assert pool.apply_async(_identity, (2,)).get(timeout=20)[0] == 2
        assert isinstance(_backend(pool)._start_error, Exception)
    _assert_cleanly_stopped(pool)


class _FailSecondGate(_WorkerStartGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def start(self, call):
        self.calls += 1
        if self.calls == 2:
            raise OSError(errno.EAGAIN, 'too many processes')
        return super().start(call)


def test_a_failed_background_start_leaves_the_pool_working_on_its_live_workers(caplog):
    gate = _FailSecondGate(0.2)
    with _parallel_pool(3, context=multiprocessing.get_context('spawn'), gate=gate) as pool:
        assert [value for value, _pid in pool.map(_identity, range(5))] == list(range(5))
        deadline = time.monotonic() + 10
        while _backend(pool)._start_error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        backend = _backend(pool)
        assert isinstance(backend._start_error, OSError)
        assert backend._target == 1
        assert [value for value, _pid in pool.map(_identity, range(3))] == [0, 1, 2]
    assert 'failed to start' in caplog.text
    _assert_cleanly_stopped(pool)


class _FailFirstGate(_WorkerStartGate):
    def start(self, call):
        raise OSError(errno.EAGAIN, 'too many processes')


def test_a_failed_first_start_still_raises_from_the_constructor():
    with pytest.raises(OSError):
        _parallel_pool(3, context=multiprocessing.get_context('spawn'), gate=_FailFirstGate(0))


@pytest.mark.parametrize('method', _methods())
def test_recycled_workers_are_replaced_without_exceeding_the_pool_size(method):
    with _parallel_pool(2, context=multiprocessing.get_context(method),
                        maxtasksperchild=1, gate=_WorkerStartGate(0)) as pool:
        rows = pool.map(_interval, range(6), chunksize=1)
        assert [value for value, *_ in rows] == list(range(6))
        assert len({pid for _value, pid, _begin, _end in rows}) == 6
        events = sorted([(begin, 1) for *_, begin, _end in rows]
                        + [(end, -1) for *_, end in rows], key=lambda item: (item[0], item[1]))
        running = peak = 0
        for _when, step in events:
            running += step
            peak = max(peak, running)
        assert peak <= 2
        assert len(pool._workers) <= 6 + 2
    _assert_cleanly_stopped(pool)


def test_default_chunks_are_sized_for_the_full_pool_while_starts_are_pending():
    with _parallel_pool(4, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(30.0)) as pool:
        result = _backend(pool).map_async(_identity, range(32))
        assert result._chunksize == 2
        assert [value for value, _pid in result.get(timeout=20)] == list(range(32))


def test_paced_pool_is_a_stock_pool_with_a_reaping_handler():
    with _parallel_pool(3, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(30.0)) as pool:
        backend = _backend(pool)
        assert isinstance(backend, _PacedPool)
        assert isinstance(backend, multiprocessing.pool.Pool)
        assert backend._target == 3 and backend._processes == 0
        assert len(backend._pool) == 1


def _memory(available, total):
    return SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=available, total=total))


def test_ram_headroom_for_every_worker_lets_the_first_starts_skip_the_stagger():
    assert R._start_burst(8, _memory(64 * GIB, 128 * GIB), platform='linux') == 8
    assert R._start_burst(8, _memory(30 * GIB, 128 * GIB), platform='linux') == 1
    assert R._start_burst(4, _memory(18 * GIB, 32 * GIB), platform='linux') == 4
    assert R._start_burst(4, _memory(17 * GIB, 32 * GIB), platform='linux') == 1
    assert R._start_burst(4, _memory(18 * GIB, 32 * GIB), platform='win32') == 1
    assert R._start_burst(1, _memory(1 * GIB, 8 * GIB), platform='linux') == 1


def test_unreadable_ram_or_a_disabled_guard_keeps_the_stagger():
    broken = SimpleNamespace(virtual_memory=lambda: (_ for _ in ()).throw(OSError()))
    assert R._start_burst(4, broken, platform='linux') == 1
    with R._ram_guard_scope({'ram_guard': False}):
        assert R._start_burst(4, _memory(100 * GIB, 128 * GIB), platform='linux') == 1


def test_an_injected_gate_is_never_bypassed_and_the_default_one_can_burst(monkeypatch):
    seen = []
    real = R._StaggeredContext.Pool

    def record(self, *args, **kwargs):
        seen.append(kwargs.get('burst'))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(R._StaggeredContext, 'Pool', record)
    monkeypatch.setattr(R, '_start_burst', lambda workers, *a, **k: workers)
    with _parallel_pool(2, context=multiprocessing.get_context('spawn'),
                        gate=_WorkerStartGate(30.0)):
        pass
    with _parallel_pool(2, context=multiprocessing.get_context('spawn')) as pool:
        assert [value for value, _pid in pool.map(_hold_identity, [0, 1])] == [0, 1]
        deadline = time.monotonic() + 10
        while _backend(pool)._live_workers() < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _backend(pool)._live_workers() == 2
    assert seen == [1, 2]


def _hold_identity(value):
    return _identity(value)


def test_burst_starts_record_the_gate_so_a_replacement_is_still_paced():
    gate = _WorkerStartGate(1.0)
    times = []
    gate.start_now(lambda: times.append(time.monotonic()))
    gate.start(lambda: times.append(time.monotonic()))
    assert times[1] - times[0] >= 0.95


def test_wait_turn_gives_up_when_asked_and_waits_otherwise():
    gate = _WorkerStartGate(0.3)
    assert gate.wait_turn(lambda: False)
    gate.start(lambda: None)
    assert not gate.wait_turn(lambda: True)
    started = time.monotonic()
    assert gate.wait_turn(lambda: False)
    assert time.monotonic() - started >= 0.25
