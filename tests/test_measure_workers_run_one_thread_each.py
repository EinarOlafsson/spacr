"""A Measure worker computes on one thread, not on every core (1.5.1.5).

On a real Mac, Measure's six default workers each sized numpy/BLAS, numba,
torch and OpenCV to all eight cores; the load reached 75 and the machine
froze. The pool now starts its workers with the thread variables set to one
and each worker's initializer holds the already-imported libraries to one.
"""
from __future__ import annotations

import multiprocessing
import os

from spacr import measure


def _what_the_worker_sees(_):
    """Report the thread settings inside a pool worker."""
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info

    env = {name: os.environ.get(name)
           for name in measure._WORKER_THREAD_VARIABLES}
    pools = sorted({int(p["num_threads"]) for p in threadpool_info()})
    return env, pools


def test_a_spawned_measure_worker_sees_one_thread_everywhere():
    """Every variable is 1 and every native pool runs one thread."""
    before = {name: os.environ.get(name)
              for name in measure._WORKER_THREAD_VARIABLES}
    ctx = multiprocessing.get_context("spawn")
    with measure._single_threaded_children():
        with ctx.Pool(2, initializer=measure._initialize_measure_worker,
                      initargs=()) as pool:
            seen = pool.map(_what_the_worker_sees, range(2))
    for env, pools in seen:
        assert env == {name: "1" for name in measure._WORKER_THREAD_VARIABLES}
        assert all(n == 1 for n in pools), pools
    after = {name: os.environ.get(name)
             for name in measure._WORKER_THREAD_VARIABLES}
    assert after == before, "the parent's own environment was not restored"


def test_the_initializer_still_hands_over_the_writer_endpoint(monkeypatch):
    """The queued-SQLite path keeps its producer handle."""
    monkeypatch.setattr(measure, "_MEASURE_WRITE_ENDPOINT", None)
    for name in measure._WORKER_THREAD_VARIABLES:
        monkeypatch.setenv(name, os.environ.get(name, "8"))
    marker = object()
    measure._initialize_measure_worker(marker)
    assert measure._MEASURE_WRITE_ENDPOINT is marker
    assert all(os.environ[n] == "1" for n in measure._WORKER_THREAD_VARIABLES)


def test_optional_thread_library_refusals_preserve_the_writer_endpoint_and_worker_budget(monkeypatch):
    import importlib

    threadpoolctl = importlib.import_module('threadpoolctl')
    torch = importlib.import_module('torch')
    cv2 = importlib.import_module('cv2')
    calls = []

    def refuse(name):
        def fail(*args, **kwargs):
            calls.append(name)
            raise RuntimeError('optional thread controller unavailable')
        return fail

    for name in measure._WORKER_THREAD_VARIABLES:
        monkeypatch.setenv(name, os.environ.get(name, '8'))
    monkeypatch.setattr(threadpoolctl, 'threadpool_limits', refuse('blas'))
    monkeypatch.setattr(torch, 'set_num_threads', refuse('torch'))
    monkeypatch.setattr(cv2, 'setNumThreads', refuse('cv2'))
    monkeypatch.setattr(measure, '_MEASURE_WRITE_ENDPOINT', None)
    endpoint = object()
    measure._initialize_measure_worker(endpoint)
    assert measure._MEASURE_WRITE_ENDPOINT is endpoint
    assert calls == ['blas', 'torch', 'cv2']
    assert all(os.environ[name] == '1' for name in measure._WORKER_THREAD_VARIABLES)
