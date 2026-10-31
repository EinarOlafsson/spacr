"""A stream router a finishing worker uninstalls stays alive for a print still using it.

CPython's ``print`` reads ``sys.stdout`` once as a BORROWED reference and then
writes every argument and separator to it. When a worker finished on another
thread mid-print, ``_unregister_worker_streams`` put the original stream back
and dropped the last reference to the router, so the print carried on writing
through freed memory: ``'_ThreadStreamRouter' object has no attribute
'_lock'`` in the Qt suite, or a segmentation fault.
"""
from __future__ import annotations

import gc
import io
import sys
import weakref

import pytest

from spacr.qt import bridge as B


@pytest.fixture
def plain_streams(monkeypatch):
    """Plain process streams and no routers, set inside the test body.

    The test runner's own capture puts its streams back between fixture
    setup and the call, so the swap is a function the test calls first.
    """
    def swap():
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setattr(B, "_STDOUT_ROUTER", None)
        monkeypatch.setattr(B, "_STDERR_ROUTER", None)
        monkeypatch.setattr(B, "_KEPT_ROUTERS",
                            {"stdout": [], "stderr": []}, raising=False)
        return out, err
    return swap


def test_the_last_worker_leaving_does_not_free_the_router_it_uninstalls(
        plain_streams):
    out, err = plain_streams()
    target = B._StreamRedirector(lambda _text: None)
    routers = list(B._register_worker_streams(target))
    assert sys.stdout is routers[0] and sys.stderr is routers[1]
    alive = [weakref.ref(router) for router in routers]

    B._unregister_worker_streams(target, *routers)
    del routers
    gc.collect()

    assert sys.stdout is out and sys.stderr is err, "originals restored"
    assert all(ref() is not None for ref in alive), (
        "an uninstalled router was freed while a print may still hold it")


def test_a_print_that_outlives_the_worker_finishes_on_the_original_stream(
        plain_streams):
    out, _err = plain_streams()
    target = B._StreamRedirector(lambda _text: None)
    holder = list(B._register_worker_streams(target))

    class _WorkerFinishesHere:
        def __str__(self):
            B._unregister_worker_streams(target, *holder)
            holder.clear()
            gc.collect()
            return "first"

    print(_WorkerFinishesHere(), "second", "third")

    assert sys.stdout is out
    assert out.getvalue() == "first second third\n"


def test_the_next_worker_reinstalls_the_kept_router(plain_streams):
    out, _err = plain_streams()
    first = B._StreamRedirector(lambda _text: None)
    routers = B._register_worker_streams(first)
    B._unregister_worker_streams(first, *routers)

    second = B._StreamRedirector(lambda _text: None)
    again = B._register_worker_streams(second)
    try:
        assert again[0] is routers[0] and again[1] is routers[1]
        assert again[0].original is out
        assert len(B._KEPT_ROUTERS["stdout"]) == 1
    finally:
        B._unregister_worker_streams(second, *again)
