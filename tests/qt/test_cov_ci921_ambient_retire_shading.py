"""Retiring the backdrop shading threads at quit, and the GL renderer name.

Covers the join loop of ``_retire_shading_threads`` (a producer whose thread
is the caller's own, is dead or is missing is not joined), the warning and
count for a thread that outlives the deadline, the never-raising ``atexit``
and ``aboutToQuit`` wrappers, ``_retire_shading_on_quit`` against an
application that cannot connect, and ``_FlowPointGraphics`` falling back to
"unknown renderer" when the driver info cannot be read.
"""
import logging
import sys
import threading
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


class _Producer:
    """A stand-in shading producer with the attributes the retire path uses."""

    def __init__(self, thread, alive=False):
        self._thread = thread
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.box = None
        self._alive = alive

    def is_alive(self):
        return self._alive


@pytest.fixture
def registry(monkeypatch):
    live = ambient.weakref.WeakSet()
    monkeypatch.setattr(ambient, "_LIVE_PRODUCERS", live)
    monkeypatch.setattr(ambient, "_SHADING_RETIRED", [False])
    return live


def test_threads_that_are_current_dead_or_missing_are_not_joined(
        registry, monkeypatch):
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    joined = []
    monkeypatch.setattr(threading.Thread, "join",
                        lambda self, *a, **k: joined.append(self))

    producers = [_Producer(threading.current_thread()), _Producer(dead),
                 _Producer(None)]
    for producer in producers:
        registry.add(producer)
    alive = ambient._retire_shading_threads(timeout_s=0.01)

    assert joined == []
    assert alive == 0
    assert all(p._stop.is_set() and p._wake.is_set() for p in producers)


def test_a_thread_that_outlives_the_deadline_is_counted_and_warned(
        registry, caplog):
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    producer = _Producer(thread, alive=True)
    registry.add(producer)
    try:
        with caplog.at_level(logging.WARNING, logger=ambient.LOG.name):
            alive = ambient._retire_shading_threads(timeout_s=0.01)
    finally:
        release.set()
        thread.join(2)
    assert alive == 1
    assert producer._stop.is_set()
    assert any("did not stop before exit" in r.getMessage()
               for r in caplog.records)
    assert ambient._SHADING_RETIRED[0] is False


def test_the_final_call_marks_shading_retired(registry):
    assert ambient._retire_shading_threads(timeout_s=0.01, final=True) == 0
    assert ambient._SHADING_RETIRED[0] is True


def test_the_atexit_hook_is_final_and_swallows_a_failure(monkeypatch):
    seen = []

    def boom(*args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("driver gone")

    monkeypatch.setattr(ambient, "_retire_shading_threads", boom)
    ambient._retire_shading_at_exit()
    assert seen == [{"final": True}]


def test_the_about_to_quit_slot_swallows_a_failure(monkeypatch):
    seen = []

    def boom(*args, **kwargs):
        seen.append((args, kwargs))
        raise RuntimeError("driver gone")

    monkeypatch.setattr(ambient, "_retire_shading_threads", boom)
    ambient._retire_shading_on_about_to_quit()
    assert seen == [((), {})]


@pytest.mark.parametrize("error", [RuntimeError, TypeError, AttributeError])
def test_an_application_that_cannot_connect_is_left_unhooked(
        monkeypatch, error):
    monkeypatch.setattr(ambient, "_QUIT_HOOKED", [None])

    class Signal:
        def connect(self, slot):
            raise error("no signal")

    app = types.SimpleNamespace(aboutToQuit=Signal())
    ambient._retire_shading_on_quit(app)
    assert ambient._QUIT_HOOKED[0] is None


def test_no_application_is_ignored_and_a_hooked_one_connects_once(
        monkeypatch):
    monkeypatch.setattr(ambient, "_QUIT_HOOKED", [None])
    connected = []

    class Signal:
        def connect(self, slot):
            connected.append(slot)

    app = types.SimpleNamespace(aboutToQuit=Signal())
    ambient._retire_shading_on_quit(None)
    assert ambient._QUIT_HOOKED[0] is None
    ambient._retire_shading_on_quit(app)
    ambient._retire_shading_on_quit(app)
    assert connected == [ambient._retire_shading_on_about_to_quit]
    assert ambient._QUIT_HOOKED[0] is app


class _Resource:
    def __init__(self, reserve=0):
        self.size = reserve

    def release(self):
        pass


class _BrokenInfoContext:
    """A context whose driver info cannot be read."""

    def __init__(self):
        self.released = False

    @property
    def info(self):
        raise RuntimeError("no driver info")

    def program(self, **kwargs):
        return _Resource()

    def buffer(self, reserve=0):
        return _Resource(reserve)

    def vertex_array(self, *args):
        return _Resource()

    def enable(self, flags):
        pass

    def release(self):
        self.released = True


def test_an_unreadable_driver_info_names_the_renderer_unknown(monkeypatch):
    module = types.ModuleType("moderngl")
    module.PROGRAM_POINT_SIZE = 1
    module.BLEND = 2
    context = _BrokenInfoContext()
    module.create_context = lambda **options: context
    monkeypatch.setitem(sys.modules, "moderngl", module)
    graphics = ambient._FlowPointGraphics()
    try:
        assert graphics.renderer == "unknown renderer"
        assert graphics._context is context
    finally:
        graphics._close()
