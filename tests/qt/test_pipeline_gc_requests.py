"""Explicit pipeline cleanup must not sweep Qt wrappers from a worker."""

import gc
import subprocess
import sys
import threading

import pytest

from spacr import _gc
from spacr.qt import gc_policy

pytestmark = pytest.mark.qt


class _Cycle:
    def __init__(self, destroyed):
        self.self_ref = self
        self.destroyed = destroyed

    def __del__(self):
        self.destroyed.append(threading.get_ident())


@pytest.fixture(autouse=True)
def _clear_request(qapp):
    _gc._requested.clear()
    yield
    _gc._requested.clear()


def test_worker_cleanup_is_serviced_by_gui_timer_below_thresholds(qapp, qtbot, monkeypatch):
    """A worker's explicit full sweep survives even when automatic GC is not due."""
    assert gc_policy.is_installed()
    assert not gc.isenabled()
    monkeypatch.setattr(gc_policy, "_saved_thresholds", (10**9,) * 3)
    destroyed = []
    for _ in range(20):
        _Cycle(destroyed)
    results = []

    def worker():
        for _ in range(3):
            results.append(_gc.collect())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "cleanup must not wait for the GUI thread"
    assert destroyed == [], "a worker ran destructors before the GUI timer"
    assert results == [0, 0, 0]
    assert _gc._requested.is_set()
    qtbot.waitUntil(lambda: len(destroyed) == 20, timeout=3000)
    assert set(destroyed) == {threading.get_ident()}
    assert not _gc._requested.is_set()
    assert gc_policy.collect_once() == -1


def test_gui_thread_cleanup_remains_synchronous(qapp):
    destroyed = []
    _Cycle(destroyed)
    assert _gc.collect() >= 1
    assert destroyed == [threading.get_ident()]
    assert not _gc._requested.is_set()


def test_headless_worker_collects_without_importing_qt():
    script = """
import gc
import sys
import threading
from spacr import _gc
gc.disable()
destroyed = []
class Cycle:
    def __init__(self):
        self.self_ref = self
    def __del__(self):
        destroyed.append(threading.get_ident())
for _ in range(20):
    Cycle()
results = []
thread = threading.Thread(target=lambda: results.append(_gc.collect()))
thread.start()
thread.join(timeout=5)
assert not thread.is_alive()
assert results[0] >= 20
assert destroyed == [thread.ident] * 20
assert not any(name.startswith('PySide6') for name in sys.modules)
assert not _gc._requested.is_set()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
