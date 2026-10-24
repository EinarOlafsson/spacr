"""Edge cases of the ambient compiler helpers during shutdown.

A helper that has already vanished, a helper that cannot be spawned and a
colored-scatter compile cancelled by shutdown must each end quietly and leave
the failure flags honest.
"""
import subprocess
import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


class _Helper:
    def __init__(self, terminate_vanishes=False, kill_vanishes=False):
        self.calls = []
        self.terminate_vanishes = terminate_vanishes
        self.kill_vanishes = kill_vanishes
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_vanishes:
            raise ProcessLookupError()

    def kill(self):
        self.calls.append("kill")
        self.alive = False
        if self.kill_vanishes:
            raise ProcessLookupError()

    def wait(self, timeout):
        self.calls.append("wait")
        if self.alive and "kill" not in self.calls:
            raise subprocess.TimeoutExpired("helper", timeout)
        return 0


def test_helper_that_vanished_before_terminate_is_still_reaped():
    helper = _Helper(terminate_vanishes=True)
    ambient._stop_prime_helper(helper, time.monotonic() + .2)
    assert helper.calls == ["terminate", "wait", "kill", "wait"]
    assert helper.poll() == 0


def test_helper_that_vanished_before_kill_is_still_reaped():
    helper = _Helper(kill_vanishes=True)
    ambient._stop_prime_helper(helper, time.monotonic() + .2)
    assert helper.calls == ["terminate", "wait", "kill", "wait"]


def test_helper_that_cannot_be_spawned_leaves_no_owned_process(monkeypatch, tmp_path):
    marker = tmp_path / "not-primed"
    monkeypatch.setattr(ambient, "_COMPILER_STOP", threading.Event())
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", None)
    monkeypatch.setattr(ambient, "_PRIME_LOCK", threading.Lock())
    monkeypatch.setattr(ambient, "_priming_allowed", lambda: True)
    monkeypatch.setattr(ambient, "_numba_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(ambient, "_primed_marker", lambda folder: marker)

    def refuse(*args, **kwargs):
        raise OSError("no process slots")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    stopped = []
    monkeypatch.setattr(ambient, "_stop_prime_helper",
                        lambda process, deadline: stopped.append(process))
    assert ambient._prime_in_helper_process() is False
    assert stopped == [] and ambient._PRIME_PROCESS is None
    assert not marker.exists()


@pytest.mark.parametrize("cancelled", [True, False])
def test_colored_scatter_compile_failure_is_a_failure_only_when_not_cancelled(
        monkeypatch, cancelled):
    stop = threading.Event()
    ready = threading.Event()
    ready.set()
    monkeypatch.setattr(ambient, "_COMPILER_STOP", stop)
    monkeypatch.setattr(ambient, "_AMBIENT_STARTUP_READY", ready)
    monkeypatch.setattr(ambient, "_on_a_worker_thread", lambda: False)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER", None)
    monkeypatch.setattr(ambient, "_SUBPIXEL_SCATTER", None)
    monkeypatch.setattr(ambient, "_COLOR_BLEND", None)
    monkeypatch.setattr(ambient, "_COLORED_SCATTER", None)
    monkeypatch.setattr(ambient, "_COLORED_SCATTER_FAILED", False)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_FAILED", False)

    class Kernel:
        nopython_signatures = (1,)

        def __call__(self, *args):
            return None

    def compiled(function):
        if function is ambient._scatter_colored_grains:
            if cancelled:
                stop.set()
            raise RuntimeError("colored scatter would not compile")
        return Kernel()

    monkeypatch.setattr(ambient, "_compiled_kernel", compiled)
    ambient._warm_packed_scatter()
    assert ambient._COLORED_SCATTER is None
    assert ambient._COLORED_SCATTER_FAILED is (not cancelled)
    assert ambient._PACKED_SCATTER_FAILED is False
    assert ambient._PACKED_SCATTER is not None
