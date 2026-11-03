"""Helper processes end with whatever started them.

The backdrop's kernel-cache helper is waited on by a non-daemon compiler
thread. Python joins such threads before it runs ``atexit`` hooks, so a
retirement registered only there let the interpreter wait for the whole
helper compile (up to four minutes) on exit.

The settings screens' GPU-count probe is started on a launcher thread. An
application quit while that launch was still running dropped the probe's
state, and the child started a moment later had no owner.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from spacr import _mask_workers as mw
from spacr.qt.widgets import ambient

ROOT = Path(__file__).resolve().parents[2]

_EXIT_SCRIPT = r"""
import subprocess, sys, threading
from pathlib import Path
from spacr.qt.widgets import ambient
pid_file = Path(sys.argv[1])
cache = Path(sys.argv[2])
ambient._numba_cache_directory = lambda: cache
ambient._priming_allowed = lambda: True
real = subprocess.Popen
spawned = threading.Event()
def stalled(command, **kwargs):
    child = real([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
    pid_file.write_text(str(child.pid))
    spawned.set()
    return child
subprocess.Popen = stalled
assert ambient._start_compiler(ambient._prime_in_helper_process, 'spacr-grain-compile')
assert spawned.wait(30)
"""


def _alive(pid):
    """Whether ``pid`` is still a running process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process check")
def test_exit_stops_the_prime_helper_before_python_joins_its_thread(tmp_path):
    """The interpreter exits within the compiler budget and reaps the helper."""
    pid_file = tmp_path / "helper.pid"
    cache = tmp_path / "numba"
    cache.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen",
               HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "config"),
               PYTHONPATH=os.pathsep.join(
                   p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p))
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-c", _EXIT_SCRIPT, str(pid_file), str(cache)],
                            env=env, capture_output=True, text=True, timeout=55)
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr[-3000:]
    pid = int(pid_file.read_text())
    assert not _alive(pid)
    assert elapsed < 30, elapsed


def test_the_retirement_runs_before_non_daemon_threads_are_joined(monkeypatch):
    """The hook is in Python's pre-join list and retires the owned compilers."""
    assert any(getattr(entry, "func", entry) is ambient._retire_compilers_before_thread_join
               for entry in threading._threading_atexits)
    calls = []
    monkeypatch.setattr(ambient, "_retire_compiler_threads", lambda: calls.append(1) or 0)
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", None)
    ambient._retire_compilers_before_thread_join()
    assert calls == []
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", object())
    ambient._retire_compilers_before_thread_join()
    assert calls == [1]

    def broken():
        """Fail the way an interpreter half torn down can."""
        raise RuntimeError("torn down")

    monkeypatch.setattr(ambient, "_retire_compiler_threads", broken)
    ambient._retire_compilers_before_thread_join()


class _SlowStartProcess:
    """A discovery child whose start blocks until the test releases it."""

    def __init__(self, fail=False):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.fail = fail
        self.started = self.stopped = self.closed = False
        self.joins = []

    def start(self):
        self.entered.set()
        assert self.release.wait(10)
        if self.fail:
            raise OSError("no interpreter")
        self.started = True

    def is_alive(self):
        return self.started and not self.stopped

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joins.append(timeout)

    def close(self):
        assert not self.is_alive()
        self.closed = True


class _Pipe:
    """One end of the discovery pipe."""

    closed = False

    def poll(self):
        return False

    def close(self):
        self.closed = True


@pytest.fixture
def slow_probe(monkeypatch):
    """Route the GPU-count probe through a child whose launch is still running."""
    from types import SimpleNamespace

    mw._stop_mask_gpu_probe()
    made = {}

    def install(fail=False):
        """Install one slow-starting fake child and return it with its pipe."""
        process = _SlowStartProcess(fail=fail)
        reader, writer = _Pipe(), _Pipe()
        context = SimpleNamespace(Pipe=lambda **_: (reader, writer),
                                  Process=lambda **_: process)
        monkeypatch.setattr(mw.multiprocessing, "get_context", lambda mode: context)
        made.update(process=process, reader=reader, writer=writer)
        return process, reader, writer

    monkeypatch.setattr(mw, "_CONTROL_GPU_COUNT", [])
    monkeypatch.setattr(mw, "_CONTROL_GPU_PROBE", None)
    yield install
    if "process" in made:
        made["process"].release.set()
    mw._stop_mask_gpu_probe()


def test_quitting_during_the_launch_leaves_the_child_owned(slow_probe):
    """A child that starts after the quit is still terminated and reaped."""
    process, reader, writer = slow_probe()
    assert mw._mask_gpu_count_for_controls() is None
    launcher = mw._CONTROL_GPU_PROBE["launcher"]
    assert process.entered.wait(5)
    began = time.monotonic()
    mw._stop_mask_gpu_probe()
    assert time.monotonic() - began < 0.5
    assert mw._CONTROL_GPU_PROBE is None and mw._CONTROL_GPU_COUNT == [0]
    assert reader.closed
    process.release.set()
    launcher.join(5)
    assert not launcher.is_alive()
    assert process.started and process.stopped and process.closed
    assert writer.closed


def test_a_launch_that_fails_after_the_quit_owns_nothing(slow_probe):
    """A cancelled launch that never started a child closes only its pipe."""
    process, reader, writer = slow_probe(fail=True)
    assert mw._mask_gpu_count_for_controls() is None
    launcher = mw._CONTROL_GPU_PROBE["launcher"]
    assert process.entered.wait(5)
    mw._stop_mask_gpu_probe()
    process.release.set()
    launcher.join(5)
    assert not launcher.is_alive()
    assert not process.started and not process.closed and process.joins == []
    assert reader.closed and writer.closed


def test_a_child_that_ignores_terminate_is_killed_after_the_quit(slow_probe, monkeypatch):
    """The launcher escalates to kill when terminate does not end the child."""
    process, _reader, _writer = slow_probe()
    signals = []
    monkeypatch.setattr(process, "terminate", lambda: signals.append("terminate"))

    def kill():
        """Record the kill and end the child."""
        signals.append("kill")
        process.stopped = True

    monkeypatch.setattr(process, "kill", kill)
    assert mw._mask_gpu_count_for_controls() is None
    launcher = mw._CONTROL_GPU_PROBE["launcher"]
    assert process.entered.wait(5)
    mw._stop_mask_gpu_probe()
    process.release.set()
    launcher.join(5)
    assert signals == ["terminate", "kill"] and process.closed
    assert process.joins == [1.0, None]


def test_a_child_that_already_exited_after_the_quit_is_only_closed(slow_probe):
    """A cancelled child that ended by itself is closed without signals."""
    process, _reader, _writer = slow_probe()
    assert mw._mask_gpu_count_for_controls() is None
    launcher = mw._CONTROL_GPU_PROBE["launcher"]
    assert process.entered.wait(5)
    mw._stop_mask_gpu_probe()
    process.stopped = True
    process.release.set()
    launcher.join(5)
    assert process.started and process.closed and process.joins == []
