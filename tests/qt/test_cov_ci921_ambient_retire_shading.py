"""Retiring the backdrop shading threads at quit, and the GL renderer name.

Covers the join loop of ``_retire_shading_threads`` (a producer whose thread
is the caller's own, is dead or is missing is not joined), the warning and
count for a thread that outlives the deadline, the never-raising ``atexit``
and ``aboutToQuit`` wrappers, ``_retire_shading_on_quit`` against an
application that cannot connect, and ``_FlowPointGraphics`` falling back to
"unknown renderer" when the driver info cannot be read.
"""
import ast
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


def test_normal_window_close_cancels_owned_compiler_helper(tmp_path):
    root = Path(__file__).resolve().parents[2]
    menu = ast.parse((root / "tests/qt/test_the_macos_application_menu.py").read_text())
    bootstrap = next(ast.literal_eval(node.value.args[0]) for node in menu.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "_OWNED_SETTINGS_PROBE"
                         for target in node.targets))
    script = textwrap.dedent(bootstrap) + '''
import subprocess, threading, time
from pathlib import Path
from PySide6.QtCore import QTimer
from spacr.qt import app as application
from spacr.qt.widgets import ambient

root = Path(os.environ["SPACR_COMPILER_EXIT_ROOT"])
assert Path(application.__file__).resolve().parents[2] == Path(os.environ["SPACR_TESTED_ROOT"])
helpers = []
real_popen = subprocess.Popen
helper_code = ''' + repr('''
import os, time
from pathlib import Path
root = Path(os.environ["SPACR_COMPILER_EXIT_ROOT"])
(root / "helper-entered").touch()
deadline = time.monotonic() + 60
while not (root / "release").exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("owned helper barrier was not released")
    time.sleep(.01)
''') + '''

def controlled_helper(arguments, *args, **kwargs):
    assert arguments[0] == sys.executable and arguments[1] == "-c"
    assert "_prime_kernel_cache" in arguments[2], arguments
    process = real_popen([sys.executable, "-c", helper_code], *args, **kwargs)
    helpers.append(process)
    return process

def forbidden_compile(function):
    (root / "unexpected-local-compile").touch()
    raise AssertionError("quit must not fall through into local compilation")

subprocess.Popen = controlled_helper
ambient._compiled_kernel = forbidden_compile
real_prime = ambient._prime_in_helper_process
entered = []

def watching_prime():
    entered.append(threading.current_thread().name)
    return real_prime()

ambient._prime_in_helper_process = watching_prime

app = QApplication([])
app.setQuitOnLastWindowClosed(True)
window = application.MainWindow()
window.show()
assert ambient._priming_allowed()
assert not ambient._primed_marker(ambient._numba_cache_directory()).exists()
ambient._AMBIENT_STARTUP_READY.set()
ambient._ready_packed_scatter()
ambient._SATIN_COMPILER.ready()
deadline = time.monotonic() + 5
while len(entered) != 2 or not (root / "helper-entered").exists():
    assert time.monotonic() < deadline, entered
    time.sleep(.01)
ambient._retire_shading_on_quit(app)
QTimer.singleShot(0, window.close)
QTimer.singleShot(10000, lambda: app.exit(91))
code = app.exec()
application._tear_down_before_finalization()
assert code == 0 and window._closing
threads = [{"name": worker.name, "daemon": worker.daemon}
           for worker in threading.enumerate()
           if worker.name in ("spacr-grain-compile", "spacr-satin-compile")]
(root / "returned.json").write_text(json.dumps({
    "code": code, "threads": threads, "helpers": [process.poll() for process in helpers],
    "prime_callers": entered}))
print("NORMAL_RETURN", flush=True)
'''
    platform = os.environ.get("SPACR_COMPILER_TEST_PLATFORM", "offscreen")
    assert platform in {"offscreen", "cocoa"}
    env = dict(os.environ, QT_QPA_PLATFORM=platform, CUDA_VISIBLE_DEVICES="",
               SPACR_NO_BACKDROP="1", SPACR_NO_NETWORK="1", SPACR_NO_SETUP="1",
               SPACR_HOME=str(tmp_path / "home"), XDG_CACHE_HOME=str(tmp_path / "cache"),
               SPACR_MENU_TEST_SETTINGS=str(tmp_path), SPACR_COMPILER_EXIT_ROOT=str(tmp_path),
               SPACR_TESTED_ROOT=str(root), PYTHONPATH=str(root),
               NUMBA_CACHE_DIR=str(tmp_path / "numba"))
    env.pop("SPACR_ISOLATED_PROFILE", None)
    env.pop("SPACR_DISTRIBUTION_SMOKE", None)
    with (tmp_path / "child.log").open("w") as stream:
        process = subprocess.Popen([sys.executable, "-c", script], cwd=root, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        waited_for_compiler = False
        try:
            deadline = time.monotonic() + 60
            returned = tmp_path / "returned.json"
            while not returned.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert returned.exists(), (tmp_path / "child.log").read_text()
            result = json.loads(returned.read_text())
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                waited_for_compiler = True
        finally:
            (tmp_path / "release").touch()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        assert process.returncode == 0, (tmp_path / "child.log").read_text()
        assert not waited_for_compiler, result
        assert result["threads"] == [], result
        assert len(result["helpers"]) == 1 and result["helpers"][0] is not None, result
        assert not (tmp_path / "unexpected-local-compile").exists()


def test_active_local_compilation_is_retained_and_reported_until_it_finishes(monkeypatch, caplog):
    stop = threading.Event()
    monkeypatch.setattr(ambient, "_COMPILER_STOP", stop, raising=False)
    monkeypatch.setattr(ambient, "_COMPILER_THREADS", set(), raising=False)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_STARTED", False)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_FAILED", False)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER", None)
    monkeypatch.setattr(ambient, "_AMBIENT_STARTUP_READY", threading.Event())
    ambient._AMBIENT_STARTUP_READY.set()
    monkeypatch.setattr(ambient, "_priming_allowed", lambda: False)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def active_local_compile(function):
        calls.append(function)
        entered.set()
        assert release.wait(5)

        def kernel(*args):
            return None

        kernel.nopython_signatures = ("owned-test",)
        return kernel

    monkeypatch.setattr(ambient, "_compiled_kernel", active_local_compile)
    ambient._ready_packed_scatter()
    assert entered.wait(5)
    worker = next(thread for thread in threading.enumerate()
                  if thread.name == "spacr-grain-compile")
    try:
        assert worker.daemon is False
        with caplog.at_level(logging.WARNING, logger=ambient.LOG.name):
            assert ambient._retire_compiler_threads(timeout_s=.02) == 1
        assert worker.is_alive()
        assert worker in ambient._COMPILER_THREADS
        assert ambient._PACKED_SCATTER is None
        assert any("compiler" in record.getMessage() and "did not stop" in record.getMessage()
                   for record in caplog.records)
        ambient._PACKED_SCATTER_STARTED = False
        satin = ambient._SatinCompiler()
        monkeypatch.setattr(satin, "_warm", lambda: calls.append("unexpected satin start"))
        ambient._ready_packed_scatter()
        satin.ready()
        assert not ambient._PACKED_SCATTER_STARTED and not satin.started
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(calls) == 1
    assert ambient._PACKED_SCATTER is None
    assert ambient._retire_compiler_threads(timeout_s=.02) == 0


@pytest.mark.parametrize("phase", ["already-primed", "helper-cancel", "ordinary-fallback"])
def test_fractal_priming_does_not_dispatch_after_compiler_shutdown(monkeypatch, phase):
    from concurrent.futures import CancelledError

    stopped, primed = threading.Event(), threading.Event()
    if phase == "already-primed":
        stopped.set()
        primed.set()
    monkeypatch.setattr(ambient, "_COMPILER_STOP", stopped, raising=False)
    monkeypatch.setattr(ambient, "_PRIMED_ONCE", primed)
    monkeypatch.setattr(ambient, "_on_a_worker_thread", lambda: True)
    called = []

    def helper():
        called.append(True)
        if phase == "helper-cancel":
            stopped.set()
        return False

    monkeypatch.setattr(ambient, "_prime_in_helper_process", helper)
    if phase == "ordinary-fallback":
        ambient._prime_before_first_kernel()
        assert called == [True]
    else:
        with pytest.raises(CancelledError):
            ambient._prime_before_first_kernel()
        assert called == ([] if phase == "already-primed" else [True])


def test_owned_helper_resisting_terminate_is_killed_and_reaped():
    calls = []

    class Helper:
        alive = True

        def poll(self):
            return None if self.alive else -9

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.alive = False

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if self.alive:
                raise subprocess.TimeoutExpired("owned helper", timeout)
            return -9

    helper = Helper()
    ambient._stop_prime_helper(helper, time.monotonic() + .1)
    assert [item if isinstance(item, str) else item[0] for item in calls] == [
        "terminate", "wait", "kill", "wait"]
    assert all(0 <= item[1] <= .1 for item in calls if isinstance(item, tuple))
    assert helper.poll() == -9


def test_helper_cleanup_error_retains_owned_process_and_reports_incomplete_drain(monkeypatch, caplog):
    class Helper:
        def poll(self):
            return None

        def terminate(self):
            raise PermissionError("owned test cleanup failure")

    helper = Helper()
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", helper)
    monkeypatch.setattr(ambient, "_COMPILER_STOP", threading.Event())
    monkeypatch.setattr(ambient, "_COMPILER_THREADS", set())
    with caplog.at_level(logging.WARNING, logger=ambient.LOG.name):
        assert ambient._retire_compiler_threads(timeout_s=.02) == 1
    assert ambient._PRIME_PROCESS is helper
    assert ambient._COMPILER_STOP.is_set()
    assert any("compiler helper did not stop" in record.getMessage() for record in caplog.records)


def test_helper_spawn_and_quit_are_serialized_and_no_late_helper_starts(monkeypatch, tmp_path):
    entered, release, terminated, retiring = (threading.Event() for _ in range(4))
    calls, failures = [], []

    class Helper:
        def poll(self):
            return -15 if terminated.is_set() else None

        def terminate(self):
            terminated.set()

        def wait(self, timeout):
            assert terminated.wait(min(timeout, 5))
            return -15

    helper = Helper()

    def spawn(*args, **kwargs):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return helper

    monkeypatch.setattr(ambient, "_COMPILER_STOP", threading.Event())
    monkeypatch.setattr(ambient, "_COMPILER_THREADS", set())
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", None)
    monkeypatch.setattr(ambient, "_numba_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(ambient, "_primed_marker", lambda path: path / "not-primed")
    monkeypatch.setattr(ambient, "_priming_allowed", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", spawn)

    def retiring_worker():
        retiring.set()
        try:
            assert ambient._retire_compiler_threads(timeout_s=2) == 0
        except BaseException as error:
            failures.append(error)

    prime = threading.Thread(target=ambient._prime_in_helper_process)
    retire = threading.Thread(target=retiring_worker)
    try:
        prime.start()
        assert entered.wait(5)
        retire.start()
        assert retiring.wait(5)
        release.set()
        prime.join(5)
        retire.join(5)
        assert not prime.is_alive() and not retire.is_alive()
        assert failures == []
        assert terminated.is_set() and ambient._PRIME_PROCESS is None
        assert ambient._prime_in_helper_process() is False
        assert calls == [True]
    finally:
        release.set()
        terminated.set()
        if prime.ident is not None:
            prime.join(5)
        if retire.ident is not None:
            retire.join(5)


@pytest.mark.parametrize("prior_failure", [False, True])
def test_new_startup_resets_canceled_compilers_but_preserves_real_failures(monkeypatch, prior_failure):
    stop = threading.Event()
    stop.set()
    satin = ambient._SatinCompiler()
    satin.started, satin.failed = True, prior_failure
    monkeypatch.setattr(ambient, "_SATIN_COMPILER", satin)
    monkeypatch.setattr(ambient, "_COMPILER_STOP", stop)
    monkeypatch.setattr(ambient, "_COMPILER_THREADS", set())
    monkeypatch.setattr(ambient, "_PRIME_PROCESS", None)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER", None)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_STARTED", True)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_FAILED", prior_failure)
    primed = threading.Event()
    primed.set()
    monkeypatch.setattr(ambient, "_PRIMED_ONCE", primed)
    monkeypatch.setattr(ambient, "_AMBIENT_STARTUP_READY", threading.Event())
    ambient._begin_ambient_startup()
    assert not stop.is_set()
    assert not primed.is_set()
    assert ambient._PACKED_SCATTER_STARTED is prior_failure
    assert satin.started is prior_failure
    assert ambient._PACKED_SCATTER_FAILED is prior_failure
    assert satin.failed is prior_failure


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
