"""A run that finished its tests has to finish reporting them too.

``--timeout`` is a per-test guard: once the last test is done it is gone, so
anything that stops the interpreter exiting stops the run somewhere between
the final test and the summary line, with no test to blame. A distributed run
turns that into hours of nothing -- workers that have already written their
results, each holding a core, and a controller waiting on all of them.

Two things are pinned here. Whatever will hold the interpreter open is NAMED
while the run can still print, and a shutdown that never ends is turned into
a stack dump and a non-zero exit instead of a wait with no end.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading

from tests.conftest import (
    SHUTDOWN_WATCHDOG_S,
    _threads_that_outlive_the_session_report,
    arm_shutdown_watchdog,
    threads_that_outlive_the_session,
)


def _parked_thread(daemon):
    """A thread that sits still until it is released, so a test can look."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=daemon,
                              name="a-parked-thread")
    thread.start()
    return thread, release


def test_a_non_daemon_thread_is_reported():
    """This is the shape that costs a run its summary line."""
    thread, release = _parked_thread(daemon=False)
    try:
        assert thread in threads_that_outlive_the_session()
    finally:
        release.set()
        thread.join(timeout=5)


def test_a_daemon_thread_is_not_reported():
    """Daemon threads cannot delay finalisation, and this suite ends with
    several of them every run -- reporting them would bury the one that
    matters."""
    thread, release = _parked_thread(daemon=True)
    try:
        assert thread not in threads_that_outlive_the_session()
    finally:
        release.set()
        thread.join(timeout=5)


def test_a_finished_thread_is_not_reported():
    """A thread that already ended holds nothing open."""
    thread, release = _parked_thread(daemon=False)
    release.set()
    thread.join(timeout=5)

    assert thread not in threads_that_outlive_the_session()


def test_the_main_thread_is_never_reported():
    """It is the thread doing the joining, not one of the threads joined."""
    assert threading.main_thread() not in threads_that_outlive_the_session()


def test_the_report_names_the_thread_and_what_it_runs():
    """A name alone sends the reader hunting; the target is the lead."""
    thread, release = _parked_thread(daemon=False)
    try:
        report = _threads_that_outlive_the_session_report([thread])
    finally:
        release.set()
        thread.join(timeout=5)

    assert "1 non-daemon thread(s)" in report
    assert "a-parked-thread" in report
    assert "joins each of them before it finalises" in report


def test_the_watchdog_can_be_turned_off():
    """Zero means off, so a debugging session is never shot in the back."""
    assert arm_shutdown_watchdog(0) is False
    assert arm_shutdown_watchdog(-1) is False


def test_the_watchdog_budget_is_generous_by_default():
    """An honestly slow teardown must never be mistaken for a stall."""
    assert SHUTDOWN_WATCHDOG_S >= 60


_STALLED_SHUTDOWN = '''\
import sys, time
sys.path.insert(0, {repo!r})
from tests.conftest import arm_shutdown_watchdog

assert arm_shutdown_watchdog(2) is True
print("armed", flush=True)
time.sleep(120)
print("the watchdog never fired", flush=True)
'''


def test_a_shutdown_that_never_ends_is_dumped_and_killed(tmp_path):
    """The whole point, driven for real: an endless wait becomes a stack dump.

    A subprocess because the watchdog kills the process it is armed in --
    proving it in-process would take the test run with it.
    """
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    script = tmp_path / "stalled_shutdown.py"
    script.write_text(textwrap.dedent(_STALLED_SHUTDOWN.format(repo=repo)))

    done = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=300)

    assert "armed" in done.stdout
    assert "the watchdog never fired" not in done.stdout, done.stdout
    assert done.returncode != 0
    assert "Timeout (0:00:02)" in done.stderr, done.stderr
    assert "Thread" in done.stderr or "Current thread" in done.stderr


_LEAKY_CONFTEST = '''\
import sys
sys.path.insert(0, {repo!r})
from tests.conftest import pytest_sessionfinish  # noqa: F401
'''

_LEAKY_TEST = '''\
import threading


def test_a_test_that_forgets_to_stop_its_thread():
    """Leaves a non-daemon thread parked forever, which is the failure mode."""
    threading.Thread(target=threading.Event().wait,
                     name="a-leaked-thread").start()
    assert True
'''


def test_a_run_whose_interpreter_will_not_exit_still_reports_and_stops(
        tmp_path):
    """The live defect end to end: named while it can print, then bounded.

    A run that passes every test and then hangs is the worst shape there is --
    ``--timeout`` is over, no test is to blame, and under ``-n`` the workers
    sit on a core each. Here the leaked thread is named in the run's own
    output and the process dies with every stack printed, instead of waiting
    for somebody to notice.
    """
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    (tmp_path / "conftest.py").write_text(_LEAKY_CONFTEST.format(repo=repo))
    (tmp_path / "test_leaks_a_thread.py").write_text(_LEAKY_TEST)

    env = dict(__import__("os").environ)
    env["SPACR_PYTEST_SHUTDOWN_WATCHDOG_S"] = "3"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "test_leaks_a_thread.py",
         "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
         "-p", "no:cov", "-c", "/dev/null"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=300)

    assert "1 passed" in done.stdout, done.stdout + done.stderr
    assert "non-daemon thread(s) are still running" in done.stdout, done.stdout
    assert "a-leaked-thread" in done.stdout, done.stdout
    assert done.returncode != 0, done.stdout
    assert "Timeout (0:00:03)" in done.stderr, done.stderr


def test_a_watchdog_that_cannot_be_armed_says_so_rather_than_going_quiet():
    """A watchdog nobody armed is as useful as no watchdog.

    faulthandler writes through a file descriptor, and a run whose stderr has
    none -- a distributed worker's, a captured stream -- cannot arm it. That
    has to be visible, because the run it was meant to bound would otherwise
    stop with no explanation at all.
    """
    import tests.conftest as suite_conftest

    said = []

    class _Reporter:
        def write_line(self, message, **kwargs):
            said.append(message)

    class _Manager:
        def get_plugin(self, name):
            return _Reporter() if name == "terminalreporter" else None

    class _Session:
        config = type("_Config", (), {"pluginmanager": _Manager()})()

    original = suite_conftest._faulthandler.dump_traceback_later

    def _refuse(*args, **kwargs):
        raise ValueError("stderr has no file descriptor here")

    suite_conftest._faulthandler.dump_traceback_later = _refuse
    try:
        suite_conftest.pytest_sessionfinish(_Session(), 0)
    finally:
        suite_conftest._faulthandler.dump_traceback_later = original

    assert any("shutdown watchdog could not be armed" in line
               for line in said), said


def test_the_report_reaches_stderr_when_nothing_is_printing_the_run(capsys):
    """A run with no terminal reporter still gets told.

    ``-p no:terminal``, a plugin that replaced it, a distributed worker --
    the report matters most exactly where the usual output channel is gone,
    so it falls back to stderr rather than disappearing.
    """
    import tests.conftest as suite_conftest

    class _Manager:
        def get_plugin(self, name):
            return None

    class _Session:
        config = type("_Config", (), {"pluginmanager": _Manager()})()

    thread, release = _parked_thread(daemon=False)
    original = suite_conftest._faulthandler.dump_traceback_later
    suite_conftest._faulthandler.dump_traceback_later = lambda *a, **k: None
    try:
        suite_conftest.pytest_sessionfinish(_Session(), 0)
    finally:
        suite_conftest._faulthandler.dump_traceback_later = original
        release.set()
        thread.join(timeout=5)

    captured = capsys.readouterr()
    assert "non-daemon thread(s) are still running" in captured.err
    assert "a-parked-thread" in captured.err


def test_a_distributed_worker_says_it_on_stderr(capsys):
    """A worker's terminal reporter prints where nobody is looking.

    The wedged process this exists for is usually a worker, and its report
    has to reach the controller's log rather than a reporter the run
    discards. Its own stderr is the channel that does.
    """
    import tests.conftest as suite_conftest

    printed = []

    class _Reporter:
        def write_line(self, message, **kwargs):
            printed.append(message)

    class _Manager:
        def get_plugin(self, name):
            return _Reporter()

    class _Config:
        pluginmanager = _Manager()
        workerinput = {"workerid": "gw3"}

    class _Session:
        config = _Config()

    thread, release = _parked_thread(daemon=False)
    original = suite_conftest._faulthandler.dump_traceback_later
    suite_conftest._faulthandler.dump_traceback_later = lambda *a, **k: None
    try:
        suite_conftest.pytest_sessionfinish(_Session(), 0)
    finally:
        suite_conftest._faulthandler.dump_traceback_later = original
        release.set()
        thread.join(timeout=5)

    captured = capsys.readouterr()
    assert "a-parked-thread" in captured.err
    assert printed == []
