"""Teardown is bounded from the last test onwards, not from the last hook.

A run that finishes every test and never prints a summary is the worst
failure this suite has: ``--timeout`` is a per-test guard and is already gone,
no test is to blame, and under ``-n`` every worker sits on a core waiting for
a controller that is waiting for them. The shutdown watchdog in
``tests/conftest.py`` turns that into a stack dump and a non-zero exit -- but
only for the part of teardown it is armed for.

The window that matters is the one between the last test and the summary
line. Plugins do their most expensive work there: pytest-cov writes the run's
coverage data from the tail of its ``pytest_runtestloop`` wrapper, and a
distributed worker reports itself finished from the tail of xdist's
``pytest_sessionfinish`` wrapper. A stall in that window has already written
its data and has not yet printed anything, which is exactly the shape that
gets reported, and a watchdog armed at the END of ``pytest_sessionfinish``
sits on the far side of it.

So both halves are driven here for real, in subprocesses: the hazard, with
the loop-end arming switched off, and the guard, with it on.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import (
    REPORT_WATCHDOG_ENV,
    REPORT_WATCHDOG_S,
    RETAINED_WIDGETS_WORTH_SAYING,
    SHUTDOWN_WATCHDOG_ENV,
    SHUTDOWN_WATCHDOG_S,
    _qt_things_report,
    qt_things_that_outlive_the_session,
    teardown_budget,
)

REPO = str(Path(__file__).resolve().parent.parent)

#: Long enough that a fired watchdog is unmistakable, short enough that the
#: hazard half of each pair does not cost the suite a minute.
BUDGET_S = 5

#: What "it never came back" costs. Comfortably above BUDGET_S so a run that
#: is merely slow is not mistaken for one that is stuck, and short enough that
#: the two demonstrations of the hazard do not dominate this file.
PATIENCE_S = 25

#: What a fired watchdog writes. Matched exactly, because "the process died"
#: is also what a crash looks like and this has to distinguish them.
FIRED = f"Timeout (0:00:0{BUDGET_S})"


_BLOCKING_PLUGIN = '''\
"""A plugin that stalls in teardown, the way a wedged run's does."""
import threading

import pytest


@pytest.hookimpl({marker})
def {hook}({args}):
    {body}
'''

#: The suite's own hooks, loaded the way a real run loads them. A SEPARATE
#: module from the stalling plugin, because two implementations of one hook in
#: one module is not two implementations -- the second name wins and the first
#: is never registered, which would quietly test nothing.
_SUITE_CONFTEST = '''\
import sys
sys.path.insert(0, {repo!r})
from tests.conftest import pytest_runtestloop, pytest_sessionfinish  # noqa
'''

_ONE_TEST = '''\
def test_the_run_itself_is_fine():
    """The tests pass. It is getting from here to the summary that fails."""
    assert True
'''


def _run_that_stalls_in(tmp_path, hook, marker, args, body, env_extra):
    """Run one passing test under a plugin that never leaves ``hook``."""
    # A real file in the probe directory keeps every supported pytest rooted
    # at the synthetic suite.  Pytest 8.0 derives the root from ``-c
    # /dev/null`` as ``/dev`` and then attempts to collect system-managed
    # entries such as ``/dev/::tmp``; collection can fail before the loop-end
    # watchdog is ever armed, which tests the filesystem instead of teardown.
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        _SUITE_CONFTEST.format(repo=REPO))
    (tmp_path / "stalling_plugin.py").write_text(
        _BLOCKING_PLUGIN.format(marker=marker, hook=hook, args=args,
                                body=body))
    (tmp_path / "test_one.py").write_text(_ONE_TEST)

    env = dict(os.environ)
    env[SHUTDOWN_WATCHDOG_ENV] = str(BUDGET_S)
    env[REPORT_WATCHDOG_ENV] = str(BUDGET_S)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "test_one.py", "-q",
         "-p", "no:randomly", "-p", "no:cacheprovider",
         "-p", "stalling_plugin", "-c", "pytest.ini"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=PATIENCE_S)


_PARK_FOREVER = "threading.Event().wait()"


def test_a_stall_in_another_plugins_sessionfinish_is_bounded(tmp_path):
    """The live defect's shape: data written, nothing printed, no exit.

    ``pytest_sessionfinish`` is one hook with many implementations, and this
    suite's runs last of them. A plugin that stalls in its own implementation
    therefore stalls before the watchdog is armed -- unless the budget started
    earlier, which is what this pins.
    """
    done = _run_that_stalls_in(
        tmp_path, "pytest_sessionfinish", "tryfirst=True",
        "session, exitstatus", _PARK_FOREVER, {})

    # The summary line is exactly what a stall here costs, so its ABSENCE is
    # the shape being reproduced: the test ran, and nobody was ever told.
    assert "[100%]" in done.stdout, done.stdout + done.stderr
    assert "passed" not in done.stdout, done.stdout
    assert done.returncode != 0
    assert FIRED in done.stderr, done.stderr
    assert "pytest_sessionfinish" in done.stderr, (
        "the dump has to name the hook that stalled, or it names nobody")


def test_that_stall_is_endless_when_the_budget_starts_at_sessionfinish(
        tmp_path):
    """The hazard, demonstrated rather than asserted.

    Switching the loop-end arming off is enough to restore the wedge, which
    is what makes the test above evidence that the arming point is what fixed
    it rather than something else in the same file.
    """
    with pytest.raises(subprocess.TimeoutExpired):
        _run_that_stalls_in(
            tmp_path, "pytest_sessionfinish", "tryfirst=True",
            "session, exitstatus", _PARK_FOREVER, {REPORT_WATCHDOG_ENV: "0"})


def test_a_stall_in_another_plugins_test_loop_teardown_is_bounded(tmp_path):
    """Where the coverage write lives, and where a wedged run was last seen.

    pytest-cov stops and saves coverage from the tail of its own
    ``pytest_runtestloop`` wrapper -- before any ``sessionfinish`` runs. A run
    found with its coverage data on disk and no summary line stalled at or
    after that point, so that point has to be inside the budget.
    """
    done = _run_that_stalls_in(
        tmp_path, "pytest_runtestloop", "hookwrapper=True, tryfirst=True",
        "session", f"yield\n    {_PARK_FOREVER}", {})

    assert done.returncode != 0
    assert FIRED in done.stderr, done.stderr
    assert "pytest_runtestloop" in done.stderr, done.stderr


def test_the_reporting_phase_gets_a_longer_budget_than_the_interpreter():
    """Combining a distributed run's coverage is slow and must not be shot."""
    assert REPORT_WATCHDOG_S > SHUTDOWN_WATCHDOG_S


def test_a_distributed_worker_is_shot_on_the_short_budget_instead():
    """None of the slow work the long budget exists for happens in a worker.

    A collocated worker saves its coverage data and stops; combining and
    reporting belong to the controller. A worker that then goes quiet is the
    process most worth killing quickly, because the whole run is waiting on it
    and until it says something there is nothing to read.
    """
    class _Controller:
        pass

    class _Worker:
        workerinput = {"workerid": "gw3"}

    assert teardown_budget(_Controller()) == REPORT_WATCHDOG_S
    assert teardown_budget(_Worker()) == SHUTDOWN_WATCHDOG_S


def test_turning_the_shutdown_watchdog_off_turns_this_one_off_too():
    """One knob, so a debugging session is never half-guarded.

    Read from the environment at import, so the value is checked the way a
    developer sets it rather than by re-deriving the arithmetic.
    """
    env = dict(os.environ)
    env[SHUTDOWN_WATCHDOG_ENV] = "0"
    env.pop(REPORT_WATCHDOG_ENV, None)
    done = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {REPO!r});"
         " import tests.conftest as c;"
         " print(c.SHUTDOWN_WATCHDOG_S, c.REPORT_WATCHDOG_S)"],
        env=env, capture_output=True, text=True, timeout=PATIENCE_S)

    assert done.stdout.split() == ["0.0", "0.0"], done.stdout + done.stderr


# ---------------------------------------------------------------------------
# Naming what Python cannot see
# ---------------------------------------------------------------------------

def test_nothing_is_claimed_when_nothing_is_holding_the_process(monkeypatch):
    """Silence when there is nothing to say, or the report is noise.

    Every run of this suite ends with daemon threads and a clean process, and
    a report that always prints teaches the reader to skip it.
    """
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    said = qt_things_that_outlive_the_session()

    assert said == [], said


def test_a_child_process_that_outlived_the_run_is_named(monkeypatch):
    """Python joins one at exit, and nothing else here would mention it."""
    class _Child:
        name = "a-forgotten-child"
        pid = 4242

    monkeypatch.setattr("multiprocessing.active_children", lambda: [_Child()])
    said = qt_things_that_outlive_the_session()

    assert any("a-forgotten-child" in line for line in said), said
    assert any("4242" in line for line in said), said


def test_the_report_says_why_the_thread_report_did_not_mention_these():
    """A reader who has just been told "no non-daemon threads" needs it."""
    report = _qt_things_report(["    a QApplication is still alive"])

    assert "1 thing(s)" in report
    assert "a QApplication is still alive" in report
    assert "None of it is joined by Python" in report


_QAPP_PROBE = '''\
import sys
sys.path.insert(0, {repo!r})
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication([])
holder = QWidget()
holder.show()
for _ in range({extra}):
    QWidget(holder)

from tests.conftest import qt_things_that_outlive_the_session
for line in qt_things_that_outlive_the_session():
    print(line.strip())
print("END")
'''


def _qapp_probe(tmp_path, extra):
    """Ask a real application, with ``extra`` spare widgets, what it says."""
    pytest.importorskip("PySide6")
    script = tmp_path / "qapp_probe.py"
    script.write_text(
        textwrap.dedent(_QAPP_PROBE.format(repo=REPO, extra=extra)))

    done = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=PATIENCE_S)
    assert "END" in done.stdout, done.stdout + done.stderr
    return done.stdout


def test_a_big_retained_widget_tree_is_named_with_its_size(tmp_path):
    """The thing the thread report structurally cannot mention.

    A ``QThread`` that has run Python is a DAEMON ``Dummy-N`` and is filtered
    out with the harmless ones; a ``QApplication`` is not a thread at all. Yet
    a retained widget tree is destroyed one object at a time on the way out,
    which is a finished run burning a core with nothing to blame.

    Driven in a subprocess against a real application rather than a stub: the
    probe's whole claim is that it reads live Qt state without touching it,
    and a stub would prove nothing about that.
    """
    out = _qapp_probe(tmp_path, RETAINED_WIDGETS_WORTH_SAYING)

    assert "a QApplication is still alive" in out, out
    assert "widget(s)" in out
    assert "1 top-level window(s)" in out, out


def test_an_ordinary_application_is_not_mentioned_at_all(tmp_path):
    """Every Qt run ends with a live QApplication, so saying so is noise.

    A report that prints on every run is a report nobody reads, and this one
    has to be worth reading on the single run where it fires.
    """
    out = _qapp_probe(tmp_path, 0)

    assert "QApplication" not in out, out
