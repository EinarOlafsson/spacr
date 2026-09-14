"""The process-wide FlowView collector is put back after every test.

``spacr.flowview.trace`` holds ONE collector for the process. ``enable()``
installs a new one into a module global and ``disable()`` does not put it
back, so whatever a test installs is what every later test in the session
sees -- including tests in other files that build a panel on it.

WHAT THAT COST, measured on this tree 2026-09-14 in alphabetical file order::

    pytest tests/flowview -q -p no:cacheprovider -p no:randomly -m "not gpu"
    150 passed, 7 errors

    AttributeError: '_Live' object has no attribute 'drain'
    spacr/flowview/panel.py:468

``_Live`` is a two-line stub from
``tests/flowview/test_cov_r8_classify_flowview_tails.py``. All seven errors
are in ``test_the_panel_is_a_box_you_can_resize.py``, whose ``panel`` fixture
builds ``FlowViewPanel(get_collector(), embedded=True)`` -- it asked for the
process collector and was handed somebody else's stub.

AND IT IS NOT FLAKY, which the ledger note for this item had backwards. The
note recorded fifteen runs on a quiet box without a reproduction and concluded
the original red was a busy machine. ``pytest-randomly`` is installed here and
shuffles the file order; pin the order with ``-p no:randomly`` and the failure
is 100% reproducible, because it is a question of which file runs first and
not of how loaded the machine is.

These tests cover the guard from five sides: the restore itself; that it
reaches an arbitrary test without being asked for AND is defined at the root
rather than in this directory, which is the difference between guarding the
whole suite and guarding one folder; that it does NOT import the tracer,
which would have spread somebody else's fault to every test in the suite;
the one case where there is no BEFORE to restore; and both cross-file
collisions, re-run for real in their own processes.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from spacr.flowview import trace
from spacr.flowview.collector import Collector
from tests.child_env import child_env
from tests.flowview_trace_state import (
    FLOWVIEW_TRACE,
    flowview_trace_module,
    flowview_trace_snapshot,
    give_back_an_untraced_process,
    restore_flowview_trace_to,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The single test that installed the stub, and the file that tripped over it.
#: Named rather than described: if either moves, the end-to-end test below
#: exits non-zero and says so instead of quietly checking nothing.
LEAKING_TEST = (
    "tests/flowview/test_cov_r8_classify_flowview_tails.py"
    "::TestTheCollectorItCollectsWith"
    "::test_a_live_graph_is_reused_rather_than_rebuilt"
)
VICTIM_FILE = "tests/flowview/test_the_panel_is_a_box_you_can_resize.py"

#: The same collision in a session where NOTHING imports the tracer while
#: collecting -- both of these reach it from inside a test body. That is the
#: case the guard cannot snapshot, and the one the reload closes.
LAZY_VICTIM_FILE = (
    "tests/qt/test_flowview_has_no_black_box_and_no_yellow_rim.py")

#: Not a FlowView test at all. It fails the moment anything in the session
#: imports ``spacr.flowview``, for a reason that predates this guard, so it is
#: the witness that the guard does not import it.
SPLIT_MODULE_GUARD = (
    "tests/qt/test_zz_a_reimported_module_is_put_back_properly.py")

#: A test as far from FlowView as this suite goes: plain ``tests/``, no Qt, no
#: tracer, nothing that has ever heard of a collector. The guard has to reach
#: it, and pytest has to name the ROOT conftest as where it comes from.
A_TEST_THAT_KNOWS_NOTHING_ABOUT_FLOWVIEW = (
    "tests/test_logging_util.py::test_setup_is_idempotent")

#: Where the fixture must be defined, as pytest reports it.
THE_ROOT_CONFTEST = "tests/conftest.py"

#: The fixture's name, once. Every assertion about it reads this.
THE_GUARD = "_the_flowview_collector_is_put_back"


def _child_pytest(targets):
    """Run ``targets`` in a fresh pytest, in the order given.

    A subprocess rather than an in-process check, for the reason
    ``tests/test_the_global_figure_style_is_put_back.py`` gives for the same
    shape: what is being tested is one test deciding state for the NEXT one,
    which is only visible across a session and is invisible from inside the
    session doing the leaking.

    ``-p no:randomly`` on purpose. The order is the whole question, and a
    shuffled child would pass roughly half the time whether or not the guard
    exists -- which is how this defect was first written off as flaky.

    :param targets: pytest file paths or node ids, relative to the repo root.
    :returns: the finished :class:`subprocess.CompletedProcess`.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", *targets,
         "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
         "-m", "not gpu"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
        env=child_env(qt=True, pythonpath=str(REPO_ROOT),
                      CUDA_VISIBLE_DEVICES="", SPACR_TEST_MEMORY_GB="8"))


def _child_fixture_report(nodeid):
    """Ask a fresh pytest which fixtures one test actually gets, and whence.

    ``--fixtures-per-test`` is pytest's own answer to the question, so this
    reads the registry rather than the source: each line is
    ``<name> -- <file>:<line>``, which carries BOTH halves of what has to be
    true -- that the guard reaches the test, and that the conftest handing it
    over is the root one.

    Verbosity has to come out positive or pytest hides every fixture whose
    name starts with an underscore, which is all of this suite's guards, so
    there is no ``-q`` here.

    :param nodeid: one test's node id, relative to the repo root.
    :returns: the finished :class:`subprocess.CompletedProcess`.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--fixtures-per-test", nodeid,
         "-v", "-p", "no:randomly", "-p", "no:cacheprovider"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
        env=child_env(qt=True, pythonpath=str(REPO_ROOT),
                      CUDA_VISIBLE_DEVICES="", SPACR_TEST_MEMORY_GB="8"))


class _Stub:
    """A collector-shaped object with nothing a real collector has."""

    def snapshot(self):
        """Answer the one call ``_collector_for_open_panel`` makes."""
        class _Graph:
            nodes = {"a": object()}
        return _Graph()


# ---------------------------------------------------------------------------
# The restore itself
# ---------------------------------------------------------------------------

def test_a_stolen_collector_is_handed_back():
    """The whole guard in one line: install a stub, restore, it is gone."""
    before = trace.get_collector()
    snapshot = flowview_trace_snapshot(trace)

    stub = _Stub()
    trace.enable(stub)
    assert trace.get_collector() is stub, (
        "enable() no longer installs the collector it is given; this file "
        "is then guarding a hazard that has moved")

    assert restore_flowview_trace_to(trace, snapshot) is True
    assert trace.get_collector() is before
    assert hasattr(trace.get_collector(), "drain"), (
        "the restored collector cannot be drained, which is the exact "
        "AttributeError this guard exists to prevent")


def test_the_enabled_flag_comes_back_with_it():
    """``enable`` is the only way back to a collector and it flips the flag.

    Restoring the collector must not leave tracing switched on for the rest
    of the session -- every traced stage in every later test would start
    emitting events, which is a second global left moved.
    """
    trace.disable()
    snapshot = flowview_trace_snapshot(trace)
    assert trace.is_enabled() is False

    trace.enable(_Stub())
    assert trace.is_enabled() is True

    restore_flowview_trace_to(trace, snapshot)
    assert trace.is_enabled() is False


def test_it_stays_enabled_when_it_was_found_enabled():
    """The other direction, or the restore is just a disable in disguise."""
    snapshot_of_the_session = flowview_trace_snapshot(trace)
    try:
        trace.enable(trace.get_collector())
        snapshot = flowview_trace_snapshot(trace)
        assert trace.is_enabled() is True

        trace.disable()
        trace.enable(_Stub())

        restore_flowview_trace_to(trace, snapshot)
        assert trace.is_enabled() is True
    finally:
        restore_flowview_trace_to(trace, snapshot_of_the_session)


def test_a_test_that_moved_nothing_is_not_written_to():
    """The common case -- and the reason the guard is affordable.

    Almost every test in the suite never touches FlowView. Reporting that it
    rewrote the globals for all of them would mean this ran a write, and a
    lock, per test for nothing.
    """
    snapshot = flowview_trace_snapshot(trace)

    assert restore_flowview_trace_to(trace, snapshot) is False


# ---------------------------------------------------------------------------
# That it reaches an arbitrary test
# ---------------------------------------------------------------------------

def test_the_guard_is_wired_into_every_test(request):
    """An autouse fixture nobody can opt out of, asked from a plain test.

    The point of putting it in the ROOT conftest: this test never requests
    it, is in a subdirectory, and still has it. A fixture that only some
    tests carry leaves the leak possible for the rest.
    """
    assert THE_GUARD in request.fixturenames


@pytest.mark.integration
def test_the_guard_is_at_the_root_and_not_in_this_directory():
    """WHERE it lives, measured on a test that is nowhere near FlowView.

    The note asked for the fixture in ``tests/flowview/conftest.py`` AND in
    ``tests/conftest.py``. One root fixture covers strictly more than both --
    but only while it is AT the root, and the test above cannot tell the
    difference: it lives in ``tests/flowview/``, so a fixture in a conftest of
    this directory would reach it just as well and leave ``tests/qt/`` and
    plain ``tests/`` unguarded with every test in this file still green.
    Measured 2026-09-14 by moving the fixture into a new
    ``tests/flowview/conftest.py``: 13 passed, nothing said a word.

    So the question is asked of a test that could not be reached from here --
    plain ``tests/``, no Qt, no tracer -- and pytest is asked to name the file
    the fixture comes from, which is the half that pins ROOT rather than
    merely REACHES.
    """
    done = _child_fixture_report(A_TEST_THAT_KNOWS_NOTHING_ABOUT_FLOWVIEW)
    output = done.stdout + done.stderr

    assert "collected 1 item" in output, (
        f"{A_TEST_THAT_KNOWS_NOTHING_ABOUT_FLOWVIEW} did not collect, so "
        "this proves nothing; the nodeid has moved and needs "
        "re-pointing:\n" + output[-3000:])

    lines = [line for line in output.splitlines()
             if line.startswith(THE_GUARD + " -- ")]
    assert lines, (
        f"a test in plain tests/ does not get {THE_GUARD}, so the guard has "
        "stopped covering everything outside tests/flowview/:\n"
        + output[-3000:])
    where = lines[0].split(" -- ", 1)[1]
    assert where.startswith(THE_ROOT_CONFTEST + ":"), (
        f"{THE_GUARD} is defined in {where}, not {THE_ROOT_CONFTEST}. A "
        "conftest covers only the directory beneath it; anywhere but the "
        "root leaves the tests that build panels from tests/qt/ and plain "
        "tests/ with no teardown at all")


def test_the_collector_this_test_was_handed_is_a_real_one():
    """The canary. Whatever ran before this, the process collector is sane.

    Under a shuffled order this test lands in a different place every run,
    so across runs it is asking the question from everywhere at once.
    """
    collector = trace.get_collector()

    assert isinstance(collector, Collector), (
        f"the process collector is a {type(collector).__name__}, which an "
        "earlier test installed and did not take back")


# ---------------------------------------------------------------------------
# Why patching the lookup function is not enough
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_patching_the_lookup_still_installs_the_stub_globally(qtbot,
                                                              monkeypatch):
    """THE INVERTED CLAIM, asserted rather than argued.

    The ledger note says the leak is a ``global`` assignment monkeypatch
    cannot observe, "so the files that patch the FUNCTION are safe". The
    second half is false, and the one test that produced the measured
    failure is a function-patching test.

    ``LazyFlowViewSection._collector_for_open_panel`` ends
    ``return enable(collector)``, so the value a patched ``get_collector``
    returned is written into the global by production code. Undoing the
    patch cannot undo that -- only the teardown in ``tests/conftest.py``
    does, which is why this test can be written at all without leaking.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens import classify
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key=classify.HOST_KEY)
    qtbot.addWidget(screen)
    section = classify.LazyFlowViewSection(screen)
    qtbot.addWidget(section)

    stub = _Stub()
    monkeypatch.setattr(trace, "get_collector", lambda: stub)
    section._collector_for_open_panel()
    monkeypatch.undo()

    assert trace.get_collector() is stub, (
        "the laundering path this guard exists for is gone; if "
        "_collector_for_open_panel no longer calls enable(), say so here "
        "rather than leaving a guard that proves nothing")
    assert not hasattr(stub, "drain"), (
        "the stub grew a drain(); it can no longer stand in for the object "
        "that broke the panel")


# ---------------------------------------------------------------------------
# The collision itself, re-run
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_the_two_files_that_collided_now_pass_in_one_session():
    """The measured failure, run again in a fresh process in fixed order.

    ``150 passed, 7 errors`` for the whole directory before the guard, and
    ``3 passed, 7 errors`` for just these two -- every error in the file that
    asks for the process collector and was handed somebody else's stub.
    """
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")

    done = _child_pytest([LEAKING_TEST, VICTIM_FILE])
    output = done.stdout + done.stderr

    assert "AttributeError" not in output, output[-3000:]
    assert done.returncode == 0, (
        "the two files that collided still do not pass together. A non-zero "
        "exit with no failures means one of the two nodeids above has moved "
        "and this guard needs re-pointing:\n" + output[-3000:])

    passed = re.search(r"(\d+) passed", output)
    assert passed is not None, output[-3000:]
    assert int(passed.group(1)) >= 10, (
        f"only {passed.group(1)} tests ran, so the collision was not "
        "reproduced end to end:\n" + output[-3000:])


# ---------------------------------------------------------------------------
# What it refuses to do: import the thing it guards
# ---------------------------------------------------------------------------

def test_the_guard_asks_sys_modules_and_never_imports(monkeypatch):
    """No tracer in the process means nothing to protect, and no import.

    An eager ``from spacr.flowview import trace`` here would be simpler and
    would drag ``spacr.flowview`` into every test in the suite -- and that
    package makes ``tests/qt/test_zz_a_reimported_module_is_put_back_properly
    .py`` fail all on its own, for a fault that is nothing to do with this.
    """
    monkeypatch.delitem(sys.modules, FLOWVIEW_TRACE, raising=False)

    assert flowview_trace_module() is None
    assert FLOWVIEW_TRACE not in sys.modules, (
        "asking for the tracer imported it")


@pytest.mark.integration
def test_the_guard_does_not_spread_the_split_module_fault():
    """The file that an eager import would have broken, run on its own.

    MEASURED 2026-09-14, and it is not this item's bug either way:
    ``spacr/flowview/__init__.py`` re-exports a function named ``export``
    over its own submodule of that name, so

        pytest tests/qt/test_no_panel_is_a_black_slab.py <this file> \
               -p no:randomly

    fails on ``test_no_spacr_submodule_is_split_from_its_package`` with this
    guard switched off entirely. What must not happen is this guard making
    that failure universal by importing the package for every test.
    """
    done = _child_pytest([SPLIT_MODULE_GUARD])
    output = done.stdout + done.stderr

    assert "spacr.flowview.export" not in output, (
        "the teardown guard imported spacr.flowview, which splits that "
        "package's `export` name and fails a file that never asked for "
        "FlowView:\n" + output[-3000:])
    assert done.returncode == 0, output[-3000:]


# ---------------------------------------------------------------------------
# The tracer a single test imports for itself
# ---------------------------------------------------------------------------

def test_a_tracer_one_test_imported_is_handed_back_empty():
    """There is no BEFORE to restore, so import-time state is rebuilt.

    Without this the hole is one test wide and a whole session long: the
    next test's setup would snapshot the STUB as its baseline and every
    restore after that would put the stub carefully back.
    """
    snapshot = flowview_trace_snapshot(trace)
    try:
        stub = _Stub()
        trace.enable(stub)

        give_back_an_untraced_process(trace)

        handed_back = trace.get_collector()
        assert handed_back is not stub
        assert isinstance(handed_back, Collector)
        assert handed_back.snapshot().nodes == {}, (
            "a reused collector, not the empty one a fresh import makes")
        assert handed_back.drain() == 0
    finally:
        restore_flowview_trace_to(trace, snapshot)


def test_the_module_object_survives_being_handed_back():
    """Reload, not eviction -- and the difference is the whole safety case.

    ``spacr.flowview.panel`` and two other modules hold ``get_collector``
    from ``from .trace import get_collector``. Dropping the tracer out of
    ``sys.modules`` would leave those three reading a dead module's global
    while everything else read a new one. Reload writes into the SAME module
    dictionary, so the old function keeps answering with the collector this
    put back.
    """
    from spacr.flowview import panel as panel_module

    snapshot = flowview_trace_snapshot(trace)
    try:
        before_module = sys.modules[FLOWVIEW_TRACE]

        give_back_an_untraced_process(trace)

        assert sys.modules[FLOWVIEW_TRACE] is before_module
        assert panel_module.get_collector() is trace.get_collector(), (
            "panel.py's bound get_collector and the tracer's own now "
            "disagree, which is the split-module hazard this avoided")
    finally:
        restore_flowview_trace_to(trace, snapshot)


@pytest.mark.integration
def test_a_lazily_imported_tracer_does_not_leak_into_the_next_file():
    """The second collision, in the session shape the snapshot cannot see.

    Neither of these two files imports the tracer while pytest is
    collecting -- both reach it from inside a test body -- so the guard has
    no BEFORE for the test that leaks. MEASURED 2026-09-14 with the reload
    removed: 3 passed, 4 errors, the same
    ``AttributeError: '_Live' object has no attribute 'drain'``.
    """
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")

    done = _child_pytest([LEAKING_TEST, LAZY_VICTIM_FILE])
    output = done.stdout + done.stderr

    assert "AttributeError" not in output, output[-3000:]
    assert done.returncode == 0, output[-3000:]

    passed = re.search(r"(\d+) passed", output)
    assert passed is not None, output[-3000:]
    assert int(passed.group(1)) >= 5, (
        f"only {passed.group(1)} tests ran, so the collision was not "
        "reproduced end to end:\n" + output[-3000:])
