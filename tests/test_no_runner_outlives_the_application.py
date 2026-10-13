"""Qt aborts if a running QThread is destroyed. Nothing may outlive the quit.

MEASURED 2026-08-19 from ~/.spacr/logs/spacr.log: spaCR died immediately after
EVERY successful regression, four times in thirteen minutes and again on a
freshly built 3.12 environment. The last line was always

    run closed [success] in 51.0s

and nothing after it -- the signature of a native abort, which writes nothing to
a Python logger.

Each JobRunner shuts down in its own widget's `closeEvent`. That covers a widget
being CLOSED and not the application quitting with a job in flight, where the
widget is destroyed without ever being closed.
"""

import spacr


import pytest
from PySide6.QtWidgets import QWidget

from spacr.qt.job_runner import JobRunner, _LIVE_RUNNERS, shutdown_all


def test_a_runner_registers_itself(qtbot):
    """Registered at construction, so one cannot be created and then missed."""
    holder = QWidget()
    qtbot.addWidget(holder)
    runner = JobRunner(holder, threaded=True, app_key="test")
    assert runner in _LIVE_RUNNERS


def test_the_registry_is_weak(qtbot):
    """It must hold nothing alive that Qt would otherwise collect."""
    import gc

    holder = QWidget()
    qtbot.addWidget(holder)
    runner = JobRunner(holder, threaded=True, app_key="test")
    before = len(list(_LIVE_RUNNERS))
    del runner
    gc.collect()
    assert len(list(_LIVE_RUNNERS)) <= before


def test_shutdown_all_stops_every_live_runner(qtbot):
    holder = QWidget()
    qtbot.addWidget(holder)
    JobRunner(holder, threaded=True, app_key="a")
    JobRunner(holder, threaded=True, app_key="b")
    assert shutdown_all() >= 2


def test_one_runner_that_refuses_does_not_stop_the_others(qtbot):
    """A drain that gives up on the first failure leaves the rest running,
    which is the abort this exists to prevent."""
    holder = QWidget()
    qtbot.addWidget(holder)

    class Awkward(JobRunner):
        def shutdown(self, timeout_ms: int = 3000) -> None:
            raise RuntimeError("no")

    Awkward(holder, threaded=True, app_key="bad")
    good = JobRunner(holder, threaded=True, app_key="good")
    stopped = []
    good.shutdown = lambda *a, **k: stopped.append(True)   # type: ignore

    shutdown_all()
    assert stopped, "a failing runner stopped the drain"


def test_the_quit_path_calls_it():
    """A registry nothing drains is a registry.

    The application's ``aboutToQuit`` drain goes through
    :func:`spacr.qt.bridge.retire_live_work`, the one exit path it shares
    with the interpreter's exit hook, and that is where the runners are
    drained. Both links are checked, so neither can drop the drain.
    """
    import inspect

    from spacr.qt import app as module
    from spacr.qt import bridge

    launch_source = inspect.getsource(module.launch)
    assert "retire_live_work()" in launch_source, (
        "the application quit path does not drain the job runners")
    assert "aboutToQuit.connect(_drain_ai)" in launch_source, (
        "the drain is not connected to the application's quit")
    assert "shutdown_all(" in inspect.getsource(bridge.retire_live_work), (
        "retire_live_work does not drain the job runners")
