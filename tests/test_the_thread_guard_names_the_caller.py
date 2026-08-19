"""Qt's own warning names nothing. This one names the caller.

    QBasicTimer::start: Timers cannot be started from another thread

No file, no function, no thread -- and three attempts at finding the caller by
reading code have missed it. The event arrives during a real run and is followed
immediately by the process dying, so there is no chance to switch
instrumentation on afterwards: it has to be on already.
"""

import logging
import threading

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest
from PySide6.QtCore import QTimer


def test_an_off_thread_start_is_recorded_with_a_stack(qtbot, caplog):
    from spacr.qt import thread_guard

    thread_guard.install()
    before = len(thread_guard.offences())

    timer = QTimer()
    qtbot.addWidget_ = None  # a bare QTimer needs no widget parent

    def start_it_from_the_wrong_thread():
        timer.start(50)

    with caplog.at_level(logging.WARNING):
        worker = threading.Thread(target=start_it_from_the_wrong_thread)
        worker.start()
        worker.join(10)

    recorded = thread_guard.offences()
    assert len(recorded) == before + 1
    # THE POINT: the stack names the function that did it.
    assert "start_it_from_the_wrong_thread" in recorded[-1]


def test_a_gui_thread_start_is_not_recorded(qtbot):
    """The guard must be silent on the ordinary path, or its output is noise
    and nobody will read the one line that matters.

    SCOPED TO THIS TIMER rather than to a global count. Since the guard began
    checking object AFFINITY it also sees Qt's own internal timers, including
    ones belonging to threads another test left running -- so a count is
    flaky in a way that says nothing about the ordinary path.
    """
    from spacr.qt import thread_guard

    thread_guard.install()
    timer = QTimer()

    def an_ordinary_gui_thread_start():
        timer.start(50)
        timer.stop()

    an_ordinary_gui_thread_start()

    assert not any("an_ordinary_gui_thread_start" in stack
                   for stack in thread_guard.offences())


def test_installing_twice_does_not_double_the_wrapper():
    """Wrapping the wrapper would double every line and make the stacks
    harder to read rather than easier."""
    from spacr.qt import thread_guard

    thread_guard.install()
    assert thread_guard.install() is False


def test_the_application_installs_it():
    import inspect

    from spacr.qt import app as module

    assert "_install_thread_guard" in inspect.getsource(module)
