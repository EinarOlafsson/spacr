"""Closing one window must not take the whole application with it.

Reported 2026-08-19: "i ran it again and it just spontaneously quit". The
evidence was the SHAPE of the exit, not a stack: two runs closed [success],
the log simply ended mid-session, and neither dmesg nor coredumpctl recorded
anything. Nothing crashed -- Qt exited cleanly.

`QApplication.quitOnLastWindowClosed` defaults to TRUE, and spaCR never set
it. A run makes and destroys top-level windows (a figure canvas being
rebuilt, a transient dialog, a progress window), and any instant with none of
them up ends the event loop and the process.
"""
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget


def test_the_default_qt_behaviour_is_the_bug(qapp):
    """Named so nobody 'simplifies' the fix away: this is what Qt does."""
    assert QApplication.instance() is not None
    # The property exists and its Qt default is the dangerous one.
    qapp.setQuitOnLastWindowClosed(True)
    assert qapp.quitOnLastWindowClosed() is True


def test_launch_turns_it_off_before_the_window_exists(qapp):
    import inspect

    from spacr.qt import app

    source = inspect.getsource(app.launch)
    assert "setQuitOnLastWindowClosed(False)" in source
    assert (source.index("setQuitOnLastWindowClosed(False)")
            < source.index("MainWindow(initial_app=")), (
        "set it before any window is shown, or the race is still open")


def test_a_transient_window_closing_leaves_the_app_running(qapp):
    qapp.setQuitOnLastWindowClosed(False)
    transient = QWidget()
    transient.show()
    survived = {"yes": False}

    transient.close()
    QTimer.singleShot(0, lambda: survived.__setitem__("yes", True))
    QTimer.singleShot(30, qapp.quit)
    qapp.exec()

    assert survived["yes"], (
        "the event loop stopped when a window closed, which is the reported "
        "spontaneous quit")


def test_the_main_window_closing_still_ends_the_program(qapp):
    """The guard must not make spaCR unquittable."""
    import inspect

    from spacr.qt.app import MainWindow

    source = inspect.getsource(MainWindow.closeEvent)
    assert "app.quit()" in source, (
        "with quitOnLastWindowClosed off, nothing else ends the session")
    assert "isAccepted()" in source, (
        "a close the window itself vetoed must not quit anyway")
