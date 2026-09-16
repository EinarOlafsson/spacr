"""The backdrop's two hooks into the application, and their absences.

The render thread is deliberately NOT parented to the widget: a QThread
whose parent is deleted while it runs prints "Destroyed while thread is
still running" and takes the process down. So the join has to be hung off
something that outlives the widget, and that is ``aboutToQuit`` on the
application -- which means both connecting and disconnecting it have to
survive there being no application to hook.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("numba")

from spacr.qt.widgets import fractal_travel as F

pytestmark = pytest.mark.qt


class _CheapEngine:
    """An engine that renders instantly, so no test waits on numba."""

    def __init__(self, thread_count):
        self.thread_count = thread_count

    def render(self, width, height, *_args, **_kwargs):
        import numpy as np

        return np.zeros((height, width, 3), dtype=np.uint8)


#: Run in a fresh interpreter by the real-quit test: frees three backdrops
#: with their screens, keeps one, quits, and reports whose render threads
#: the application's ``aboutToQuit`` hooks joined.
_REAL_QUIT_CHILD = r'''
import json

import numpy as np
from PySide6.QtCore import SIGNAL, QEvent, QTimer
from PySide6.QtWidgets import QApplication, QWidget

import spacr
from spacr.qt.widgets import fractal_travel as F


class Cheap:
    def __init__(self, thread_count):
        self.thread_count = thread_count

    def render(self, width, height, *_args, **_kwargs):
        return np.zeros((height, width, 3), dtype=np.uint8)


app = QApplication([])
F.OrbitEngine = Cheap


def build():
    return F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                              F.RuntimeControls(),
                              F.HardwareProfile(logical_cpus=4))


def flush():
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


freed = []
for _ in range(3):
    screen = QWidget()
    backdrop = build()
    backdrop.setParent(screen)
    freed.append(backdrop._thread)
    screen.deleteLater()
    flush()
live = build()

joined = []
real_join = F._quit_and_join_thread


def spy(thread):
    if thread is live._thread:
        joined.append("live")
    elif any(thread is gone for gone in freed):
        joined.append("freed")
    else:
        joined.append("unknown")
    real_join(thread)


F._quit_and_join_thread = spy
QTimer.singleShot(0, app.quit)
app.exec()
F._quit_and_join_thread = real_join

live.shutdown()
live.deleteLater()
flush()
print("REPORT " + json.dumps({
    "spacr": spacr.__file__,
    "joined_at_quit": joined,
    "receivers_after_teardown": app.receivers(SIGNAL("aboutToQuit()")),
}), flush=True)
'''


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    made = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                              F.RuntimeControls(),
                              F.HardwareProfile(logical_cpus=4))
    yield made
    made.shutdown()
    made.deleteLater()


class TestShuttingDown:

    def test_a_shutdown_stops_the_timer_and_joins_the_thread(self, widget):
        widget.shutdown()

        assert widget._stopped is True
        assert not widget._timer.isActive()
        assert not widget._thread.isRunning(), "the render thread outlived it"

    def test_a_second_shutdown_is_a_no_op(self, widget):
        """"Safe to call twice" is in the docstring, and the close event
        calls it as well as the screen teardown."""
        widget.shutdown()
        widget.shutdown()

        assert widget._stopped is True

    def test_a_cleared_hook_does_not_stop_the_join(self, widget):
        """Whatever the disconnect does, the thread still gets joined.

        Joining is the one thing shutdown exists to do: an unjoined
        render thread is what prints "Destroyed while thread is still
        running" and takes the process with it. Anything raised on the
        way there would skip it.
        """
        widget._app_quit_join = None

        widget.shutdown()                          # must not raise

        assert widget._stopped is True
        assert not widget._thread.isRunning(), (
            "the thread was left running by a failed disconnect")

    def test_the_disconnect_guard_is_belt_and_braces_and_why(self, widget):
        """THE PIN for the except.

        ``except (RuntimeError, TypeError)`` covers the two ways PySide
        refuses a disconnect: a dead signal source, and an argument that
        is not a slot. Neither can happen here.

        A dead source would mean the QApplication was destroyed -- and
        then ``QApplication.instance()`` is None and the disconnect is
        skipped by the guard above it. A non-slot argument would mean
        ``_app_quit_join`` held something other than the lambda
        ``__init__`` puts there.

        What PySide does for the case that CAN arise -- a hook that was
        never connected, or was already disconnected -- is return False.
        That is asserted below, because a PySide that began raising for
        it instead is exactly the change that would make this handler
        live, and it should fail here rather than take the process down
        on a second shutdown.
        """
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance()
        assert application is not None

        assert application.aboutToQuit.disconnect(lambda: None) is False

        with pytest.raises(TypeError):
            application.aboutToQuit.disconnect("not a slot")

        # And the reason the code passes the lambda rather than None:
        # None means "disconnect EVERYTHING", which would take every
        # other widget's aboutToQuit hook with it.
        assert application.aboutToQuit.disconnect(None) is True, (
            "there was no connection to remove, so this test proved "
            "nothing about what None does")

        source = inspect.getsource(F._make_cpu_widget)
        assert "except (RuntimeError, TypeError):" in source
        assert "self._app_quit_join = (" in source, (
            "_app_quit_join is no longer set to a lambda in __init__, so "
            "the disconnect can now be handed something that is not a slot")

    def test_closing_the_widget_shuts_it_down(self, widget):
        widget.close()

        assert widget._stopped is True


class TestTheApplicationHookItself:

    def test_the_join_is_hung_off_the_application_not_the_widget(self,
                                                                 widget):
        assert callable(widget._app_quit_join)
        source = inspect.getsource(F._make_cpu_widget)
        assert "aboutToQuit.connect(self._app_quit_join)" in source

    def test_both_hooks_are_guarded_against_there_being_no_application(self):
        """THE PIN, for both arcs.

        ``QApplication.instance()`` is None only before one is made or
        after it is destroyed -- never inside a running test, because the
        fixture needs an application to build a widget at all.

        The guards are not decoration: this widget is also constructed by
        the offscreen thumbnail path and by ``--help``-style probes that
        import the module without starting an application, and an
        AttributeError on None there would turn a missing backdrop into a
        failed launch.
        """
        source = inspect.getsource(F._make_cpu_widget)
        assert source.count("application = QApplication.instance()") == 2, (
            "one of the two application lookups changed shape")
        assert source.count("if application is not None:") == 2, (
            "an application lookup is no longer guarded against None")

    def test_a_backdrop_freed_with_its_screen_takes_its_quit_hook_along(
            self, qapp, monkeypatch):
        """Deleting the screen takes the backdrop's ``aboutToQuit`` hook too.

        A backdrop is reparented into its screen and deleted with it, and a
        child freed with its parent is never sent ``closeEvent`` -- so
        ``shutdown``, which disconnects the hook, never runs. Before the fix
        every such teardown left one connection on the application, holding
        a finished render thread until quit: three screens, three receivers.

        Counted with ``receivers`` rather than by emitting ``aboutToQuit``,
        which would run every other test's hooks on the shared application.
        """
        import shiboken6
        from PySide6.QtCore import SIGNAL, QEvent
        from PySide6.QtWidgets import QWidget

        monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
        before = qapp.receivers(SIGNAL("aboutToQuit()"))
        for _ in range(3):
            # Counted per pass, so a hook left by the previous pass fails the
            # final assertion rather than this precondition.
            was = qapp.receivers(SIGNAL("aboutToQuit()"))
            screen = QWidget()
            backdrop = F._make_cpu_widget(
                F.Settings(pattern="orbit", backend="cpu"),
                F.RuntimeControls(), F.HardwareProfile(logical_cpus=4))
            backdrop.setParent(screen)
            # ASSERTED, or the final count proves nothing: a backdrop that
            # never hooked aboutToQuit would pass it just as well.
            assert qapp.receivers(SIGNAL("aboutToQuit()")) == was + 1, (
                "the backdrop did not hook aboutToQuit")
            screen.deleteLater()
            qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            qapp.processEvents()
            assert not shiboken6.isValid(backdrop), (
                "the screen was deleted without its backdrop")

        assert qapp.receivers(SIGNAL("aboutToQuit()")) == before, (
            "a backdrop freed with its screen left its aboutToQuit hook "
            "connected")

    def test_a_real_quit_joins_the_live_backdrop_and_no_freed_one(self):
        """What the application does when it quits, in a fresh interpreter.

        A child process, because quitting the shared test application is not
        something one test may do to the rest. Three backdrops are freed with
        their screens and one is left showing; ``_quit_and_join_thread`` is
        spied on, so the child reports whose threads the quit reached. Before
        the fix it reached all four -- three of them for widgets that no
        longer existed. The live one is then shut down and freed, the path on
        which the hook is disconnected twice, and that has to stay silent.
        """
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        import spacr

        # Rooted at the checkout this test imported, not wherever an editable
        # install points: the child has to run the code under test.
        root = Path(spacr.__file__).resolve().parent.parent
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                           else []))
        done = subprocess.run([sys.executable, "-c", _REAL_QUIT_CHILD],
                              cwd=root, env=env, capture_output=True,
                              text=True, timeout=300)

        assert done.returncode == 0, done.stderr
        lines = [line for line in done.stdout.splitlines()
                 if line.startswith("REPORT ")]
        assert lines, f"the child reported nothing\n{done.stderr}"
        report = json.loads(lines[-1][len("REPORT "):])
        assert Path(report["spacr"]).resolve().parent.parent == root
        assert report["joined_at_quit"] == ["live"], (
            "the quit ran a hook other than the live backdrop's -- a freed "
            "backdrop's aboutToQuit hook was still connected")
        assert report["receivers_after_teardown"] == 0
        assert "Traceback" not in done.stderr, done.stderr
        assert "Failed to disconnect" not in done.stderr, done.stderr

    def test_an_application_exists_for_every_test_in_this_file(self, qapp):
        from PySide6.QtWidgets import QApplication

        assert QApplication.instance() is not None
        assert qapp is not None
