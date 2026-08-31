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

    def test_an_application_exists_for_every_test_in_this_file(self, qapp):
        from PySide6.QtWidgets import QApplication

        assert QApplication.instance() is not None
        assert qapp is not None
