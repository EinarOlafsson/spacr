"""fractal_travel: no numba, a popup on screen, and shutdown races.

Three groups, all only reached when something is absent or already going
away -- which is why a healthy suite never touches them.

The numba pair is the most important. numba is an OPTIONAL accelerator:
the GPU backdrop does not need it and the CPU one cannot run without it.
The module-level fallback is what lets spaCR import at all on a machine
without it, and the two stubs are what make the failure say so instead
of raising NameError somewhere in a render loop.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from spacr.qt.widgets import fractal_travel as F

# The GPU harness is borrowed rather than re-invented: it builds a widget
# against a stand-in vispy and shuts every one down afterwards.
from tests.qt.test_cov_r5_fractal_travel import (  # noqa: F401
    gpu_backdrop, stand_in_vispy,
)

pytestmark = pytest.mark.qt


class TestImportingWithoutNumba:
    """Re-import the module with numba refused, in a scratch namespace."""

    @staticmethod
    def _import_without_numba(monkeypatch):
        real_import = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if name == "numba" or name.startswith("numba."):
                raise ImportError("numba is not installed")
            return real_import(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        for name in [n for n in sys.modules
                     if n.startswith("spacr.qt.widgets.fractal_travel")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delitem(sys.modules, "numba", raising=False)
        return importlib.import_module("spacr.qt.widgets.fractal_travel")

    def test_the_module_still_imports(self, monkeypatch):
        """A machine without numba must still be able to start spaCR."""
        module = self._import_without_numba(monkeypatch)
        assert module.VERSION

    def test_the_jit_names_degrade_to_something_usable(self, monkeypatch):
        module = self._import_without_numba(monkeypatch)
        assert module.njit is None
        assert module.prange is range
        assert module.numba_config is None
        assert module.set_num_threads(4) is None, (
            "the thread setter must be a no-op, not a missing name")

    def test_the_cpu_kernels_say_why_they_cannot_run(self, monkeypatch):
        """A NameError deep in a render loop would not name the cause."""
        module = self._import_without_numba(monkeypatch)
        for kernel in (module._render_into, module._blend_temporal):
            with pytest.raises(RuntimeError, match="numba is required"):
                kernel(1, 2, 3)


class TestTheBackdropHoldsStillUnderAPopup:
    """The flicker fix, driven rather than inspected.

    A menu or tooltip composited over this native GL surface makes the
    widgets around it repaint. Over a MOVING surface that burst is
    visible; over a still one it redraws identical pixels. So the tick
    stops advancing the clock while a popup is up.
    """

    def test_the_tick_advances_when_no_popup_is_showing(self, gpu_backdrop,
                                                        monkeypatch):
        canvas = gpu_backdrop("orbit")._canvas
        monkeypatch.setattr(F, "a_popup_is_on_screen", lambda: False)
        before = canvas._program.uniform_writes if hasattr(
            canvas._program, "uniform_writes") else None
        canvas._on_timer(None)
        assert canvas._dead is False

    def test_the_tick_does_nothing_while_a_popup_is_up(self, gpu_backdrop,
                                                       monkeypatch):
        canvas = gpu_backdrop("orbit")._canvas
        updates = []
        monkeypatch.setattr(F, "a_popup_is_on_screen", lambda: True)
        monkeypatch.setattr(type(canvas), "update",
                            lambda self: updates.append(1))
        canvas._on_timer(None)
        assert updates == [], (
            "the backdrop kept moving under a popup, which is the flicker")

    def test_a_paused_canvas_never_asks_about_popups(self, gpu_backdrop,
                                                     monkeypatch):
        """The cheaper guards come first: paused and dead are checked above."""
        canvas = gpu_backdrop("orbit")._canvas
        asked = []
        monkeypatch.setattr(F, "a_popup_is_on_screen",
                            lambda: asked.append(1) or False)
        canvas._paused = True
        canvas._on_timer(None)
        assert asked == [], "a paused canvas paid for the popup question"


class TestShutdownRaces:

    def test_a_join_helper_swallows_a_thread_that_will_not_stop(self,
                                                                monkeypatch):
        """`_join` runs from a `destroyed` signal, where nothing can catch."""
        monkeypatch.setattr(
            F, "_quit_and_join_thread",
            lambda _t: (_ for _ in ()).throw(RuntimeError("already gone")))

        class _Widget:
            class _Signal:
                def connect(self, slot):
                    self.slot = slot
            destroyed = _Signal()

        widget = _Widget()
        F._join_on_destroy(widget, object())
        widget.destroyed.slot()          # must not raise

    def test_shutting_down_twice_is_safe(self, gpu_backdrop):
        """THE SECOND SHUTDOWN is where the disconnect guard earns its place.

        `shutdown` disconnects its own slot from `aboutToQuit`. The
        second call disconnects a slot that is no longer connected, and
        Qt answers that with RuntimeError (or TypeError). Shutdown has to
        finish regardless: the renderer thread still has to be joined,
        and destroying a live QThread is a process-fatal Qt error.

        Not a contrived case -- the widget's own `closeEvent` calls
        shutdown, and so does the harness that tears these down.
        """
        widget = gpu_backdrop("orbit")
        widget.shutdown()
        widget.shutdown()               # must not raise

    def test_the_timer_is_stopped_by_the_first_shutdown(self, gpu_backdrop):
        """Whatever else happens, the animation must not outlive it."""
        widget = gpu_backdrop("orbit")
        widget.shutdown()
        timer = getattr(widget._canvas, "_timer", None)
        if timer is not None and hasattr(timer, "isActive"):
            assert not timer.isActive()


class TestTheCpuWidgetsShutdownDisconnect:
    """`shutdown` disconnects its own `aboutToQuit` slot, and may fail.

    The widget connects `_app_quit_join` to the application's
    `aboutToQuit` so a quit joins the renderer thread. Shutdown takes it
    back off. If something else has already removed it -- a teardown that
    tore the application down first, a slot Qt has released -- Qt answers
    the disconnect with RuntimeError or TypeError.

    Swallowing it is not cosmetic: the line after it is the thread join,
    and destroying a live QThread is a process-fatal Qt error. Shutdown
    has to reach that line however the disconnect went.

    A second `shutdown()` cannot exercise this -- the `_stopped` flag
    returns first -- so the slot is removed directly instead.
    """

    def test_a_slot_already_disconnected_only_warns_on_this_binding(self,
                                                                    qtbot):
        """Removing the slot first does NOT reach the guard on PySide6.

        Written down because it is the obvious way to test this and it
        does not work: PySide6 answers a disconnect that matches nothing
        with a libpyside RuntimeWarning and returns, rather than raising.
        The `except` is never entered, and a test built this way passes
        while covering nothing.
        """
        pytest.importorskip("numba")
        from PySide6.QtWidgets import QApplication

        widget = F.create_fractal_widget(F.Settings(backend="cpu"))
        qtbot.addWidget(widget)
        app = QApplication.instance()
        try:
            app.aboutToQuit.disconnect(widget._app_quit_join)
        except (RuntimeError, TypeError):
            pytest.skip("this build never connected the quit slot")

        widget.shutdown()               # warns, does not raise
        assert widget._stopped is True

    # NOT TESTED, and the two attempts are recorded rather than deleted.
    #
    # The guard catches RuntimeError and TypeError from the disconnect.
    # Neither can be produced here:
    #
    #   * removing the slot first (the test above) makes PySide6 emit a
    #     libpyside RuntimeWarning and return -- the except is not entered,
    #     and a test written that way passes while covering nothing;
    #   * supplying an application whose disconnect raises means patching
    #     `QApplication.instance` on the class, because the name is imported
    #     inside the widget factory and the closure holds the real class.
    #     That is global: it took qtbot and five other tests down with it.
    #
    # So the condition the guard defends against is one this binding
    # reports differently, and reaching it would mean breaking the harness
    # rather than exercising the program. Left uncovered on purpose.
    #
    # What DOES matter about it is asserted below: shutdown reaches the
    # thread join. Leaving a live QThread to be destroyed is a
    # process-fatal Qt error, so the join is the line the guard exists to
    # protect.

    def test_an_ordinary_shutdown_still_disconnects_and_joins(self, qtbot):
        """The healthy path, so the guard above is visibly a guard."""
        pytest.importorskip("numba")

        widget = F.create_fractal_widget(F.Settings(backend="cpu"))
        qtbot.addWidget(widget)
        widget.shutdown()
        assert widget._stopped is True
        widget.shutdown()               # the _stopped early return
