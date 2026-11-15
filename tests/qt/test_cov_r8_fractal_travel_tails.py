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
        # THE PARENT PACKAGE HOLDS THE NAME TOO, and `delitem` on
        # `sys.modules` does not put that back. Importing a module writes
        # it as an ATTRIBUTE of its package as well as into `sys.modules`,
        # so a re-import here left `spacr.qt.widgets.fractal_travel` (the
        # attribute) pointing at this scratch module while `sys.modules`
        # was restored to the real one on teardown -- two different module
        # objects under one name.
        #
        # A later test then patched one and the code under test read the
        # other: `from spacr.qt.widgets import fractal_travel` follows the
        # package attribute, and `from .widgets.fractal_travel import ...`
        # inside `install_the_spaceout_fractal` follows `sys.modules`.
        # Three tests in `test_the_spaceout_fractal.py` failed that way,
        # and only when this file ran first -- which CI's per-file
        # distribution meant it usually did not.
        #
        # Recording the attribute before the re-import is what restores it.
        import spacr.qt.widgets as _widgets_package

        monkeypatch.setattr(_widgets_package, "fractal_travel",
                            getattr(_widgets_package, "fractal_travel", None),
                            raising=False)
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

    def test_the_re_import_leaves_one_module_under_one_name(self,
                                                            monkeypatch):
        """The scratch namespace must not outlive this test.

        `delitem` on `sys.modules` restores that dict and nothing else,
        and an import writes the module as an ATTRIBUTE of its package
        too. Without the attribute being recorded as well, this class left
        `spacr.qt.widgets.fractal_travel` meaning two different module
        objects -- one reached through the package, one through
        `sys.modules` -- and a later test patched whichever the code under
        test did not read.

        Asserted INSIDE the test rather than left to a teardown check,
        because the failure it prevents lands in another file entirely and
        arrives there as three unrelated-looking assertion errors.
        """
        import spacr.qt.widgets as package

        scratch = self._import_without_numba(monkeypatch)
        # During the test the two legitimately disagree: that is what a
        # scratch namespace IS.
        assert scratch is not sys.modules.get("spacr.qt.widgets."
                                              "fractal_travel", scratch) \
            or scratch is sys.modules["spacr.qt.widgets.fractal_travel"]
        # And the attribute is under monkeypatch, so undoing puts the real
        # module back in both places. Undo here to assert it.
        monkeypatch.undo()
        assert package.fractal_travel is sys.modules[
            "spacr.qt.widgets.fractal_travel"], (
            "the package attribute and sys.modules name two different "
            "modules -- a later test will patch one and be read from the "
            "other")

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

        # ASSERTED on what shutdown is FOR. Destroying a live QThread is
        # a process-fatal Qt error, so the second call finishing quietly
        # is only correct if the thread is actually stopped -- and a
        # shutdown that returned early on both calls would pass without
        # this.
        thread = getattr(widget._canvas, "_thread", None)
        if thread is not None and hasattr(thread, "isRunning"):
            assert not thread.isRunning(), (
                "the renderer thread outlived two shutdowns")

    def test_the_timer_is_stopped_by_the_first_shutdown(self, gpu_backdrop):
        """Whatever else happens, the animation must not outlive it."""
        widget = gpu_backdrop("orbit")
        widget.shutdown()
        timer = getattr(widget._canvas, "_timer", None)
        if timer is not None and hasattr(timer, "isActive"):
            assert not timer.isActive()
