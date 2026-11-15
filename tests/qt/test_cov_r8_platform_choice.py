"""Which Qt platform plugin spaCR asks for, and when it declines to.

WHY THIS IS LOAD-BEARING. On a native Wayland session Qt hands vispy an
OpenGL ES context; vispy compiles the fractal shaders as desktop GLSL 120
and ES answers "unsupported version 120", so the backdrop never draws a
frame. The same compositor rounds the opaque surface, which is where the
black corners outside the rounded edges came from. Under `xcb` both are
gone, confirmed by running all three entry points.

The failure mode these tests exist to catch is not "it forgets to set
xcb" -- that is loud. It is the opposite: OVERRULING a choice somebody
made deliberately. The test suite sets QT_QPA_PLATFORM=offscreen, and a
helper that stamped xcb over it would take the whole suite off a
headless machine.
"""
from __future__ import annotations


class TestEveryEntryPointAsks:
    """A fix that reaches one command and not the others is not a fix.

    This machine has three separate spaCR installs, and an evening was
    lost to a platform preference that lived in `spaceout` while the user
    was running `spacr`. The helper is in `spacr.qt` for that reason, and
    these assert that each way in actually calls it.
    """

    def test_run_asks(self):
        import inspect

        from spacr.qt import run

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(run)

    def test_run_without_setup_asks(self):
        import inspect

        from spacr.qt import run_without_setup

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(run_without_setup)

    def test_spaceout_asks(self):
        import inspect

        from spacr.qt.spaceout import main

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(main)

    def test_safespacr_inherits_it_by_going_through_run(self):
        """Safe mode has no platform logic of its own, and needs none."""
        import inspect

        from spacr.qt.safespacr import main

        assert "run(" in inspect.getsource(main)
