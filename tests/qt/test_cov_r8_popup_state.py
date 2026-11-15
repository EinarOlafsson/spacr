"""`popup_state` — what the backdrop asks before it moves.

The flicker this exists to stop is a real one: a menu or a tooltip
composited over the native GL backdrop made the widgets around it
repaint, and the user saw the dock and the header flicker. Holding the
animation still while a popup is up takes a menu from 238 painted
widgets to 5 and a tooltip from 90 to 0.

So these tests are about the two things that can silently undo that: the
tooltip half being forgotten (a tooltip is NOT an activePopupWidget, and
watching only menus would have left half the bug in place), and the
helper being allowed to raise into an animation tick.
"""
from __future__ import annotations



class TestTheAnimationTicksActuallyAskIt:
    """The helper is only useful if the ticks consult it."""

    def test_the_gpu_fractal_tick_consults_it(self):
        import inspect

        from spacr.qt.widgets import fractal_travel

        src = inspect.getsource(fractal_travel)
        assert "a_popup_is_on_screen()" in src, (
            "the GPU backdrop stopped asking whether a popup is up")

    def test_the_ambient_engine_tick_deliberately_does_not(self):
        """AND THE AMBIENT TICK DELIBERATELY DOES NOT ASK (385).

        This used to assert the opposite. The hold was borrowed from the
        GL fix, and the GL fix's own diagnosis names its subject: "a popup
        composited over the NATIVE GL backdrop". The ambient widget has no
        GL surface and no native window, and the burst the hold was
        counting turned out not to be the popup's -- offscreen, widget
        repaints over 1.2 s were 1,590 with no popup at all and 1,592 with
        a menu up, against 2 with the animation held. Holding for the popup
        stopped the animation, which stopped a burst that was there the
        whole time.

        What it cost was the whole of 385: the theme froze whenever
        Preferences or the Help menu was open, and Preferences is where
        the theme's own controls live.
        """
        import inspect

        from spacr.qt.widgets import ambient

        tick = inspect.getsource(ambient.AmbientWidget._on_tick)
        assert "a_popup_is_on_screen()" not in tick, (
            "the ambient tick holds still under a popup again -- 385 says "
            "the theme must keep moving while Preferences or Help is open")
