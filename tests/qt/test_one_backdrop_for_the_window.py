"""There is ONE animated backdrop in the window, not one per screen.

WHAT THIS IS ABOUT. ``MainWindow._backdrop_the_dock_column`` puts a single
live backdrop on the central widget, behind the dock slot and the stack both,
and its docstring states the rule with the symptom that produced it: two
animations running out of step across a seam. Both screen classes were
supposed to see that backdrop and decline to build their own.

The guard could never fire. It asked ``self.window()`` from the screen's own
``__init__``, where the screen has no parent -- and a widget with no parent IS
its own window, so it asked the SCREEN whether it had a ``window_backdrop``
and got None every time. HomePage had no guard at all.

Measured on the maintainer's 3840x2160 screen at font_scale 2 before the fix:
two visible backdrops on Home (3840x2114 with 3400x2114 laid over it) and two
on a module screen, each shading and blitting a full-size field at 12.5 fps,
the lower one covered over 87 % of its area -- 950 paints/s and 393 Mpx/s with
nobody touching the machine. Idle GUI-thread CPU fell from ~16 % of a core to
~10 % once the second one went.

The check cannot live in the screen, so these tests assert the property from
the window's side, which is the only place the question has an answer.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _visible_backdrops(window):
    """Every animated backdrop actually on screen in ``window``."""
    from spacr.qt.widgets.ambient import AmbientWidget

    return [w for w in window.findChildren(AmbientWidget) if w.isVisible()]


def test_home_shows_exactly_one_backdrop(qtbot, qt_theme_applied):
    """Home must not lay a second animated field over the window's."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)

    if window.window_backdrop() is None:
        pytest.skip("the ambient backdrop is off in this configuration")
    assert len(_visible_backdrops(window)) == 1


def test_opening_a_module_does_not_add_a_second_backdrop(qtbot,
                                                         qt_theme_applied):
    """A module screen joins the window's backdrop rather than bringing one."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)

    if window.window_backdrop() is None:
        pytest.skip("the ambient backdrop is off in this configuration")

    window._on_nav_selected("mask")
    qtbot.wait(50)
    assert len(_visible_backdrops(window)) == 1


def test_a_screen_that_gave_up_its_backdrop_does_not_paint_over_the_window(
        qtbot, qt_theme_applied):
    """``page_fill`` must stay None, or the screen paints out the animation.

    ``page_fill`` returns a flat colour whenever the screen's own ``_ambient``
    is None. A screen that merely LOST its backdrop would therefore paint that
    colour straight over the window's -- the black slab reported three times.
    """
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)

    if window.window_backdrop() is None:
        pytest.skip("the ambient backdrop is off in this configuration")

    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens["mask"]
    assert screen.page_fill() is None, (
        "the screen would paint a flat page over the window's animation"
    )
