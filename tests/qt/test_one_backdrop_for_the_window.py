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


def test_home_still_has_a_backdrop_the_user_can_see(qtbot, qt_theme_applied):
    """THE DEDUP IS OFF, AND THIS IS WHAT REPLACED THE ASSERTION.

    This test used to demand exactly ONE visible backdrop, and it passed
    while the maintainer's home screen was black. Both halves of that are
    the point.

    The dedup retired the screen's own backdrop and left the window's,
    which sits behind the stack -- and HomePage's plain ``QWidget``
    containers paint ``bg`` over it, which on the dark theme is
    ``#000000``. Counting widgets could not see that, because the widget
    that was counted was the one being covered up.

    THE EXACT-ONE ASSERTION IS BACK, 2026-09-11, AND SO IS THE
    MEASUREMENT IT WAS MADE CONDITIONAL ON -- that was the condition this
    docstring set and it has been met, not waived. The containers stopped
    painting an opaque `bg` over the window's backdrop
    (`theme._window_block` takes the transparent shape when an animated
    backdrop is running), the dedup is on again, and
    `test_the_home_screen_is_not_black_on_a_real_display` below reads the
    pixels from X. Counting alone shipped a black window; counting beside a
    picture does not.
    """
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)

    if window.window_backdrop() is None:
        pytest.skip("the ambient backdrop is off in this configuration")
    visible = _visible_backdrops(window)
    assert visible, "the home screen has no visible animated backdrop at all"
    assert len(visible) == 1, (
        f"{len(visible)} animated backdrops are visible on Home; two "
        f"full-size fields shaded and blitted per frame was 950 paints/s "
        f"and 393 Mpx/s for a picture nobody could see")


def test_opening_a_module_keeps_a_backdrop_too(qtbot, qt_theme_applied):
    """The same, for a module screen, and for the same reason.

    This one asserted that opening a module added no second backdrop. It
    was true, it was measured, and it is what left the module pages
    without a visible one.
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
    assert _visible_backdrops(window), (
        "the module screen has no visible animated backdrop at all")


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


def test_the_screen_stops_deferring_once_the_window_backdrop_goes(
        qtbot, qt_theme_applied):
    """And it must start painting again the moment there is nothing to defer to.

    The test above pins one half of the rule and, on its own, invites the
    other half to be got wrong -- which is what happened.
    ``_uses_window_backdrop`` was set and never cleared, so a screen that
    had once shared the window's animation went on returning None from
    ``page_fill`` after the animation was switched off in Preferences.
    With nothing behind it and nothing painted by it, the page fell back
    to the flat ``surface`` slab: the black page, reported three times and
    then reintroduced by the fix for it.

    The flag is a claim about the window as it stands NOW, so
    ``refresh_theme`` -- which is what Preferences calls -- reconciles it
    on every cached screen, not only on the one in front of the user.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow
    from PySide6.QtWidgets import QApplication

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
    assert screen.page_fill() is None, "precondition: the window is animating"

    # RESTORED IN `finally`, AND THE FIRST VERSION OF THIS TEST WAS NOT.
    # `set_ambient_enabled` writes to the REAL QSettings -- the developer's
    # own, not a tmp_path -- so leaving it False turned the animation off in
    # the running application and failed twelve preference tests in every
    # later run. A test that changes a persisted preference owns putting it
    # back, whatever it asserts in between.
    was_enabled = prefs.get_ambient_enabled()
    try:
        prefs.set_ambient_enabled(False)
        prefs.apply_preferences_to_app(QApplication.instance())
        window.refresh_theme()
        qtbot.wait(50)

        assert window.window_backdrop() is None, (
            "precondition: turning ambient off must retire the window's "
            "backdrop")
        assert getattr(screen, "_uses_window_backdrop", False) is False, (
            "the screen still claims a window backdrop that is gone")
        assert screen.page_fill() is not None, (
            "with no animation behind it the screen must paint the page "
            "colour; returning None leaves the flat `surface` slab -- the "
            "black page")
    finally:
        prefs.set_ambient_enabled(was_enabled)
        prefs.apply_preferences_to_app(QApplication.instance())


@pytest.mark.gui
def test_the_home_screen_is_not_black_on_a_real_display(qtbot):
    """THE MEASUREMENT THE EXACT-ONE ASSERTION ABOVE IS CONDITIONAL ON.

    MARKED `gui` BECAUSE IT CANNOT BE FAKED. The offscreen platform does
    not composite, and `QWidget.grab()` cannot capture the GL-backed
    `AmbientWidget` at all -- it renders a WORKING backdrop as black, which
    is how a green suite shipped a black window in the first place. The
    only honest reading is `QScreen.grabWindow` on a raised, settled window
    under a real display, and the suite runs `-m "not gui"`, so this costs
    nothing where it could only lie.

    THE THRESHOLDS COME FROM MEASUREMENTS, not from taste. Home screen,
    dark theme, the maintainer's display, five runs with this fix in:

        chromatic  46.9  51.9  52.6  64.1  65.2 %
        pure black  0.0   0.0   0.0   0.0   0.0 %

    and the same configuration with the window block left opaque, which is
    the regression: 3.0 % chromatic and 20.4 % pure black. The backdrop is
    ANIMATED, so the chromatic figure moves with whichever frame the grab
    catches -- hence a floor at 25 %, well under the lowest reading and
    well over the broken one, rather than a number near either.
    """
    from PySide6.QtCore import QElapsedTimer, Qt
    from PySide6.QtWidgets import QApplication

    from spacr.qt import preferences
    from spacr.qt.app import MainWindow

    if not preferences.get_ambient_enabled():
        pytest.skip("the ambient backdrop is off in this configuration")

    app = QApplication.instance()
    # THROUGH THE REAL PATH, or this measures an unthemed window. The
    # stylesheet is what makes the containers transparent over the
    # backdrop, and without it parts of the window paint their default
    # black -- 5.4 % of the frame, measured, which is a fact about the
    # test rather than about the product.
    preferences.apply_preferences_to_app(app)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    window.show()
    window.raise_()
    window.activateWindow()
    # NOT `processEvents()` IN A COUNTED LOOP. Forty calls take
    # microseconds; a compositing window manager needs milliseconds to map
    # and composite, and a grab fired before the window is on screen
    # returns whatever is stacked above it -- which on a dark desktop is a
    # convincing near-black that looks exactly like the bug this asserts
    # against. That mistake cost a session's worth of wrong conclusions.
    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < 2500:
        app.processEvents()

    image = window.screen().grabWindow(window.winId()).toImage()
    assert not image.isNull() and image.width() > 100, (
        "the grab came back empty; run "
        "tools/can_this_display_be_measured.py first")

    chromatic = black = total = 0
    for y in range(0, image.height(), 3):
        for x in range(0, image.width(), 3):
            colour = image.pixelColor(x, y)
            red, green, blue = colour.red(), colour.green(), colour.blue()
            total += 1
            if red == green == blue == 0:
                black += 1
            if max(red, green, blue) - min(red, green, blue) >= 12:
                chromatic += 1

    black_share = 100.0 * black / total
    chromatic_share = 100.0 * chromatic / total
    assert black_share < 5.0, (
        f"{black_share:.1f}% of the home screen is pure black; the "
        f"regression this guards measured 20.4%")
    assert chromatic_share > 25.0, (
        f"only {chromatic_share:.1f}% of the home screen is coloured; the "
        f"regression this guards measured 3.0% and the fix measures 47-65%")
