"""A field frame on the Mask Generation screen costs about what it costs on Home.

The capture gate on the workstation (real application, GNOME, 2560x1440)
measured the field painting at about 57 fps on Home and 42-45 on Mask
Generation. A field frame is a synchronous repaint of the backdrop, and Qt
repaints every widget above it with it, so a frame's GUI cost is the
backdrop's blit plus every visible widget's paint event. The frame rate is
whatever that cost leaves of the 16.7 ms tick.

Measured offscreen at 2560x1440 on 2026-10-12: GUI-thread CPU per
synchronous field frame with the shading thread's frame already published,
cheapest of 15, median of four interleaved rounds, three runs, on a 32-thread
machine at load 50-65. Before: Home 38.9-42.3 ms, Mask 52.9-62.2 ms (1.36-1.52
times Home). After: Home 35.8-38.5 ms, Mask 40.4-50.8 ms (1.05-1.42).

What Mask had and Home did not was paint hooks on the settings form. Every
widget's paint event went through three Python event filters (the hub, the
field's own application filter, and the field fade, which the hub asked
about every paint): 2915 Python calls a Mask frame, 2056 a Home frame. The
fade drew two antialiased seventeen-stop gradient rounded rectangles under
each of the twelve fields, and it, the five switches and the splitter
handles each built a ``QSettings`` to resolve the theme, 21 store reads a
frame. Now the field hears the application only for the pointer, the fade
is installed on the fields and drawn from a cached pixmap, and the paint-time
theme is resolved once per preference application
(:func:`spacr.qt.theme._painting_theme`): 1101 and 691 Python calls a frame,
no store reads. What still separates the two screens is Qt painting the
stylesheet of the form itself: 3.6 megapixels of translucent rounded card
backgrounds on Mask against 1.0 on Home, and more styled controls.

Three assertions. The ratio is a ceiling against a regression, load-robust
because both screens are measured interleaved in one process, so a busy
machine slows both alike. The store reads and the filter calls are counts
and do not depend on load at all.
"""
from __future__ import annotations

import statistics
import time

import pytest

pytest.importorskip("PySide6")

#: Most a Mask field frame may cost, in Home field frames.
#: 1.05-1.42 measured after the fix, 1.36-1.52 before it.
MASK_OVER_HOME = 1.6

FRAMES = 15
ROUNDS = 4


def _hide_tours(window) -> None:
    for child in window.findChildren(object):
        if type(child).__name__ == "_TourOverlay":
            child.hide()


def _frame_ms(backdrop, frames=FRAMES) -> float:
    """Cheapest GUI-thread CPU of one synchronous field frame, in ms.

    The minimum, because on a machine with other work on it every other
    sample measures some of that work too.
    """
    spent = []
    for _frame in range(frames):
        started = time.thread_time()
        backdrop.repaint()
        spent.append((time.thread_time() - started) * 1000.0)
    return min(spent)


@pytest.fixture
def applied(qapp):
    """The preferences applied the way the application applies them.

    That is what installs the field fade, which ``qt_theme_applied`` alone
    does not; the application's look is put back afterwards.
    """
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.field_fade import (invalidate_field_fade,
                                             uninstall_field_fade)

    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    apply_preferences_to_app(qapp)
    yield qapp
    uninstall_field_fade(qapp)
    invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


@pytest.fixture
def big_window(qtbot, applied):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(2560, 1440)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    qtbot.waitUntil(lambda: backdrop.frames_painted > 3, timeout=20000)
    window._on_nav_selected("mask")
    qtbot.wait(300)
    window._on_nav_selected("__home__")
    qtbot.wait(300)
    yield window, backdrop
    backdrop._timer.stop()


def _show(window, key, qtbot) -> None:
    window._on_nav_selected(key)
    qtbot.wait(100)
    _hide_tours(window)
    qtbot.wait(20)


def test_a_mask_field_frame_costs_about_a_home_one(big_window, qtbot):
    window, backdrop = big_window
    backdrop._timer.stop()
    home, mask = [], []
    for _round in range(ROUNDS):
        _show(window, "__home__", qtbot)
        home.append(_frame_ms(backdrop))
        _show(window, "mask", qtbot)
        mask.append(_frame_ms(backdrop))
    ratio = statistics.median(m / h for m, h in zip(mask, home))
    assert ratio <= MASK_OVER_HOME, (
        f"a Mask field frame costs {ratio:.2f} Home frames "
        f"(Home {statistics.median(home):.1f} ms, "
        f"Mask {statistics.median(mask):.1f} ms per frame at 2560x1440)")


def test_a_mask_field_frame_reads_no_preference_store(big_window, qtbot,
                                                     monkeypatch):
    from spacr.qt import preferences

    window, backdrop = big_window
    _show(window, "mask", qtbot)
    backdrop._timer.stop()
    backdrop.repaint()
    reads = []
    real = preferences._settings

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(preferences, "_settings", counting)
    for _frame in range(5):
        backdrop.repaint()
    assert reads == [], f"{len(reads) / 5:.1f} preference reads per field frame"


def test_no_paint_hook_is_asked_about_every_widget(big_window, qtbot):
    """The field and the fade hear their own widgets, not the whole frame.

    Every widget above the field is sent a paint event on every frame,
    180 of them on Mask. Before 2026-10-12 three Python filters were asked
    about each one (the hub, the field's own application filter and the
    fade, which the hub asked about every paint): 2915 Python calls per
    Mask frame and 2056 per Home frame. Now the field hears the application
    only for the pointer, through the hub, and the fade is installed on the
    fields themselves: 1101 and 691.
    """
    import sys

    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.field_fade import FIELD_TYPES

    window, backdrop = big_window
    _show(window, "mask", qtbot)
    backdrop._timer.stop()
    backdrop.repaint()
    asked = {"ambient.py": 0, "field_fade.py": 0}

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code.co_name == "eventFilter":
            name = frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
            if name in asked:
                asked[name] += 1

    frames = 4
    sys.setprofile(profile)
    try:
        for _frame in range(frames):
            backdrop.repaint()
    finally:
        sys.setprofile(None)
    fields = sum(1 for widget in window.findChildren(QWidget)
                 if isinstance(widget, FIELD_TYPES) and widget.isVisible())
    assert asked["field_fade.py"] / frames <= fields, asked
    assert asked["ambient.py"] / frames <= 5, asked
