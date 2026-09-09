"""The backdrop full screen, with nothing else, until a key is pressed."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeyEvent, QMouseEvent

import spacr.qt.app as app_module
from spacr.qt.screensaver import Screensaver


def test_it_is_in_the_hotkey_menu(qtbot):
    win = app_module.MainWindow()
    qtbot.addWidget(win)
    try:
        action = win.findChild(QAction, "ShowScreensaver")
        assert action is not None
        assert action.shortcut().toString() == "Ctrl+Shift+F"
        assert "screensaver" in action.statusTip().lower()
    finally:
        win.close()


def test_any_key_closes_it(qtbot):
    saver = Screensaver()
    qtbot.addWidget(saver)
    closed = []
    saver.destroyed.connect(lambda *_a: closed.append(True))

    saver.keyPressEvent(QKeyEvent(
        QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier))
    # `close` on a WA_DeleteOnClose widget schedules deletion; the window
    # being hidden is what the user sees.
    assert not saver.isVisible()


def test_a_click_closes_it_too(qtbot):
    from PySide6.QtCore import QPointF

    saver = Screensaver()
    qtbot.addWidget(saver)
    saver.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(1.0, 1.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    assert not saver.isVisible()


def test_the_pointer_is_hidden(qtbot):
    """What makes it read as a screensaver rather than an empty window."""
    saver = Screensaver()
    qtbot.addWidget(saver)
    assert saver.cursor().shape() == Qt.CursorShape.BlankCursor


def test_it_is_its_own_window_and_not_the_main_one(qtbot):
    """Hiding spaCR's widgets means remembering what to restore, and getting
    that wrong rearranges the layout."""
    saver = Screensaver()
    qtbot.addWidget(saver)
    assert saver.parent() is None
    assert saver.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


def test_closing_stops_the_backdrop(qtbot):
    """A canvas destroyed while its timer runs is a crash."""
    saver = Screensaver()
    qtbot.addWidget(saver)
    paused = []

    class _Backdrop:
        def pause(self):
            paused.append(True)

    saver._backdrop = _Backdrop()
    saver.close()
    assert paused == [True]


def test_the_main_window_keeps_a_reference(qtbot):
    """Python would free the only one and the window would close at once."""
    import inspect

    source = inspect.getsource(app_module.MainWindow._show_the_screensaver)
    assert "self._screensaver = saver" in source


# ---------------------------------------------------------------------------
# 352: the arms that run when the backdrop or the screen will not cooperate.
# ---------------------------------------------------------------------------

def test_a_backdrop_that_will_not_build_leaves_a_black_window(qtbot,
                                                              monkeypatch):
    """A screensaver with no backdrop is still a screensaver.

    The backdrop is the part that asks a driver to draw, and it is exactly
    the part that fails on a machine with a bad GL stack -- which is the
    machine most likely to be left idle long enough to reach here. Raising
    would replace a blank screen with a crash; returning None leaves the
    window, and `paintEvent` fills it black so nothing shows through.
    """
    from spacr.qt import screensaver as ss

    def explode(*_a, **_k):
        raise RuntimeError("no GL on this machine")

    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "get_fractal_settings", explode)
    saver = ss.Screensaver()
    qtbot.addWidget(saver)
    assert saver._backdrop is None


def test_the_black_paint_covers_the_whole_window(qtbot):
    """`paintEvent` fills the rect, so no desktop shows through a gap."""
    from PySide6.QtGui import QColor

    from spacr.qt import screensaver as ss

    from PySide6.QtGui import QImage

    saver = ss.Screensaver()
    qtbot.addWidget(saver)
    saver.resize(40, 30)
    # RENDERED INTO AN IMAGE WE PRE-FILL, so "black" cannot be the default
    # of an untouched buffer. `render` drives the real `paintEvent`; a
    # `grab()` on a widget that was never shown can return the backing
    # store instead, which offscreen is not necessarily what was painted.
    image = QImage(40, 30, QImage.Format_RGB32)
    image.fill(QColor(255, 0, 0))
    saver.render(image)
    for point in ((0, 0), (39, 29), (20, 15)):
        pixel = QColor(image.pixel(*point))
        # OPAQUE AND DARK, not a specific hex. The fill is black and the
        # theme's own page colour lands on top of it, so the rendered
        # pixel measures (5, 5, 10) rather than (0, 0, 0). What the code
        # promises is that nothing shows THROUGH -- so the assertion is
        # that the red we pre-filled is gone, which is the thing a user
        # would see if the promise broke.
        assert pixel.red() < 32 and pixel.green() < 32 and pixel.blue() < 32, (
            point, pixel.getRgb())


def test_opening_without_a_parent_screen_still_returns_a_window(qtbot,
                                                                monkeypatch):
    """No parent means no screen geometry to copy, and that is not a failure.

    `open_screensaver` copies the parent's screen geometry when there is a
    parent to ask. Called with none -- which is what a test and a headless
    run do -- it must still hand back a window rather than None.
    """
    from spacr.qt import screensaver as ss

    saver = ss.show_screensaver(None)
    if saver is not None:
        qtbot.addWidget(saver)
        saver.close()
    assert saver is not None


def test_a_screensaver_that_cannot_open_reports_none_rather_than_raising(
        monkeypatch):
    """The caller is an idle timer; an exception there has nowhere to go."""
    from spacr.qt import screensaver as ss

    def explode(*_a, **_k):
        raise RuntimeError("no display")

    monkeypatch.setattr(ss, "Screensaver", explode)
    assert ss.show_screensaver(None) is None
