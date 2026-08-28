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
