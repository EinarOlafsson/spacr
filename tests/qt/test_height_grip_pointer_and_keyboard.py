"""A panel resize must follow global pointer motion and preserve keyboard access."""
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from spacr.qt.widgets.height_grip import HeightGrip


@pytest.fixture
def grip(qtbot, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_font_scale", lambda: 1.0)
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    target = QWidget(window)
    layout.addWidget(target)
    handle = HeightGrip(target, 100, 400, window)
    layout.addWidget(handle)
    handle.resize_target(200)
    window.show()
    yield handle


def _mouse(handle, event_type, global_y, button, buttons):
    event = QMouseEvent(event_type, QPointF(10, 4), QPointF(50, global_y),
                        button, buttons, Qt.NoModifier)
    QApplication.sendEvent(handle, event)


def test_drag_uses_global_displacement_and_emits_only_on_release(grip):
    changed = []
    grip.height_changed.connect(changed.append)
    _mouse(grip, QEvent.MouseButtonPress, 300, Qt.LeftButton, Qt.LeftButton)
    _mouse(grip, QEvent.MouseMove, 365, Qt.NoButton, Qt.LeftButton)
    assert grip.target_height() == 265
    # Local pointer position is unchanged: the grip moves with the panel.
    _mouse(grip, QEvent.MouseMove, 325, Qt.NoButton, Qt.LeftButton)
    assert grip.target_height() == 225
    assert changed == []
    _mouse(grip, QEvent.MouseButtonRelease, 325, Qt.LeftButton, Qt.NoButton)
    assert changed == [225]
    _mouse(grip, QEvent.MouseMove, 390, Qt.NoButton, Qt.NoButton)
    assert grip.target_height() == 225


def test_unpressed_or_right_button_motion_never_resizes(grip):
    changed = []
    grip.height_changed.connect(changed.append)
    _mouse(grip, QEvent.MouseMove, 800, Qt.NoButton, Qt.NoButton)
    _mouse(grip, QEvent.MouseButtonPress, 300, Qt.RightButton, Qt.RightButton)
    _mouse(grip, QEvent.MouseMove, 800, Qt.NoButton, Qt.RightButton)
    _mouse(grip, QEvent.MouseButtonRelease, 800, Qt.RightButton, Qt.NoButton)
    assert grip.target_height() == 200
    assert changed == []


@pytest.mark.parametrize("key, expected", [
    (Qt.Key_Down, 212), (Qt.Key_Plus, 212), (Qt.Key_Equal, 212),
    (Qt.Key_Up, 188), (Qt.Key_Minus, 188),
    (Qt.Key_PageDown, 260), (Qt.Key_PageUp, 140),
    (Qt.Key_End, 400), (Qt.Key_Home, 100),
])
def test_keyboard_resize_has_the_documented_step_and_bounds(grip, key, expected):
    changed = []
    grip.height_changed.connect(changed.append)
    QApplication.sendEvent(grip, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    assert grip.target_height() == expected
    assert changed == [expected]


def test_unhandled_key_is_not_consumed(grip):
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_X, Qt.NoModifier)
    grip.keyPressEvent(event)
    assert not event.isAccepted()
    assert grip.target_height() == 200


def test_context_menu_reset_restores_the_original_height(grip):
    grip.resize_target(350)
    opened = []
    changed = []
    grip.height_changed.connect(changed.append)

    def choose_reset():
        menu = QApplication.activePopupWidget()
        opened.append([action.text() for action in menu.actions()])
        menu.actions()[0].trigger()
        menu.close()

    timer = QTimer(grip)
    timer.setSingleShot(True)
    timer.timeout.connect(choose_reset)
    timer.start(50)
    try:
        grip.customContextMenuRequested.emit(grip.rect().center())
    finally:
        timer.stop()
    assert opened == [["Reset height"]]
    assert grip.target_height() == 200
    assert changed == [200]
