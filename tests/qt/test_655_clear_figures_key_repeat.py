"""Holding a clear key must not repeatedly discard arriving figures."""
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from spacr.qt.widgets.figure_queue import _ClearFiguresLabel


def test_held_space_activates_clear_once_like_a_native_button(qtbot, activate_focus_window):
    host = QWidget()
    layout = QVBoxLayout(host)
    native = QPushButton("Native action")
    clear = _ClearFiguresLabel(host)
    for control in (native, clear):
        layout.addWidget(control)
    qtbot.addWidget(host)
    host.show()
    activate_focus_window(host)
    counts = [[], []]
    native.clicked.connect(lambda: counts[0].append(True))
    clear.clicked.connect(lambda: counts[1].append(True))
    for control in (native, clear):
        control.setFocus(Qt.TabFocusReason)
        qtbot.waitUntil(control.hasFocus)
        QApplication.sendEvent(control, QKeyEvent(QEvent.KeyPress, Qt.Key_Space,
                                                 Qt.NoModifier, " ", False))
        for _ in range(2):
            QApplication.sendEvent(control, QKeyEvent(QEvent.KeyPress, Qt.Key_Space,
                                                     Qt.NoModifier, " ", True))
        QApplication.sendEvent(control, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space,
                                                 Qt.NoModifier, " ", False))
    assert counts[0] == [True]
    assert counts[1] == counts[0]
    assert clear._flash.active
    assert clear.hasFocus()
