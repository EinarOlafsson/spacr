"""Popup stacking policy survives main-window clicks and popup reopening."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QWidget

from spacr.qt import dialogs
from spacr.qt.widgets import glass


@pytest.mark.parametrize("treatment", ["detached", "glassed", "filtered"])
def test_popup_keeps_transient_owner_without_global_top_hint(qtbot, treatment):
    main = QWidget()
    qtbot.addWidget(main)
    layout = QVBoxLayout(main)
    button = QPushButton("Main action", main)
    layout.addWidget(button)
    clicks = []
    button.clicked.connect(lambda: clicks.append(True))
    main.show()
    popup = QDialog(main)
    qtbot.addWidget(popup)
    popup.setWindowModality(Qt.NonModal)
    popup.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    if treatment == "detached":
        dialogs.detach_from_window_manager(popup)
    elif treatment == "glassed":
        glass.make_frameless(popup)
    else:
        dialogs.detach_all_dialogs(QApplication.instance())
    popup.show()
    qtbot.waitExposed(popup)
    native = popup.windowHandle()
    assert native is not None
    assert native.transientParent() is main.windowHandle()
    for _ in range(2):
        main.raise_()
        main.activateWindow()
        qtbot.mouseClick(button, Qt.LeftButton)
        assert popup.isVisible()
        assert popup.parentWidget() is main
        assert not popup.windowFlags() & Qt.WindowStaysOnTopHint
        assert not popup.windowHandle().flags() & Qt.WindowStaysOnTopHint
        assert popup.windowHandle().transientParent() is main.windowHandle()
        popup.hide()
        popup.show()
        qtbot.waitExposed(popup)
    assert len(clicks) == 2
    assert popup.windowModality() == Qt.NonModal


def test_top_popup_preserves_modal_verdict_and_dismissal(qtbot):
    main = QWidget()
    qtbot.addWidget(main)
    main.show()
    popup = QDialog(main)
    qtbot.addWidget(popup)
    dialogs.detach_from_window_manager(popup)
    popup.open()
    qtbot.waitExposed(popup)
    assert popup.windowModality() == Qt.WindowModal
    popup.accept()
    assert popup.result() == QDialog.Accepted
    assert not popup.windowFlags() & Qt.WindowStaysOnTopHint
    popup.show()
    qtbot.keyClick(popup, Qt.Key_Escape)
    assert not popup.isVisible()
    assert popup.result() == QDialog.Rejected
    assert not main.windowFlags() & Qt.WindowStaysOnTopHint
