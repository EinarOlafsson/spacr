"""Dialog preparation preserves deliberate positions and survives no display owner."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget

from spacr.qt import dialogs, hidpi

pytestmark = pytest.mark.qt


def test_active_main_owns_a_moved_dialog_without_changing_its_geometry(qtbot, qapp):
    first, active = QMainWindow(), QMainWindow()
    for window in (first, active):
        qtbot.addWidget(window)
        window.resize(480, 320)
        window.show()
        qtbot.waitExposed(window)
    qapp.setActiveWindow(active)
    assert qapp.activeWindow() is active
    hidden = QWidget()
    qtbot.addWidget(hidden)
    dialog = QDialog(hidden)
    qtbot.addWidget(dialog)
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setGeometry(83, 97, 310, 230)
    geometry = dialog.geometry()

    assert dialogs._prepare_dialog_owner(dialog) is active
    assert dialog.parentWidget() is active
    assert dialog.geometry() == geometry
    assert dialog.windowModality() == Qt.ApplicationModal
    assert not dialog.isVisible()
    assert dialogs._prepare_dialog_owner(dialog) is active
    assert dialog.geometry() == geometry


def test_absent_application_leaves_an_unowned_dialog_unchanged(qtbot, monkeypatch):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.setGeometry(61, 79, 250, 180)
    geometry = dialog.geometry()
    monkeypatch.setattr(QApplication, 'instance', staticmethod(lambda: None))

    assert dialogs._visible_dialog_owner(dialog) is None
    assert dialogs._prepare_dialog_owner(dialog) is None
    assert dialog.parentWidget() is None
    assert dialog.geometry() == geometry


def test_unavailable_screen_does_not_move_or_later_jump_an_unshown_dialog(qtbot, monkeypatch):
    owner = QWidget()
    qtbot.addWidget(owner)
    owner.show()
    qtbot.waitExposed(owner)
    dialog = QDialog(owner)
    qtbot.addWidget(dialog)
    dialog.resize(270, 190)
    geometry = dialog.geometry()
    assert dialog.windowHandle() is None
    assert not dialog.testAttribute(Qt.WA_Moved)
    screen = hidpi.screen_for_widget(owner)
    assert screen is not None
    monkeypatch.setattr(hidpi, 'screen_for_widget', lambda widget: None)

    assert dialogs._place_dialog_on_first_show(dialog) is False
    assert dialog.geometry() == geometry
    assert dialog.property('spacrInitiallyPlaced') is True
    monkeypatch.setattr(hidpi, 'screen_for_widget', lambda widget: screen)
    assert dialogs._place_dialog_on_first_show(dialog) is False
    assert dialog.geometry() == geometry
