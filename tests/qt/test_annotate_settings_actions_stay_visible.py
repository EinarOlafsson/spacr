"""Item 350 / GitHub #128: a small settings window keeps its actions usable."""

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QScrollArea

from spacr.qt.annotate_engine import AnnotateSettings
from spacr.qt.dialogs import make_the_window_resizable
from spacr.qt.screens.annotate import _SettingsDialog


@pytest.mark.parametrize("size", [(640, 480), (480, 320)])
@pytest.mark.parametrize("font_factor", [1.0, 1.5])
def test_settings_actions_remain_clickable_while_the_form_scrolls(
        qtbot, qt_theme_applied, size, font_factor):
    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    font = QFont(dialog.font())
    font.setPointSizeF(font.pointSizeF() * font_factor)
    dialog.setFont(font)
    make_the_window_resizable(dialog)
    dialog.show()
    dialog.resize(*size)
    qtbot.wait(80)

    assert dialog.height() <= size[1]
    scroll = dialog.findChildren(QScrollArea)[0]
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0, "the small window must expose the whole form by scrolling"
    buttons = dialog.findChild(QDialogButtonBox)
    for position in (bar.minimum(), bar.maximum() // 2, bar.maximum()):
        bar.setValue(position)
        qtbot.wait(20)
        for standard in (QDialogButtonBox.Ok, QDialogButtonBox.Cancel):
            button = buttons.button(standard)
            assert button.visibleRegion().contains(button.rect()), (
                standard, position, button.geometry())
            rect = button.rect().translated(button.mapTo(dialog, QPoint(0, 0)))
            assert dialog.rect().contains(rect)
    assert scroll.viewport().rect().contains(
        dialog._queue_limit.mapTo(scroll.viewport(), dialog._queue_limit.rect().center()))
    buttons.button(QDialogButtonBox.Cancel).click()
    assert dialog.result() == QDialog.Rejected


def test_settings_open_inside_the_available_screen(qtbot, qt_theme_applied):
    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    make_the_window_resizable(dialog)
    dialog.show()
    qtbot.wait(80)
    from spacr.qt.hidpi import screen_for_widget
    available = screen_for_widget(dialog).availableGeometry()
    assert dialog.width() <= available.width()
    assert dialog.height() <= available.height()
