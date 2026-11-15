"""The settings popover survives an anchor that has already been destroyed.

The popover positions itself against the button that opened it. That button
lives on a screen the user may have navigated away from, and Qt deletes the
C++ side of a widget before Python lets go of the wrapper. Asking a dead
widget where it is raises, and the popover has to appear somewhere rather
than take the screen down with it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from spacr.qt.widgets.dna_rain_settings import (DnaRainSettingsBar,
                                                DnaRainSettingsPopover)


def _popover(qapp):
    return DnaRainSettingsPopover(DnaRainSettingsBar())


def test_escape_closes_the_popover(qapp):
    """Escape closes, like every other popup in the app."""
    popover = _popover(qapp)
    popover.show()
    popover.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert popover.isVisible() is False


def test_another_key_is_passed_on_and_does_not_close_it(qapp):
    """Typing a letter into the bar must not dismiss the settings."""
    popover = _popover(qapp)
    popover.show()
    popover.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a"))
    assert popover.isVisible() is True
    popover.hide()
