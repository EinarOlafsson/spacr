"""The settings popover survives an anchor that has already been destroyed.

The popover positions itself against the button that opened it. That button
lives on a screen the user may have navigated away from, and Qt deletes the
C++ side of a widget before Python lets go of the wrapper. Asking a dead
widget where it is raises, and the popover has to appear somewhere rather
than take the screen down with it.
"""
from __future__ import annotations

import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets.dna_rain_settings import (DnaRainSettingsBar,
                                                DnaRainSettingsPopover)


def _popover(qapp):
    return DnaRainSettingsPopover(DnaRainSettingsBar())


def test_a_destroyed_anchor_is_treated_as_an_anchor_at_the_origin(qapp):
    """Losing the button must not lose the settings it opened."""
    popover = _popover(qapp)
    popover.adjustSize()

    at_origin = QPushButton("settings")
    at_origin.setFixedSize(0, 1)
    at_origin.move(0, 0)
    popover._position_near(at_origin)
    expected = popover.pos()

    dead = QPushButton("settings")
    shiboken6.delete(dead)
    popover._position_near(dead)
    assert popover.pos() == expected


def test_a_live_anchor_is_positioned_against_where_it_actually_is(qapp):
    """The fallback must not shadow the ordinary placement."""
    popover = _popover(qapp)
    popover.adjustSize()

    near = QPushButton("settings")
    near.setFixedSize(80, 24)
    near.move(0, 0)
    popover._position_near(near)
    first = popover.pos()

    far = QPushButton("settings")
    far.setFixedSize(80, 24)
    far.move(400, 500)
    popover._position_near(far)
    assert popover.pos() != first


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
