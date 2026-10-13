"""The Preferences gear has to be big enough to see.

Reported as "I cannot see the gear that should represent the button" --
and the icon was never missing. `setIconSize` was never called, so it took
Qt's 16px default while this project ships a 1.5 font scale, leaving a
16px glyph in a 44px button with 1.5x text beside it.
"""

import pytest

from PySide6.QtCore import QSize

from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen


@pytest.fixture
def screen(qt_theme_applied, qtbot):
    widget = AppScreen("mask")
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qt_theme_applied.processEvents()
    return widget


class TestTheGearIsVisible:

    def test_the_icon_is_actually_there(self, screen):
        """Distinguishes 'missing glyph' from 'too small' -- they are
        different bugs and were confused in the first diagnosis."""
        icon = screen._btn_preferences.icon()
        assert not icon.isNull()
        assert not icon.pixmap(24, 24).isNull()

    def test_it_is_painted_rather_than_transparent(self, screen):
        """A glyph re-inked to the background colour is present and
        invisible, which looks identical to a missing one."""
        image = screen._btn_preferences.icon().pixmap(24, 24).toImage()
        opaque = sum(
            1
            for x in range(0, image.width(), 2)
            for y in range(0, image.height(), 2)
            if image.pixelColor(x, y).alpha() > 20)
        assert opaque > 10, "the gear renders but paints almost nothing"

    def test_it_is_a_readable_fraction_of_the_button(self, screen):
        """16px in a 44px button is what was reported as invisible."""
        button = screen._btn_preferences
        assert button.iconSize().width() >= 18
        assert button.iconSize().width() / button.height() > 0.35

    def test_it_grows_with_the_font_scale(self, qt_theme_applied, qtbot):
        """The real cause. A fixed 16px icon beside 1.5x text reads as
        absent, and this project's DEFAULT scale is 1.5 -- so the shipped
        configuration was the broken one and the tests pinned 1.0.
        """
        from spacr.qt import theme

        sizes = {}
        for scale in (1.0, 1.5):
            preferences.set_font_scale(scale)
            qt_theme_applied.setStyleSheet(theme.stylesheet())
            widget = AppScreen("mask")
            qtbot.addWidget(widget)
            sizes[scale] = widget._btn_preferences.iconSize().width()
        preferences.set_font_scale(1.0)
        assert sizes[1.5] > sizes[1.0], (
            f"icon did not scale: {sizes}")
