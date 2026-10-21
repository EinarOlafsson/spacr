"""The Preferences gear has to be big enough to see.

Reported as "I cannot see the gear that should represent the button" --
and the icon was never missing. `setIconSize` was never called, so it took
Qt's 16px default in the originally reported 1.5x font configuration,
leaving a 16px glyph in a 44px button with 1.5x text beside it.
"""

import pytest

from PySide6.QtCore import QSize

from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen


@pytest.fixture
def _geometry_at_design_scale():
    """Keep saved and active zoom local to the icon's literal pixels."""
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 1.0)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(1.0)
            yield
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


@pytest.fixture
def _geometry_at_default_scale(qapp):
    """Measure the gear with fresh font and GUI preferences."""
    from spacr.qt import gui_scale, theme

    original_sheet = qapp.styleSheet()
    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    try:
        with pytest.MonkeyPatch.context() as patch:
            try:
                for key in keys:
                    settings.remove(key)
                patch.setattr(gui_scale, "_SCALE", preferences.get_gui_scale())
                qapp.setStyleSheet(theme.stylesheet(
                    font_scale=preferences.get_font_scale()))
                yield
            finally:
                for key in keys:
                    if key in previous:
                        settings.setValue(key, previous[key])
                    else:
                        settings.remove(key)
    finally:
        qapp.setStyleSheet(original_sheet)


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

    def test_it_is_a_readable_fraction_of_the_button(
            self, _geometry_at_design_scale, screen):
        """16px in a 44px button is what was reported as invisible."""
        button = screen._btn_preferences
        assert button.iconSize().width() >= 18
        assert button.iconSize().width() / button.height() > 0.35

    def test_the_fresh_default_gear_is_readable_and_painted(
            self, _geometry_at_default_scale, screen):
        from spacr.qt import gui_scale

        button = screen._btn_preferences
        original = gui_scale._original_for(button, "iconSize")
        size = original(button) if original is not None else button.iconSize()
        assert size.width() / button.height() > 0.35
        image = button.icon().pixmap(size).toImage()
        opaque = sum(image.pixelColor(x, y).alpha() > 20
                     for x in range(image.width())
                     for y in range(image.height()))
        assert opaque > 10, "the default gear paints almost nothing"

    def test_it_grows_with_the_font_scale(self, qt_theme_applied, qtbot):
        """The real cause. A fixed 16px icon beside 1.5x text reads as
        absent in the originally reported 1.5x configuration. Both sizes
        must follow the selected font scale.
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
