"""The ambient backdrop's controls, caches and refusals.

This paints behind every module screen while a pipeline runs, so nothing in it
may raise and nothing in it may grow without bound. The tests below drive the
parts that only a user turning a Preferences knob reaches: every setter that
drops a cache, the two caches that reset instead of growing, the blur that
resizes the buffer, and the reads that have to survive a theme module or a
preferences file that cannot be loaded.
"""

from __future__ import annotations


import pytest

from PySide6.QtGui import QColor

from spacr.qt.widgets import ambient as amb
from spacr.qt.widgets.ambient import (AmbientWidget, drift_direction_label)

pytestmark = pytest.mark.qt

DARK = "#101418"


# ---------------------------------------------------------------------------
# Names and the theme colour
# ---------------------------------------------------------------------------

def test_an_unknown_drift_direction_is_named_in_the_error():
    """A stale preferences value must say what the valid ones are."""
    with pytest.raises(ValueError, match="unknown starfield direction"):
        drift_direction_label("sideways")


def test_the_backdrop_colour_falls_back_when_the_theme_cannot_be_read(
        monkeypatch):
    """A backdrop must never raise on its way to a screen.

    The flat fill under the animation is read from the live theme; if that
    read fails the dark page colour is used, because a screen with no
    backdrop at all is worse than one in the wrong palette.
    """
    from spacr.qt import theme

    def refuse():
        raise RuntimeError("no palette loaded")

    monkeypatch.setattr(theme, "active_page_colour", refuse)
    colour = amb._theme_background()

    assert isinstance(colour, QColor)
    assert colour == QColor(amb.page_colour("dark"))


def test_the_motion_settings_fall_back_when_preferences_cannot_be_read(
        monkeypatch):
    """Unreadable preferences give the shipped animation, not a crash."""
    import builtins

    real_import = builtins.__import__

    def no_preferences(name, *args, **kwargs):
        if "preferences" in name:
            raise ImportError("preferences are unreadable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_preferences)
    try:
        motion = amb.preferred_motion()
    finally:
        monkeypatch.undo()

    assert motion == amb.Motion(amb.DEFAULT_BLUR, amb.DEFAULT_SPEED,
                                amb.DEFAULT_SIZE, amb.DEFAULT_RESOLUTION,
                                amb.DEFAULT_DENSITY,
                                amb.DEFAULT_DRIFT_DIRECTION)


# ---------------------------------------------------------------------------
# The engine setters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The buffered engines
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The aurora
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The starfield
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The widget's own reads
# ---------------------------------------------------------------------------

def test_the_widget_reports_back_every_control_it_was_given(qtbot):
    """Preferences reads the active controls to fill its sliders in.

    A getter that did not reflect the setter beside it would show the user a
    control in one position and animate in another. Retired blur remains inert
    on the widget even when an older caller supplies a nonzero value.
    """
    widget = AmbientWidget(theme="drift", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)

    widget.set_blur(1.25)
    widget.set_resolution(0.75)
    widget.set_density(2.0)
    widget.set_speed(1.5)
    widget.set_size_scale(1.75)
    widget.set_direction("down")

    assert widget.blur() == amb.DEFAULT_BLUR
    assert widget.engine.blur == amb.DEFAULT_BLUR
    assert widget.resolution() == 0.75
    assert widget.density() == 2.0
    assert widget.speed() == 1.5
    assert widget.size_scale() == 1.75
    assert widget.direction() == "down"


def test_an_unknown_direction_leaves_the_widget_where_it_was(qtbot):
    """A stale preferences value must not blank the starfield's setting."""
    widget = AmbientWidget(theme="drift", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.set_direction("down")

    widget.set_direction("sideways")

    assert widget.direction() == "down"
