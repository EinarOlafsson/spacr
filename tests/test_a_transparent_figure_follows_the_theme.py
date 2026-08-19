"""Transparent means the ground is the viewer's, so the ink is theirs too.

Asked for twice: "the background of the figures should be transparent and the
lines should be white in dark mode and black in light mode", and then reported
as a fault -- "a lot of the text in the figures is black in dark mode and the
axes as well".

`transparent` used to keep the PRINT ink on the ground it removes, arguing
that dark ink on a transparent ground is still unreadable on a dark slide.
That argument is right about `print` and wrong about this one: transparent
MEANS the ground is whatever the figure is pasted onto, and the only thing
that knows what that is, is the user -- who says so by their theme.
"""
import pytest

from spacr import figure_style
from spacr.figure_style import (DARK_GRID, DARK_INK, PRINT_GRID, PRINT_GROUND,
                                PRINT_INK, saved_figure_appearance, theme_ink)


@pytest.fixture()
def theme(monkeypatch):
    def use(name):
        monkeypatch.setattr("spacr.qt.preferences.resolve_effective_theme",
                            lambda: name)
    return use


def test_dark_mode_gets_light_ink(theme):
    theme("dark")

    look = saved_figure_appearance("transparent")

    assert look.ground is None
    assert look.ink == DARK_INK
    assert look.grid == DARK_GRID


def test_light_mode_is_the_print_pair_exactly(theme):
    """So switching a light-mode user to transparent moves only the ground."""
    theme("light")

    look = saved_figure_appearance("transparent")

    assert look.ground is None
    assert look.ink == PRINT_INK
    assert look.grid == PRINT_GRID


@pytest.mark.parametrize("name", ["space", "cell", "glass"])
def test_every_theme_that_is_not_light_is_treated_as_dark(theme, name):
    # `resolve_effective_theme` says so itself: Space and Cell are dark.
    theme(name)

    assert theme_ink() == (DARK_INK, DARK_GRID)


def test_print_is_untouched_by_any_of_this(theme):
    """A figure going into a manuscript must not change with the UI theme."""
    theme("dark")

    look = saved_figure_appearance("print")

    assert (look.ground, look.ink, look.grid) == (PRINT_GROUND, PRINT_INK,
                                                  PRINT_GRID)


def test_screen_is_still_a_no_op(theme):
    theme("dark")

    look = saved_figure_appearance("screen")

    assert (look.ground, look.ink, look.grid) == (None, None, None)


def test_a_headless_run_answers_the_print_pair(monkeypatch):
    """No application means no theme to follow, and it must not raise."""
    import builtins

    real_import = builtins.__import__

    def no_preferences(name, *args, **kwargs):
        if name.endswith("qt.preferences") or name == "spacr.qt.preferences":
            raise ImportError("no GUI here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_preferences)

    assert theme_ink() == (PRINT_INK, PRINT_GRID)
