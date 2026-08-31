"""The chaining strip must not paint a black box above the Run button.

WHAT WENT WRONG. `install_chaining` puts the strip into the runtime
panel immediately above the actions row -- the last thing the eye
crosses on the way to Run. The screen's page-surface sweep runs when the
screen is BUILT, and the strip arrives afterwards, so the sweep never
saw the rows inside it. An anonymous QWidget holding a layout inherits
the blanket `QWidget { background-color: bg }` rule and paints the
WINDOW colour, which is not a surface and which no page-opacity setting
can reach.

The user reported it twice: "in regression there is a field over the run
button, same row as the use it boton that has a black field behind it",
and "remove the black box in regression above the run button and below
inputs".

The sweep is a heuristic with two halves, and both are asserted here:
anonymous scaffolding is tagged, and anything NAMED -- including the
strip itself, which is a QFrame that paints on purpose -- is left alone.
A fix that tagged everything would take the strip's own surface away and
trade a black box for an invisible one.
"""
from __future__ import annotations

import pytest

from spacr.qt.chaining import ChainingBar, install_chaining
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.theme import TRANSPARENT_PROPERTY


@pytest.fixture
def regression_screen(qtbot):
    scr = AppScreen("regression")
    qtbot.addWidget(scr)
    scr._clear_page_surfaces()          # as building the screen does
    return scr


def _tagged(widget) -> bool:
    return bool(widget.property(TRANSPARENT_PROPERTY))


def test_the_pinned_row_does_not_paint_the_window_colour(regression_screen):
    """THE REPORTED BUG. The row holding the pin and "Use it"."""
    bar = install_chaining(regression_screen)
    assert bar is not None, "regression should carry a chaining strip"
    assert _tagged(bar._pinned_row), (
        "the row behind the pinned input and its Use it button is opaque "
        "again -- that is the black box above Run")


def test_the_continue_row_is_swept_too(regression_screen):
    """The same scaffolding, one row down; it had the same defect."""
    bar = install_chaining(regression_screen)
    assert bar is not None
    assert _tagged(bar._next_row)


def test_every_anonymous_container_in_the_strip_is_swept(regression_screen):
    """Written against the shape, not against two attribute names.

    A row added to this strip later would be a new black box, and naming
    the two that exist today would not catch it.
    """
    from PySide6.QtWidgets import QWidget

    bar = install_chaining(regression_screen)
    assert bar is not None
    missed = [w for w in bar.findChildren(QWidget)
              if type(w) is QWidget and not w.objectName()
              and not _tagged(w)]
    assert missed == [], (
        f"{len(missed)} anonymous container(s) in the strip still paint "
        "the window colour")


def test_the_strip_itself_keeps_its_surface(regression_screen):
    """The other half of the heuristic.

    `ChainingBar` is a QFrame -- a component that paints deliberately.
    Sweeping it would make the strip itself vanish, which is a worse bug
    than the one being fixed and a much harder one to see.
    """
    bar = install_chaining(regression_screen)
    assert bar is not None
    assert not _tagged(bar), "the strip was swept along with its scaffolding"


def test_the_named_labels_keep_their_surface(regression_screen):
    """A named widget is something the designer styled on purpose."""
    bar = install_chaining(regression_screen)
    assert bar is not None
    for label in (bar._pinned, bar._source, bar._stale, bar._fix):
        assert label.objectName(), "this test is asserting nothing"
        assert not _tagged(label), (
            f"{label.objectName()} lost the fill it was given on purpose")


def test_installing_twice_returns_the_same_strip(regression_screen):
    """The early return must not build a second, unswept strip.

    `install_chaining` answers an existing bar rather than adding
    another; a second one would sit below the first, unswept, as a fresh
    black box.
    """
    first = install_chaining(regression_screen)
    second = install_chaining(regression_screen)
    assert first is second
    assert len(regression_screen.findChildren(ChainingBar)) == 1


def test_a_sweep_that_fails_still_leaves_a_working_strip(regression_screen,
                                                        monkeypatch):
    """Decoration must never stop the strip from being installed.

    A screen that opens without the strip is the old behaviour and is
    always better than a screen that does not open -- the same rule the
    function's own docstring states.
    """
    import spacr.qt.theme as theme

    def explode(*_a, **_k):
        raise RuntimeError("no style on this platform")

    monkeypatch.setattr(theme, "clear_container_surfaces", explode)
    bar = install_chaining(regression_screen)
    assert bar is not None, "a failed sweep took the strip down with it"
    assert bar._btn_use.text() == "Use it"
