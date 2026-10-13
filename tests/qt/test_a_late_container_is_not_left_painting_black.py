"""A container built after construction must still lose its background.

THE BUG THIS PINS, reported 2026-09-13: "there is now a black box in the
background of the settings container... when i opened it i saw the black box
and then went to a different module measure and saw the same thing then i
want back and it was gone."

`_clear_page_surfaces` tags every layout container so the backdrop reaches
the eye, and it runs during construction -- so it tags what exists THEN.
Anything the screen builds afterwards inherits the blanket
``QWidget { background-color: bg }`` rule and paints the window colour as a
solid rectangle on top of the animation.

IT LOOKED INTERMITTENT BECAUSE THE REPAIR WAS ACCIDENTAL. `showEvent` calls
`refresh_ambient_background`, which re-tags -- but only when the ambient
preference actually changed, which its own docstring states. Leaving the
screen and returning sometimes took that path. So the box appeared on first
open and was gone on the second, which reads like a paint race and is not
one: it is a container nobody ever tagged, repaired by a side effect.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget                       # noqa: E402

from spacr.qt.theme import TRANSPARENT_PROPERTY             # noqa: E402


def _a_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


def test_a_container_added_after_construction_is_tagged_on_first_show(qtbot):
    """The actual defect: built late, shown, and still opaque."""
    screen = _a_screen(qtbot)
    # ANONYMOUS ON PURPOSE. `clear_container_surfaces` treats an unnamed
    # QWidget as scaffolding that should show what is behind it, and a NAMED
    # one as something the designer styled deliberately, which keeps its
    # fill. Naming this would test the opposite of the contract -- the first
    # draft of this test did, and failed for that reason rather than for the
    # bug it is about.
    late = QWidget(screen)
    assert late.property(TRANSPARENT_PROPERTY) in (None, False), (
        "the fixture is wrong: this widget was already tagged")

    screen.show()
    qtbot.waitExposed(screen)

    assert late.property(TRANSPARENT_PROPERTY) is True, (
        "a container built after construction still paints the window "
        "colour over the backdrop on the screen's first show")


def test_the_sweep_does_not_run_again_on_every_show(qtbot):
    """Tagging walks every child and re-polishes it; Mask has 201 settings.

    Correctness would allow running it on every show. This asserts the
    guard, because a sweep on every tab switch is a cost a user feels and
    the next person to touch `showEvent` should have to defeat a test to
    reintroduce it.
    """
    screen = _a_screen(qtbot)
    calls = []
    original = screen._clear_page_surfaces
    screen._clear_page_surfaces = lambda: (calls.append(1), original())[1]

    screen.show()
    qtbot.waitExposed(screen)
    after_first = len(calls)
    screen.hide()
    screen.show()
    qtbot.waitExposed(screen)

    assert after_first == 1, f"expected one sweep on first show, got {after_first}"
    assert len(calls) == 1, (
        f"the sweep ran again on a later show ({len(calls)} times total)")
