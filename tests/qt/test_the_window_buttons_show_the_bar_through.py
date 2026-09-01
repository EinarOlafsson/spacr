"""The three window marks paint no plate of their own.

Reported 2026-09-01: "the x square and minus in the top right dont
always have the same background as the container, please remove or make
transparent their background color if possible."

They used to be painted the menu bar's colour, read once from the theme
at construction. That is a SNAPSHOT: the bar repaints for a theme
change, a palette change, and on macOS for a translucency the copied
value never had, and each of those leaves three plates in the old
colour. Matching by copying is the bug. Showing through cannot drift,
because there is nothing to keep in step.

Transparent is safe HERE and not for the bar itself -- these sit inside
the menu bar, which paints its own surface, whereas the bar is a
top-level surface and would show the desktop through.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_mod

CHROME = ("MinimiseWindow", "FullScreenToggle", "CloseWindow")


def _chrome_qss() -> str:
    """The stylesheet the corner widget is given, read from the source.

    Reading the SOURCE rather than building a MainWindow: constructing
    one pulls in the whole application, and the claim under test is
    about what the rule says.
    """
    import inspect

    source = inspect.getsource(app_mod.MainWindow._install_fullscreen_button)
    match = re.search(r'corner\.setStyleSheet\((.*?)\)\n', source, re.S)
    assert match, "the corner widget no longer sets a stylesheet"
    return match.group(1)


def test_the_corner_paints_no_surface():
    qss = _chrome_qss()
    assert "background: transparent;" in qss
    assert "QWidget#WindowChrome {" in qss


def test_no_state_of_a_chrome_button_paints_a_plate():
    """Hover, pressed, checked and disabled included -- a plate in any one
    of them is the mismatch coming back for that state only."""
    qss = _chrome_qss()
    for state in ("QToolButton,", "QToolButton:hover,",
                  "QToolButton:pressed,", "QToolButton:checked,",
                  "QToolButton:disabled"):
        assert state in qss, f"{state} lost its rule"
    assert qss.count("background: transparent;") >= 2
    assert "background: #" not in qss, (
        "a literal colour is back in the chrome stylesheet; that is the "
        "snapshot which drifts from the bar")


def test_the_bar_colour_is_no_longer_copied_into_the_corner():
    """THE REGRESSION. A call to menu_bar_background() here means the
    colour is being snapshotted again."""
    import inspect

    source = inspect.getsource(app_mod.MainWindow._install_fullscreen_button)
    # CODE lines only. The comment explains the bug and names the call it
    # removed; matching the bare word fails on its own explanation.
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "menu_bar_background()" not in code, (
        "the corner reads the bar's colour again; it cannot stay in step "
        "with a bar that repaints")


def test_the_hover_state_is_a_glyph_repaint_not_a_plate():
    """WHY removing the plate costs nothing.

    _ChromeButton swaps the ICON for one painted in the hover colour, so
    the feedback survives having no background at all.
    """
    import inspect

    source = inspect.getsource(app_mod._ChromeButton)
    assert "setIcon" in source
    assert "_hover_colour" in source
    assert "setStyleSheet" not in source, (
        "the button paints its own background again")


@pytest.mark.parametrize("name", CHROME)
def test_each_button_still_has_a_hover_colour(name):
    """So 'transparent everywhere' did not quietly remove the feedback."""
    assert name in app_mod.CHROME_HOVER
    assert app_mod.CHROME_HOVER[name].startswith("#")
