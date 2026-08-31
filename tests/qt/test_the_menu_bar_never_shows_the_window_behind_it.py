"""Hovering the menu bar showed a black box behind the label.

Reported from macOS: "there are black boxes behind the minimize,
fullscreen and close icons in the top right... the black boxes appear
only after hovering the mouse over the icon. same happens to the spacr
and Help text." And, in the same breath, "the bar is transparent and has
a gradient so it is hard to see the spaCR and Help".

Both come from ``background: transparent``. Transparent means PAINT
NOTHING, and what is behind this bar is the window -- whose palette
Window role is the splash colour, ``#000000``. On Linux the bar's own
fill covers that and nothing is ever seen; on macOS the hover repaint
clears to the window first, and pure black arrives as a plate behind
whatever was hovered.

The "gradient" is the same hole seen from the other side: the bar is the
frameless window's title bar, so a fully transparent one shows the
animated backdrop moving underneath the only two words on it.

These tests are written against the GENERATED stylesheet rather than a
screenshot, because the defect is a colour that was never asked for
rather than one that looks wrong -- and because it cannot be reproduced
on the platform CI runs on.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from spacr.qt.theme import (MENU_BAR_ALPHA, menu_bar_background, palette_for,
                            stylesheet)

#: The rules that were `transparent` and paint the reported black.
HOVER_RULES = (
    r"QMenuBar \{(.*?)\}",
    r"QMenuBar::item \{(.*?)\}",
    r"QMenuBar::item:selected, QMenuBar::item:pressed \{(.*?)\}",
)


def _block(css: str, pattern: str) -> str:
    """One QSS block's body, with its comments stripped."""
    found = re.search(pattern, css, re.S)
    assert found is not None, f"no rule matching {pattern}"
    return re.sub(r"/\*.*?\*/", "", found.group(1), flags=re.S)


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("pattern", HOVER_RULES)
def test_no_menu_bar_rule_paints_nothing(theme, pattern):
    """None of the three is transparent, in either theme.

    The hover rule is the one that was reported, but all three are
    asserted: `QMenuBar::item` is what a hover repaint starts from, and
    an opaque hover over a transparent item would still flash black.
    """
    body = _block(stylesheet(theme), pattern)
    assert "transparent" not in body, (
        "a menu bar rule paints nothing, so the window's black shows "
        f"through it:\n{body.strip()}")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_bar_is_completely_opaque(theme):
    """Nothing shows through it at all.

    This asked for "a little transparent" first, and 0.94 was tried.
    Seen on a real screen it was still wrong: "remove the transparency
    for the bar and it will be perfect". The bar is the frameless
    window's title bar, so anything behind it is motion underneath text,
    and a little motion under text is not better than none.

    Asserted on the ALPHA and on the rendered colour together, because
    the rendered colour is what the flat themes are separately required
    to keep as plain hex.
    """
    assert MENU_BAR_ALPHA == 1.0, (
        f"{MENU_BAR_ALPHA} lets the backdrop through the title bar")
    colour = menu_bar_background(theme)
    assert not colour.startswith("rgba("), (
        f"{colour} is translucent; the bar was asked to be solid")
    assert colour in _block(stylesheet(theme), HOVER_RULES[0])


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_bar_is_one_flat_colour_and_not_a_gradient(theme):
    """"dont use a gradient". A gradient behind two words is why they
    were hard to read."""
    body = _block(stylesheet(theme), HOVER_RULES[0])
    assert "gradient" not in body.lower(), (
        f"the menu bar draws a gradient:\n{body.strip()}")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_bar_colour_is_the_surface_colour_at_that_alpha(theme):
    """The colour is DERIVED, not a second hand-written constant.

    This is what stops the corner chrome and the bar drifting apart:
    `spacr.qt.app` styles the window chrome with a stylesheet of its own
    and reads the same function, so there is one colour rather than two
    that have to be kept equal.
    """
    red, green, blue = (int(part) for part in
                        re.findall(r"\d+", menu_bar_background(theme))[:3])
    surface = palette_for(theme)["surface"].lstrip("#")
    assert (red, green, blue) == tuple(
        int(surface[i:i + 2], 16) for i in (0, 2, 4))


def test_the_window_chrome_paints_the_same_colour_as_the_bar(qtbot,
                                                             qt_theme_applied,
                                                             tmp_path,
                                                             monkeypatch):
    """The corner marks sit ON the bar, so they must paint ITS colour.

    Driven through a real window rather than by reading app.py, because
    the failure being prevented is the corner keeping a colour of its own
    after the bar's changed -- which a source grep would not notice.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import app as app_mod

    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    chrome = window._window_buttons
    sheet = chrome.styleSheet()
    assert "transparent" not in sheet, (
        "the window chrome paints nothing, which is the black box")
    assert menu_bar_background() in sheet
