"""The three marks top right, and the menu bar beside them.

Three things asked for once the chrome was in place:

* "the x in the top right should be the same height and width as the
  square" -- they sit side by side and are read as a pair, so they occupy
  the same box;
* "when the mouse is overed over of clicked on help or spaCR they should
  be blue" -- the same accent the dock's open section header takes;
* "when i press spaCR in the top left corner in fullscreen mode the drop
  down spawns under help not spaCR". A menu opens where the BAR SAYS its
  action is, so the bar has to be re-laid when the window state changes.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QSize, Qt

from spacr.qt import theme
from spacr.qt.app import MainWindow


def _extent(icon):
    """(width, height) of the drawn mark inside an 18 px icon."""
    image = icon.pixmap(QSize(18, 18)).toImage()
    lit = [(x, y)
           for x in range(image.width())
           for y in range(image.height())
           if image.pixelColor(x, y).alpha() > 40]
    assert lit, "the icon draws nothing"
    xs = [x for x, _y in lit]
    ys = [y for _x, y in lit]
    return max(xs) - min(xs), max(ys) - min(ys), min(xs), min(ys)


def test_the_x_is_the_size_of_the_square(qapp):
    """Same box, same origin -- not merely the same nominal icon size."""
    close_w, close_h, close_x, close_y = _extent(MainWindow._close_icon())
    full_w, full_h, full_x, full_y = _extent(MainWindow._fullscreen_icon())

    assert (close_w, close_h) == (full_w, full_h)
    assert (close_x, close_y) == (full_x, full_y)


def test_the_minus_sits_inside_that_same_box(qapp):
    """It is one rule wide, but it spans the pair's width."""
    minus_w, _h, minus_x, _y = _extent(MainWindow._minimise_icon())
    full_w, _fh, full_x, _fy = _extent(MainWindow._fullscreen_icon())

    assert minus_w == full_w
    assert minus_x == full_x


@pytest.mark.parametrize("theme_name", ("dark", "light"))
def test_the_menu_bar_words_light_in_the_accent(theme_name):
    """Hovered AND pressed, and as the word rather than a plate."""
    sheet = theme.stylesheet(theme_name)
    palette = theme.PALETTES[theme_name] if hasattr(theme, "PALETTES") else None

    start = sheet.find("QMenuBar::item:selected")
    assert start != -1, "the menu bar has no hover rule"
    rule = sheet[start:sheet.find("}", start)]

    assert "QMenuBar::item:pressed" in rule, "pressing it does nothing"
    # NO CONTRASTING PLATE -- asserted as the bar's own colour rather
    # than as `transparent`, which is what this used to demand.
    #
    # The two are identical to look at: painting the colour already
    # underneath you changes no pixel. They are not identical when the
    # paint is SKIPPED, which is what `transparent` means. What is behind
    # this bar is the window, whose palette Window role is the splash
    # colour -- pure black. On Linux the bar's own fill covers that; on
    # macOS the hover repaint clears to the window first, and the black
    # arrived as a plate behind the hovered word. Reported 2026-08-31:
    # "black boxes... appear only after hovering".
    #
    # So the intent this test defends is unchanged -- the word lights, no
    # plate grows -- and only the means of getting it has moved.
    assert f"background: {theme.menu_bar_background(theme_name)}" in rule, (
        "the hover paints something other than the bar's own colour, so "
        "it grows a plate -- or paints nothing, and shows the window's "
        "black through the hole")
    if palette is not None:
        assert palette["accent"].lower() in rule.lower()


def test_the_menu_bar_is_relaid_when_the_window_state_changes(qtbot,
                                                              qt_theme_applied):
    """The fix for a menu opening under the wrong word.

    Asserted on the HANDLER rather than on a popup position: a menu's
    placement needs a real screen, and what went wrong is that the bar's
    action rectangles were stale when Qt read them.
    """
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    bar = win.menuBar()

    seen = []
    original = bar.updateGeometry
    bar.updateGeometry = lambda: (seen.append(True), original())[1]

    win.changeEvent(QEvent(QEvent.Type.WindowStateChange))

    assert seen, "a window-state change does not re-lay the menu bar"
    # And every action still has a rectangle to open under.
    for action in bar.actions():
        assert bar.actionGeometry(action).isValid()
