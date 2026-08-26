"""The Console control on Annotate's bottom row, measured in pixels.

It had a black box behind it. A ``QToolButton`` with no rule of its own is
drawn by the widget style from the palette's Button role -- a dark slab
under the caption -- and being checkable bought it nothing, because that
slab looks the same whether the pane is open or shut.

What it should be is text: the theme's own foreground colour, no plate, and
the accent while the console is open. That is a STATE, so it holds with
nobody touching the button, the same shape a checkable fold button's stage
fill holds (:mod:`spacr.qt.widgets.fold_strip`).

**Measured, not read.** The button is rendered over a magenta ground with
only its children drawn, so every pixel that is not magenta is a pixel the
button painted. A plated button paints essentially all of them; a caption
on the page paints the glyphs and nothing else. The toggle is driven with a
real mouse click, because the question is what a user sees after clicking
it, not what the slot does when called.
"""
from __future__ import annotations

from collections import Counter

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

#: Rendered over this; nothing in the theme is magenta.
SENTINEL = QColor(255, 0, 255)

#: A caption with no plate behind it covers a small share of its own
#: rectangle. The measured value is about 0.14; a plated button measures
#: 0.999, so anything in between is a plate that has grown or shrunk rather
#: than a threshold anybody has to tune.
INK_ONLY = 0.5


@pytest.fixture
def annotate_screen(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)
    return screen


def _ink(widget) -> tuple:
    """``(painted share, colour counts)`` of what the widget draws itself."""
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(SENTINEL)
    widget.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    counts = Counter()
    for y in range(image.height()):
        for x in range(image.width()):
            counts[QColor(image.pixel(x, y)).name().lower()] += 1
    total = image.width() * image.height()
    painted = total - counts[SENTINEL.name().lower()]
    return painted / float(total), counts


def test_the_console_switch_has_no_plate_behind_it(annotate_screen):
    """White text on the page, not a caption sitting on a box."""
    from spacr.qt.theme import active_palette

    palette = active_palette()
    share, counts = _ink(annotate_screen._console_switch)
    assert share < INK_ONLY, (
        f"the switch painted {share:.1%} of its own rectangle, which is a "
        "plate rather than a caption")
    assert counts[palette["fg"].lower()] > 0, (
        "the caption is not drawn in the theme's foreground colour")
    assert counts[palette["accent"].lower()] == 0, (
        "the switch is already lit with the console shut")


def test_clicking_it_lights_it_blue_and_it_stays_lit(annotate_screen,
                                                     qt_theme_applied):
    """Driven by a real click, and measured with nothing hovering it."""
    from spacr.qt.theme import active_palette

    palette = active_palette()
    switch = annotate_screen._console_switch
    assert not switch.isChecked()

    QTest.mouseClick(switch, Qt.LeftButton, Qt.NoModifier,
                     QPoint(switch.width() // 2, switch.height() // 2))
    qt_theme_applied.processEvents()

    assert switch.isChecked(), "the click did not open the console"
    assert not annotate_screen._console_wrap.isHidden()

    share, counts = _ink(switch)
    assert share < INK_ONLY, (
        "lighting the switch gave it a plate it did not have before")
    assert counts[palette["accent"].lower()] > 0, (
        "the switch did not turn blue while the console is open")
    assert counts[palette["fg"].lower()] == 0, (
        "the caption is still drawn in the resting colour")


def test_clicking_it_again_puts_the_caption_back(annotate_screen,
                                                 qt_theme_applied):
    """The lit state follows the pane, in both directions."""
    from spacr.qt.theme import active_palette

    palette = active_palette()
    switch = annotate_screen._console_switch
    middle = QPoint(switch.width() // 2, switch.height() // 2)
    QTest.mouseClick(switch, Qt.LeftButton, Qt.NoModifier, middle)
    QTest.mouseClick(switch, Qt.LeftButton, Qt.NoModifier, middle)
    qt_theme_applied.processEvents()

    assert not switch.isChecked()
    assert annotate_screen._console_wrap.isHidden()
    _, counts = _ink(switch)
    assert counts[palette["fg"].lower()] > 0
    assert counts[palette["accent"].lower()] == 0


def test_the_lit_colour_comes_from_the_theme_rather_than_a_literal(
        annotate_screen, qt_theme_applied):
    """One place a colour is written down, as the fold switches do it.

    The screen is built first on purpose. This module is imported lazily by
    the application, so its block is registered minutes after the only
    stylesheet that would have carried it -- the screen's constructor is
    what re-applies the sheet, and a rule that only appears when somebody
    remembers to call for it is a rule that is missing in the running app.
    """
    from spacr.qt.screens.annotate import CONSOLE_SWITCH_NAME
    from spacr.qt.theme import active_palette

    sheet = qt_theme_applied.styleSheet()
    assert f"QToolButton#{CONSOLE_SWITCH_NAME}" in sheet, (
        "the switch has no rule in the application stylesheet, so it cannot "
        "follow a theme change")
    palette = active_palette()
    block = sheet.split(f"QToolButton#{CONSOLE_SWITCH_NAME}", 1)[1]
    block = block[:block.find("QToolButton#%s:hover" % CONSOLE_SWITCH_NAME)
                  + 400]
    assert palette["accent"] in block
    assert "background: transparent" in block
