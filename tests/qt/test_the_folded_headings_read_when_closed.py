"""198: a folded sub-heading reads when it is not highlighted.

    "the text in the measurements tab for the sub categories should be white
    when they are not highlighted (in dark mode oposite in bright mode)."

IT WAS BACKWARDS. Both header styles -- `CollapsibleSection`'s own sheet and
the theme's `QToolButton#SectionHeader` -- painted the resting state in the
MUTED colour and the theme foreground only on hover or when open.

THE UNHIGHLIGHTED STATE IS THE ONE A USER READS. On a screen with sixteen
folded categories at most one is open, and the rest are what they are
scanning to decide where to go. Dimming them says "this is secondary" about
the only thing on the page that is not.

AND IT WAS NEVER ONLY THE MEASUREMENTS TAB. Both widgets are used by every
module screen, so fixing the one tab it was reported on would have left the
same fault on twenty-seven others.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _resting_colour(sheet: str, selector: str) -> str:
    """The `color:` of a selector's resting block."""
    at = sheet.index(selector + " {")
    block = sheet[at:sheet.index("}", at)]
    for line in block.splitlines():
        if "color:" in line and "background" not in line:
            return line.split("color:")[1].strip().rstrip(";").lower()
    raise AssertionError(f"{selector} sets no colour")


class TestTheThemeHeaderReadsAtRest:

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_it_is_the_foreground_not_the_muted_ink(self, theme):
        from spacr.qt.theme import palette_for, stylesheet

        colour = _resting_colour(stylesheet(theme),
                                 "QToolButton#SectionHeader")
        palette = palette_for(theme)

        assert colour == palette["fg"].lower()
        assert colour != palette["fg_muted"].lower()

    def test_the_two_themes_are_opposite(self):
        """"white ... in dark mode oposite in bright mode"."""
        from spacr.qt.theme import stylesheet

        dark = _resting_colour(stylesheet("dark"), "QToolButton#SectionHeader")
        light = _resting_colour(stylesheet("light"),
                                "QToolButton#SectionHeader")

        assert dark != light
        assert _lightness(dark) > 200, dark
        assert _lightness(light) < 60, light

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_the_highlight_is_still_visible(self, theme):
        """The condition on this change: open and hover must still be
        distinguishable -- by something that is not the text going away."""
        from spacr.qt.theme import stylesheet

        sheet = stylesheet(theme)
        at = sheet.index("QToolButton#SectionHeader:checked {")
        block = sheet[at:sheet.index("}", at)]

        assert "background" in block
        assert "border-bottom" in block


def _lightness(colour: str) -> float:
    text = colour.lstrip("#")
    if len(text) != 6:
        pytest.skip(f"not a hex colour: {colour}")
    return sum(int(text[i:i + 2], 16) for i in (0, 2, 4)) / 3


class TestTheFoldingSectionReadsAtRest:
    """`CollapsibleSection` carries its own sheet, so the theme's rule does
    not reach it and it had the same fault independently."""

    def test_the_resting_colour_is_the_palette_text(self, qtbot):
        from spacr.qt.widgets.collapsible_section import CollapsibleSection
        from PySide6.QtWidgets import QLabel

        section = CollapsibleSection("Attached databases", QLabel("body"))
        qtbot.addWidget(section)

        sheet = section._header.styleSheet()
        assert "color: palette(text)" in sheet
        assert "palette(mid)" not in sheet

    def test_and_it_still_marks_the_open_one(self, qtbot):
        from spacr.qt.widgets.collapsible_section import CollapsibleSection
        from PySide6.QtWidgets import QLabel

        section = CollapsibleSection("Attached databases", QLabel("body"))
        qtbot.addWidget(section)

        sheet = section._header.styleSheet()
        assert "QToolButton:hover" in sheet
        assert "background" in sheet.split("QToolButton:hover")[1]


class TestNoLiteralInk:
    """`#FFFFFF` here is the same fault instruction 178 removed from eleven
    figure call sites: it reads on one theme and vanishes on the other, and
    the author sees only the one they use."""

    def test_the_folding_header_names_no_literal(self, qtbot):
        from spacr.qt.widgets.collapsible_section import CollapsibleSection
        from PySide6.QtWidgets import QLabel

        section = CollapsibleSection("x", QLabel("body"))
        qtbot.addWidget(section)

        sheet = section._header.styleSheet().lower()
        for literal in ("#fff", "#ffffff", "white", "#000", "black"):
            assert f"color: {literal}" not in sheet, literal


class TestItHoldsOnAScreenThatFolds:
    """A rule can be right and overridden, so this reads the built widget."""

    @pytest.mark.parametrize("app_key", ["regression", "measure"])
    def test_the_headers_use_the_themes_foreground(self, qtbot, app_key):
        from PySide6.QtWidgets import QToolButton

        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen(app_key)
        qtbot.addWidget(screen)

        headers = [b for b in screen.findChildren(QToolButton)
                   if b.objectName() == "SectionHeader"]
        if not headers:
            pytest.skip(f"{app_key} has no folding section headers")

        # None of them may be styled by an inline sheet that re-dims them.
        for header in headers:
            own = header.styleSheet().lower()
            assert "fg_muted" not in own
            assert "palette(mid)" not in own


class TestMeasuredOffThePixels:
    """The acceptance criterion 198 asks for: read the RENDERED widget, not
    the sheet. A rule can be correct and then lost to a later selector, an
    inline sheet, or a disabled palette -- none of which reading the sheet
    would catch.

    The plate is filled with the theme background first. The header's own
    background is `transparent`, so an unfilled pixmap leaves the untouched
    pixels at zero, which reads as a black plate on BOTH themes -- and so
    passes by luck on dark and fails confusingly on light.
    """

    @pytest.mark.parametrize("theme,floor", [("dark", 200), ("light", 60)])
    def test_the_resting_text_contrasts_with_the_plate(self, qtbot, theme,
                                                       floor):
        from PySide6.QtGui import QColor, QPixmap
        from PySide6.QtWidgets import QApplication, QToolButton

        from spacr.qt.screens.app_screen import AppScreen
        from spacr.qt.theme import palette_for, stylesheet

        palette = palette_for(theme)
        QApplication.instance().setStyleSheet(stylesheet(theme))
        try:
            screen = AppScreen("measure")
            qtbot.addWidget(screen)
            screen.resize(900, 900)

            headers = [b for b in screen.findChildren(QToolButton)
                       if b.objectName() == "SectionHeader"]
            assert headers, "the Measurements tab folds -- it has headers"

            header = headers[0]
            header.setChecked(False)          # the state the user READS
            header.style().unpolish(header)
            header.style().polish(header)
            header.resize(300, 28)

            plate = QPixmap(header.size())
            plate.fill(QColor(palette["bg"]))
            header.render(plate)
            image = plate.toImage()

            greys = [
                sum((image.pixelColor(x, y).red(),
                     image.pixelColor(x, y).green(),
                     image.pixelColor(x, y).blue())) / 3
                for y in range(image.height())
                for x in range(image.width())
            ]
            background = max(set(greys), key=greys.count)
            ink = min(greys) if background > 128 else max(greys)

            # The ink lands on the right side of mid-grey for the theme...
            if theme == "dark":
                assert ink >= floor, f"resting text too dark on dark: {ink}"
            else:
                assert ink <= floor, f"resting text too light on light: {ink}"
            # ...and is actually legible against what is behind it.
            assert abs(background - ink) > 120
        finally:
            QApplication.instance().setStyleSheet("")
