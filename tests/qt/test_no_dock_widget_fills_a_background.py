"""Nothing in the dock fills a background. Only the slab is painted.

THE BLACK BOX, reported four times on 2026-09-03 and blamed on the
stylesheet three of them:

    "replace the black box behind thicons with a translucent box"
    "the black box is still there and the transparent box ... is not"
    "the black box is still there. the flickering is still there"
    "blinking is still there and black box is still there"

IT WAS NEVER THE STYLESHEET. A widget that overrides `paintEvent` and does
not paint a background still gets one: Qt fills its rectangle with the
palette brush before `paintEvent` runs, and that brush is `surface_alt` --
`#161719`, the exact colour measured behind every icon.

MEASURED, a plain button and a dock row carrying the SAME object name, each
rendered over magenta:

    plain QPushButton#SidebarItem      #ff00ff   (the QSS was always right)
    _DockRow                           #161719   (the box)
    _DockRow with WA_NoSystemBackground #ff00ff  (the fix)

The lesson worth keeping is the middle line. Every earlier fix edited
`QPushButton#SidebarItem` in `theme.py`, and every one of them was checked
against a widget that had no custom `paintEvent` -- so the stylesheet
measured clean, and the running application drew the box.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QPushButton

MARKER = "#ff00ff"


def _fill_over_marker(widget, *, hover=False):
    """What ``widget`` leaves on a magenta ground, beside its icon."""
    widget.setFixedSize(160, 44)
    widget.ensurePolished()
    if hover:
        widget.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    shot = QPixmap(widget.size())
    shot.fill(QColor(MARKER))
    widget.render(shot)
    return shot.toImage().pixelColor(widget.width() - 8,
                                     widget.height() // 2).name()


@pytest.mark.parametrize("state", ["rest", "hover", "checked"])
def test_a_dock_row_fills_nothing_in_any_state(state, qtbot,
                                               qt_theme_applied):
    """The row itself, which is where the box actually was."""
    from spacr.qt.app import _DockRow

    row = _DockRow("x")
    qtbot.addWidget(row)
    row.setObjectName("SidebarItem")
    if state == "checked":
        row.setCheckable(True)
        row.setChecked(True)
    seen = _fill_over_marker(row, hover=(state == "hover"))
    assert seen == MARKER, (
        f"a dock row fills {seen} behind its icon when {state} -- that is "
        "the black box")


def test_the_row_declares_that_it_paints_its_own_background():
    """Stated as well as measured, because the measurement is indirect.

    A future edit that drops this attribute brings the box back, and the
    test above would then be the only thing between it and a release.
    """
    from spacr.qt.app import _DockRow

    row = _DockRow("x")
    try:
        assert row.testAttribute(
            Qt.WidgetAttribute.WA_NoSystemBackground), (
            "the row no longer declares WA_NoSystemBackground")
    finally:
        row.deleteLater()


@pytest.mark.parametrize("state", ["rest", "hover", "checked"])
def test_the_stylesheet_was_never_the_problem(state, qtbot,
                                              qt_theme_applied):
    """A PLAIN button with the dock's object name fills nothing either.

    This is the control that was missing. It passed before the fix and it
    passes after, which is exactly the point: it is what made three
    stylesheet edits look correct while the dock still drew a box.
    """
    button = QPushButton("x")
    qtbot.addWidget(button)
    button.setObjectName("SidebarItem")
    if state == "checked":
        button.setCheckable(True)
        button.setChecked(True)
    assert _fill_over_marker(button, hover=(state == "hover")) == MARKER


def test_the_generic_button_rule_still_paints_ordinary_buttons(
        qtbot, qt_theme_applied):
    """The dock is the exception, not the rule.

    `QPushButton { background-color: surface_alt }` is what every ordinary
    button in the application is drawn with, so a fix that reached all of
    them would have flattened every dialog. Asserted so a later "simplify"
    cannot quietly do that.
    """
    button = QPushButton("x")
    qtbot.addWidget(button)
    assert _fill_over_marker(button) != MARKER, (
        "ordinary buttons stopped being painted")
