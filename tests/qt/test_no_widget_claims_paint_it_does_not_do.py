"""Nothing claims `WA_OpaquePaintEvent` while painting nothing.

`WA_OpaquePaintEvent` is a PROMISE: this widget fills every pixel of its own
rect, so Qt may skip erasing the background before a repaint. A widget that
makes the promise and then paints nothing leaves whatever was already on
screen, and transparent children draw on top of it.

REPORTED 2026-09-05, and all of it one symptom:

    "in the bottom left corner there is text that overlaps (new text is
     pasted over old text)"
    "the bar on the top with the spacr and help text and the text in the
     bottom left and right (the version) are all flickering"
    "strange flicker when i hover the modual buttons, settings categories"

MEASURED: `MainWindow` set `setAutoFillBackground(True)` and
`WA_OpaquePaintEvent, True` together, which is consistent -- but applying the
application stylesheet clears `autoFillBackground` again, so by the time the
window was shown it read `autoFill=False, opaquePaint=True`. The promise
outlived the thing that kept it, and every text surface over the animated
backdrop paid for it.

The window still paints: the stylesheet fills it, and a styled background is
drawn whether or not anything claims opacity. What was removed is the claim.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

#: Widgets that genuinely do fill every pixel, and are allowed the promise.
#: Each one paints an image or a solid ground across its whole rect.
ALLOWED = {"AmbientWidget", "LoadingScreen", "ScreenSaver",
           "FractalTravelWidget", "SpaceBackdrop"}


def _claims_opacity(widget) -> bool:
    try:
        return bool(widget.testAttribute(
            Qt.WidgetAttribute.WA_OpaquePaintEvent))
    except RuntimeError:
        return False


def test_the_main_window_does_not_claim_what_the_sheet_takes_away(
        qtbot, qt_theme_applied):
    """The exact pair that produced the overlapping text."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1440, 900)
    win.show()

    claims = _claims_opacity(win)
    fills = win.autoFillBackground()
    assert not (claims and not fills), (
        f"the window claims WA_OpaquePaintEvent while autoFillBackground is "
        f"{fills} -- it is promising to paint pixels it does not paint, and "
        "Qt will stop erasing behind it")


def test_no_chrome_widget_claims_opacity_without_filling(qtbot,
                                                         qt_theme_applied):
    """The same check across every widget the window builds.

    A widget that paints a whole image or a solid ground is allowed the
    promise -- those are named in ALLOWED and each one keeps it. Anything
    else claiming it, without `autoFillBackground`, is the defect.
    """
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1440, 900)
    win.show()

    liars = []
    for child in win.findChildren(QWidget):
        if type(child).__name__ in ALLOWED:
            continue
        if _claims_opacity(child) and not child.autoFillBackground():
            liars.append(f"{type(child).__name__}"
                         f"#{child.objectName() or '-'}")
    assert not liars, (
        "these widgets promise to paint every pixel and do not fill: "
        f"{sorted(set(liars))}")
