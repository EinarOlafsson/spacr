"""The grey panel the annotation grid is laid out on, measured at its corner.

It had square corners while every panel beside it was rounded. The fix is
not a number invented here: :data:`spacr.qt.theme.RADIUS` already carries
the corner radii the application uses, and ``md`` is the one the settings
cards, the tab panes and the chart frames all round at.

**Measured, not read.** A stylesheet string cannot say whether a corner is
round -- the rule can be present and the widget can still paint square,
which is exactly what happened when the backdrop's colour was set on the
scroll VIEWPORT: an unselectored ``setStyleSheet`` is inherited by every
descendant and outranks the application sheet, so it blanked the panel
inside it. So the corner is rendered and compared, pixel for pixel, with
a reference widget carrying a known radius and the same fill.
"""
from __future__ import annotations

import re

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QRegion
from PySide6.QtWidgets import QWidget

#: Rendered over this. Nothing in the theme is magenta, so a pixel that
#: still reads as the sentinel is a pixel the panel did not paint.
SENTINEL = QColor(255, 0, 255)

#: How many rows of the top-left corner the profile covers. Comfortably more
#: than the largest radius the theme declares.
CORNER_ROWS = 14


def _corner_profile(widget, rows: int = CORNER_ROWS) -> tuple:
    """How far the fill is bitten back on each of the top ``rows`` rows.

    One number per row: the count of leading pixels the widget left alone.
    A square panel answers all zeros; a rounded one answers a descending
    run whose shape IS the radius.
    """
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(SENTINEL)
    widget.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    profile = []
    for y in range(rows):
        run = 0
        while run < rows and QColor(image.pixel(run, y)).rgb() == SENTINEL.rgb():
            run += 1
        profile.append(run)
    return tuple(profile)


def _declared_rule(app) -> tuple:
    """The fill and the radius the live stylesheet gives the backdrop."""
    from spacr.qt.screens.annotate import GRID_OBJECT_NAME

    body = re.search(r"QWidget#%s\s*\{([^}]*)\}" % GRID_OBJECT_NAME,
                     app.styleSheet())
    assert body is not None, (
        "the backdrop has no rule in the application stylesheet at all")
    fill = re.search(r"background:\s*([^;]+);", body.group(1))
    radius = re.search(r"border-radius:\s*(\d+)px", body.group(1))
    assert fill is not None and radius is not None
    return fill.group(1).strip(), int(radius.group(1))


@pytest.fixture
def annotate_screen(qtbot, qt_theme_applied, monkeypatch):
    """The screen, showing its grid page rather than the empty state."""
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)
    screen._content_stack.setCurrentWidget(screen._grid_scroll)
    qt_theme_applied.processEvents()
    return screen


def test_the_backdrop_rounds_at_the_radius_the_theme_declares(
        annotate_screen, qt_theme_applied):
    """The corner is the theme's ``md``, and measurably not ``sm`` or ``lg``."""
    from spacr.qt.theme import RADIUS

    fill, declared = _declared_rule(qt_theme_applied)
    assert declared == RADIUS["md"], (
        f"the backdrop declares {declared}px, which is not the theme's md "
        f"({RADIUS['md']}px) that the panels beside it use")

    holder = annotate_screen._grid_holder
    measured = _corner_profile(holder)
    assert any(measured), (
        "the backdrop painted right into its corner: it is still a square "
        "box")

    def reference(radius: int) -> tuple:
        probe = QWidget()
        probe.setObjectName("BackdropRadiusReference")
        probe.setStyleSheet(
            "QWidget#BackdropRadiusReference { background: %s; "
            "border-radius: %dpx; }" % (fill, radius))
        probe.resize(holder.size())
        probe.show()
        qt_theme_applied.processEvents()
        try:
            return _corner_profile(probe)
        finally:
            probe.hide()
            probe.deleteLater()

    assert measured == reference(RADIUS["md"]), (
        f"the corner measured {measured}, which is not what a "
        f"{RADIUS['md']}px radius draws")
    assert measured != reference(RADIUS["sm"])
    assert measured != reference(RADIUS["lg"])


def test_nothing_paints_a_square_grey_behind_the_rounded_corner(
        annotate_screen, qt_theme_applied):
    """The container behind the panel has to leave the corner alone.

    The viewport used to be filled with the same grey. A rounded panel on
    top of that is a rounded panel nobody can see: the corners are filled
    by the widget underneath and the box reads square anyway.
    """
    viewport = annotate_screen._grid_scroll.viewport()
    image = QImage(viewport.size(), QImage.Format_ARGB32)
    image.fill(SENTINEL)
    viewport.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    corner = QColor(image.pixel(0, 0)).rgb()
    assert corner == SENTINEL.rgb(), (
        "something painted the panel's top-left corner, so the rounding "
        "cannot be seen")
