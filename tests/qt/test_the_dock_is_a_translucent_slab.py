"""The dock is a translucent rounded box, not a black column.

Asked for on 2026-09-03: "replace the black box behind thicons with a
translucent box with rounded edges. the translucent box should also highlight
upon hover."

THE BOX IS THE DOCK PANEL, and finding that took two attempts. The first read
it as one plate per row, which is a reasonable reading of "highlights upon
hover" and is also implemented -- see `_DockRow._paint_plate`. It was not what
was reported, because on the dark theme `theme.DOCK_FILL` is literally
``"transparent"``: the dock column showed the page's pure black straight
through, so what sat behind the icons was a black box the width of the dock
and the height of the window. That is the box, and nothing per-row could
replace it.

MEASURED, one 1440x900 MainWindow on the dark theme, the dock rendered over
its own page colour:

    behind the dock   #000000
    slab at rest      #0d0d0d  (+13)
    slab hovered      #171717  (+10 more)
    corner outside    #000000  (unpainted, so the corners are round)

Painted rather than styled: a `border-radius` in QSS reaches a plain
``QWidget`` only with ``WA_StyledBackground`` set, and that attribute makes
the widget opaque -- which is the one thing a translucent slab must not be.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPixmap

import spacr.qt.theme as theme_module
from spacr.qt.app import MainWindow, Sidebar


@pytest.fixture(params=["dark", "light", "space", "cell"])
def dock(request, qtbot, qt_theme_applied, monkeypatch):
    """One themed window's dock, with the palette forced to `request.param`.

    `active_palette` reads the user's saved preference, so a test that only
    applied a stylesheet would measure whatever theme the developer last
    chose. Patched, not written to settings.
    """
    name = request.param
    monkeypatch.setattr(theme_module, "active_palette",
                        lambda: theme_module.palette_for(name))
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1440, 900)
    win.show()
    win._sidebar.theme_name = name
    yield win._sidebar
    win.hide()


def _render(dock, name):
    """The dock painted over its own page colour, as the window shows it."""
    page = QColor(theme_module.palette_for(name)["bg"])
    shot = QPixmap(dock.size())
    shot.fill(page)
    dock.render(shot)
    return shot.toImage()


def _step(a, b):
    return max(abs(a.red() - b.red()), abs(a.green() - b.green()),
               abs(a.blue() - b.blue()))


def test_the_dock_draws_a_translucent_slab_over_the_page(dock):
    """Visible against the page, and nowhere near opaque."""
    name = dock.theme_name
    image = _render(dock, name)
    box = dock.plate_rect()
    page = QColor(theme_module.palette_for(name)["bg"])
    inside = image.pixelColor(int(box.center().x()), int(box.top()) + 30)

    assert _step(inside, page) >= 6, (
        f"the slab {inside.name()} is invisible against the page "
        f"{page.name()}")
    assert _step(inside, page) < 90, (
        f"the slab {inside.name()} is opaque, not translucent")


def test_the_slab_has_corners_the_page_shows_through(dock):
    """Rounded edges: the dock's own corner pixel is NOT the slab."""
    name = dock.theme_name
    image = _render(dock, name)
    page = QColor(theme_module.palette_for(name)["bg"])
    corner = image.pixelColor(0, 0)
    assert _step(corner, page) == 0, (
        f"the dock's corner is {corner.name()}, not the page {page.name()} "
        "-- the slab is square, or it is not inset")
    box = dock.plate_rect()
    assert box.left() > 0 and box.top() > 0, "the slab is flush with the edge"
    assert Sidebar.PLATE_RADIUS_PX > 0


def test_the_slab_highlights_while_the_pointer_is_in_the_dock(dock):
    """"the translucent box should also highlight upon hover"."""
    name = dock.theme_name
    box = dock.plate_rect()
    point = (int(box.center().x()), int(box.top()) + 30)

    dock._pointer_inside = False
    rest = _render(dock, name).pixelColor(*point)
    dock._pointer_inside = True
    hot = _render(dock, name).pixelColor(*point)
    dock._pointer_inside = False

    assert _step(hot, rest) >= 4, (
        f"the slab does not brighten on hover: {rest.name()} -> {hot.name()}")


def test_leaving_the_dock_puts_the_slab_back(dock, qapp):
    """A slab left bright is a dock that looks hovered when nothing is."""
    from PySide6.QtCore import QEvent

    dock._pointer_inside = True
    qapp.sendEvent(dock, QEvent(QEvent.Type.Leave))
    assert dock._pointer_inside is False
