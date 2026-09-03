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


def _render(dock, name, background=None):
    """The dock painted over its page colour, WITHOUT the window fill.

    `QWidget.render` defaults to `DrawWindowBackground | DrawChildren`, and
    the window fill comes from the PALETTE whatever `paintEvent` does -- so a
    plain render reports the palette's Window colour for a widget that paints
    a translucent slab, which is the very thing this file exists to tell
    apart. Dropping the flag leaves only what the dock and its children draw.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QRegion
    from PySide6.QtWidgets import QWidget

    page = QColor(background or theme_module.palette_for(name)["bg"])
    shot = QPixmap(dock.size())
    shot.fill(page)
    dock.render(shot, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
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


def test_the_slab_does_not_react_to_the_pointer(dock, qapp):
    """ONE STATE. It brightened while the pointer was in the dock for a few
    hours on 2026-09-03 and was asked for gone with everything else: "i just
    want the transparent dock holder with rounded edges, the icons and when
    hovered the icons turn blue and you see the text which is also blue.
    nothing else."

    It also cost a repaint of the WHOLE dock on every Enter and Leave, and a
    Leave arrives each time the pointer crosses from the column's own surface
    onto one of its rows -- see
    `test_the_dock_does_not_relayout_on_hover.py`, which counts them.
    """
    from PySide6.QtCore import QEvent

    name = dock.theme_name
    box = dock.plate_rect()
    point = (int(box.center().x()), int(box.top()) + 30)
    before = _render(dock, name).pixelColor(*point)

    qapp.sendEvent(dock, QEvent(QEvent.Type.Enter))
    assert _render(dock, name).pixelColor(*point).name() == before.name(), (
        "the slab changed when the pointer entered the dock")
    qapp.sendEvent(dock, QEvent(QEvent.Type.Leave))
    assert _render(dock, name).pixelColor(*point).name() == before.name()
    assert not hasattr(Sidebar, "PLATE_ALPHA_HOVER"), (
        "the slab has a hover state again")


def test_nothing_is_painted_over_the_slab(dock):
    """The slab is the ONLY box in the dock.

    THE DEFECT THIS PINS was reported three times and measured on
    2026-09-03: every row ran `drawControl(CE_PushButton)` first, on the
    belief that QStyleSheetStyle would paint its QSS background. It never
    did -- a `paintEvent` that goes straight to `drawControl` skips the pass
    a stylesheet's background is filled in -- so what was rendered was a
    NATIVE button panel from the palette's Button role: an opaque `#161719`
    rectangle behind every icon, drawn OVER the translucent slab.

    Checked by forcing the slab to a colour nothing else uses and asking
    whether it survives to the middle of a row. Sampling for "not the page
    colour" would have passed all along -- the button panel is not the page
    colour either.
    """
    from PySide6.QtGui import QColor, QPixmap

    name = dock.theme_name
    row = next(r for r in dock._items
               if str(r.property("navKey")) == "mask")
    page = QColor(theme_module.palette_for(name)["bg"])

    original = Sidebar.PLATE_ALPHA
    marker = "#ff00ff"
    patched = dict(theme_module.palette_for(name))
    patched["fg"] = marker
    real = theme_module.active_palette
    theme_module.active_palette = lambda: patched
    Sidebar.PLATE_ALPHA = 1.0
    try:
        image = _render(dock, name, page.name())
        # To the right of the icon, well inside the row and inside the slab.
        top_left = row.mapTo(dock, row.rect().topLeft())
        x = top_left.x() + row.width() - 30
        y = top_left.y() + row.height() // 2
        seen = image.pixelColor(x, y)
    finally:
        Sidebar.PLATE_ALPHA = original
        theme_module.active_palette = real

    assert seen.name().lower() == marker, (
        f"something paints {seen.name()} over the slab in the middle of a "
        f"row -- the slab was forced to {marker}")


def test_leaving_the_dock_leaves_no_row_looking_hovered(dock, qapp):
    """The one thing `leaveEvent` still does."""
    from PySide6.QtCore import QEvent

    for row in dock._items[:3]:
        row._hovered = True
    qapp.sendEvent(dock, QEvent(QEvent.Type.Leave))
    still_hot = [str(r.property("navKey")) for r in dock._items
                 if getattr(r, "_hovered", False)]
    assert not still_hot, f"these rows still look hovered: {still_hot}"
