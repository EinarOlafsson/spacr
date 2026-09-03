"""Hovering the dock must not invalidate its layout.

Reported 2026-09-03: "an issue i have noticed on my home computer but not on
my work computer is that hovering on the dock induces flickering. and only if
i hover."

MEASURED, one 1440x900 MainWindow with the dark stylesheet, pointer swept
across five rows and off the dock:

    setFixedHeight calls   355  ->  0
    updateGeometry calls   355  ->  0

355 is 71 rows x 5 dock Leave events. `Sidebar.leaveEvent` called
`_rest_every_icon()`, which walks every row through `_place_icon` and so
through `_DockRow.set_row_height` -- `setFixedHeight` plus `updateGeometry`
on each, whether or not the height had changed. A dock Leave arrives every
time the pointer crosses from the column's own surface onto one of its rows,
so an ordinary sweep asked for a full relayout of the dock per row. Whether
that presents as visible flicker depends on the compositor, which is why one
machine showed it and another did not.

Counted rather than screenshotted on purpose: a flicker is two frames, and a
test that grabs frames measures the compositor. What the application controls
is whether it asks for the relayout, so that is what this asserts.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.app import MainWindow, _DockRow


@pytest.fixture(scope="module")
def window(qapp, qt_theme_applied):
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    qapp.processEvents()
    yield win
    win.hide()
    qapp.processEvents()


def _sweep(dock, rows, qapp):
    """Move the pointer over each row, then off the dock, as Qt would."""
    for row in rows:
        local = QPointF(row.width() / 2, row.height() / 2)
        globally = QPointF(row.mapToGlobal(QPoint(int(local.x()),
                                                  int(local.y()))))
        QApplication.sendEvent(row, QEnterEvent(local, local, globally))
        QApplication.sendEvent(row, QMouseEvent(
            QEvent.Type.MouseMove, local, globally,
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier))
        QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))
        # The dock's OWN Leave, which is what a pointer crossing from the
        # column onto a row delivers and what used to do the damage.
        QApplication.sendEvent(dock, QEvent(QEvent.Type.Leave))
    qapp.processEvents()


def test_a_hover_sweep_asks_for_no_geometry_change(window, qapp,
                                                   monkeypatch):
    """Not one `setFixedHeight` or `updateGeometry` across the whole sweep."""
    dock = window._sidebar
    rows = [r for r in dock._items if not r.isHidden()][:5]
    assert len(rows) == 5, "need five visible dock rows"

    calls = {"height": 0, "geometry": 0}
    real_height = _DockRow.setFixedHeight
    real_geometry = _DockRow.updateGeometry

    def counted_height(self, value):
        calls["height"] += 1
        return real_height(self, value)

    def counted_geometry(self):
        calls["geometry"] += 1
        return real_geometry(self)

    monkeypatch.setattr(_DockRow, "setFixedHeight", counted_height)
    monkeypatch.setattr(_DockRow, "updateGeometry", counted_geometry)
    _sweep(dock, rows, qapp)

    assert calls == {"height": 0, "geometry": 0}, (
        f"hovering the dock asked for a relayout: {calls}")


def test_the_sweep_still_leaves_no_row_looking_hovered(window, qapp):
    """The behaviour `leaveEvent` is actually for, kept.

    A dock Leave that left one row blue with its name showing is a row that
    looks hovered when nothing is -- which is what the removed icon reset
    used to be paired with.
    """
    dock = window._sidebar
    rows = [r for r in dock._items if not r.isHidden()][:5]
    _sweep(dock, rows, qapp)
    still_hot = [str(r.property("navKey")) for r in dock._items
                 if getattr(r, "_hovered", False)]
    assert not still_hot, f"these rows still look hovered: {still_hot}"


def test_a_font_scale_change_still_resizes_every_row(window, qapp):
    """The guards must not make a real change a no-op.

    `set_row_height` returns early on an unchanged height, and
    `_place_icon` skips an unchanged icon size. A font-scale change arrives
    through `refresh_icons` and has to get through both.
    """
    from spacr.qt import preferences as prefs

    dock = window._sidebar
    before = {r: (r.height(), r.iconSize().width()) for r in dock._items}
    original = prefs.get_font_scale()
    try:
        prefs.set_font_scale(min(2.0, original * 1.5))
        dock._forget_icon_sizes()
        dock.refresh_icons()
        qapp.processEvents()
        after = {r: (r.height(), r.iconSize().width()) for r in dock._items}
        assert after != before, (
            "a font-scale change did not move a single row")
        grew = [r for r in dock._items if after[r][0] > before[r][0]]
        assert grew, "no row got taller at a larger font scale"
    finally:
        prefs.set_font_scale(original)
        dock._forget_icon_sizes()
        dock.refresh_icons()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# The blink
# ---------------------------------------------------------------------------

def test_a_dock_leave_does_not_unhover_the_row_under_the_pointer(window,
                                                                 qapp):
    """The blink, and it was never a repaint problem.

    Reported 2026-09-03: "if i hover quickly an element in the dock it blinks
    blue a bunch of times then stays blue after a while."

    Qt sends a widget a Leave whenever a CHILD takes the pointer. So moving
    from the dock's own surface onto one of its rows delivers Enter to the
    row and Leave to the dock -- and `Sidebar.leaveEvent` then cleared the
    hover ink off every row, including the one the pointer had just landed
    on. The ink went on with the Enter and off with the Leave, over and over;
    it "stayed blue" whenever the two happened to arrive the other way round.

    MEASURED before the fix: `Enter(row)` left `_hovered` True and the
    `Leave(Sidebar)` immediately after it set it back to False.
    """
    from PySide6.QtGui import QCursor

    dock = window._sidebar
    row = next(r for r in dock._items if not r.isHidden())
    # The cursor really is over the dock during this sequence; that is what
    # tells a child-took-the-pointer Leave from a left-the-dock one.
    QCursor.setPos(dock.mapToGlobal(QPoint(dock.width() // 2, 200)))
    qapp.processEvents()

    local = QPointF(row.width() / 2, row.height() / 2)
    QApplication.sendEvent(row, QEnterEvent(
        local, local,
        QPointF(row.mapToGlobal(QPoint(int(local.x()), int(local.y()))))))
    assert row._hovered is True, "the row did not take the hover at all"

    QApplication.sendEvent(dock, QEvent(QEvent.Type.Leave))
    assert row._hovered is True, (
        "a dock Leave un-hovered the row the pointer is sitting on -- "
        "this is the blink")


def test_the_ink_moves_cleanly_from_one_row_to_the_next(window, qapp):
    """Exactly one row is inked while the pointer crosses the dock."""
    from PySide6.QtGui import QCursor

    dock = window._sidebar
    rows = [r for r in dock._items if not r.isHidden()][:4]
    # The window is module-scoped, so start from a known state rather than
    # from whatever the previous test left hovered.
    for row in dock._items:
        row._hovered = False
    QCursor.setPos(dock.mapToGlobal(QPoint(dock.width() // 2, 200)))
    qapp.processEvents()

    for row in rows:
        # Qt sends the row being left its own Leave before the next Enter;
        # this loop does the same so it measures the handover rather than a
        # sequence Qt never delivers.
        for other in dock._items:
            other._hovered = False
        local = QPointF(row.width() / 2, row.height() / 2)
        QApplication.sendEvent(row, QEnterEvent(
            local, local,
            QPointF(row.mapToGlobal(QPoint(int(local.x()),
                                           int(local.y()))))))
        QApplication.sendEvent(dock, QEvent(QEvent.Type.Leave))
        inked = [str(r.property("navKey")) for r in dock._items
                 if getattr(r, "_hovered", False)]
        assert inked == [str(row.property("navKey"))], (
            f"hovering {row.property('navKey')!r} left {inked} inked")
        QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))


def test_the_pointer_really_leaving_still_clears_every_row(window, qapp):
    """The case `leaveEvent` exists for, kept.

    A pointer can leave the dock off the bottom row onto the empty stretch
    below it, where no other row will ever be entered -- so a row left inked
    there would look hovered with nothing hovering it.
    """
    from PySide6.QtGui import QCursor

    dock = window._sidebar
    for row in dock._items[:3]:
        row._hovered = True
    # Well outside the dock.
    QCursor.setPos(dock.mapToGlobal(QPoint(dock.width() + 400, 400)))
    qapp.processEvents()
    QApplication.sendEvent(dock, QEvent(QEvent.Type.Leave))
    still = [str(r.property("navKey")) for r in dock._items
             if getattr(r, "_hovered", False)]
    assert not still, f"these rows stayed inked after the pointer left: {still}"
