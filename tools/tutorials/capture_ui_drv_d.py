"""Helpers for the drv-d lesson drivers (lessons 50-81): clips that contain their own actions.

``capture_ui.clip`` wraps a whole block. Some scenes cannot sit in a ``with``
(a scene whose recording must span a click that opens a modal dialog, with the
steps inside the dialog run from its own event loop), so these three calls
split it: :func:`begin` before the first action, :func:`snap` where the old
driver captured inside the dialog, :func:`finish` once the dialog is closed.
Outside the GNOME host they reproduce the old order exactly (one still capture).
"""
from __future__ import annotations

import capture_ui


def begin(capture, settle, name, lead=0.8):
    """Host: start recording ``name`` now (returns True). Xvfb: nothing, returns False."""
    started = bool(capture_ui.live(capture) and capture.begin(name))
    if started:
        capture_ui.wait(capture, lead, settle)
    return started


def snap(capture, name):
    """Xvfb: the still capture the old driver took here. Host: nothing (the clip is running)."""
    if not capture_ui.live(capture):
        capture(name)


def finish(capture, started):
    """Host: stop the clip begun by :func:`begin` (it holds to its planned duration)."""
    if started:
        capture.end()


def hover_rows(capture, view, rows=4, dwell=1.0):
    """Host: the pointer rests on the first rows of a list/tree/table, top to bottom."""
    if not capture_ui.live(capture) or view is None or not view.isVisible():
        return
    h = capture_ui._h()
    model = view.model()
    if model is None:
        return
    port = view.viewport()
    shown = 0
    for row in range(model.rowCount(view.rootIndex())):
        rect = view.visualRect(model.index(row, 0, view.rootIndex()))
        if rect.isValid() and port.rect().contains(rect.center()):
            h.hover(port, rect.center())
            h.pause(capture, dwell)
            shown += 1
            if shown >= rows:
                return


def drag_scroll_bar_horizontal(capture, bar, value):
    """Host: drag the handle of a horizontal scroll bar towards ``value`` (a real drag).

    ``capture_ui.scroll_to_value`` / ``capture_home_live.drag_scroll_bar`` assume a
    vertical bar; this is the horizontal counterpart. Xvfb: ``setValue``.
    """
    value = max(bar.minimum(), min(bar.maximum(), int(value)))
    if not capture_ui.live(capture):
        bar.setValue(value)
        return
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    h = capture_ui._h()
    for _ in range(4):
        if abs(bar.value() - value) <= max(2, bar.singleStep()):
            return
        option = QStyleOptionSlider()
        bar.initStyleOption(option)
        handle = bar.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarSlider, bar)
        groove = bar.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarGroove, bar)
        travel = max(1, groove.width() - handle.width())
        span = max(1, bar.maximum() - bar.minimum())
        dx = int(round((value - bar.value()) * travel / span))
        if not dx:
            return
        h.drag(capture, bar, handle.center(), (dx, 0))
