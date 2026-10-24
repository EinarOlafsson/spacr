"""Helpers for the lesson 09-23 scene drivers: a clip that spans a dialog, typed text, a modal wait.

These sit beside :mod:`capture_ui` (which another group edits) and use only its
public functions. In the capture host every call is a human action at the real
pointer; elsewhere (the Xvfb recorder) each does what the driver did before.
"""
from __future__ import annotations

import capture_ui


class Rec:
    """A scene clip started BEFORE the actions and stopped where the frame used to be taken.

    ``with capture_ui.clip(...)`` cannot span code that runs inside a modal
    dialog's own event loop (a callback fires from ``exec()``). ``Rec`` can:
    ``start()`` before the click that opens the dialog, ``stop()`` inside the
    dialog callback once the narrated state shows.

    Host: ``start`` begins the recording, ``stop`` holds the planned duration
    and ends it (a body longer than the plan makes a longer clip, never shorter).
    Xvfb: ``start`` does nothing, ``stop`` takes the old still ``capture(name)``.
    """

    def __init__(self, capture, name, settle, lead=0.8, **capture_kwargs):
        self.capture, self.name, self.settle = capture, name, settle
        self.lead, self.kwargs, self.started = lead, capture_kwargs, False

    def start(self):
        if capture_ui.live(self.capture) and not self.started:
            self.started = bool(self.capture.begin(self.name))
            if self.started:
                capture_ui.wait(self.capture, self.lead, self.settle)
        return self

    def point(self, *widgets, dwell=1.2):
        capture_ui.point_at(self.capture, *widgets, dwell=dwell)

    def stop(self):
        if capture_ui.live(self.capture):
            if self.started:
                self.capture.end()
                self.started = False
        else:
            self.capture(self.name, **self.kwargs)

    def abort(self):
        """Close a host clip after a failure, without taking a still."""
        if self.started:
            self.capture.end()
            self.started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, trace):
        if kind is None:
            self.stop()
        else:
            self.abort()
        return False


def type_text(capture, edit, value, *, tab=True):
    """Replace the text of a line edit by typing it: host through the pointer and keyboard."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if capture_ui.live(capture):
        capture_ui.set_text(capture, edit, str(value), commit=tab)
        return
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
    if str(value):
        QTest.keyClicks(edit, str(value))
    else:
        QTest.keyClick(edit, Qt.Key_Backspace)
    if tab:
        QTest.keyClick(edit, Qt.Key_Tab)


def search_typed(capture, bar, query):
    """Type ``query`` into a settings search box: host through the pointer, Xvfb as the driver did."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if capture_ui.live(capture):
        capture_ui.search_settings(capture, bar, query)
        return
    bar._input.setFocus()
    QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(bar._input, query)


def modal_wait(capture, delay_ms, handler, timeout=60.0):
    """Run ``handler`` once the modal opened by the next click shows (see ``capture_ui.when_modal``)."""
    capture_ui.when_modal(capture, delay_ms, handler, timeout=timeout)


def watchdog_ms(capture, base_ms):
    """A dialog watchdog: the pointer needs real travel time in the host."""
    return base_ms * 5 if capture_ui.live(capture) else base_ms


def hover(capture, widget, settle, seconds=0.8):
    """Rest the pointer on a control (host: it glides there; Xvfb: a synthetic move)."""
    from PySide6.QtTest import QTest
    QTest.mouseMove(widget)
    settle(seconds)


def dismiss_menu(capture, menu):
    """Close an open menu: Escape on the real keyboard in the host, else ``close()``."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if capture_ui.live(capture):
        QTest.keyClick(menu, Qt.Key_Escape)
    else:
        menu.close()


def pick_list_row(capture, view, row):
    """Click the check box of a checkable list row (scrolled into view by the wheel in the host)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    item = view.item(row)
    if capture_ui.live(capture):
        capture_ui._wheel_item_into_view(capture, view, item)
    else:
        view.scrollToItem(item)
    rect = view.visualItemRect(item)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 10, rect.center().y()))


def table_row_to_top(capture, table, row):
    """Scroll a table so ``row`` is at the top: wheel turns in the host, ``scrollToItem`` elsewhere."""
    from PySide6.QtWidgets import QAbstractItemView
    if not capture_ui.live(capture):
        table.scrollToItem(table.item(row, 0), QAbstractItemView.PositionAtTop)
        return
    bar = table.verticalScrollBar()
    if table.verticalScrollMode() == QAbstractItemView.ScrollPerItem:
        target = row
    else:
        target = bar.value() + table.rowViewportPosition(row)
    capture_ui.scroll_to_value(capture, table, target)


def table_into_view(capture, table, row, column=0):
    """Bring a table cell into view: wheel turns in the host, ``scrollTo`` elsewhere."""
    if not capture_ui.live(capture):
        table.scrollTo(table.model().index(row, column))
        return
    h = capture_ui._h()
    index = table.model().index(row, column)
    for _ in range(80):
        rect = table.visualRect(index)
        port = table.viewport().rect()
        if port.contains(rect.center()):
            return
        bar = table.verticalScrollBar()
        before = bar.value()
        h.wheel(capture, table.viewport(), 1 if rect.center().y() > port.bottom() else -1, port.center())
        if bar.value() == before:
            return
