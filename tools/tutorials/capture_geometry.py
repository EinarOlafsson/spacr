"""Describe visible Qt controls in the actual recording window's coordinates."""


def foreground_dialogs(dialogs, active_modal=None, active_popup=None):
    """Draw the actual active modal and its popup above background windows."""
    result = list(dialogs)
    for active in (active_modal, active_popup):
        if active is not None and any(item is active for item in result):
            result = [item for item in result if item is not active] + [active]
    return result


def capture_rect(widget, window):
    """Return a clipped client rectangle, or None for a hidden/off-frame widget."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QWidget

    if not widget.isVisible():
        return None
    point = widget.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
    x, y = max(0, point.x()), max(0, point.y())
    # SaveFigureDialog has public width/height spin-box attributes. Call
    # native QWidget accessors rather than those unrelated instance names.
    right = min(QWidget.width(window), point.x() + QWidget.width(widget))
    bottom = min(QWidget.height(window), point.y() + QWidget.height(widget))
    if right <= x or bottom <= y:
        return None
    return [x, y, right - x, bottom - y]
