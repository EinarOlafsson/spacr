"""Keep the operating system's standard arrow in every interactive context."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget

from ..gil_priority import _watch_application_events


def arrow_cursor(active=False):
    """Return the native OS arrow without customizing its artwork or size.

    :param active: retained for compatibility; both states use the OS arrow.
    :returns: a native arrow that follows the user's OS pointer preferences.
    """
    return QCursor(Qt.ArrowCursor)


_POINTER_EVENTS = frozenset({
    QEvent.CursorChange, QEvent.Enter, QEvent.Show, QEvent.MouseMove})


class _CursorPolicy(QObject):
    """Preserve gestures while suppressing application cursor substitutions."""

    def __init__(self, application):
        """Bind the native-arrow policy to its application and guard recursive cursor events."""
        super().__init__(application)
        self._changing = False

    def eventFilter(self, watched, event):
        """Restore the native arrow on pointer events without consuming the original gesture.

        A ``Show`` of a widget that never had a cursor set on it is passed
        over: such a widget shows its parent's cursor, which this policy has
        already answered for, and a module screen's first show delivers one
        ``Show`` per child -- several thousand -- that used to read and
        compare a cursor each (7 % of a slow module's first open).
        """
        if self._changing:
            return False
        kind = event.type()
        if kind not in _POINTER_EVENTS:
            return False
        if not isinstance(watched, QWidget):
            return False
        if (kind == QEvent.Show
                and not watched.testAttribute(Qt.WA_SetCursor)
                and QApplication.overrideCursor() is None):
            return False
        self._changing = True
        try:
            if watched.cursor().shape() != Qt.ArrowCursor:
                watched.setCursor(arrow_cursor())
            override = QApplication.overrideCursor()
            if override is not None and override.shape() != Qt.ArrowCursor:
                QApplication.changeOverrideCursor(arrow_cursor())
        except RuntimeError:
            pass
        finally:
            self._changing = False
        return False


def install_cursor_policy(application=None):
    """Install the native-arrow policy once; leave all gestures unchanged."""
    app = application or QApplication.instance()
    if app is None or getattr(app, '_spacr_cursor_policy', None) is not None:
        return False
    policy = _CursorPolicy(app)
    app._spacr_cursor_policy = policy
    _watch_application_events(app, policy, _POINTER_EVENTS)
    for widget in QApplication.allWidgets():
        policy.eventFilter(widget, QEvent(QEvent.CursorChange))
    return True
