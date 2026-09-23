"""Keep the operating system's standard arrow in every interactive context."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget


def arrow_cursor(active=False):
    """Return the native OS arrow without customizing its artwork or size.

    :param active: retained for compatibility; both states use the OS arrow.
    :returns: a native arrow that follows the user's OS pointer preferences.
    """
    return QCursor(Qt.ArrowCursor)


class _CursorPolicy(QObject):
    """Preserve gestures while suppressing application cursor substitutions."""

    def __init__(self, application):
        super().__init__(application)
        self._changing = False

    def eventFilter(self, watched, event):
        if self._changing or event.type() not in (
                QEvent.CursorChange, QEvent.Enter, QEvent.Show, QEvent.MouseMove):
            return False
        if not isinstance(watched, QWidget):
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
    app.installEventFilter(policy)
    for widget in QApplication.allWidgets():
        policy.eventFilter(widget, QEvent(QEvent.CursorChange))
    return True
