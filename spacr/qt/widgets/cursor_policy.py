"""Keep one arrow silhouette and use blue to indicate interactive cursors."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from ..theme import active_palette

_CURSORS = {}


def arrow_cursor(active=False):
    """Return the shared arrow, blue for interactive contexts.

    :param active: whether the original cursor requested a different shape.
    :returns: an arrow with opaque fill and a white outline in dark themes,
        or black outline in light themes. Idle and active silhouettes match.
    """
    light = QColor(active_palette()['fg']).lightness() < 128
    key = bool(active), light
    if key not in _CURSORS:
        pixmap = QPixmap(28, 34)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        shape = QPainterPath(QPointF(4, 3))
        for x, y in ((4, 25), (10, 20), (15, 30), (19, 28), (14, 18), (23, 18)):
            shape.lineTo(x, y)
        shape.closeSubpath()
        rim = QColor('black' if light else 'white')
        fill = QColor('#168cff') if active else QColor('white' if light else 'black')
        painter.setPen(QPen(rim, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(fill)
        painter.drawPath(shape)
        painter.end()
        _CURSORS[key] = QCursor(pixmap, 4, 3)
    return _CURSORS[key]


def _is_ours(cursor):
    """Recognize cached arrows so applying the policy cannot recurse."""
    if cursor.shape() != Qt.BitmapCursor:
        return False
    key = cursor.pixmap().cacheKey()
    return any(value.pixmap().cacheKey() == key for value in _CURSORS.values())


class _CursorPolicy(QObject):
    """Normalize widget cursors once, including lazily constructed popups."""

    def __init__(self, application):
        """Own the application filter and guard nested CursorChange events."""
        super().__init__(application)
        self._changing = False

    def eventFilter(self, watched, event):
        """Preserve gestures while replacing cursor shapes with arrow colors."""
        if self._changing or event.type() not in (QEvent.CursorChange, QEvent.Enter, QEvent.Show, QEvent.PaletteChange):
            return False
        if not isinstance(watched, QWidget):
            return False
        self._changing = True
        try:
            cursor = watched.cursor()
            if not _is_ours(cursor):
                active = cursor.shape() != Qt.ArrowCursor
                watched.setProperty('spacrCursorActive', active)
                watched.setCursor(arrow_cursor(active))
            else:
                active = next(key[0] for key, value in _CURSORS.items()
                              if value.pixmap().cacheKey() == cursor.pixmap().cacheKey())
                watched.setProperty('spacrCursorActive', active)
                if event.type() == QEvent.PaletteChange:
                    watched.setCursor(arrow_cursor(active))
            override = QApplication.overrideCursor()
            if override is not None and not _is_ours(override):
                QApplication.changeOverrideCursor(arrow_cursor(override.shape() != Qt.ArrowCursor))
        except RuntimeError:
            pass
        finally:
            self._changing = False
        return False


def install_cursor_policy(application=None):
    """Install the application-wide fixed-arrow policy once.

    :param application: QApplication; defaults to the current instance.
    :returns: whether a new filter was installed. Widget gestures and native
        compositor operations remain owned by their existing handlers.
    """
    app = application or QApplication.instance()
    if app is None or getattr(app, '_spacr_cursor_policy', None) is not None:
        return False
    policy = _CursorPolicy(app)
    app._spacr_cursor_policy = policy
    app.installEventFilter(policy)
    for widget in QApplication.allWidgets():
        policy.eventFilter(widget, QEvent(QEvent.CursorChange))
    return True
