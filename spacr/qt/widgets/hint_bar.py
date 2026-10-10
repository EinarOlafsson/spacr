"""Window-level, non-modal hover help for Qt controls.

:class:`HintBar` keeps four visible lines at a fixed height. Longer help can
be scrolled within that space while the pointer is over a widget. On pointer
leave, it restores the default message only when that widget's text is still
displayed, avoiding an intermediate reset between adjacent controls.

:meth:`HintBar.explain` registers either explicit text or the widget's current
tooltip. Successful registration clears the tooltip, supplies the same text
as the accessible description only when no accessible description is already
present, and installs the bar as an event filter. Source strings are
translated when displayed. Widgets without available text are not registered.

Use :func:`hint_bar_of` to locate the :class:`HintBar` in a widget's top-level
window, or :func:`explain_through_the_bar` to register a widget only when such
a bar exists. The bar retains the registration mapping and event filter for
its lifetime.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QTextDocument, QTextOption
from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout, QWidget

#: What the bar says when nothing is under the pointer.
DEFAULT_HINT = "Hover a control to see what it does."

#: The objectName the stylesheet selects on, shared with the Home screen's
#: bar so the two are one thing wearing one style rather than two that
#: happen to look alike today.
BAR_NAME = "HintBar"


class _HintResizeHandle(QFrame):
    """Move the top edge of a help strip without resizing its window."""

    def __init__(self, bar: HintBar) -> None:
        """Keep the strip whose height the drag changes."""
        super().__init__(bar)
        self._bar = bar
        self._drag = None
        self.setObjectName("HintBarResizeHandle")
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setFixedHeight(8)
        self.setCursor(Qt.SizeVerCursor)

    def mousePressEvent(self, event) -> None:
        """Start a vertical drag at the currently reserved height."""
        if event.button() == Qt.LeftButton:
            self._drag = (event.globalPosition().y(), self._bar.height())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Trade tab space for help space within the visible dialog."""
        if self._drag is not None:
            start_y, start_height = self._drag
            self._bar._set_manual_height(
                start_height + round(start_y - event.globalPosition().y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Persist the chosen height after the user releases the grip."""
        if self._drag is not None and event.button() == Qt.LeftButton:
            self._drag = None
            self._bar.helpHeightCommitted.emit(self._bar.height())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class HintBar(QLabel):
    """A fixed help strip and the register of what each widget should say.

    :param default: what the line says when nothing is hovered. It is
        restored whenever a widget with no registered hint takes the pointer,
        so it should read as a prompt rather than as a blank.
    :param parent: parent widget.
    """

    helpHeightCommitted = Signal(int)

    def __init__(self, default: str = DEFAULT_HINT,
                 parent: Optional[QWidget] = None) -> None:
        """Build the help strip that replaces per-control tooltips.

        Four lines stay visible: moving between controls cannot change the
        dialog's layout. Longer or translated help scrolls inside the strip
        without losing its full text or accessible description.

        :param default: what to show with nothing hovered.
        :param parent: parent widget, or ``None``.
        """
        super().__init__("", parent)
        self._default = default
        self._hints: Dict[QWidget, str] = {}
        self.setObjectName(BAR_NAME)
        self.setAlignment(Qt.AlignJustify | Qt.AlignVCenter)
        self.setWordWrap(True)
        self._visible_text = default
        self.setAccessibleName(default)
        self._manual_height = None
        self._resize_handle = _HintResizeHandle(self)
        try:
            from ..i18n import tr
            resize_tip = tr("Drag this edge to make the help area taller or shorter.")
        except Exception:
            resize_tip = "Drag this edge to make the help area taller or shorter."
        self._resize_handle.setAccessibleName(resize_tip)
        self._resize_handle.setAccessibleDescription(resize_tip)
        self._view = QTextEdit(self)
        self._view.setObjectName("HintBarText")
        self._view.setReadOnly(True)
        self._view.setAcceptRichText(False)
        self._view.setFocusPolicy(Qt.StrongFocus)
        self._view.setTabChangesFocus(True)
        self._view.setFrameShape(QFrame.NoFrame)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self._view.document().setDocumentMargin(0)
        self._view.setAlignment(Qt.AlignJustify)
        self._view.setStyleSheet(
            "background: transparent; border: none; padding: 0px; margin: 0px;")
        inner = QVBoxLayout(self)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)
        inner.addWidget(self._resize_handle)
        inner.addWidget(self._view)
        self._view.setPlainText(default)
        self.setFixedHeight(self._minimum_help_height())

    def _minimum_help_height(self) -> int:
        """Leave four painted text rows and the resize edge visible."""
        lines = max(self.fontMetrics().lineSpacing() * 4, self._four_painted_rows())
        margins = self.contentsMargins()
        handle = getattr(self, "_resize_handle", None)
        edge = handle.height() if handle is not None else 8
        return max(28, lines + 12,
                   lines + margins.top() + margins.bottom() + edge)

    def _four_painted_rows(self) -> int:
        """Measure four laid-out rows, which can exceed four rounded line spacings."""
        view = getattr(self, "_view", None)
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(view.font() if view is not None else self.font())
        document.setPlainText("\n".join(("Xg",) * 4))
        frame = 0 if view is None else 2 * view.frameWidth()
        return math.ceil(document.size().height()) + frame

    def _maximum_help_height(self) -> int:
        """Keep the preceding page and the dialog's action buttons visible."""
        minimum = self._minimum_help_height()
        window = self.window()
        layout = window.layout() if window is not self else None
        index = layout.indexOf(self) if layout is not None else -1
        above = layout.itemAt(index - 1).widget() if index > 0 else None
        if above is None or not self.isVisible():
            return minimum
        kept = max(above.minimumSizeHint().height(), 100)
        return max(minimum, self.height() + max(0, above.height() - kept))

    def _set_manual_height(self, height: int) -> None:
        """Bound a drag by actual space above the footer."""
        wanted = max(self._minimum_help_height(), int(height))
        self.setFixedHeight(min(wanted, self._maximum_help_height()))
        self._manual_height = self.height()

    def showEvent(self, event) -> None:
        """Restore a saved height after the Preferences layout is measured.

        :param event: the Qt show event.
        """
        super().showEvent(event)
        self._sync_help_font()
        self.setFixedHeight(max(self._minimum_help_height(),
                                self._manual_height or 0))
        if self._manual_height is not None:
            QTimer.singleShot(0, self, lambda: self._set_manual_height(
                self._manual_height) if self.isVisible() else None)

    def setText(self, text: str) -> None:
        """Show all help in the fixed strip, scrolling when it exceeds four lines.

        :param text: the complete translated help sentence.
        """
        self._visible_text = text
        self.setAccessibleName(text)
        self._view.setPlainText(text)
        self._view.verticalScrollBar().setValue(0)

    def text(self) -> str:
        """Return the complete sentence currently shown in the strip."""
        return self._visible_text

    def changeEvent(self, event):
        """Keep the reserved four lines in sync with the painted font.

        :param event: the Qt font or style change event.
        :returns: None.
        """
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.ApplicationFontChange,
                            QEvent.StyleChange):
            self._sync_help_font()
            self.setFixedHeight(max(self._minimum_help_height(),
                                    getattr(self, "_manual_height", None) or 0))

    def _sync_help_font(self) -> None:
        """Paint the help at the strip's font size despite inherited editor rules."""
        view = getattr(self, "_view", None)
        if view is not None:
            view.setFont(self.font())
            pixels = self.fontInfo().pixelSize()
            view.setStyleSheet(
                "background: transparent; border: none; padding: 0px; margin: 0px;"
                f"font-size: {pixels}px;")


    def explain(self, widget: QWidget, text: str = "") -> str:
        """Have ``widget`` write ``text`` here while the pointer is on it.

        :param widget: the control to watch.
        :param text: what to say. Empty takes the widget's own tooltip,
            which is then cleared -- the sentence moves rather than being
            said twice in two places.
        :returns: the sentence registered, or ``""`` if there was none, in
            which case nothing is watched: a control with nothing to say
            should not blank the bar when the pointer crosses it.
        """
        sentence = (text or widget.toolTip() or "").strip()
        if not sentence:
            return ""
        widget.setToolTip("")
        if not widget.accessibleDescription():
            widget.setAccessibleDescription(sentence)
        self._hints[widget] = sentence
        widget.installEventFilter(self)
        return sentence

    def explains(self, widget: QWidget) -> str:
        """What ``widget`` will write here, or ``""`` if it writes nothing.

        :param widget: a control that may have been registered with
            :meth:`explain`.
        """
        return self._hints.get(widget, "")

    def count(self) -> int:
        """How many controls report to this bar."""
        return len(self._hints)


    def reset(self) -> None:
        """Say the default again."""
        self.setText(self._translated(self._default))

    def _translated(self, text: str) -> str:
        """Translate one hint.

        :param text: the English source.
        :returns: the translation, or the source unchanged when the catalogue
            cannot be reached -- English help beats no help.
        """
        try:
            from ..i18n import tr
        except Exception:
            return text
        return tr(text)

    def eventFilter(self, obj, event):                  # noqa: N802
        """Watch the widgets whose hints this bar shows.

        :param obj: the object the event is for.
        :param event: the event.
        :returns: True to stop the event going further.
        """
        kind = event.type()
        view = getattr(self, "_view", None)
        reading_long_help = view is not None and (
            view.verticalScrollBar().maximum() > 0
            or view.document().blockCount() > 4)
        if kind == QEvent.Enter:
            sentence = self._hints.get(obj)
            if sentence and not (
                    obj is self._resize_handle and reading_long_help):
                self.setText(self._translated(sentence))
        elif kind in (QEvent.Leave, QEvent.HoverLeave):
            if not reading_long_help and self._hints.get(obj) and \
                    self.text() == self._translated(self._hints[obj]):
                self.reset()
        return super().eventFilter(obj, event)


def hint_bar_of(widget: QWidget) -> Optional[HintBar]:
    """The :class:`HintBar` belonging to ``widget``'s window, if it has one.

    Lets a helper deep in a form hand a sentence to the bar without the
    caller having to thread it down through every layer.

    :param widget: any widget, or ``None``; its top-level window is searched
        for a :class:`HintBar` child.
    """
    window = widget.window() if widget is not None else None
    if window is None:
        return None
    found = window.findChild(HintBar)
    return found


def explain_through_the_bar(widget: QWidget, text: str = "") -> bool:
    """Register ``widget`` with its window's bar. False when there is none.

    The caller decides what to do without one -- usually leave the tooltip
    where it is, which is better than a control that explains itself
    nowhere.

    :param widget: the control to register; its window's bar is found with
        :func:`hint_bar_of`. Also returns False when the widget has nothing
        to say.
    """
    bar = hint_bar_of(widget)
    if bar is None:
        return False
    return bool(bar.explain(widget, text))
