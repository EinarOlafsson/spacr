"""
EmptyState — reusable big "nothing here yet" panel with icon, title,
subtitle, and an optional call-to-action button.

Used on screens whose primary content requires the user to first pick
a source folder / open a database (annotate, make-masks) — instead of
showing an empty grid or blank canvas, these screens show this panel centered.
"""
from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtCore import QEvent, Qt, QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import SPACING
from ..gui_scale import _set_actual_dim, add_listener


class _WrappedLabel(QLabel):
    """QLabel that always sizes to fit its wrapped text at a fixed width.

    QLabel's default sizeHint ignores word-wrap even when it's on, so
    overflow text bleeds over neighbor widgets in a QVBoxLayout. We use
    QFontMetrics.boundingRect(width, TextWordWrap, ...) to compute the
    correct height and pin the label to that exact size.
    """
    def __init__(self, text: str, wrap_width: int = 560, parent=None):
        """Build a label whose height is measured, not left to the layout.

        :param text: the sentence to wrap.
        :param wrap_width: the width to wrap at, and the label's fixed
            width. THE HEIGHT IS COMPUTED FROM IT rather than requested: a
            wrapping label that asks the layout for its height gets one
            answer before the layout runs and another after, which is how an
            empty state ends up clipped to a single line.
        :param parent: parent widget.
        """
        super().__init__(text, parent)
        self._wrap_width = wrap_width
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(wrap_width)
        self._fit_wrapped_height()
        add_listener(self._fit_wrapped_height)

    def _fit_wrapped_height(self, _scale=None) -> None:
        """Reconcile physical height after measurement or a finished GUI zoom.

        :param _scale: unused zoom notification; measure the current font
            and contents width instead.
        """
        from shiboken6 import isValid

        if not isValid(self):
            return
        height = self._wrapped_height()
        _set_actual_dim(self, "minh", height)
        _set_actual_dim(self, "maxh", height)

    def _wrapped_height(self) -> int:
        """Compute label height needed to render the wrapped text."""
        from PySide6.QtCore import QRect
        fm = self.fontMetrics()
        rect = fm.boundingRect(
            QRect(0, 0, max(1, self.contentsRect().width()), 10_000),
            Qt.TextWordWrap,
            self.text(),
        )
        return rect.height() + 8

    def setText(self, text: str) -> None:
        """Replace the label text and re-fix the label to its wrapped height."""
        super().setText(text)
        self._fit_wrapped_height()

    def event(self, event: QEvent) -> bool:
        """Remeasure after Qt applies the font or style used to paint the text.

        :param event: the event delivered by Qt.
        :returns: whether the base label handled the event.
        """
        handled = super().event(event)
        if hasattr(self, "_wrap_width") and event.type() in (
            QEvent.FontChange, QEvent.ApplicationFontChange,
            QEvent.StyleChange, QEvent.Polish, QEvent.Resize,
        ):
            height = self._wrapped_height()
            if height != self.height():
                self._fit_wrapped_height()
        return handled


class EmptyState(QWidget):
    """Centered vertical stack: icon → title → subtitle → CTA button.

    Any element can be omitted by passing an empty string / None.

    :param title: heading, in the page's title weight. Omitted when empty.
    :param subtitle: one or two sentences under the title saying what the
        reader can do about the emptiness. Omitted when empty.
    :param icon: mark above the title. Omitted when ``None``.
    :param cta_label: caption for the action button. No button is built
        when this is empty, so a state with nothing to offer costs no
        widget rather than showing a disabled one.
    :param on_action: called when that button is pressed. The
        :attr:`action_triggered` signal is emitted either way, so a host may
        connect to it instead.
    :param parent: owning widget, or ``None``.
    """

    action_triggered = Signal()

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        icon: Optional[QIcon] = None,
        cta_label: str = "",
        on_action: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        """Build the placeholder shown where content would be.

        :param title: the headline.
        :param subtitle: the sentence under it, saying what to do.
        :param icon: an icon above the text, or ``None``.
        :param cta_label: the action button's label; empty shows no button.
        :param on_action: called when that button is pressed.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["xxl"], SPACING["xxl"],
                                  SPACING["xxl"], SPACING["xxl"])
        outer.setSpacing(SPACING["md"])
        outer.addStretch(1)

        if icon is not None:
            icon_lbl = QLabel()
            pix = icon.pixmap(QSize(80, 80))
            icon_lbl.setPixmap(pix)
            icon_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(icon_lbl, alignment=Qt.AlignCenter)

        if title:
            title_lbl = QLabel(title)
            title_lbl.setObjectName("TitleHeading")
            title_lbl.setAlignment(Qt.AlignCenter)
            outer.addWidget(title_lbl, alignment=Qt.AlignCenter)

        if subtitle:
            sub_lbl = _WrappedLabel(subtitle, wrap_width=560)
            sub_lbl.setObjectName("SubtitleSmall")
            outer.addWidget(sub_lbl, alignment=Qt.AlignHCenter)

        if cta_label:
            btn = QPushButton(cta_label)
            btn.setObjectName("PrimaryButton")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(40)
            btn.setMinimumWidth(220)
            btn.clicked.connect(self.action_triggered.emit)
            if on_action:
                btn.clicked.connect(on_action)
            self._cta_button = btn
            outer.addSpacing(SPACING["md"])
            outer.addWidget(btn, alignment=Qt.AlignCenter)
        else:
            self._cta_button = None

        outer.addStretch(1)

    @property
    def cta_button(self) -> Optional[QPushButton]:
        """The primary CTA button, or ``None`` when no ``cta_label`` was set."""
        return self._cta_button
