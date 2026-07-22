"""
EmptyState — reusable big "nothing here yet" panel with icon, title,
subtitle, and an optional call-to-action button.

Used on screens whose primary content requires the user to first pick
a source folder / open a database (annotate, make-masks) — instead of
showing an empty grid or blank canvas we render this centered.
"""
from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING


class _WrappedLabel(QLabel):
    """QLabel that always sizes to fit its wrapped text at a fixed width.

    QLabel's default sizeHint ignores word-wrap even when it's on, so
    overflow text bleeds over neighbor widgets in a QVBoxLayout. We use
    QFontMetrics.boundingRect(width, TextWordWrap, ...) to compute the
    correct height and pin the label to that exact size.
    """
    def __init__(self, text: str, wrap_width: int = 560, parent=None):
        super().__init__(text, parent)
        self._wrap_width = wrap_width
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(wrap_width, self._wrapped_height())

    def _wrapped_height(self) -> int:
        """Compute label height needed to render the wrapped text."""
        from PySide6.QtCore import QRect
        fm = self.fontMetrics()
        rect = fm.boundingRect(
            QRect(0, 0, self._wrap_width, 10_000),
            Qt.TextWordWrap,
            self.text(),
        )
        return rect.height() + 8

    def setText(self, text: str) -> None:
        """Replace the label text and re-fix the label to its wrapped height."""
        super().setText(text)
        self.setFixedSize(self._wrap_width, self._wrapped_height())


class EmptyState(QWidget):
    """Centered vertical stack: icon → title → subtitle → CTA button.

    Any element can be omitted by passing an empty string / None.
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
