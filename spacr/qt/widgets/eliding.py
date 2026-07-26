"""Widgets that shrink their text to an ellipsis instead of clipping it.

Qt's :class:`~PySide6.QtWidgets.QLabel` and
:class:`~PySide6.QtWidgets.QPushButton` both *clip* text that doesn't fit
the geometry the layout gave them — the last characters simply stop
being painted, with no ellipsis and no hint that anything is missing.
On a navigation surface that is a real bug: a user cannot click what
they cannot read, and "Annotator Agreeme" looks like a typo rather than
a too-narrow tile.

The two widgets here keep the full string, render an elided copy when
the width is genuinely too small, and put the full string in the
tooltip while that's the case. They also report a *stable* size hint,
computed from the full text rather than from whatever elided copy is
currently displayed — without that the hint would shrink as soon as the
text elided, the layout would hand back a different width, and the
label would oscillate.

Both expose :meth:`full_text` and :meth:`is_elided` so tests (and
callers) can assert "either it fits, or it is elided and the tooltip
carries the whole name".
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QPushButton, QSizePolicy


class ElidingLabel(QLabel):
    """A QLabel that elides rather than clips.

    :param text: the full text to display.
    :param parent: optional parent widget.
    :param mode: where the ellipsis goes; defaults to ``Qt.ElideRight``.
    """

    #: Never elide below roughly this many characters — a label reduced
    #: to a bare "…" carries no information at all.
    _MIN_CHARS = 4

    def __init__(self, text: str = "", parent=None,
                 mode: Qt.TextElideMode = Qt.ElideRight):
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self._elided = False
        self.setText(text)

    # -- text ----------------------------------------------------------
    def setText(self, text: str) -> None:      # noqa: N802 (Qt casing)
        """Set the full text, then render as much of it as fits."""
        self._full_text = text or ""
        self._refresh()

    def full_text(self) -> str:
        """The complete, un-elided text as handed to :meth:`setText`."""
        return self._full_text

    def is_elided(self) -> bool:
        """True when the displayed text is a shortened copy."""
        return self._elided

    def required_width(self) -> int:
        """Width in px at which the full text renders without eliding."""
        self.ensurePolished()
        m = self.contentsMargins()
        return (QFontMetrics(self.font()).horizontalAdvance(self._full_text)
                + m.left() + m.right())

    def available_text_width(self) -> int:
        """Px currently available to draw text in, margins removed."""
        return self._available_width()

    # -- geometry ------------------------------------------------------
    def sizeHint(self) -> QSize:               # noqa: N802
        """Hint the width of the *full* text so the layout can grant it."""
        base = super().sizeHint()
        return QSize(max(base.width(), self.required_width()), base.height())

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        """Allow shrinking to a few characters so the parent can cap us."""
        base = super().minimumSizeHint()
        fm = QFontMetrics(self.font())
        m = self.contentsMargins()
        floor = fm.horizontalAdvance("M" * self._MIN_CHARS + "…")
        return QSize(min(base.width(), floor + m.left() + m.right()),
                     base.height())

    def resizeEvent(self, event) -> None:      # noqa: N802
        """Re-elide whenever the layout hands us a different width."""
        super().resizeEvent(event)
        self._refresh()

    # -- internals -----------------------------------------------------
    def _available_width(self) -> int:
        m = self.contentsMargins()
        return self.width() - m.left() - m.right()

    def _refresh(self) -> None:
        """Show the full text when it fits, an elided copy when it doesn't."""
        fm = QFontMetrics(self.font())
        available = self._available_width()
        needed = fm.horizontalAdvance(self._full_text)
        # Before the first layout pass the widget still carries Qt's
        # default 100 px size, which would elide almost everything and
        # leave a stale tooltip behind. Wait for a real geometry.
        if not self.testAttribute(Qt.WA_Resized):
            available = max(available, needed)
        if available <= 0 or needed <= available:
            self._elided = False
            QLabel.setText(self, self._full_text)
            if self.toolTip() == self._full_text:
                self.setToolTip("")
            return
        self._elided = True
        QLabel.setText(
            self, fm.elidedText(self._full_text, self._elide_mode, available))
        # A user who cannot read the whole name must still be able to
        # discover it — the tooltip is the only place left to put it.
        self.setToolTip(self._full_text)


class ElidingPushButton(QPushButton):
    """A QPushButton whose label elides rather than clips.

    Used for the sidebar navigation items, where the column has a fixed
    width and the app names keep getting longer.

    :param text: the full button text.
    :param parent: optional parent widget.
    :param mode: where the ellipsis goes; defaults to ``Qt.ElideRight``.
    """

    _MIN_CHARS = 6

    def __init__(self, text: str = "", parent=None,
                 mode: Qt.TextElideMode = Qt.ElideRight):
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self._elided = False
        # Horizontally shrinkable: without this the layout treats the
        # size hint as a hard minimum and squeezes the *whole* sidebar
        # instead of shortening one label.
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Preferred)
        self.setSizePolicy(policy)
        self.setText(text)

    # -- text ----------------------------------------------------------
    def setText(self, text: str) -> None:      # noqa: N802
        """Set the full text, then render as much of it as fits."""
        self._full_text = text or ""
        self._refresh()

    def full_text(self) -> str:
        """The complete, un-elided text as handed to :meth:`setText`."""
        return self._full_text

    def is_elided(self) -> bool:
        """True when the displayed text is a shortened copy."""
        return self._elided

    # -- geometry ------------------------------------------------------
    def sizeHint(self) -> QSize:               # noqa: N802
        """Hint the width the *full* text needs, elided or not.

        ``QPushButton.sizeHint`` measures the text currently set, which
        after eliding is shorter than the real name; adding back the
        difference keeps the hint stable across elide/unelide.
        """
        base = super().sizeHint()
        fm = QFontMetrics(self.font())
        extra = (fm.horizontalAdvance(self._full_text)
                 - fm.horizontalAdvance(super().text()))
        return QSize(base.width() + max(0, extra), base.height())

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        """Allow shrinking to a handful of characters plus the icon."""
        base = QPushButton.sizeHint(self)
        fm = QFontMetrics(self.font())
        chrome = base.width() - fm.horizontalAdvance(super().text())
        floor = fm.horizontalAdvance("M" * self._MIN_CHARS + "…")
        return QSize(min(base.width(), chrome + floor), base.height())

    def resizeEvent(self, event) -> None:      # noqa: N802
        """Re-elide whenever the layout hands us a different width."""
        super().resizeEvent(event)
        self._refresh()

    def available_text_width(self) -> int:
        """Px left for the label once the icon and style padding are paid for.

        Derived from the button's own size hint (hint minus the advance of
        the text it currently shows == everything that is not text), so it
        follows the active style instead of assuming padding values.
        """
        fm = QFontMetrics(self.font())
        chrome = (QPushButton.sizeHint(self).width()
                  - fm.horizontalAdvance(super().text()))
        return self.width() - chrome

    # -- internals -----------------------------------------------------
    def _refresh(self) -> None:
        fm = QFontMetrics(self.font())
        available = self.available_text_width()
        needed = fm.horizontalAdvance(self._full_text)
        # See ElidingLabel._refresh — don't elide against the default
        # 100 px size a widget carries before its first layout pass.
        if not self.testAttribute(Qt.WA_Resized):
            available = max(available, needed)
        if available <= 0 or needed <= available:
            self._elided = False
            QPushButton.setText(self, self._full_text)
            return
        self._elided = True
        QPushButton.setText(
            self, fm.elidedText(self._full_text, self._elide_mode, available))
