"""Qt labels and buttons that elide text without losing the full value.

The widgets render an ellipsis when space is limited and expose the complete
text in a tooltip. Stable size hints prevent layouts from oscillating between
full and shortened text.

PROGRESS LINE. :class:`ProgressLine` is a slim progress bar whose numbers sit
beside it, where nothing can cut them.

WHY (item 502). The theme draws every ``QProgressBar`` as an 8 px track
(``height: 8px; max-height: 8px`` in :mod:`spacr.qt.theme`). A bar that also
paints its own text -- "step 2 of 3", "45%", "312 MB / 690 MB (45%)" -- draws
a 13 px caption into those 8 px, so only the top half of each glyph reaches
the screen, and at 50 % GUI scale the track is 4 px and almost nothing does.
The maintainer's report was "the text e.g. step 1 of 3 or 10% is cut off".

THE DESIGN the maintainer chose is "thin bar + label":

    [████████░░░░░░░░░░░░]  step 2 of 3 · 45%
    Downloading torch… 312 MB of 690 MB · 4.2 MB/s · 1 min 30 s left

* the bar paints no text at all and stays the slim track the theme draws;
* the COUNT beside it (the step and the percentage, the numbers a person is
  watching) is a plain label that is never elided -- its horizontal size
  policy is ``Minimum``, so a layout cannot give it less than its text needs;
* an optional DETAIL line below carries the part that changes and can be
  long (a file name, a speed, a time left); that part elides, and
  :meth:`ProgressLine.displayed_text` reports what is really painted.

No size here is computed from font metrics and then handed to a size setter,
so the GUI-scale layer (:mod:`spacr.qt.gui_scale`) scales each size once: the
label follows the scaled font, the spacing follows the scaled layout.

The widget answers the parts of ``QProgressBar``'s API the call sites use
(``setRange``, ``setValue``, ``setFormat`` with ``%p``/``%v``/``%m``,
``setTextVisible``, ``format``, ``value`` ...), so swapping one in for a bar
changes one line at each site.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)


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
        """Create a label that shortens its text rather than forcing a width.

        :param text: the full text; kept, so widening restores it.
        :param parent: parent widget, or ``None``.
        :param mode: where the ellipsis goes.
        """
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self._elided = False
        self.setText(text)

    def setText(self, text: str) -> None:      # noqa: N802 (Qt casing)
        """Set the full text, then render as much of it as fits.

        :param text: the full label text; ``None`` is treated as ``""``. When
            it has to be elided, the full text becomes the tooltip.
        """
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
        """Re-elide whenever the layout hands us a different width.

        :param event: the resize event, passed to the base class; the text is
            then re-fitted to the label's new width.
        """
        super().resizeEvent(event)
        self._refresh()

    def _available_width(self) -> int:
        """Return the width left for text after the contents margins.

        :returns: the usable width in pixels.
        """
        m = self.contentsMargins()
        return self.width() - m.left() - m.right()

    def _refresh(self) -> None:
        """Show the full text when it fits, an elided copy when it doesn't."""
        fm = QFontMetrics(self.font())
        available = self._available_width()
        needed = fm.horizontalAdvance(self._full_text)
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
        """Create a button that shortens its label rather than forcing a width.

        The horizontal policy is loosened deliberately: without it the layout
        treats the size hint as a hard minimum and squeezes the whole sidebar
        instead of shortening one label.

        :param text: the full text; kept, so widening restores it.
        :param parent: parent widget, or ``None``.
        :param mode: where the ellipsis goes.
        """
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self._elided = False
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Preferred)
        self.setSizePolicy(policy)
        self.setText(text)

    def setText(self, text: str) -> None:      # noqa: N802
        """Set the full text, then render as much of it as fits.

        :param text: the full button label; ``None`` is treated as ``""``.
        """
        self._full_text = text or ""
        self._refresh()

    def full_text(self) -> str:
        """The complete, un-elided text as handed to :meth:`setText`."""
        return self._full_text

    def is_elided(self) -> bool:
        """True when the displayed text is a shortened copy."""
        return self._elided

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
        """Re-elide whenever the layout hands us a different width.

        :param event: the resize event, passed to the base class; the label is
            then re-fitted to the button's new width.
        """
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

    def _refresh(self) -> None:
        """Re-elide the label to whatever width the button now has.

        Nothing is elided before the first layout pass: a widget carries a
        default 100 px until then, and eliding against it would shorten a label
        that was never actually short of room.
        """
        fm = QFontMetrics(self.font())
        available = self.available_text_width()
        needed = fm.horizontalAdvance(self._full_text)
        if not self.testAttribute(Qt.WA_Resized):
            available = max(available, needed)
        if available <= 0 or needed <= available:
            self._elided = False
            QPushButton.setText(self, self._full_text)
            return
        self._elided = True
        QPushButton.setText(
            self, fm.elidedText(self._full_text, self._elide_mode, available))


SEPARATOR = " · "


class ProgressLine(QWidget):
    """A thin bar with its count beside it and an optional detail line below.

    :param parent: parent widget, or ``None``.
    :param detail: whether to build the eliding detail line under the bar.
    :param count_below: put the count on its own line under the bar, at the
        left, instead of beside it -- for a side panel whose content can be
        wider than the pane, where the right end of a row is scrolled away.
    """

    def __init__(self, parent=None, *, detail: bool = True,
                 count_below: bool = False):
        """Build the bar, the count label and, if asked, the detail line.

        :param parent: parent widget, or ``None``.
        :param detail: build the detail line under the bar.
        :param count_below: the count under the bar instead of beside it.
        """
        super().__init__(parent)
        self.setObjectName("ProgressLine")
        self._format = "%p%"
        self._text_visible = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.bar = QProgressBar(self)
        self.bar.setTextVisible(False)
        self.bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.bar, 1, Qt.AlignVCenter)

        self.count = QLabel("", self)
        self.count.setObjectName("ProgressLineCount")
        self.count.setTextFormat(Qt.PlainText)
        self.count.setWordWrap(False)
        self.count.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        if count_below:
            self.count.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            outer.addLayout(row)
            outer.addWidget(self.count)
        else:
            self.count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(self.count, 0, Qt.AlignVCenter)
            outer.addLayout(row)

        self.detail = None
        if detail:
            self.detail = ElidingLabel("", self)
            self.detail.setObjectName("ProgressLineDetail")
            self.detail.setTextFormat(Qt.PlainText)
            self.detail.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Preferred)
            self.detail.hide()
            outer.addWidget(self.detail)
        self._refresh()

    def setRange(self, minimum: int, maximum: int) -> None:
        """Set the bar's range; ``(0, 0)`` is the busy state with no number.

        :param minimum: the value at an empty bar.
        :param maximum: the value at a full bar.
        """
        self.bar.setRange(int(minimum), int(maximum))
        self._refresh()

    def setMinimum(self, minimum: int) -> None:
        """Set the value at an empty bar.

        :param minimum: the new minimum.
        """
        self.bar.setMinimum(int(minimum))
        self._refresh()

    def setMaximum(self, maximum: int) -> None:
        """Set the value at a full bar.

        :param maximum: the new maximum.
        """
        self.bar.setMaximum(int(maximum))
        self._refresh()

    def setValue(self, value: int) -> None:
        """Move the bar and the numbers beside it.

        :param value: the new value, clamped by the bar to its range.
        """
        self.bar.setValue(int(value))
        self._refresh()

    def reset(self) -> None:
        """Empty the bar the way ``QProgressBar.reset`` does."""
        self.bar.reset()
        self._refresh()

    def value(self) -> int:
        """The bar's current value."""
        return self.bar.value()

    def minimum(self) -> int:
        """The value at an empty bar."""
        return self.bar.minimum()

    def maximum(self) -> int:
        """The value at a full bar."""
        return self.bar.maximum()

    def setFormat(self, text: str) -> None:
        """Set the count template, with ``QProgressBar``'s ``%p %v %m``.

        A template with no ``%p`` gets the percentage appended after a
        separator whenever the bar has a range, so the number is never lost.

        :param text: the template, e.g. ``"step 2 of 3"`` or
            ``"%v / %m jobs"``.
        """
        self._format = str(text or "")
        self._refresh()

    def format(self) -> str:
        """The count template last handed to :meth:`setFormat`."""
        return self._format

    def setTextVisible(self, visible: bool) -> None:
        """Show or hide the count beside the bar; the bar never paints text.

        :param visible: whether the count label is shown.
        """
        self._text_visible = bool(visible)
        self.count.setVisible(self._text_visible)
        self._refresh()

    def isTextVisible(self) -> bool:
        """Whether the count beside the bar is shown."""
        return self._text_visible

    def set_detail(self, text: str) -> None:
        """Set the line under the bar; it elides when the window is narrow.

        :param text: the changing part -- a file name, a speed, a time left.
        """
        if self.detail is None:
            return
        self.detail.setText(str(text or ""))
        self.detail.setVisible(bool(text))

    def percent(self):
        """The whole percentage the bar shows, or ``None`` while busy."""
        low, high = self.bar.minimum(), self.bar.maximum()
        value = self.bar.value()
        if high <= low or value < low:
            return None
        return int((value - low) * 100 / (high - low))

    def text(self) -> str:
        """The count as it is written beside the bar."""
        return self.count.text()

    def count_text(self) -> str:
        """The count as it is written beside the bar (never elided)."""
        return self.count.text()

    def detail_text(self) -> str:
        """The full detail line, before any eliding."""
        return self.detail.full_text() if self.detail is not None else ""

    def displayed_text(self) -> str:
        """What is really painted: the count, then the detail as elided.

        Hidden parts are left out, so a test compares against the screen.
        """
        parts = []
        if self._text_visible and self.count.text():
            parts.append(self.count.text())
        if self.detail is not None and not self.detail.isHidden():
            parts.append(QLabel.text(self.detail))
        return "\n".join(parts)

    def _refresh(self) -> None:
        """Write the count from the template and the bar's position."""
        percent = self.percent()
        value, top = self.bar.value(), self.bar.maximum()
        busy = percent is None
        text = self._format
        if "%p" in text and busy:
            for token in ("(%p%)", "%p%", "%p"):
                text = text.replace(token, "")
        text = (text.replace("%p", "" if busy else str(percent))
                .replace("%v", str(max(value, self.bar.minimum())))
                .replace("%m", str(top))).strip()
        if not busy and "%p" not in self._format:
            text = f"{text}{SEPARATOR}{percent}%" if text else f"{percent}%"
        self.count.setText(text)
