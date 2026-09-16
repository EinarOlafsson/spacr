"""UsageBar — labeled slim progress bar for RAM/GPU/CPU indicators."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ..theme import SPACING


class UsageBar(QWidget):
    """Labeled thin progress bar with a right-aligned percent readout.

    BOTH TEXT COLUMNS ARE FIXED-WIDTH ON PURPOSE, and that is what made them
    a bug at a large font scale. The caption column is wider than "RAM" needs
    so the four bars of the System card line up; the readout column is fixed
    so the bar does not jitter sideways as 9 % becomes 10 %. Neither reason
    survives being written as a constant: see :meth:`_size_the_columns`.

    :param label: text shown to the left of the bar (e.g. "RAM", "GPU").
    :param parent: parent widget; ownership only.
    """

    #: The 1.0 widths, kept as the ALIGNMENT FLOOR rather than as the answer.
    #: A column sized to its own text would put every row's bar at a
    #: different x; these keep the four rows of the System card in one line.
    LABEL_W = 48
    PCT_W = 40

    #: The widest readout :meth:`set_value` can ever produce. Sizing to the
    #: longest string a control can hold, rather than to the one it happens
    #: to be showing, is the rule -- and here it also makes
    #: the width deterministic, since the number on screen is live RAM.
    WIDEST_PCT = "100%"

    def __init__(self, label: str, parent=None):
        """Build one labelled resource meter.

        The row itself must paint nothing: the application sheet's blanket
        background rule would otherwise put an opaque rectangle behind the bar,
        whatever is behind the page.

        :param label: what is being measured.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setObjectName("UsageBarRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget#UsageBarRow { background: transparent; }"
            "QProgressBar { background: transparent; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        self._label = QLabel(label)
        self._label.setObjectName("Muted")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setObjectName("UsageBar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, 1)

        self._pct = QLabel("0%")
        self._pct.setObjectName("Muted")
        self._pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._pct)

        self._size_the_columns()


    def _size_the_columns(self) -> None:
        """Set both fixed widths from the font actually in use.

        A FIXED SIZE ONLY HOLDS AT ONE FONT SCALE, caught by a sweep at the
        largest scale preferences offers. `setFixedWidth(48)` and `setFixedWidth(40)` were written at
        100 %, and at 200 % the glyphs doubled while the boxes did not:

            'VRAM'  48 px of column, 66 px of text
            'RAM'   48 px of column, 52 px of text
            '100%'  40 px of column, 62 px of text   (and 46 at 150 %)

        so every System card in the application cut its own labels in half.
        `scaled_px` exists for precisely this -- its docstring says "any
        control tuned to match a text width goes wrong at large font scales"
        -- and it is the floor here rather than the whole answer, because a
        scale is not the only way a glyph gets wider: a theme font, a
        translated caption or a longer bar name would each overflow a purely
        proportional box. The floor keeps the rows aligned; the metrics keep
        the text whole; whichever is larger wins.
        """
        from ..preferences import scaled_px

        if getattr(self, "_pct", None) is None:
            return

        caption = self._label.fontMetrics().horizontalAdvance(
            self._label.text())
        self._label.setFixedWidth(max(scaled_px(self.LABEL_W), caption))
        readout = self._pct.fontMetrics().horizontalAdvance(self.WIDEST_PCT)
        self._pct.setFixedWidth(max(scaled_px(self.PCT_W), readout))

    def changeEvent(self, event) -> None:
        """Re-measure when the font under this row changes.

        The stylesheet is what carries the font scale, and it is installed on
        the application AFTER these widgets are built -- a preferences save
        re-installs it on a live window. Both arrive as a font change on the
        widgets whose resolved font moved, so the columns are re-measured
        there rather than being right only for the sheet that happened to be
        current at construction.
        """
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.ApplicationFontChange,
                            QEvent.StyleChange):
            self._size_the_columns()

    def showEvent(self, event) -> None:
        """Re-measure on the way to being seen.

        A widget's QSS font is resolved when it is polished, which for a
        screen built into a stacked page happens on the way to the first
        show. Measuring here as well as on the font change is what keeps a
        row that was never sent a font change -- because its own font was
        inherited whole -- from being sized by the default metrics.
        """
        self._size_the_columns()
        super().showEvent(event)

    def set_value(self, pct: float) -> None:
        """Set the bar value, clamped to 0-100, and re-color at 75/90 %.

        :param pct: value in ``0.0``-``100.0``; out-of-range values clamp.
        """
        pct = max(0, min(100, int(round(pct))))
        self._bar.setValue(pct)
        self._pct.setText(f"{pct}%")
        if pct >= 90:
            self._bar.setObjectName("UsageBarError")
        elif pct >= 75:
            self._bar.setObjectName("UsageBarWarn")
        else:
            self._bar.setObjectName("UsageBar")
        self._bar.style().unpolish(self._bar)
        self._bar.style().polish(self._bar)
