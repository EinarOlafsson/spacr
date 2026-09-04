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
    #: to be showing, is instruction 350's own rule -- and here it also makes
    #: the width deterministic, since the number on screen is live RAM.
    WIDEST_PCT = "100%"

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("UsageBarRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Two rules, and the second is the load-bearing one.
        #
        # The ROW must paint nothing, or the blanket
        # `QWidget { background-color: bg }` fills it with the WINDOW colour
        # inside the System card.
        #
        # So must the bar's TRACK, and that was the part missing. The
        # application sheet gives `QProgressBar#UsageBar` a `surface_alt` fill
        # *at page opacity*, and the card it sits in is already `surface_alt`
        # at page opacity — so the track laid a second copy of the same
        # translucent grey over the first and read as a band the slider could
        # not thin: measured, at a requested 30 % the card passed 0.70 of the
        # backdrop and the track only 0.49.
        #
        # One of the four bars escaped that, and only by accident: the CPU bar
        # sits in a wrapper carrying an unqualified `background: transparent`,
        # and in Qt a sheet set on an ANCESTOR beats the application sheet
        # irrespective of selector specificity, so the wrapper's rule reached
        # the bar and cancelled the fill. RAM, GPU and VRAM go straight into
        # the card body, whose sheet is qualified (`QWidget#CardBody`) and
        # never reached theirs. Saying it here is what makes all four behave
        # the same wherever they are put.
        #
        # The selector is name-agnostic on purpose: `set_value` renames the bar
        # to UsageBarWarn / UsageBarError past 75 / 90 %, and the only
        # QProgressBar under this row is that one bar. `::chunk` is a separate
        # sub-control, so the filled part keeps its accent / warning / error
        # colour — only the empty track goes away.
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

    # ------------------------------------------------------------------
    # The two fixed widths, measured rather than assumed
    # ------------------------------------------------------------------

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

        # `setStyleSheet` in `__init__` delivers a StyleChange to this row
        # BEFORE either label exists, so the hook below can arrive during
        # construction. Nothing to size yet is not an error.
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
        # Force restyle since QSS keys on objectName.
        self._bar.style().unpolish(self._bar)
        self._bar.style().polish(self._bar)
