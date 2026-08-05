"""UsageBar — labeled slim progress bar for RAM/GPU/CPU indicators."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ..theme import SPACING


class UsageBar(QWidget):
    """Labeled thin progress bar with a right-aligned percent readout.

    :param label: text shown to the left of the bar (e.g. "RAM", "GPU").
    """

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
        self._label.setFixedWidth(48)
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
        self._pct.setFixedWidth(40)
        layout.addWidget(self._pct)

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
