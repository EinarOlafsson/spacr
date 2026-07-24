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
        # The global `QWidget { background: bg }` rule would otherwise paint
        # this row solid black inside the System card. Make the row itself
        # transparent so the card's dark-gray surface shows around the bar.
        self.setObjectName("UsageBarRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("QWidget#UsageBarRow { background: transparent; }")
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
