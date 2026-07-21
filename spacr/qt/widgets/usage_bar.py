"""UsageBar — labeled slim progress bar for RAM/GPU/CPU indicators."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ..theme import SPACING


class UsageBar(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
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

    def set_value(self, pct: float):
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
