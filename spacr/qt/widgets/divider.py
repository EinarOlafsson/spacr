"""Divider — thin themed separator, horizontal or vertical."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame


class Divider(QFrame):
    def __init__(self, orientation: Qt.Orientation = Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.setObjectName("Divider")
        if orientation == Qt.Horizontal:
            self.setFrameShape(QFrame.HLine)
            self.setFixedHeight(1)
        else:
            self.setFrameShape(QFrame.VLine)
            self.setFixedWidth(1)
