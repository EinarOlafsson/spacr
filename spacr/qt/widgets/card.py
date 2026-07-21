"""
Card — QFrame with rounded border, optional title bar, and a body widget.

Consumers add content to `card.body_layout` (a QVBoxLayout).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget

from ..theme import SPACING


class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("CardTitle")
            outer.addWidget(title_label)
        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setObjectName("CardSubtitle")
            sub_label.setWordWrap(True)
            outer.addWidget(sub_label)

        if title or subtitle:
            divider = QFrame()
            divider.setObjectName("Divider")
            divider.setFrameShape(QFrame.HLine)
            outer.addWidget(divider)

        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(SPACING["sm"])
        outer.addWidget(self.body, 1)
