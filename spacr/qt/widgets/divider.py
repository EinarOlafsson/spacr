"""Divider — thin themed separator, horizontal or vertical."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame


class Divider(QFrame):
    """Thin themed separator line, horizontal or vertical.

    :param orientation: ``Qt.Horizontal`` (default) or ``Qt.Vertical``.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, orientation: Qt.Orientation = Qt.Horizontal, parent=None):
        """Build a one-pixel rule.

        :param orientation: horizontal or vertical; it is fixed to one pixel on
            the axis it does not span.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setObjectName("Divider")
        if orientation == Qt.Horizontal:
            self.setFrameShape(QFrame.HLine)
            self.setFixedHeight(1)
        else:
            self.setFrameShape(QFrame.VLine)
            self.setFixedWidth(1)
