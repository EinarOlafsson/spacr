"""
Section — collapsible-looking QGroupBox with a QFormLayout body.

Used by settings screens to group related fields.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QFormLayout, QVBoxLayout, QWidget

from ..theme import SPACING


class Section(QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title.upper(), parent)
        self._form = QFormLayout()
        self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form.setFormAlignment(Qt.AlignTop)
        self._form.setContentsMargins(SPACING["sm"], SPACING["md"],
                                       SPACING["sm"], SPACING["sm"])
        self._form.setHorizontalSpacing(SPACING["md"])
        self._form.setVerticalSpacing(SPACING["sm"])
        self.setLayout(self._form)

    def add_row(self, label: str, widget: QWidget) -> None:
        """Add a labeled row to the section's form."""
        self._form.addRow(label, widget)

    def add_widget(self, widget: QWidget) -> None:
        """Add a full-width widget (spans both label and field columns)."""
        self._form.addRow(widget)
