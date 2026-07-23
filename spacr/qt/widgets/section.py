"""
Section — a collapsible group with a header row (chevron + title) and
a QFormLayout body that expands/collapses on click. Used by settings
screens to group related fields; every section is collapsed by
default so users see one row per category instead of a wall of
controls.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import SPACING


class Section(QFrame):
    """Collapsible section with an animated chevron header + form body."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent=None, expanded: bool = False):
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self._expanded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header = QToolButton(self)
        self._header.setObjectName("SectionHeader")
        self._header.setText(title.upper())
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.RightArrow)
        self._header.setCheckable(True)
        self._header.setChecked(False)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setMinimumHeight(34)
        self._header.clicked.connect(self._on_toggle)
        outer.addWidget(self._header)

        # Body
        self._body = QWidget(self)
        self._body.setObjectName("SectionBody")
        self._form = QFormLayout(self._body)
        self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form.setFormAlignment(Qt.AlignTop)
        self._form.setContentsMargins(SPACING["md"], SPACING["md"],
                                       SPACING["md"], SPACING["md"])
        self._form.setHorizontalSpacing(SPACING["md"])
        self._form.setVerticalSpacing(SPACING["sm"])
        self._body.setVisible(False)
        outer.addWidget(self._body)

        if expanded:
            self.set_expanded(True)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def add_row(self, label: str, widget: QWidget) -> None:
        """Add a labeled row to the section's form body."""
        self._form.addRow(label, widget)

    def add_widget(self, widget: QWidget) -> None:
        """Add a full-width (label-less) widget to the section's form body."""
        self._form.addRow(widget)

    def title(self) -> str:
        """Return the section's header text."""
        return self._header.text()

    def set_hint(self, text: str) -> None:
        """Attach a hover tooltip to the section's header.

        The tooltip appears when the user hovers the header, whether
        the section is currently expanded or collapsed — same UX as
        every other Qt tooltip.

        :param text: tooltip text (plain or HTML; empty clears it).
        """
        self._header.setToolTip(text or "")

    def set_expanded(self, on: bool) -> None:
        """Expand or collapse the section body programmatically."""
        self._header.setChecked(on)
        self._on_toggle(on)

    def is_expanded(self) -> bool:
        """Return True when the section body is currently visible."""
        return self._expanded

    # ------------------------------------------------------------------
    def _on_toggle(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self._header.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self._body.setVisible(self._expanded)
        self.toggled.emit(self._expanded)
