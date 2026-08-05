"""
Section — a collapsible group with a header row (chevron + title) and
a QFormLayout body that expands/collapses on click. Used by settings
screens to group related fields; every section is collapsed by
default so users see one row per category instead of a wall of
controls.
"""
from __future__ import annotations

import re
from typing import Optional, Union

from PySide6.QtCore import Qt, Signal
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

from ..theme import SPACING, STAGE_LABEL, STAGE_NOTE


class Section(QFrame):
    """Collapsible section with an animated chevron header + form body."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent=None, expanded: bool = False):
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self._expanded = False
        self._maturity = "stable"
        self._hint = ""
        self._row_widgets = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header = QToolButton(self)
        self._header.setObjectName("SectionHeader")
        # A QToolButton reads '&' as a mnemonic marker, so the categories
        # named "Plate Layout & Controls" and "Embedding & Clustering"
        # rendered as "PLATE LAYOUT _CONTROLS" -- the ampersand swallowed and
        # the following letter underlined as an accelerator that goes
        # nowhere. Section headers are not keyboard shortcuts, so the '&' is
        # escaped for display. `title()` still answers with the real text.
        self._title = title.upper()
        self._refresh_header_text()
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
    def add_row(
        self,
        label: Union[str, QWidget],
        widget: QWidget,
        info_widget: Optional[QWidget] = None,
        wrap_label: bool = False,
    ) -> None:
        """Add a labeled row, optionally with an information-link icon.

        :param wrap_label: build the ``SettingLabelWithInfo`` host even with
            no info widget to put in it. The host is what right-aligns the
            label against its field; it used to arrive only as a side effect
            of there being a dot, so removing the dot from the settings form
            left every label left-aligned and turned half of each row's
            width into page showing through rather than category surface.
            ``_row_widgets`` still records the caller's label, not the host,
            so anything reading rows back gets the ``QLabel`` it passed in.
        """
        form_label = label
        if info_widget is not None or wrap_label:
            form_label = QWidget(self._body)
            form_label.setObjectName("SettingLabelWithInfo")
            label_row = QHBoxLayout(form_label)
            label_row.setContentsMargins(0, 0, 0, 0)
            label_row.setSpacing(SPACING["xs"])
            label_row.addStretch(1)
            if isinstance(label, QWidget):
                label_row.addWidget(label)
            else:
                label_row.addWidget(QLabel(str(label), form_label))
            if info_widget is not None:
                label_row.addWidget(info_widget)
        self._form.addRow(form_label, widget)
        self._row_widgets.append((label, widget))
        self._apply_maturity(label, setting=True)
        self._apply_maturity(widget, setting=True)

    def add_widget(self, widget: QWidget) -> None:
        """Add a full-width (label-less) widget to the section's form body."""
        self._form.addRow(widget)
        self._row_widgets.append((None, widget))
        self._apply_maturity(widget, setting=True)

    def title(self) -> str:
        """Return the section's header text, un-escaped."""
        return self._title

    def header(self) -> QToolButton:
        """Return the clickable header button (chevron + category title).

        Public because a screen needs a precise hover target for the
        *category* it represents: the section itself covers the whole form
        once expanded, so filtering events on it would report the category
        while the pointer is over one of its settings.
        """
        return self._header

    def set_hint(self, text: str) -> None:
        """Attach a hover tooltip to the section's header.

        The tooltip appears when the user hovers the header, whether
        the section is currently expanded or collapsed — same UX as
        every other Qt tooltip.

        :param text: tooltip text (plain or HTML; empty clears it).
        """
        self._hint = text or ""
        self._refresh_tooltip()

    def set_maturity(self, stage: str) -> None:
        """Colour this section and every setting row by maturity stage.

        ``stable``/``beta``/``alpha`` use the exact hues shown in Home's
        maturity legend. Unknown values deliberately fall back to stable.
        """
        stage = str(stage or "stable").lower()
        if stage not in STAGE_LABEL:
            stage = "stable"
        self._maturity = stage
        for target in (self, self._header, self._body):
            self._apply_maturity(target)
        for label, widget in self._row_widgets:
            self._apply_maturity(label, setting=True)
            self._apply_maturity(widget, setting=True)
        self.setAccessibleDescription(
            f"{STAGE_LABEL[stage]} maturity settings section"
        )
        self._refresh_header_text()
        self._refresh_tooltip()

    def maturity(self) -> str:
        """Return ``stable``, ``beta`` or ``alpha`` for this section."""
        return self._maturity

    def set_expanded(self, on: bool) -> None:
        """Expand or collapse the section body programmatically."""
        self._header.setChecked(on)
        self._on_toggle(on)

    def is_expanded(self) -> bool:
        """Return True when the section body is currently visible."""
        return self._expanded

    def _apply_maturity(self, widget, *, setting: bool = False) -> None:
        if not isinstance(widget, QWidget):
            return
        prop = "settingMaturity" if setting else "maturity"
        widget.setProperty(prop, self._maturity)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)

    def _refresh_header_text(self) -> None:
        text = self._title
        if self._maturity != "stable":
            stage = STAGE_LABEL[self._maturity].upper()
            # Category names historically carried their own ``(BETA)`` or
            # were simply named ``Beta``. Maturity styling then appended a
            # second ``· BETA`` badge, producing ``BETA · BETA``. Keep the
            # original title for configuration lookups, but render one badge.
            text = re.sub(
                rf"\s*(?:\(\s*{re.escape(stage)}\s*\)|{re.escape(stage)})\s*$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            text = f"{text}   ·   {stage}" if text else f"·   {stage}"
        text = text.replace("&", "&&")
        self._header.setText(text)

    def _refresh_tooltip(self) -> None:
        # Stable is the normal case, so preserve existing curated tooltips
        # byte-for-byte. Beta/alpha need the caution text because their colour
        # carries information the old tooltip did not.
        note = (
            STAGE_NOTE.get(self._maturity, "")
            if self._maturity != "stable"
            else ""
        )
        parts = [part for part in (self._hint, note) if part]
        self._header.setToolTip("\n\n".join(parts))

    # ------------------------------------------------------------------
    def _on_toggle(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self._header.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self._body.setVisible(self._expanded)
        self.toggled.emit(self._expanded)
