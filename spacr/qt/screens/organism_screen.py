"""Sourced organism guides, interactive cell diagrams and assay navigation.

The information and module panes scroll independently and share a horizontal
splitter. Existing assays keep their registry keys; proposals cannot run.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..organisms import ORGANISMS
from ..preferences import scaled_px
from ..theme import SPACING, TILE_H, TILE_ICON_PX, TILE_MAX_W, TILE_W, font_px, make_transparent
from ..widgets.home import AppTile
from ..widgets.organism_diagram import OrganismDiagram


_IMAGES = Path(__file__).resolve().parents[2] / "resources" / "images"
APP_KEY = "toxoplasma"
FOLDED_APPS = ("analyze_plaques", "recruitment", "invasion", "replication", 'host_pathogen')


class OrganismScreen(QWidget):
    """Show one organism and its existing or proposed image-analysis modules.

    :param app_key: ``toxoplasma``, ``plasmodium`` or ``candida``.
    :param host: optional main window receiving module navigation requests.
    :param parent: owning Qt widget.
    """

    module_requested = Signal(str)

    def __init__(self, app_key: str, host=None, parent=None):
        """Build the independently scrollable introduction and Home tile grid."""
        super().__init__(parent)
        self.app_key = app_key
        self.setObjectName("OrganismScreen")
        self.organism = ORGANISMS[app_key]
        if host is not None:
            self.module_requested.connect(host._on_nav_selected)
        self._columns = 0
        self._tiles = []
        root = QVBoxLayout(self)
        heading = QHBoxLayout()
        title = QLabel(tr(self.organism["name"]), self)
        title.setObjectName("SectionTitle")
        title.setStyleSheet(f"font-size: {font_px(22)}px; font-weight: 600;")
        heading.addWidget(title, 1)
        root.addLayout(heading)
        self._splitter = QSplitter(Qt.Horizontal, self)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(scaled_px(1))
        self._splitter.setAccessibleName(tr("Information and modules divider"))
        root.addWidget(self._splitter, 1)
        self._intro = self._build_intro()
        self._scroll = self._pane(self._intro)
        self._scroll.setMinimumWidth(scaled_px(240))
        self._splitter.addWidget(self._scroll)
        self._modules = QWidget()
        self._grid = QGridLayout(self._modules)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(scaled_px(SPACING["xs"]))
        self._module_scroll = self._pane(self._modules)
        self._module_scroll.setMinimumWidth(scaled_px(TILE_W + 24))
        self._splitter.addWidget(self._module_scroll)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([scaled_px(480), scaled_px(780)])
        self._splitter.handle(1).setToolTip(tr("Drag to resize the information pane."))
        self._module_scroll.viewport().installEventFilter(self)
        self._build_tiles()
        self._splitter.splitterMoved.connect(self._reflow)
        QTimer.singleShot(0, self._reflow)

    @staticmethod
    def _pane(widget: QWidget) -> QScrollArea:
        """Wrap content in a frameless, independently scrollable pane."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(widget)
        make_transparent(widget)
        return scroll

    @staticmethod
    def _paragraph(text: str, name: str = "") -> QLabel:
        """Create readable wrapping prose that follows the pane's width."""
        label = QLabel(tr(text))
        label.setObjectName(name)
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return label

    def _build_intro(self) -> QWidget:
        """Add the cell, several assay-referenced sections and cited resources."""
        panel = QWidget()
        panel.setStyleSheet(f"QLabel {{ font-size: {font_px(14)}px; }}")
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, scaled_px(10), 0)
        column.setSpacing(scaled_px(12))
        column.addWidget(self._paragraph(self.organism["description"], "OrganismDescription"))
        self._diagram = OrganismDiagram(self.app_key, _IMAGES / self.organism["diagram"])
        column.addWidget(self._diagram)
        column.addWidget(self._paragraph(self.organism["diagram_note"]))
        records = json.loads((_IMAGES / "organism_sources.json").read_text())
        record = next(row for row in records if row["file"] == self.organism["diagram"])
        credit = self._link(record["credit"] + " · SwissBioPics", record["source_page"])
        credit.setObjectName("OrganismImageCredit")
        column.addWidget(credit)
        column.addWidget(self._link("CC BY 4.0", record["licence_url"]))
        modules = {key: title for key, title, _, _ in self.organism["modules"] if key}
        for title, text, keys in self.organism["sections"]:
            section = self._paragraph(title, "OrganismSectionTitle")
            section.setStyleSheet(f"font-size: {font_px(17)}px; font-weight: 600;")
            column.addWidget(section)
            column.addWidget(self._paragraph(text, "OrganismSectionText"))
            for key in keys:
                link = self._link(tr(modules[key]), key, external=False)
                link.setObjectName("OrganismModuleLink")
                link.linkActivated.connect(self._open_module_link)
                column.addWidget(link)
        resources = self._paragraph("Sources and research resources", "OrganismSectionTitle")
        resources.setStyleSheet(f"font-size: {font_px(17)}px; font-weight: 600;")
        column.addWidget(resources)
        column.addWidget(self._link(tr("Biology source: CDC"), self.organism["source"]))
        for label, url in self.organism["links"]:
            column.addWidget(self._link(tr(label), url))
        column.addStretch(1)
        return panel

    def _open_module_link(self, key: str) -> None:
        """Navigate only to a live assay listed on this organism page."""
        if key == "starplast" and self.app_key == "toxoplasma":
            from ..starplast import open_starplast

            open_starplast(self)
            return
        if any(row[0] == key for row in self.organism["modules"] if row[0]):
            self.module_requested.emit(key)

    @staticmethod
    def _link(label: str, url: str, external: bool = True) -> QLabel:
        """Build an external source link or an internal assay navigation link.

        :param label: translated display text.
        :param url: source URL or existing assay registry key.
        :param external: whether Qt opens the URL in the external browser.
        :returns: a wrapping link label with an accessible name.
        """
        widget = QLabel(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>')
        widget.setOpenExternalLinks(external)
        widget.setWordWrap(True)
        widget.setToolTip(url if external else label)
        widget.setAccessibleName(label)
        widget.setStyleSheet(f"font-size: {font_px(12)}px;")
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return widget

    def _build_tiles(self) -> None:
        """Use Home's tile class, width bounds, height, policy and icon size."""
        from ..app import _icon_for_app, app_stage
        from ..iconset import app_icon

        for key, title, description, icon in self.organism["modules"]:
            artwork = (_icon_for_app(icon) if key else
                       app_icon(icon, override=f"organism_{icon}.svg"))
            tile = AppTile(
                tr(title), tr(description), artwork,
                width=scaled_px(TILE_W), height=scaled_px(TILE_H),
                icon_px=scaled_px(TILE_ICON_PX),
                stage=app_stage(key) if key and key != "starplast" else "alpha", parent=self._modules)
            tile.setProperty("organismModuleKey", key or "")
            tile.setMaximumWidth(scaled_px(TILE_MAX_W))
            tile.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            if key:
                tile.setToolTip(tr(title) + "\n" + tr(description))
                tile.clicked.connect(lambda checked=False, target=key:
                                     self._open_module_link(target))
            else:
                note = tr("Coming soon") + " — " + tr(title) + "\n" + tr(description)
                tile.setToolTip(note)
                tile.setAccessibleDescription(note)
                tile.setEnabled(False)
                tile.setAttribute(Qt.WA_AlwaysShowToolTips, True)
                opacity = QGraphicsOpacityEffect(tile)
                opacity.setOpacity(0.45)
                tile.setGraphicsEffect(opacity)
            self._tiles.append(tile)

    def _reflow(self, *args) -> None:
        """Fit Home tiles to the current right-pane width after divider moves."""
        if not self._tiles:
            return
        available = self._module_scroll.viewport().width()
        spacing = scaled_px(SPACING["xs"])
        columns = max(1, (available + spacing) // (scaled_px(TILE_W) + spacing))
        if columns == self._columns:
            return
        self._columns = columns
        for tile in self._tiles:
            self._grid.removeWidget(tile)
        for index, tile in enumerate(self._tiles):
            self._grid.addWidget(tile, index // columns, index % columns)
        for column in range(self._grid.columnCount()):
            self._grid.setColumnStretch(column, int(column < columns))
        for row in range(self._grid.rowCount()):
            self._grid.setRowStretch(row, 0)
        self._grid.setRowStretch((len(self._tiles) - 1) // columns + 1, 1)

    def eventFilter(self, watched, event) -> bool:
        """Reflow after viewport resize, including vertical scrollbar changes.

        :param watched: Qt object that received the event.
        :param event: Qt event to inspect before delegating to the base filter.
        :returns: the base event filter's result.
        """
        if event.type() == QEvent.Resize and watched is self._module_scroll.viewport():
            QTimer.singleShot(0, self._reflow)
        return super().eventFilter(watched, event)
