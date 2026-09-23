"""Persistent, selectable explanations of every element in a pipeline."""
from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextDocument
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import active_palette, font_px
from .workflow_diagram import _api_link


def _connection_card(data, edge):
    """Explain a handoff without repeating the endpoint artifact inventories."""
    names = [escape(tr(data['modules'][edge[key]]['name'])) for key in ('from', 'to')]
    text = edge.get('handoff') or tr("Follow the pathway instructions for this connection.")
    links = ' · '.join(link for link in (_api_link(data, edge['from']), _api_link(data, edge['to'])) if link)
    return ('<b>' + ' → '.join(names) + '.</b> ' + escape(tr(text)) + ' ' + links)


def _module_card(data, key):
    """Keep a module's guidance, input/output locations and API in one text flow."""
    module = data['modules'][key]
    parts = ['<b>' + escape(tr(module['name'])) + '.</b>', escape(tr(module.get('guidance', '')))]
    for role, title in (('inputs', tr('Inputs')), ('outputs', tr('Outputs'))):
        records = []
        for artifact in module.get(role, ()):
            info = data.get('artifacts', {}).get(artifact, {})
            record = escape(tr(info.get('title', artifact)))
            location = info.get('location')
            if location:
                record += ' (' + escape(tr(location).rstrip('.')) + ')'
            records.append(record)
        if records:
            parts.append('<b>' + escape(title) + ':</b> ' + '; '.join(records) + '.')
    parts.append(_api_link(data, key))
    return ' '.join(part for part in parts if part)


class PipelineDetails(QScrollArea):
    """Show a pipeline's dependency stages as module and connection cards.

    :param parent: owning dialog. Cards fill its available width, and text
        wraps when the window width changes. Selecting a graph element only
        changes its border; it never rebuilds the text or scrolls the panel.
    """

    def __init__(self, parent=None):
        """Create a rounded, translucent pane with a stable scroll position."""
        super().__init__(parent)
        self.setObjectName("PipelineDetails")
        self.setWidgetResizable(True)
        self.setMinimumHeight(200)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        palette = active_palette()
        colour = QColor(palette["surface_hi"])
        self.setStyleSheet(f"QScrollArea#PipelineDetails {{ background: rgba({colour.red()}, {colour.green()}, {colour.blue()}, 130); "
                          f"border: 1px solid {palette['border']}; border-radius: 12px; }}")
        self.viewport().setAutoFillBackground(False)
        self.cards = {}
        self._selected = None

    def set_pipeline(self, entry, diagram):
        """Build cards once for a newly selected pipeline, in dependency order.

        :param entry: pathway metadata including title and summary.
        :param diagram: matching WorkflowView with nodes, positions and edges.
        """
        old = self.takeWidget()
        if old is not None:
            old.deleteLater()
        self.cards = {}
        self._selected = None
        content = QWidget()
        content.setObjectName("PipelineExplanationContent")
        content.setStyleSheet("QWidget#PipelineExplanationContent { background: transparent; }")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        header = QLabel('<b>' + escape(tr(entry['title'])) + '.</b> ' +
                        escape(tr(entry.get('description') or entry.get('summary', ''))))
        header.setWordWrap(True)
        header.setStyleSheet(f"background: transparent; font-size: {font_px('body')}px;")
        layout.addWidget(header)
        stages = {}
        for key, node in diagram.nodes.items():
            stages.setdefault(node.pos().x(), []).append(key)
        for stage in sorted(stages):
            group = QWidget()
            group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            grid = QGridLayout(group)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(8)
            keys = sorted(stages[stage], key=lambda k: diagram.nodes[k].pos().y())
            for column, key in enumerate(keys):
                lane = QWidget()
                lane.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                lane_layout = QVBoxLayout(lane)
                lane_layout.setAlignment(Qt.AlignTop)
                lane_layout.setContentsMargins(0, 0, 0, 0)
                lane_layout.setSpacing(6)
                lane_layout.addWidget(self._card(key, _module_card(diagram.data, key)))
                for edge in diagram.links:
                    if edge['from'] == key:
                        identifier = edge['from'] + '→' + edge['to']
                        lane_layout.addWidget(self._card(identifier, _connection_card(diagram.data, edge)))
                grid.addWidget(lane, column // 2, column % 2, Qt.AlignTop)
                grid.setColumnStretch(column % 2, 1)
            layout.addWidget(group)
            if stage != max(stages):
                arrow = QLabel('↓')
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet('color: #168cff; background: transparent; font-size: 16px;')
                layout.addWidget(arrow)
        layout.addStretch()
        self.setWidget(content)

    def _card(self, identifier, html):
        """Create a rounded explanation card whose final links open the API."""
        card = QFrame()
        card.setObjectName("PipelineExplanationCard")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QLabel(html)
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setMinimumWidth(0)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(label)
        self.cards[identifier] = card
        self._style_card(card, False)
        return card

    @staticmethod
    def _style_card(card, selected):
        """Change only the border color so selection cannot alter geometry."""
        palette = active_palette()
        border = '#168cff' if selected else palette['border']
        card.setProperty('selected', selected)
        colour = QColor(palette["surface_hi"])
        card.setStyleSheet(f"QFrame#PipelineExplanationCard {{ background: rgba({colour.red()}, {colour.green()}, {colour.blue()}, 150); "
                          f"border: 2px solid {border}; border-radius: 10px; }} "
                          f"QLabel {{ border: none; background: transparent; font-size: {font_px('body')}px; }}")

    def select_element(self, identifier):
        """Outline the corresponding module or connection without moving text.

        :param identifier: module key, or source and destination joined by →.
        """
        if self._selected in self.cards:
            self._style_card(self.cards[self._selected], False)
        self._selected = identifier
        if identifier in self.cards:
            self._style_card(self.cards[identifier], True)

    def toPlainText(self):
        """Return all visible explanation text for accessibility and inspection."""
        document = QTextDocument()
        document.setHtml("<br>".join(label.text() for label in self.widget().findChildren(QLabel))
                         if self.widget() is not None else "")
        return document.toPlainText()
