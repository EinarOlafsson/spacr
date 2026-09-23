"""Interactive workflow diagrams built from the shared module and artifact map.

Solid arrows describe documented handoffs. Dashed arrows connect matching
artifact types or pathway prerequisites; their explanations distinguish these
from verified handoffs. Nodes and edges never start analysis. Hover or select
one to read its description in the owning dialog's fixed details area.
"""
from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFrame, QGraphicsItem,
    QGraphicsPathItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QPushButton, QTextBrowser, QVBoxLayout,
)

from ..i18n import tr
from ..theme import active_palette, font_px


def workflow_map(path=None):
    """Read the bundled module map, or an explicitly supplied JSON path.

    :returns: the map containing modules, artifacts, pathways and connections.
    :raises OSError: the file cannot be read.
    :raises ValueError: the JSON is invalid or lacks the required collections.
    """
    source = Path(path) if path is not None else (
        Path(__file__).resolve().parents[2] / "resources/module_workflows.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data.get("modules"), dict) or not isinstance(data.get("artifacts"), dict):
        raise ValueError("Workflow map needs modules and artifacts.")
    return data


def connections(data, keys=None, steps=None):
    """Return documented handoffs and explicitly distinguished inferred links.

    :param data: shared workflow map.
    :param keys: optional module subset; None includes every mapped module.
    :param steps: pathway steps, whose ``after`` dependencies define its edges.
        None builds the full network, including matching artifact types.
    :returns: dictionaries with ``from``, ``to``, ``artifacts``, ``handoff``
        and ``kind`` (documented, compatible or prerequisite). Inferred
        compatibility describes matching types, not an automatic file import.
    """
    modules = data["modules"]
    selected = list(keys if keys is not None else modules)
    documented = {(edge["from"], edge["to"]): edge
                  for edge in data.get("connections", ())}
    if steps is not None:
        pairs = [(before, step["module"]) for step in steps
                 for before in step.get("after", ())]
    else:
        pairs = [(source, target) for source in selected for target in selected
                 if source != target and ((source, target) in documented or
                    set(modules[source].get("outputs", ())) &
                    set(modules[target].get("inputs", ())))]
    result = []
    for source, target in pairs:
        if source not in selected or target not in selected:
            continue
        if (source, target) in documented:
            result.append(dict(documented[source, target], kind="documented"))
            continue
        shared = sorted(set(modules.get(source, {}).get("outputs", ())) &
                        set(modules.get(target, {}).get("inputs", ())))
        result.append({"from": source, "to": target, "artifacts": shared,
                       "handoff": "", "kind": "compatible" if shared else "prerequisite"})
    return result


def _artifact_names(data, keys):
    """Resolve artifact titles without discarding an unknown artifact key."""
    return [tr(data.get("artifacts", {}).get(key, {}).get("title", key)) for key in keys]


def node_description(data, key):
    """Return escaped, translated HTML describing a module and its data ports."""
    module = data["modules"][key]
    parts = [f'<b>{escape(tr(module["name"]))}</b>', escape(tr(module.get("guidance", "")))]
    for role, title in (("inputs", tr("Inputs")), ("outputs", tr("Outputs"))):
        entries = []
        for artifact in module.get(role, ()):
            info = data.get("artifacts", {}).get(artifact, {})
            entries.append(escape(tr(info.get("title", artifact))) + ": " +
                           escape(tr(info.get("location", ""))))
        parts.append(f'<b>{escape(title)}</b><br>' + '<br>'.join(entries))
    return '<br><br>'.join(parts)


def edge_description(data, edge):
    """Explain the direction, artifact locations and limits of a connection."""
    source, target = (tr(data["modules"][edge[k]]["name"]) for k in ("from", "to"))
    if edge["kind"] == "documented":
        explanation = tr(edge["handoff"])
    elif edge["kind"] == "compatible":
        explanation = tr("Matching data types. Check the destination's required fields, object identities and settings; this is not an automatic transfer.")
    else:
        explanation = tr("Pathway prerequisite. Follow the pathway instructions; no direct file handoff is declared for this pair.")
    artifacts = [data["artifacts"].get(key, {"title": key}) for key in edge["artifacts"]]
    return (f'<b>{escape(source)} → {escape(target)}</b><br><br>{escape(explanation)}' +
            ''.join('<br><br><b>' + escape(tr(item["title"])) + '</b><br>' +
                    escape(tr(item.get("location", ""))) for item in artifacts))


def _positions(keys, edges):
    """Lay out documented dependencies by depth; place remaining nodes below.

    Cycles are kept in a bounded final layer, so malformed or cyclic maps
    cannot make layout loop forever. Compatibility edges do not impose rank.
    """
    parents = {key: set() for key in keys}
    for edge in edges:
        if edge["kind"] != "compatible":
            parents[edge["to"]].add(edge["from"])
    ranked, pending = {}, list(keys)
    while pending:
        ready = [key for key in pending if parents[key] <= ranked.keys()]
        if not ready:
            for key in pending:
                ranked[key] = max(ranked.values(), default=0) + 1
            break
        for key in ready:
            ranked[key] = max((ranked[p] + 1 for p in parents[key]), default=0)
            pending.remove(key)
    columns = defaultdict(list)
    linked = {edge[k] for edge in edges if edge["kind"] != "compatible" for k in ("from", "to")}
    for key in keys:
        if key in linked:
            columns[ranked[key]].append(key)
    positions = {key: QPointF(depth * 325, row * 225)
                 for depth, group in columns.items() for row, key in enumerate(group)}
    bottom = max((point.y() + 260 for point in positions.values()), default=0)
    remaining = [key for key in keys if key not in positions]
    for index, key in enumerate(remaining):
        positions[key] = QPointF((index % 5) * 325, bottom + (index // 5) * 225)
    return positions


class _Node(QGraphicsItem):
    """One focusable module, including its input and output artifact titles."""

    def __init__(self, view, key):
        super().__init__()
        self.view, self.key = view, key
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsFocusable)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(2)
        self.highlighted = False

    def boundingRect(self):
        """Return stable geometry, independent of hover and selection."""
        return QRectF(0, 0, 260, 84 if self.view.compact else 180)

    def paint(self, painter, option, widget=None):
        """Paint a rounded module card with separate input and output rows."""
        palette = active_palette()
        painter.setRenderHint(QPainter.Antialiasing)
        fill = QColor(palette["surface_hi"])
        painter.setBrush(fill)
        painter.setPen(QPen(QColor(palette["accent"] if self.highlighted else palette["border"]), 1.5))
        painter.drawRoundedRect(self.boundingRect().adjusted(1, 1, -1, -1), 16, 16)
        font = QFont()
        font.setPixelSize(font_px(24 if self.view.compact else 17, scale=1))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(palette["fg"]))
        module = self.view.data["modules"][self.key]
        painter.drawText(QRectF(14, 9, 232, 66 if self.view.compact else 40), Qt.AlignVCenter | Qt.TextWordWrap, tr(module["name"]))
        if self.view.compact:
            return
        font.setBold(False)
        font.setPixelSize(font_px(14, scale=1))
        painter.setFont(font)
        for y, role, title in ((54, "inputs", tr("Inputs")), (113, "outputs", tr("Outputs"))):
            names = ', '.join(_artifact_names(self.view.data, module.get(role, ()))) or tr("None declared")
            painter.setPen(QColor(palette["accent"] if role == "outputs" else palette["fg_muted"]))
            text = title + ': ' + names
            metrics = QFontMetrics(font)
            words, lines, line = text.split(), [], ''
            for word in words:
                candidate = (line + ' ' + word).strip()
                if metrics.horizontalAdvance(candidate) > 232 and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            count = max(1, 57 // metrics.height())
            if len(lines) > count:
                lines[count - 1] = metrics.elidedText(' '.join(lines[count - 1:]), Qt.ElideRight, 232)
            painter.drawText(QRectF(14, y, 232, 57), Qt.AlignLeft, '\n'.join(lines[:count]))

    def hoverEnterEvent(self, event):
        """Describe this node without resizing the diagram."""
        self.view.describe_node(self.key)
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event):
        """Keep this node's description visible after the click."""
        self.view.describe_node(self.key)
        self.setFocus()
        event.accept()

    def focusInEvent(self, event):
        """Expose the same explanation to keyboard focus."""
        self.view.describe_node(self.key)
        super().focusInEvent(event)


class _Edge(QGraphicsPathItem):
    """Directed connection with a wider invisible hit target for hovering."""

    def __init__(self, view, edge, start, end):
        path = QPainterPath(start)
        bend = max(40, abs(end.x() - start.x()) / 2)
        path.cubicTo(start + QPointF(bend, 0), end - QPointF(bend, 0), end)
        super().__init__(path)
        self.view, self.edge = view, edge
        self.end = end
        self.highlighted = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIsFocusable)
        self.setZValue(0)

    def shape(self):
        """Make thin arrows selectable without requiring pixel-perfect aim."""
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())

    def boundingRect(self):
        """Include both the arrowhead and the wider hover target."""
        return self.path().boundingRect().adjusted(-14, -14, 14, 14)

    def paint(self, painter, option, widget=None):
        """Draw solid documented handoffs and dashed candidate connections."""
        palette = active_palette()
        colour = QColor(palette["accent"] if self.highlighted else palette["fg_muted"])
        colour.setAlphaF(1 if self.highlighted else (.55 if self.edge["kind"] == "documented" else .18))
        pen = QPen(colour, 2.2 if self.highlighted else 1)
        if self.edge["kind"] != "documented":
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())
        painter.setBrush(colour)
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([self.end, self.end + QPointF(-10, -5), self.end + QPointF(-10, 5)]))

    def hoverEnterEvent(self, event):
        """Explain the connection in the fixed details panel."""
        self.view.describe_edge(self)
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event):
        """Select this edge and retain its explanation."""
        self.view.describe_edge(self)
        self.setFocus()
        event.accept()


class WorkflowView(QGraphicsView):
    """Zoomable, pannable map whose selection emits explanatory HTML.

    :param data: shared workflow map.
    :param keys: optional subset of module keys for one pathway.
    :param steps: optional pathway dependencies; no sequence links are invented.
    :param parent: owning widget.
    :param compact: show module titles only; full input/output descriptions
        remain in the emitted explanation. Suitable for the complete network.
    """

    explanation = Signal(str)
    activated = Signal()

    def __init__(self, data, keys=None, steps=None, parent=None, *, compact=False):
        """Build nodes and directed connections once; hover changes only ink."""
        super().__init__(parent)
        self.compact = compact
        self._fitted = False
        self._auto_fit = True
        self.data = data
        self.keys = list(keys if keys is not None else data["modules"])
        self.links = connections(data, self.keys, steps)
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QGraphicsView { background: transparent; border: none; }")
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.nodes, self.edges = {}, []
        positions = _positions(self.keys, self.links)
        if compact:
            positions = {key: QPointF((i % 9) * 310, (i // 9) * 125)
                         for i, key in enumerate(self.keys)}
        for key in self.keys:
            node = _Node(self, key)
            node.setPos(positions[key])
            self.scene().addItem(node)
            self.nodes[key] = node
        for link in self.links:
            middle = 42 if compact else 90
            start = positions[link["from"]] + QPointF(260, middle)
            end = positions[link["to"]] + QPointF(0, middle)
            edge = _Edge(self, link, start, end)
            self.scene().addItem(edge)
            self.edges.append(edge)
        self.setSceneRect(self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20))

    def fit_diagram(self):
        """Fit every module into the viewport; preserve geometry and text size."""
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._fitted = True
        self._auto_fit = True

    def resizeEvent(self, event):
        """Refit after actual layout sizing until the user pans or zooms."""
        super().resizeEvent(event)
        if self._auto_fit and self.scene() is not None:
            self.fit_diagram()

    def showEvent(self, event):
        """Fit once when a viewport first acquires a useful size."""
        super().showEvent(event)
        if not self._fitted:
            self.fit_diagram()

    def zoom(self, factor):
        """Scale the view within readable bounds without relaying out nodes."""
        target = self.transform().m11() * factor
        if .04 <= target <= 3:
            self._auto_fit = False
            self.scale(factor, factor)

    def wheelEvent(self, event):
        """Use Ctrl+wheel for zoom; ordinary wheel scrolls the diagram."""
        if event.modifiers() & Qt.ControlModifier:
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        else:
            super().wheelEvent(event)

    def _highlight(self, keys, chosen=None):
        """Emphasize the selected neighborhood and keep unrelated edges faint."""
        for key, node in self.nodes.items():
            node.highlighted = key in keys
            node.update()
        for edge in self.edges:
            edge.highlighted = edge is chosen or (chosen is None and bool(keys & {edge.edge['from'], edge.edge['to']}))
            edge.setZValue(1 if edge.highlighted else 0)
            edge.update()

    def describe_node(self, key, *, center=False):
        """Describe a module; optionally zoom to it for keyboard selection."""
        self._highlight({key})
        self.explanation.emit(node_description(self.data, key))
        self.activated.emit()
        if center:
            self._auto_fit = False
            self.resetTransform()
            self.centerOn(self.nodes[key])

    def describe_edge(self, edge, *, center=False):
        """Describe a connection and highlight its two endpoint modules."""
        self._highlight({edge.edge['from'], edge.edge['to']}, edge)
        self.explanation.emit(edge_description(self.data, edge.edge))
        self.activated.emit()
        if center:
            self._auto_fit = False
            self.fitInView(edge.boundingRect().united(self.nodes[edge.edge['from']].sceneBoundingRect()).united(
                self.nodes[edge.edge['to']].sceneBoundingRect()).adjusted(-30, -30, 30, 30), Qt.KeepAspectRatio)


class DiagramDialog(QDialog):
    """A diagram window with a rounded, 80-percent opaque background."""

    def __init__(self, parent=None):
        """Keep the background translucent without reducing text opacity."""
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("QDialog { background: transparent; }")

    def paintEvent(self, event):
        """Paint one 80-percent surface beneath the diagram and its controls."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = active_palette()
        background = QColor(palette["surface"])
        background.setAlphaF(.8)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 16, 16)


def details_box(parent=None):
    """Build a fixed-height, scrollable explanation area shared by diagrams."""
    box = QTextBrowser(parent)
    box.setObjectName("WorkflowDetails")
    box.setFixedHeight(font_px("body") * 10)
    box.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
    box.setHtml(escape(tr("Hover or select a module or connection to read its inputs, outputs and explanation here.")))
    return box


class SpacrFlowchartDialog(DiagramDialog):
    """Show every mapped module and directed input/output connection.

    :param parent: owning main window.
    :param data: workflow map override; None reads the bundled map.
    """

    def __init__(self, parent=None, data=None):
        """Build the network, keyboard selectors, view controls and details."""
        super().__init__(parent)
        self.setObjectName("SpacrFlowchartDialog")
        self.setWindowTitle(tr("spaCR flowchart"))
        self.resize(1180, 780)
        layout = QVBoxLayout(self)
        note = QLabel(tr("Solid arrows: documented handoffs. Dashed arrows: matching data types, requiring compatibility checks. Drag to pan; Ctrl+wheel to zoom."))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.view = WorkflowView(data if data is not None else workflow_map(), parent=self, compact=True)
        toolbar = QHBoxLayout()
        self.module_picker = QComboBox()
        self.module_picker.setAccessibleName(tr("Module"))
        self.module_picker.addItem(tr("Select a module…"), None)
        for key in sorted(self.view.keys, key=lambda k: tr(self.view.data["modules"][k]["name"])):
            self.module_picker.addItem(tr(self.view.data["modules"][key]["name"]), key)
        self.module_picker.currentIndexChanged.connect(self._module_selected)
        toolbar.addWidget(self.module_picker, 1)
        self.edge_picker = QComboBox()
        self.edge_picker.setAccessibleName(tr("Connection"))
        self.edge_picker.addItem(tr("Select a connection…"), None)
        for index, edge in enumerate(self.view.edges):
            label = ' → '.join(tr(self.view.data["modules"][edge.edge[k]]["name"]) for k in ('from', 'to'))
            self.edge_picker.addItem(label, index)
        self.edge_picker.currentIndexChanged.connect(self._edge_selected)
        toolbar.addWidget(self.edge_picker, 1)
        for text, action in ((tr("−"), lambda: self.view.zoom(1 / 1.2)),
                             (tr("+"), lambda: self.view.zoom(1.2)),
                             (tr("Fit"), self.view.fit_diagram)):
            button = QPushButton(text)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        self.compatible = QCheckBox(tr("Show matching data types"))
        self.compatible.setChecked(False)
        self.compatible.toggled.connect(self._show_compatible)
        layout.addWidget(self.compatible)
        self._show_compatible(False)
        layout.addWidget(self.view, 1)
        self.details = details_box(self)
        self.view.explanation.connect(self.details.setHtml)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _module_selected(self, index):
        """Focus a module selected with the keyboard or module picker."""
        key = self.module_picker.itemData(index)
        if key is not None:
            self.view.describe_node(key, center=True)

    def _edge_selected(self, index):
        """Focus a selected edge and display its complete handoff explanation."""
        number = self.edge_picker.itemData(index)
        if number is not None:
            if self.view.edges[number].edge['kind'] != 'documented':
                self.compatible.setChecked(True)
            self.view.describe_edge(self.view.edges[number], center=True)

    def _show_compatible(self, show):
        """Reveal candidate data-type links without confusing them with handoffs."""
        for edge in self.view.edges:
            edge.setVisible(show or edge.edge['kind'] == 'documented')


def show_spacr_flowchart(window):
    """Open or raise the window's single nonmodal spaCR workflow diagram."""
    dialog = getattr(window, "_spacr_flowchart", None)
    if dialog is None:
        dialog = SpacrFlowchartDialog(window)
        window._spacr_flowchart = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
