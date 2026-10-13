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
    QLabel, QPushButton, QTextBrowser, QVBoxLayout, QSplitter,
)

from ..i18n import tr
from ..theme import active_palette, font_px


def workflow_map(path=None):
    """Read the bundled module map, or an explicitly supplied JSON path.

    :param path: JSON file; None selects the bundled module_workflows.json.
    :returns: the parsed map; modules and artifacts must be dictionaries.
        The bundled map also supplies pathways and connections.
    :raises OSError: the file cannot be read.
    :raises ValueError: invalid JSON or missing module/artifact dictionaries.
    """
    source = Path(path) if path is not None else (
        Path(__file__).resolve().parents[2] / "resources/module_workflows.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or not isinstance(data.get("modules"), dict)
            or not isinstance(data.get("artifacts"), dict)):
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


def _api_link(data, key):
    """Link a mapped module to the same localized API route used by Help."""
    from ..help_search import api_url

    module = data['modules'][key]
    symbol = module.get('api_module')
    if not symbol:
        return ''
    caption = tr("{module} API", module=tr(module['name']))
    return f'<a href="{escape(api_url(symbol), quote=True)}">{escape(caption)}</a>'


def node_description(data, key):
    """Describe a module, its API link and its input/output artifacts.

    :param data: shared workflow map.
    :param key: module key present in the map.
    :returns: escaped, translated HTML for the fixed details panel.
    """
    module = data["modules"][key]
    parts = [f'<b>{escape(tr(module["name"]))}.</b>',
             escape(tr(module.get("guidance", "")))]
    for role, title in (("inputs", tr("Inputs")), ("outputs", tr("Outputs"))):
        if key.startswith("input:") and not module.get(role):
            continue
        entries = []
        for artifact in module.get(role, ()):
            info = data.get("artifacts", {}).get(artifact, {})
            location = tr(info.get("location", "")).rstrip('.')
            entries.append(escape(tr(info.get("title", artifact))) +
                           (' (' + escape(location) + ')' if location else ''))
        parts.append(f'<b>{escape(title)}:</b> ' +
                     ('; '.join(entries) or escape(tr('None declared'))) + '.')
    parts.append(_api_link(data, key))
    return ' '.join(part for part in parts if part)


def edge_description(data, edge):
    """Explain a connection and link both endpoint APIs.

    :param data: shared workflow map containing both endpoint modules.
    :param edge: connection record returned by :func:`connections`.
    :returns: escaped, translated HTML describing artifact locations and
        whether the edge is a documented handoff, a matching data type or a
        pathway prerequisite. Inferred links do not establish file compatibility.
    """
    source, target = (tr(data["modules"][edge[k]]["name"]) for k in ("from", "to"))
    if edge["kind"] == "documented":
        explanation = tr(edge["handoff"])
    elif edge["kind"] == "compatible":
        explanation = tr("Matching data types. Check the destination's required fields, object identities and settings; this is not an automatic transfer.")
    else:
        explanation = tr("Pathway prerequisite. Follow the pathway instructions; no direct file handoff is declared for this pair.")
    artifacts = [data["artifacts"].get(key, {"title": key}) for key in edge["artifacts"]]
    return (f'<b>{escape(source)} → {escape(target)}.</b> {escape(explanation)} ' +
            ' '.join('<b>' + escape(tr(item["title"])) + ':</b> ' +
                     escape(tr(item.get("location", ""))) for item in artifacts) + ' ' +
            ' · '.join(link for link in (_api_link(data, edge['from']),
                                        _api_link(data, edge['to'])) if link))


def _positions(keys, edges, data=None, *, compact=False):
    """Lay out documented dependencies in columns ordered by data flow.

    Independent sources move beside the latest branch they feed, unless
    that would delay an earlier dependency chain. Terminal consumers spread
    across later columns. Modules without documented handoffs are placed by
    their input/output types. Within each column, neighboring branches are
    aligned while the main pipeline stays first. Cycles remain bounded
    rather than looping.
    """
    parents = {key: set() for key in keys}
    children = {key: set() for key in keys}
    for edge in edges:
        if edge["kind"] != "compatible":
            parents[edge["to"]].add(edge["from"])
            children[edge["from"]].add(edge["to"])
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
    linked = {edge[k] for edge in edges if edge["kind"] != "compatible" for k in ("from", "to")}
    for key in keys:
        if parents[key] or not children[key]:
            continue
        reachable, waiting = set(), list(children[key])
        while waiting:
            child = waiting.pop()
            for descendant in children[child] - reachable:
                reachable.add(descendant)
                waiting.append(descendant)
        if not children[key] & reachable:
            ranked[key] = max(ranked[child] for child in children[key]) - 1
    for key in sorted(keys, key=lambda k: ranked[k]):
        if parents[key]:
            ranked[key] = max(ranked[key], max(ranked[parent] + 1 for parent in parents[key]))
    columns = defaultdict(list)
    core = [key for key in ('mask', 'measure', 'annotate', 'classify_merged', 'regression') if key in keys]
    ordered = sorted(keys, key=lambda key: (key not in core, not bool(parents[key]), keys.index(key)))
    maximum = max(ranked.values(), default=0)
    for key in ordered:
        if key in linked and (children[key] or key in core or not compact):
            columns[ranked[key]].append(key)
    for key in ordered:
        if key in linked and not children[key] and key not in core and compact:
            depth = min(range(ranked[key], maximum + 1),
                        key=lambda d: len(columns[d]) + .3 * (d - ranked[key]))
            ranked[key] = depth
            columns[depth].append(key)
    modules = (data or {}).get('modules', {})
    for key in ordered:
        if key in linked:
            continue
        inputs = set(modules.get(key, {}).get('inputs', ()))
        outputs = set(modules.get(key, {}).get('outputs', ()))
        producers = [ranked[k] for k in linked if inputs & set(modules.get(k, {}).get('outputs', ()))]
        consumers = [ranked[k] for k in linked if outputs & set(modules.get(k, {}).get('inputs', ()))]
        first = min(maximum, min(producers) + 1) if producers else 0
        last = max(first, min(consumers) - 1) if consumers else maximum
        suggested = min(range(first, last + 1),
                        key=lambda d: len(columns[d]) + .2 * (d - first))
        ranked[key] = suggested
        columns[suggested].append(key)
    for key in sorted(keys, key=lambda k: ranked[k], reverse=True):
        if key not in core and children[key]:
            latest = min(ranked[child] for child in children[key]) - 1
            if latest > ranked[key]:
                columns[ranked[key]].remove(key)
                ranked[key] = latest
                columns[latest].append(key)
    rows = {key: row for group in columns.values() for row, key in enumerate(group)}
    for _ in range(2):
        for neighbors, reverse in ((parents, False), (children, True)):
            for depth in sorted(columns, reverse=reverse):
                group = columns[depth]
                scores = {
                    key: (key not in core, not bool(neighbors[key]),
                          sum(rows[peer] for peer in neighbors[key]) / len(neighbors[key])
                          if neighbors[key] else rows[key], rows[key])
                    for key in group
                }
                group.sort(key=scores.__getitem__)
                rows.update((key, row) for row, key in enumerate(group))
    row_height = 125 if compact else 225
    return {key: QPointF(depth * 325, row * row_height)
            for depth, group in sorted(columns.items()) for row, key in enumerate(group)}


class _Node(QGraphicsItem):
    """One focusable module, including its input and output artifact titles."""

    def __init__(self, view, key):
        """Create a focusable module node associated with its workflow and registry key."""
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
        font.setPixelSize(font_px((19 if self.key.startswith("input:") else 24) if self.view.compact else 17, scale=1))
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
        """Build a smooth connection ending at the arrow base and retain its handoff metadata."""
        base = end - QPointF(12, 0)
        path = QPainterPath(start)
        bend = max(24, abs(base.x() - start.x()) / 2)
        path.cubicTo(start + QPointF(bend, 0), base - QPointF(bend, 0), base)
        super().__init__(path)
        self.view, self.edge = view, edge
        self.end = end
        self.arrowhead = QPolygonF([end, base + QPointF(0, -6), base + QPointF(0, 6)])
        self.highlighted = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIsFocusable)
        self.setZValue(0)

    def shape(self):
        """Make thin arrows selectable without requiring pixel-perfect aim."""
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        target = stroker.createStroke(self.path())
        target.addPolygon(self.arrowhead)
        return target

    def boundingRect(self):
        """Include both the arrowhead and the wider hover target."""
        return self.path().boundingRect().adjusted(-14, -14, 14, 14)

    def paint(self, painter, option, widget=None):
        """Draw solid documented handoffs and dashed candidate connections."""
        palette = active_palette()
        colour = QColor(palette["accent"] if self.highlighted else palette["fg_muted"])
        colour.setAlphaF(1 if self.highlighted else (.55 if self.edge["kind"] == "documented" else .18))
        pen = QPen(colour, 2.2 if self.highlighted else 1)
        pen.setCapStyle(Qt.FlatCap)
        if self.edge["kind"] != "documented":
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())
        painter.setBrush(colour)
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(self.arrowhead)

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
    selection_changed = Signal(str)

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
        positions = _positions(self.keys, self.links, data, compact=compact)
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
        """Fit the complete diagram with preserved aspect ratio and resume auto-fit.

        :returns: None; updates the view transform and automatic resize behavior.
        """
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._fitted = True
        self._auto_fit = True

    def resizeEvent(self, event):
        """Refit on resize while automatic fitting remains enabled.

        Manual zoom or explicit centering disables automatic fitting until
        :meth:`fit_diagram` is called again.

        :param event: Qt resize event forwarded to the base view.
        :returns: None.
        """
        super().resizeEvent(event)
        if self._auto_fit and self.scene() is not None:
            self.fit_diagram()

    def showEvent(self, event):
        """Fit once when a viewport first acquires a useful size.

        :param event: Qt show event forwarded to the base view.
        :returns: None.
        """
        super().showEvent(event)
        if not self._fitted:
            self.fit_diagram()

    def zoom(self, factor):
        """Scale the view without changing node layout.

        :param factor: multiplier for the current view scale. The change is
            applied only when the resulting scale is between 0.04 and 3,
            inclusive; accepted changes disable automatic fitting.
        :returns: None.
        """
        target = self.transform().m11() * factor
        if .04 <= target <= 3:
            self._auto_fit = False
            self.scale(factor, factor)

    def wheelEvent(self, event):
        """Use Ctrl+wheel for zoom; ordinary wheel scrolls the diagram.

        :param event: Qt wheel event; control-modified events are accepted
            here, and other events are forwarded to the base view.
        :returns: None.
        """
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
        """Highlight a module and emit its explanation without running analysis.

        :param key: module key present in this view.
        :param center: when True, reset the zoom and center the module,
            disabling automatic fitting; defaults to False.
        :returns: None; emits explanation HTML and the activated signal.
        """
        self._highlight({key})
        self.activated.emit()
        self.explanation.emit(node_description(self.data, key))
        self.selection_changed.emit(key)
        if center:
            self._auto_fit = False
            self.resetTransform()
            self.centerOn(self.nodes[key])

    def describe_edge(self, edge, *, center=False):
        """Describe a connection and highlight its two endpoint modules.

        :param edge: graphical edge item from this view's edges list.
        :param center: when True, fit the edge and both endpoint nodes into
            the viewport and disable automatic fitting; defaults to False.
        :returns: None; emits explanation HTML and the activated signal.
        """
        self._highlight({edge.edge['from'], edge.edge['to']}, edge)
        self.activated.emit()
        self.explanation.emit(edge_description(self.data, edge.edge))
        self.selection_changed.emit(edge.edge["from"] + "→" + edge.edge["to"])
        if center:
            self._auto_fit = False
            self.fitInView(edge.boundingRect().united(self.nodes[edge.edge['from']].sceneBoundingRect()).united(
                self.nodes[edge.edge['to']].sceneBoundingRect()).adjusted(-30, -30, 30, 30), Qt.KeepAspectRatio)


class DiagramDialog(QDialog):
    """A diagram window with a rounded, 80-percent opaque background.

    :param parent: owning widget; defaults to None for a top-level window.
    """

    def __init__(self, parent=None):
        """Keep the background translucent without reducing text opacity."""
        super().__init__(parent)
        self.setProperty("spacrNoGlass", True)
        from .glass import make_frameless, install_glass_everywhere

        install_glass_everywhere()
        make_frameless(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("QDialog { background: transparent; }")

    def paintEvent(self, event):
        """Paint one 80-percent surface beneath the diagram and its controls.

        :param event: Qt paint event; the complete background is repainted.
        :returns: None; text and child controls retain their own opacity.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = active_palette()
        background = QColor(palette["surface"])
        background.setAlphaF(.8)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 16, 16)


def details_box(parent=None):
    """Build a resizable, scrollable explanation area shared by diagrams.

    :param parent: owning widget; defaults to None.
    :returns: QTextBrowser with external API links enabled and introductory
        text. Its minimum height is six body-font lines; the splitter sets its height.
    """
    box = QTextBrowser(parent)
    box.setObjectName("WorkflowDetails")
    box.setMinimumHeight(font_px("body") * 6)
    box.setOpenExternalLinks(True)
    palette = active_palette()
    colour = QColor(palette["surface_hi"])
    box.setStyleSheet(f"QTextBrowser {{ background: rgba({colour.red()}, {colour.green()}, {colour.blue()}, 150); "
                     f"border: 1px solid {palette['border']}; border-radius: 12px; padding: 12px; "
                     f"font-size: {font_px('body') + 2}px; }}")
    box.setHtml(escape(tr("Hover or select a module or connection to read its inputs, outputs and explanation here.")))
    return box


def diagram_splitter(parent=None):
    """Build a vertical pane divider with a thin blue, draggable handle.

    :param parent: owning dialog or container.
    :returns: non-collapsing QSplitter; callers add the diagram and details.
    """
    splitter = QSplitter(Qt.Vertical, parent)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(1)
    splitter.setStyleSheet("QSplitter::handle:vertical { background: #168cff; }")
    return splitter


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
        note = QLabel(tr("Solid arrows: documented handoffs. Dashed arrows: matching data types, requiring compatibility checks."))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.view = WorkflowView(data if data is not None else workflow_map(), parent=self)
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
        from PySide6.QtGui import QKeySequence
        modifier = QKeySequence("Ctrl+Z").toString(QKeySequence.NativeText).removesuffix("Z").rstrip("+")
        self.navigation_hint = QLabel(tr(
            "Hold {key} and scroll the mouse wheel to zoom. Drag empty space to move around. Fit shows the whole map.",
            key=modifier))
        self.navigation_hint.setObjectName("WorkflowNavigationHint")
        self.navigation_hint.setWordWrap(True)
        layout.addWidget(self.navigation_hint)
        self._show_compatible(False)
        self.splitter = diagram_splitter(self)
        self.splitter.addWidget(self.view)
        layout.addWidget(self.splitter, 1)
        self.details = details_box(self)
        self.view.explanation.connect(self.details.setHtml)
        self.splitter.addWidget(self.details)
        self.splitter.setSizes([460, 240])
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
    """Open or raise the window's single nonmodal spaCR workflow diagram.

    :param window: owning main window, which retains the dialog for reuse.
    :returns: the shown and activated :class:`SpacrFlowchartDialog`.
    """
    dialog = getattr(window, "_spacr_flowchart", None)
    if dialog is None:
        dialog = SpacrFlowchartDialog(window)
        window._spacr_flowchart = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
