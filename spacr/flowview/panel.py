"""Optional, self-contained Qt live panel for a FlowView collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .collector import Collector
from .export import export as export_graph
from .layout import GraphLayout, layout_graph
from .model import Edge, Node, RunGraph
from .theme import CANVAS, CARD, TEXT_PRIMARY, TEXT_SECONDARY
from .trace import get_collector

QT_INSTALL_COMMAND = "pip install spacr[flowview]"
QT_MISSING_MESSAGE = (
    "The FlowView live panel requires PySide6. "
    f"Install it with `{QT_INSTALL_COMMAND}`."
)

try:
    from PySide6.QtCore import QRectF, Qt, QTimer
    from PySide6.QtGui import QBrush, QCloseEvent, QColor, QPainter, QWheelEvent
    from PySide6.QtWidgets import (
        QFileDialog,
        QFrame,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    from .items import EdgeItem, NodeItem
except ImportError as error:
    QT_AVAILABLE = False
    _QT_IMPORT_ERROR = error
else:
    QT_AVAILABLE = True
    _QT_IMPORT_ERROR = None


if not QT_AVAILABLE:

    class _MissingQtWidget:
        """Placeholder that names the command needed for live rendering."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ImportError(QT_MISSING_MESSAGE) from _QT_IMPORT_ERROR

    class FlowGraphicsView(_MissingQtWidget):
        """Unavailable graphics-view placeholder."""

    class FlowViewPanel(_MissingQtWidget):
        """Unavailable live-panel placeholder."""

else:

    def _topology_key(graph: RunGraph) -> tuple[tuple[str, ...], tuple[str, ...]]:
        nodes = tuple(sorted(graph.nodes))
        edges = tuple(sorted(repr(edge) for edge in graph.edges))
        return nodes, edges


    def _json_mapping(values: dict[str, object]) -> str:
        return json.dumps(
            values,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
            default=str,
        )


    def inspector_text(node: Node) -> str:
        """Format complete, copyable details for the selected node."""

        if node.started_at is not None and node.ended_at is not None:
            duration: object = node.ended_at - node.started_at
        else:
            duration = "—"
        progress: object = list(node.progress) if node.progress is not None else "—"
        return "\n".join(
            (
                node.label,
                f"Identifier: {node.id}",
                f"Kind: {node.kind.value}",
                f"State: {node.state.value}",
                f"Started: {node.started_at if node.started_at is not None else '—'}",
                f"Ended: {node.ended_at if node.ended_at is not None else '—'}",
                f"Duration: {duration}",
                f"Progress: {json.dumps(progress, ensure_ascii=False)}",
                "",
                "Parameters:",
                _json_mapping(node.params),
                "",
                "Metrics:",
                _json_mapping(node.metrics),
                "",
                "Error:",
                node.error if node.error is not None else "—",
            )
        )


    class FlowGraphicsView(QGraphicsView):
        """A panning graphics view with bounded cursor-centred wheel zoom."""

        MIN_ZOOM = 0.25
        MAX_ZOOM = 4.0
        ZOOM_STEP = 1.15

        def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
            super().__init__(scene, parent)
            self._zoom = 1.0
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setViewportUpdateMode(
                QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
            )
            self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setBackgroundBrush(QBrush(QColor(CANVAS)))

        @property
        def zoom(self) -> float:
            """Current bounded zoom multiplier."""

            return self._zoom

        def zoom_by(self, wheel_delta: int) -> bool:
            """Apply one wheel zoom step and report whether the view changed."""

            if wheel_delta == 0:
                return False
            requested = self._zoom * (
                self.ZOOM_STEP if wheel_delta > 0 else 1.0 / self.ZOOM_STEP
            )
            bounded = max(self.MIN_ZOOM, min(self.MAX_ZOOM, requested))
            if bounded == self._zoom:
                return False
            self.scale(bounded / self._zoom, bounded / self._zoom)
            self._zoom = bounded
            return True

        def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt virtual name
            """Translate vertical wheel movement into bounded zoom."""

            if self.zoom_by(event.angleDelta().y()):
                event.accept()
            else:
                event.ignore()


    class FlowViewPanel(QWidget):
        """A boxed live run graph suitable for insertion below Classify settings."""

        SAMPLE_NOTE = "Live display is sampling updates because the event queue filled."

        def __init__(
            self,
            collector: Collector,
            parent: QWidget | None = None,
            *,
            refresh_interval_ms: int = 50,
            export_path_provider: Callable[[], str | Path | None] | None = None,
            auto_start: bool = True,
            embedded: bool = False,
        ) -> None:
            if refresh_interval_ms <= 0:
                raise ValueError("refresh_interval_ms must be greater than zero")
            super().__init__(parent)
            self.setObjectName("FlowViewPanel")
            self._embedded = bool(embedded)
            panel_surface = (
                "background: transparent; border: none;"
                if self._embedded
                else f"background: {CARD}; border: 1px solid #FFFFFF1A;"
            )
            self.setStyleSheet(
                f"#FlowViewPanel {{ {panel_surface} }}"
                f"QLabel, QPlainTextEdit {{ color: {TEXT_PRIMARY}; }}"
                f"QPlainTextEdit {{ background: {CANVAS}; border: 1px solid #FFFFFF1A; }}"
            )
            self._collector = collector
            try:
                self._follow_global_collector = collector is get_collector()
            except Exception:
                self._follow_global_collector = False
            self._export_path_provider = export_path_provider
            self._snapshot = collector.snapshot()
            self._revision = -1
            self._topology: tuple[tuple[str, ...], tuple[str, ...]] | None = None
            self._layout: GraphLayout | None = None
            self._selected_node_id: str | None = None
            self._node_items: dict[str, NodeItem] = {}
            self._edge_items: dict[Edge, EdgeItem] = {}

            outer = QVBoxLayout(self)
            outer.setContentsMargins(10, 10, 10, 10)
            outer.setSpacing(8)
            toolbar = QHBoxLayout()
            self.title_label = QLabel("FlowView")
            self.title_label.setObjectName("FlowViewTitle")
            self.title_label.setVisible(not self._embedded)
            toolbar.addWidget(self.title_label)
            toolbar.addStretch(1)
            self.export_button = QPushButton("Export…")
            self.export_button.setToolTip("Export this run as SVG, HTML, or JSON")
            toolbar.addWidget(self.export_button)
            outer.addLayout(toolbar)

            self.sample_note = QLabel(self.SAMPLE_NOTE)
            self.sample_note.setStyleSheet(f"color: {TEXT_SECONDARY};")
            self.sample_note.setWordWrap(True)
            self.sample_note.hide()
            outer.addWidget(self.sample_note)

            self.scene = QGraphicsScene(self)
            self.scene.setBackgroundBrush(QBrush(QColor(CANVAS)))
            self.view = FlowGraphicsView(self.scene, self)
            self.inspector = QPlainTextEdit(self)
            self.inspector.setReadOnly(True)
            self.inspector.setPlaceholderText("Select a stage to inspect its run details.")
            self.inspector.setMinimumHeight(118)
            splitter = QSplitter(Qt.Orientation.Vertical, self)
            splitter.addWidget(self.view)
            splitter.addWidget(self.inspector)
            splitter.setStretchFactor(0, 4)
            splitter.setStretchFactor(1, 1)
            outer.addWidget(splitter, 1)

            self.export_status = QLabel("")
            self.export_status.setStyleSheet(f"color: {TEXT_SECONDARY};")
            outer.addWidget(self.export_status)

            self.export_button.clicked.connect(self._export_current)
            self.scene.selectionChanged.connect(self._selection_changed)
            self.timer = QTimer(self)
            self.timer.setInterval(int(refresh_interval_ms))
            self.timer.timeout.connect(self.refresh)
            self.refresh(force=True)
            if auto_start:
                self.start()

        @property
        def graph(self) -> RunGraph:
            """Most recent immutable snapshot displayed by the panel."""

            return self._snapshot

        def start(self) -> bool:
            """Start approximately 20 Hz collection, unless already active."""

            if self.timer.isActive():
                return False
            self.timer.start()
            return True

        def stop(self) -> bool:
            """Stop collection, returning whether a running timer was stopped."""

            if not self.timer.isActive():
                return False
            self.timer.stop()
            return True

        def refresh(self, *, force: bool = False) -> bool:
            """Drain events and repaint only when the graph content changed."""

            if self._follow_global_collector:
                try:
                    current_collector = get_collector()
                except Exception:
                    current_collector = self._collector
                if current_collector is not self._collector:
                    self._collector = current_collector
                    self._revision = -1

            self._collector.drain()
            self.sample_note.setVisible(bool(self._collector.sampled))
            revision = self._collector.revision
            if not force and revision == self._revision:
                return False
            graph = self._collector.snapshot()
            self._snapshot = graph
            self._render_graph(graph)
            self._revision = revision
            return True

        def _render_graph(self, graph: RunGraph) -> None:
            layout = layout_graph(graph)
            topology = _topology_key(graph)
            if topology != self._topology or layout != self._layout:
                self._rebuild_scene(graph, layout)
            else:
                for node_id, item in self._node_items.items():
                    item.update_node(graph.nodes[node_id], layout[node_id])
                for edge, item in self._edge_items.items():
                    item.set_source_running(
                        graph.nodes[edge.src].state.value == "running"
                    )
            self._topology = topology
            self._layout = layout
            if self._selected_node_id in graph.nodes:
                self._show_inspector(graph.nodes[self._selected_node_id])
            elif self._selected_node_id is not None:
                self._selected_node_id = None
                self.inspector.clear()

        def _rebuild_scene(self, graph: RunGraph, layout: GraphLayout) -> None:
            selected_node_id = self._selected_node_id
            self.scene.clear()
            self._node_items.clear()
            self._edge_items.clear()
            for edge in sorted(graph.edges, key=repr):
                item = EdgeItem(
                    edge,
                    layout[edge.src],
                    layout[edge.dst],
                    source_running=graph.nodes[edge.src].state.value == "running",
                )
                self.scene.addItem(item)
                self._edge_items[edge] = item
            for node_id in sorted(graph.nodes):
                item = NodeItem(graph.nodes[node_id], layout[node_id])
                self.scene.addItem(item)
                self._node_items[node_id] = item
            self.scene.setSceneRect(QRectF(0.0, 0.0, layout.width, layout.height))
            if selected_node_id in self._node_items:
                self._selected_node_id = selected_node_id
                self._node_items[selected_node_id].setSelected(True)

        def _selection_changed(self) -> None:
            selected = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, NodeItem)
            ]
            if selected:
                self._selected_node_id = selected[0].node_id
                self._show_inspector(selected[0].node)
            else:
                self._selected_node_id = None
                self.inspector.clear()

        def _show_inspector(self, node: Node) -> None:
            self.inspector.setPlainText(inspector_text(node))

        def _export_current(self) -> Path | None:
            if self._export_path_provider is None:
                chosen, _filter = QFileDialog.getSaveFileName(
                    self,
                    "Export FlowView run",
                    f"{self._snapshot.run_id}.svg",
                    "SVG (*.svg);;HTML (*.html);;JSON (*.json)",
                )
                path: str | Path | None = chosen
            else:
                path = self._export_path_provider()
            if not path:
                self.export_status.setText("Export cancelled.")
                return None
            suffix = Path(path).suffix.casefold().lstrip(".")
            fmt = suffix if suffix in {"svg", "html", "json"} else "svg"
            try:
                exported = export_graph(self._snapshot, path, fmt=fmt)
            except Exception as error:
                self.export_status.setText(f"Export failed: {error}")
                return None
            self.export_status.setText(f"Exported {exported.name}")
            return exported

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt virtual name
            """Stop timers and release scene-owned graphics objects."""

            self.stop()
            self.scene.clear()
            self._node_items.clear()
            self._edge_items.clear()
            super().closeEvent(event)


__all__ = [
    "FlowGraphicsView",
    "FlowViewPanel",
    "QT_AVAILABLE",
    "QT_INSTALL_COMMAND",
    "QT_MISSING_MESSAGE",
]

if QT_AVAILABLE:
    __all__.append("inspector_text")
