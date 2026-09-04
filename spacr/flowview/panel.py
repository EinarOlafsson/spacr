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

#: The hairline a divider is drawn as, and what it becomes under the pointer.
#:
#: WRITTEN AS `rgba()`, NOT `#RRGGBBAA`. A Qt stylesheet reads an eight-digit
#: hex as #AARRGGBB, so `#FFFFFF1A` -- white at 10% alpha in CSS -- parsed as
#: opaque rgb(255, 255, 26) and drew the bright yellow rim reported here on
#: 2026-09-01. `rgba()` means the same thing in both dialects.
_SPLIT_LINE = "rgba(255, 255, 255, 0.16)"
_SPLIT_HOVER = "rgba(95, 168, 199, 0.9)"

QT_INSTALL_COMMAND = "pip install spacr[flowview]"
QT_MISSING_MESSAGE = (
    "The FlowView live panel requires PySide6. "
    f"Install it with `{QT_INSTALL_COMMAND}`."
)

try:
    from PySide6.QtCore import QRectF, QSize, Qt, QTimer
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
            """Refuse construction, naming the missing Qt import as the cause."""
            del args, kwargs
            raise ImportError(QT_MISSING_MESSAGE) from _QT_IMPORT_ERROR

    class FlowGraphicsView(_MissingQtWidget):
        """Placeholder used when PySide6 is unavailable.

        Construction raises :class:`ImportError` with the
        ``pip install spacr[flowview]`` remediation.
        """

    class FlowViewPanel(_MissingQtWidget):
        """Placeholder used when PySide6 is unavailable.

        Construction raises :class:`ImportError` with the
        ``pip install spacr[flowview]`` remediation.
        """

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
        """With PySide6, ``FlowGraphicsView`` provides panning and bounded zoom.

        :param scene: graphics scene displayed by the view.
        :param parent: optional Qt parent widget.
        """

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
            # TRANSPARENT, not CANVAS. This brush is the near-black rectangle
            # reported on 2026-09-01, and it is set HERE -- clearing the
            # scene's brush in the panel left this one painting over it, which
            # is why the box stayed black through the first attempt. The
            # viewport must stop filling itself as well, or Qt paints the
            # palette colour underneath before either brush is consulted.
            self.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
            self.viewport().setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

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


    class _PanelHeightGrip(QFrame):
        """Thin drag handle along the panel's lower edge.

        The panel is one row of a scrolling settings column, so without this
        it gets whatever height the column gives it and puts the graph and
        the inspector both behind an inner scrollbar inside an outer one.
        Dragging this sets the height; double-clicking gives it back.

        :param panel: the :class:`FlowViewPanel` this resizes. ALSO ITS
            QWIDGET PARENT, so the grip is laid out under the panel it drags
            and cannot outlive it.
        """

        HEIGHT = 7

        def __init__(self, panel: "FlowViewPanel") -> None:
            """Build the handle and give it a vertical-resize cursor."""
            super().__init__(panel)
            self._panel = panel
            self._press_y: float | None = None
            self._start_height = 0
            self.setCursor(Qt.SizeVerCursor)
            self.setFixedHeight(self.HEIGHT)
            # Drawn here rather than by the app theme: FlowView renders
            # standalone too, and must not need the Qt palette to have a
            # visible edge.
            self.setStyleSheet(
                "QFrame { background: transparent; border: none;"
                f" border-bottom: 1px solid {_SPLIT_LINE}; }}"
                "QFrame:hover {"
                f" border-bottom: 2px solid {_SPLIT_HOVER}; }}")
            self.setToolTip("Drag to make FlowView taller or shorter. "
                            "Double-click to fit its contents.")

        def sizeHint(self) -> QSize:                     # noqa: N802
            """Wide and thin -- the handle is an edge, not a bar."""
            return QSize(80, self.HEIGHT)

        def mousePressEvent(self, event) -> None:        # noqa: N802
            """Remember where the drag started, and from what height."""
            if event.button() == Qt.LeftButton:
                self._press_y = event.globalPosition().y()
                self._start_height = self._panel.height()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:         # noqa: N802
            """Resize by how far the pointer moved SINCE THE PRESS.

            Measured from the press rather than the last move, so a drag
            that outruns the redraw lands where the pointer is instead of
            accumulating rounding.
            """
            if self._press_y is not None and event.buttons() & Qt.LeftButton:
                delta = event.globalPosition().y() - self._press_y
                self._panel.set_user_height(self._start_height + int(delta))
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event) -> None:      # noqa: N802
            """End the drag."""
            self._press_y = None
            super().mouseReleaseEvent(event)

        def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
            """Give the height back to the content."""
            if event.button() == Qt.LeftButton:
                self._panel.reset_user_height()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)


    class FlowViewPanel(QWidget):
        """With PySide6, ``FlowViewPanel`` renders and inspects live snapshots.

        :param collector: event collector whose snapshots the panel renders.
        :param parent: optional Qt parent widget.
        :param refresh_interval_ms: positive timer interval, in milliseconds,
            between automatic refreshes.
        :param export_path_provider: optional callback returning the export
            destination, or ``None`` to cancel. Without one, export opens a
            Qt file dialog.
        :param auto_start: whether to start the refresh timer after the initial
            forced render.
        :param embedded: whether to use a transparent, borderless surface and
            hide the panel title for embedding in another screen.
        """

        SAMPLE_NOTE = "Live display is sampling updates because the event queue filled."

        #: How tall the inspector starts. 118 px was about four lines, so
        #: every stage worth inspecting needed scrolling before it could be
        #: read. Start taller while keeping the inspector expandable.
        INSPECTOR_MIN_HEIGHT = 240

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
            # NO RIM, and the reason is worth keeping: the border here read
            # `#FFFFFF1A`, which in a CSS file means white at 10% alpha
            # (#RRGGBBAA) but in a QT STYLESHEET is parsed as #AARRGGBB --
            # opaque rgb(255, 255, 26). That is the bright yellow rectangle
            # around the inspector, reported on 2026-09-01 as "the yellow rim".
            # The same literal in `export.py` is correct, because that one
            # really is CSS and a browser really does read #RRGGBBAA.
            #
            # It is removed rather than corrected to a faint white, which is
            # what was asked for: the panel sits inside a section that already
            # draws the only box this needs.
            panel_surface = (
                "background: transparent; border: none;"
                if self._embedded
                else "background: transparent; border: none;"
                     " border-radius: 8px;"
            )
            self.setStyleSheet(
                f"#FlowViewPanel {{ {panel_surface} }}"
                f"QLabel, QPlainTextEdit {{ color: {TEXT_PRIMARY}; }}"
                # TRANSPARENT, ROUNDED, RIMLESS. The inspector was a black
                # rectangle with the yellow border above; it now shows the
                # page behind it like every other surface on the screen.
                "QPlainTextEdit {"
                " background: transparent; border: none;"
                " border-radius: 8px; }"
            )
            #: Pixels the user dragged the panel to, or None for content.
            self._user_height: int | None = None
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
            # THE OTHER BLACK BOX. The scene painted CANVAS (#0E1216), which
            # is a near-black rectangle sitting on top of whatever the screen
            # behind it is showing. Transparent lets the page through, and the
            # nodes carry their own fills so nothing becomes unreadable.
            self.scene.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
            self.view = FlowGraphicsView(self.scene, self)
            # THE SCENE BRUSH IS NOT ENOUGH. A QGraphicsView paints its own
            # widget background and its viewport's before the scene is drawn,
            # so clearing only the brush left the same near-black rectangle on
            # screen. All three have to give way for the page to show through.
            self.view.setStyleSheet(
                "QGraphicsView { background: transparent; border: none;"
                " border-radius: 8px; }")
            self.view.setFrameShape(QFrame.Shape.NoFrame)
            self.view.viewport().setAutoFillBackground(False)
            self.inspector = QPlainTextEdit(self)
            self.inspector.setReadOnly(True)
            self.inspector.setPlaceholderText("Select a stage to inspect its run details.")
            # TALLER TO START, AND FREE TO GROW. 118 px showed about four
            # lines, so every stage worth inspecting needed scrolling
            # immediately. The splitter below gives it a real share of the
            # height rather than the sliver a minimum alone would earn it.
            self.inspector.setMinimumHeight(self.INSPECTOR_MIN_HEIGHT)
            splitter = QSplitter(Qt.Orientation.Vertical, self)
            splitter.addWidget(self.view)
            splitter.addWidget(self.inspector)
            # THE INSPECTOR GETS A REAL SHARE. At 4:1 it was a sliver that
            # collapsed to its minimum the moment the graph had anything in
            # it; the graph still leads, but the pane underneath is now a
            # place text can actually be read, and the splitter handle stays
            # so either can be given the whole height.
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            splitter.setCollapsible(1, False)
            # THE HANDLE HAS TO BE VISIBLE TO BE FOUND. A QSplitter draws
            # nothing by default on this style, so the divider between the
            # graph and the inspector was a 6 px band of nothing -- draggable,
            # but only by somebody who already knew it was there. It is drawn
            # as the same hairline the console's resize handle uses, so the
            # two affordances on this screen do not look like two.
            splitter.setHandleWidth(_PanelHeightGrip.HEIGHT)
            splitter.setStyleSheet(
                "QSplitter::handle:vertical {"
                f" background: transparent;"
                f" border-bottom: 1px solid {_SPLIT_LINE}; }}"
                "QSplitter::handle:vertical:hover {"
                f" border-bottom: 2px solid {_SPLIT_HOVER}; }}")
            self._splitter = splitter
            outer.addWidget(splitter, 1)

            # AND ONE BELOW THE TEXT BOX. The splitter divides the panel's own
            # height between graph and inspector; this one changes how much
            # height the panel has at all, so the pair reads as "the graph
            # against the text" and "the two of them against the page".
            self._bottom_grip = _PanelHeightGrip(self)
            outer.addWidget(self._bottom_grip)

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
            """Start refreshing at the configured interval; report if started."""

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
            """Drain events and repaint after a change or whenever ``force=True``.

            :param force: repaint even when the collector revision is unchanged.
            :returns: whether the graph was repainted.
            """

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

        def set_user_height(self, height: int) -> None:
            """Hold the panel at ``height`` pixels until it is given back.

            Clamped at the floor the graph and inspector need between them,
            so a drag that runs off the top cannot squeeze the panel to
            nothing and leave no edge to drag back.
            """
            floor = self.INSPECTOR_MIN_HEIGHT + _PanelHeightGrip.HEIGHT
            self._user_height = max(int(height), floor)
            self.setFixedHeight(self._user_height)

        def reset_user_height(self) -> None:
            """Give the height back to the content and the layout."""
            self._user_height = None
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.updateGeometry()

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
