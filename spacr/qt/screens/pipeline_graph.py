"""Pipeline Graph — what produced what, and which of it is still true.

A spaCR project accumulates files faster than anyone can hold in their
head: masks, a measurements database, crops, model weights, predictions,
regression tables, figures. Some of those were made from each other. Some
were made from a *previous* version of each other, because the mask step
was re-run with a different diameter on Tuesday and nothing downstream was.
Nothing in the GUI showed that, and a stale number does not look stale — it
looks like a number.

This screen draws the DAG:

::

    ┌──────────────┐   ┌────────────────┐   ┌───────────────┐
    │ mask         │──▶│ measure        │──▶│ classify      │
    │ masks   OK   │   │ db      STALE  │   │ preds   STALE │
    └──────────────┘   └────────────────┘   └───────────────┘

Boxes are artifacts, arrows are "was made from", and the colour is the
verdict :func:`spacr.artifacts.Registry.is_stale` returns — with its
``reasons`` shown in the panel beside the graph, because "stale" on its own
is an accusation, not an explanation. Click a box and the panel names the
run that produced it, the settings digest, the spaCR version, and what
re-running it would invalidate.

Three deliberate choices:

**The graph is drawn from provenance, not from the module list.** What the
pipeline is *supposed* to do is :data:`spacr.ports.PORTS`, and that is
drawn too — dimmed, underneath, as the "declared order" strip — but the
boxes and arrows are what actually happened. A project where measure read
last week's masks does not look like the tidy diagram, and that difference
is the entire value of the screen.

**Missing is not stale.** A deleted file is red and says so; an outdated
one is amber. :mod:`spacr.artifacts` refuses to conflate them and neither
does this.

**Nothing here writes.** No re-run button, no "mark as fresh", no delete.
The screen answers a question; acting on the answer is the module's job,
and a one-click re-run wired to a graph is how somebody overwrites the
artifact they were trying to check.

The layout is computed by a pure function (:func:`layout_rects`) off the
graph's layers, so the arrangement is testable without pixels, and the
heavy part — opening the registry and asking it about every artifact — runs
through :class:`spacr.qt.job_runner.JobRunner`, off the GUI thread.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...pipeline_graph import (STATE_CURRENT, STATE_MISSING, STATE_STALE,
                               PipelineGraph, build_graph, stale_summary,
                               to_dot)
from ..job_runner import JobRunner
from ..theme import SPACING, active_palette, pane_surface, register_widget_qss

__all__ = ["APP_KEY", "GraphCanvas", "PipelineGraphScreen", "layout_rects",
           "make_pipeline_graph_screen", "register"]

#: The app key this screen is registered under. Load-bearing: saved user
#: state, the command palette and the sidebar all key off it.
APP_KEY = "pipeline_graph"

#: Sidebar / tile name.
APP_NAME = "Pipeline Graph"

#: One-line summary; the tooltip and status tip.
APP_DESCRIPTION = (
    "The DAG of what produced what, with everything stale or missing marked")

#: The paragraph under this app's header, handed to the seam as ``intro``.
APP_INTRO = (
    "Every registered output of a project, drawn as the graph of what was "
    "made from what. Each box carries the run that produced it, the settings "
    "digest and the spaCR version; the colour is the artifact registry's "
    "verdict on whether it still follows from its inputs. Amber is stale — an "
    "input moved on or a material setting changed after this was written — "
    "and red is missing from disk. Click a box for the reasons and for what "
    "re-running it would invalidate.")

#: Why there is no ``spacr-run pipeline_graph``; reaches
#: ``spacr.cli.INTERACTIVE_ONLY``, which prints it instead of "unknown
#: module".
APP_CLI_NOTE = (
    "Pipeline Graph is an interactive view of one project's provenance DAG; "
    "headless, call spacr.pipeline_graph.build_graph(project) and "
    "format_graph(graph) for the same content as text, or to_dot(graph) for "
    "a Graphviz figure.")

#: "Pipeline Graph" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr.
APP_TRANSLATIONS = (
    "Pipelinediagram",
    "Pipeline-Graph",
    "Grafo de la tubería",
    "流程图",
    "Grafo do pipeline",
    "पाइपलाइन ग्राफ़",
    "파이프라인 그래프",
    "Vinnsluferilsrit",
    "Graphe du pipeline",
)

#: Box geometry, in device-independent pixels. Named because the layout
#: function and the painter must agree and a literal in two places is a
#: drawing that drifts.
NODE_WIDTH = 190
NODE_HEIGHT = 74
COLUMN_GAP = 64
ROW_GAP = 22
MARGIN = 18

#: One colour per state, as ``(fill, border)``. Deliberately not from the
#: palette: these three have to stay distinguishable in both themes and to
#: mean the same thing in the exported DOT file, which has no palette.
STATE_COLOURS: Dict[str, Tuple[str, str]] = {
    STATE_CURRENT: ("#2f6b3f", "#63b175"),
    STATE_STALE: ("#7a5a1e", "#d6a740"),
    STATE_MISSING: ("#7a2a2a", "#d46a6a"),
}


def _graph_qss(palette: dict, opacity) -> str:
    """QSS for the canvas frame and the verdict strip.

    The canvas paints itself, so all this does is give it the same recessed
    surface every other pane on the screen sits on — a graph floating on the
    window background reads as a dialog that failed to draw.
    """
    surface = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QScrollArea#PipelineGraphCanvasArea {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QFrame#PipelineGraphBanner {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#PipelineGraphVerdict {{
    font-weight: 600;
}}
QLabel#PipelineGraphVerdict[problem="true"] {{
    color: {palette["warning"]};
}}
"""


# ``replace=True`` because this module owns the name: a reimport (a test that
# reloads it, a plugin that pulls it in twice) must re-register the same block
# rather than raise on the duplicate and leave the screen unstyled.
register_widget_qss("PipelineGraphCanvasArea", _graph_qss, replace=True)


# ---------------------------------------------------------------------------
# Layout — a pure function, so the arrangement is testable without pixels
# ---------------------------------------------------------------------------

def layout_rects(graph: PipelineGraph) -> Dict[str, QRect]:
    """Place every node of ``graph`` on a grid: one column per layer.

    Column is :attr:`spacr.pipeline_graph.Node.depth`, which is the longest
    distance from a root — so an arrow never points backwards. Row is the
    node's position within its layer, in the order the graph already sorted
    them (newest first), so re-building the same graph twice draws the same
    picture.

    :param graph: the graph to lay out.
    :returns: ``{artifact_id: QRect}``, empty for an empty graph.
    """
    rects: Dict[str, QRect] = {}
    for column, row_ids in enumerate(graph.layers):
        for row, artifact_id in enumerate(row_ids):
            rects[artifact_id] = QRect(
                MARGIN + column * (NODE_WIDTH + COLUMN_GAP),
                MARGIN + row * (NODE_HEIGHT + ROW_GAP),
                NODE_WIDTH, NODE_HEIGHT)
    return rects


def canvas_size(graph: PipelineGraph) -> Tuple[int, int]:
    """The pixel size the whole graph needs, as ``(width, height)``."""
    rects = layout_rects(graph)
    if not rects:
        return (400, 200)
    right = max(r.right() for r in rects.values()) + MARGIN
    bottom = max(r.bottom() for r in rects.values()) + MARGIN
    return (int(right), int(bottom))


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

class GraphCanvas(QWidget):
    """Draws one :class:`~spacr.pipeline_graph.PipelineGraph`.

    Custom painting rather than ``QGraphicsScene``: the graph is a few dozen
    boxes with straight-ish edges, and a scene brings its own selection
    model, its own coordinate system and its own set of ways to leak a
    C++ object into a Python-owned widget. The whole drawing is one
    ``paintEvent`` over a dict of rectangles that a test can read directly.

    :param parent: Qt parent.
    :ivar selected: artifact id of the box the user last clicked, or ``""``.
    """

    #: Emitted with an artifact id when a box is clicked; ``""`` on a click
    #: that landed on the background.
    node_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: Optional[PipelineGraph] = None
        self._rects: Dict[str, QRect] = {}
        self._visible: set = set()
        self.selected: str = ""
        self.setMinimumSize(400, 200)
        self.setMouseTracking(False)

    def set_graph(self, graph: Optional[PipelineGraph],
                  visible: Optional[set] = None) -> None:
        """Show ``graph``, drawing only the ids in ``visible`` when given."""
        self._graph = graph
        self._rects = layout_rects(graph) if graph is not None else {}
        if graph is None:
            self._visible = set()
        elif visible is None:
            self._visible = set(self._rects)
        else:
            self._visible = set(visible) & set(self._rects)
        if self.selected not in self._visible:
            self.selected = ""
        width, height = (canvas_size(graph) if graph is not None
                         else (400, 200))
        self.setMinimumSize(max(400, width), max(200, height))
        self.resize(max(400, width), max(200, height))
        self.update()

    def graph(self) -> Optional[PipelineGraph]:
        """The graph currently drawn, or ``None``."""
        return self._graph

    def node_rects(self) -> Dict[str, QRect]:
        """``{artifact id: rectangle}`` for every box currently drawn."""
        return {key: QRect(rect) for key, rect in self._rects.items()
                if key in self._visible}

    def node_at(self, x: int, y: int) -> str:
        """Artifact id of the box containing this point, or ``""``."""
        for artifact_id, rect in self._rects.items():
            if artifact_id in self._visible and rect.contains(int(x), int(y)):
                return artifact_id
        return ""

    def select(self, artifact_id: str) -> None:
        """Select a box by id (``""`` clears) and emit :attr:`node_clicked`."""
        self.selected = artifact_id if artifact_id in self._visible else ""
        self.update()
        self.node_clicked.emit(self.selected)

    # -- Qt ---------------------------------------------------------------

    def mousePressEvent(self, event) -> None:      # noqa: N802 - Qt override
        """Select whatever box was clicked."""
        position = event.position()
        self.select(self.node_at(int(position.x()), int(position.y())))
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:           # noqa: N802 - Qt override
        """Paint the edges, then the boxes, then the selection ring."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            if self._graph is None or not self._visible:
                self._paint_empty(painter)
                return
            self._paint_edges(painter)
            self._paint_nodes(painter)
        finally:
            painter.end()

    def _paint_empty(self, painter: QPainter) -> None:
        """The "nothing to draw" state, which is a sentence, not a blank."""
        palette = active_palette()
        painter.setPen(QPen(QColor(palette["fg_muted"])))
        painter.drawText(
            self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN),
            int(Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap),
            "Nothing registered here yet. Runs started from this version of "
            "spaCR record their outputs and appear in this graph.")

    def _paint_edges(self, painter: QPainter) -> None:
        """One curve per provenance edge, dashed when the source is gone."""
        assert self._graph is not None
        palette = active_palette()
        for edge in self._graph.edges:
            source = self._rects.get(edge.source)
            target = self._rects.get(edge.target)
            if target is None or edge.target not in self._visible:
                continue
            if source is None or edge.source not in self._visible:
                continue
            start = (source.right(), source.center().y())
            end = (target.left(), target.center().y())
            path = QPainterPath()
            path.moveTo(*start)
            midpoint = (start[0] + end[0]) / 2.0
            path.cubicTo(midpoint, start[1], midpoint, end[1], *end)
            pen = QPen(QColor(STATE_COLOURS[STATE_MISSING][1]
                              if edge.dangling else palette["border"]))
            pen.setWidth(2)
            if edge.dangling:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    def _paint_nodes(self, painter: QPainter) -> None:
        """One rounded box per artifact: module, kind, file, state."""
        assert self._graph is not None
        palette = active_palette()
        metrics = QFontMetrics(self.font())
        for artifact_id, rect in self._rects.items():
            if artifact_id not in self._visible:
                continue
            node = self._graph.node(artifact_id)
            if node is None:                        # pragma: no cover
                continue
            fill, border = STATE_COLOURS.get(
                node.state, STATE_COLOURS[STATE_CURRENT])
            pen = QPen(QColor(border))
            pen.setWidth(3 if artifact_id == self.selected else 1)
            painter.setPen(pen)
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(rect, 8, 8)

            painter.setPen(QPen(QColor("#ffffff")))
            inner = rect.adjusted(10, 8, -10, -8)
            painter.drawText(
                inner, int(Qt.AlignTop | Qt.AlignLeft),
                metrics.elidedText(node.module, Qt.ElideRight, inner.width()))
            painter.setPen(QPen(QColor(palette["fg_muted"])))
            painter.drawText(
                inner.adjusted(0, 18, 0, 0), int(Qt.AlignTop | Qt.AlignLeft),
                metrics.elidedText(node.kind, Qt.ElideRight, inner.width()))
            painter.setPen(QPen(QColor("#ffffff")))
            name = os.path.basename(node.path.rstrip(os.sep)) or node.path
            painter.drawText(
                inner.adjusted(0, 34, 0, 0), int(Qt.AlignTop | Qt.AlignLeft),
                metrics.elidedText(name, Qt.ElideMiddle, inner.width()))
            painter.drawText(
                inner, int(Qt.AlignBottom | Qt.AlignRight),
                node.state.upper() if node.state != STATE_CURRENT else "")


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

class PipelineGraphScreen(QWidget):
    """Pick a project; see its provenance DAG with staleness marked.

    :param parent: Qt parent.
    :param project: open straight onto this project root, skipping the
        folder picker.
    :param threaded: ``False`` builds the graph inline, so a test drives the
        screen synchronously without the behaviour diverging.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation worked. Failures land here and in the banner — never in a
        modal dialog, which hangs a headless run.
    """

    #: Emitted with the :class:`~spacr.pipeline_graph.PipelineGraph` after
    #: every load, including the empty ones.
    graph_loaded = Signal(object)

    def __init__(self, parent=None, project: str = "", threaded: bool = True):
        super().__init__(parent)
        self._graph: Optional[PipelineGraph] = None
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self.last_error: str = ""

        self._build_ui()
        if project:
            self.load_project(project)
        else:
            self._set_verdict(
                "Choose a spaCR project folder to draw what it has produced.",
                problem=False)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "pipeline_graph")

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Picker, verdict strip, then the canvas beside the detail pane."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel(APP_NAME)
        title.setObjectName("ScreenTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "What produced what, and which of it still follows from its "
            "inputs.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        picker = QHBoxLayout()
        picker.setSpacing(SPACING["sm"])
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("Project folder")
        self._project_edit.returnPressed.connect(self._on_project_entered)
        picker.addWidget(QLabel("Project"))
        picker.addWidget(self._project_edit, 1)
        self._browse_button = QPushButton("Browse…")
        self._browse_button.clicked.connect(self._on_browse)
        picker.addWidget(self._browse_button)
        self._reload_button = QPushButton("Redraw")
        self._reload_button.setToolTip(
            "Re-read the artifact registry and re-check every artifact's "
            "provenance. Nothing is written.")
        self._reload_button.clicked.connect(self._on_reload)
        picker.addWidget(self._reload_button)
        outer.addLayout(picker)

        self._banner = QFrame()
        self._banner.setObjectName("PipelineGraphBanner")
        banner_row = QHBoxLayout(self._banner)
        banner_row.setContentsMargins(SPACING["md"], SPACING["sm"],
                                      SPACING["md"], SPACING["sm"])
        self._verdict = QLabel("")
        self._verdict.setObjectName("PipelineGraphVerdict")
        self._verdict.setWordWrap(True)
        banner_row.addWidget(self._verdict, 1)
        self._copy_button = QPushButton("Copy Graphviz")
        self._copy_button.setToolTip(
            "Put the graph on the clipboard as Graphviz DOT, for a methods "
            "figure.")
        self._copy_button.clicked.connect(self._on_copy_dot)
        banner_row.addWidget(self._copy_button)
        outer.addWidget(self._banner)

        filters = QHBoxLayout()
        filters.setSpacing(SPACING["md"])
        self._filters: Dict[str, QCheckBox] = {}
        for state, label, tip in (
                (STATE_CURRENT, "Current",
                 "Artifacts that still follow from everything they were "
                 "made from."),
                (STATE_STALE, "Stale",
                 "An input was produced again, or a material setting "
                 "changed, after this was written."),
                (STATE_MISSING, "Missing",
                 "The path this was registered at is no longer on disk.")):
            box = QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(tip)
            box.toggled.connect(self._on_filter_toggled)
            filters.addWidget(box)
            self._filters[state] = box
        filters.addStretch(1)
        self._declared = QLabel("")
        self._declared.setObjectName("Muted")
        self._declared.setWordWrap(True)
        filters.addWidget(self._declared, 2)
        outer.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("PipelineGraphCanvasArea")
        self._scroll.setWidgetResizable(False)
        self._canvas = GraphCanvas()
        self._canvas.node_clicked.connect(self._on_node_clicked)
        self._scroll.setWidget(self._canvas)
        splitter.addWidget(self._scroll)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumWidth(300)
        self._details.setPlainText(
            "Click a box for the run that produced it, its settings digest, "
            "and what re-running it would invalidate.")
        splitter.addWidget(self._details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

    # -- loading ----------------------------------------------------------

    def load_project(self, project: str) -> None:
        """Build and draw the graph for ``project``, off the GUI thread.

        Returns as soon as the job is submitted; :attr:`graph_loaded` fires
        when the graph is drawn. A project with no registry is not a failure
        — it draws the declared module order and says nothing has run.
        """
        project = str(project or "").strip()
        self.last_error = ""
        self._project_edit.setText(project)
        if not project:
            self._set_verdict("Choose a spaCR project folder.", problem=False)
            return
        self._set_verdict(f"Reading {os.path.basename(project) or project}…",
                          problem=False)
        self._jobs.cancel()
        self._jobs.submit(lambda root=project: build_graph(root),
                          self._on_graph_ready)

    def graph(self) -> Optional[PipelineGraph]:
        """The graph currently drawn, or ``None``."""
        return self._graph

    def _on_graph_ready(self, graph: Optional[PipelineGraph]) -> None:
        """Draw a freshly built graph. Runs on the GUI thread."""
        self._graph = graph
        if graph is None:                            # pragma: no cover
            self._set_verdict("The graph could not be built.", problem=True)
            return
        summary = stale_summary(graph)
        message = summary["verdict"]
        if graph.notes:
            message = f"{message} {graph.notes[0]}"
        self._set_verdict(
            message, problem=bool(summary["n_stale"] or summary["n_missing"]))
        declared = " → ".join(
            "/".join(row) for row in graph.modules.layers[:6])
        self._declared.setText(
            f"Declared order: {declared}" if declared else "")
        self._redraw()
        self.graph_loaded.emit(graph)

    def _redraw(self) -> None:
        """Push the graph into the canvas honouring the state filters."""
        if self._graph is None:
            self._canvas.set_graph(None)
            return
        wanted = {state for state, box in self._filters.items()
                  if box.isChecked()}
        visible = {node.artifact_id for node in self._graph.nodes
                   if node.state in wanted}
        self._canvas.set_graph(self._graph, visible)

    # -- details ----------------------------------------------------------

    def describe(self, artifact_id: str) -> str:
        """The detail block for one artifact, as plain text.

        Split out from the widget so a test can assert on the content
        without reading a ``QTextEdit`` back.
        """
        if self._graph is None or not artifact_id:
            return ""
        node = self._graph.node(artifact_id)
        if node is None:
            return ""
        lines = [f"{node.module} → {node.kind}",
                 f"state: {node.state}",
                 f"path: {node.path}"]
        if node.role:
            lines.append(f"role: {node.role}")
        if node.run_id:
            lines.append(f"run id: {node.run_id}")
        if node.created_utc:
            lines.append(f"produced: {node.created_utc}")
        if node.spacr_version:
            lines.append(f"spaCR: {node.spacr_version}")
        if node.settings_hash:
            lines.append(f"settings digest: {node.settings_hash[:16]}")
        lines.append(f"size: {node.size_bytes} bytes in {node.n_files} file(s)")
        if node.reasons:
            lines.append("")
            lines.append("Why it is flagged:")
            lines.extend(f"  • {reason}" for reason in node.reasons)
        upstream = self._graph.upstream(artifact_id)
        if upstream:
            lines.append("")
            lines.append("Made from:")
            lines.extend(f"  • {n.module} {n.kind} ({n.state})"
                         for n in upstream)
        downstream = self._graph.downstream(artifact_id)
        lines.append("")
        if downstream:
            lines.append("Re-running this would invalidate:")
            lines.extend(f"  • {n.module} {n.kind} at {n.path}"
                         for n in downstream)
        else:
            lines.append("Nothing was derived from this.")
        return "\n".join(lines)

    # -- slots ------------------------------------------------------------

    def _on_node_clicked(self, artifact_id: str) -> None:
        """Fill the detail pane for the clicked box."""
        text = self.describe(artifact_id)
        self._details.setPlainText(
            text or "Click a box for its provenance.")

    def _on_filter_toggled(self, _checked: bool) -> None:
        """Re-draw with the current state filters."""
        self._redraw()

    def _on_project_entered(self) -> None:
        """Load whatever was typed into the project box."""
        self.load_project(self._project_edit.text())

    def _on_reload(self) -> None:
        """Re-read the registry for the project already chosen."""
        self.load_project(self._project_edit.text())

    def _on_browse(self) -> None:                    # pragma: no cover - modal
        """Ask for a project folder and load it."""
        chosen = QFileDialog.getExistingDirectory(self, "Choose a project")
        if chosen:
            self.load_project(chosen)

    def _on_copy_dot(self) -> None:
        """Put the graph on the clipboard as Graphviz DOT."""
        if self._graph is None:
            self._set_verdict("There is no graph to copy yet.", problem=True)
            return
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(to_dot(self._graph))
            self._set_verdict(
                f"{len(self._graph)} node(s) copied as Graphviz DOT.",
                problem=False)

    def _on_job_failed(self, message: str) -> None:
        """Report a background failure inline; never a modal."""
        self.last_error = message
        self._set_verdict(f"Could not read that project: {message}",
                          problem=True)

    def _set_verdict(self, text: str, *, problem: bool) -> None:
        """Write the banner and repolish it for the problem colour."""
        self._verdict.setText(text)
        self._verdict.setProperty("problem", "true" if problem else "false")
        style = self._verdict.style()
        if style is not None:
            style.unpolish(self._verdict)
            style.polish(self._verdict)

    # -- lifecycle --------------------------------------------------------

    def is_busy(self) -> bool:
        """True while a graph is still being built."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def closeEvent(self, event) -> None:             # noqa: N802 - Qt override
        """Drain the worker before the widget goes."""
        self._jobs.shutdown()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_pipeline_graph_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory the registry calls to build this screen."""
    return PipelineGraphScreen()


def register() -> bool:
    """Add Pipeline Graph to the app registry. Idempotent.

    :returns: True when this call added the row, False when it was already
        there — which is what a second import, or a plugin that pulls the
        module in again, must not treat as an error.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
        factory=make_pipeline_graph_screen, stage=STAGE_ALPHA,
        title="Pipeline Graph", intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/pipeline_graph",
        translations=APP_TRANSLATIONS)
    return True


register()
