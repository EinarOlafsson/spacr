"""Drawing gates — the chart, the shapes on it, and the hierarchy beside it.

The chrome over :mod:`spacr.qt.widgets.gate_spec`. Everything about what a gate
*means* — the geometry, the chain, the percentages, the filter clause — is in
there; this module is three drawing tools and a tree.

:class:`GateCanvas` subclasses
:class:`spacr.qt.widgets.graph_builder.GraphCanvas` for the same reason
:class:`spacr.qt.widgets.trellis_view.TrellisCanvas` does: the marks, the hue
order, the density raster and the large-data policy are one implementation.
What it adds is a *mode*. With no tool armed the drag is the inherited brush;
with a tool armed the drag draws a shape and nothing is published until the
gate is named.

The three tools, and why the drag means different things
---------------------------------------------------------

* **Threshold** — a horizontal sweep on a one-column plot. Only the x extent is
  read, because the y axis of a histogram is a count and gating on a count is
  not a thing anyone means.
* **Rectangle** — a drag on a two-column plot, both extents read.
* **Polygon** — click a vertex at a time, then close it. Three vertices
  minimum, enforced where the gate is built rather than where it is applied.

The population is the parent's, always
---------------------------------------

A gate is drawn **inside** whatever is selected in the tree, and the canvas
shows that parent's population rather than the whole table — because a gate
drawn on a picture of everything and then applied to a subset is a gate nobody
placed. The header says which population is on screen and how many objects that
is.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QPushButton,
    QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from ...selection import DataFilter
from ..theme import SPACING, active_palette, mark_surface
from .graph_builder import GraphCanvas
from .graph_spec import BAR, HISTOGRAM, GraphSpec
from .gate_spec import (
    ELLIPSE, EllipseGate,
    GATE_KINDS, POLYGON, RECTANGLE, THRESHOLD, Gate, GateError, GateSet,
    PolygonGate, RectGate, ThresholdGate,
)

LOG = logging.getLogger("spacr.qt.gate_editor")

#: Key the gate tree's stylesheet is registered under.
QSS_NAME = "GateHierarchy"


def _gate_tree_qss(palette, opacity=None) -> str:
    """Colours for the gate list.

    It had no block at all, so its rows fell back to Qt's default text
    colour -- black -- on the theme's surface. On the dark themes that is
    black on grey and simply cannot be read.

    Transparent backgrounds, theme foreground: the panel behind it already
    carries the page opacity, and painting a colour here would freeze one
    opacity into the list while everything around it kept following the
    preference.
    """
    return f"""
    QTreeWidget#GateHierarchy {{
        background: transparent;
        color: {palette['fg']};
        border: none;
    }}
    QTreeWidget#GateHierarchy::item {{
        color: {palette['fg']};
        padding: 2px 4px;
    }}
    QTreeWidget#GateHierarchy::item:selected {{
        background: {palette['accent']};
        color: {palette['bg']};
    }}
    QHeaderView::section {{
        background: transparent;
        color: {palette['fg_muted']};
        border: none;
        padding: 2px 4px;
    }}
    QWidget#GateTree {{
        background: transparent;
    }}
    """


try:
    from ..theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(QSS_NAME, _gate_tree_qss, replace=True)
except Exception:      # pragma: no cover - decoration is not load-bearing
    LOG.debug("could not register the gate tree stylesheet", exc_info=True)

__all__ = ["GateCanvas", "GateTree", "GateEditorPanel", "TOOL_LABELS"]

#: The tool a fresh editor starts on. RECTANGLE, not brush: "drag to draw a
#: box" is what a user tries first, and starting on the brush meant a drag
#: highlighted instead of drawing and the editor looked as though it could
#: not make a gate at all.
DEFAULT_TOOL = RECTANGLE

#: What each tool is called, and what the gesture is.
TOOL_LABELS = {
    "": "Brush (no gate) — drag to highlight, as everywhere else",
    THRESHOLD: "Threshold — drag across a histogram to cut one column",
    RECTANGLE: "Rectangle — drag a box on a two-column plot",
    ELLIPSE: "Oval — drag a box; the oval is drawn inside it",
    POLYGON: "Polygon — click each vertex, then Close",
}


class GateCanvas(GraphCanvas):
    #: Gating is the one place a filter must NOT move the axes. A gate is
    #: drawn in data coordinates on a particular view; rescaling to the rows
    #: it kept moves that view out from under it, which reads as the plot
    #: zooming into the gate and makes the gate impossible to drag.
    RESCALE_ON_FILTER = False

    """The plot, with gates drawn on it and a tool that draws more.

    Emits :attr:`gate_drawn` with a finished :class:`~spacr.qt.widgets.gate_spec.Gate`
    that has **no name yet** — naming is the host's job, because a gate is not
    a gate until it is named, and a dialog does not belong in a canvas.
    """

    #: A shape was completed. Carries a gate named ``"(unnamed)"``.
    gate_drawn = Signal(object)
    #: A gate was moved or resized in place. Carries the EDITED gate; the
    #: panel replaces the one of the same name.
    gate_edited = Signal(object)

    #: A polygon gained or lost a vertex — for a host showing the count.
    polygon_changed = Signal(int)

    def __init__(self, parent=None, *, link=None, source: str = "gate_editor"):
        super().__init__(parent, link=link, source=source)
        self._tool = DEFAULT_TOOL
        #: How near the first vertex a click has to land to close a polygon.
        #: Pixels, because "close enough to click" is a screen property.
        self.CLOSE_RADIUS_PX = 12.0
        #: Set while a gate is being dragged. `None` whenever it is not,
        #: which is what every handler gates on.
        self._move_name: Optional[str] = None
        self._move_from: Optional[Tuple[float, float]] = None
        self._pending: List[Tuple[float, float]] = []
        self._gates = GateSet()
        self._active: Optional[str] = None
        self._artists: List[object] = []

    # -- the tool ---------------------------------------------------------
    @property
    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        """Arm a drawing tool, or ``""`` to go back to brushing."""
        if tool and tool not in GATE_KINDS:
            raise GateError(
                f"unknown gate tool {tool!r}; the tools are "
                f"{', '.join(GATE_KINDS)}")
        self._tool = tool
        self.clear_pending()

    def pending_vertices(self) -> Tuple[Tuple[float, float], ...]:
        """The polygon vertices clicked so far."""
        return tuple(self._pending)

    def clear_pending(self) -> None:
        self._pending = []
        self.polygon_changed.emit(0)
        self.render_now()

    # -- the gates --------------------------------------------------------
    @property
    def gates(self) -> GateSet:
        return self._gates

    def set_gates(self, gates: GateSet, *, active: Optional[str] = None) -> None:
        """Show ``gates``, with ``active`` as the population being drawn on."""
        self._gates = gates
        self._active = active
        self.render_now()

    @property
    def active_gate(self) -> Optional[str]:
        """The gate whose population the canvas is showing."""
        return self._active

    def population(self) -> Optional[pd.DataFrame]:
        """The rows on screen: the active gate's population, or the table."""
        if self._frame is None:
            return None
        base, _note = self._apply_filter(self._frame)
        if not self._active:
            return base
        try:
            return self._gates.population(base, self._active)
        except GateError as exc:
            LOG.info("the active gate does not apply here: %s", exc)
            return base

    # -- rendering --------------------------------------------------------
    def render_now(self) -> None:
        """Draw the parent's population, then the gates on top of it."""
        frame, self._frame = self._frame, self.population()
        try:
            super().render_now()
        finally:
            self._frame = frame
        self._draw_gates()

    def _draw_gates(self) -> None:
        """Outline every gate that is drawn on these two columns."""
        self._artists = []
        axes = self.panel_axes()
        if not axes:
            return
        palette = active_palette()
        for ax in axes.values():
            for gate in self._gates.gates:
                if not self._gate_is_on_these_axes(gate):
                    continue
                self._outline(ax, gate, palette)
            if self._pending:
                self._outline_pending(ax, palette)
        self._canvas.draw_idle()

    def _gate_is_on_these_axes(self, gate: Gate) -> bool:
        """Whether ``gate`` belongs to the measurements currently plotted.

        A gate is a statement about two named columns. Drawing one on a
        different pair is meaningless -- the outline would sit at
        coordinates that mean something else entirely -- and NOT drawing it
        when the user comes back to its own pair is how a gate seems to have
        vanished.

        A one-column gate (a threshold) needs only its column on screen, on
        either axis: a histogram puts it on x, and a scatter may put it on
        either.
        """
        spec = self._spec
        showing = {c for c in (getattr(spec, "x", None),
                               getattr(spec, "y", None)) if c}
        needed = set(gate.columns)
        if not needed:
            return False
        return needed <= showing

    def _outline(self, ax, gate: Gate, palette) -> None:
        """Draw ``gate`` if it is a gate on the columns currently plotted.

        A gate on other columns is not drawn rather than approximated onto
        these axes: an outline in the wrong units is worse than no outline.
        """
        accent = palette["accent"]
        if isinstance(gate, ThresholdGate):
            if gate.column not in (self._spec.x, self._spec.y):
                return
            for bound in (gate.low, gate.high):
                if bound is not None:
                    self._artists.append(
                        ax.axvline(bound, color=accent, linewidth=1.4,
                                   linestyle="--", zorder=7))
            return
        if not (self._spec.x == getattr(gate, "x_column", None)
                and self._spec.y == getattr(gate, "y_column", None)):
            return
        from matplotlib.patches import Polygon as MplPolygon
        if isinstance(gate, RectGate):
            x0, x1, y0, y1 = self._rect_bounds(ax, gate)
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        elif isinstance(gate, PolygonGate):
            points = list(gate.vertices)
        else:  # pragma: no cover - three kinds, all handled
            return
        patch = MplPolygon(points, closed=True, fill=False, edgecolor=accent,
                           linewidth=1.4, zorder=7)
        ax.add_patch(patch)
        self._artists.append(patch)
        ax.annotate(gate.name, points[0], color=accent, fontsize=7,
                    xytext=(2, 2), textcoords="offset points", zorder=8)

    def _rect_bounds(self, ax, gate: RectGate) -> Tuple[float, float, float, float]:
        """A rectangle's corners, with an unbounded side taken to the axis."""
        x_low, x_high = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        return (gate.x_low if gate.x_low is not None else x_low,
                gate.x_high if gate.x_high is not None else x_high,
                gate.y_low if gate.y_low is not None else y_low,
                gate.y_high if gate.y_high is not None else y_high)

    def _outline_pending(self, ax, palette) -> None:
        xs = [v[0] for v in self._pending]
        ys = [v[1] for v in self._pending]
        self._artists.append(ax.plot(
            xs, ys, color=palette["warning"], linewidth=1.2,
            marker="o", markersize=3, zorder=8)[0])

    # -- drawing gates ----------------------------------------------------
    def gate_at(self, x: float, y: float) -> Optional[str]:
        """Name of the topmost gate containing ``(x, y)``, or None.

        Topmost = last drawn, which is the one the user sees on top and
        therefore the one they mean by clicking there.

        Deliberately does NOT consult `population()`. Hit-testing a gate is
        pure geometry -- is this coordinate inside this shape -- and needs no
        rows at all. Asking for the population first meant that whenever it
        was unavailable, dragging died silently: `population()` returns None
        when the active gate no longer exists, which is exactly the state
        left behind by DELETING a gate. The remaining gate then looked
        "fixed and I cannot move".

        Only gates on the current axes are tested, so a gate belonging to a
        different pair cannot be grabbed invisibly.
        """
        probe = pd.DataFrame({})
        hit: Optional[str] = None
        for gate in self.gates.gates:
            if not self._gate_is_on_these_axes(gate):
                continue
            columns = gate.columns
            if not columns:
                continue
            try:
                probe = pd.DataFrame({columns[0]: [float(x)]})
                if len(columns) > 1:
                    probe[columns[1]] = [float(y)]
                if bool(gate.mask(probe)[0]):
                    hit = gate.name
            except Exception:
                # A gate on columns this scatter is not showing cannot be
                # hit-tested here, and must not stop the ones that can.
                continue
        return hit

    def _on_press(self, event) -> None:
        # A press inside an existing gate MOVES it, whatever tool is armed.
        # Checked first because "the closed gate should be draggable" has to
        # hold without the user first disarming the tool they drew it with --
        # nobody thinks of that, and the gate then looks stuck.
        #
        # The one exception is mid-polygon: there the user is placing
        # vertices, and a vertex that happens to land inside an older gate
        # must not drag it.
        mid_polygon = self._tool == POLYGON and bool(self._pending)
        if not mid_polygon and event.inaxes is not None \
                and event.xdata is not None and event.ydata is not None:
            name = self.gate_at(float(event.xdata), float(event.ydata))
            if name:
                self._move_name = name
                self._move_from = (float(event.xdata), float(event.ydata))
                return
        if self._tool != POLYGON:
            super()._on_press(event)
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        # Clicking the FIRST vertex again closes the shape. That is what
        # everyone tries, and the "Close polygon" button was the only way to
        # do it -- so a polygon looked impossible to finish.
        if len(self._pending) >= 3 and self._near_first_vertex(event, x, y):
            self.close_polygon_now()
            return
        self._pending.append((x, y))
        self.polygon_changed.emit(len(self._pending))
        self._draw_gates()

    def _near_first_vertex(self, event, x: float, y: float) -> bool:
        """Whether ``(x, y)`` is close enough to the first vertex to close.

        Measured in PIXELS, not data units: "close enough to click" is a
        property of the screen, and a data-unit tolerance would be
        unusable on one axis and impossible on the other whenever the two
        measurements have different ranges -- which is nearly always.
        """
        if not self._pending:
            return False
        ax = getattr(event, "inaxes", None)
        first = self._pending[0]
        try:
            fx, fy = ax.transData.transform(first)
            px, py = ax.transData.transform((x, y))
        except Exception:
            return False
        return ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5 <= self.CLOSE_RADIUS_PX

    def close_polygon_now(self) -> None:
        """Close the pending polygon.

        `close_polygon` ALREADY emits `gate_drawn`. Emitting again here made
        one drawn polygon prompt for a name twice and create two identical
        gates -- which is exactly what was reported. This wrapper exists only
        so the click-the-first-vertex path and the Close button share a name.
        """
        self.close_polygon()

    def _on_motion(self, event) -> None:
        if getattr(self, "_move_name", None):
            # The move is applied on RELEASE, not per motion event: a gate
            # is re-evaluated against the whole table to redraw, and doing
            # that on every mouse move makes a large frame unusable.
            return
        if self._tool == POLYGON:
            return
        super()._on_motion(event)

    def _on_release(self, event) -> None:
        name = getattr(self, "_move_name", None)
        if name:
            start = getattr(self, "_move_from", None)
            self._move_name = None
            self._move_from = None
            if (start is None or event.inaxes is None
                    or event.xdata is None or event.ydata is None):
                return
            dx = float(event.xdata) - start[0]
            dy = float(event.ydata) - start[1]
            if dx == 0 and dy == 0:
                # A click, not a drag. Selecting rather than moving by zero
                # keeps a stray click from marking the gate set dirty.
                self.set_gates(self.gates, active=name)
                return
            try:
                gate = self.gates.get(name)
            except Exception:
                # The gate went away between press and release -- another
                # view can remove one while a drag is in flight.
                return
            self.gate_edited.emit(gate.translated(dx, dy))
            return
        if self._tool in ("", POLYGON):
            super()._on_release(event)
            return
        origin, self._drag_origin = self._drag_origin, None
        if self._drag_patch is not None:
            try:
                self._drag_patch.remove()
            except (ValueError, NotImplementedError):  # pragma: no cover
                pass
            self._drag_patch = None
        if origin is None or event.inaxes is not origin[0]:
            return
        if event.xdata is None or event.ydata is None:
            return
        _ax, x0, y0 = origin
        gate = self.gate_from_drag(x0, y0, float(event.xdata),
                                   float(event.ydata))
        if gate is not None:
            self.gate_drawn.emit(gate)

    def _make_drag_patch(self, x0: float, y0: float):
        """Preview the shape the armed tool will actually make.

        A rectangular preview for an elliptical gate tells the user the wrong
        thing about what they are about to draw -- reported as "the oval
        looks like a square when dragged but does in fact generate an oval
        gate".
        """
        if self._tool == ELLIPSE:
            from matplotlib.patches import Ellipse

            return Ellipse((x0, y0), 0.0, 0.0, **self._drag_patch_style())
        return super()._make_drag_patch(x0, y0)

    def _update_drag_patch(self, patch, x0: float, y0: float,
                           x1: float, y1: float) -> None:
        if self._tool == ELLIPSE:
            # Inscribed in the swept box, exactly as EllipseGate.from_drag
            # builds it -- so the preview and the gate are the same shape.
            patch.set_center(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
            patch.set_width(abs(x1 - x0))
            patch.set_height(abs(y1 - y0))
            return
        super()._update_drag_patch(patch, x0, y0, x1, y1)

    def gate_from_drag(self, x0: float, y0: float, x1: float, y1: float,
                       *, name: str = "(unnamed)") -> Optional[Gate]:
        """Build the armed tool's gate from a swept rectangle.

        Public so the interaction can be driven without synthesising mouse
        events — the same seam the Graph Builder's :meth:`brush` provides.
        """
        spec = self._spec
        if self._tool == THRESHOLD:
            column = spec.x or spec.y
            if not column:
                return None
            # Only the horizontal sweep is read: on a histogram the vertical
            # axis is a count, and gating on a count is not a thing anyone
            # means.
            return ThresholdGate(name=name, parent=self._active,
                                 column=column, low=x0, high=x1)
        if self._tool == RECTANGLE:
            if not (spec.x and spec.y):
                return None
            return RectGate(name=name, parent=self._active,
                            x_column=spec.x, y_column=spec.y,
                            x_low=x0, x_high=x1, y_low=y0, y_high=y1)
        if self._tool == ELLIPSE:
            if not (spec.x and spec.y):
                return None
            if x0 == x1 or y0 == y1:
                # A zero-width drag would be an ellipse with a zero radius,
                # which EllipseGate refuses. Nothing drawn is the right
                # answer to nothing dragged.
                return None
            return EllipseGate.from_drag(name, spec.x, spec.y,
                                         x0, y0, x1, y1, parent=self._active)
        return None

    def close_polygon(self, *, name: str = "(unnamed)") -> Optional[Gate]:
        """Finish the polygon being clicked out.

        :returns: the gate, or ``None`` when fewer than three vertices have
            been clicked — the canvas does not raise at the user for clicking
            twice and changing their mind.
        """
        spec = self._spec
        if len(self._pending) < 3 or not (spec.x and spec.y):
            return None
        gate = PolygonGate(name=name, parent=self._active,
                           x_column=spec.x, y_column=spec.y,
                           vertices=tuple(self._pending))
        self._pending = []
        self.polygon_changed.emit(0)
        self.gate_drawn.emit(gate)
        return gate


class GateTree(QWidget):
    """The gating hierarchy, with each gate's n and its percentage of parent.

    Both percentages are shown, from
    :class:`~spacr.qt.widgets.gate_spec.GateStats` — 90% of a parent that is 2%
    of the table is 1.8% of the objects, and a strategy that prints only the
    first is flattering itself.
    """

    #: The selected gate changed — carries the name, or ``""`` for the root.
    active_changed = Signal(str)
    #: A gate was deleted.
    gates_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GateTree")
        self._gates = GateSet()
        self._frame: Optional[pd.DataFrame] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("GateHierarchy")
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Gate", "n", "% parent", "% all"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setToolTip(
            "The gating hierarchy. Select a gate to draw the next one inside "
            "it; the plot then shows that gate's population.")
        self.tree.currentItemChanged.connect(self._on_selection)
        # `GateEditorPanel` is transparent scaffolding by design (see
        # the GraphBuilder block), so the hierarchy has nothing behind
        # it and is the page itself.
        mark_surface(self.tree)
        outer.addWidget(self.tree, 1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._remove = QPushButton("Delete gate", self)
        self._remove.setToolTip(
            "Deletes the gate and everything gated inside it — a child whose "
            "parent is gone is a gate on a population that no longer exists.")
        self._remove.clicked.connect(self.remove_selected)
        row.addWidget(self._remove)
        row.addStretch(1)
        outer.addLayout(row)

    def set_gates(self, gates: GateSet, frame: Optional[pd.DataFrame]) -> None:
        self._gates = gates
        self._frame = frame
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree and recompute every count."""
        current = self.active_gate()
        self.tree.clear()
        if self._frame is None:
            return
        try:
            stats = {s.name: s for s in self._gates.stats(self._frame)}
        except GateError as exc:
            LOG.info("gates do not apply to this table: %s", exc)
            stats = {}
        items: Dict[str, QTreeWidgetItem] = {}
        for gate in self._gates.order():
            stat = stats.get(gate.name)
            labels = [gate.name, "", "", ""]
            if stat is not None:
                labels = [gate.name, f"{stat.n_in:,}",
                          f"{100.0 * stat.of_parent:.1f}%",
                          f"{100.0 * stat.of_total:.1f}%"]
            item = QTreeWidgetItem(labels)
            item.setData(0, Qt.UserRole, gate.name)
            item.setToolTip(0, gate.describe())
            parent_item = items.get(gate.parent) if gate.parent else None
            if parent_item is None:
                self.tree.addTopLevelItem(item)
            else:
                parent_item.addChild(item)
            items[gate.name] = item
        self.tree.expandAll()
        if current in items:
            self.tree.setCurrentItem(items[current])

    def active_gate(self) -> str:
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else ""

    def select(self, name: str) -> None:
        for index in range(self.tree.topLevelItemCount()):
            if self._select_in(self.tree.topLevelItem(index), name):
                return
        self.tree.setCurrentItem(None)

    def _select_in(self, item: QTreeWidgetItem, name: str) -> bool:
        if item.data(0, Qt.UserRole) == name:
            self.tree.setCurrentItem(item)
            return True
        return any(self._select_in(item.child(i), name)
                   for i in range(item.childCount()))

    def remove_selected(self) -> None:
        name = self.active_gate()
        if not name:
            return
        self._gates.remove(name)
        self.refresh()
        self.gates_changed.emit()
        self.active_changed.emit(self.active_gate())

    def _on_selection(self, *_args) -> None:
        self.active_changed.emit(self.active_gate())


class _ClusterSettingsDialog(QDialog):
    """DBSCAN's two parameters, with what they mean in the units they act in.

    `eps` is in SCALED units while scaling is on, which is what makes one
    default work across measurements whose ranges differ by orders of
    magnitude -- `cell_area` runs to thousands and `eccentricity` to one, and
    unscaled DBSCAN on that pair clusters on area alone. The checkbox says so
    rather than leaving the user to discover it by getting one blob.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            from ..dialogs import detach_from_window_manager
            detach_from_window_manager(self)
        except Exception:
            pass
        self.setWindowTitle("Cluster settings")
        form = QFormLayout(self)

        self._eps = QDoubleSpinBox(self)
        self._eps.setRange(0.01, 100.0)
        self._eps.setSingleStep(0.05)
        self._eps.setDecimals(2)
        self._eps.setValue(0.30)
        self._eps.setToolTip(
            "Neighbourhood radius. Larger merges nearby populations into "
            "one; smaller splits one into several.")
        form.addRow("eps (radius)", self._eps)

        self._min_samples = QSpinBox(self)
        self._min_samples.setRange(2, 10000)
        self._min_samples.setValue(10)
        self._min_samples.setToolTip(
            "Objects needed to seed a population. Anything sparser is "
            "treated as debris and left out of every gate.")
        form.addRow("min samples", self._min_samples)

        self._scale = QCheckBox("Standardise both axes first", self)
        self._scale.setChecked(True)
        self._scale.setToolTip(
            "On unless you know otherwise. Without it, the axis with the "
            "larger numeric range decides the clustering on its own.")
        form.addRow("", self._scale)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def eps(self) -> float:
        return float(self._eps.value())

    def min_samples(self) -> int:
        return int(self._min_samples.value())

    def scale(self) -> bool:
        return bool(self._scale.isChecked())


class GateEditorPanel(QWidget):
    """Canvas, tools and hierarchy: the whole gating surface.

    :meth:`publish` is the point of the screen — it turns the selected gate
    into a :class:`~spacr.selection.DataFilter` clause and pushes it onto the
    shared filter, so every open view narrows to the gated population.
    """

    gates_changed = Signal()
    #: Selecting a gate asks the screen to show the measurements it was drawn
    #: on. Carries ``(x_column, y_column)``; y is empty for a one-column gate.
    axes_requested = Signal(str, str)

    def __init__(self, parent=None, *, link=None,
                 source: str = "gate_editor"):
        super().__init__(parent)
        self.setObjectName("GateEditorPanel")
        self._gates = GateSet()
        self._frame: Optional[pd.DataFrame] = None
        self._namer = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(SPACING["xs"])
        self._tool = QComboBox(self)
        self._tool.setObjectName("GateToolPicker")
        for key in ("",) + GATE_KINDS:
            self._tool.addItem(TOOL_LABELS[key].split(" — ")[0], key)
        self._tool.setToolTip("\n".join(TOOL_LABELS.values()))
        index = self._tool.findData(DEFAULT_TOOL)
        if index >= 0:
            self._tool.setCurrentIndex(index)
        # No `canvas.set_tool` here: the canvas is built further down and
        # already starts on DEFAULT_TOOL. Calling it at this point read the
        # attribute before it existed.
        self._tool.currentIndexChanged.connect(self._on_tool_changed)
        tools.addWidget(QLabel("Tool", self))
        tools.addWidget(self._tool)

        self._close = QPushButton("Close polygon", self)
        self._close.setEnabled(False)
        self._close.clicked.connect(self._on_close_polygon)
        tools.addWidget(self._close)

        self._cluster = QPushButton("Cluster…", self)
        self._cluster.setToolTip(
            "Find dense populations with DBSCAN and turn each one into a "
            "gate you can edit, nest and save like any other.")
        self._cluster.clicked.connect(self._on_cluster)
        tools.addWidget(self._cluster)

        self._apply = QPushButton("Apply as filter", self)
        self._apply.setObjectName("PrimaryButton")
        self._apply.setToolTip(
            "Publish the selected gate as the shared filter, so every open "
            "view narrows to its population.")
        self._apply.clicked.connect(self.publish)
        tools.addWidget(self._apply)

        self._status = QLabel("no gates", self)
        self._status.setObjectName("GateStatus")
        self._status.setWordWrap(True)
        tools.addWidget(self._status, 1)
        outer.addLayout(tools)

        # A SPLITTER, not a QHBoxLayout. The gate list sits between the
        # scatter and the filter column, and in a box layout with a hard
        # 320px cap it could not be resized at all: dragging the outer
        # splitter moved the filter column and took the canvas AND the gate
        # list with it as one block. Its own handle makes it independent,
        # which is what "the gate box should be independent" asks for.
        self.body = QSplitter(Qt.Horizontal, self)
        self.body.setChildrenCollapsible(False)

        self.canvas = GateCanvas(self, link=link, source=source)
        self.canvas.gate_drawn.connect(self._on_gate_drawn)
        self.canvas.gate_edited.connect(self._on_gate_edited)
        self.canvas.polygon_changed.connect(self._on_polygon_changed)
        self.body.addWidget(self.canvas)

        self.tree = GateTree(self)
        # No maximum. A cap cannot be dragged past, so a gate whose name or
        # statistics were wider than 320px had nowhere to be read. A minimum
        # stays, so the handle cannot hide the list entirely.
        self.tree.setMinimumWidth(220)
        self.tree.active_changed.connect(self._on_active_changed)
        self.tree.gates_changed.connect(self._on_tree_changed)
        self.body.addWidget(self.tree)

        # The scatter takes the slack when the panel is resized; the gate
        # list keeps whatever width the user gave it.
        self.body.setStretchFactor(0, 1)
        self.body.setStretchFactor(1, 0)
        outer.addWidget(self.body, 1)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        self._frame = frame
        self.canvas.set_frame(frame)
        self.tree.set_gates(self._gates, frame)
        self._refresh_status()

    def set_spec(self, spec: GraphSpec) -> None:
        self.canvas.set_spec(spec)

    @property
    def gates(self) -> GateSet:
        return self._gates

    def set_gates(self, gates: GateSet) -> None:
        """Replace the whole set — loading a saved gating strategy."""
        self._gates = gates
        self.canvas.set_gates(gates, active=self.tree.active_gate() or None)
        self.tree.set_gates(gates, self._frame)
        self._refresh_status()
        self.gates_changed.emit()

    def set_namer(self, namer) -> None:
        """Install ``namer() -> str`` to name a freshly drawn gate.

        Injectable so a test can name gates without a modal dialog standing in
        a headless run's way — the same reason
        :class:`~spacr.qt.widgets.data_filter_panel.DataFilterPanel` takes an
        injectable link.
        """
        self._namer = namer

    # -- drawing ----------------------------------------------------------
    def _on_tool_changed(self, *_args) -> None:
        tool = self._tool.currentData() or ""
        self.canvas.set_tool(tool)
        self._close.setEnabled(False)
        self._refresh_status()

    def _on_polygon_changed(self, count: int) -> None:
        self._close.setEnabled(count >= 3)
        if count:
            self._status.setText(
                f"{count} vertex(es) — three or more make a region")

    def _on_cluster(self) -> None:
        """Find dense populations and add one gate per cluster.

        Clusters become REAL gates rather than a separate kind of selection,
        so each is editable, nestable, serialisable and usable as a filter
        the moment it appears -- everything a hand-drawn gate can do, because
        it is one.
        """
        from PySide6.QtWidgets import QMessageBox

        frame = self.canvas.population()
        if frame is None or frame.empty:
            QMessageBox.information(
                self, "Nothing to cluster",
                "Load a table before clustering.")
            return
        x_column = getattr(self.canvas, "x_column", None) or ""
        y_column = getattr(self.canvas, "y_column", None) or ""
        if not x_column or not y_column:
            QMessageBox.information(
                self, "Pick two measurements",
                "Clustering needs an X and a Y measurement.")
            return

        dialog = _ClusterSettingsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        from .gate_spec import ClusterError, cluster_gates
        try:
            found = cluster_gates(
                frame, x_column, y_column,
                eps=dialog.eps(), min_samples=dialog.min_samples(),
                scale=dialog.scale(), parent=self.canvas.active_gate())
        except ClusterError as exc:
            # Named, not swallowed: every one of these messages says what to
            # change, and a silent empty result reads as a broken button.
            QMessageBox.warning(self, "Could not cluster", str(exc))
            return

        if not found:
            QMessageBox.information(
                self, "No clusters",
                "DBSCAN found only sparse points at these settings. Raise "
                "eps to group them more loosely, or lower min_samples.")
            return

        gates = self.canvas.gates
        for gate in found:
            gates.add(gate)
        self.canvas.set_gates(gates, active=found[0].name)
        self._refresh_status()

    def _on_close_polygon(self) -> None:
        self.canvas.close_polygon()

    def _on_gate_edited(self, gate: Gate) -> None:
        """Replace a gate that was dragged on the canvas.

        By NAME, so the hierarchy is untouched: a moved child stays a child.
        `GateSet.add` replaces an existing name rather than appending, which
        is what makes this a one-liner instead of a remove-then-add that
        could lose the gate if the add failed.
        """
        # `gates` is a PROPERTY on both this panel and the canvas. Calling
        # it raised TypeError on every drag -- which is what the user saw.
        gates = self.gates
        gates.add(gate)
        self.canvas.set_gates(gates, active=gate.name)
        self._refresh_status()

    def _on_gate_drawn(self, gate: Gate) -> None:
        name = self._ask_name()
        if not name:
            self.canvas.render_now()
            return
        try:
            self._gates.add(gate.rename(name))
        except GateError as exc:
            self._status.setText(str(exc))
            return
        self.canvas.set_gates(self._gates,
                              active=self.tree.active_gate() or None)
        self.tree.set_gates(self._gates, self._frame)
        self.tree.select(name)
        self._refresh_status()
        self.gates_changed.emit()

    def _ask_name(self) -> str:
        if self._namer is not None:
            return str(self._namer() or "")
        name, ok = QInputDialog.getText(
            self, "Name this gate",
            "A gate is not a gate until it is named — the name is what makes "
            "it re-appliable and what the hierarchy is read by:")
        return name.strip() if ok else ""

    def _on_active_changed(self, name: str) -> None:
        self.canvas.set_gates(self._gates, active=name or None)
        self._refresh_status()
        # Choosing a gate should show you that gate. It is drawn on two named
        # measurements, so selecting one whose axes are not on screen used to
        # select something invisible -- and any attempt to drag it did
        # nothing, because it was not being drawn or hit-tested there.
        if not name:
            return
        try:
            columns = self._gates.get(name).columns
        except Exception:
            return
        if columns:
            self.axes_requested.emit(columns[0],
                                     columns[1] if len(columns) > 1 else "")

    def _on_tree_changed(self) -> None:
        self.canvas.set_gates(self._gates,
                              active=self.tree.active_gate() or None)
        self._refresh_status()
        self.gates_changed.emit()

    # -- publishing -------------------------------------------------------
    def publish(self) -> Optional[DataFilter]:
        """Push the selected gate onto the shared filter.

        Composed onto whatever the Local Data Filter already published rather
        than replacing it: a gate and a filter are both ways of narrowing the
        population, and a screen with both must not have one silently undo the
        other.

        NOTE, and the next thing to change here: the user has asked that
        applying a gate HIGHLIGHT its points and leave the rest of the graph
        on screen, rather than hide the rows outside it --

            "i dont want it to zoom in the first place. i want it to
             highlight the datapoints in the gate and show the gate but also
             show the rest of the graph."

        That is a SELECTION, not a filter, and the distinction already
        exists: `link.set_selection` rings rows and keeps every one of them
        on screen, while `link.set_filter` removes them. The Graph Builder's
        own test states both behaviours side by side. Switching this to a
        selection is what makes the axes stop moving for the right reason,
        rather than because rescaling was suppressed.

        Keep the filter available -- narrowing to a gate is a real thing to
        want -- but it should be the explicit second action, not what the
        primary button does.
        """
        name = self.tree.active_gate()
        if not name:
            self._status.setText("Select a gate in the hierarchy first.")
            return None
        frame = self._frame
        if frame is None:
            self._status.setText("Load a table first.")
            return None
        try:
            inside = self._gates.mask(frame, name)
        except GateError as exc:
            self._status.setText(str(exc))
            return None

        # A SELECTION, not a filter. Applying a gate highlights the objects
        # inside it and leaves every other point on screen:
        #
        #   "i want it to highlight the datapoints in the gate and show the
        #    gate but also show the rest of the graph"
        #
        # Filtering removed the outside rows, and the axes then rescaled to
        # what was left -- which is what read as the plot zooming into the
        # gate, and what moved the ground out from under the gate outline so
        # it could not be dragged.
        #
        # Narrowing the population to a gate is still a real thing to want,
        # but it is a second, explicit act. It is not what pressing the
        # primary button should do.
        try:
            self.canvas.publish_selection(frame.loc[inside])
        except Exception as exc:
            # A highlight needs object keys to name the rows to everyone
            # else. A table without them cannot be published, and saying so
            # is better than a traceback -- the gate itself is still drawn
            # and still usable locally.
            self._status.setText(
                f"{int(inside.sum()):,} object(s) in {name}, but they cannot "
                f"be shared with other views: {exc}")
            return None
        self._status.setText(
            f"{int(inside.sum()):,} object(s) highlighted by {name}")
        return None

    def status(self) -> str:
        return self._status.text()

    def _refresh_status(self) -> None:
        if self._frame is None:
            self._status.setText("no table loaded")
            return
        active = self.tree.active_gate()
        population = self.canvas.population()
        n = 0 if population is None else len(population)
        where = f"inside {active}" if active else "the whole table"
        self._status.setText(
            f"drawing on {where} · {n:,} objects"
            + (f" · {len(self._gates)} gate(s)" if len(self._gates) else ""))

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.canvas.close()
        super().closeEvent(event)
