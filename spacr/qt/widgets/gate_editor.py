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
    QComboBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ...selection import DataFilter
from ..theme import SPACING, active_palette
from .graph_builder import GraphCanvas
from .graph_spec import BAR, HISTOGRAM, GraphSpec
from .gate_spec import (
    GATE_KINDS, POLYGON, RECTANGLE, THRESHOLD, Gate, GateError, GateSet,
    PolygonGate, RectGate, ThresholdGate,
)

LOG = logging.getLogger("spacr.qt.gate_editor")

__all__ = ["GateCanvas", "GateTree", "GateEditorPanel", "TOOL_LABELS"]

#: What each tool is called, and what the gesture is.
TOOL_LABELS = {
    "": "Brush (no gate) — drag to highlight, as everywhere else",
    THRESHOLD: "Threshold — drag across a histogram to cut one column",
    RECTANGLE: "Rectangle — drag a box on a two-column plot",
    POLYGON: "Polygon — click each vertex, then Close",
}


class GateCanvas(GraphCanvas):
    """The plot, with gates drawn on it and a tool that draws more.

    Emits :attr:`gate_drawn` with a finished :class:`~spacr.qt.widgets.gate_spec.Gate`
    that has **no name yet** — naming is the host's job, because a gate is not
    a gate until it is named, and a dialog does not belong in a canvas.
    """

    #: A shape was completed. Carries a gate named ``"(unnamed)"``.
    gate_drawn = Signal(object)
    #: A polygon gained or lost a vertex — for a host showing the count.
    polygon_changed = Signal(int)

    def __init__(self, parent=None, *, link=None, source: str = "gate_editor"):
        super().__init__(parent, link=link, source=source)
        self._tool = ""
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
                self._outline(ax, gate, palette)
            if self._pending:
                self._outline_pending(ax, palette)
        self._canvas.draw_idle()

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
    def _on_press(self, event) -> None:
        if self._tool != POLYGON:
            super()._on_press(event)
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        self._pending.append((float(event.xdata), float(event.ydata)))
        self.polygon_changed.emit(len(self._pending))
        self._draw_gates()

    def _on_motion(self, event) -> None:
        if self._tool == POLYGON:
            return
        super()._on_motion(event)

    def _on_release(self, event) -> None:
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


class GateEditorPanel(QWidget):
    """Canvas, tools and hierarchy: the whole gating surface.

    :meth:`publish` is the point of the screen — it turns the selected gate
    into a :class:`~spacr.selection.DataFilter` clause and pushes it onto the
    shared filter, so every open view narrows to the gated population.
    """

    gates_changed = Signal()

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
        self._tool.currentIndexChanged.connect(self._on_tool_changed)
        tools.addWidget(QLabel("Tool", self))
        tools.addWidget(self._tool)

        self._close = QPushButton("Close polygon", self)
        self._close.setEnabled(False)
        self._close.clicked.connect(self._on_close_polygon)
        tools.addWidget(self._close)

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

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(SPACING["sm"])
        self.canvas = GateCanvas(self, link=link, source=source)
        self.canvas.gate_drawn.connect(self._on_gate_drawn)
        self.canvas.polygon_changed.connect(self._on_polygon_changed)
        body.addWidget(self.canvas, 1)

        self.tree = GateTree(self)
        self.tree.setMaximumWidth(320)
        self.tree.active_changed.connect(self._on_active_changed)
        self.tree.gates_changed.connect(self._on_tree_changed)
        body.addWidget(self.tree)
        outer.addLayout(body, 1)

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

    def _on_close_polygon(self) -> None:
        self.canvas.close_polygon()

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
        """
        name = self.tree.active_gate()
        if not name:
            self._status.setText("Select a gate in the hierarchy first.")
            return None
        try:
            data_filter = self._gates.filter_for(name, self.canvas.link.filter)
        except GateError as exc:
            self._status.setText(str(exc))
            return None
        self.canvas.publish_filter(data_filter)
        self._status.setText(f"filtering on {data_filter.describe()}")
        return data_filter

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
