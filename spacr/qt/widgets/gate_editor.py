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

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QButtonGroup, QLabel,
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QPushButton,
    QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from ...selection import DataFilter
from ..theme import SPACING, active_palette, mark_surface
from .graph_builder import GraphCanvas
from .graph_spec import BAR, HISTOGRAM, GraphSpec
from .gate_spec import (
    BOX, BoxGate, ELLIPSE, EllipseGate, Handle, WAND,
    GATE_KINDS, POLYGON, RECTANGLE, THRESHOLD, Gate, GateError, GateSet,
    PolygonGate, RectGate, ThresholdGate,
    CYLINDER, PRISM,
    COMPOSITE,
    CylinderGate, PrismGate,
)
from .toggle import Toggle

LOG = logging.getLogger("spacr.qt.gate_editor")


def _project(ax, point):
    """A data point's position in the 3D axes' own 2D coordinates.

    Wrapped because matplotlib has moved this: `proj3d.proj_transform` takes
    the projection matrix as `ax.M` in some versions and `ax.get_proj()` in
    others, and the import path has changed too. One place to be wrong.
    """
    from mpl_toolkits.mplot3d import proj3d

    matrix = getattr(ax, "M", None)
    if matrix is None:
        matrix = ax.get_proj()
    x, y, _z = proj3d.proj_transform(point[0], point[1], point[2], matrix)
    return (x, y)


def fit_to_text(widget, *, padding: int = 16, lines: int = 1) -> None:
    """Size ``widget`` so its own text cannot be clipped.

    Measured with the widget's REAL font metrics, so it follows the theme, the
    platform and the user's DPI rather than a number that was right on one
    machine. Height is set as well as width: the reports are "cutt of on the
    sides usually its from the top asn sometimes botom", and a control sized
    only horizontally clips its ascenders exactly the way described.

    A minimum, never a fixed size -- a layout may still give the widget more,
    and a widget that cannot grow is the other half of this same bug.
    """
    metrics = widget.fontMetrics()
    text = widget.text() if hasattr(widget, "text") else ""
    width = metrics.horizontalAdvance(str(text) or "MM") + padding
    height = metrics.height() * max(1, lines) + max(8, padding // 2)
    widget.setMinimumSize(max(width, widget.minimumWidth()),
                          max(height, widget.minimumHeight()))

#: Colours gates are drawn in, cycled by position in the gate set. Chosen to
#: stay apart from each other AND from a viridis-coloured cloud underneath --
#: a gate outline in the same green as the density it sits on is invisible
#: exactly where it matters. Also distinguishable in the common forms of
#: colour blindness, since the colour is the only thing telling two gates
#: apart on the plot.
GATE_COLOURS: Tuple[str, ...] = (
    "#ff4d6d",   # rose
    "#4cc9f0",   # cyan
    "#ffb703",   # amber
    "#b388ff",   # violet
    "#06d6a0",   # mint
    "#ff8fab",   # pink
    "#8ecae6",   # pale blue
    "#f4a261",   # sand
)

#: Key the gate tree's stylesheet is registered under.
QSS_NAME = "GateHierarchy"


def _gate_tree_qss(palette, opacity=None) -> str:
    """Colours for the gate list.

    It had no block at all, so its rows fell back to Qt's default text
    colour -- black -- on the theme's surface. On the dark themes that is
    black on grey and simply cannot be read.

    Theme foreground, and no background of its own: painting a colour here
    would freeze one opacity into the list while everything around it kept
    following the preference. The tree is marked as a surface (see
    `GateTree`), so the theme's ``*[spacrSurface="true"]`` rule supplies the
    fill at the user's page opacity -- declaring ``background: transparent``
    here would beat that rule and leave the list with no surface at all.
    """
    return f"""
    QTreeWidget#GateHierarchy {{
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
    POLYGON: "Polygon — click each vertex, then Close. In 3D the vertices "
             "land on the anchor plane and the shape becomes a prism",
    WAND: "Wand — click a population; the gate grows to fit it",
    BOX: "Box — three measurements at once, made from the 3D view",
    CYLINDER: "Cylinder — an oval drawn on one plane of the 3D view, "
              "extended along the third measurement",
    PRISM: "Prism — a polygon drawn on one plane of the 3D view, "
           "extended along the third measurement",
    COMPOSITE: "Combined — other gates added to or subtracted from each "
               "other, chosen in the gates panel",
}


class GateCanvas(GraphCanvas):
    """A :class:`GraphCanvas` you can draw gates on.

    Adds interactive gate drawing, dragging and hit-testing to the shared
    canvas, and pins the axes while a gate exists -- see
    :data:`RESCALE_ON_FILTER` for why that is not optional here.
    """

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
    #: A wand click could not become a gate. Carries the reason, which always
    #: names the setting or the gesture that would fix it.
    wand_failed = Signal(str)

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
        #: Gates the user has toggled OFF. Names, not gates, so a gate that
        #: is edited in place keeps its toggle.
        self._disabled: set = set()
        try:
            self._canvas.mpl_connect("scroll_event", self._on_scroll)
            # A spin ends here. `snap_to_axis` is read on release rather
            # than during the drag: snapping mid-turn would fight the user's
            # hand, and the point of snapping is only about the FINAL view.
            self._canvas.mpl_connect("button_release_event",
                                     self._on_button_release)
        except Exception:      # pragma: no cover - no canvas in a bare test
            LOG.debug("no scroll events available", exc_info=True)
        #: Which plane the pending polygon's vertices were clicked on, as
        #: (first, second). A polygon spanning two planes is not one shape.
        self._pending_plane: Optional[Tuple[str, str]] = None
        #: Set while an anchor point is being pulled: (gate name, role).
        self._resize: Optional[Tuple[str, str]] = None
        #: The dashed shape following the mouse mid-drag. Its artists are
        #: tracked separately from `_artists` so a motion event can replace
        #: it without redrawing every gate and every highlight -- which is a
        #: mask over the whole table per gate, per mouse move.
        self._ghost: List[object] = []
        #: How near an anchor a press has to land to grab it. Pixels: "close
        #: enough to grab" is a property of the screen, not of the data.
        self.HANDLE_RADIUS_PX = 9.0
        #: Set by `apply_settings`. Defaults match GateEditorSettings, so the
        #: canvas draws the same with or without a settings object.
        self._settings = None
        self._highlight_gated = True
        self._line_width = 0.5
        self._colour_map = "viridis"
        self._resolution = "points"
        self._bins = 200
        self._show_grid = False
        self._x_scale = "linear"
        self._y_scale = "linear"
        self._colour_by = "density"
        #: Limits set by the wheel, or None to follow the data.
        self._zoom = None
        #: "2D", "3D" or "xD" -- see `set_mode`.
        self._mode = "2D"
        self._z_column = ""
        #: How far the volume is zoomed in. 1.0 is the data's own extent.
        self._volume_zoom = 1.0
        #: (elevation, azimuth) once the user has turned it, else None.
        self._view_angles = None
        #: Which axis the volume spins about: "x", "y", "z" or "" for free.
        self._spin_axis = "z"
        self._spin_from = None
        #: Where a draw-in-the-volume drag started, or None.
        self._volume_drag = None

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
        """The rows on screen -- the whole table.

        This used to return the ACTIVE GATE'S population, and `render_now`
        plots whatever this returns, so selecting or drawing a gate replotted
        only the objects inside it. That is textbook hierarchical gating, and
        it is not what was asked for: "draw a gate on the graph ... but never
        zoom into the gated data ... be able to select this gate in the gate
        panel and toggle it on and off."

        It is also the whole of the stuck state -- "the only way to get back
        to the main figure is to delete a gate" -- because clearing the active
        name was the only thing that ever restored the full view, and deleting
        was the only thing that cleared it.

        Gates are overlays now: outlined on the full plot, highlighting their
        own objects. `_active` survives only as the PARENT of the next gate
        drawn, which is a statement about the hierarchy, not about the view.
        """
        if self._frame is None:
            return None
        base, _note = self._apply_filter(self._frame)
        return base

    def apply_settings(self, settings) -> None:
        """Take the drawing settings and redraw once.

        Every one of these has to reach the DRAWING. The settings window
        shipped with fields the canvas never read, so the colour map said
        viridis while the points stayed blue and nothing but the sampling
        appeared to do anything.

        Guarded with getattr so a partial settings object -- an older saved
        set, a test double -- cannot stop the editor from drawing at all.
        """
        self._settings = settings
        tool = getattr(settings, "default_tool", None)
        if tool and tool in GATE_KINDS and not self._tool:
            self._tool = tool
        self._highlight_gated = bool(getattr(settings, "highlight_gated", True))
        self._line_width = float(getattr(settings, "gate_line_width", 0.5))
        self.POINT_SIZE_BASE = float(getattr(settings, "point_size", 6.0)) ** 2
        self.POINT_ALPHA = float(getattr(settings, "point_opacity", 0.6))
        self._colour_map = str(getattr(settings, "colour_map", "viridis"))
        self._resolution = str(getattr(settings, "resolution_mode", "points"))
        self._bins = int(getattr(settings, "bins", 200))
        self._show_grid = bool(getattr(settings, "show_grid", False))
        scale_for = getattr(settings, "scale_for", None)
        if callable(scale_for):
            self._x_scale, self._y_scale = scale_for("x"), scale_for("y")
        else:
            self._x_scale = "log" if getattr(settings, "log_x", False) else "linear"
            self._y_scale = "log" if getattr(settings, "log_y", False) else "linear"
        self._colour_by = str(getattr(settings, "colour_by", "density"))
        self.render_now()

    # -- the settings, reaching the drawing -------------------------------
    def point_colormap(self):
        """The colour map the user chose, by name.

        An unknown name falls back rather than raising: matplotlib's registry
        changes between versions, and a colour map that no longer exists must
        not take the whole plot with it (INVARIANTS 10).
        """
        from matplotlib import colormaps
        try:
            return colormaps[self._colour_map]
        except (KeyError, TypeError):
            LOG.info("no colour map called %r; using the theme's",
                     self._colour_map)
            return super().point_colormap()

    def decorate_axes(self, ax) -> None:
        """Grid and log scales.

        Log is applied only where it is legal: a log axis over data that
        reaches zero or below draws nothing at all, which reads as the plot
        having broken rather than as the setting being inapplicable.
        """
        palette = active_palette()
        # Line properties only when enabling: matplotlib warns that supplying
        # them with False turns the grid ON, which is the opposite of asked.
        if self._show_grid:
            ax.grid(True, color=palette["fg_muted"], alpha=0.25, linewidth=0.5)
        else:
            ax.grid(False)
        ax.set_axisbelow(True)
        # Decided by the DATA, not by the axis limits. The limits are padded
        # outward, so a measurement whose smallest value is 1 gets a lower
        # limit near -4 and looked non-positive -- which is why log X never
        # applied while log Y, on a column with larger numbers and therefore
        # proportionally smaller padding, sometimes did.
        spec = self._spec
        for scale, column, setter in (
                (self._x_scale, getattr(spec, "x", None), ax.set_xscale),
                (self._y_scale, getattr(spec, "y", None), ax.set_yscale)):
            if scale == "linear" or not column:
                continue
            if scale in ("log", "logit") and not self._column_is_positive(column):
                LOG.info("%s scale skipped: %s reaches zero or below",
                         scale, column)
                continue
            try:
                setter(scale)
            except Exception:
                LOG.info("axis scale %r did not apply", scale, exc_info=True)

    def _column_is_positive(self, column: str) -> bool:
        """Whether every finite value of ``column`` is above zero.

        A log axis over data that reaches zero draws nothing at all, which
        reads as the plot having broken rather than as the setting being
        inapplicable to this measurement.
        """
        frame = self._frame
        if frame is None or column not in frame.columns:
            return False
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        return bool(len(values)) and float(values.min()) > 0

    def _draw_plain_points(self, ax, x, y, rows, palette):
        """Colour the cloud by DENSITY when there is no colour column.

        "cmap dosnt seem to be allpied to the data , they are always blue."
        A cytometry scatter has no colour axis, so the base canvas drew one
        flat colour and the chosen map had nothing to colour. Density is what
        the map should show: on a crowded plot the overlap is the reading, and
        a single colour hides it entirely.

        The binned resolution modes replace the points outright; this is the
        `points` mode, which keeps one marker per object.
        """
        if self._resolution != "points":
            finite = np.isfinite(x) & np.isfinite(y)
            return self._draw_binned(ax, x[finite], y[finite])
        values = self._colour_values(x, y, rows)
        if values is None:
            return ax.scatter(x, y, s=self._sizes(rows),
                              color=self._series_colour(0),
                              linewidths=0.0, alpha=self.POINT_ALPHA)
        return ax.scatter(x, y, s=self._sizes(rows), c=values,
                          cmap=self.point_colormap(),
                          linewidths=0.0, alpha=self.POINT_ALPHA)

    def _colour_values(self, x, y, rows):
        """The per-point value the colour map is applied to, or None for flat.

        A named column wins over density, so "colour by pathogen count" means
        that and not an approximation of it. A column that is missing or
        non-numeric falls back to density rather than to an error -- the
        colour axis is decoration (INVARIANTS 10).
        """
        choice = self._colour_by
        if choice == "flat":
            return None
        if choice and choice != "density" and choice in rows.columns:
            values = pd.to_numeric(rows[choice], errors="coerce").to_numpy(float)
            if np.isfinite(values).any():
                return values
            LOG.info("column %r has no numeric values; colouring by density",
                     choice)
        return self._density(x, y)

    def _draw_density(self, ax, rows, palette) -> None:
        """Large tables take a different path, and it has to obey the settings.

        Past its large-data threshold the base canvas rasterises with imshow
        and never calls `_draw_points` at all -- which is why "viridis does
        work but not when there are more than 50000 data points" and why
        hexbin appeared to do nothing: both live in the points path.

        The chosen resolution mode is honoured here too. `points` on a table
        this size still means the raster, because one marker per object is
        what the threshold exists to avoid.
        """
        spec = self._spec
        if self._resolution == "points" or not (spec.x and spec.y):
            super()._draw_density(ax, rows, palette)
            return
        x = pd.to_numeric(rows[spec.x], errors="coerce").to_numpy(float)
        y = pd.to_numeric(rows[spec.y], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        self._draw_binned(ax, x[finite], y[finite])

    def _draw_binned(self, ax, x, y):
        """hexbin / histogram / density, one implementation for both paths."""
        if len(x) == 0:
            return None
        if self._resolution == "hexbin":
            return ax.hexbin(x, y, gridsize=max(10, min(self._bins // 4, 200)),
                             cmap=self.point_colormap(), mincnt=1,
                             linewidths=0.0)
        counts, xe, ye = np.histogram2d(
            x, y, bins=max(10, min(self._bins, 1000)))
        counts = counts.T
        if self._resolution == "density" and counts.sum() > 0:
            counts = counts / counts.sum()
        masked = np.ma.masked_where(counts <= 0, counts)
        return ax.pcolormesh(xe, ye, masked, cmap=self.point_colormap(),
                             shading="auto")

    def _density(self, x, y):
        """A per-point density, binned rather than kernel-estimated.

        A Gaussian KDE over a million objects is minutes; a 2D histogram
        lookup is milliseconds and produces the same reading at the
        resolution a screen can show.
        """
        finite = np.isfinite(x) & np.isfinite(y)
        out = np.zeros(len(x))
        if not finite.any():
            return out
        bins = max(10, min(self._bins, 512))
        counts, xe, ye = np.histogram2d(x[finite], y[finite], bins=bins)
        xi = np.clip(np.digitize(x[finite], xe) - 1, 0, counts.shape[0] - 1)
        yi = np.clip(np.digitize(y[finite], ye) - 1, 0, counts.shape[1] - 1)
        out[finite] = counts[xi, yi]
        return out

    # -- which gates are showing -----------------------------------------
    def is_gate_enabled(self, name: str) -> bool:
        """Whether ``name`` is drawn. Unknown gates are on: a gate that has
        never been toggled has never been turned off."""
        return name not in self._disabled

    def set_gate_enabled(self, name: str, on: bool) -> None:
        """Turn a gate's outline and highlight on or off.

        Off means NOT DRAWN, never deleted and never removed from the set:
        the gate keeps its shape, its parent and its children, and comes back
        exactly as it was. Its rows stay on the plot either way -- toggling
        changes what is marked, not what exists.
        """
        if on:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)
        self.render_now()

    @property
    def enabled_gates(self) -> Tuple[str, ...]:
        """The names currently drawn, in definition order."""
        return tuple(g.name for g in self._gates.gates
                     if self.is_gate_enabled(g.name))

    # -- 3D ---------------------------------------------------------------
    def set_mode(self, mode: str, *, z_column: str = "") -> None:
        """Switch between the 2D scatter and the 3D volume."""
        self._mode = mode if mode in ("2D", "3D", "xD") else "2D"
        self._z_column = z_column or self._z_column
        self.render_now()

    def _render_volume(self) -> bool:
        """Draw the 3D view. Returns False if it cannot, so 2D takes over.

        A real third axis rather than a projection trick: matplotlib's Axes3D
        gives depth sorting and, more to the point, DRAG-ROTATION for free.
        Rotation is the whole reason to be in 3D -- a fixed view of a volume
        tells you less than two scatters.
        """
        spec = self._spec
        z = self._z_column
        frame = self.population()
        if not (spec.x and spec.y and z) or frame is None or frame.empty:
            return False
        if any(c not in frame.columns for c in (spec.x, spec.y, z)):
            return False

        from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 - registers 3d

        palette = active_palette()
        self._figure.clear()
        self._axes = {}
        ax = self._figure.add_subplot(projection="3d")
        self._apply_spin_speed(ax)
        self._draw_anchor_aura(ax)

        x = pd.to_numeric(frame[spec.x], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[spec.y], errors="coerce").to_numpy(float)
        zs = pd.to_numeric(frame[z], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(zs)

        if not self._draw_voxels(ax, x[finite], y[finite], zs[finite]):
            ax.scatter(x[finite], y[finite], zs[finite],
                       s=max(1.0, self.POINT_SIZE_BASE / 4.0),
                       c=self._density(x, y)[finite],
                       cmap=self.point_colormap(),
                       depthshade=False, linewidths=0.0,
                       alpha=self.POINT_ALPHA)

        for axis, label in ((ax.xaxis, spec.x), (ax.yaxis, spec.y),
                            (ax.zaxis, z)):
            axis.set_pane_color((0, 0, 0, 0))
            axis.line.set_color(palette["fg_muted"])
        ax.set_xlabel(spec.x, color=palette["fg"], fontsize=8)
        ax.set_ylabel(spec.y, color=palette["fg"], fontsize=8)
        ax.set_zlabel(z, color=palette["fg"], fontsize=8)
        ax.tick_params(colors=palette["fg_muted"], labelsize=7)
        self._figure.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))

        # Matplotlib's own drag-rotation is free rotation, which is what the
        # axis lock exists to replace. Disabled so the two cannot fight.
        try:
            ax.disable_mouse_rotation()
        except Exception:      # pragma: no cover - older matplotlib
            LOG.debug("could not take over 3d rotation", exc_info=True)
        if self._view_angles is not None:
            ax.view_init(elev=self._view_angles[0], azim=self._view_angles[1])
        if self._volume_zoom != 1.0:
            self._apply_volume_zoom(ax)

        self._draw_volume_gates(ax, frame, palette)
        # Keyed like every other panel, so `panel_axes()` keeps its contract
        # and nothing downstream has to know this one is three-dimensional.
        self._axes = {(0, 0): ax}
        self._canvas.draw_idle()
        return True

    #: Points above which the volume is drawn as voxels instead of dots.
    #:
    #: Not a taste threshold. A scatter of a million points in 3D is slower
    #: to draw than to compute, and every dot is drawn over by the ones in
    #: front of it -- so past this the picture stops improving and only the
    #: frame rate changes. Below it the dots ARE the better picture, because
    #: an individual object can be seen and clicked.
    VOXEL_THRESHOLD = 20000

    def _draw_voxels(self, ax, x, y, z) -> bool:
        """Draw the volume as occupancy voxels. False if it should not be.

        THIS IS WHAT `voxel_bins` IS FOR, and until now nothing read it --
        the setting was declared, given a control, saved, reloaded and
        ignored, which is the phantom-control defect instruction 77 swept
        for.

        A voxel is drawn where objects ARE, sized by how many. Occupancy
        rather than a surface: a surface implies a boundary the data has not
        got, while a cloud of sized markers says "this many here" and is the
        same claim the 2D density plot makes.

        :returns: False when the point count does not justify it, so the
            caller scatters as before.
        """
        bins = int(getattr(self._settings, "voxel_bins", 0) or 0)
        if bins < 2 or len(x) < self.VOXEL_THRESHOLD:
            return False
        try:
            counts, edges = np.histogramdd(
                np.column_stack([x, y, z]), bins=(bins, bins, bins))
        except Exception:
            LOG.debug("could not bin the volume", exc_info=True)
            return False
        filled = counts > 0
        if not filled.any():
            return False
        centres = [(e[:-1] + e[1:]) / 2.0 for e in edges]
        ix, iy, iz = np.nonzero(filled)
        weight = counts[filled]
        # Area, not radius, tracks the count: a marker whose RADIUS was the
        # count would exaggerate a busy voxel by its square.
        sizes = 6.0 + 40.0 * (weight / weight.max())
        ax.scatter(centres[0][ix], centres[1][iy], centres[2][iz],
                   s=sizes, c=weight, cmap=self.point_colormap(),
                   depthshade=False, linewidths=0.0,
                   alpha=min(1.0, self.POINT_ALPHA * 2))
        return True

    def volume_axis_map(self):
        """How screen pixels map to data on the two axes facing the viewer.

        Returns ``(x_column, y_column, invert)`` where ``invert(dx, dy)`` turns
        a movement in PIXELS into one in data units, or None when the view is
        not square-on.

        Built by projecting the data's own corners and measuring where they
        land, rather than by trusting a formula for the projection matrix:
        matplotlib has changed how `ax.M` is spelled more than once, and a
        drag that silently lands in the wrong measurement is worse than one
        that refuses.

        The axis that barely moves on screen is the one pointing at the
        viewer -- that is what "square-on" means, and it is the axis a drag
        cannot say anything about.
        """
        ax = self.axes_at(0, 0)
        spec = self._spec
        if ax is None or not hasattr(ax, "get_zlim"):
            return None
        columns = (spec.x, spec.y, self._z_column)
        if not all(columns):
            return None

        limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
        origin = [lo for lo, _hi in limits]
        spans = [hi - lo for lo, hi in limits]
        if any(not np.isfinite(s) or s == 0 for s in spans):
            return None

        def screen(point):
            try:
                projected = ax.transData.transform(
                    ax.get_proj() is not None and _project(ax, point) or (0, 0))
            except Exception:
                return None
            return np.asarray(projected, dtype=float)

        base = screen(origin)
        if base is None:
            return None
        moves = []
        for axis in range(3):
            point = list(origin)
            point[axis] += spans[axis]
            landed = screen(point)
            if landed is None:
                return None
            moves.append(landed - base)

        # The depth axis is whichever moved least on screen -- the one
        # pointing most nearly at the viewer.
        #
        # NO square-on requirement. It used to refuse anything more than a few
        # degrees off, which made drawing feel broken at every angle a user
        # actually leaves the volume at: "the mouse needs to be decoupled from
        # spinning. if the gate is on None then spin. if the gate is on any of
        # the gating mechanisms then allow drawing." The tool decides, and
        # nothing else does.
        #
        # Off-square the mapping is read in the plane through the middle of
        # the depth axis, so a drag is exact there and increasingly
        # approximate towards the front and back faces. That is a real limit
        # and it is the reason the gate leaves the depth axis UNBOUNDED --
        # it is honest about the one measurement the gesture cannot pin down.
        lengths = [float(np.hypot(*m)) for m in moves]
        depth = int(np.argmin(lengths))
        if max(lengths) <= 0:
            return None

        kept = [a for a in range(3) if a != depth]
        matrix = np.column_stack([moves[a] / spans[a] for a in kept])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            return None
        inverse = np.linalg.inv(matrix)

        def invert(dx, dy):
            data = inverse @ np.asarray([dx, dy], dtype=float)
            return float(data[0]), float(data[1])

        return columns[kept[0]], columns[kept[1]], invert, depth

    def screen_to_volume(self, event):
        """Data coordinates on the two visible axes for a screen point."""
        mapping = self.volume_axis_map()
        ax = self.axes_at(0, 0)
        if mapping is None or ax is None:
            return None
        first, second, invert, depth = mapping
        limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
        origin = [lo for lo, _hi in limits]
        # Read at the MIDDLE of the depth axis rather than its near face, so
        # the error from a tilted view is centred instead of accumulating in
        # one direction.
        anchor = list(origin)
        anchor[depth] = (limits[depth][0] + limits[depth][1]) / 2.0
        base = np.asarray(ax.transData.transform(_project(ax, anchor)),
                          dtype=float)
        dx = float(getattr(event, "x", 0) or 0) - base[0]
        dy = float(getattr(event, "y", 0) or 0) - base[1]
        first_delta, second_delta = invert(dx, dy)
        kept = [a for a in range(3) if a != depth]
        return (first, origin[kept[0]] + first_delta,
                second, origin[kept[1]] + second_delta)

    def box_from_view(self) -> Optional[Gate]:
        """A box gate enclosing the volume's current limits.

        The view is the gesture. Spinning and zooming until a population
        fills the box is already the act of choosing it, and a rectangle
        dragged on a rotated projection has no defined extent along the axis
        pointing at the viewer -- reading one off would invent a number.
        """
        ax = self.axes_at(0, 0)
        spec = self._spec
        if ax is None or not hasattr(ax, "get_zlim"):
            return None
        if not (spec.x and spec.y and self._z_column):
            return None
        return BoxGate.from_limits(
            "(unnamed)", (spec.x, spec.y, self._z_column),
            (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))

    def _draw_box(self, ax, gate, colour) -> None:
        """The twelve edges of a box, drawn in the volume."""
        frame = self.population()
        def bound(low, high, column):
            if low is not None and high is not None:
                return float(low), float(high)
            values = pd.to_numeric(frame[column], errors="coerce") \
                if frame is not None and column in frame.columns else None
            lo = float(low) if low is not None else (
                float(np.nanmin(values)) if values is not None else 0.0)
            hi = float(high) if high is not None else (
                float(np.nanmax(values)) if values is not None else 1.0)
            return lo, hi

        x0, x1 = bound(gate.x_low, gate.x_high, gate.x_column)
        y0, y1 = bound(gate.y_low, gate.y_high, gate.y_column)
        z0, z1 = bound(gate.z_low, gate.z_high, gate.z_column)
        corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1)
                   for z in (z0, z1)]
        edges = [(a, b) for i, a in enumerate(corners)
                 for b in corners[i + 1:]
                 if sum(p != q for p, q in zip(a, b)) == 1]
        for a, b in edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color=colour, linewidth=self._line_width + 0.4, alpha=0.9)

    def _draw_volume_gates(self, ax, frame, palette) -> None:
        """Show each shown gate's objects in the volume.

        A 2D gate is a statement about two of the three measurements, so in a
        volume it is a COLUMN through the cloud rather than a closed region.
        Marking its objects says exactly that, and says it without pretending
        the gate bounds a depth it never mentioned.
        """
        spec = self._spec
        for gate in self._gates.gates:
            if not self.is_gate_enabled(gate.name):
                continue
            try:
                inside = self._gates.mask(frame, gate.name)
            except Exception:
                continue
            if not bool(np.any(inside)):
                continue
            colour = self.gate_colour(gate.name)
            if isinstance(gate, BoxGate) and gate.z_column == self._z_column:
                self._draw_box(ax, gate, colour)
            picked = frame.loc[inside]
            ax.scatter(
                pd.to_numeric(picked[spec.x], errors="coerce"),
                pd.to_numeric(picked[spec.y], errors="coerce"),
                pd.to_numeric(picked[self._z_column], errors="coerce"),
                s=18, facecolor="none", edgecolor=colour,
                linewidths=0.7, depthshade=False)

    def _on_button_release(self, event) -> None:
        """Square the volume up when a spin ends, if the setting says so.

        `snap_to_axis` was declared, given a control, saved and reloaded, and
        READ BY NOTHING -- a control that turns nothing is a promise the
        application does not keep. This is where it turns something.

        Only in 3D, and only when the view has actually been turned: snapping
        a volume nobody rotated would move a view the user set deliberately.
        """
        if self._mode not in ("3D", "xD"):
            return
        if not bool(getattr(self._settings, "snap_to_axis", False)):
            return
        if self._view_angles is None:
            return
        try:
            self.snap_to_nearest_axis()
        except Exception:
            LOG.debug("could not snap the view", exc_info=True)

    def _apply_spin_speed(self, axes) -> None:
        """Scale how far a drag turns the volume.

        matplotlib has no public setting for this: ``Axes3D`` converts the
        drag straight into degrees inside ``_on_move``. So the method is
        WRAPPED rather than reimplemented -- the wrapper scales the reported
        cursor movement and lets matplotlib do the rest, which keeps the
        rotation matplotlib's and the speed ours.

        Guarded end to end: a matplotlib whose internals moved leaves the
        rotation at its normal speed, which is a setting not taking effect
        rather than a volume that will not turn.
        """
        speed = float(getattr(self._settings, "spin_speed", 1.0) or 1.0)
        original = getattr(axes, "_on_move", None)
        if original is None or getattr(original, "_spacr_wrapped", False):
            return
        if abs(speed - 1.0) < 1e-9:
            return

        def scaled(event):
            try:
                start_x = getattr(axes, "_sx", None)
                start_y = getattr(axes, "_sy", None)
                if start_x is not None and event.x is not None:
                    event.x = start_x + (event.x - start_x) * speed
                if start_y is not None and event.y is not None:
                    event.y = start_y + (event.y - start_y) * speed
            except Exception:
                LOG.debug("could not scale the spin", exc_info=True)
            return original(event)

        scaled._spacr_wrapped = True
        try:
            axes._on_move = scaled
        except Exception:
            LOG.debug("this matplotlib does not allow a spin-speed wrap",
                      exc_info=True)

    def snap_to_nearest_axis(self) -> Tuple[float, float]:
        """Turn the volume square-on to whichever axis it is nearest.

        A volume stopped at an arbitrary angle cannot be read off at all --
        the point of snapping is that a 3D gate is always finally judged from
        a view where one measurement is flat.
        """
        axes = self.axes_at(0, 0)
        if axes is None:
            return (0.0, 0.0)
        elevation = min((0.0, 90.0, -90.0),
                        key=lambda e: abs(e - float(axes.elev)))
        azimuth = min((0.0, 90.0, 180.0, 270.0, 360.0),
                      key=lambda a: abs(a - (float(axes.azim) % 360)))
        azimuth = azimuth % 360
        axes.view_init(elev=elevation, azim=azimuth)
        self._view_angles = (elevation, azimuth)
        self._canvas.draw_idle()
        return (elevation, azimuth)

    # -- rendering --------------------------------------------------------
    def render_now(self) -> None:
        """Draw the parent's population, then the gates on top of it.

        In 3D the volume replaces all of it: the gate tools are 2D gestures
        on a flat axes, and running them against a rotated projection would
        produce gates whose coordinates mean nothing.
        """
        # xD renders as a volume as well when it has a third component: the
        # user picked PC1, PC2 and PC3 and got a 2D scatter, which reads as
        # the third component having been ignored.
        if self._mode in ("3D", "xD") and self._render_volume():
            return
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
        frame = self.population()
        for ax in axes.values():
            for gate in self._gates.gates:
                if not self._gate_is_on_these_axes(gate):
                    continue
                if not self.is_gate_enabled(gate.name):
                    continue
                if self._highlight_gated:
                    self._highlight(ax, gate, frame, palette)
                self._outline(ax, gate, palette)
                self._draw_handles(ax, gate, palette)
            if self._pending:
                self._outline_pending(ax, palette)
        self._canvas.draw_idle()

    def _highlight(self, ax, gate: Gate, frame, palette) -> None:
        """Mark the objects inside ``gate``, leaving every other point alone.

        This is what replaced replotting the gate's population: "i want it to
        highlight the datapoints in the gate and show the gate but also show
        the rest of the graph." The mask comes from the GateSet, so a child
        gate marks its own population and not its parent's.

        Failure here is silent and total -- a gate whose columns are missing
        from this table simply is not highlighted. The outline still draws, so
        the user sees the gate; a traceback out of a paint path would take the
        whole plot with it, and the highlight is decoration (INVARIANTS 10).
        """
        spec = self._spec
        if frame is None or frame.empty or not (spec.x and spec.y):
            return
        if spec.x not in frame.columns or spec.y not in frame.columns:
            return
        try:
            inside = self._gates.mask(frame, gate.name)
        except Exception:
            LOG.debug("cannot highlight %s here", gate.name, exc_info=True)
            return
        if inside is None or not bool(inside.any()):
            return
        marked = frame.loc[inside]
        self._artists.append(
            ax.scatter(marked[spec.x], marked[spec.y], s=14,
                       facecolor="none", edgecolor=self.gate_colour(gate.name),
                       linewidths=0.7, zorder=6))

    def _as_flat(self, gate: Gate) -> Gate:
        """A box seen from the front, so the 2D tools can draw and edit it.

        Its outline, handles and drag then all work unchanged, and the depth
        the flat view cannot express is left alone rather than silently reset.
        """
        if isinstance(gate, BoxGate):
            return gate.to_rect()
        return gate

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
        if isinstance(gate, BoxGate):
            # A box is drawn flat as its rectangle when its x and y are up.
            return {gate.x_column, gate.y_column} <= showing
        needed = set(gate.columns)
        if not needed:
            return False
        return needed <= showing

    def gate_colour(self, name: str) -> str:
        """The colour a gate is drawn in, stable for the life of the set.

        By POSITION in the gate set, so a gate keeps its colour while others
        are added, and so the outline, the ringed objects and the row in the
        gate list all agree -- which is the point: a plot with four gates on
        it should be readable without clicking each one.

        Falls back to the accent when a gate is not in the set (a shape being
        dragged out has no position yet).
        """
        names = list(self._gates.names)
        if name not in names:
            return active_palette()["accent"]
        return GATE_COLOURS[names.index(name) % len(GATE_COLOURS)]

    def _outline(self, ax, gate: Gate, palette) -> None:
        """Draw ``gate`` if it is a gate on the columns currently plotted.

        A gate on other columns is not drawn rather than approximated onto
        these axes: an outline in the wrong units is worse than no outline.
        """
        accent = self.gate_colour(gate.name)
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
        points = self._gate_points(ax, self._as_flat(gate))
        if not points:
            return
        patch = MplPolygon(points, closed=True, fill=False, edgecolor=accent,
                           linewidth=self._line_width, zorder=7)
        ax.add_patch(patch)
        self._artists.append(patch)
        ax.annotate(gate.name, points[0], color=accent, fontsize=7,
                    xytext=(2, 2), textcoords="offset points", zorder=8)

    def _gate_points(self, ax, gate: Gate) -> List[Tuple[float, float]]:
        """The outline of ``gate`` as a closed run of points.

        One geometry for the solid outline, the dashed ghost and the drag
        preview, so a gate cannot be drawn one shape and committed as
        another -- which is exactly what happened to the oval.

        EllipseGate had NO branch here at all: an oval was previewed while
        being dragged and then vanished the moment it became a gate. It is
        approximated as a polygon rather than an `Ellipse` patch so the ghost
        and the outline share this one path.
        """
        if isinstance(gate, RectGate):
            x0, x1, y0, y1 = self._rect_bounds(ax, gate)
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if isinstance(gate, PolygonGate):
            return list(gate.vertices)
        if isinstance(gate, EllipseGate):
            angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            return [(gate.x_centre + gate.x_radius * float(np.cos(a)),
                     gate.y_centre + gate.y_radius * float(np.sin(a)))
                    for a in angles]
        return []

    def _view(self, ax) -> Tuple[float, float, float, float]:
        """The visible limits, for placing handles on unbounded sides."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return float(x0), float(x1), float(y0), float(y1)

    def _handles_for(self, ax, gate: Gate) -> Tuple[Handle, ...]:
        """``gate``'s anchor points, or none if it is not on these axes."""
        if not self._gate_is_on_these_axes(gate):
            return ()
        try:
            return self._as_flat(gate).handles(self._view(ax))
        except Exception:
            LOG.debug("no handles for %s", gate.name, exc_info=True)
            return ()

    def _draw_handles(self, ax, gate: Gate, palette) -> None:
        """Draw ``gate``'s anchors: squares for corners, circles for sides."""
        handles = self._handles_for(ax, gate)
        if not handles:
            return
        accent = self.gate_colour(gate.name)
        for corner, marker in ((True, "s"), (False, "o")):
            picked = [h for h in handles if h.corner is corner]
            if not picked:
                continue
            self._artists.append(ax.plot(
                [h.x for h in picked], [h.y for h in picked],
                linestyle="none", marker=marker, markersize=5,
                markerfacecolor=palette["bg"], markeredgecolor=accent,
                markeredgewidth=1.2, zorder=9)[0])

    def handle_at(self, event) -> Optional[Tuple[str, str]]:
        """The anchor under the pointer as ``(gate name, role)``, or None.

        Measured in pixels for the same reason polygon-closing is: a tolerance
        in data units would be unusable on one axis whenever the two
        measurements have different ranges, which is nearly always.

        Only ENABLED gates are grabbable. A hidden gate is not on screen, and
        an invisible anchor that catches the mouse is indistinguishable from
        the plot being broken.
        """
        ax = getattr(event, "inaxes", None)
        # Pixel coordinates, via getattr: a real matplotlib event always
        # carries them, but a synthetic one raised from code (a test, a
        # scripted gate) need not, and no anchor is grabbable without them.
        ex, ey = getattr(event, "x", None), getattr(event, "y", None)
        if ax is None or ex is None or ey is None:
            return None
        best: Optional[Tuple[str, str]] = None
        best_distance = self.HANDLE_RADIUS_PX
        for gate in self._gates.gates:
            if not self.is_gate_enabled(gate.name):
                continue
            for handle in self._handles_for(ax, gate):
                try:
                    px, py = ax.transData.transform((handle.x, handle.y))
                except Exception:
                    continue
                distance = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
                if distance <= best_distance:
                    best_distance = distance
                    best = (gate.name, handle.role)
        return best

    # -- the shape that follows the mouse ---------------------------------
    def _clear_ghost(self) -> None:
        for artist in self._ghost:
            try:
                artist.remove()
            except Exception:
                pass
        self._ghost = []

    def _show_ghost(self, gate: Optional[Gate]) -> None:
        """Draw ``gate`` dashed, as the placeholder following the mouse.

        The gate being dragged is "picked up": its prospective shape is drawn
        dashed while the mouse moves and only becomes real on release. The
        committed gate stays drawn underneath, so the user can see where it
        was as well as where it is going.

        Only the ghost's own artists are touched. Redrawing every gate on
        every motion event would re-mask the whole table per gate per mouse
        move, which is what made applying the move on release necessary in
        the first place.
        """
        self._clear_ghost()
        axes = self.panel_axes()
        if gate is None or not axes:
            self._canvas.draw_idle()
            return
        from matplotlib.patches import Polygon as MplPolygon
        palette = active_palette()
        for ax in axes.values():
            if isinstance(gate, ThresholdGate):
                for bound in (gate.low, gate.high):
                    if bound is not None:
                        self._ghost.append(ax.axvline(
                            bound, color=palette["warning"], linewidth=1.4,
                            linestyle=":", zorder=10))
                continue
            points = self._gate_points(ax, gate)
            if not points:
                continue
            patch = MplPolygon(points, closed=True, fill=False,
                               edgecolor=palette["warning"], linewidth=1.4,
                               linestyle="--", zorder=10)
            ax.add_patch(patch)
            self._ghost.append(patch)
        self._canvas.draw_idle()

    def _dragged_to(self, event) -> Optional[Gate]:
        """The gate as it would be if the mouse were released here.

        One function for both gestures and for both the ghost and the commit,
        so the dashed shape cannot promise something the release does not do.
        """
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        x, y = float(event.xdata), float(event.ydata)
        if self._resize is not None:
            name, role = self._resize
            try:
                return self._gates.get(name).with_handle(role, x, y)
            except Exception:
                LOG.debug("cannot resize %s by %s", name, role, exc_info=True)
                return None
        name = self._move_name
        start = self._move_from
        if not name or start is None:
            return None
        try:
            return self._gates.get(name).translated(x - start[0], y - start[1])
        except Exception:
            return None

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

    def set_spin_axis(self, axis: str) -> None:
        """Lock the volume's rotation to one axis.

        "i want to be able to spinn allong axees not meev freely ... say click
        the y axis, then i should be able to spin on the x axis." Free
        rotation reaches angles from which nothing can be read, and getting
        back to a square-on view by hand is not realistic. Locked, a drag is
        one rotation about one axis and every view stays interpretable.
        """
        self._spin_axis = axis if axis in ("x", "y", "z", "") else "z"

    def _in_volume(self) -> bool:
        """Whether the volume is what is currently drawn."""
        return (self._mode in ("3D", "xD")
                and hasattr(self.axes_at(0, 0), "get_zlim"))

    def _volume_press(self, event) -> bool:
        """Start a spin. Returns True when the event belongs to the volume.

        The gate tools must not see it. That is the bug behind "i cant zoom in
        or spin on any of the axees. if i press pollygon and tried to draw a
        gate, then i could all of a suded spinn the graph" -- the 2D press
        handler was consuming the drag, and only the polygon tool, which
        ignores drags, let it through to matplotlib.
        """
        if not self._in_volume():
            return False
        if event.inaxes is None:
            return True
        # A tool armed AND a square-on view means the user is DRAWING, not
        # spinning. Both conditions matter: without a tool a drag is
        # navigation, and off-square the depth axis is tilted so a dragged
        # shape would be a shape in no particular measurement.
        if self._tool:
            corner = self.screen_to_volume(event)
            if corner is not None:
                self._volume_drag = corner
                return True
        self._spin_from = (float(getattr(event, "x", 0) or 0),
                           float(getattr(event, "y", 0) or 0))
        return True

    def _volume_motion(self, event) -> bool:
        if not self._in_volume():
            return False
        if self._volume_drag is not None:
            self._show_volume_drag(event)
            return True
        if self._spin_from is None or event.inaxes is None:
            return True
        ax = self.axes_at(0, 0)
        if ax is None or not hasattr(ax, "view_init"):
            return True
        x, y = float(getattr(event, "x", 0) or 0), float(getattr(event, "y", 0) or 0)
        dx, dy = x - self._spin_from[0], y - self._spin_from[1]
        self._spin_from = (x, y)

        elevation, azimuth = float(ax.elev), float(ax.azim)
        if self._spin_axis == "z":
            # Spinning about the vertical axis is a change of azimuth only:
            # the horizon stays level, which is what makes it readable.
            azimuth += dx * 0.5
        elif self._spin_axis in ("x", "y"):
            elevation = max(-90.0, min(90.0, elevation + dy * 0.5))
        else:
            azimuth += dx * 0.5
            elevation = max(-90.0, min(90.0, elevation + dy * 0.5))
        ax.view_init(elev=elevation, azim=azimuth)
        self._view_angles = (elevation, azimuth)
        self._canvas.draw_idle()
        return True

    def _volume_release(self, event) -> bool:
        if not self._in_volume():
            return False
        if self._volume_drag is not None:
            gate = self._gate_from_volume_drag(event)
            self._volume_drag = None
            self._clear_ghost()
            if gate is not None:
                self.gate_drawn.emit(gate)
            else:
                self.render_now()
            return True
        self._spin_from = None
        return True

    def _volume_scroll(self, event) -> bool:
        """Zoom the volume by scaling all three axes about their centres."""
        if not self._in_volume():
            return False
        ax = self.axes_at(0, 0)
        if ax is None or not hasattr(ax, "get_zlim"):
            return True
        step = getattr(event, "step", 0) or (
            1 if getattr(event, "button", "") == "up" else -1)
        self._volume_zoom = max(0.05, min(50.0,
                                          self._volume_zoom * (1.25 ** step)))
        self._apply_volume_zoom(ax)
        self._canvas.draw_idle()
        return True

    def _show_volume_drag(self, event) -> None:
        """The rectangle being swept, drawn flat on the snapped view."""
        corner = self.screen_to_volume(event)
        start = self._volume_drag
        ax = self.axes_at(0, 0)
        if corner is None or start is None or ax is None:
            return
        self._clear_ghost()
        first, x0, second, y0 = start
        _f, x1, _s, y1 = corner
        limits = {"x": ax.get_xlim3d(), "y": ax.get_ylim3d(),
                  "z": ax.get_zlim3d()}
        spec = self._spec
        depth_column = next(c for c in (spec.x, spec.y, self._z_column)
                            if c not in (first, second))
        depth = limits[{spec.x: "x", spec.y: "y",
                        self._z_column: "z"}[depth_column]]
        palette = active_palette()

        # Drawn at BOTH ends of the depth axis, which is what the gate is: a
        # rectangle extended all the way through the volume.
        order = {spec.x: 0, spec.y: 1, self._z_column: 2}
        for far in depth:
            points = []
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
                point = [None, None, None]
                point[order[first]] = px
                point[order[second]] = py
                point[order[depth_column]] = far
                points.append(point)
            xs, ys, zs = zip(*points)
            self._ghost.extend(ax.plot(xs, ys, zs, color=palette["warning"],
                                       linewidth=1.2, linestyle="--"))
        self._canvas.draw_idle()

    def _gate_from_volume_drag(self, event) -> Optional[Gate]:
        """The box a drag on the snapped view describes.

        Bounded on the two measurements the user could actually see, and
        UNBOUNDED on the one pointing at them. That is the honest reading of
        the gesture: they said nothing about depth, so the gate says nothing
        about depth -- and a box with an unbounded axis is exactly a
        rectangle extended through the volume.
        """
        corner = self.screen_to_volume(event)
        start = self._volume_drag
        if corner is None or start is None:
            return None
        first, x0, second, y0 = start
        _f, x1, _s, y1 = corner
        if x0 == x1 or y0 == y1:
            return None

        spec = self._spec
        depth_column = next(c for c in (spec.x, spec.y, self._z_column)
                            if c not in (first, second))

        # THE TOOL DECIDES THE SHAPE, on the plane the drag happened in.
        # Instruction 52 asks for a circle, a rectangle and a polygon on the
        # anchor plane; the first two are drags and are here. All three
        # extrude along the same depth axis, and all three are UNBOUNDED on
        # it for the reason in this method's docstring.
        if self._tool == ELLIPSE:
            return CylinderGate(
                name="(unnamed)",
                u_column=first, v_column=second, axis_column=depth_column,
                u_centre=(x0 + x1) / 2.0, v_centre=(y0 + y1) / 2.0,
                u_radius=abs(x1 - x0) / 2.0, v_radius=abs(y1 - y0) / 2.0)

        bounds = {first: (min(x0, x1), max(x0, x1)),
                  second: (min(y0, y1), max(y0, y1))}
        def side(column):
            return bounds.get(column, (None, None))
        x_low, x_high = side(spec.x)
        y_low, y_high = side(spec.y)
        z_low, z_high = side(self._z_column)
        return BoxGate(name="(unnamed)",
                       x_column=spec.x, y_column=spec.y,
                       z_column=self._z_column,
                       x_low=x_low, x_high=x_high,
                       y_low=y_low, y_high=y_high,
                       z_low=z_low, z_high=z_high)

    def anchor_plane(self) -> Optional[Tuple[str, str, str]]:
        """``(first, second, normal)`` of the plane a drag would land on.

        Instruction 52 point 1 asks for the anchor plane to be VISIBLE before
        the user commits to drawing. It is already implicit -- a drag on the
        snapped view is read in the two measurements facing the camera -- but
        implicit is exactly the problem: the affordance has to say which
        surface the next shape lands on.

        :returns: None in 2D, or when the view is not square-on to a face and
            so has no plane to name.
        """
        if self._mode not in ("3D", "xD"):
            return None
        axes = self.axes_at(0, 0)
        spec = self._spec
        if axes is None or not (spec.x and spec.y and self._z_column):
            return None
        try:
            elevation = float(axes.elev)
            azimuth = float(axes.azim) % 360
        except Exception:
            return None
        # Square-on only. Off a face there is no plane a drag means, which is
        # why `snap_to_axis` exists -- saying nothing is better than naming
        # a plane the user is not looking at.
        if abs(elevation) > 1e-6 and abs(abs(elevation) - 90.0) > 1e-6:
            return None
        if abs(abs(elevation) - 90.0) < 1e-6:
            return (spec.x, spec.y, self._z_column)
        if min(abs(azimuth), abs(azimuth - 360)) < 1e-6:
            return (spec.y, self._z_column, spec.x)
        if abs(azimuth - 90.0) < 1e-6 or abs(azimuth - 270.0) < 1e-6:
            return (spec.x, self._z_column, spec.y)
        if abs(azimuth - 180.0) < 1e-6:
            return (spec.y, self._z_column, spec.x)
        return None

    def _draw_anchor_aura(self, ax) -> None:
        """The blue hue on the plane the next shape would land on.

        A translucent FILLED quad rather than an edge highlight, because
        point 1 asks for it to be visible from any camera angle and an edge
        disappears the moment it points at the viewer.
        """
        plane = self.anchor_plane()
        if plane is None:
            return
        first, second, normal = plane
        spec = self._spec
        axis_of = {spec.x: "x", spec.y: "y", self._z_column: "z"}
        limits = {"x": ax.get_xlim3d(), "y": ax.get_ylim3d(),
                  "z": ax.get_zlim3d()}
        try:
            u0, u1 = limits[axis_of[first]]
            v0, v1 = limits[axis_of[second]]
            far = limits[axis_of[normal]][0]
        except KeyError:
            return
        order = {spec.x: 0, spec.y: 1, self._z_column: 2}
        corners = []
        for pu, pv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            point = [None, None, None]
            point[order[first]] = pu
            point[order[second]] = pv
            point[order[normal]] = far
            corners.append(point)
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            quad = Poly3DCollection(
                [corners], facecolor=active_palette()["accent"], alpha=0.12,
                edgecolor=active_palette()["accent"], linewidths=0.8)
            ax.add_collection3d(quad)
            self._artists.append(quad)
        except Exception:
            LOG.debug("could not draw the anchor plane", exc_info=True)

    def _apply_volume_zoom(self, ax) -> None:
        """Scale the three axes about the data's centre.

        The limits, not the camera: a gate is a statement in data units, and a
        camera trick would leave the outlines somewhere other than the objects
        they enclose.
        """
        spec = self._spec
        frame = self.population()
        if frame is None:
            return
        factor = 1.0 / float(self._volume_zoom)
        for column, setter in ((spec.x, ax.set_xlim3d),
                               (spec.y, ax.set_ylim3d),
                               (self._z_column, ax.set_zlim3d)):
            if not column or column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            centre = float(values.mean())
            half = max(float(values.std()) * 3.0,
                       (float(values.max()) - float(values.min())) / 2.0) or 1.0
            setter(centre - half * factor, centre + half * factor)

    def _on_press(self, event) -> None:
        if self._volume_press(event):
            return
        # A press inside an existing gate MOVES it, whatever tool is armed.
        # Checked first because "the closed gate should be draggable" has to
        # hold without the user first disarming the tool they drew it with --
        # nobody thinks of that, and the gate then looks stuck.
        #
        # The one exception is mid-polygon: there the user is placing
        # vertices, and a vertex that happens to land inside an older gate
        # must not drag it.
        mid_polygon = self._tool == POLYGON and bool(self._pending)
        if not mid_polygon:
            # An anchor point is tested BEFORE the shape, because every anchor
            # sits on or inside its own gate. Testing the shape first would
            # mean a press on a corner moved the whole gate and resizing were
            # unreachable.
            grabbed = self.handle_at(event)
            if grabbed is not None:
                self._resize = grabbed
                return
        if not mid_polygon and event.inaxes is not None \
                and event.xdata is not None and event.ydata is not None:
            name = self.gate_at(float(event.xdata), float(event.ydata))
            if name:
                self._move_name = name
                self._move_from = (float(event.xdata), float(event.ydata))
                return
        if self._tool == WAND:
            self._wand_at(event)
            return
        if self._tool != POLYGON:
            super()._on_press(event)
            return
        if self._mode in ("3D", "xD"):
            # In the volume, `event.xdata` is a projected screen coordinate
            # and means nothing in data units. The same reader the drag tools
            # use answers this properly, in the two measurements facing the
            # camera.
            placed = self.screen_to_volume(event)
            if placed is None:
                return
            first, x, second, y = placed
            plane = (first, second)
            if self._pending and self._pending_plane != plane:
                # The view turned mid-polygon. Vertices from two planes are
                # not one shape, and quietly mixing them would produce a
                # prism whose outline nobody drew.
                LOG.debug("polygon abandoned: the view turned mid-shape")
                self._pending = []
            self._pending_plane = plane
            if len(self._pending) >= 3 and self._near_first_vertex(event, x, y):
                self.close_polygon_now()
                return
            self._pending.append((x, y))
            self.polygon_changed.emit(len(self._pending))
            self._draw_gates()
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

    def _wand_at(self, event) -> None:
        """Grow a gate from a click and offer it like any other drawn gate.

        The wand emits `gate_drawn`, so it lands in the same naming and
        undo path as a dragged shape -- it is a way of PRODUCING a polygon,
        not a fourth kind of gate.

        A click that cannot grow one reports why in the status line rather
        than raising: the two things that make it fail, clicking in empty
        space and a tolerance too small for this cloud, are both things the
        user fixes by clicking again.
        """
        from .gate_spec import WandError, wand_gate

        spec = self._spec
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        frame = self.population()
        if frame is None or frame.empty or not (spec.x and spec.y):
            self.wand_failed.emit(
                "the wand needs a table and two measurements on screen")
            return
        settings = self._settings
        try:
            gate = wand_gate(
                frame, spec.x, spec.y,
                float(event.xdata), float(event.ydata),
                tolerance=float(getattr(settings, "wand_tolerance", 0.05)),
                max_radius=float(getattr(settings, "wand_max_radius", 0.35)))
        except WandError as exc:
            self.wand_failed.emit(str(exc))
            return
        except GateError as exc:
            self.wand_failed.emit(str(exc))
            return
        self.gate_drawn.emit(gate)

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

    def _on_scroll(self, event) -> None:
        """Zoom about the pointer with the wheel.

        About the POINTER rather than the centre: zooming toward what you are
        looking at is what every map does, and centre-zoom means chasing a
        feature back into view after every notch.

        Data limits, not a transform, so the gates -- which are drawn in data
        coordinates -- stay exactly where they belong on the measurements.
        """
        if self._volume_scroll(event):
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or event.xdata is None or event.ydata is None:
            return
        step = getattr(event, "step", 0) or (
            1 if getattr(event, "button", "") == "up" else -1)
        factor = 0.8 ** float(step)
        def zoomed(limits, anchor):
            low, high = limits
            return (anchor + (low - anchor) * factor,
                    anchor + (high - anchor) * factor)

        # REMEMBERED, not just set. A redraw re-applies the computed scales
        # after the marks are drawn, so limits set here alone are undone by
        # the next render -- which is every gate edit.
        self._zoom = (zoomed(ax.get_xlim(), float(event.xdata)),
                      zoomed(ax.get_ylim(), float(event.ydata)))
        self.render_now()

    def reset_view(self) -> None:
        """Back to the limits the data asks for, and the starting angle."""
        self._zoom = None
        self._volume_zoom = 1.0
        self._view_angles = None
        self.render_now()

    #: Kept as the old name: `reset_zoom` was the 2D-only version.
    reset_zoom = reset_view

    def _apply_scales(self, ax, kind, scales, panel) -> None:
        """Let a wheel zoom outlive the redraw that follows it.

        The computed scales are applied AFTER the marks are drawn, so limits
        set by the wheel alone are undone by the next render -- and a render
        happens on every gate edit. Re-applying here is what makes the zoom a
        state of the view rather than a gesture that survives until the next
        click.
        """
        super()._apply_scales(ax, kind, scales, panel)
        if self._zoom is None:
            return
        (x_limits, y_limits) = self._zoom
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    def _on_motion(self, event) -> None:
        if self._volume_motion(event):
            return
        if self._resize is not None or getattr(self, "_move_name", None):
            # The EDIT is applied on release -- a gate is re-masked against
            # the whole table to redraw, and doing that per mouse move makes
            # a large frame unusable. What follows the mouse is a dashed
            # placeholder in the shape of the gate, which costs one polygon.
            self._show_ghost(self._dragged_to(event))
            return
        if self._tool == POLYGON:
            return
        super()._on_motion(event)

    def _on_release(self, event) -> None:
        if self._volume_release(event):
            return
        if self._resize is not None:
            name, _role = self._resize
            edited = self._dragged_to(event)
            self._resize = None
            self._clear_ghost()
            if edited is None:
                # Released off the axes, or the pull would have collapsed the
                # shape. The gate is redrawn as it was rather than left with
                # a ghost hanging over it.
                self.render_now()
                return
            self.gate_edited.emit(edited)
            return
        name = getattr(self, "_move_name", None)
        if name:
            start = getattr(self, "_move_from", None)
            self._move_name = None
            self._move_from = None
            self._clear_ghost()
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

    # A drawn gate is TOP-LEVEL. It used to take its parent from the active
    # gate, and drawing one selects it -- so the second gate nested inside the
    # first, the third inside the second, and so on without anyone asking:
    # "in the gate view it looks like the second gate is in the first and the
    # thired gate is in the second". Worse, a nested gate is ANDed with its
    # ancestors (`GateSet.mask` walks the path), so those gates were quietly
    # not the shapes that were drawn.
    #
    # The hierarchy itself is kept -- it round-trips through save/load and
    # clustering still uses it -- but nesting is now something a caller asks
    # for, never a side effect of what happens to be selected.
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
            return ThresholdGate(name=name, column=column, low=x0, high=x1)
        if self._tool == RECTANGLE:
            if not (spec.x and spec.y):
                return None
            return RectGate(name=name,
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
            return EllipseGate.from_drag(name, spec.x, spec.y, x0, y0, x1, y1)
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
        if self._mode in ("3D", "xD") and self._pending_plane:
            first, second = self._pending_plane
            normal = next((c for c in (spec.x, spec.y, self._z_column)
                           if c not in (first, second)), "")
            if not normal:
                return None
            # Unbounded along the normal, like every other shape drawn on the
            # anchor plane: they said nothing about depth.
            gate = PrismGate(name=name, u_column=first, v_column=second,
                             axis_column=normal,
                             vertices=tuple(self._pending))
            self._pending = []
            self._pending_plane = None
            self.polygon_changed.emit(0)
            self.gate_drawn.emit(gate)
            return gate
        gate = PolygonGate(name=name,
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
    #: A gate was ticked or unticked — carries the name and whether it is on.
    enabled_changed = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GateTree")
        self._gates = GateSet()
        self._frame: Optional[pd.DataFrame] = None
        self._colour_source = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("GateHierarchy")
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Gate", "n", "% parent", "% all"])
        header = self.tree.header()
        # Every column visible on startup. Stretching column 0 and leaving the
        # rest at their default width pushed n / % parent / % all off the edge
        # of a narrow panel, so the counts -- the reason the tree has columns
        # at all -- could not be seen until the user resized something.
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(44)
        self.tree.setToolTip(
            "The gates you have drawn. Tick one to show it on the plot and "
            "highlight its objects, untick it to hide it. Selecting a gate "
            "sets the axes to its measurements and makes the next gate you "
            "draw a child of it — it never changes what the plot shows.")
        self.tree.currentItemChanged.connect(self._on_selection)
        self.active_changed.connect(self._rebuild_thresholds)
        self.tree.itemChanged.connect(self._on_item_changed)
        #: Gates the user has unticked. The tree owns this because the tick
        #: is in the tree; the canvas is told, and does not have to be asked.
        self._disabled: set = set()
        #: Set while `refresh` is rebuilding, because setting a check state
        #: fires `itemChanged` and a rebuild would otherwise report every
        #: gate as freshly toggled by the user.
        self._rebuilding = False
        # `GateEditorPanel` is transparent scaffolding by design (see
        # the GraphBuilder block), so the hierarchy has nothing behind
        # it and is the page itself.
        mark_surface(self.tree)
        outer.addWidget(self.tree, 1)

        # Instruction 52 point 4: "the user should also be able to set
        # thresholds for each individual gate for the measurements they are
        # defined by". One row per measurement the SELECTED gate can take a
        # threshold on -- which for a cylinder is its normal, and is how its
        # height is bounded.
        #
        # Rebuilt on selection rather than kept for every gate: a panel
        # holding rows for gates nobody has selected is a panel that has to
        # keep them in step with edits made elsewhere.
        self._thresholds = QWidget(self)
        self._threshold_form = QFormLayout(self._thresholds)
        self._threshold_form.setContentsMargins(0, 0, 0, 0)
        self._threshold_rows: Dict[str, Tuple[QLineEdit, QLineEdit]] = {}
        #: Which gate the rows above belong to. Remembered rather than
        #: re-read from the selection: if the selection moves between a row
        #: being filled in and the edit landing, re-reading would put the
        #: number on the wrong gate.
        self._threshold_gate: str = ""
        self._thresholds.setVisible(False)
        outer.addWidget(self._thresholds)

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
        self._rebuilding = True
        try:
            self._rebuild(current)
        finally:
            self._rebuilding = False

    def _rebuild(self, current: str) -> None:
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
            item.setCheckState(0, Qt.Unchecked if gate.name in self._disabled
                               else Qt.Checked)
            colour = self._colour_for(gate.name)
            if colour:
                # The gate's own colour, on its name. This is the half that
                # makes colour-coding useful: a colour on the plot that is not
                # also in the list is a colour with nothing to look it up in.
                item.setForeground(0, QBrush(QColor(colour)))
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

    def _colour_for(self, name: str) -> str:
        """The gate's colour, asked of whoever is drawing it.

        The canvas owns the mapping so the two cannot disagree; the tree only
        displays it. Returns "" when there is no canvas -- the tree is usable
        on its own, and a missing colour is not worth failing over.
        """
        source = getattr(self, "_colour_source", None)
        if source is None:
            return ""
        try:
            return str(source(name) or "")
        except Exception:
            return ""

    def set_colour_source(self, source) -> None:
        """Tell the tree where gate colours come from -- see `_colour_for`."""
        self._colour_source = source
        self.refresh()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """A tick changed — unless the tree is rebuilding itself."""
        if self._rebuilding or column != 0 or item is None:
            return
        name = item.data(0, Qt.UserRole)
        if not name:
            return
        on = item.checkState(0) == Qt.Checked
        if on:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)
        self.enabled_changed.emit(name, on)

    def is_enabled(self, name: str) -> bool:
        """Whether ``name`` is ticked. Unknown gates are on."""
        return name not in self._disabled

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

    def _rebuild_thresholds(self, name: str) -> None:
        """Show one low/high pair per measurement the selected gate can bound.

        Blank means UNBOUNDED, not zero. That distinction is the whole
        interface here: a cylinder with no bound on its normal means the 2D
        oval extended through the volume, and a cylinder bounded 0..0 means
        nothing at all.
        """
        while self._threshold_form.rowCount():
            self._threshold_form.removeRow(0)
        self._threshold_rows = {}
        self._threshold_gate = str(name or "")
        gate = None
        if name and name in self._gates:
            gate = self._gates.get(name)
        offered = {}
        if gate is not None:
            try:
                offered = gate.thresholds()
            except Exception:
                LOG.debug("could not read thresholds", exc_info=True)
        self._thresholds.setVisible(bool(offered))
        for column, (low, high) in offered.items():
            pair = QWidget(self._thresholds)
            line = QHBoxLayout(pair)
            line.setContentsMargins(0, 0, 0, 0)
            low_edit, high_edit = QLineEdit(pair), QLineEdit(pair)
            for edit, value, hint in ((low_edit, low, "min"),
                                      (high_edit, high, "max")):
                edit.setText("" if value is None else f"{float(value):g}")
                edit.setPlaceholderText(hint)
                edit.setToolTip(
                    f"Threshold on {column} for this gate. Leave it EMPTY "
                    f"for no bound \u2014 empty is unbounded, not zero.")
                edit.editingFinished.connect(
                    lambda c=column: self._apply_threshold(c))
                line.addWidget(edit)
            self._threshold_rows[column] = (low_edit, high_edit)
            self._threshold_form.addRow(column, pair)

    def _apply_threshold(self, column: str) -> None:
        """Put an edited pair back on the gate."""
        name = self._threshold_gate
        if not name or name not in self._gates:
            return
        low_edit, high_edit = self._threshold_rows.get(column, (None, None))
        if low_edit is None:
            return

        def value(edit):
            text = edit.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        try:
            updated = self._gates.get(name).with_threshold(
                column, value(low_edit), value(high_edit))
        except GateError:
            LOG.debug("gate %s cannot take a threshold on %s", name, column)
            return
        self._gates.add(updated)
        self._rebuild_thresholds(name)
        self.gates_changed.emit()

    def _on_selection(self, *_args) -> None:
        self.active_changed.emit(self.active_gate())


@dataclass(frozen=True)
class _ClusterRun:
    """One clustering pass's parameters, from wherever they came from.

    The dialog and the Search tab are two editors of the same five numbers.
    Naming them once here is what lets `run_cluster` have ONE body -- and
    the modal's own docstring records what happens when two editors of the
    same settings drift: it opened on hardcoded 0.30/10 while Gate Settings
    offered 0.5/20, and the values the user set were discarded.
    """

    eps: float
    min_samples: int
    scale: bool
    walk: bool
    walk_steps: int
    method: str


class _ClusterSettingsDialog(QDialog):
    """DBSCAN's two parameters, with what they mean in the units they act in.

    `eps` is in SCALED units while scaling is on, which is what makes one
    default work across measurements whose ranges differ by orders of
    magnitude -- `cell_area` runs to thousands and `eccentricity` to one, and
    unscaled DBSCAN on that pair clusters on area alone. The checkbox says so
    rather than leaving the user to discover it by getting one blob.

    SEEDED FROM THE SAVED GATE SETTINGS, which it did not used to be. Gate
    Settings has offered `cluster_eps`, `cluster_min_samples` and
    `cluster_scale` for as long as this dialog has existed, and this dialog
    opened on its own hardcoded 0.30/10 regardless -- so values the user set
    deliberately were discarded, and the two disagreed about the defaults as
    well (0.5 and 20 against 0.30 and 10). `settings` is optional only
    because the dialog is constructible before `apply_settings` has run.
    """

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        try:
            from ..dialogs import detach_from_window_manager
            detach_from_window_manager(self)
        except Exception:
            pass
        self.setWindowTitle("Cluster settings")
        form = QFormLayout(self)

        # One place decides each default: the GateEditorSettings dataclass. Reading
        # through getattr keeps an older saved settings object -- one written
        # before a field existed -- from raising here.
        from .gate_settings import GateEditorSettings
        fallback = GateEditorSettings()
        source = settings if settings is not None else fallback

        def _setting(name):
            value = getattr(source, name, None)
            return getattr(fallback, name) if value is None else value

        self._eps = QDoubleSpinBox(self)
        self._eps.setRange(0.01, 100.0)
        self._eps.setSingleStep(0.05)
        self._eps.setDecimals(2)
        self._eps.setValue(float(_setting("cluster_eps")))
        self._eps.setToolTip(
            "Neighbourhood radius. Larger merges nearby populations into "
            "one; smaller splits one into several.")
        form.addRow("eps (radius)", self._eps)

        self._min_samples = QSpinBox(self)
        self._min_samples.setRange(2, 10000)
        self._min_samples.setValue(int(_setting("cluster_min_samples")))
        self._min_samples.setToolTip(
            "Objects needed to seed a population. Anything sparser is "
            "treated as debris and left out of every gate.")
        form.addRow("min samples", self._min_samples)

        self._scale = Toggle("Standardise both axes first", self)
        self._scale.setChecked(bool(_setting("cluster_scale")))
        self._scale.setToolTip(
            "On unless you know otherwise. Without it, the axis with the "
            "larger numeric range decides the clustering on its own.")
        form.addRow("", self._scale)

        self._walk = Toggle("Walk eps and use the best radius", self)
        self._walk.setChecked(bool(_setting("cluster_walk")))
        self._walk.setToolTip(
            "Try a range of radii around the one above, score each by how "
            "well separated the populations are, and cluster at the best "
            "one. Use it when you do not know what eps should be.")
        form.addRow("", self._walk)

        self._walk_steps = QSpinBox(self)
        self._walk_steps.setRange(2, 200)
        self._walk_steps.setValue(int(_setting("cluster_walk_steps")))
        self._walk_steps.setToolTip(
            "How many radii to try. Each one is a full DBSCAN pass, so this "
            "is what the search costs.")
        self._walk_steps.setEnabled(self._walk.isChecked())
        self._walk.toggled.connect(self._walk_steps.setEnabled)
        form.addRow("walk steps", self._walk_steps)

        #: Not offered again here -- the algorithm is a Gate Settings
        #: decision, and this dialog is the per-run tuning of it. Carried so
        #: the run uses the method that was chosen, which is the whole
        #: defect: the picker existed and `cluster_gates` ran DBSCAN anyway.
        self._method = str(_setting("cluster_method"))

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

    def walk(self) -> bool:
        return bool(self._walk.isChecked())

    def walk_steps(self) -> int:
        return int(self._walk_steps.value())

    def method(self) -> str:
        return self._method


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
    #: The Settings button was pressed. The panel does not own the settings
    #: window -- the screen does, because sampling is the screen's job.
    settings_requested = Signal()
    #: The xD projection was switched on or off. Carries a bool.
    projection_requested = Signal(bool)
    #: A gating mode was chosen: "2D" or "3D".
    mode_requested = Signal(str)
    #: The volume's spin axis changed: "x", "y" or "z".
    spin_axis_changed = Signal(str)

    def __init__(self, parent=None, *, link=None,
                 source: str = "gate_editor"):
        super().__init__(parent)
        self.setObjectName("GateEditorPanel")
        self._gates = GateSet()
        self._frame: Optional[pd.DataFrame] = None
        self._namer = None
        #: The gate-editor settings, kept because the CLUSTER button needs
        #: them and the canvas only takes the drawing ones. None until
        #: `apply_settings` runs, which is why every read below falls back to
        #: the dataclass default rather than assuming this is set.
        self._settings = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(SPACING["xs"])
        self._tool = QComboBox(self)
        self._tool.setObjectName("GateToolPicker")
        for key in ("",) + GATE_KINDS:
            if key in (BOX, CYLINDER, PRISM, COMPOSITE):
                # Not drag tools. The first three come from the 3D view: a
                # shape
                # dragged on a rotated projection has no defined extent
                # along the axis pointing at the viewer, so offering them
                # here would promise a gesture that cannot work. A composite
                # is not drawn at all -- it is made from gates that already
                # exist, in the gates panel.
                continue
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

        # No Close polygon button. Clicking the first vertex closes the shape,
        # which is what everyone tries; once that worked, the button was a
        # second way to do one thing and the only one that had to be found.

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.setObjectName("GateSettingsButton")
        self._settings_button.setToolTip("Gate editor settings")
        self._settings_button.clicked.connect(self.settings_requested.emit)
        fit_to_text(self._settings_button)
        tools.addWidget(self._settings_button)

        self._reset_view = QPushButton("Reset view", self)
        self._reset_view.setToolTip(
            "Back to the limits the data asks for, after zooming or spinning "
            "too far. Gates are untouched — this moves the view, never them.")
        self._reset_view.clicked.connect(self.reset_view)
        fit_to_text(self._reset_view)
        tools.addWidget(self._reset_view)

        self._cluster = QPushButton("Cluster…", self)
        self._cluster.setToolTip(
            "Find dense populations with DBSCAN and turn each one into a "
            "gate you can edit, nest and save like any other.")
        self._cluster.clicked.connect(self._on_cluster)
        fit_to_text(self._cluster)
        tools.addWidget(self._cluster)

        # 2D / 3D / xD, right of Cluster. Checkable and exclusive: the mode is
        # one choice, and three buttons that can all be on describe a state
        # the editor does not have.
        self._mode_buttons: Dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode in ("2D", "3D"):
            button = QPushButton(mode, self)
            button.setCheckable(True)
            button.setChecked(mode == "2D")
            # Width from the TEXT, not a number. A fixed 38px clipped "2D",
            # "3D" and "xD" on the sides, and any fixed size is a promise
            # about a font the app does not control -- the user's theme, DPI
            # and platform all change it.
            fit_to_text(button, padding=18)
            button.clicked.connect(
                lambda _checked=False, m=mode: self.mode_requested.emit(m))
            group.addButton(button)
            self._mode_buttons[mode] = button
            tools.addWidget(button)

        # OUTSIDE the exclusive group, deliberately. xD is not a third
        # dimensionality: it says what the AXES ARE -- components rather than
        # raw measurements -- and that is orthogonal to how many are drawn.
        # Gating PC1 vs PC2 in 2D and PC1/PC2/PC3 in 3D are both things
        # people want, and one exclusive group could express neither.
        self._xd_button = QPushButton("xD", self)
        self._xd_button.setCheckable(True)
        self._xd_button.setToolTip(
            "Project the chosen measurements onto components and gate on "
            "those. Independent of 2D/3D \u2014 pick how many axes there "
            "are separately.\n\nWhich measurements are reduced is the xD "
            "tab of the settings.")
        fit_to_text(self._xd_button, padding=18)
        self._xd_button.toggled.connect(self.projection_requested.emit)
        tools.addWidget(self._xd_button)

        # Which axis the volume spins about. Shown only in 3D, because in 2D
        # there is nothing to spin and a dead control is worse than no
        # control.
        self._box_gate = QPushButton("Box gate", self)
        self._box_gate.setToolTip(
            "Turn what is currently in view into a gate on all three "
            "measurements. Spin and zoom until a population fills the box, "
            "then keep it — on a rotated projection the view IS the gesture, "
            "and a shape dragged on it would have no defined depth.")
        self._box_gate.clicked.connect(self.gate_from_view)
        fit_to_text(self._box_gate)
        tools.addWidget(self._box_gate)

        self._spin_label = QLabel("spin", self)
        tools.addWidget(self._spin_label)
        self._spin_buttons: Dict[str, QPushButton] = {}
        spin_group = QButtonGroup(self)
        spin_group.setExclusive(True)
        for axis in ("x", "y", "z"):
            button = QPushButton(axis.upper(), self)
            button.setCheckable(True)
            button.setChecked(axis == "z")
            button.setToolTip(
                f"Spin about {axis.upper()}. Locked to one axis, a drag is "
                f"one rotation and every view stays readable; free rotation "
                f"reaches angles nothing can be read from.")
            button.clicked.connect(
                lambda _checked=False, a=axis: self.spin_axis_changed.emit(a))
            fit_to_text(button, padding=14)
            spin_group.addButton(button)
            self._spin_buttons[axis] = button
            tools.addWidget(button)
        self.set_spin_controls_visible(False)

        # No Apply button. A gate highlights its objects the moment it is
        # shown (the tick in the gate list), so a button whose job was "now
        # make it count" describes a step that no longer exists.

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
        self.canvas.wand_failed.connect(self._status.setText)
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
        self.tree.enabled_changed.connect(self.canvas.set_gate_enabled)
        self.tree.set_colour_source(self.canvas.gate_colour)
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
        self._refresh_status()

    def _on_polygon_changed(self, count: int) -> None:
        if count:
            self._status.setText(
                f"{count} vertex(es) — three or more make a region")

    def _on_cluster(self) -> None:
        """The Cluster… button: ask, then run."""
        self.run_cluster(ask=True)

    def run_cluster(self, *, ask: bool = True) -> None:
        """Find dense populations and add one gate per cluster.

        Clusters become REAL gates rather than a separate kind of selection,
        so each is editable, nestable, serialisable and usable as a filter
        the moment it appears -- everything a hand-drawn gate can do, because
        it is one.

        :param ask: open the parameter dialog first. The Cluster… button
            does; the Search TAB does not, because the tab IS the parameter
            editor -- asking again there would be asking twice for the same
            numbers. Both read the same settings object, which is what stops
            the two from disagreeing.
        """
        from PySide6.QtWidgets import QMessageBox

        frame = self.canvas.population()
        if frame is None or frame.empty:
            QMessageBox.information(
                self, "Nothing to cluster",
                "Load a table before clustering.")
            return
        # The axes live on the SPEC. `canvas.x_column` has never existed, so
        # getattr always returned the default and clustering refused two
        # measurements that were plainly chosen -- "when i press cluster i get
        # 'Clustering needs an X and a Y measurement.' when both are cohosen".
        spec = self.canvas.spec
        x_column = getattr(spec, "x", None) or ""
        y_column = getattr(spec, "y", None) or ""
        if not x_column or not y_column:
            QMessageBox.information(
                self, "Pick two measurements",
                "Clustering needs an X and a Y measurement.")
            return

        if ask:
            dialog = _ClusterSettingsDialog(self, settings=self._settings)
            if dialog.exec() != QDialog.Accepted:
                return
            params = _ClusterRun(dialog.eps(), dialog.min_samples(),
                                 dialog.scale(), dialog.walk(),
                                 dialog.walk_steps(), dialog.method())
        else:
            settings = self._settings
            params = _ClusterRun(
                float(getattr(settings, "cluster_eps", 0.5)),
                int(getattr(settings, "cluster_min_samples", 20)),
                bool(getattr(settings, "cluster_scale", True)),
                bool(getattr(settings, "cluster_walk", False)),
                int(getattr(settings, "cluster_walk_steps", 12)),
                str(getattr(settings, "cluster_method", "dbscan")))

        from .gate_spec import (ClusterError, best_cluster_candidate,
                                cluster_gates, cluster_walk_candidates)
        eps = params.eps
        chosen = None
        try:
            if params.walk:
                candidates = cluster_walk_candidates(
                    frame, x_column, y_column,
                    eps=eps, min_samples=params.min_samples,
                    scale=params.scale, steps=params.walk_steps,
                    method=params.method)
                chosen = best_cluster_candidate(candidates)
                if chosen is None:
                    # Named rather than silently falling back to the typed
                    # eps: a walk that found nothing defensible is a result
                    # about the DATA, and clustering at the original radius
                    # anyway would present it as if the search had endorsed
                    # it.
                    tried = ", ".join(f"{c.eps:.3g}" for c in candidates)
                    QMessageBox.information(
                        self, "The walk found nothing to recommend",
                        "No radius produced two or more populations while "
                        "keeping most of the objects.\n\nTried: "
                        f"{tried}\n\nLower min samples, pick measurements "
                        "that separate the populations, or turn the walk "
                        "off and set eps yourself.")
                    return
                eps = chosen.eps
            found = cluster_gates(
                frame, x_column, y_column,
                eps=eps, min_samples=params.min_samples,
                scale=params.scale, method=params.method,
                parent=self.canvas.active_gate())
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
        if chosen is not None:
            # What the walk decided, in the units the user typed in, so the
            # number can be carried back to Gate Settings by hand. A search
            # that silently substitutes a parameter is worse than one that
            # never ran.
            QMessageBox.information(
                self, "Walk finished",
                f"Clustered at eps {chosen.eps:.3g}, which gave "
                f"{chosen.clusters} populations and left "
                f"{chosen.noise_fraction:.0%} of objects outside them.")

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

    def set_projection_active(self, on: bool) -> None:
        """Show the xD button as on or off without re-emitting.

        Used when a projection was asked for and could not be made: the
        button must not keep claiming something that did not happen.
        """
        button = getattr(self, "_xd_button", None)
        if button is None:
            return
        blocked = button.blockSignals(True)
        button.setChecked(bool(on))
        button.blockSignals(blocked)

    def set_spin_controls_visible(self, visible: bool) -> None:
        self._box_gate.setVisible(visible)
        self._spin_label.setVisible(visible)
        for button in self._spin_buttons.values():
            button.setVisible(visible)

    def gate_from_view(self) -> None:
        """Make a box gate out of what the volume currently shows."""
        gate = self.canvas.box_from_view()
        if gate is None:
            self._status.setText(
                "a box gate needs three measurements on screen; choose a Z")
            return
        self._on_gate_drawn(gate)

    def reset_view(self) -> None:
        """Undo a zoom and a spin in one place.

        One button for both because from the user's side there is one
        problem -- "the graph is not where it was" -- and having to know
        whether they zoomed or rotated to get out of it is the kind of
        distinction only the implementation cares about.
        """
        self.canvas.reset_view()

    def apply_settings(self, settings) -> None:
        """Take the settings that change how the gates surface draws.

        The canvas takes the drawing ones. Sampling is the screen's job --
        it owns the table and the read -- and the 3D ones belong to a
        workspace that does not exist yet. A setting silently read in two
        places is how the two get to disagree.

        The CLUSTERING ones are kept here rather than passed on, because the
        Cluster button is on this panel and used to ignore them entirely.
        """
        self._settings = settings
        self.canvas.apply_settings(settings)
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._frame is None:
            self._status.setText("no table loaded")
            return
        # It used to say "drawing on inside <gate>", which was true when
        # selecting a gate replotted its population. The plot always shows
        # the whole table now, so saying otherwise would be a lie about the
        # thing the user is looking at.
        active = self.tree.active_gate()
        population = self.canvas.population()
        n = 0 if population is None else len(population)
        parts = [f"{n:,} objects"]
        if len(self._gates):
            showing = len(self.canvas.enabled_gates)
            parts.append(f"{showing} of {len(self._gates)} gate(s) shown")
        if active:
            parts.append(f"next gate inside {active}")
        self._status.setText(" · ".join(parts))

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.canvas.close()
        super().closeEvent(event)
