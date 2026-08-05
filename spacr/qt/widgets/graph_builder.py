"""Graph Builder — drag a column onto a channel and the chart appears.

The direct-manipulation surface that replaces "which of the forty ``plot_*``
functions do I want, and what does it expect?". Six drop zones — x, y, colour,
size, facet-row, facet-column — a well of the columns worth plotting, and a
canvas that re-renders the moment a zone changes.

What is in here and what is not
-------------------------------

Everything about *what to draw* lives in :mod:`spacr.qt.widgets.graph_spec`:
the spec object, the plot-type inference, the facet grid, the shared scales
and the large-data policy. This module is the chrome and the matplotlib calls.
The split is not tidiness — small multiples, the gate editor, the feature
explorer and the campaign control charts are all "the graph builder with one
more rule", and they need the engine without inheriting a drag-and-drop panel.

Linked, and asymmetric on purpose
---------------------------------

:class:`GraphCanvas` mixes in
:class:`spacr.qt.linked_selection.LinkedView`, so it is one of the views that
talk to each other:

* a **brush** (drag a rectangle across a panel) publishes the rows it swept as
  the shared selection;
* an incoming **selection** rings those rows and dims the rest. It never
  removes a point — a selection highlights, it does not hide;
* an incoming **filter** does remove rows, and the axes re-scale to what is
  left, because a filter genuinely narrows the population.

The brush is evaluated as a *predicate over the frame*, not as a hit test
against drawn marks, which is what keeps it exact when a panel was drawn as a
density raster or from a sample.

Colour
------

Categorical series take a fixed eight-hue order — never cycled, never
re-assigned when a filter changes the series count, so a gene keeps its colour
between two charts. The order is the validated reference palette (light and
dark steps kept separately rather than flipped), and a continuous colour
column gets a single-hue light-to-dark ramp. Beyond eight levels the extras
fold into one "other" grey rather than inventing hues nobody can tell apart.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QMimeData, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ...selection import Selection, object_keys
from ..linked_selection import LinkedView
from ..theme import (RADIUS, SPACING, active_palette, font_px,
                     make_transparent, paint_panel, register_widget_qss)
from .graph_spec import (
    BAR, BINNED, BOX, CHANNELS, COLOUR, EMPTY, FACET_COL, FACET_ROW, HEATMAP,
    HISTOGRAM, LINE, MISSING_LEVEL, PLOT_KINDS, SCATTER, SIZE, VIOLIN, X, Y,
    GraphSpec, RenderData, brush_mask, facet_grid, plottable_columns,
    prepare_data, scales_for, value_axes,
)
from .toggle import Toggle

LOG = logging.getLogger("spacr.qt.graph_builder")

__all__ = [
    "COLUMN_MIME", "CHANNEL_LABELS", "ColumnWell", "DropZone", "GraphCanvas",
    "GraphBuilderPanel", "categorical_colours", "sequential_colours",
]

#: The drag payload. Its own type rather than ``text/plain`` so a column
#: dragged out of the well cannot be dropped into an unrelated text field, and
#: a path dragged in from the file manager cannot be read as a column name.
COLUMN_MIME = "application/x-spacr-graph-column"

#: Drop-zone captions, and the order they are laid out in.
CHANNEL_LABELS = {
    X: "X",
    Y: "Y",
    COLOUR: "Colour",
    SIZE: "Size",
    FACET_ROW: "Facet ↓",
    FACET_COL: "Facet →",
}

#: One-line "what does this zone do" for the tooltips.
CHANNEL_HINTS = {
    X: "Horizontal axis. One continuous column alone draws a histogram.",
    Y: "Vertical axis. A categorical column here and a continuous one on X "
       "draws boxes.",
    COLOUR: "Hue. A categorical column takes the fixed series order; a "
            "continuous one takes a light-to-dark ramp.",
    SIZE: "Mark area. Point plots only — an aggregate ignores it.",
    FACET_ROW: "One row of panels per level, with shared axes.",
    FACET_COL: "One column of panels per level, with shared axes.",
}

#: The categorical series order, light-surface steps then dark-surface steps.
#: Assigned by position and never cycled: the ninth level is folded into
#: :data:`OTHER_COLOUR`, because a generated ninth hue is one nobody can
#: separate from the eight already there.
_SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767")

#: Where levels past the eighth go.
OTHER_COLOUR = "#898781"
OTHER_LABEL = "other"

#: Single-hue magnitude ramp, light → dark, for a continuous colour column and
#: for the density raster. One hue, never a rainbow.
_RAMP_LIGHT = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281")
_RAMP_DARK = ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4")

#: How much opacity an unselected mark keeps while a selection is live. Dim,
#: never hidden — the shape of what was *not* selected is half the answer.
DIMMED_ALPHA = 0.16

#: Redraws are coalesced this long, so dragging a spinbox costs one render.
DEBOUNCE_MS = 120


def _is_light_surface() -> bool:
    """Whether the active theme's chart surface is a light one.

    Derived from the surface colour rather than a theme name so a theme added
    later gets the right series steps without touching this file.
    """
    try:
        surface = active_palette()["surface"]
        text = str(surface).lstrip("#")[:6]
        r, g, b = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return False
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5


def categorical_colours() -> Tuple[str, ...]:
    """The fixed eight-hue series order for the active theme."""
    return _SERIES_LIGHT if _is_light_surface() else _SERIES_DARK


def sequential_colours() -> Tuple[str, ...]:
    """The single-hue magnitude ramp for the active theme, light → dark."""
    return _RAMP_LIGHT if _is_light_surface() else _RAMP_DARK


def _colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "spacr_graph_seq", list(sequential_colours()))


def _orientation(vertical: bool) -> dict:
    """``boxplot``/``violinplot`` orientation, spelled the way this matplotlib
    wants it.

    3.10 replaced ``vert=True`` with ``orientation="vertical"`` and warns on
    the old spelling. spaCR is installed against both, and a
    ``PendingDeprecationWarning`` per panel per render is noise that hides the
    warnings worth reading.
    """
    import matplotlib
    parts = matplotlib.__version__.split(".")
    try:
        modern = (int(parts[0]), int(parts[1])) >= (3, 10)
    except (IndexError, ValueError):  # pragma: no cover - odd version string
        modern = True
    if modern:
        return {"orientation": "vertical" if vertical else "horizontal"}
    return {"vert": bool(vertical)}


# ---------------------------------------------------------------------------
# The well of columns, and the six zones
# ---------------------------------------------------------------------------

class ColumnWell(QWidget):
    """The list of plottable columns, filtered by a search box, draggable out.

    Only the columns :func:`spacr.qt.widgets.graph_spec.plottable_columns`
    offers — the same rule the Local Data Filter uses to decide what is worth
    a control. A measurement table has hundreds of columns and listing all of
    them is the same as listing none.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GraphColumnWell")
        self._columns: Tuple[str, ...] = ()
        self._kinds: Dict[str, str] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        self._search = QLineEdit(self)
        self._search.setObjectName("GraphColumnSearch")
        self._search.setPlaceholderText("Find a column…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refilter)
        outer.addWidget(self._search)

        self._list = _DraggableList(self)
        self._list.setObjectName("GraphColumnList")
        outer.addWidget(self._list, 1)

        self._count = QLabel("no table loaded", self)
        self._count.setObjectName("GraphColumnCount")
        outer.addWidget(self._count)

    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        if frame is None:
            self._columns = ()
            self._kinds = {}
        else:
            self._columns = plottable_columns(frame)
            from .graph_spec import column_kinds
            self._kinds = column_kinds(frame)
        self._refilter()

    def columns(self) -> Tuple[str, ...]:
        """Every offered column, whatever the search box currently shows."""
        return self._columns

    def visible_columns(self) -> List[str]:
        return [self._list.item(i).data(Qt.UserRole)
                for i in range(self._list.count())]

    def _refilter(self) -> None:
        needle = self._search.text().strip().lower()
        self._list.clear()
        for name in self._columns:
            if needle and needle not in name.lower():
                continue
            kind = self._kinds.get(name, "")
            item = QListWidgetItem(f"{name}   ·  {kind[:4]}")
            item.setData(Qt.UserRole, name)
            item.setToolTip(f"{name} — {kind}\nDrag onto a channel.")
            self._list.addItem(item)
        shown = self._list.count()
        total = len(self._columns)
        self._count.setText(
            "no table loaded" if not total
            else f"{shown} of {total} columns"
            if shown != total else f"{total} columns")


class _DraggableList(QListWidget):
    """A list whose items leave as :data:`COLUMN_MIME` payloads."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(False)

    def mimeData(self, items) -> QMimeData:  # noqa: N802 - Qt name
        payload = QMimeData()
        names = [i.data(Qt.UserRole) for i in items if i.data(Qt.UserRole)]
        if names:
            payload.setData(COLUMN_MIME, names[0].encode("utf-8"))
            # A plain-text copy as well, so dropping a column into a text
            # field elsewhere pastes its name rather than nothing.
            payload.setText(names[0])
        return payload


class DropZone(QFrame):
    """One channel's drop target.

    Emits :attr:`column_changed` with ``(channel, column_or_empty)``. The
    empty string rather than ``None`` so the signal can be typed ``str, str``
    and connected across a queued connection without a custom metatype.
    """

    column_changed = Signal(str, str)

    def __init__(self, channel: str, parent=None):
        super().__init__(parent)
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}")
        self.channel = channel
        self._column: Optional[str] = None
        self.setObjectName("GraphDropZone")
        self.setAcceptDrops(True)
        self.setProperty("filled", False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(CHANNEL_HINTS.get(channel, ""))

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING["sm"], SPACING["xs"],
                               SPACING["xs"], SPACING["xs"])
        row.setSpacing(SPACING["xs"])

        self._name = QLabel(CHANNEL_LABELS[channel], self)
        self._name.setObjectName("GraphDropZoneName")
        row.addWidget(self._name)

        self._value = QLabel("drop a column", self)
        self._value.setObjectName("GraphDropZoneValue")
        self._value.setWordWrap(False)
        row.addWidget(self._value, 1)

        self._clear = QPushButton("×", self)
        self._clear.setObjectName("GraphDropZoneClear")
        self._clear.setFixedWidth(20)
        self._clear.setToolTip(f"Take the column off {CHANNEL_LABELS[channel]}")
        self._clear.setVisible(False)
        self._clear.clicked.connect(lambda: self.set_column(None))
        row.addWidget(self._clear)

    # -- state ---------------------------------------------------------
    @property
    def column(self) -> Optional[str]:
        return self._column

    def set_column(self, column: Optional[str]) -> None:
        """Put ``column`` on this channel (``None`` empties it) and announce it.

        Silent when nothing changes: the panel rebuilds the chart on every
        emission, and a re-drop of the same column would otherwise cost a full
        re-render for no visible difference.
        """
        column = str(column) if column else None
        if column == self._column:
            return
        self._column = column
        self._value.setText(column or "drop a column")
        self._value.setToolTip(column or "")
        self._clear.setVisible(bool(column))
        self.setProperty("filled", bool(column))
        # Qt does not restyle on a property change by itself.
        self.style().unpolish(self)
        self.style().polish(self)
        self.column_changed.emit(self.channel, column or "")

    # -- drag and drop --------------------------------------------------
    def _accepts(self, event) -> bool:
        return event.mimeData() is not None and \
            event.mimeData().hasFormat(COLUMN_MIME)

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._accepts(event):
            self.setProperty("hovered", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):  # noqa: N802 - Qt name
        self.setProperty("hovered", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802 - Qt name
        if not self._accepts(event):
            event.ignore()
            return
        raw = bytes(event.mimeData().data(COLUMN_MIME)).decode("utf-8")
        self.setProperty("hovered", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.set_column(raw or None)
        event.acceptProposedAction()


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

def page_alpha() -> float:
    """The page-opacity preference as a plain float, for matplotlib.

    Matplotlib takes alpha as a number, not as a QSS colour, so
    :func:`~spacr.qt.theme.pane_surface` is no help to an axes patch.
    Degrades to the theme's designed scrim when preferences cannot be
    read, which is what a first run mid-generation gets.
    """
    from ..theme import panel_alpha
    theme = "dark"
    opacity = None
    try:
        from ..preferences import get_pane_opacity, resolve_effective_theme
        theme = resolve_effective_theme()
        opacity = get_pane_opacity()
    except Exception:
        pass
    return float(panel_alpha(theme, "surface_alt", opacity))


def _page_surface_axes(ax, palette) -> None:
    """Give ``ax`` a plotting area that follows the page-opacity slider.

    The axes keep a fill — the plotting area is meant to read as a panel
    within the panel — but at the page alpha, so the preference reaches
    the plot rather than stopping at its frame. ``set_facecolor`` with a
    raw hex would be opaque by construction and would hide the panel the
    canvas painted underneath.
    """
    ax.patch.set_facecolor(palette["surface_alt"])
    ax.patch.set_alpha(page_alpha())


#: Built once, on first use — see :func:`_canvas_class`.
_CANVAS_CLASS = None


def _canvas_class():
    """The figure canvas: a deferred draw it owns, on a page panel.

    Two problems, one class, because both need the same subclass and no
    module in this package may import matplotlib's Qt backend at import
    time.

    *The timer.* Matplotlib's Qt canvas schedules its idle draw with a
    static ``QTimer.singleShot``, which is not owned by the canvas and can
    therefore fire after Qt has deleted it — a segfault on close. An owned
    timer dies with the widget. The same fix
    :class:`spacr.qt.widgets.umap_explorer.ImageUmapExplorer` carries.

    *The slab.* ``FigureCanvasQT.__init__`` sets ``WA_OpaquePaintEvent``
    and the figure carries a solid ``facecolor``: two opaque things
    stacked, with square corners where every other container on the page
    is rounded. QSS reaches neither — a ``WA_OpaquePaintEvent`` widget
    never lets the sheet's background through — so the Graph Builder, the
    Trellis, the Gate Editor, Tabulate, PCA and the Feature Explorer all
    showed one flat rectangle whatever the page-opacity slider said. The
    panel is therefore drawn in ``paintEvent``, under a figure whose own
    patch is fully transparent, exactly as Training Runs does it.

    Cached, so the six screens share one class rather than one per canvas.
    """
    global _CANVAS_CLASS
    if _CANVAS_CLASS is not None:
        return _CANVAS_CLASS
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class OwnedTimerFigureCanvas(FigureCanvasQTAgg):

        def __init__(self, figure, *, panel: bool = True):
            """:param panel: draw the page surface under the figure.

            ``False`` for a canvas that is already sitting ON a panel — the
            scree plot inside the PCA shelf, say. Two surfaces stacked read
            0.49 at a requested 30 %, a shade no position of the slider can
            reach, so the inner one shows the outer panel through instead.
            """
            super().__init__(figure)
            self._spacr_draw_timer = QTimer(self)
            self._spacr_draw_timer.setSingleShot(True)
            self._spacr_draw_timer.timeout.connect(self._spacr_draw)
            self._spacr_panel = bool(panel)
            self.setAttribute(Qt.WA_OpaquePaintEvent, False)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            make_transparent(self)
            # Whatever is below is the surface now. Leaving the patch opaque
            # would paint the old rectangle straight back over it.
            figure.patch.set_alpha(0.0)

        def paintEvent(self, event):  # noqa: N802 - Qt name
            """Draw the page panel, then let matplotlib draw over it."""
            if self._spacr_panel:
                painter = QPainter(self)
                paint_panel(painter, self, role="surface", inset=0.5)
                painter.end()
            super().paintEvent(event)

        def draw_idle(self):
            self._draw_pending = True
            if not self._spacr_draw_timer.isActive():
                self._spacr_draw_timer.start(0)

        def _spacr_draw(self):
            if not self._draw_pending:
                return
            self._draw_pending = False
            try:
                self.draw()
            except RuntimeError:
                return

        def cancel_pending_draw(self):
            self._spacr_draw_timer.stop()
            self._draw_pending = False

    _CANVAS_CLASS = OwnedTimerFigureCanvas
    return OwnedTimerFigureCanvas


class GraphCanvas(LinkedView, QWidget):
    """The chart itself: a spec in, a faceted figure out, brushing back.

    Linked to the shared selection as ``source`` (``"graph_builder"`` by
    default). Pass ``link`` a private
    :class:`~spacr.qt.linked_selection.LinkedSelection` in tests so a run does
    not narrow every other open view.
    """

    #: Emitted after every render with the :class:`RenderData` that was drawn,
    #: so a host can put the large-data notice in its own status bar.
    rendered = Signal(object)

    def __init__(self, parent=None, *, link=None, source: str = "graph_builder"):
        super().__init__(parent)
        self.setObjectName("GraphCanvas")
        self._frame: Optional[pd.DataFrame] = None
        self._spec = GraphSpec()
        self._kinds: Dict[str, str] = {}
        self._keyed = False
        self._filter_note = ""

        # What the last render produced — kept so brushing, highlighting and
        # the tests can ask what is on screen without re-deriving it.
        self._visible: Optional[pd.DataFrame] = None
        self._render_data: Optional[RenderData] = None
        self._grid = None
        self._brush_grid = None
        self._scales = None
        self._axes: Dict[Tuple[int, int], object] = {}
        self._axes_at: Dict[int, Tuple[int, int]] = {}
        self._overlays: Dict[Tuple[int, int], Optional[Callable]] = {}
        #: Whether the drawn kind can move its highlight without a redraw.
        self._live_highlight = False
        self._selected_mask: Optional[np.ndarray] = None
        self._drag_origin: Optional[Tuple[object, float, float]] = None
        self._drag_patch = None

        self._build_ui()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.render_now)
        self.link_selection(source, link=link)

    def _build_ui(self) -> None:
        from matplotlib.figure import Figure

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        # No `facecolor` and no inline `background:` — the canvas paints the
        # page panel in its own `paintEvent` and its figure patch is
        # transparent, so either of those would put an opaque rectangle back
        # over it and stop the page-opacity slider reaching the chart.
        self._figure = Figure(figsize=(7.5, 5.0))
        self._canvas = _canvas_class()(self._figure)
        outer.addWidget(self._canvas, 1)

        self._notice = QLabel("", self)
        self._notice.setObjectName("GraphNotice")
        self._notice.setWordWrap(True)
        outer.addWidget(self._notice)

        for event, slot in (("button_press_event", self._on_press),
                            ("motion_notify_event", self._on_motion),
                            ("button_release_event", self._on_release)):
            self._canvas.mpl_connect(event, slot)

    # -- inputs ---------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Point the canvas at a table.

        Channels naming a column the new table does not have are emptied
        rather than carried over: a spec that half-resolves would draw a chart
        of fewer variables than the zones claim.
        """
        self._frame = frame
        self._kinds = self._spec.kinds_for(frame) if frame is not None else {}
        self._keyed = False
        if frame is not None:
            try:
                object_keys(frame)
                self._keyed = True
            except Exception:
                # No object key columns: the chart still draws, it just cannot
                # join the linked selection. Said out loud in the notice
                # rather than silently publishing nothing.
                self._keyed = False
        if frame is not None:
            spec = self._spec
            for channel in CHANNELS:
                column = spec.column_for(channel)
                if column and column not in frame.columns:
                    spec = spec.with_channel(channel, None)
            self._spec = spec
        self.render_now()

    @property
    def spec(self) -> GraphSpec:
        return self._spec

    @property
    def kinds(self) -> Dict[str, str]:
        """The loaded table's column kinds, with the spec's role overrides.

        Public because "is this column continuous here?" is the question every
        caller of :meth:`spec` asks next, and re-deriving it would risk two
        answers.
        """
        return dict(self._kinds)

    def set_spec(self, spec: GraphSpec, *, immediate: bool = True) -> None:
        """Replace the spec and redraw."""
        self._spec = spec
        if self._frame is not None:
            self._kinds = spec.kinds_for(self._frame)
        if immediate:
            self.render_now()
        else:
            self._debounce.start()

    def set_channel(self, channel: str, column: Optional[str]) -> None:
        self.set_spec(self._spec.with_channel(channel, column))

    # -- what the last render produced -----------------------------------
    @property
    def render_data(self) -> Optional[RenderData]:
        return self._render_data

    @property
    def grid(self):
        return self._grid

    @property
    def scales(self):
        return self._scales

    def panel_axes(self) -> Dict[Tuple[int, int], object]:
        """``{(row, col): Axes}`` for every panel, empty ones included."""
        return dict(self._axes)

    def axes_at(self, row: int = 0, col: int = 0):
        return self._axes.get((row, col))

    def notice(self) -> str:
        """The line under the chart: what was drawn, and out of how much."""
        return self._notice.text()

    def selected_count(self) -> int:
        """Rows of the drawn frame the shared selection names."""
        if self._selected_mask is None:
            return 0
        return int(self._selected_mask.sum())

    # -- rendering --------------------------------------------------------
    def render_now(self) -> None:
        """Rebuild the figure from the current frame, spec, filter and selection."""
        self._debounce.stop()
        self._figure.clear()
        # `clear()` restores the rc facecolor AND its alpha, so the
        # transparency the canvas set has to be re-asserted or the first
        # redraw paints the opaque rectangle straight back over the panel.
        self._figure.patch.set_alpha(0.0)
        self._axes = {}
        self._axes_at = {}
        self._overlays = {}
        self._drag_patch = None
        palette = active_palette()

        if self._frame is None or self._frame.empty:
            self._render_message(
                "Load a table, then drag a column onto X or Y.")
            return

        self._visible, self._filter_note = self._apply_filter(self._frame)
        spec = self._spec
        kinds = self._kinds
        kind = spec.resolved_kind(kinds)
        if kind == EMPTY:
            self._render_message(
                "Drag a column onto X or Y.\n"
                "One continuous column draws a histogram; two draw a scatter; "
                "one of each draws boxes.")
            return

        data = prepare_data(self._visible, spec, kinds)
        grid = facet_grid(data.frame, spec, levels_source=self._visible)
        # A second grid over the *unsampled* rows, so a brush selects every
        # row inside the rectangle rather than only the ones drawn.
        self._brush_grid = (grid if data.frame is self._visible
                            else facet_grid(self._visible, spec,
                                            levels_source=self._visible))
        scales = scales_for(data.frame, spec, kinds, grid)
        self._render_data = data
        self._grid = grid
        self._scales = scales
        self._live_highlight = (kind == SCATTER and data.strategy != BINNED)
        self._selected_mask = self._selection_mask(data.frame)

        nrows, ncols = grid.shape
        axes = self._figure.subplots(
            nrows, ncols, squeeze=False,
            sharex=bool(spec.shared_x), sharey=bool(spec.shared_y))
        for panel in grid.panels:
            ax = axes[panel.row][panel.col]
            self._axes[(panel.row, panel.col)] = ax
            self._axes_at[id(ax)] = (panel.row, panel.col)
            self._style_axes(ax, palette)
            rows = panel.frame(data.frame)
            mask = (self._selected_mask[panel.index]
                    if self._selected_mask is not None else None)
            overlay = self._draw_panel(ax, rows, mask, kind, data, palette)
            self._overlays[(panel.row, panel.col)] = overlay
            self._apply_scales(ax, kind, scales, panel)
            self._label_panel(ax, panel, grid, nrows, ncols, palette)

        self._draw_legend(kind, palette)
        self._figure.tight_layout(pad=0.8)
        self._canvas.draw_idle()
        self._notice.setText(self._notice_text(data, grid))
        self.rendered.emit(data)

    def _render_message(self, text: str) -> None:
        palette = active_palette()
        ax = self._figure.add_subplot(111)
        _page_surface_axes(ax, palette)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                color=palette["fg_muted"], fontsize=10,
                transform=ax.transAxes)
        self._render_data = None
        self._grid = None
        self._scales = None
        self._selected_mask = None
        self._canvas.draw_idle()
        self._notice.setText("")

    def _apply_filter(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """``frame`` narrowed by the shared filter, plus a note if it could not be.

        A filter naming a column this table does not have is reported rather
        than swallowed: the alternative is a chart of more rows than the
        filter panel says are in the population.
        """
        try:
            return self.linked_visible(frame), ""
        except Exception as exc:
            LOG.info("the shared filter does not apply to this table: %s", exc)
            return frame, f" · the shared filter does not apply here ({exc})"

    def _selection_mask(self, frame: pd.DataFrame) -> Optional[np.ndarray]:
        """Which drawn rows the shared selection names, or ``None`` at rest.

        ``None`` and "an all-False mask" are different: the first is nobody
        having selected anything, the second is a brush that caught nothing.
        Only the second dims the rest of the chart.
        """
        selection = self.link.selection
        if not selection.is_active or not self._keyed or frame.empty:
            return None
        try:
            return selection.mask_for(frame)
        except Exception:
            LOG.debug("could not resolve the shared selection here",
                      exc_info=True)
            return None

    # -- drawing ----------------------------------------------------------
    def _style_axes(self, ax, palette) -> None:
        """Recessive chrome: hairline grid, two spines, muted ticks."""
        _page_surface_axes(ax, palette)
        ax.grid(True, color=palette["border_soft"], linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(palette["border"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=palette["fg_muted"], labelsize=8, length=3)

    def _series_colour(self, index: int) -> str:
        order = categorical_colours()
        return order[index] if index < len(order) else OTHER_COLOUR

    def _level_colours(self, values: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Per-row hue for a categorical colour column, by fixed level order.

        The level's position in :attr:`Scales.colour_levels` picks the hue, not
        its rank in this panel — so a filter that removes a gene does not
        repaint the genes that survive.
        """
        levels = list(self._scales.colour_levels or ())
        index = {level: i for i, level in enumerate(levels)}
        text = values.astype(str)
        colours = np.array(
            [self._series_colour(index.get(v, len(levels))) for v in text],
            dtype=object)
        return colours, levels

    def _sizes(self, rows: pd.DataFrame) -> np.ndarray:
        spec = self._spec
        base = np.full(len(rows), 16.0)
        limits = getattr(self._scales, "size_limits", None)
        if not spec.size or spec.size not in rows.columns or not limits:
            return base
        values = pd.to_numeric(rows[spec.size], errors="coerce").to_numpy(float)
        low, high = limits
        span = (high - low) or 1.0
        scaled = np.clip((values - low) / span, 0.0, 1.0)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        return 10.0 + scaled * 130.0

    def _draw_panel(self, ax, rows, mask, kind, data, palette
                    ) -> Optional[Callable]:
        """Draw one panel; return an updater for a cheap highlight repaint.

        The updater exists only for point marks, where a selection change is
        a change of two artists. Aggregates redraw — their overlay is a
        recomputed reduction, not a re-styled artist.
        """
        if rows.empty:
            ax.text(0.5, 0.5, "no rows", ha="center", va="center",
                    color=palette["fg_muted"], fontsize=8,
                    transform=ax.transAxes)
            return None
        if kind == SCATTER and data.strategy == BINNED:
            self._draw_density(ax, rows, palette)
            return None
        if kind in (SCATTER, LINE):
            return self._draw_points(ax, rows, mask, kind, palette)
        if kind == HISTOGRAM:
            self._draw_histogram(ax, rows, mask, palette)
            return None
        if kind == BAR:
            self._draw_bar(ax, rows, mask, palette)
            return None
        if kind in (BOX, VIOLIN):
            self._draw_distribution(ax, rows, kind, palette)
            return None
        if kind == HEATMAP:
            self._draw_heatmap(ax, rows, palette)
            return None
        return None  # pragma: no cover - every kind is handled above

    def _xy(self, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        spec, scales = self._spec, self._scales
        def axis(column, levels):
            if not column or column not in rows.columns:
                return np.zeros(len(rows))
            if levels:
                position = {level: i for i, level in enumerate(levels)}
                return np.array([position.get(str(v), np.nan)
                                 for v in rows[column].astype(str)], dtype=float)
            return pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
        return (axis(spec.x, scales.x_levels), axis(spec.y, scales.y_levels))

    def _draw_points(self, ax, rows, mask, kind, palette) -> Callable:
        spec = self._spec
        x, y = self._xy(rows)
        if kind == LINE:
            order = np.argsort(x, kind="stable")
            ax.plot(x[order], y[order], color=self._series_colour(0),
                    linewidth=2.0, solid_capstyle="round")
            return lambda _m: None
        if spec.colour and self._scales.colour_levels:
            colours, _levels = self._level_colours(rows[spec.colour])
            base = ax.scatter(x, y, s=self._sizes(rows), c=list(colours),
                              linewidths=0.0, alpha=0.75)
        elif spec.colour and self._scales.colour_limits:
            values = pd.to_numeric(rows[spec.colour],
                                   errors="coerce").to_numpy(float)
            low, high = self._scales.colour_limits
            base = ax.scatter(x, y, s=self._sizes(rows), c=values,
                              cmap=_colormap(), vmin=low, vmax=high,
                              linewidths=0.0, alpha=0.8)
        else:
            base = ax.scatter(x, y, s=self._sizes(rows),
                              color=self._series_colour(0),
                              linewidths=0.0, alpha=0.7)
        ring = ax.scatter([], [], s=54, facecolors="none",
                          edgecolors=palette["fg"], linewidths=1.4, zorder=5)

        def update(new_mask) -> None:
            if new_mask is None:
                base.set_alpha(0.7)
                ring.set_offsets(np.empty((0, 2)))
                return
            base.set_alpha(DIMMED_ALPHA)
            picked = np.column_stack([x[new_mask], y[new_mask]]) \
                if new_mask.any() else np.empty((0, 2))
            ring.set_offsets(picked)

        update(mask)
        return update

    def _draw_density(self, ax, rows, palette) -> None:
        """A 2-D histogram raster: every row counted, none drawn twice."""
        spec, scales = self._spec, self._scales
        x = pd.to_numeric(rows[spec.x], errors="coerce").to_numpy(float)
        y = pd.to_numeric(rows[spec.y], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        edges_x = scales.x_edges if scales.x_edges is not None else spec.bins
        edges_y = scales.y_edges if scales.y_edges is not None else spec.bins
        counts, ex, ey = np.histogram2d(x, y, bins=[edges_x, edges_y])
        weighted = None
        if spec.colour and scales.colour_limits:
            values = pd.to_numeric(rows[spec.colour],
                                   errors="coerce").to_numpy(float)[finite]
            total, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y],
                                         weights=np.nan_to_num(values))
            with np.errstate(invalid="ignore", divide="ignore"):
                weighted = np.where(counts > 0, total / counts, np.nan)
        image = weighted if weighted is not None else np.where(counts > 0,
                                                               counts, np.nan)
        ax.imshow(image.T, origin="lower", aspect="auto", cmap=_colormap(),
                  extent=(ex[0], ex[-1], ey[0], ey[-1]),
                  interpolation="nearest")

    def _draw_histogram(self, ax, rows, mask, palette) -> None:
        spec, scales = self._spec, self._scales
        column = spec.x or spec.y
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        edges = scales.x_edges if scales.x_edges is not None else spec.bins
        if spec.colour and scales.colour_levels:
            colours, levels = self._level_colours(rows[spec.colour])
            bottom = None
            for i, level in enumerate(levels):
                pick = finite & (colours == self._series_colour(i))
                counts, ex = np.histogram(values[pick], bins=edges)
                centres = (ex[:-1] + ex[1:]) / 2.0
                width = np.diff(ex) * 0.92
                ax.bar(centres, counts, width=width, bottom=bottom,
                       color=self._series_colour(i), linewidth=0.0,
                       label=level)
                bottom = counts if bottom is None else bottom + counts
        else:
            counts, ex = np.histogram(values[finite], bins=edges)
            centres = (ex[:-1] + ex[1:]) / 2.0
            ax.bar(centres, counts, width=np.diff(ex) * 0.92,
                   color=self._series_colour(0), linewidth=0.0)
        if mask is not None and mask.any():
            counts, ex = np.histogram(values[finite & mask], bins=edges)
            centres = (ex[:-1] + ex[1:]) / 2.0
            ax.bar(centres, counts, width=np.diff(ex) * 0.5,
                   color=palette["fg"], alpha=0.85, linewidth=0.0,
                   label="selected")

    def _draw_bar(self, ax, rows, mask, palette) -> None:
        spec, scales = self._spec, self._scales
        column = spec.x or spec.y
        levels = list(scales.x_levels or scales.y_levels or ())
        text = rows[column].astype(str).mask(rows[column].isna(), MISSING_LEVEL)
        counts = text.value_counts()
        heights = [float(counts.get(level, 0)) for level in levels]
        ax.bar(range(len(levels)), heights, width=0.78,
               color=self._series_colour(0), linewidth=0.0)
        if mask is not None and mask.any():
            picked = text[mask].value_counts()
            ax.bar(range(len(levels)),
                   [float(picked.get(level, 0)) for level in levels],
                   width=0.42, color=palette["fg"], alpha=0.85, linewidth=0.0,
                   label="selected")

    def _draw_distribution(self, ax, rows, kind, palette) -> None:
        spec, scales = self._spec, self._scales
        categorical_on_x = bool(scales.x_levels)
        cat_column = spec.x if categorical_on_x else spec.y
        num_column = spec.y if categorical_on_x else spec.x
        levels = list((scales.x_levels if categorical_on_x
                       else scales.y_levels) or ())
        text = rows[cat_column].astype(str).mask(rows[cat_column].isna(),
                                                 MISSING_LEVEL)
        values = pd.to_numeric(rows[num_column], errors="coerce")
        groups, positions = [], []
        for i, level in enumerate(levels):
            picked = values[(text == level).to_numpy()].dropna().to_numpy(float)
            if picked.size:
                groups.append(picked)
                positions.append(i)
        if not groups:
            return
        if kind == VIOLIN:
            parts = ax.violinplot(groups, positions=positions,
                                  showmedians=True, widths=0.7,
                                  **_orientation(categorical_on_x))
            for body in parts["bodies"]:
                body.set_facecolor(self._series_colour(0))
                body.set_alpha(0.55)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_color(palette["fg_muted"])
        else:
            drawn = ax.boxplot(
                groups, positions=positions, widths=0.6, patch_artist=True,
                manage_ticks=False, **_orientation(categorical_on_x),
                medianprops={"color": palette["fg"], "linewidth": 1.4},
                flierprops={"marker": ".", "markersize": 2.5,
                            "markerfacecolor": palette["fg_muted"],
                            "markeredgecolor": "none", "alpha": 0.5})
            for box in drawn["boxes"]:
                box.set_facecolor(self._series_colour(0))
                box.set_alpha(0.7)
                box.set_linewidth(0.0)
            for key in ("whiskers", "caps"):
                for line in drawn[key]:
                    line.set_color(palette["fg_muted"])
                    line.set_linewidth(0.9)

    def _draw_heatmap(self, ax, rows, palette) -> None:
        spec, scales = self._spec, self._scales
        x_levels = list(scales.x_levels or ())
        y_levels = list(scales.y_levels or ())
        table = pd.crosstab(
            rows[spec.y].astype(str).mask(rows[spec.y].isna(), MISSING_LEVEL),
            rows[spec.x].astype(str).mask(rows[spec.x].isna(), MISSING_LEVEL))
        table = table.reindex(index=y_levels, columns=x_levels, fill_value=0)
        counts = table.to_numpy(dtype=float)
        ax.imshow(np.where(counts > 0, counts, np.nan), origin="lower",
                  aspect="auto", cmap=_colormap(),
                  extent=(-0.5, len(x_levels) - 0.5,
                          -0.5, len(y_levels) - 0.5),
                  interpolation="nearest")

    # -- axes -------------------------------------------------------------
    def _apply_scales(self, ax, kind, scales, panel) -> None:
        """Give every panel the *same* limits, ticks and orders.

        Set explicitly from :func:`~spacr.qt.widgets.graph_spec.scales_for`
        rather than left to matplotlib's ``sharex``: sharing makes the panels
        agree with *each other*, but they agree on whatever the first panel
        happened to autoscale to, which is not necessarily wide enough for the
        rest. Computing the limits over the whole frame is what makes them
        bound every panel.
        """
        spec = self._spec
        counts_on_y = kind in (HISTOGRAM, BAR)
        if scales.x_levels is not None:
            ax.set_xticks(range(len(scales.x_levels)))
            ax.set_xticklabels(scales.x_levels, rotation=30, ha="right",
                               fontsize=7)
            if spec.shared_x:
                ax.set_xlim(-0.6, len(scales.x_levels) - 0.4)
        elif spec.shared_x and scales.x_limits is not None:
            ax.set_xlim(*scales.x_limits)
        if counts_on_y:
            if spec.shared_y and scales.count_limit:
                ax.set_ylim(0, scales.count_limit)
        elif scales.y_levels is not None:
            ax.set_yticks(range(len(scales.y_levels)))
            ax.set_yticklabels(scales.y_levels, fontsize=7)
            if spec.shared_y:
                ax.set_ylim(-0.6, len(scales.y_levels) - 0.4)
        elif spec.shared_y and scales.y_limits is not None:
            ax.set_ylim(*scales.y_limits)

    def _label_panel(self, ax, panel, grid, nrows, ncols, palette) -> None:
        """Axis names on the outside edges only — the shared-axis convention.

        The value column is named on x and the count on y for a histogram or
        bar chart whichever zone it was dropped in, which is the same
        indirection :func:`~spacr.qt.widgets.graph_spec.value_axes` applies to
        the scales. Labelling from ``spec.x``/``spec.y`` directly would put
        the name on the axis the data is not on.
        """
        spec = self._spec
        kind = spec.resolved_kind(self._kinds)
        x_column, y_column = value_axes(spec, self._kinds)
        counts_on_y = kind in (HISTOGRAM, BAR)
        if panel.row == nrows - 1:
            ax.set_xlabel(x_column or "", color=palette["fg_dim"], fontsize=9)
        if panel.col == 0:
            ax.set_ylabel("count" if counts_on_y else (y_column or ""),
                          color=palette["fg_dim"], fontsize=9)
        if grid.is_faceted:
            title = panel.title()
            if title:
                ax.set_title(f"{title}  ·  n={panel.n:,}",
                             color=palette["fg_dim"], fontsize=8, pad=3)
        elif panel.n:
            ax.set_title(f"n={panel.n:,}", color=palette["fg_muted"],
                         fontsize=8, pad=3, loc="right")

    def _draw_legend(self, kind, palette) -> None:
        """A legend whenever there are two or more series — identity is never
        colour alone."""
        levels = getattr(self._scales, "colour_levels", None)
        if not levels or len(levels) < 2:
            return
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=self._series_colour(i),
                          markeredgecolor="none", label=str(level))
                   for i, level in enumerate(levels[:len(categorical_colours())])]
        if len(levels) > len(categorical_colours()):
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markersize=6, markerfacecolor=OTHER_COLOUR,
                                  markeredgecolor="none",
                                  label=f"{OTHER_LABEL} "
                                        f"({len(levels) - len(categorical_colours())})"))
        legend = self._figure.legend(
            handles=handles, loc="upper right", frameon=False, fontsize=8,
            title=self._spec.colour, title_fontsize=8)
        for text in legend.get_texts():
            text.set_color(palette["fg_dim"])
        if legend.get_title() is not None:
            legend.get_title().set_color(palette["fg_muted"])

    def _notice_text(self, data: RenderData, grid) -> str:
        parts = [data.notice]
        if grid.notice:
            parts.append(grid.notice)
        if not self._keyed:
            parts.append("no object keys in this table — brushing cannot "
                         "publish a selection")
        if self._filter_note:
            parts.append(self._filter_note.strip(" ·"))
        if self._selected_mask is not None:
            parts.append(f"{int(self._selected_mask.sum()):,} highlighted")
        return " · ".join(p for p in parts if p)

    # -- linked selection -------------------------------------------------
    def on_linked_filter_changed(self, data_filter) -> None:
        """A filter genuinely narrows the population: redraw and re-scale."""
        self._debounce.start()

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """A selection only highlights — never a row fewer on screen."""
        if self._render_data is None:
            return
        if not self._live_highlight:
            # An aggregate's highlight is a recomputed reduction, not a
            # re-styled artist, so it costs a redraw. Point marks do not:
            # two artists move, which is what keeps a lasso in another view
            # from re-rendering fifty thousand marks here.
            self.render_now()
            return
        self._selected_mask = self._selection_mask(self._render_data.frame)
        for (row, col), overlay in self._overlays.items():
            if overlay is None:
                continue
            panel = self._grid.panel(row, col)
            mask = (self._selected_mask[panel.index]
                    if self._selected_mask is not None else None)
            overlay(mask)
        self._canvas.draw_idle()
        self._notice.setText(self._notice_text(self._render_data, self._grid))

    # -- brushing ---------------------------------------------------------
    def brush(self, x0: float, y0: float, x1: float, y1: float, *,
              row: int = 0, col: int = 0,
              publish: bool = True) -> Optional[Selection]:
        """Select every row of one panel inside the rectangle, and publish it.

        Evaluated against the panel's **unsampled** rows, so a brush over a
        density raster or a sampled scatter still names every row in the
        rectangle rather than only the ones that got drawn.

        :returns: the published :class:`~spacr.selection.Selection`, or
            ``None`` when this table carries no object keys to name rows with.
        """
        if (self._visible is None or self._brush_grid is None
                or not self._keyed):
            return None
        try:
            panel = self._brush_grid.panel(row, col)
        except IndexError:
            return None
        rows = panel.frame(self._visible)
        keep = brush_mask(rows, self._spec, self._kinds,
                          x0, y0, x1, y1, self._scales)
        picked = rows.loc[keep]
        if not publish:
            return Selection.from_frame(picked, source=self.link_source)
        return self.publish_selection(picked)

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        self._drag_origin = (event.inaxes, float(event.xdata),
                             float(event.ydata))

    def _on_motion(self, event) -> None:
        if self._drag_origin is None or event.inaxes is not self._drag_origin[0]:
            return
        if event.xdata is None or event.ydata is None:
            return
        from matplotlib.patches import Rectangle
        ax, x0, y0 = self._drag_origin
        if self._drag_patch is None:
            palette = active_palette()
            self._drag_patch = Rectangle(
                (x0, y0), 0, 0, facecolor=palette["accent"], alpha=0.18,
                edgecolor=palette["accent"], linewidth=1.0, zorder=6)
            ax.add_patch(self._drag_patch)
        self._drag_patch.set_bounds(
            min(x0, event.xdata), min(y0, event.ydata),
            abs(event.xdata - x0), abs(event.ydata - y0))
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        origin, self._drag_origin = self._drag_origin, None
        if self._drag_patch is not None:
            try:
                self._drag_patch.remove()
            except (ValueError, NotImplementedError):
                pass
            self._drag_patch = None
        if origin is None or event.inaxes is not origin[0]:
            return
        ax, x0, y0 = origin
        if event.xdata is None or event.ydata is None:
            return
        where = self._axes_at.get(id(ax))
        if where is None:
            return
        span_x = abs(float(event.xdata) - x0)
        span_y = abs(float(event.ydata) - y0)
        width = abs(np.diff(ax.get_xlim())[0]) or 1.0
        height = abs(np.diff(ax.get_ylim())[0]) or 1.0
        if span_x < width * 0.01 and span_y < height * 0.01:
            # A click, not a drag: back to the resting state, which is a
            # different thing from an empty selection.
            self.clear_linked_selection()
            return
        self.brush(x0, y0, float(event.xdata), float(event.ydata),
                   row=where[0], col=where[1])

    # -- teardown ----------------------------------------------------------
    def closeEvent(self, event):  # noqa: N802 - Qt name
        try:
            self.unlink_selection()
        except (RuntimeError, TypeError):
            pass
        self._debounce.stop()
        canvas = getattr(self, "_canvas", None)
        if canvas is not None and hasattr(canvas, "cancel_pending_draw"):
            canvas.cancel_pending_draw()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# The whole surface
# ---------------------------------------------------------------------------

class GraphBuilderPanel(QWidget):
    """The well, the six zones, the plot-type override and the canvas."""

    spec_changed = Signal(object)

    def __init__(self, parent=None, *, link=None,
                 source: str = "graph_builder"):
        super().__init__(parent)
        self.setObjectName("GraphBuilderPanel")
        self._zones: Dict[str, DropZone] = {}
        self._building = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        shelf = QWidget(self)
        shelf.setObjectName("GraphShelf")
        shelf_layout = QVBoxLayout(shelf)
        shelf_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                        SPACING["sm"], SPACING["sm"])
        shelf_layout.setSpacing(SPACING["sm"])

        self.well = ColumnWell(shelf)
        shelf_layout.addWidget(self.well, 1)

        zones = QGridLayout()
        zones.setContentsMargins(0, 0, 0, 0)
        zones.setSpacing(SPACING["xs"])
        for i, channel in enumerate(CHANNELS):
            zone = DropZone(channel, shelf)
            zone.column_changed.connect(self._on_zone_changed)
            self._zones[channel] = zone
            zones.addWidget(zone, i // 2, i % 2)
        shelf_layout.addLayout(zones)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING["xs"])
        self._kind = QComboBox(shelf)
        self._kind.setObjectName("GraphKindPicker")
        self._kind.addItem("Automatic", "")
        for kind in PLOT_KINDS:
            if kind != EMPTY:
                self._kind.addItem(kind.capitalize(), kind)
        self._kind.setToolTip(
            "The plot type is inferred from the columns dropped. Pick one "
            "here to pin it instead.")
        self._kind.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("Plot", shelf), 0, 0)
        controls.addWidget(self._kind, 0, 1)

        self._bins = QSpinBox(shelf)
        self._bins.setRange(2, 200)
        self._bins.setValue(30)
        self._bins.setToolTip("Bins per axis, for histograms and the density "
                              "raster.")
        self._bins.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("Bins", shelf), 1, 0)
        controls.addWidget(self._bins, 1, 1)

        self._shared_x = Toggle("Shared X", shelf)
        self._shared_y = Toggle("Shared Y", shelf)
        for box in (self._shared_x, self._shared_y):
            box.setChecked(True)
            box.setToolTip("Every panel on the same scale. Off makes panels "
                           "incomparable — which is occasionally what you want.")
            box.toggled.connect(self._on_controls_changed)
        controls.addWidget(self._shared_x, 2, 0)
        controls.addWidget(self._shared_y, 2, 1)
        shelf_layout.addLayout(controls)

        self._clear = QPushButton("Clear all channels", shelf)
        self._clear.setObjectName("GraphClearButton")
        self._clear.clicked.connect(self.clear_channels)
        shelf_layout.addWidget(self._clear)

        splitter.addWidget(shelf)

        self.canvas = GraphCanvas(self, link=link, source=source)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

    # -- data -----------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        self.well.set_frame(frame)
        self.canvas.set_frame(frame)
        self._sync_zones()

    @property
    def spec(self) -> GraphSpec:
        return self.canvas.spec

    def set_spec(self, spec: GraphSpec) -> None:
        """Push a whole spec in — restoring a saved chart, or a preset."""
        self.canvas.set_spec(spec)
        self._sync_zones()

    def clear_channels(self) -> None:
        for zone in self._zones.values():
            zone.set_column(None)

    def zone(self, channel: str) -> DropZone:
        return self._zones[channel]

    # -- wiring ----------------------------------------------------------
    def _sync_zones(self) -> None:
        """Make the zones and the controls show what the spec actually says."""
        self._building = True
        try:
            spec = self.canvas.spec
            for channel, zone in self._zones.items():
                zone.set_column(spec.column_for(channel))
            index = self._kind.findData(spec.kind or "")
            if index >= 0:
                self._kind.setCurrentIndex(index)
            self._bins.setValue(spec.bins)
            self._shared_x.setChecked(spec.shared_x)
            self._shared_y.setChecked(spec.shared_y)
        finally:
            self._building = False

    def _on_zone_changed(self, channel: str, column: str) -> None:
        if self._building:
            return
        self.canvas.set_channel(channel, column or None)
        self.spec_changed.emit(self.canvas.spec)

    def _on_controls_changed(self, *_args) -> None:
        if self._building:
            return
        from dataclasses import replace as _replace
        spec = _replace(
            self.canvas.spec,
            kind=(self._kind.currentData() or None),
            bins=self._bins.value(),
            shared_x=self._shared_x.isChecked(),
            shared_y=self._shared_y.isChecked())
        self.canvas.set_spec(spec, immediate=False)
        self.spec_changed.emit(spec)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.canvas.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Styling, through the seam rather than by editing theme.py
# ---------------------------------------------------------------------------

def _graph_builder_qss(palette, opacity) -> str:
    from ..theme import block_surface
    surface_alt = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QWidget#GraphShelf, QWidget#TrellisShelf {{
    background: {surface_alt};
    border-radius: {RADIUS["md"]}px;
}}
/* Scaffolding, whatever it is called: the panels hold a splitter edge to
 * edge and the canvas below paints the page surface. Without this they
 * take the blanket `QWidget` rule, which is the WINDOW colour and not a
 * surface, so no page-opacity setting could ever reach them. */
QWidget#GraphBuilderPanel, QWidget#TrellisPanel, QWidget#GraphCanvas,
QWidget#PCAPanel, QWidget#GateEditorPanel, QWidget#FeatureExplorerPanel {{
    background: transparent;
}}
QFrame#GraphDropZone {{
    background: transparent;
    border: 1px dashed {palette["border"]};
    border-radius: {RADIUS["sm"]}px;
    min-height: 34px;
}}
QFrame#GraphDropZone[filled="true"] {{
    border: 1px solid {palette["accent"]};
    background: {palette["accent_soft"]};
}}
QFrame#GraphDropZone[hovered="true"] {{
    border: 1px solid {palette["accent_hi"]};
}}
QLabel#GraphDropZoneName {{
    color: {palette["fg_muted"]};
    font-weight: 600;
}}
QLabel#GraphDropZoneValue {{
    color: {palette["fg"]};
}}
QFrame#GraphDropZone[filled="false"] QLabel#GraphDropZoneValue {{
    color: {palette["fg_muted"]};
    font-style: italic;
}}
QPushButton#GraphDropZoneClear {{
    border: none;
    background: transparent;
    color: {palette["fg_muted"]};
}}
QLabel#GraphNotice, QLabel#GraphColumnCount {{
    color: {palette["fg_muted"]};
    font-size: {font_px(11)}px;
}}
QListWidget#GraphColumnList {{
    background: transparent;
    border: 1px solid {palette["border_soft"]};
    border-radius: {RADIUS["sm"]}px;
}}
"""


register_widget_qss("GraphBuilder", _graph_builder_qss, replace=True)
