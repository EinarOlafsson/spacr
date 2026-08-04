"""The trellis surface — the Graph Builder's canvas, laid out as a grid.

:class:`TrellisCanvas` **subclasses**
:class:`spacr.qt.widgets.graph_builder.GraphCanvas` rather than reimplementing
it. That is the whole design decision in this file. The mark drawing, the fixed
categorical hue order, the density raster, the selection ring, the legend and
the brush are one implementation, and a second one would be a second set of
rules to keep in step with the first — the mistake
:mod:`spacr.qt.widgets.pivot_builder` avoids by not drawing anything at all.

What the subclass changes is the two things a trellis is:

* **the layout** comes from :func:`spacr.qt.widgets.trellis_spec.trellis`,
  which knows about wrapping and about blank slots, rather than from
  ``facet_grid`` directly;
* **the scales are set per panel from that panel's own group**. Every panel's
  limits are written explicitly rather than left to matplotlib's ``sharex``:
  sharing makes panels agree with *each other*, but on whatever the first one
  autoscaled to, which is not necessarily wide enough for the rest. Under
  :data:`~spacr.qt.widgets.trellis_spec.SCALE_SHARED` — the default — every
  panel is handed the identical tuple, which is a property a test can assert
  and an eye can trust.

Two conventions the grid earns
------------------------------

**Inner tick labels are hidden only when the axis is genuinely shared.** A
trellis with per-panel scales that hides its inner ticks is a lie with a
tidy layout; if a panel has its own limits it prints its own numbers.

**Every panel's title carries its n**, from
:meth:`~spacr.qt.widgets.trellis_spec.TrellisPanel.label`. Blank slots at the
end of a wrapped grid are hidden entirely — they are not panels with no data,
they are the remainder of a division.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ...selection import Selection
from ..theme import SPACING, active_palette
from .graph_builder import ColumnWell, DropZone, GraphCanvas
from .graph_spec import (
    BAR, BINNED, CHANNELS, EMPTY, FACET_COL, FACET_ROW, HISTOGRAM, PLOT_KINDS,
    SCATTER, GraphSpec, value_axes,
)
from .trellis_spec import (
    MAX_WRAP, SCALE_LABELS, SCALE_MODES, SCALE_SHARED, Trellis, TrellisSpec,
    trellis,
)

LOG = logging.getLogger("spacr.qt.trellis")

__all__ = ["TrellisCanvas", "TrellisPanelWidget"]


class TrellisCanvas(GraphCanvas):
    """A grid of the same chart, one panel per group, on shared axes.

    Everything :class:`~spacr.qt.widgets.graph_builder.GraphCanvas` does —
    linked selection, brushing, the large-data policy, the colour order — with
    the layout and the scales taken from
    :class:`~spacr.qt.widgets.trellis_spec.TrellisSpec`.
    """

    #: Emitted after every render with the :class:`Trellis` that was drawn.
    trellis_rendered = Signal(object)

    def __init__(self, parent=None, *, link=None, source: str = "trellis"):
        super().__init__(parent, link=link, source=source)
        # After the base constructor, which builds the figure and subscribes
        # to the link but does not render — so nothing reads these before
        # they exist.
        self._trellis_spec = TrellisSpec()
        self._trellis: Optional[Trellis] = None

    # -- the spec ---------------------------------------------------------
    @property
    def trellis_spec(self) -> TrellisSpec:
        return self._trellis_spec

    @property
    def trellis(self) -> Optional[Trellis]:
        """The last computed grid, or ``None`` before the first render."""
        return self._trellis

    def set_trellis_spec(self, spec: TrellisSpec, *,
                         immediate: bool = True) -> None:
        """Replace the whole spec and redraw."""
        self._trellis_spec = spec
        # The inherited helpers read `self._spec`; keeping the two in step is
        # what lets every drawing method be reused unchanged.
        self._spec = spec.graph
        if self._frame is not None:
            self._kinds = spec.graph.kinds_for(self._frame)
        if immediate:
            self.render_now()
        else:
            self._debounce.start()

    def set_spec(self, spec: GraphSpec, *, immediate: bool = True) -> None:
        """Replace only the inner chart spec, keeping the grid's own options."""
        self.set_trellis_spec(self._trellis_spec.with_graph(spec),
                              immediate=immediate)

    def set_channel(self, channel: str, column: Optional[str]) -> None:
        self.set_spec(self._spec.with_channel(channel, column))

    # -- rendering --------------------------------------------------------
    def render_now(self) -> None:
        """Rebuild the grid from the frame, the spec, the filter and the
        selection."""
        self._debounce.stop()
        self._figure.clear()
        self._axes = {}
        self._axes_at = {}
        self._overlays = {}
        self._drag_patch = None
        palette = active_palette()

        if self._frame is None or self._frame.empty:
            self._trellis = None
            self._render_message(
                "Load a table, then drag a column onto X or Y and a grouping "
                "column onto Facet ↓ or Facet →.")
            return

        self._visible, self._filter_note = self._apply_filter(self._frame)
        spec = self._trellis_spec
        self._spec = spec.graph
        kinds = self._kinds
        if spec.graph.resolved_kind(kinds) == EMPTY:
            self._trellis = None
            self._render_message(
                "Drag a column onto X or Y to say what each panel shows,\n"
                "then a grouping column onto Facet ↓ or Facet → to repeat it.")
            return

        result = trellis(self._visible, spec)
        kind = spec.graph.resolved_kind(kinds)
        self._trellis = result
        self._render_data = result.data
        self._grid = result.grid
        self._brush_grid = result.grid
        self._scales = result.shared
        self._live_highlight = (kind == SCATTER
                                and result.data.strategy != BINNED)
        self._selected_mask = self._selection_mask(result.frame)

        nrows, ncols = result.shape
        # `sharex`/`sharey` are deliberately off: every panel's limits are
        # written from its own scale group below, which is stronger — and
        # under a free or per-row mode, sharing would be wrong.
        axes = self._figure.subplots(nrows, ncols, squeeze=False,
                                     sharex=False, sharey=False)
        for panel in result.panels:
            ax = axes[panel.row][panel.col]
            self._axes[(panel.row, panel.col)] = ax
            self._axes_at[id(ax)] = (panel.row, panel.col)
            self._overlays[(panel.row, panel.col)] = None
            if not panel.occupied:
                # The remainder of a wrapped division. Not a panel with no
                # data — there is no group here at all — so it is not drawn.
                ax.set_visible(False)
                continue
            self._style_axes(ax, palette)
            rows = panel.frame(result.frame)
            mask = (self._selected_mask[panel.index]
                    if self._selected_mask is not None else None)
            # The inherited drawing helpers read `self._scales`; pointing it
            # at this panel's group is what makes a free-scale histogram use
            # this panel's bin edges rather than the grid's.
            previous, self._scales = self._scales, panel.scales
            try:
                self._overlays[(panel.row, panel.col)] = self._draw_panel(
                    ax, rows, mask, kind, result.data, palette)
            finally:
                self._scales = previous
            self._apply_panel_scales(ax, kind, panel.scales)
            self._label_trellis_panel(ax, panel, result, palette)

        self._draw_legend(kind, palette)
        self._figure.tight_layout(pad=0.7)
        self._canvas.draw_idle()
        self._notice.setText(self._trellis_notice(result))
        self.rendered.emit(result.data)
        self.trellis_rendered.emit(result)

    def _apply_panel_scales(self, ax, kind, scales) -> None:
        """Write this panel's limits, ticks and orders explicitly.

        Always written, never left to an autoscale: under
        :data:`~spacr.qt.widgets.trellis_spec.SCALE_SHARED` every panel is
        handed the same numbers, which is what "shared" has to mean if a shift
        between panels is to read as a shift.
        """
        counts_on_y = kind in (HISTOGRAM, BAR)
        if scales.x_levels is not None:
            ax.set_xticks(range(len(scales.x_levels)))
            ax.set_xticklabels(scales.x_levels, rotation=30, ha="right",
                               fontsize=7)
            ax.set_xlim(-0.6, len(scales.x_levels) - 0.4)
        elif scales.x_limits is not None:
            ax.set_xlim(*scales.x_limits)
        if counts_on_y:
            if scales.count_limit:
                ax.set_ylim(0, scales.count_limit)
        elif scales.y_levels is not None:
            ax.set_yticks(range(len(scales.y_levels)))
            ax.set_yticklabels(scales.y_levels, fontsize=7)
            ax.set_ylim(-0.6, len(scales.y_levels) - 0.4)
        elif scales.y_limits is not None:
            ax.set_ylim(*scales.y_limits)

    def _label_trellis_panel(self, ax, panel, result: Trellis, palette) -> None:
        """Title with n, axis names on the edges, ticks where they are needed."""
        spec = self._trellis_spec
        kind = spec.graph.resolved_kind(self._kinds)
        x_column, y_column = value_axes(spec.graph, self._kinds)
        counts_on_y = kind in (HISTOGRAM, BAR)
        nrows, ncols = result.shape

        below = (result.panel(panel.row + 1, panel.col)
                 if panel.row + 1 < nrows else None)
        is_bottom = below is None or not below.occupied
        is_left = panel.col == 0

        if is_bottom:
            ax.set_xlabel(x_column or "", color=palette["fg_dim"], fontsize=9)
        if is_left:
            ax.set_ylabel("count" if counts_on_y else (y_column or ""),
                          color=palette["fg_dim"], fontsize=9)
        # Inner ticks are hidden only where the axis really is shared. A panel
        # with its own limits prints its own numbers, or the layout is tidier
        # than it is true.
        if spec.scale_x == SCALE_SHARED and not is_bottom:
            ax.tick_params(labelbottom=False)
        if spec.scale_y == SCALE_SHARED and not is_left:
            ax.tick_params(labelleft=False)

        title = panel.label()
        if title:
            colour = (palette["warning"] if panel.is_low_n
                      else palette["fg_dim"])
            ax.set_title(title, color=colour, fontsize=8, pad=3)

    def _trellis_notice(self, result: Trellis) -> str:
        parts = [result.summary()]
        if not self._keyed:
            parts.append("no object keys in this table — brushing cannot "
                         "publish a selection")
        if self._filter_note:
            parts.append(self._filter_note.strip(" ·"))
        if self._selected_mask is not None:
            parts.append(f"{int(self._selected_mask.sum()):,} highlighted")
        return " · ".join(p for p in parts if p)

    # -- brushing ---------------------------------------------------------
    def brush(self, x0: float, y0: float, x1: float, y1: float, *,
              row: int = 0, col: int = 0,
              publish: bool = True) -> Optional[Selection]:
        """Select the rows of one panel inside a swept rectangle.

        Delegated to :meth:`~spacr.qt.widgets.trellis_spec.Trellis.brush`,
        which evaluates the rectangle against the **unsampled** rows and
        against that panel's own scales — so it is exact over a density raster
        and correct under a per-panel scale, where the drawn coordinates of a
        categorical axis differ from the grid's.
        """
        if self._trellis is None or self._visible is None or not self._keyed:
            return None
        try:
            keep = self._trellis.brush(x0, y0, x1, y1, row=row, col=col)
        except IndexError:
            return None
        picked = self._visible.loc[keep]
        if not publish:
            return Selection.from_frame(picked, source=self.link_source)
        return self.publish_selection(picked)

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """Move the highlight; redraw only when the marks cannot be re-styled."""
        if self._trellis is None:
            return
        if not self._live_highlight:
            self.render_now()
            return
        self._selected_mask = self._selection_mask(self._trellis.frame)
        for (row, col), overlay in self._overlays.items():
            if overlay is None:
                continue
            panel = self._trellis.panel(row, col)
            mask = (self._selected_mask[panel.index]
                    if self._selected_mask is not None else None)
            overlay(mask)
        self._canvas.draw_idle()
        self._notice.setText(self._trellis_notice(self._trellis))


class TrellisPanelWidget(QWidget):
    """The whole surface: a column well, the six zones, the grid's options,
    and the canvas.

    The well and the drop zones are
    :class:`~spacr.qt.widgets.graph_builder.ColumnWell` and
    :class:`~spacr.qt.widgets.graph_builder.DropZone` unchanged, so a column
    dragged here and a column dragged in the Graph Builder are the same
    gesture with the same payload type.
    """

    spec_changed = Signal(object)

    def __init__(self, parent=None, *, link=None, source: str = "trellis"):
        super().__init__(parent)
        self.setObjectName("TrellisPanel")
        self._zones: Dict[str, DropZone] = {}
        self._building = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        shelf = QWidget(self)
        shelf.setObjectName("TrellisShelf")
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
        self._kind.setObjectName("TrellisKindPicker")
        self._kind.addItem("Automatic", "")
        for kind in PLOT_KINDS:
            if kind != EMPTY:
                self._kind.addItem(kind.capitalize(), kind)
        self._kind.setToolTip("The plot repeated in every panel.")
        self._kind.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("Plot", shelf), 0, 0)
        controls.addWidget(self._kind, 0, 1)

        self._scale_x = QComboBox(shelf)
        self._scale_y = QComboBox(shelf)
        for box, axis in ((self._scale_x, "X"), (self._scale_y, "Y")):
            box.setObjectName(f"TrellisScale{axis}")
            for mode in SCALE_MODES:
                box.addItem(SCALE_LABELS[mode].split(" — ")[0], mode)
            box.setToolTip(
                f"{axis} scale.\n" + "\n".join(SCALE_LABELS[m]
                                               for m in SCALE_MODES))
            box.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("X scale", shelf), 1, 0)
        controls.addWidget(self._scale_x, 1, 1)
        controls.addWidget(QLabel("Y scale", shelf), 2, 0)
        controls.addWidget(self._scale_y, 2, 1)

        self._wrap = QSpinBox(shelf)
        self._wrap.setRange(0, MAX_WRAP)
        self._wrap.setSpecialValueText("off")
        self._wrap.setToolTip(
            "With one facet channel in use, lay the levels out this many "
            "panels wide instead of in one long strip. Ignored when both "
            "facet channels are in use.")
        self._wrap.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("Wrap at", shelf), 3, 0)
        controls.addWidget(self._wrap, 3, 1)

        self._bins = QSpinBox(shelf)
        self._bins.setRange(2, 200)
        self._bins.setValue(30)
        self._bins.setToolTip("Bins per axis, for histograms and the density "
                              "raster.")
        self._bins.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(QLabel("Bins", shelf), 4, 0)
        controls.addWidget(self._bins, 4, 1)
        shelf_layout.addLayout(controls)

        self._clear = QPushButton("Clear all channels", shelf)
        self._clear.clicked.connect(self.clear_channels)
        shelf_layout.addWidget(self._clear)

        splitter.addWidget(shelf)
        self.canvas = TrellisCanvas(self, link=link, source=source)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        self.well.set_frame(frame)
        self.canvas.set_frame(frame)
        self._sync()

    @property
    def spec(self) -> TrellisSpec:
        return self.canvas.trellis_spec

    def set_spec(self, spec: TrellisSpec) -> None:
        self.canvas.set_trellis_spec(spec)
        self._sync()

    def zone(self, channel: str) -> DropZone:
        return self._zones[channel]

    def clear_channels(self) -> None:
        for zone in self._zones.values():
            zone.set_column(None)

    # -- wiring -----------------------------------------------------------
    def _on_zone_changed(self, channel: str, column: str) -> None:
        if self._building:
            return
        self.canvas.set_channel(channel, column or None)
        self.spec_changed.emit(self.canvas.trellis_spec)

    def _on_controls_changed(self, *_args) -> None:
        if self._building:
            return
        spec = self.canvas.trellis_spec
        graph = replace(spec.graph.with_kind(self._kind.currentData() or None),
                        bins=int(self._bins.value()))
        spec = spec.with_graph(graph).with_scales(
            self._scale_x.currentData(), self._scale_y.currentData())
        spec = spec.with_wrap(int(self._wrap.value()))
        self.canvas.set_trellis_spec(spec)
        self.spec_changed.emit(spec)

    def _sync(self) -> None:
        """Make the controls show what the spec actually says."""
        self._building = True
        try:
            spec = self.canvas.trellis_spec
            for channel, zone in self._zones.items():
                zone.set_column(spec.graph.column_for(channel))
            index = self._kind.findData(spec.graph.kind or "")
            if index >= 0:
                self._kind.setCurrentIndex(index)
            self._bins.setValue(spec.graph.bins)
            for box, mode in ((self._scale_x, spec.scale_x),
                              (self._scale_y, spec.scale_y)):
                position = box.findData(mode)
                if position >= 0:
                    box.setCurrentIndex(position)
            self._wrap.setValue(spec.wrap)
        finally:
            self._building = False

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.canvas.close()
        super().closeEvent(event)
