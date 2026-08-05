"""Small multiples — one chart repeated per group, on axes that really are shared.

A trellis is the cheapest correct comparison there is: the same plot, once per
plate or once per gene, laid out in a grid so the eye does the differencing. It
is also the easiest one to break, in a way that produces a figure everybody
believes. Three rules, and this module exists to hold all three in one place.

**Shared axes, by default and loudly.** Two panels whose y axes differ by an
order of magnitude look identical, so a per-panel autoscale turns "the knockdown
halved the count" into "the two panels look the same". Every scale here is
computed over the whole grid unless the user explicitly asks otherwise, and
when they do ask otherwise :attr:`Trellis.notice` says so in words — because a
free-scale trellis screenshotted into a slide has no other way to admit it.
:data:`SCALE_ROW` and :data:`SCALE_COL` are the middle grounds: panels down a
column of the grid share, or panels across a row do, which is what you want when
the rows are three different measurements and the columns are three conditions.

**Every panel prints its n.** Not a toggle. A box over four objects and a box
over four thousand are the same box, and the number that separates them is the
one an option would let people turn off. The same stance
:mod:`spacr.qt.widgets.pivot_spec` takes about cells, and :data:`LOW_N` is
imported from there rather than written again.

**Empty panels are drawn.** Inherited whole from
:func:`spacr.qt.widgets.graph_spec.facet_grid`: the grid is the full cartesian
product of the two facet channels, so "plate 3 / row H was measured and nothing
survived the filter" and "there is no plate 3 / row H" stay different pictures.

What this adds over the Graph Builder
-------------------------------------

The Graph Builder already facets — it has to, since dragging a column onto
Facet ↓ has to do something. What is here and not there:

* **two-way faceting as the subject** rather than a decoration, including
  :func:`wrap` for the single-channel case, so twelve plates are a 4 × 3 block
  rather than a strip twelve panels wide;
* **per-panel scale groups** — the four :data:`SCALE_MODES` above, per axis,
  where the Graph Builder has one shared/not-shared boolean per axis;
* **per-panel n**, and a summary of the spread of n across the grid, because
  the first question about any trellis is whether the small panels are small
  because of biology or because of sampling.

Colour and size are never per-panel
-----------------------------------

Whatever the scale mode, the colour levels, the colour ramp limits and the size
limits are computed once over the whole grid. A gene that is blue in one panel
and orange in the next is not a legend, it is a trap; and a mark area that means
200 px² on the left and 20 000 px² on the right is worse, because nothing on
screen says so.

No Qt in here — pure pandas and numpy, like the modules it builds on.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_spec import (
    BAR, FULL, HISTOGRAM, FacetGrid, GraphSpec, RenderData, Scales, SpecError,
    _level_series, brush_mask, facet_grid, prepare_data, scales_for,
)
from .pivot_spec import LOW_N

__all__ = [
    "SCALE_SHARED", "SCALE_FREE", "SCALE_ROW", "SCALE_COL", "SCALE_MODES",
    "SCALE_LABELS", "LOW_N", "MAX_WRAP",
    "TrellisSpec", "TrellisPanel", "Trellis", "trellis", "wrap_positions",
]

#: One set of limits for every panel. The default, and the whole point.
SCALE_SHARED = "shared"

#: Every panel autoscales to its own rows. Comparable within a panel and
#: nowhere else — the notice says so.
SCALE_FREE = "free"

#: Panels in the same grid **row** share limits. For a grid whose rows are
#: different measurements (area, intensity, eccentricity) and whose columns are
#: conditions: comparing across the row is the question, and forcing an
#: intensity axis onto an area panel would flatten both.
SCALE_ROW = "row"

#: Panels in the same grid **column** share limits.
SCALE_COL = "col"

SCALE_MODES: Tuple[str, ...] = (SCALE_SHARED, SCALE_FREE, SCALE_ROW, SCALE_COL)

#: What each mode is, in the words the option list uses.
SCALE_LABELS: Dict[str, str] = {
    SCALE_SHARED: "shared — one scale for the whole grid (compare anywhere)",
    SCALE_FREE: "free — every panel autoscales (compare inside a panel only)",
    SCALE_ROW: "per row — panels across a row share (compare along a row)",
    SCALE_COL: "per column — panels down a column share (compare down a column)",
}

#: Widest a wrapped grid may be. Past this the panels are too narrow to carry
#: an axis, and the answer is a filter rather than a wider figure.
MAX_WRAP = 12


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrellisSpec:
    """A :class:`~spacr.qt.widgets.graph_spec.GraphSpec` plus the grid's rules.

    Composition rather than a subclass: the graph spec is a complete,
    serialisable description of one chart, and a trellis is that chart *and* a
    layout. Keeping them apart means a trellis can hand its inner spec to the
    Graph Builder, the gate editor or the feature explorer unchanged, and a
    chart built anywhere can be dropped into a grid without conversion.

    :param graph: what to draw in each panel — the six channels, the kind, the
        bins, the point budget. Its ``facet_row`` / ``facet_col`` are the grid.
    :param scale_x: one of :data:`SCALE_MODES` for the horizontal axis.
    :param scale_y: likewise for the vertical one.
    :param wrap: for a grid faceted on **one** channel only, lay the levels out
        this many panels wide instead of in a single strip. ``0`` keeps the
        strip. Ignored with a notice when both facet channels are in use —
        wrapping a two-way grid would put unrelated levels in the same row.
    :raises SpecError: on an unknown scale mode or a wrap beyond
        :data:`MAX_WRAP`, at the point the spec is built.
    """

    graph: GraphSpec = field(default_factory=GraphSpec)
    scale_x: str = SCALE_SHARED
    scale_y: str = SCALE_SHARED
    wrap: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphSpec):
            object.__setattr__(self, "graph",
                               GraphSpec.from_dict(dict(self.graph)))
        for name in ("scale_x", "scale_y"):
            mode = str(getattr(self, name))
            if mode not in SCALE_MODES:
                raise SpecError(
                    f"{name} is {mode!r}; the scale modes are "
                    f"{', '.join(SCALE_MODES)}")
            object.__setattr__(self, name, mode)
        wrap = int(self.wrap)
        if wrap < 0 or wrap > MAX_WRAP:
            raise SpecError(
                f"wrap is {self.wrap}; it must be 0 (no wrapping) or at most "
                f"{MAX_WRAP} panels wide")
        object.__setattr__(self, "wrap", wrap)

    # -- the inner spec, reachable without reaching through --------------
    @property
    def x(self) -> Optional[str]:
        return self.graph.x

    @property
    def y(self) -> Optional[str]:
        return self.graph.y

    @property
    def colour(self) -> Optional[str]:
        return self.graph.colour

    @property
    def facet_row(self) -> Optional[str]:
        return self.graph.facet_row

    @property
    def facet_col(self) -> Optional[str]:
        return self.graph.facet_col

    @property
    def is_two_way(self) -> bool:
        """Both facet channels are in use — the case the grid is *for*."""
        return bool(self.graph.facet_row) and bool(self.graph.facet_col)

    @property
    def is_faceted(self) -> bool:
        return bool(self.graph.facet_row) or bool(self.graph.facet_col)

    @property
    def shares_everything(self) -> bool:
        """Both axes shared — the state in which the grid is comparable."""
        return self.scale_x == SCALE_SHARED and self.scale_y == SCALE_SHARED

    # -- edits ------------------------------------------------------------
    def with_graph(self, graph: GraphSpec) -> "TrellisSpec":
        return replace(self, graph=graph)

    def with_channel(self, channel: str, column: Optional[str]) -> "TrellisSpec":
        return replace(self, graph=self.graph.with_channel(channel, column))

    def with_kind(self, kind: Optional[str]) -> "TrellisSpec":
        return replace(self, graph=self.graph.with_kind(kind))

    def with_scales(self, scale_x: Optional[str] = None,
                    scale_y: Optional[str] = None) -> "TrellisSpec":
        return replace(self, scale_x=scale_x or self.scale_x,
                       scale_y=scale_y or self.scale_y)

    def with_wrap(self, wrap: int) -> "TrellisSpec":
        return replace(self, wrap=int(wrap))

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"graph": self.graph.to_dict(), "scale_x": self.scale_x,
                "scale_y": self.scale_y, "wrap": self.wrap}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrellisSpec":
        data = dict(payload)
        known = {k: v for k, v in data.items()
                 if k in {"scale_x", "scale_y", "wrap"}}
        known["graph"] = GraphSpec.from_dict(data.get("graph") or {})
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrellisSpec":
        return cls.from_dict(json.loads(text))

    def describe(self, kinds: Optional[Mapping[str, str]] = None) -> str:
        parts = [self.graph.describe(kinds)]
        if self.scale_x != SCALE_SHARED or self.scale_y != SCALE_SHARED:
            parts.append(f"x scale: {self.scale_x} · y scale: {self.scale_y}")
        if self.wrap:
            parts.append(f"wrapped at {self.wrap}")
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------

def wrap_positions(count: int, wrap: int) -> Tuple[Tuple[int, int], ...]:
    """Grid positions for ``count`` levels laid out ``wrap`` panels wide.

    Row-major, so reading order and level order are the same order. Returned
    rather than applied, so the caller can also work out how many trailing
    slots are blank — a wrapped grid of seven levels at three wide has two
    blanks, and they are *placeholders*, not empty groups.
    """
    if wrap < 1:
        return tuple((i, 0) for i in range(count))
    return tuple((i // wrap, i % wrap) for i in range(count))


@dataclass(frozen=True)
class TrellisPanel:
    """One panel: which group it holds, which rows, and its own scales.

    ``index`` holds *positional* indices into the frame the trellis was
    computed over, exactly as
    :class:`~spacr.qt.widgets.graph_spec.FacetPanel` does and for the same
    reason — a measurement frame carries a duplicated index often enough that
    positions are the only safe currency.

    :param occupied: ``False`` for the blank slots at the end of a wrapped
        grid. A blank slot has no group at all, which is different from a group
        with no rows: the first is "the grid is 3 wide and 7 does not divide by
        3", the second is a fact about the data. The renderer hides the first
        and draws the second.
    """

    row: int
    col: int
    row_level: Optional[str]
    col_level: Optional[str]
    index: np.ndarray
    scales: Scales
    occupied: bool = True

    @property
    def n(self) -> int:
        """Rows in this panel. Printed on every panel, always."""
        return int(len(self.index))

    @property
    def is_empty(self) -> bool:
        """A real group with no rows. Drawn, empty, with its n of 0."""
        return self.occupied and self.n == 0

    @property
    def is_low_n(self) -> bool:
        """At or below :data:`LOW_N`, and not empty — worth flagging."""
        return self.occupied and 0 < self.n <= LOW_N

    def frame(self, source: pd.DataFrame) -> pd.DataFrame:
        return source.iloc[self.index]

    def title(self) -> str:
        """The group, without the n."""
        parts = [p for p in (self.row_level, self.col_level) if p is not None]
        return " · ".join(parts)

    def label(self) -> str:
        """What goes above the panel: the group **and** its n.

        Always both. See the module docstring — this is the number that stops
        a box over four objects reading like a box over four thousand.
        """
        if not self.occupied:
            return ""
        title = self.title()
        count = f"n = {self.n:,}"
        return f"{title}  ·  {count}" if title else count


@dataclass(frozen=True)
class Trellis:
    """A computed grid: the panels, their scales, and what to say about it.

    :param frame: the rows that were laid out — post-filter, post-large-data
        policy. Panel indices are positions into *this*.
    :param source: the rows before the large-data policy, for an exact brush.
    :param data: the large-data decision, carried whole so the caller can print
        :attr:`~spacr.qt.widgets.graph_spec.RenderData.notice` unchanged.
    :param shared: the whole-grid scales, computed whatever the modes are, so a
        caption can say what the shared limits *would* have been.
    """

    spec: TrellisSpec
    frame: pd.DataFrame
    source: pd.DataFrame
    kinds: Mapping[str, str]
    grid: FacetGrid
    panels: Tuple[TrellisPanel, ...]
    shape: Tuple[int, int]
    shared: Scales
    data: RenderData
    notice: str = ""

    # -- reading the grid --------------------------------------------------
    @property
    def n_panels(self) -> int:
        """Rows × columns — **including** empty panels and blank slots."""
        return len(self.panels)

    @property
    def n_occupied(self) -> int:
        """Panels that stand for a real group, empty or not."""
        return sum(1 for p in self.panels if p.occupied)

    @property
    def n_empty(self) -> int:
        """Real groups with no rows. Non-zero is a fact worth reading."""
        return sum(1 for p in self.panels if p.is_empty)

    def panel(self, row: int, col: int) -> TrellisPanel:
        return self.panels[row * self.shape[1] + col]

    def scales_at(self, row: int, col: int) -> Scales:
        return self.panel(row, col).scales

    def n_at(self, row: int, col: int) -> int:
        return self.panel(row, col).n

    def n_range(self) -> Optional[Tuple[int, int]]:
        """``(smallest, largest)`` n over the non-empty panels."""
        counts = [p.n for p in self.panels if p.occupied and p.n]
        return (min(counts), max(counts)) if counts else None

    def low_n_panels(self) -> Tuple[TrellisPanel, ...]:
        return tuple(p for p in self.panels if p.is_low_n)

    # -- the honesty line --------------------------------------------------
    def axes_are_comparable(self) -> bool:
        """Whether a difference between two panels is a difference in the data."""
        return self.spec.shares_everything

    def summary(self) -> str:
        """One line under the grid: shape, n spread, and every caveat."""
        rows, cols = self.shape
        parts = [f"{rows} × {cols} = {self.n_panels} panels"]
        if self.n_empty:
            parts.append(f"{self.n_empty} with no rows")
        span = self.n_range()
        if span is not None:
            parts.append(f"n per panel {span[0]:,}–{span[1]:,}"
                         if span[0] != span[1] else f"n = {span[0]:,} each")
        low = len(self.low_n_panels())
        if low:
            parts.append(f"{low} panel(s) at n ≤ {LOW_N}")
        if self.notice:
            parts.append(self.notice)
        return " · ".join(parts)

    def brush(self, x0: float, y0: float, x1: float, y1: float, *,
              row: int = 0, col: int = 0) -> np.ndarray:
        """Rows of :attr:`source` inside a rectangle swept on one panel.

        Evaluated against the **unsampled** rows and against that panel's own
        scales, so a brush stays exact when the panel was drawn from a sample
        or as a density raster, and a categorical axis under a free scale is
        matched against the levels that panel actually drew.
        """
        panel = self.panel(row, col)
        inside = brush_mask(self.source, self.spec.graph, self.kinds,
                            x0, y0, x1, y1, panel.scales)
        if not panel.occupied:
            return np.zeros(len(self.source), dtype=bool)
        return inside & self._panel_membership(panel)

    def _panel_membership(self, panel: TrellisPanel) -> np.ndarray:
        """Which rows of :attr:`source` belong to ``panel``'s group."""
        keep = np.ones(len(self.source), dtype=bool)
        for column, level in ((self.spec.facet_row, panel.row_level),
                              (self.spec.facet_col, panel.col_level)):
            if column and level is not None and column in self.source.columns:
                keep &= (_level_series(self.source, column).to_numpy() == level)
        return keep


# ---------------------------------------------------------------------------
# Computing it
# ---------------------------------------------------------------------------

def _panel_top(frame: pd.DataFrame, spec: GraphSpec, panel: TrellisPanel,
               kind: str, x_edges) -> float:
    """The tallest bar or bin in one panel, given the edges it is drawn with.

    The edges come from the panel's **x** group, not its y group: the count is
    on y but the *binning* is on x, and computing the tallest bin against the
    wrong edges is how a shared count axis ends up shorter than the bar it is
    supposed to contain.
    """
    column = spec.x or spec.y
    if not column or column not in frame.columns or not panel.occupied \
            or panel.n == 0:
        return 0.0
    rows = frame.iloc[panel.index]
    if kind == HISTOGRAM and x_edges is not None:
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size:
            counts, _ = np.histogram(values, bins=x_edges)
            return float(counts.max())
        return 0.0
    if kind == BAR:
        counts = _level_series(rows, column).value_counts()
        if len(counts):
            return float(counts.max())
    return 0.0


#: Headroom above the tallest bar, so it is not flush with the top spine.
_COUNT_HEADROOM = 1.08


def _group_key(mode: str, panel: TrellisPanel) -> Any:
    """What ``panel`` shares a scale with under ``mode``.

    One constant for :data:`SCALE_SHARED` (everything is one group), the
    panel's own position for :data:`SCALE_FREE` (every panel is its own), and
    the grid row or column for the two middle grounds.
    """
    if mode == SCALE_SHARED:
        return 0
    if mode == SCALE_FREE:
        return (panel.row, panel.col)
    return panel.row if mode == SCALE_ROW else panel.col


def _groups(mode: str, panels: Sequence[TrellisPanel]
            ) -> Tuple[Dict[Any, List[int]], List[Any]]:
    """``({key: panel positions}, key per panel)`` under ``mode``.

    The per-panel key list is returned alongside the groups so the caller looks
    its group up rather than searching for it; a scan per panel is O(panels²),
    which a 12 × 12 grid notices.
    """
    groups: Dict[Any, List[int]] = {}
    keys: List[Any] = []
    for position, panel in enumerate(panels):
        key = _group_key(mode, panel)
        groups.setdefault(key, []).append(position)
        keys.append(key)
    return groups, keys


def _scales_for_group(frame: pd.DataFrame, spec: GraphSpec,
                      kinds: Mapping[str, str],
                      panels: Sequence[TrellisPanel]) -> Scales:
    """Scales over exactly the rows in ``panels``.

    ``count_limit`` is left alone here and filled in afterwards: it needs the
    x group's bin edges and the y group's membership at the same time, and
    neither group knows about the other until both have been computed.
    """
    indices = [p.index for p in panels if p.occupied and p.n]
    rows = (frame.iloc[np.concatenate(indices)] if indices
            else frame.iloc[np.zeros(0, dtype=int)])
    return scales_for(rows, spec, kinds, None)


def trellis(frame: pd.DataFrame, spec: Optional[TrellisSpec] = None, *,
            levels_source: Optional[pd.DataFrame] = None) -> Trellis:
    """Lay ``frame`` out as small multiples.

    :param frame: the rows to draw — already narrowed by whatever filter the
        views share.
    :param levels_source: where the facet *levels* come from when that is not
        ``frame``; passed straight through to
        :func:`~spacr.qt.widgets.graph_spec.facet_grid`, so a level that exists
        in the population but drew no rows still gets its panel.
    :returns: a :class:`Trellis` whose panels are the full grid, blanks and
        empties included, each carrying its own :class:`Scales` and its n.
    """
    spec = spec or TrellisSpec()
    graph = spec.graph
    kinds = graph.kinds_for(frame)
    notices: List[str] = []

    data = prepare_data(frame, graph, kinds)
    grid = facet_grid(data.frame, graph,
                      levels_source=levels_source
                      if levels_source is not None else frame)
    if grid.notice:
        notices.append(grid.notice)

    placed = _place(grid, spec, notices)
    panels_shape, seats = placed

    # Scales. Colour and size come from the whole grid, always — see the module
    # docstring. Only the two positional axes take the mode.
    shared = _scales_for_group(data.frame, graph, kinds, seats)
    x_groups, x_keys = _groups(spec.scale_x, seats)
    y_groups, y_keys = _groups(spec.scale_y, seats)
    x_scales = {key: _scales_for_group(
        data.frame, graph, kinds, [seats[i] for i in positions])
        for key, positions in x_groups.items()}
    y_scales = {key: _scales_for_group(
        data.frame, graph, kinds, [seats[i] for i in positions])
        for key, positions in y_groups.items()}

    panels: List[TrellisPanel] = []
    for position, seat in enumerate(seats):
        xs, ys = x_scales[x_keys[position]], y_scales[y_keys[position]]
        panels.append(replace(seat, scales=Scales(
            x_limits=xs.x_limits, x_levels=xs.x_levels, x_edges=xs.x_edges,
            y_limits=ys.y_limits, y_levels=ys.y_levels, y_edges=ys.y_edges,
            colour_levels=shared.colour_levels,
            colour_limits=shared.colour_limits,
            size_limits=shared.size_limits)))

    kind = graph.resolved_kind(kinds)
    if kind in (HISTOGRAM, BAR):
        # The count axis is the y axis, so it shares along the *y* groups —
        # but each panel's bars are counted with its own x group's edges.
        # Sharing the value axis of an aggregate is the same rule as sharing a
        # data axis; forgetting it is the usual way a faceted histogram lies.
        tops = [_panel_top(data.frame, graph, panel, kind, panel.scales.x_edges)
                for panel in panels]
        for key, positions in y_groups.items():
            tallest = max((tops[i] for i in positions), default=0.0)
            limit = tallest * _COUNT_HEADROOM if tallest > 0 else None
            for i in positions:
                panels[i] = replace(
                    panels[i],
                    scales=replace(panels[i].scales, count_limit=limit))

    if not spec.shares_everything:
        notices.append(_free_scale_warning(spec))
    if data.notice and data.strategy != FULL:
        notices.append(data.notice)

    return Trellis(
        spec=spec, frame=data.frame, source=frame, kinds=kinds, grid=grid,
        panels=tuple(panels), shape=panels_shape, shared=shared, data=data,
        notice="; ".join(dict.fromkeys(n for n in notices if n)))


def _free_scale_warning(spec: TrellisSpec) -> str:
    """The sentence a non-shared grid must carry.

    Written as what the reader must *not* do, rather than as a description of
    the setting: "x scale: free" is a preference, "panels are not comparable
    left to right" is the consequence.
    """
    parts = []
    for axis, mode in (("horizontally", spec.scale_x), ("vertically", spec.scale_y)):
        if mode == SCALE_SHARED:
            continue
        where = {SCALE_FREE: "panels are NOT comparable to each other",
                 SCALE_ROW: "only panels in the same row are comparable",
                 SCALE_COL: "only panels in the same column are comparable"}[mode]
        parts.append(f"{axis}: {where}")
    return "axes are not shared — " + "; ".join(parts)


def _place(grid: FacetGrid, spec: TrellisSpec, notices: List[str]
           ) -> Tuple[Tuple[int, int], List[TrellisPanel]]:
    """Turn a :class:`FacetGrid` into seated panels, wrapping if asked.

    Returns the grid shape and the panels in row-major order, with
    ``scales=Scales()`` — the caller fills those in once the groups are known.
    """
    empty = Scales()
    faceted = [TrellisPanel(row=p.row, col=p.col, row_level=p.row_level,
                            col_level=p.col_level, index=p.index, scales=empty)
               for p in grid.panels]
    if not spec.wrap:
        return grid.shape, faceted
    if spec.is_two_way:
        notices.append(
            "wrap ignored: the grid is faceted both ways, and wrapping it "
            "would put unrelated levels in the same row")
        return grid.shape, faceted

    count = len(faceted)
    width = min(spec.wrap, count) or 1
    height = int(math.ceil(count / width))
    positions = wrap_positions(count, width)
    seated: Dict[Tuple[int, int], TrellisPanel] = {}
    for (row, col), panel in zip(positions, faceted):
        seated[(row, col)] = replace(panel, row=row, col=col)
    out: List[TrellisPanel] = []
    for row in range(height):
        for col in range(width):
            out.append(seated.get((row, col), TrellisPanel(
                row=row, col=col, row_level=None, col_level=None,
                index=np.zeros(0, dtype=int), scales=empty, occupied=False)))
    return (height, width), out
