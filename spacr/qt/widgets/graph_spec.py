"""The Graph Builder's *spec*: what is on which channel, and nothing else.

A JMP-style graph builder is two things that want very much to be one thing:
a pile of drag-and-drop chrome, and a small declarative object saying "``area``
is on x, ``gene`` is on colour, facet by plate down the rows". This module is
the second one, kept deliberately apart from the first.

Why the separation is the important decision
--------------------------------------------

Four later screens — small multiples, the gate editor, the feature explorer and
the campaign control charts — are all "a Graph Builder with one extra rule".
If the axis and facet logic lives inside a ``QWidget``, each of them either
re-derives it or inherits a widget it does not want. So everything in here is

* **pure pandas and numpy, with no Qt**, like :mod:`spacr.selection` — testable
  without a display, usable from a notebook or from ``spacr-run``;
* **serialisable** — :meth:`GraphSpec.to_dict` round-trips through JSON, so a
  chart is something a settings file, a report or a macro can carry;
* **immutable** — every channel change returns a new spec, which is what makes
  "undo the last drag" and "compare two specs" trivial rather than a diffing
  problem.

The four pieces
---------------

:class:`GraphSpec`
    Which column is on which of the six channels (x, y, colour, size,
    facet-row, facet-column), the plot type (or ``None`` for "infer it"), and
    the handful of options that change *what is computed* rather than what it
    looks like.

:func:`facet_grid`
    The panel layout — the **full cartesian product** of the row and column
    levels, empty combinations included. A missing panel and an empty panel say
    different things ("this table has no plate 3 / row H" versus "plate 3 row H
    was measured and nothing survived the filter"), and only one of them is
    true; drawing the grid complete is what keeps the reader from guessing.

:func:`scales_for`
    One set of limits, bin edges, category orders and colour levels for *every*
    panel. Shared axes are the default because comparing across panels is the
    entire point of faceting, and two panels whose y axes differ by an order of
    magnitude look identical.

:func:`prepare_data`
    The large-data policy, stated rather than hidden. See below.

Large data
----------

spaCR measurement tables run to 10^5–10^6 object rows and nobody wants a
scatter of a million overlapping dots. Three strategies, chosen by what the
plot actually needs, and **the chosen one is always named in
:attr:`RenderData.notice`** so a subset can never be mistaken for the whole:

* **aggregate plots use every row, always.** A histogram, bar, box, violin or
  heatmap is already a reduction — sampling before aggregating would change the
  answer for no gain, so :data:`AGGREGATE_KINDS` never sample regardless of
  size.
* **scatter/line up to** :data:`DEFAULT_POINT_BUDGET` **rows: every row is a
  mark.**
* **scatter above the budget: 2-D density binning** — every row is counted into
  a shared-edge 2-D histogram and the panel is drawn as a raster. Nothing is
  dropped, the density is quantitative, and a brush is still exact because a
  rectangle brush is a predicate on x and y evaluated against the *full* frame,
  not against the pixels.
* **scatter above the budget where binning cannot answer the question** — a
  categorical colour or a size channel needs per-point marks — falls back to a
  **seeded uniform sample**, with the count, the fraction and the word
  "sampled" in the notice.

The one thing not on offer is quietly plotting the head of the frame.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .data_filter_panel import classify_columns

__all__ = [
    "SpecError",
    "CHANNELS", "X", "Y", "COLOUR", "SIZE", "FACET_ROW", "FACET_COL",
    "POSITIONAL_CHANNELS", "FACET_CHANNELS",
    "PLOT_KINDS", "AGGREGATE_KINDS",
    "SCATTER", "LINE", "HISTOGRAM", "BAR", "BOX", "VIOLIN", "HEATMAP", "EMPTY",
    "CONTINUOUS", "CATEGORICAL", "UNPLOTTABLE",
    "MISSING_LEVEL", "MAX_FACET_LEVELS", "MAX_PANELS",
    "DEFAULT_POINT_BUDGET",
    "GraphSpec", "column_kinds", "plottable_columns", "infer_kind",
    "value_axes", "brush_mask", "FULL", "BINNED", "SAMPLED",
    "FacetPanel", "FacetGrid", "facet_grid",
    "Scales", "scales_for",
    "RenderData", "prepare_data",
]


class SpecError(ValueError):
    """A spec that cannot mean anything — an unknown channel or plot type.

    Raised at the point the spec is built rather than at render time. A
    misspelled channel that fell through to "no column on x" would draw a
    plausible-looking chart of the wrong thing, which is worse than a
    traceback next to the line that caused it.
    """


# ---------------------------------------------------------------------------
# The six channels
# ---------------------------------------------------------------------------

X = "x"
Y = "y"
COLOUR = "colour"
SIZE = "size"
FACET_ROW = "facet_row"
FACET_COL = "facet_col"

#: Every drop zone, in the order the panel lays them out.
CHANNELS: Tuple[str, ...] = (X, Y, COLOUR, SIZE, FACET_ROW, FACET_COL)

#: The two that decide the plot type. Colour and size decorate; facets
#: replicate; only these choose what kind of chart this is.
POSITIONAL_CHANNELS: Tuple[str, ...] = (X, Y)

#: The two that split one chart into a grid of them.
FACET_CHANNELS: Tuple[str, ...] = (FACET_ROW, FACET_COL)

# ---------------------------------------------------------------------------
# The plot types
# ---------------------------------------------------------------------------

SCATTER = "scatter"
LINE = "line"
HISTOGRAM = "histogram"
BAR = "bar"
BOX = "box"
VIOLIN = "violin"
HEATMAP = "heatmap"
EMPTY = "empty"

#: Everything a user may pick from the override menu, plus ``EMPTY`` which is
#: only ever inferred (there is nothing to override when nothing is dropped).
PLOT_KINDS: Tuple[str, ...] = (
    SCATTER, LINE, HISTOGRAM, BAR, BOX, VIOLIN, HEATMAP, EMPTY)

#: Kinds that reduce many rows to few marks. They never sample: the reduction
#: *is* the answer, and computing it on a tenth of the rows would move it.
AGGREGATE_KINDS = frozenset({HISTOGRAM, BAR, BOX, VIOLIN, HEATMAP})

# ---------------------------------------------------------------------------
# Column kinds
# ---------------------------------------------------------------------------

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
UNPLOTTABLE = "skip"

#: What a missing facet value is called. Rows with a NaN facet key get their
#: own panel rather than disappearing — a facet that silently drops a tenth of
#: the table is exactly the failure the empty-panel rule exists to prevent.
MISSING_LEVEL = "(missing)"

#: Facet levels per axis. Twelve panels down a page is already a lot; beyond
#: this the grid is unreadable and the answer is a filter, not a taller figure.
MAX_FACET_LEVELS = 12

#: Hard cap on drawn panels, whatever the two level counts multiply to.
MAX_PANELS = MAX_FACET_LEVELS * MAX_FACET_LEVELS

#: Individual marks a scatter will draw before switching to density binning or
#: sampling. Chosen where an Agg canvas still redraws in well under a second.
DEFAULT_POINT_BUDGET = 50_000


def column_kinds(frame: pd.DataFrame) -> Dict[str, str]:
    """Sort ``frame``'s columns into :data:`CONTINUOUS` / :data:`CATEGORICAL`
    / :data:`UNPLOTTABLE`.

    A thin re-reading of
    :func:`spacr.qt.widgets.data_filter_panel.classify_columns` — the Local
    Data Filter's rule — rather than a second classifier. The two screens must
    agree about what ``cell_count`` is, or a column offered as a tick list in
    the filter and as a continuous axis in the plot would give a user two
    different mental models of the same table.

    The translation is one-to-one: a column the filter offers as a *range* is
    continuous; one it offers as *ticks* is categorical; one it skips
    (high-cardinality free text, or a key that identifies rather than
    describes) is not worth an axis either.
    """
    translation = {"range": CONTINUOUS, "category": CATEGORICAL,
                   "skip": UNPLOTTABLE}
    return {name: translation[kind]
            for name, kind in classify_columns(frame).items()}


def plottable_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns worth offering in the drag well, sorted.

    Same rule, same reason as the filter panel's picker: a measurement table
    has hundreds of columns, and offering all of them is the same as offering
    none.
    """
    return tuple(sorted(name for name, kind in column_kinds(frame).items()
                        if kind != UNPLOTTABLE))


def _axis_kind(column: Optional[str], kinds: Mapping[str, str]) -> Optional[str]:
    """The kind an axis should treat ``column`` as, or ``None`` for no column.

    An :data:`UNPLOTTABLE` column reaching an axis — a spec restored from a
    file written against a different table, say — is drawn as categorical
    rather than refused. It will be a crowded chart, which is a visible,
    correctable state; refusing would leave the user with a blank canvas and
    nothing to click.
    """
    if not column:
        return None
    kind = kinds.get(column, CATEGORICAL)
    return CONTINUOUS if kind == CONTINUOUS else CATEGORICAL


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphSpec:
    """Which column is on which channel, and what to draw.

    Frozen: every edit returns a new spec (:meth:`with_channel`,
    :meth:`with_kind`), so the panel can keep a history for undo and two specs
    can be compared with ``==``.

    :param x: column on the horizontal axis, or ``None``.
    :param y: column on the vertical axis, or ``None``.
    :param colour: column mapped to hue. Categorical → the fixed
        eight-colour order; continuous → one-hue light-to-dark ramp.
    :param size: column mapped to mark area. Only meaningful for point marks;
        an aggregate plot ignores it rather than pretending.
    :param facet_row: column whose levels become rows of panels.
    :param facet_col: column whose levels become columns of panels.
    :param kind: one of :data:`PLOT_KINDS`, or ``None`` to infer it from what
        was dropped (:func:`infer_kind`). ``None`` and "the inferred value"
        are kept apart on purpose: a spec that inferred ``scatter`` and one
        that *pinned* ``scatter`` behave differently the moment the user drags
        a categorical column onto x.
    :param roles: per-column override of :func:`column_kinds` —
        ``{"cell_count": "continuous"}``. The table-wide rule is a good guess
        and not always the right one, and the alternative to an override here
        is the user editing the table.
    :param bins: histogram / density bin count per axis.
    :param shared_x: every panel gets the same x limits. Default on.
    :param shared_y: likewise for y.
    :param point_budget: individual marks before :func:`prepare_data` switches
        to binning or sampling.
    :param seed: the sampler's seed, so "the same chart" is the same chart —
        a screenshot in a report and the screen it came from must not differ
        by a random draw.
    :raises SpecError: on an unknown plot kind, a non-positive bin count or a
        role that is neither continuous nor categorical.
    """

    x: Optional[str] = None
    y: Optional[str] = None
    colour: Optional[str] = None
    size: Optional[str] = None
    facet_row: Optional[str] = None
    facet_col: Optional[str] = None
    kind: Optional[str] = None
    roles: Mapping[str, str] = field(default_factory=dict)
    bins: int = 30
    shared_x: bool = True
    shared_y: bool = True
    point_budget: int = DEFAULT_POINT_BUDGET
    seed: int = 0

    def __post_init__(self) -> None:
        for channel in CHANNELS:
            value = getattr(self, channel)
            # "" and None both mean "empty zone"; normalising here is what
            # lets `if spec.x:` be the whole test everywhere else.
            object.__setattr__(self, channel,
                               str(value) if value else None)
        if self.kind is not None:
            kind = str(self.kind)
            if kind not in PLOT_KINDS:
                raise SpecError(
                    f"unknown plot kind {kind!r}; choose one of "
                    f"{', '.join(PLOT_KINDS)}, or None to infer it from the "
                    f"columns dropped")
            object.__setattr__(self, "kind", kind)
        roles = {}
        for name, role in dict(self.roles).items():
            if role not in (CONTINUOUS, CATEGORICAL):
                raise SpecError(
                    f"role override for {name!r} is {role!r}; it must be "
                    f"{CONTINUOUS!r} or {CATEGORICAL!r}")
            roles[str(name)] = role
        object.__setattr__(self, "roles", roles)
        if int(self.bins) < 1:
            raise SpecError(f"bins must be at least 1, not {self.bins}")
        object.__setattr__(self, "bins", int(self.bins))
        object.__setattr__(self, "point_budget", max(1, int(self.point_budget)))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "shared_x", bool(self.shared_x))
        object.__setattr__(self, "shared_y", bool(self.shared_y))

    # -- channels ------------------------------------------------------
    @property
    def channels(self) -> Dict[str, Optional[str]]:
        """``{channel: column or None}`` for all six, in :data:`CHANNELS` order."""
        return {channel: getattr(self, channel) for channel in CHANNELS}

    def column_for(self, channel: str) -> Optional[str]:
        """The column on ``channel``. :raises SpecError: on an unknown channel."""
        if channel not in CHANNELS:
            raise SpecError(
                f"unknown channel {channel!r}; the drop zones are "
                f"{', '.join(CHANNELS)}")
        return getattr(self, channel)

    def with_channel(self, channel: str, column: Optional[str]) -> "GraphSpec":
        """A copy with ``column`` on ``channel`` (``None`` empties the zone)."""
        if channel not in CHANNELS:
            raise SpecError(
                f"unknown channel {channel!r}; the drop zones are "
                f"{', '.join(CHANNELS)}")
        return replace(self, **{channel: (str(column) if column else None)})

    def with_kind(self, kind: Optional[str]) -> "GraphSpec":
        """A copy pinned to ``kind``, or back to inferring when ``None``."""
        return replace(self, kind=kind)

    def with_role(self, column: str, role: Optional[str]) -> "GraphSpec":
        """A copy treating ``column`` as ``role``; ``None`` restores the rule."""
        roles = dict(self.roles)
        if role is None:
            roles.pop(str(column), None)
        else:
            roles[str(column)] = role
        return replace(self, roles=roles)

    def used_columns(self) -> Tuple[str, ...]:
        """Every column named by a channel, de-duplicated, in channel order."""
        seen: Dict[str, None] = {}
        for channel in CHANNELS:
            column = getattr(self, channel)
            if column:
                seen.setdefault(column, None)
        return tuple(seen)

    @property
    def is_empty(self) -> bool:
        """Nothing on x and nothing on y: there is no chart to draw yet."""
        return not self.x and not self.y

    # -- kinds ---------------------------------------------------------
    def kinds_for(self, frame: pd.DataFrame) -> Dict[str, str]:
        """:func:`column_kinds` of ``frame`` with this spec's overrides applied."""
        kinds = column_kinds(frame)
        kinds.update({name: role for name, role in self.roles.items()
                      if name in frame.columns})
        return kinds

    def resolved_kind(self, kinds: Mapping[str, str]) -> str:
        """The kind that will actually be drawn — the pin, or the inference."""
        return self.kind or infer_kind(self, kinds)

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain JSON-able dict. Every field, always — a stable schema beats
        a compact one for something later screens read."""
        payload: Dict[str, Any] = {channel: getattr(self, channel)
                                   for channel in CHANNELS}
        payload.update({
            "kind": self.kind,
            "roles": dict(sorted(self.roles.items())),
            "bins": self.bins,
            "shared_x": self.shared_x,
            "shared_y": self.shared_y,
            "point_budget": self.point_budget,
            "seed": self.seed,
        })
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphSpec":
        """Rebuild from :meth:`to_dict`.

        Unknown keys are ignored and missing keys take their defaults, so a
        spec written by an older (or newer) build still opens. A spec that
        will not load is a chart the user cannot get back.
        """
        fields = set(CHANNELS) | {"kind", "roles", "bins", "shared_x",
                                  "shared_y", "point_budget", "seed"}
        known = {key: value for key, value in dict(payload).items()
                 if key in fields}
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GraphSpec":
        return cls.from_dict(json.loads(text))

    # -- for a caption --------------------------------------------------
    def describe(self, kinds: Optional[Mapping[str, str]] = None) -> str:
        """One human line, for the chart's caption and the window title."""
        kinds = dict(kinds or {})
        parts = [f"{channel.replace('_', ' ')}: {column}"
                 for channel, column in self.channels.items() if column]
        if not parts:
            return "nothing dropped yet"
        kind = self.resolved_kind(kinds)
        pinned = " (pinned)" if self.kind else ""
        return f"{kind}{pinned} · " + " · ".join(parts)


def infer_kind(spec: GraphSpec, kinds: Mapping[str, str]) -> str:
    """The plot type the dropped columns imply.

    Only x and y decide. Colour, size and the facets never change *what kind
    of chart this is* — dragging ``gene`` onto colour must not silently turn a
    scatter into something else, or the chart the user built stops being the
    chart they are looking at.

    ============================  ==================================
    x, y                          kind
    ============================  ==================================
    nothing                       :data:`EMPTY`
    one continuous                :data:`HISTOGRAM`
    one categorical               :data:`BAR` (counts per level)
    two continuous                :data:`SCATTER`
    one of each                   :data:`BOX`
    two categorical               :data:`HEATMAP` (a contingency count)
    ============================  ==================================

    :data:`VIOLIN` and :data:`LINE` are reachable only as an explicit
    override: a violin claims a density estimate the data may not support at
    small n, and joining points with a line asserts an order between them that
    a measurement table does not have.
    """
    x_kind = _axis_kind(spec.x, kinds)
    y_kind = _axis_kind(spec.y, kinds)
    present = [k for k in (x_kind, y_kind) if k is not None]
    if not present:
        return EMPTY
    if len(present) == 1:
        return HISTOGRAM if present[0] == CONTINUOUS else BAR
    if x_kind == CONTINUOUS and y_kind == CONTINUOUS:
        return SCATTER
    if x_kind == CATEGORICAL and y_kind == CATEGORICAL:
        return HEATMAP
    return BOX


# ---------------------------------------------------------------------------
# Faceting
# ---------------------------------------------------------------------------

_DIGIT_RUN = re.compile(r"(\d+)")


def _sort_key(text: str):
    """Numeric-aware ordering, so plate 2 sorts before plate 10.

    The whole-string ``float()`` this used to be only delivered that for a
    level that was a bare number. A level of ``"P10"`` is not, so it fell to
    the text branch and sorted *before* ``"P2"`` — and plate, well and
    condition levels almost always carry a prefix, so the docstring's own
    example was the case that did not work. Facet panels came out
    P1, P10, P11, P2.

    A level that IS a bare number keeps the old whole-string comparison, so
    ``-10`` still sorts before ``-5`` and numeric levels still come ahead of
    prefixed ones. Everything else is split on runs of digits and compared
    chunk by chunk, digits as numbers and the rest as text.
    """
    try:
        value = float(text)
    except (TypeError, ValueError):
        pass
    else:
        # NaN has no order, and one in a sort key makes the whole sort
        # arbitrary rather than wrong in one place. Treat it as text.
        if value == value:
            return ((0, value, ""),)
    key = []
    for index, chunk in enumerate(_DIGIT_RUN.split(str(text))):
        if index % 2:                       # split() alternates text, digits
            key.append((1, float(chunk), ""))
        else:
            key.append((2, 0.0, chunk))
    return tuple(key)


def _level_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """``column`` as strings, with NaN folded into :data:`MISSING_LEVEL`."""
    series = frame[column]
    text = series.astype(str)
    return text.mask(series.isna(), MISSING_LEVEL)


def _levels(frame: pd.DataFrame, column: str,
            max_levels: int) -> Tuple[Tuple[str, ...], int]:
    """The kept levels of ``column`` and how many were cut."""
    if column not in frame.columns:
        raise SpecError(
            f"facet column {column!r} is not in this table; it has "
            f"{len(frame.columns)} columns and none of them is that one")
    found = sorted({str(v) for v in _level_series(frame, column).unique()},
                   key=_sort_key)
    return tuple(found[:max_levels]), max(0, len(found) - max_levels)


@dataclass(frozen=True)
class FacetPanel:
    """One panel of the grid, and the rows that belong in it.

    ``index`` holds *positional* indices into the frame :func:`facet_grid` was
    given, not label indices: a measurement frame carries a duplicated or
    reset index often enough that positions are the only safe currency.
    """

    row: int
    col: int
    row_level: Optional[str]
    col_level: Optional[str]
    index: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.index))

    @property
    def is_empty(self) -> bool:
        """No rows. Still drawn — see :class:`FacetGrid`."""
        return self.n == 0

    def frame(self, source: pd.DataFrame) -> pd.DataFrame:
        """This panel's rows out of ``source``."""
        return source.iloc[self.index]

    def title(self) -> str:
        """The panel's own label, empty when the grid is not faceted."""
        parts = [p for p in (self.row_level, self.col_level) if p is not None]
        return " · ".join(parts)


@dataclass(frozen=True)
class FacetGrid:
    """The complete panel layout — including the combinations with no rows.

    An empty panel is **drawn empty**, never skipped. "Plate 3 / row H has no
    surviving cells" and "there is no plate 3 / row H" are different facts,
    and a grid that closes up the gaps tells the reader the second one when
    the first is true.

    :param hidden_rows: rows excluded because their facet level did not make
        the :data:`MAX_FACET_LEVELS` cut. Non-zero means the grid is not the
        whole table, and :attr:`notice` says so.
    """

    row_column: Optional[str]
    col_column: Optional[str]
    row_levels: Tuple[Optional[str], ...]
    col_levels: Tuple[Optional[str], ...]
    panels: Tuple[FacetPanel, ...]
    hidden_rows: int = 0
    notice: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.row_levels), len(self.col_levels)

    @property
    def n_panels(self) -> int:
        """Rows × columns — the count *including* empty panels."""
        return len(self.panels)

    @property
    def is_faceted(self) -> bool:
        return bool(self.row_column or self.col_column)

    def panel(self, row: int, col: int) -> FacetPanel:
        return self.panels[row * len(self.col_levels) + col]


def facet_grid(frame: pd.DataFrame, spec: GraphSpec, *,
               levels_source: Optional[pd.DataFrame] = None,
               max_levels: int = MAX_FACET_LEVELS,
               max_panels: int = MAX_PANELS) -> FacetGrid:
    """Split ``frame`` into the grid ``spec``'s facet channels describe.

    :param frame: the rows to place into panels (post-filter, post-sample).
    :param levels_source: where the *levels* come from, when that is not
        ``frame``. The renderer passes the pre-sample frame, so a level that
        exists in the population but drew no rows in the sample still gets its
        panel — drawn empty, which is the honest picture — instead of the grid
        silently changing shape with the sample.
    :param max_levels: per axis; beyond it levels are cut and counted.
    :param max_panels: hard ceiling on rows × columns.
    :returns: a :class:`FacetGrid` whose ``panels`` are the **full** cartesian
        product in row-major order.
    """
    source = frame if levels_source is None else levels_source
    notices = []

    def axis(column: Optional[str]) -> Tuple[Tuple[Optional[str], ...], int]:
        if not column:
            return (None,), 0
        levels, cut = _levels(source, column, max_levels)
        if not levels:
            # A facet column with no levels at all — everything filtered out,
            # or an all-NaN column. One panel, drawn empty. A zero-column grid
            # is not a figure matplotlib (or anyone) can draw, and "your
            # filter matches nothing" is an answer worth rendering.
            return (None,), 0
        if cut:
            notices.append(
                f"{column} has {len(levels) + cut} levels; the first "
                f"{len(levels)} are drawn")
        return levels, cut

    row_levels, _row_cut = axis(spec.facet_row)
    col_levels, _col_cut = axis(spec.facet_col)

    # Trim the *columns* axis first when the product is too big: a grid is
    # read down the page, so losing a column costs less than losing a row.
    while len(row_levels) * len(col_levels) > max_panels:
        if len(col_levels) >= len(row_levels) and len(col_levels) > 1:
            col_levels = col_levels[:-1]
        elif len(row_levels) > 1:
            row_levels = row_levels[:-1]
        else:  # pragma: no cover - unreachable while max_panels >= 1
            break
        notices.append(f"grid capped at {max_panels} panels")

    n = len(frame)
    # Only split on an axis that actually has levels: an axis that degenerated
    # to `(None,)` above is a single panel, and matching rows against a level
    # of ``None`` would put every one of them nowhere.
    row_live = bool(spec.facet_row) and row_levels != (None,)
    col_live = bool(spec.facet_col) and col_levels != (None,)
    row_keys = (_level_series(frame, spec.facet_row).to_numpy()
                if row_live and n else None)
    col_keys = (_level_series(frame, spec.facet_col).to_numpy()
                if col_live and n else None)

    positions = np.arange(n)
    kept = np.ones(n, dtype=bool)
    if row_keys is not None:
        kept &= np.isin(row_keys, np.asarray(row_levels, dtype=object))
    if col_keys is not None:
        kept &= np.isin(col_keys, np.asarray(col_levels, dtype=object))
    hidden = int((~kept).sum())

    panels = []
    for r, row_level in enumerate(row_levels):
        row_mask = kept if row_keys is None else (kept & (row_keys == row_level))
        for c, col_level in enumerate(col_levels):
            mask = (row_mask if col_keys is None
                    else (row_mask & (col_keys == col_level)))
            panels.append(FacetPanel(
                row=r, col=c, row_level=row_level, col_level=col_level,
                index=positions[mask]))

    if hidden:
        notices.append(f"{hidden:,} row(s) outside the drawn levels")
    return FacetGrid(
        row_column=spec.facet_row, col_column=spec.facet_col,
        row_levels=tuple(row_levels), col_levels=tuple(col_levels),
        panels=tuple(panels), hidden_rows=hidden,
        notice="; ".join(dict.fromkeys(notices)))


# ---------------------------------------------------------------------------
# Shared scales
# ---------------------------------------------------------------------------

def _numeric(frame: pd.DataFrame, column: Optional[str]) -> Optional[np.ndarray]:
    if not column or column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _limits(values: Optional[np.ndarray], pad: float = 0.05
            ) -> Optional[Tuple[float, float]]:
    """``(low, high)`` with a margin, or ``None`` when there is nothing finite.

    A degenerate column (one distinct value, or one row) is widened rather
    than returned flat: matplotlib silently expands a zero-width axis, and it
    does it differently per panel, which would break shared axes in exactly
    the case that is hardest to notice.
    """
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        span = abs(low) * 0.05 or 0.5
        return low - span, high + span
    margin = (high - low) * pad
    return low - margin, high + margin


@dataclass(frozen=True)
class Scales:
    """One set of limits and orders for **every** panel.

    Computed over the whole frame, never per panel: that is what "shared axes"
    means, and it is why the later trellis screen can reuse this instead of
    re-deriving it. When ``spec.shared_x`` is off, the renderer autoscales x
    per panel and :attr:`x_limits` is ignored — the field is still filled, so
    a caption can say what the shared limits *would* have been.

    :param count_limit: the tallest bar or bin across all panels, so a
        histogram grid's count axis is comparable too. Sharing the value axis
        of an aggregate is the same rule as sharing a data axis; forgetting it
        is the usual way a faceted histogram lies.
    """

    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = None
    x_levels: Optional[Tuple[str, ...]] = None
    y_levels: Optional[Tuple[str, ...]] = None
    x_edges: Optional[np.ndarray] = None
    y_edges: Optional[np.ndarray] = None
    colour_levels: Optional[Tuple[str, ...]] = None
    colour_limits: Optional[Tuple[float, float]] = None
    size_limits: Optional[Tuple[float, float]] = None
    count_limit: Optional[float] = None

    def x_positions(self) -> Optional[Dict[str, int]]:
        """Level → tick position, so a categorical x lines up across panels."""
        if self.x_levels is None:
            return None
        return {level: i for i, level in enumerate(self.x_levels)}

    def y_positions(self) -> Optional[Dict[str, int]]:
        if self.y_levels is None:
            return None
        return {level: i for i, level in enumerate(self.y_levels)}


def _category_levels(frame: pd.DataFrame, column: Optional[str],
                     limit: int = 60) -> Optional[Tuple[str, ...]]:
    if not column or column not in frame.columns:
        return None
    found = sorted({str(v) for v in _level_series(frame, column).unique()},
                   key=_sort_key)
    return tuple(found[:limit])


def value_axes(spec: GraphSpec, kinds: Mapping[str, str]
               ) -> Tuple[Optional[str], Optional[str]]:
    """Which column each *axis* actually carries, once the kind is known.

    Almost always ``(spec.x, spec.y)``. The exception is a lone column on Y
    that infers a histogram or a bar chart: those draw the column along the
    horizontal axis and counts up the vertical one, whichever zone it was
    dropped in. Without this the scales would be computed for an axis nothing
    is drawn on, panels would autoscale independently, and a faceted
    "histogram of Y" would quietly stop sharing its bins.
    """
    kind = spec.resolved_kind(kinds)
    if kind in (HISTOGRAM, BAR) and not spec.x and spec.y:
        return spec.y, None
    return spec.x, spec.y


def scales_for(frame: pd.DataFrame, spec: GraphSpec,
               kinds: Mapping[str, str],
               grid: Optional[FacetGrid] = None) -> Scales:
    """Limits, bin edges, level orders and colour levels shared by every panel.

    :param frame: the rows that will be drawn — post-filter and post-sample,
        so the limits bound what is actually on screen.
    :param grid: needed only for :attr:`Scales.count_limit`, which is the
        maximum over panels and therefore cannot be computed from the frame
        alone.
    """
    kind = spec.resolved_kind(kinds)
    x_column, y_column = value_axes(spec, kinds)
    x_kind = _axis_kind(x_column, kinds)
    y_kind = _axis_kind(y_column, kinds)

    x_values = _numeric(frame, x_column) if x_kind == CONTINUOUS else None
    y_values = _numeric(frame, y_column) if y_kind == CONTINUOUS else None
    x_limits = _limits(x_values)
    y_limits = _limits(y_values)
    x_levels = (_category_levels(frame, x_column)
                if x_kind == CATEGORICAL else None)
    y_levels = (_category_levels(frame, y_column)
                if y_kind == CATEGORICAL else None)

    x_edges = y_edges = None
    if kind in (HISTOGRAM, SCATTER, LINE) and x_limits is not None:
        x_edges = np.linspace(x_limits[0], x_limits[1], spec.bins + 1)
    if kind == SCATTER and y_limits is not None:
        y_edges = np.linspace(y_limits[0], y_limits[1], spec.bins + 1)

    colour_levels = None
    colour_limits = None
    if spec.colour:
        if _axis_kind(spec.colour, kinds) == CATEGORICAL:
            colour_levels = _category_levels(frame, spec.colour)
        else:
            colour_limits = _limits(_numeric(frame, spec.colour), pad=0.0)
    size_limits = (_limits(_numeric(frame, spec.size), pad=0.0)
                   if spec.size else None)

    count_limit = None
    if kind in (HISTOGRAM, BAR) and grid is not None:
        count_limit = _count_limit(frame, spec, grid, kind, x_edges, x_levels)

    return Scales(
        x_limits=x_limits, y_limits=y_limits,
        x_levels=x_levels, y_levels=y_levels,
        x_edges=x_edges, y_edges=y_edges,
        colour_levels=colour_levels, colour_limits=colour_limits,
        size_limits=size_limits, count_limit=count_limit)


def _count_limit(frame: pd.DataFrame, spec: GraphSpec, grid: FacetGrid,
                 kind: str, x_edges, x_levels) -> Optional[float]:
    """The tallest bar/bin across every panel, for a shared count axis."""
    column = spec.x or spec.y
    if not column or column not in frame.columns:
        return None
    tallest = 0.0
    for panel in grid.panels:
        if panel.is_empty:
            continue
        rows = panel.frame(frame)
        if kind == HISTOGRAM and x_edges is not None:
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size:
                counts, _ = np.histogram(values, bins=x_edges)
                tallest = max(tallest, float(counts.max()))
        elif kind == BAR:
            counts = _level_series(rows, column).value_counts()
            if len(counts):
                tallest = max(tallest, float(counts.max()))
    if tallest <= 0:
        return None
    return tallest * 1.08


# ---------------------------------------------------------------------------
# The large-data policy
# ---------------------------------------------------------------------------

#: Every row is an individual mark.
FULL = "full"
#: Every row is counted, but into a 2-D density raster rather than marks.
BINNED = "binned"
#: A seeded uniform subset is drawn; the rest are counted only in the notice.
SAMPLED = "sampled"


@dataclass(frozen=True)
class RenderData:
    """What the renderer should draw, and what it must say about it.

    :attr:`notice` is **not** optional decoration. It is the difference
    between a chart of a million cells and a chart of fifty thousand of them,
    and a screenshot that does not carry it is a result nobody can check.
    """

    frame: pd.DataFrame
    strategy: str
    n_total: int
    n_shown: int
    notice: str = ""

    @property
    def is_complete(self) -> bool:
        """Whether every row is represented — as a mark or in a bin."""
        return self.strategy in (FULL, BINNED)


def prepare_data(frame: pd.DataFrame, spec: GraphSpec,
                 kinds: Mapping[str, str]) -> RenderData:
    """Decide how many rows get drawn, and say so.

    See the module docstring for the policy. The short version: aggregates use
    everything, point plots use everything up to ``spec.point_budget``, and
    above that they either bin (nothing lost) or sample (said out loud).
    """
    kind = spec.resolved_kind(kinds)
    total = int(len(frame))
    if kind in AGGREGATE_KINDS or kind == EMPTY:
        return RenderData(frame=frame, strategy=FULL, n_total=total,
                          n_shown=total,
                          notice=(f"{total:,} rows" if total else "no rows"))
    if total <= spec.point_budget:
        return RenderData(frame=frame, strategy=FULL, n_total=total,
                          n_shown=total,
                          notice=(f"{total:,} rows" if total else "no rows"))

    # Above the budget. Binning keeps every row, so it is preferred — but it
    # can only draw what a raster can carry: a density, optionally shaded by
    # the mean of a continuous colour column. A categorical colour or a size
    # channel needs one mark per row, and for those the only honest option
    # left is a sample the chart admits to.
    per_point_encoding = bool(spec.size) or (
        bool(spec.colour) and _axis_kind(spec.colour, kinds) == CATEGORICAL)
    if kind == SCATTER and not per_point_encoding:
        return RenderData(
            frame=frame, strategy=BINNED, n_total=total, n_shown=total,
            notice=(f"{total:,} rows drawn as a {spec.bins}×{spec.bins} "
                    f"density — every row is counted"))

    budget = min(spec.point_budget, total)
    # Positional, seeded, and sorted back into the frame's own order — not
    # `DataFrame.sample`, whose result has to be re-sorted by *index*, which
    # is not the row order for a frame that arrived from a filter or a join.
    picked = np.sort(np.random.default_rng(spec.seed).choice(
        total, size=budget, replace=False))
    sample = frame.iloc[picked]
    reason = ("a categorical colour" if spec.colour and not spec.size
              else "a size channel" if spec.size and not spec.colour
              else "per-point colour and size" if spec.size and spec.colour
              else "a line")
    return RenderData(
        frame=sample, strategy=SAMPLED, n_total=total, n_shown=int(len(sample)),
        notice=(f"showing a random {len(sample):,} of {total:,} rows "
                f"({100.0 * len(sample) / total:.1f}%) — {reason} needs one "
                f"mark per row. Filters and selections still see all "
                f"{total:,}."))


def brush_mask(frame: pd.DataFrame, spec: GraphSpec, kinds: Mapping[str, str],
               x0: float, y0: float, x1: float, y1: float,
               scales: Optional[Scales] = None) -> np.ndarray:
    """Rows of ``frame`` inside the rectangle a user dragged on a panel.

    A brush is a *predicate*, not a hit test against drawn marks, which is why
    it stays exact when the panel was binned or sampled: the rectangle is
    evaluated against whatever frame it is handed, and the renderer hands it
    the unsampled one.

    A categorical axis is matched by the level under the swept tick positions,
    so brushing three boxes of a box plot selects those three groups.

    On a histogram or a bar chart the vertical axis is a *count*, not a
    variable, so only the horizontal sweep constrains anything — brushing
    across four bins selects the rows in those four bins, whatever height the
    drag happened to start at.
    """
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    keep = np.ones(len(frame), dtype=bool)
    if len(frame) == 0:
        return keep

    x_column, y_column = value_axes(spec, kinds)
    if spec.resolved_kind(kinds) in (HISTOGRAM, BAR):
        y_column = None
    for column, kind_of_axis, lo, hi, levels in (
            (x_column, _axis_kind(x_column, kinds), lo_x, hi_x,
             getattr(scales, "x_levels", None) if scales else None),
            (y_column, _axis_kind(y_column, kinds), lo_y, hi_y,
             getattr(scales, "y_levels", None) if scales else None)):
        if not column or column not in frame.columns:
            continue
        if kind_of_axis == CONTINUOUS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            keep &= np.isfinite(values) & (values >= lo) & (values <= hi)
        elif levels:
            swept = {level for i, level in enumerate(levels)
                     if lo - 0.5 <= i <= hi + 0.5}
            keep &= _level_series(frame, column).isin(swept).to_numpy()
    return keep
