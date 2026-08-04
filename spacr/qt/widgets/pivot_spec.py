"""Tabulate — the pivot table's engine: rows, columns, aggregations, and n.

JMP's *Tabulate* is a drag-and-drop pivot: put ``plateID`` down the rows,
``gene`` across the columns, tick *mean* and *sd*, and read the table. It is
the thing people reach for before they plot anything, because a number they can
copy into a slide is often the whole answer.

Like :mod:`spacr.qt.widgets.graph_spec`, this half is **pure pandas and numpy
with no Qt** — testable without a display, usable from a notebook, and
available to whatever screen wants a summary table without inheriting a
drag-and-drop panel.

n is not optional
-----------------
Every cell of a :class:`PivotResult` carries its **n**, whether or not the user
ticked it, and the panel prints it. A mean over four objects and a mean over
four thousand are the same three digits on screen, and the difference between
them is the difference between a result and a coincidence. Making n opt-in
would make the most important number on the table the one nobody turns on.

For the same reason ``sd`` and ``sem`` of a single object are **NaN, not
zero**. One measurement has no spread; a zero there reads as "perfectly
reproducible", which is the opposite of what it means.

Empty is not zero
-----------------
A cell for a combination that has **no rows at all** is empty in every layer,
including n. ``0`` is a count of something, and a grid where "no objects were
measured in D7" looks identical to "some were measured and none survived the
filter" is a grid nobody can read. :attr:`PivotResult.present` keeps the two
apart, and :meth:`PivotResult.is_empty` is what the renderer asks.

The distinction that *is* real is kept: a cell with rows whose value column is
entirely NaN reads ``n = 0`` — objects were measured there and none of them
produced this measurement. That is a fact worth showing, and it is different
from the well not existing.

The full grid, for the same reason the facet grid is full
---------------------------------------------------------
Rows and columns are the **cartesian product** of their keys' levels, empty
combinations included, exactly as
:func:`spacr.qt.widgets.graph_spec.facet_grid` draws empty panels: a table
read by position needs its positions to line up, and one that closes up its
gaps tells the reader "there is no row H" when the truth is "row H was
measured and is empty". Above :data:`MAX_ROWS` × :data:`MAX_COLS` the product
is abandoned for the observed combinations only, and :attr:`PivotResult.notice`
says so — an unreadable table is not more honest than a smaller one that
admits what it left out.

The plate hierarchy
-------------------
``plateID`` / ``rowID`` / ``columnID`` / ``fieldID`` is what spaCR users
actually pivot on, so multiple keys per axis nest in the order they were
dropped and :data:`WELL_HIERARCHY` is offered as a preset. Nothing here is
special-cased for them; they are ordinary columns that happen to be the common
answer.

Feeding the chart
-----------------
:meth:`PivotResult.to_long` returns one row per cell with one column per
statistic, which is exactly the shape
:class:`~spacr.qt.widgets.graph_builder.GraphBuilderPanel` wants: ``x =
plateID``, ``y = mean``, ``size = n``. The pivot does not draw anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Both imported rather than re-derived. `_level_series` is how a NaN key
# becomes a visible level instead of a dropped row, and `_sort_key` is why
# plate 2 sorts before plate 10 — and the pivot's row order and the chart's
# facet order have to be the same order, or pivoting by plate and then
# plotting by plate would disagree about which plate is first.
from .graph_spec import MISSING_LEVEL, _level_series, _sort_key

__all__ = [
    "PivotError", "MISSING_LEVEL",
    "N", "MEAN", "MEDIAN", "SD", "SEM", "MIN", "MAX", "QUANTILE",
    "AGGREGATIONS", "AGGREGATION_LABELS", "COUNT_ONLY",
    "MAX_ROWS", "MAX_COLS", "MAX_CELLS", "LOW_N",
    "WELL_HIERARCHY", "FIELD_HIERARCHY",
    "PivotSpec", "PivotResult", "pivot", "format_value",
]


class PivotError(ValueError):
    """A pivot that cannot mean anything, with the reason in the message."""


# ---------------------------------------------------------------------------
# The aggregations
# ---------------------------------------------------------------------------

N = "n"
MEAN = "mean"
MEDIAN = "median"
SD = "sd"
SEM = "sem"
MIN = "min"
MAX = "max"
QUANTILE = "quantile"

#: Every aggregation, in the order a panel lists them.
AGGREGATIONS: Tuple[str, ...] = (N, MEAN, MEDIAN, SD, SEM, MIN, MAX, QUANTILE)

#: What each one is called, and the thing about it worth knowing.
AGGREGATION_LABELS: Dict[str, str] = {
    N: "n — objects with a value here. Always computed, always shown.",
    MEAN: "mean",
    MEDIAN: "median — what to read instead of the mean when n is small",
    SD: "sd — sample standard deviation (ddof 1); blank at n = 1",
    SEM: "sem — sd ÷ √n; blank at n = 1",
    MIN: "min",
    MAX: "max",
    QUANTILE: "quantile — linear interpolation, at the fraction below",
}

#: The pseudo value column used when no value column is chosen: the table is
#: then a contingency count and ``n`` counts *rows* rather than values.
COUNT_ONLY = ""

#: Displayed rows before the cartesian product is abandoned for the observed
#: combinations. A pivot longer than this is not read, it is exported.
MAX_ROWS = 2_000
#: Displayed columns. Far smaller than :data:`MAX_ROWS`: a table is read down
#: the page, and a hundred columns is already a horizontal scroll.
MAX_COLS = 200
#: Hard ceiling on cells, whatever the two multiply to.
MAX_CELLS = 100_000

#: At or below this n, a cell is worth flagging. Not a rule about
#: significance — just the point where a mean is one object's opinion.
LOW_N = 5

#: The two presets worth a button. Ordinary columns; they are only the common
#: answer to "what do I pivot on".
WELL_HIERARCHY: Tuple[str, ...] = ("plateID", "rowID", "columnID")
FIELD_HIERARCHY: Tuple[str, ...] = ("plateID", "rowID", "columnID", "fieldID")

#: What pandas calls each aggregation. ``n`` is ``count`` — the number of
#: *non-null* values, which is the n a mean was actually taken over, not the
#: number of rows that happened to be grouped together.
_PANDAS_NAMES: Dict[str, str] = {
    N: "count", MEAN: "mean", MEDIAN: "median", SD: "std", SEM: "sem",
    MIN: "min", MAX: "max",
}


def format_value(value: float, *, digits: int = 4) -> str:
    """One number for a table cell, or ``''`` for a missing one.

    Blank rather than ``nan``: a cell that reads ``nan`` is read as an error in
    the software, and a cell that reads ``0`` is worse. See the module
    docstring — an sd of a single object is genuinely nothing, and blank is
    what nothing looks like.
    """
    if value is None or not np.isfinite(value):
        return ""
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.{digits}g}"


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def _clean(names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """De-duplicated, blank-free, order preserved."""
    seen: Dict[str, None] = {}
    for name in names or ():
        if name is not None and str(name) != "":
            seen.setdefault(str(name), None)
    return tuple(seen)


@dataclass(frozen=True)
class PivotSpec:
    """What goes down the rows, across the columns, and into the cells.

    Frozen and JSON round-tripping like
    :class:`~spacr.qt.widgets.graph_spec.GraphSpec`, and for the same reason:
    a table is something a settings file, a report or a macro should be able to
    carry, and every edit returning a new spec makes "undo the last drag"
    trivial.

    :param rows: keys nesting down the rows, outermost first.
    :param cols: keys nesting across the columns, outermost first.
    :param values: the columns to aggregate. Empty means a contingency table:
        ``n`` counts rows and nothing else is computed.
    :param aggs: which aggregations to compute. :data:`N` is always added —
        see the module docstring.
    :param quantile: the fraction :data:`QUANTILE` reports, in ``[0, 1]``.
    :raises PivotError: on an unknown aggregation, a quantile outside ``[0,
        1]``, or a column used on two axes at once — at the point the spec is
        built rather than when the table comes out wrong.
    """

    rows: Tuple[str, ...] = ()
    cols: Tuple[str, ...] = ()
    values: Tuple[str, ...] = ()
    aggs: Tuple[str, ...] = (N, MEAN, SD)
    quantile: float = 0.75

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "values"):
            object.__setattr__(self, name, _clean(getattr(self, name)))
        clash = set(self.rows) & set(self.cols)
        if clash:
            raise PivotError(
                f"{', '.join(sorted(clash))} is on both the row and the "
                f"column axis. One column cannot nest inside itself; every "
                f"cell off the diagonal would be empty by construction.")
        aggs: List[str] = [N]
        for agg in self.aggs or ():
            if agg not in AGGREGATIONS:
                raise PivotError(
                    f"unknown aggregation {agg!r}; choose from "
                    f"{', '.join(AGGREGATIONS)}")
            if agg not in aggs:
                aggs.append(agg)
        object.__setattr__(self, "aggs", tuple(aggs))
        q = float(self.quantile)
        if not 0.0 <= q <= 1.0:
            raise PivotError(
                f"quantile is a fraction and must be in [0, 1], not "
                f"{self.quantile}")
        object.__setattr__(self, "quantile", q)

    # -- edits -----------------------------------------------------------
    def with_rows(self, rows: Sequence[str]) -> "PivotSpec":
        return replace(self, rows=tuple(rows))

    def with_cols(self, cols: Sequence[str]) -> "PivotSpec":
        return replace(self, cols=tuple(cols))

    def with_values(self, values: Sequence[str]) -> "PivotSpec":
        return replace(self, values=tuple(values))

    def with_aggs(self, aggs: Sequence[str]) -> "PivotSpec":
        return replace(self, aggs=tuple(aggs))

    @property
    def is_empty(self) -> bool:
        """Nothing on any axis and nothing to count: no table yet."""
        return not (self.rows or self.cols or self.values)

    @property
    def layers(self) -> Tuple[Tuple[str, str], ...]:
        """``(value, agg)`` pairs, in the order a cell stacks them.

        With no value column there is one layer, ``(COUNT_ONLY, N)`` — the
        contingency count.
        """
        if not self.values:
            return ((COUNT_ONLY, N),)
        return tuple((value, agg)
                     for value in self.values for agg in self.aggs)

    def used_columns(self) -> Tuple[str, ...]:
        return _clean(tuple(self.rows) + tuple(self.cols) + tuple(self.values))

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"rows": list(self.rows), "cols": list(self.cols),
                "values": list(self.values), "aggs": list(self.aggs),
                "quantile": self.quantile}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PivotSpec":
        fields = {"rows", "cols", "values", "aggs", "quantile"}
        known = {k: v for k, v in dict(payload).items() if k in fields}
        for key in ("rows", "cols", "values", "aggs"):
            if key in known:
                known[key] = tuple(known[key] or ())
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PivotSpec":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        parts = []
        if self.rows:
            parts.append("rows: " + " / ".join(self.rows))
        if self.cols:
            parts.append("columns: " + " / ".join(self.cols))
        parts.append("cells: " + (", ".join(
            f"{agg}({value})" if value else agg
            for value, agg in self.layers)))
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PivotResult:
    """A computed table: one 2-D array per ``(value, agg)`` layer, plus n.

    :param row_levels: one tuple of level strings per displayed row, aligned
        with :attr:`row_keys`.
    :param col_levels: likewise across.
    :param layers: ``{(value, agg): array}``, each ``(nrow, ncol)`` of float,
        NaN where the cell is empty **or** the statistic does not exist
        (``sd`` at n=1).
    :param sizes: rows of the source frame in each cell — the *row* count, not
        the value count. ``sizes == 0`` and ``present == False`` are the same
        thing here, and both are what makes a cell blank.
    :param present: whether the combination has any rows at all.
    :param hidden_rows: source rows whose keys fell outside the displayed
        levels. Non-zero means the table is not the whole frame.
    """

    row_keys: Tuple[str, ...]
    col_keys: Tuple[str, ...]
    row_levels: Tuple[Tuple[str, ...], ...]
    col_levels: Tuple[Tuple[str, ...], ...]
    layers: Mapping[Tuple[str, str], np.ndarray]
    sizes: np.ndarray
    present: np.ndarray
    spec: PivotSpec
    n_source_rows: int = 0
    hidden_rows: int = 0
    notice: str = ""

    # -- shape ------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.row_levels), len(self.col_levels)

    @property
    def n_cells(self) -> int:
        return len(self.row_levels) * len(self.col_levels)

    @property
    def layer_keys(self) -> Tuple[Tuple[str, str], ...]:
        return self.spec.layers

    # -- reading a cell ---------------------------------------------------
    def is_empty(self, row: int, col: int) -> bool:
        """No rows of the source frame landed here. Renders blank, not 0."""
        return not bool(self.present[row, col])

    def value_at(self, value: str, agg: str, row: int, col: int) -> float:
        """One statistic, or NaN when it is empty or does not exist."""
        try:
            layer = self.layers[(value, agg)]
        except KeyError:
            raise PivotError(
                f"this table has no {agg}({value or 'rows'}) layer; it has "
                f"{', '.join(f'{a}({v})' for v, a in self.layer_keys)}"
            ) from None
        return float(layer[row, col])

    def n_at(self, value: str, row: int, col: int) -> Optional[int]:
        """The n behind a cell's statistics, or ``None`` when it is empty.

        ``None`` and ``0`` are different: ``None`` is "nothing was measured
        in this combination", ``0`` is "objects were measured and none of them
        has a value for this feature".
        """
        if self.is_empty(row, col):
            return None
        count = self.layers[(value, N)][row, col]
        return None if not np.isfinite(count) else int(count)

    def row_label(self, row: int, sep: str = " · ") -> str:
        return sep.join(self.row_levels[row])

    def col_label(self, col: int, sep: str = " · ") -> str:
        return sep.join(self.col_levels[col])

    def low_n_cells(self, threshold: int = LOW_N) -> int:
        """Non-empty cells whose n is at or below ``threshold``."""
        total = 0
        for value in (self.spec.values or (COUNT_ONLY,)):
            counts = self.layers[(value, N)]
            with np.errstate(invalid="ignore"):
                total += int(np.sum(np.isfinite(counts)
                                    & (counts <= threshold)
                                    & self.present))
        return total

    def n_range(self) -> Optional[Tuple[int, int]]:
        """``(smallest, largest)`` n over the non-empty cells, or ``None``.

        The n behind the *statistics* — values, not rows — because that is the
        number a mean was taken over. They differ whenever a value column has
        NaN in it, and the smaller one is the one that matters.
        """
        value = (self.spec.values or (COUNT_ONLY,))[0]
        counts = self.layers[(value, N)][self.present]
        counts = counts[np.isfinite(counts)]
        if not counts.size:
            return None
        return int(counts.min()), int(counts.max())

    def summary(self) -> str:
        """One line under the table: shape, n range, and what was left out."""
        rows, cols = self.shape
        filled = int(self.present.sum())
        parts = [f"{rows:,} × {cols:,} = {self.n_cells:,} cells, "
                 f"{filled:,} with data"]
        span = self.n_range()
        if span is not None:
            parts.append(f"n per cell {span[0]:,}–{span[1]:,}")
        low = self.low_n_cells()
        if low:
            parts.append(f"{low:,} cell(s) at n ≤ {LOW_N}")
        if self.hidden_rows:
            parts.append(f"{self.hidden_rows:,} source row(s) outside the "
                         f"shown levels")
        if self.notice:
            parts.append(self.notice)
        return " · ".join(parts)

    # -- frames -----------------------------------------------------------
    def to_long(self) -> pd.DataFrame:
        """One row per **non-empty** cell; one column per statistic.

        The frame to hand the Graph Builder: ``x = plateID``, ``y = mean``,
        ``size = n``, ``colour = gene``. Empty cells are omitted rather than
        carried as NaN rows — a scatter of nothing is a mark at the origin
        waiting to happen, and :attr:`present` is where "which combinations
        were empty" lives.

        With more than one value column, ``value_column`` names which one the
        row is about, so the frame stays tidy instead of growing a column per
        (value × agg) pair.
        """
        rows: List[Dict[str, Any]] = []
        values = self.spec.values or (COUNT_ONLY,)
        for r in range(len(self.row_levels)):
            for c in range(len(self.col_levels)):
                if self.is_empty(r, c):
                    continue
                base: Dict[str, Any] = {}
                for key, level in zip(self.row_keys, self.row_levels[r]):
                    base[key] = level
                for key, level in zip(self.col_keys, self.col_levels[c]):
                    base[key] = level
                base["cell_rows"] = int(self.sizes[r, c])
                for value in values:
                    record = dict(base)
                    record["value_column"] = value or "rows"
                    for agg in (self.spec.aggs if self.spec.values else (N,)):
                        record[agg] = float(self.layers[(value, agg)][r, c])
                    rows.append(record)
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        """The spreadsheet shape: row keys as leading columns, one column per
        ``(column level × value × agg)``.

        What the CSV export writes and what a user pastes into a slide. The
        multi-level column header is flattened into one readable string because
        a CSV has one header row, and a reader who has to reassemble three of
        them has been handed a puzzle rather than a table.
        """
        data: Dict[str, Any] = {}
        for i, key in enumerate(self.row_keys):
            data[key] = [levels[i] for levels in self.row_levels]
        if not self.row_keys:
            data["rows"] = ["all"] * len(self.row_levels)
        for c, col_levels in enumerate(self.col_levels):
            prefix = " · ".join(col_levels)
            for value, agg in self.layer_keys:
                name = f"{agg}({value})" if value else "n"
                header = f"{prefix} · {name}" if prefix else name
                data[header] = [self.layers[(value, agg)][r, c]
                                for r in range(len(self.row_levels))]
        return pd.DataFrame(data)

    def to_csv(self, path: str) -> str:
        """Write :meth:`to_frame` to ``path``. Returns the path."""
        self.to_frame().to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _levels_of(frame: pd.DataFrame, key: str) -> Tuple[str, ...]:
    if key not in frame.columns:
        raise PivotError(
            f"{key!r} is not a column of this table; it has "
            f"{len(frame.columns)} columns and none of them is that one")
    labels = _level_series(frame, key)
    return tuple(sorted({str(v) for v in labels.unique()}, key=_sort_key))


def _axis_levels(frame: pd.DataFrame, keys: Sequence[str],
                 observed: Sequence[Tuple[str, ...]], limit: int,
                 notices: List[str], what: str
                 ) -> Tuple[Tuple[Tuple[str, ...], ...], int]:
    """The displayed combinations for one axis, and how many were cut.

    The cartesian product while it fits, the observed combinations when it does
    not, truncated when even those do not — every step named in ``notices``,
    because a table that quietly stopped being the whole table is the one
    failure this module exists to prevent.
    """
    if not keys:
        return ((),), 0
    per_key = [_levels_of(frame, key) for key in keys]
    total = 1
    for levels in per_key:
        total *= max(1, len(levels))
    if total <= limit:
        combos = tuple(product(*per_key))
    else:
        combos = tuple(sorted(
            dict.fromkeys(observed),
            key=lambda combo: tuple(_sort_key(part) for part in combo)))
        notices.append(
            f"{total:,} {what} combinations would be the full grid; only the "
            f"{len(combos):,} that occur are shown")
    cut = 0
    if len(combos) > limit:
        cut = len(combos) - limit
        combos = combos[:limit]
        notices.append(f"{what} capped at {limit:,}; {cut:,} not shown")
    return combos, cut


def pivot(frame: pd.DataFrame, spec: Optional[PivotSpec] = None) -> PivotResult:
    """Compute the table ``spec`` describes over ``frame``.

    The policy is in the module docstring; the short version is that every cell
    carries its n, an empty combination is empty rather than zero, ``sd`` and
    ``sem`` are ``ddof=1`` and therefore blank at n=1, and the grid is the full
    cartesian product until that stops being readable.

    :raises PivotError: for a spec that cannot describe a table over this
        frame, with the reason in the message.
    """
    spec = spec or PivotSpec()
    notices: List[str] = []
    n_source = int(len(frame))

    for value in spec.values:
        if value not in frame.columns:
            raise PivotError(
                f"value column {value!r} is not in this table. Drop a column "
                f"onto the values well that exists here, or reload the table.")

    keys = tuple(spec.rows) + tuple(spec.cols)
    n_rows_keys = len(spec.rows)
    if keys:
        labels = pd.DataFrame(
            {f"__k{i}": _level_series(frame, key).astype(str).to_numpy()
             for i, key in enumerate(keys)})
    else:
        labels = pd.DataFrame(index=pd.RangeIndex(n_source))
    key_columns = list(labels.columns)

    observed = [tuple(row) for row in labels.to_numpy()] if key_columns else []
    row_observed = [combo[:n_rows_keys] for combo in observed]
    col_observed = [combo[n_rows_keys:] for combo in observed]

    row_levels, _row_cut = _axis_levels(
        frame, spec.rows, row_observed, MAX_ROWS, notices, "row")
    col_levels, _col_cut = _axis_levels(
        frame, spec.cols, col_observed, MAX_COLS, notices, "column")
    while len(row_levels) * len(col_levels) > MAX_CELLS and len(row_levels) > 1:
        # Trim rows, not columns: a column removed loses a whole series, a row
        # removed loses one group, and the table is read down the page.
        row_levels = row_levels[:len(row_levels) - 1]
        notices.append(f"grid capped at {MAX_CELLS:,} cells")

    row_at = {combo: i for i, combo in enumerate(row_levels)}
    col_at = {combo: i for i, combo in enumerate(col_levels)}
    shape = (len(row_levels), len(col_levels))

    sizes = np.zeros(shape, dtype=np.int64)
    present = np.zeros(shape, dtype=bool)
    layers: Dict[Tuple[str, str], np.ndarray] = {
        key: np.full(shape, np.nan, dtype=float) for key in spec.layers}

    work = labels.copy()
    for value in spec.values:
        numeric = pd.to_numeric(frame[value], errors="coerce")
        if n_source and not numeric.notna().any() and frame[value].notna().any():
            # Text dropped on the values well. Every statistic would come out
            # blank and every n zero, which reads as "no data" rather than as
            # "you cannot average a gene name" — so it is said instead.
            notices.append(
                f"{value!r} is not numeric, so there is nothing to aggregate; "
                f"put it on rows or columns instead")
        work[value] = numeric.to_numpy()

    if key_columns:
        grouped = work.groupby(key_columns, dropna=False, sort=False,
                               observed=True)
    else:
        # No keys at all: one cell, the whole frame. `groupby` on a constant
        # is the same computation with the same code path below.
        work["__all"] = 0
        grouped = work.groupby(["__all"], dropna=False, sort=False,
                               observed=True)

    group_sizes = grouped.size()
    group_index = group_sizes.index
    size_values = group_sizes.to_numpy()
    # One numpy column per (value, agg), positionally aligned to the group
    # index. Reindexed onto it rather than trusted to match, and read
    # positionally rather than through `.loc` — a label lookup per group per
    # value turns a 50 000-group pivot into a visible pause.
    columns: Dict[Tuple[str, str], np.ndarray] = {}
    for value in spec.values:
        wanted = [_PANDAS_NAMES[a] for a in spec.aggs if a in _PANDAS_NAMES]
        table = grouped[value].agg(wanted)
        if isinstance(table, pd.Series):  # pragma: no cover - `n` is always in
            table = table.to_frame(name=wanted[0])
        if QUANTILE in spec.aggs:
            table = table.assign(
                **{"__q": grouped[value].quantile(spec.quantile)})
        table = table.reindex(group_index)
        for agg in spec.aggs:
            name = "__q" if agg == QUANTILE else _PANDAS_NAMES[agg]
            columns[(value, agg)] = table[name].to_numpy(dtype=float)

    hidden = 0
    for position, group_key in enumerate(group_index):
        size = int(size_values[position])
        if not key_columns:
            r = c = 0
        else:
            combo = (tuple(str(part) for part in group_key)
                     if isinstance(group_key, tuple) else (str(group_key),))
            r = row_at.get(combo[:n_rows_keys], -1)
            c = col_at.get(combo[n_rows_keys:], -1)
            if r < 0 or c < 0:
                hidden += size
                continue
        sizes[r, c] = size
        present[r, c] = True
        if not spec.values:
            layers[(COUNT_ONLY, N)][r, c] = float(size)
            continue
        for key, values_array in columns.items():
            layers[key][r, c] = values_array[position]

    if hidden:
        notices.append(f"{hidden:,} source row(s) outside the shown levels")

    return PivotResult(
        row_keys=tuple(spec.rows), col_keys=tuple(spec.cols),
        row_levels=row_levels, col_levels=col_levels, layers=layers,
        sizes=sizes, present=present, spec=spec, n_source_rows=n_source,
        hidden_rows=hidden, notice="; ".join(dict.fromkeys(notices)))
