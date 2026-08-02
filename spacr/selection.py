"""A shared selection and filter model, so the views can talk to each other.

spaCR ships four views over the same measurement table — the UMAP, the plate
heatmap, the database browser and the annotation grid — and until now none of
them knew the others existed. Lassoing a cluster in the UMAP told you nothing
about where those cells sat on the plate, and narrowing to high-count wells in
one view narrowed nothing anywhere else.

This module is the piece underneath that. It is deliberately **pure pandas and
numpy**, with no Qt anywhere:

* it can be tested without a display, which the Qt half cannot;
* the same filter can be applied headless in ``spacr-run`` or a notebook, so a
  selection made in the GUI is expressible as something reproducible;
* and the expensive part — evaluating a filter over a million-row table — is
  kept away from the event loop by construction.

:mod:`spacr.qt.linked_selection` wraps it in a ``QObject`` with signals.

Two ideas, kept separate on purpose
-----------------------------------

**A filter** narrows the population everyone is looking at: "wells with at
least 200 cells, plate 3 only". It is declarative, cheap to describe, and
survives being written to a settings file.

**A selection** is the subset the user has pointed at *inside* that
population: the cluster they lassoed. It is transient and arbitrary — there is
no predicate that describes it — so it is carried as explicit keys.

Conflating the two is the usual mistake. A filter can be re-applied to a
different table (another plate, a re-run) and still mean something; a
selection cannot, because it names individual objects.

Identity
--------

Objects are identified by :func:`object_keys` — the schema's own row key
(``plateID``, ``rowID``, ``columnID``, ``fieldID``, ``object_label``), joined
with the schema separator. That is the one identity every table in
``measurements.db`` already agrees on, so a key from the UMAP means the same
row in the plate view without a lookup table in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema

__all__ = [
    "FilterError",
    "RangeFilter",
    "CategoryFilter",
    "DataFilter",
    "Selection",
    "object_keys",
    "OBJECT_KEY_COLUMNS",
]


class FilterError(ValueError):
    """A filter that cannot be applied to the frame it was handed.

    Raised rather than silently dropped. A filter naming a column that is not
    there has almost always been carried over from a different table, and
    quietly ignoring it would narrow the population by less than the user
    asked for while the UI still showed the filter as active — which reads as
    "these are all the cells that match" when it is not.
    """


#: The columns that identify one measured object, in the order they are
#: joined. Taken from :mod:`spacr.schema` rather than written out here, so a
#: schema change moves this with it.
OBJECT_KEY_COLUMNS: Tuple[str, ...] = (
    schema.FIELD_KEY_COLUMNS + (schema.OBJECT_LABEL_KEY,)
)


def object_keys(df: pd.DataFrame,
                *, timelapse: bool = False) -> pd.Index:
    """Return one stable key per row of ``df``.

    :param df: any frame carrying the object key columns.
    :param timelapse: include the timepoint, so the same object at two frames
        is two keys rather than one. A timelapse table that leaves this False
        collapses every frame of an object onto a single key, which is the
        bug that has bitten this codebase repeatedly in the other direction
        (see ``schema.parse_prcfo``).
    :returns: a :class:`pandas.Index` of ``str``, aligned to ``df.index``.
    :raises FilterError: if any key column is missing.
    """
    cols = list(OBJECT_KEY_COLUMNS)
    if timelapse:
        # Insert the timepoint before the object label, matching the order
        # `ObjectTableSchema.row_key_columns(timelapse=True)` uses.
        cols = list(schema.TIMEPOINT_KEY_COLUMNS) + [schema.OBJECT_LABEL_KEY]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FilterError(
            f"cannot build object keys: {df.shape[0]}-row frame is missing "
            f"{missing}. Available columns include "
            f"{sorted(df.columns)[:8]}...")
    if df.empty:
        return pd.Index([], dtype=object)
    parts = [df[c].astype(str) for c in cols]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep=schema.KEY_SEPARATOR)
    return pd.Index(joined.to_numpy(), dtype=object)


@dataclass(frozen=True)
class RangeFilter:
    """Keep rows whose ``column`` lies within ``[low, high]``.

    ``None`` on either bound means unbounded on that side, which is what a
    slider dragged to its end should mean — not "exclude everything".

    NaN never passes. A measurement that could not be computed is not a
    measurement inside the range, and letting it through would put objects
    with no value into a population the user defined by value.
    """

    column: str
    low: Optional[float] = None
    high: Optional[float] = None

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise FilterError(
                f"range filter names column {self.column!r}, which this frame "
                f"does not have")
        values = pd.to_numeric(df[self.column], errors="coerce")
        keep = values.notna().to_numpy()
        if self.low is not None:
            keep &= (values >= self.low).to_numpy()
        if self.high is not None:
            keep &= (values <= self.high).to_numpy()
        return keep

    def describe(self) -> str:
        if self.low is None and self.high is None:
            return f"{self.column}: any"
        if self.low is None:
            return f"{self.column} ≤ {self.high:g}"
        if self.high is None:
            return f"{self.column} ≥ {self.low:g}"
        return f"{self.low:g} ≤ {self.column} ≤ {self.high:g}"


@dataclass(frozen=True)
class CategoryFilter:
    """Keep rows whose ``column`` is one of ``values``.

    An EMPTY ``values`` keeps nothing, and that is deliberate: unticking every
    box in a category list means "show me none of these", and quietly
    reinterpreting it as "show me all of them" would silently widen the
    population the user is looking at. The UI is expected to make an empty
    selection visible; this class will not paper over it.
    """

    column: str
    values: Tuple[Any, ...]

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise FilterError(
                f"category filter names column {self.column!r}, which this "
                f"frame does not have")
        wanted = {str(v) for v in self.values}
        return df[self.column].astype(str).isin(wanted).to_numpy()

    def describe(self) -> str:
        if not self.values:
            return f"{self.column}: none"
        shown = ", ".join(str(v) for v in self.values[:3])
        more = f" +{len(self.values) - 3}" if len(self.values) > 3 else ""
        return f"{self.column} ∈ {{{shown}{more}}}"


@dataclass
class DataFilter:
    """An AND of :class:`RangeFilter` and :class:`CategoryFilter` clauses.

    Declarative and re-appliable: the same filter means something on a
    re-run's table, which is what separates it from a selection.
    """

    clauses: list = field(default_factory=list)

    def add(self, clause) -> "DataFilter":
        """Add a clause, replacing any existing one on the same column.

        Replacing rather than appending is what makes a slider a slider: a
        widget that emits on every drag would otherwise stack a hundred
        near-identical range clauses and turn an O(1) filter into an O(n) one.
        """
        self.clauses = [c for c in self.clauses
                        if c.column != clause.column] + [clause]
        return self

    def remove(self, column: str) -> "DataFilter":
        """Drop the clause on ``column``, if any. Unknown columns are fine."""
        self.clauses = [c for c in self.clauses if c.column != column]
        return self

    def clear(self) -> "DataFilter":
        self.clauses = []
        return self

    @property
    def is_empty(self) -> bool:
        return not self.clauses

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask over ``df``'s rows.

        An empty filter keeps everything, which is the identity a "no filter"
        state should have.
        """
        keep = np.ones(len(df), dtype=bool)
        for clause in self.clauses:
            keep &= clause.mask(df)
        return keep

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` narrowed to the rows this filter keeps."""
        return df.loc[self.mask(df)]

    def describe(self) -> str:
        """One human line, for the header of whatever view is filtered.

        A filtered view that does not say it is filtered is how someone
        reports a result computed on a fifth of their data.
        """
        if not self.clauses:
            return "no filter"
        return " and ".join(c.describe() for c in self.clauses)


@dataclass
class Selection:
    """The keys the user has pointed at, plus the filter they sit inside.

    ``keys`` of ``None`` means "nothing selected", which is different from an
    empty index meaning "an explicit selection that happens to be empty" —
    a lasso around blank space. Views draw those two differently: the first
    is the resting state, the second is a result.
    """

    keys: Optional[pd.Index] = None
    source: str = ""

    @property
    def is_active(self) -> bool:
        return self.keys is not None

    def __len__(self) -> int:
        return 0 if self.keys is None else len(self.keys)

    def mask_for(self, df: pd.DataFrame,
                 *, timelapse: bool = False) -> np.ndarray:
        """Boolean mask of the rows of ``df`` that are in this selection.

        With no selection every row is in it — the resting state highlights
        nothing rather than everything, but a caller asking "which rows are
        selected" when nothing is gets the whole frame rather than an empty
        one, which is what keeps ``df[sel.mask_for(df)]`` meaning "the data
        the user is looking at".
        """
        if self.keys is None:
            return np.ones(len(df), dtype=bool)
        if df.empty:
            return np.zeros(0, dtype=bool)
        # `Index.isin` already returns an ndarray — unlike `Series.isin`,
        # which returns a Series. Calling `.to_numpy()` on it raises.
        return np.asarray(
            object_keys(df, timelapse=timelapse).isin(self.keys), dtype=bool)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source: str = "",
                   *, timelapse: bool = False) -> "Selection":
        """Select exactly the rows of ``df``."""
        return cls(keys=object_keys(df, timelapse=timelapse), source=source)

    @classmethod
    def none(cls) -> "Selection":
        return cls(keys=None, source="")
