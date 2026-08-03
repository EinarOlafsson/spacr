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

Three ideas, kept separate on purpose
-------------------------------------

**A filter** narrows the population everyone is looking at: "wells with at
least 200 cells, plate 3 only". It is declarative, cheap to describe, and
survives being written to a settings file.

**A selection** is the subset the user has pointed at *inside* that
population: the cluster they lassoed. It is transient and arbitrary — there is
no predicate that describes it — so it is carried as explicit keys.

Conflating the two is the usual mistake. A filter can be re-applied to a
different table (another plate, a re-run) and still mean something; a
selection cannot, because it names individual objects.

**A request** (:class:`ObjectRequest`) is neither: it is one act of routing.
"Open exactly these twelve objects, because they are the cells this model
called infected and the annotator called uninfected." A filter and a selection
are *state* that every view reads; a request is an *event* that travels once,
from the view that made it to the view that can show it. It carries the reason
with it, because a view showing twelve crops out of ninety thousand has to be
able to say why those twelve — otherwise they read as the whole dataset.

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
from types import MappingProxyType
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
    "ObjectRequest",
    "object_keys",
    "as_key_index",
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
    def from_keys(cls, keys: Any, source: str = "",
                  *, timelapse: bool = False) -> "Selection":
        """Select exactly ``keys`` — anything :func:`as_key_index` accepts.

        The counterpart to :meth:`from_frame` for a view that never had the
        frame: a scatter plot holding an array of keys, or a screen restoring
        a selection from a settings file.
        """
        return cls(keys=as_key_index(keys, timelapse=timelapse), source=source)

    @classmethod
    def none(cls) -> "Selection":
        return cls(keys=None, source="")


# ---------------------------------------------------------------------------
# "Show me exactly these objects"
# ---------------------------------------------------------------------------

def as_key_index(keys: Any, *, timelapse: bool = False) -> pd.Index:
    """Coerce whatever names a set of objects into object keys.

    :param keys: one of

        * a :class:`Selection` — its keys. A *resting* selection (``keys is
          None``) raises: "open nothing" is not a request, and turning it into
          an empty one would silently open an empty view.
        * a :class:`pandas.DataFrame` carrying :data:`OBJECT_KEY_COLUMNS` —
          :func:`object_keys` of it.
        * a single ``str`` — ONE key. Special-cased on purpose: a string is
          iterable, so falling through to the iterable branch would open one
          object per character, which is the sort of thing that produces a
          grid of 23 empty tiles rather than an error.
        * any other iterable of keys — an Index, a Series, a list — coerced
          with ``str``.

    :param timelapse: passed to :func:`object_keys` for the frame case, so a
        timelapse table keys each frame of an object separately.
    :returns: a :class:`pandas.Index` of ``str``.
    :raises TypeError: if ``keys`` is not something that can name objects.
    :raises ValueError: for a resting :class:`Selection`.

    Order is preserved and duplicates are dropped. Order is load-bearing: it
    is what carries "worst errors first" from a confusion-matrix cell through
    to whatever opens them, and a duplicated key would draw the same crop
    twice in the grid.
    """
    if isinstance(keys, Selection):
        if keys.keys is None:
            raise ValueError(
                "a resting Selection names no objects; there is nothing to "
                "open. Check `selection.is_active` first — 'nothing selected' "
                "and 'an empty selection' are different states.")
        values: Iterable[Any] = list(keys.keys)
    elif isinstance(keys, pd.DataFrame):
        values = list(object_keys(keys, timelapse=timelapse))
    elif isinstance(keys, str):
        values = [keys]
    else:
        try:
            values = list(keys)
        except TypeError:
            raise TypeError(
                f"cannot read object keys out of {type(keys).__name__}; pass "
                f"a DataFrame, a Selection, a key string, or an iterable of "
                f"key strings") from None
    seen = set()
    unique = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return pd.Index(unique, dtype=object)


@dataclass(frozen=True, eq=False)
class ObjectRequest:
    """One "open exactly these objects" act, on its way to whatever shows them.

    Built by the view that asked and handed, unchanged, to the opener
    registered for :attr:`kind` — see
    :func:`spacr.qt.linked_selection.open_objects`. Openers take this one
    object rather than a handful of arguments so the request can grow a field
    without breaking every registered opener.

    :param keys: anything :func:`as_key_index` accepts. Normalised to a
        :class:`pandas.Index` of :data:`OBJECT_KEY_COLUMNS` keys on
        construction, so an opener may assume ``request.keys`` is an Index of
        ``str``, in the caller's order, without duplicates.
    :param reason: why these objects, in the words the receiving view will
        put on screen ("predicted infected, annotated uninfected"). Required
        and non-blank: a grid showing twelve crops out of ninety thousand and
        not saying why is read as the whole dataset.
    :param source: the view that asked ("umap", "classifier_evaluation").
    :param kind: the destination. Left empty by callers who take the default;
        the router stamps the kind it actually dispatched to, so an opener
        registered for two kinds can tell which one it was reached through.
    :param timelapse: whether ``keys`` carry a timepoint, so the receiver
        resolves them against its own table the same way they were built.
    :param context: free-form extras for the destination — per-key scores to
        sort by, a column to annotate into. Copied and made read-only, so a
        caller mutating their dict cannot change a request already sent.
    :raises ValueError: on a blank ``reason``.

    An EMPTY request is legal. A confusion-matrix cell holding no errors is a
    real answer, and the destination saying "0 objects · no errors in this
    cell" is more use than an exception the caller has to catch.
    """

    keys: Any
    reason: str
    source: str = ""
    kind: str = ""
    timelapse: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "keys", as_key_index(self.keys, timelapse=self.timelapse))
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError(
                "an object request needs a reason: it becomes the line the "
                "receiving view shows above a subset, and without it a "
                "handful of crops reads as the whole population")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def __len__(self) -> int:
        return len(self.keys)

    def select_from(self, df: pd.DataFrame) -> pd.DataFrame:
        """The rows of ``df`` this request names, **in the request's order**.

        Not a mask, and not ``df``'s order: the caller's order is the answer
        for a request built worst-first, and a boolean mask would silently
        re-sort it back into table order. Keys with no row in ``df`` are
        dropped — a request can name objects a narrower table does not carry,
        and that is a smaller result, not an error.
        """
        if df.empty:
            return df.iloc[:0]
        rank_of = {key: i for i, key in enumerate(self.keys)}
        ranks = np.array(
            [rank_of.get(k, -1) for k in object_keys(df, timelapse=self.timelapse)],
            dtype=np.int64)
        keep = ranks >= 0
        kept = df.loc[keep]
        return kept.iloc[np.argsort(ranks[keep], kind="stable")]

    def as_selection(self) -> Selection:
        """The same objects as a :class:`Selection`, to publish as a highlight.

        Opening a subset and highlighting it everywhere else are two acts, and
        this is the seam between them: a receiver that wants the plate view to
        light up the crops it just opened publishes this, rather than the
        router doing it behind the user's back and wiping the lasso they made
        it with.
        """
        return Selection(keys=self.keys, source=self.source)

    def describe(self) -> str:
        """One line for the receiving view's header."""
        n = len(self.keys)
        noun = "object" if n == 1 else "objects"
        whence = f" (from {self.source})" if self.source else ""
        return f"{n} {noun} · {self.reason}{whence}"
