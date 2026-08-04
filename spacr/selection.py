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

**The object type is part of that identity.** It did not used to be, and the
collapse was real: object tables are one type per table, so a nucleus
labelled 1 and a pathogen labelled 1 in the same field composed to the
*identical* key. A cell's own children are exactly the objects most likely to
collide, which is where object linking is most useful — four objects opened
as three crops, and which one you got depended on the row order of
``png_list`` (:func:`spacr.active_learning.crops_for_object_keys` keeps the
first). The type now goes into the object component of the key, exactly as
:func:`spacr.schema.object_id` writes it::

    plate1_r1_c1_f1_7          # type not stated  (what spaCR always wrote)
    plate1_r1_c1_f1_nucleus7   # type stated
    plate1_r1_c1_f1_pathogen7  # a different object, and now a different key

A frame states its type either by carrying :data:`OBJECT_TYPE_COLUMN` or by
the reader passing ``object_type=`` — readers know which table they read, and
:func:`with_object_type` is the one line that puts it on the frame.

Two rules keep the change from breaking anything already written:

*Untyped keys do not move.* The untyped form is byte for byte what it was, so
every key in every stored selection, exported ``.h5ad`` and prediction bundle
still composes and parses as it did. There is nothing on disk to migrate.

*An untyped key is LESS SPECIFIC, not wrong.* ``plate1_r1_c1_f1_7`` means
"the object labelled 7 in that field", which is exactly what it always meant
— so it matches that object whatever its type, rather than silently matching
one of them. Symmetrically a typed key still matches a row that has not said
what it is. :meth:`Selection.mask_for` and :meth:`ObjectRequest.select_from`
both apply that rule, and it is the whole migration: an old key is read as
the thing it always said, never re-read as something narrower.
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
    "untyped_object_keys",
    "with_object_type",
    "key_object_type",
    "untyped_object_key",
    "match_keys",
    "as_key_index",
    "OBJECT_KEY_COLUMNS",
    "OBJECT_TYPE_COLUMN",
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
#:
#: :data:`OBJECT_TYPE_COLUMN` is deliberately **not** in here. These are the
#: columns a frame must have before it can be keyed at all, and every object
#: table has them; the type is a refinement a frame may or may not carry, and
#: adding it to this tuple would turn every ``all(c in frame.columns for c in
#: OBJECT_KEY_COLUMNS)`` check in the GUI — the check that decides whether a
#: view can join the shared selection — from True to False on every table
#: spaCR has ever written.
OBJECT_KEY_COLUMNS: Tuple[str, ...] = (
    schema.FIELD_KEY_COLUMNS + (schema.OBJECT_LABEL_KEY,)
)

#: The optional column naming which object table a row came from.
#:
#: Present, it makes the key say which of a cell's children an object is.
#: Absent, keys are untyped and mean exactly what they always meant. See the
#: module docstring; :func:`with_object_type` is how a reader puts it on.
OBJECT_TYPE_COLUMN: str = schema.OBJECT_TYPE_KEY

#: A reversible escape for the two characters that would otherwise make a key
#: unsplittable or ambiguous.
#:
#: ``schema._check_plate`` refuses a separator in a plate id for exactly this
#: reason, and ``object_keys`` had no equivalent: ``("p_x", "r1", "c1", "f1",
#: 1)`` and ``("p", "x_r1", "c1", "f1", 1)`` both composed to
#: ``"p_x_r1_c1_f1_1"``. Two distinct objects, one key, and every view that
#: resolves a key back to a row shows or annotates the wrong object.
#:
#: Escaping rather than refusing, because a key is built from data that is
#: already on disk: refusing would take a view that shows the wrong object and
#: turn it into a view that raises, and the user cannot go back and rename the
#: plate in a database they already have. ``%`` has to be escaped too or the
#: escape is not reversible — a plate literally named ``p%5Fx`` would collide
#: with a plate named ``p_x``.
_KEY_ESCAPES: Tuple[Tuple[str, str], ...] = (
    ("%", "%25"),
    (schema.KEY_SEPARATOR, "%5F"),
)


def with_object_type(df: pd.DataFrame, object_type: Any) -> pd.DataFrame:
    """Return a copy of ``df`` stamped with the object table it came from.

    The one line a reader adds after loading a table, so every key built from
    the frame afterwards says which of a cell's children it names. Readers are
    where this belongs: a frame does not know what it is, but whatever ran
    ``SELECT * FROM nucleus`` does.

    An ``object_type`` spaCR does not key objects by — ``png_list``, a
    summary, a user's own table — is a **no-op**, not an error. Such a table's
    rows are keyed the way they always were, which is the correct answer for
    something that is not one of the four analysis compartments.

    :param df: any frame.
    :param object_type: an object table name, or ``None``.
    :returns: ``df`` unchanged, or a copy carrying :data:`OBJECT_TYPE_COLUMN`.
    """
    if not schema.is_object_type(object_type):
        return df
    out = df.copy()
    out[OBJECT_TYPE_COLUMN] = str(object_type).strip().lower()
    return out


def _escape_component(values: pd.Series) -> pd.Series:
    """Percent-escape ``%`` and the key separator, in that order."""
    out = values
    for character, escape in _KEY_ESCAPES:
        out = out.str.replace(character, escape, regex=False)
    return out


def _object_prefixes(df: pd.DataFrame, object_type: Any) -> Optional[pd.Series]:
    """The per-row type prefix, or ``None`` when the frame states no type.

    ``None`` rather than a column of ``''`` so the caller can take the fast
    path — an untyped frame is the overwhelmingly common one and must not pay
    for a feature it is not using.
    """
    if object_type is not None:
        prefix = schema.object_type_prefix(object_type)
        return pd.Series(prefix, index=df.index, dtype=object)
    if OBJECT_TYPE_COLUMN not in df.columns:
        return None
    raw = df[OBJECT_TYPE_COLUMN].astype(str).str.strip().str.lower()
    blank = raw.isin(("", "nan", "none", "null")) | df[OBJECT_TYPE_COLUMN].isna()
    stated = raw[~blank]
    if stated.empty:
        return None
    # Validate the vocabulary once per call rather than once per row: an
    # object table has a handful of distinct types and tens of millions of
    # rows, and `object_type_prefix` raising per row would be the slow part
    # of the only function on the lasso's hot path.
    for value in stated.unique():
        schema.object_type_prefix(value)
    return raw.where(~blank, "")


def _key_columns(timelapse: bool) -> list:
    cols = list(OBJECT_KEY_COLUMNS)
    if timelapse:
        # Insert the timepoint before the object label, matching the order
        # `ObjectTableSchema.row_key_columns(timelapse=True)` uses.
        cols = list(schema.TIMEPOINT_KEY_COLUMNS) + [schema.OBJECT_LABEL_KEY]
    return cols


def _compose(df: pd.DataFrame, cols: list,
             prefixes: Optional[pd.Series]) -> pd.Index:
    """Join the key columns, escaping only if a component needs it."""
    parts = [df[c].astype(str) for c in cols]
    if prefixes is not None:
        parts[-1] = prefixes.astype(str).str.cat(parts[-1])
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep=schema.KEY_SEPARATOR)
    # One pass over the composed key tells us whether any component smuggled
    # a separator in: a well-formed key has exactly one per join. Checking
    # here rather than per column keeps the common case — nothing to escape,
    # and the key is byte for byte what it always was — at two extra passes
    # instead of ten.
    expected = len(cols) - 1
    needs_escape = bool(
        (joined.str.count(schema.KEY_SEPARATOR) != expected).any()
        or joined.str.contains("%", regex=False).any())
    if needs_escape:
        parts = [_escape_component(df[c].astype(str)) for c in cols]
        if prefixes is not None:
            parts[-1] = prefixes.astype(str).str.cat(parts[-1])
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.str.cat(p, sep=schema.KEY_SEPARATOR)
    return pd.Index(joined.to_numpy(), dtype=object)


def object_keys(df: pd.DataFrame, *, timelapse: bool = False,
                object_type: Any = None) -> pd.Index:
    """Return one stable key per row of ``df``.

    :param df: any frame carrying the object key columns.
    :param timelapse: include the timepoint, so the same object at two frames
        is two keys rather than one. A timelapse table that leaves this False
        collapses every frame of an object onto a single key, which is the
        bug that has bitten this codebase repeatedly in the other direction
        (see ``schema.parse_prcfo``).
    :param object_type: the object table these rows came from. Overrides
        :data:`OBJECT_TYPE_COLUMN` when the frame also carries one. ``None``
        means "read it off the frame, and leave the keys untyped if it does
        not say" — *not* "these are cells".
    :returns: a :class:`pandas.Index` of ``str``, aligned to ``df.index``.
    :raises FilterError: if any key column is missing.
    :raises spacr.schema.KeyParseError: if a stated type is not one spaCR
        keys objects by.

    A typed key is exactly the ``prcfo`` :func:`spacr.schema.compose_prcfo`
    writes for the same object, so the two identities converge as soon as the
    type is known. They differ only in the untyped case, where this joins the
    label bare and ``prcfo`` writes ``'o7'`` — kept that way on purpose, since
    changing it would move every key that already exists.
    """
    cols = _key_columns(timelapse)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FilterError(
            f"cannot build object keys: {df.shape[0]}-row frame is missing "
            f"{missing}. Available columns include "
            f"{sorted(df.columns)[:8]}...")
    if df.empty:
        return pd.Index([], dtype=object)
    return _compose(df, cols, _object_prefixes(df, object_type))


def untyped_object_keys(df: pd.DataFrame,
                        *, timelapse: bool = False) -> pd.Index:
    """The keys ``df`` would have had before object types existed.

    Not a legacy shim: it is the *less specific* name for the same rows, and
    it is what makes an old key go on meaning what it always meant. A key
    naming no type says "the object labelled 7 in that field" and has to match
    that object whatever its type — see :meth:`Selection.mask_for`.
    """
    cols = _key_columns(timelapse)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FilterError(
            f"cannot build object keys: {df.shape[0]}-row frame is missing "
            f"{missing}. Available columns include "
            f"{sorted(df.columns)[:8]}...")
    if df.empty:
        return pd.Index([], dtype=object)
    return _compose(df, cols, None)


def _split_key(key: Any) -> Optional[Tuple[str, Optional[str], str]]:
    """``(everything before the object, type or None, label)``, or ``None``.

    ``None`` for anything that is not an object key. That matters more than it
    looks: a crop path and a file name travel through the same routing
    contract (:func:`spacr.active_learning.crops_for_object_keys` resolves
    all three), and ``'…/plate1_r1_c1_f1_o7.png'`` would otherwise split into
    an object labelled ``'7.png'`` and be rewritten out from under the caller
    that was relying on it to name a file.

    The label is required to be **digits**, which is what
    :func:`spacr.schema.validate_object_table_frame` already enforces on
    ``object_label``. That is the check that tells a key from a path.
    """
    text = str(key)
    head, separator, token = text.rpartition(schema.KEY_SEPARATOR)
    if not separator:
        return None
    object_type, label = schema.split_object_id(token, require_prefix=False)
    if not label or not label.isdigit():
        return None
    return (head + separator, object_type, label)


def key_object_type(key: Any) -> Optional[str]:
    """The object type ``key`` states, or ``None`` when it states none.

    ``None`` means *not stated*. It does not mean "cell", and nothing here
    will ever guess: every key spaCR wrote before today is untyped, so a
    default would put a type on the whole world's existing data.
    """
    split = _split_key(key)
    return None if split is None else split[1]


def untyped_object_key(key: Any) -> str:
    """``key`` with the object type taken back off, for a looser comparison.

    ``'p_r1_c1_f1_nucleus7'`` → ``'p_r1_c1_f1_7'``, and a key that already
    states no type is returned unchanged. A ``prcfo`` reduces the same way
    (``'p_r1_c1_f1_o7'`` → ``'p_r1_c1_f1_7'``), which is what lets a key
    copied out of a crop table match a key built from a measurement table.

    Anything this cannot read as an object key — a crop path, a file name — is
    returned untouched rather than mangled: those travel through the same
    routing contract and must survive it.
    """
    split = _split_key(key)
    if split is None:
        return str(key)
    head, _object_type, label = split
    return head + label


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


def _match(typed_rows: pd.Index, untyped_rows: pd.Index,
           wanted: Iterable[Any]) -> np.ndarray:
    """Which rows ``wanted`` names, matching types by specificity.

    The one answer to "does this key name this row". See
    :meth:`Selection.mask_for` for the rule and why it is that rule; the
    reason it lives in one function is that :meth:`ObjectRequest.select_from`,
    :func:`match_keys` and every linked view need the *same* answer. Two
    implementations of this question is how a selection and the crops it
    opens come to disagree, and that is invisible until somebody counts tiles.
    """
    keys = [str(k) for k in wanted]
    # `Index.isin` already returns an ndarray — unlike `Series.isin`, which
    # returns a Series. Calling `.to_numpy()` on it raises.
    mask = np.asarray(typed_rows.isin(keys), dtype=bool)
    if not keys or mask.all():
        return mask

    loose = [k for k in keys if key_object_type(k) is None]
    narrowed = {untyped_object_key(k) for k in keys
                if key_object_type(k) is not None}
    if loose:
        # A key naming no type names the object whatever its type.
        mask |= np.asarray(untyped_rows.isin(loose), dtype=bool)
    if narrowed:
        # A typed key still names a row that has not said what it is — but
        # only such a row. Without the `row_untyped` guard a selection of
        # `nucleus1` would light up `pathogen1`, which is the collapse
        # rebuilt one level up.
        row_untyped = np.asarray(typed_rows) == np.asarray(untyped_rows)
        mask |= (row_untyped
                 & np.asarray(untyped_rows.isin(narrowed), dtype=bool))
    return mask


def match_keys(keys: Any, wanted: Iterable[Any]) -> np.ndarray:
    """Boolean mask over ``keys`` of the ones ``wanted`` names.

    The key-to-key form of :meth:`Selection.mask_for`, for a view that holds
    an array of keys rather than the frame they came from — a scatter plot, a
    UMAP that derived its point identity once at load, a tree. Those views
    reached for a bare ``Index.isin``, which asks for exact equality, and
    exact equality is the wrong question the moment one side of a link states
    an object type and the other does not: a table publishing ``…_f1_cell1``
    highlighted **nothing at all** in a UMAP whose points are keyed
    ``…_f1_1``. Silence, in the one place a linked view has to be loud.

    :param keys: the view's own keys, in its own order.
    :param wanted: the keys to match against — a selection's, a request's.
    :returns: a boolean :class:`numpy.ndarray` aligned to ``keys``.
    """
    typed_rows = pd.Index([str(k) for k in keys], dtype=object)
    untyped_rows = pd.Index([untyped_object_key(k) for k in typed_rows],
                            dtype=object)
    return _match(typed_rows, untyped_rows, wanted)


def _match_frame(df: pd.DataFrame, wanted: Iterable[Any], *,
                 timelapse: bool = False,
                 object_type: Any = None) -> np.ndarray:
    """:func:`match_keys` for a caller that still has the frame.

    Built off the key columns rather than off strings, so the untyped form is
    *rebuilt* exactly rather than parsed back out of a composed key.
    """
    typed_rows = object_keys(df, timelapse=timelapse, object_type=object_type)
    untyped_rows = untyped_object_keys(df, timelapse=timelapse)
    return _match(typed_rows, untyped_rows, wanted)


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

    def mask_for(self, df: pd.DataFrame, *, timelapse: bool = False,
                 object_type: Any = None) -> np.ndarray:
        """Boolean mask of the rows of ``df`` that are in this selection.

        With no selection every row is in it — the resting state highlights
        nothing rather than everything, but a caller asking "which rows are
        selected" when nothing is gets the whole frame rather than an empty
        one, which is what keeps ``df[sel.mask_for(df)]`` meaning "the data
        the user is looking at".

        **Types are matched by specificity, not by equality**, and that is the
        whole of the object-type migration:

        * two typed keys match when they agree — a nucleus 1 is not a
          pathogen 1, which is the collapse this exists to end;
        * a key stating no type matches a row of *any* type. It says "the
          object labelled 7 in that field", which is exactly what it said
          before types existed, so every selection ever saved goes on naming
          what it named. It is deliberately not narrowed to one type: an old
          key that quietly resolved to one of four objects is the bug, and
          replacing it with a *different* silent choice would not be a fix.
        * a typed key matches a row stating no type. The row has not
          contradicted it; it has said nothing.

        :param object_type: the table ``df`` came from, when the frame does
            not carry :data:`OBJECT_TYPE_COLUMN` itself.
        """
        if self.keys is None:
            return np.ones(len(df), dtype=bool)
        if df.empty:
            return np.zeros(0, dtype=bool)
        return _match_frame(df, self.keys, timelapse=timelapse,
                            object_type=object_type)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source: str = "",
                   *, timelapse: bool = False,
                   object_type: Any = None) -> "Selection":
        """Select exactly the rows of ``df``."""
        return cls(keys=object_keys(df, timelapse=timelapse,
                                    object_type=object_type), source=source)

    @classmethod
    def from_keys(cls, keys: Any, source: str = "",
                  *, timelapse: bool = False,
                  object_type: Any = None) -> "Selection":
        """Select exactly ``keys`` — anything :func:`as_key_index` accepts.

        The counterpart to :meth:`from_frame` for a view that never had the
        frame: a scatter plot holding an array of keys, or a screen restoring
        a selection from a settings file.
        """
        return cls(keys=as_key_index(keys, timelapse=timelapse,
                                     object_type=object_type), source=source)

    @classmethod
    def none(cls) -> "Selection":
        return cls(keys=None, source="")


# ---------------------------------------------------------------------------
# "Show me exactly these objects"
# ---------------------------------------------------------------------------

def as_key_index(keys: Any, *, timelapse: bool = False,
                 object_type: Any = None) -> pd.Index:
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
    :param object_type: likewise — the object table a frame came from.
    :returns: a :class:`pandas.Index` of ``str``.
    :raises TypeError: if ``keys`` is not something that can name objects.
    :raises ValueError: for a resting :class:`Selection`.

    Order is preserved and duplicates are dropped. Order is load-bearing: it
    is what carries "worst errors first" from a confusion-matrix cell through
    to whatever opens them, and a duplicated key would draw the same crop
    twice in the grid.

    Key **strings are passed through untouched**, including untyped ones. It
    is tempting to normalise them here, and it would be wrong: a caller may
    legitimately hand over a ``prcfo``, a crop path or a file name (see
    :func:`spacr.active_learning.crops_for_object_keys`), and rewriting those
    into something that looks like an object key would break the resolution
    they were relying on. Untyped and typed keys are reconciled where they are
    *compared* — :func:`_match_keys` — not where they are collected.
    """
    if isinstance(keys, Selection):
        if keys.keys is None:
            raise ValueError(
                "a resting Selection names no objects; there is nothing to "
                "open. Check `selection.is_active` first — 'nothing selected' "
                "and 'an empty selection' are different states.")
        values: Iterable[Any] = list(keys.keys)
    elif isinstance(keys, pd.DataFrame):
        values = list(object_keys(keys, timelapse=timelapse,
                                  object_type=object_type))
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

    def select_from(self, df: pd.DataFrame, *,
                    object_type: Any = None) -> pd.DataFrame:
        """The rows of ``df`` this request names, **in the request's order**.

        Not a mask, and not ``df``'s order: the caller's order is the answer
        for a request built worst-first, and a boolean mask would silently
        re-sort it back into table order. Keys with no row in ``df`` are
        dropped — a request can name objects a narrower table does not carry,
        and that is a smaller result, not an error.

        Types are matched by specificity, exactly as
        :meth:`Selection.mask_for` describes: an exact match first, then a key
        that states no type, then a typed key against a row that states none.
        Trying them in that order is what keeps a request naming both a
        nucleus 1 and a pathogen 1 opening as two rows rather than one.

        :param object_type: the table ``df`` came from, when the frame does
            not carry :data:`OBJECT_TYPE_COLUMN` itself.
        """
        if df.empty:
            return df.iloc[:0]
        exact = {key: i for i, key in enumerate(self.keys)}
        loose: dict = {}
        narrowed: dict = {}
        for i, key in enumerate(self.keys):
            if key_object_type(key) is None:
                loose.setdefault(key, i)
            else:
                narrowed.setdefault(untyped_object_key(key), i)
        typed_rows = object_keys(df, timelapse=self.timelapse,
                                 object_type=object_type)
        if loose or narrowed:
            plain_rows = untyped_object_keys(df, timelapse=self.timelapse)
        else:
            plain_rows = typed_rows

        def rank(typed: str, plain: str) -> int:
            found = exact.get(typed)
            if found is None:
                found = loose.get(plain)
            if found is None and typed == plain:
                found = narrowed.get(plain)
            return -1 if found is None else found

        ranks = np.array([rank(t, p) for t, p in zip(typed_rows, plain_rows)],
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
