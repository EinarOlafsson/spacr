"""``V9`` ``B20`` — cell → nucleus → pathogen, as the tree it already is.

Every object table in ``measurements.db`` is flat, and the relationships
between them are already stored: :mod:`spacr.schema` gives ``nucleus`` and
``pathogen`` a ``cell_id`` pointing at the cell they sit inside
(:data:`spacr.schema.CHILD_OBJECT_TABLES`). Nothing has ever *shown* that.
"Cell 41 in field 3 has one nucleus and four pathogens, and one of those
pathogens has an area that would be impossible inside that nucleus" is a
question the database can answer and the GUI could not ask.

This module is the answer, in plain pandas with no Qt, so the tree can be
built in a notebook and tested without a display.

Why a tree and not a join
-------------------------

A join answers "give me every pathogen with its cell's area". A tree answers
"what is inside this cell", which is a different question and the one a person
looking at a crop actually has. The difference shows up in the failures: a
join silently drops a cell with no pathogens and silently drops a pathogen
whose ``cell_id`` names a cell that is not there. Both of those are findings —
the first is the negative control working, the second is a segmentation bug —
so :func:`build_forest` keeps childless parents and :func:`orphans` returns
the unattached children rather than discarding them.

Identity is the shared one
--------------------------

Nodes are keyed by :func:`spacr.selection.object_keys`, the same string the
UMAP, the plate view and the crop grid use, so selecting a node in the tree
publishes something every other view already understands. A child's parent is
resolved *within its own field*: ``cell_id`` is a label, not a key, and label
7 exists in every field on the plate.

The shared key carries the object type — this module is why
-----------------------------------------------------------

:data:`spacr.selection.OBJECT_KEY_COLUMNS` used to be the field plus the
object label, with no table in it, so a nucleus labelled 1 and a pathogen
labelled 1 in the same field had **the same key**. A lineage tree is where
that became visible, because a cell's own children are exactly the objects
most likely to collide: four objects opened as three crops and which one you
got depended on the row order of ``png_list``.

:func:`spacr.selection.object_keys` now writes the object type into the key,
so a node's shared key already says which table it came from and
:attr:`LineageNode.node_id` is the same identity in a different spelling.
Both are kept. ``node_id`` is what the tree addresses its rows by and cannot
collide by construction; :meth:`LineageNode.key_collisions` compares the two
and is now a **regression test rather than a warning** — it finds nothing on a
correctly keyed forest, and :func:`build_forest` still takes ``typed=False``
so the collapse can be reproduced on demand rather than only remembered.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field as _field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import schema
from .active_learning import _object_label

__all__ = [
    "LineageError",
    "LineageNode",
    "ROOT_TABLE",
    "LINEAGE_TABLES",
    "child_tables",
    "field_key",
    "node_key",
    "build_forest",
    "tree_for",
    "orphans",
    "lineage_frame",
    "read_object_tables",
    "describe_forest",
    "forest_key_collisions",
    "ID_SEPARATOR",
]

#: What separates the table from the object key in
#: :attr:`LineageNode.node_id`. A colon, because
#: :data:`spacr.schema.KEY_SEPARATOR` is ``'_'`` and is already inside the key
#: — joining on it again would make ``node_id`` as ambiguous as the thing it
#: exists to disambiguate.
ID_SEPARATOR = ":"


class LineageError(ValueError):
    """A set of tables that cannot be assembled into a lineage.

    Raised rather than returning an empty forest: "this cell has no children"
    and "the cell table has no object_label column so nothing could be
    matched" render identically as a leaf node, and only one of them is a
    result.
    """


#: The table whose rows are the roots of the tree. Cells contain nuclei and
#: pathogens; nothing contains a cell.
ROOT_TABLE = "cell"

#: The tables this module assembles, root first. ``cytoplasm`` is a parent
#: table in the schema but carries no ``cell_id``, so it has no place in a
#: containment tree — it is the cell minus its children, not a child.
LINEAGE_TABLES: Tuple[str, ...] = (ROOT_TABLE,) + schema.CHILD_OBJECT_TABLES


def child_tables() -> Tuple[str, ...]:
    """The tables whose rows carry a parent link, from the schema itself.

    Read off :data:`spacr.schema.OBJECT_TABLE_SCHEMAS` rather than written
    out here, so a table that gains a ``parent_column`` joins the tree
    without an edit in this file. ``organelle`` is in
    :data:`spacr.schema.CHILD_OBJECT_TABLES` but has no declared schema
    entry, so it is included by name — its per-object table is optional and
    its rows carry ``cell_id`` when it exists.
    """
    declared = tuple(
        name for name, contract in schema.OBJECT_TABLE_SCHEMAS.items()
        if contract.parent_column)
    extra = tuple(name for name in schema.CHILD_OBJECT_TABLES
                  if name not in declared)
    return declared + extra


def field_key(row: Mapping[str, Any]) -> str:
    """The ``prcf`` of one row — the field a label is unique within.

    ``cell_id`` is an object *label*, and label 7 exists in every field of
    every plate. Matching children to parents on the label alone attaches
    every field's nuclei to every field's cell 7, which produces a tree that
    looks plausible and is wrong everywhere.
    """
    return schema.KEY_SEPARATOR.join(
        str(row[column]) for column in schema.FIELD_KEY_COLUMNS)


def node_key(row: Mapping[str, Any],
             object_type: Optional[str] = None) -> str:
    """The shared object key of one row: field, then the typed object id.

    :param row: an object-table row.
    :param object_type: which table it came from. ``None`` builds the untyped
        key spaCR wrote before object types existed — the one that gives a
        cell's nucleus 1 and its pathogen 1 the same name.
    """
    label = str(row[schema.OBJECT_LABEL_KEY])
    if object_type is not None and schema.is_object_type(object_type):
        label = schema.object_id(label, object_type=object_type)
    return schema.KEY_SEPARATOR.join([field_key(row), label])


@dataclass(frozen=True)
class LineageNode:
    """One object and everything inside it.

    :ivar key: the shared object key — the same string every other view uses.
    :ivar table: which object table the row came from.
    :ivar label: the integer object label within its field.
    :ivar field: the ``prcf`` this object belongs to.
    :ivar children: contained objects, grouped table by table in
        :data:`LINEAGE_TABLES` order and by label within a table, so two runs
        of the same data draw the same tree.
    :ivar row: the source row, so a view can show a measurement beside the
        name without going back to the frame.
    """

    key: str
    table: str
    label: int
    field: str
    children: Tuple["LineageNode", ...] = ()
    row: Mapping[str, Any] = _field(default_factory=dict)

    # -- identity ------------------------------------------------------------
    @property
    def node_id(self) -> str:
        """``'pathogen:plate1_r1_c1_f1_pathogen1'`` — the table, then the key.

        Distinct by construction, which :attr:`key` was not until the object
        type went into it. Now that it has, this is the same identity said
        twice — and it stays, because it is what proves the other one:
        :meth:`key_collisions` is exactly the comparison between them, and a
        second identity that cannot collide is what makes the first one's
        collisions detectable rather than invisible.
        """
        return f"{self.table}{ID_SEPARATOR}{self.key}"

    # -- walking ------------------------------------------------------------
    def walk(self, depth: int = 0) -> Iterable[Tuple[int, "LineageNode"]]:
        """This node then its descendants, depth-first, with their depth."""
        yield depth, self
        for child in self.children:
            yield from child.walk(depth + 1)

    def descendants(self) -> Tuple["LineageNode", ...]:
        """Everything below this node, depth-first, excluding itself."""
        return tuple(node for depth, node in self.walk() if depth)

    def node_ids(self) -> Tuple[str, ...]:
        """Every :attr:`node_id` below (and at) this node, depth-first.

        Always distinct — unlike :meth:`keys`, which cannot be.
        """
        return tuple(node.node_id for _depth, node in self.walk())

    def keys(self) -> Tuple[str, ...]:
        """This node's key and every descendant's, depth-first, de-duplicated.

        The order a selection made on this node publishes in, so the parent
        comes first and the crops open with the cell at the front of the grid.

        **As long as the subtree**, now that the shared key carries the object
        type. It used not to be: a nucleus 1 and a pathogen 1 inside the same
        cell were one key, de-duplicating was the only honest thing to do
        (sending the same key twice would draw the same crop twice), and four
        objects opened as three. :meth:`key_collisions` is the check that this
        no longer happens — it returns nothing on a typed forest.
        """
        out: List[str] = []
        seen = set()
        for _depth, node in self.walk():
            if node.key not in seen:
                seen.add(node.key)
                out.append(node.key)
        return tuple(out)

    def key_collisions(self) -> Dict[str, Tuple[str, ...]]:
        """``{shared key: the tables that share it}``, for the ones that do.

        **Empty on a correctly keyed forest** — that is now the assertion this
        method exists to make, not a hope. Non-empty means this subtree holds
        objects that every other view will treat as one: the crop grid shows
        one of them, and which one depends on the order ``png_list`` happens
        to be in. That was the ordinary case before the object type went into
        the key; it is a regression now, and it is still detectable because
        :attr:`node_id` cannot collide even when :attr:`key` can.
        """
        by_key: Dict[str, List[str]] = {}
        for _depth, node in self.walk():
            by_key.setdefault(node.key, []).append(node.table)
        return {key: tuple(tables) for key, tables in by_key.items()
                if len(tables) > 1}

    def counts(self) -> Dict[str, int]:
        """How many of each table are inside this node (itself included)."""
        out: Dict[str, int] = {}
        for _depth, node in self.walk():
            out[node.table] = out.get(node.table, 0) + 1
        return out

    def find(self, key: str) -> Optional["LineageNode"]:
        """The node with this key, anywhere below (or at) this one."""
        for _depth, node in self.walk():
            if node.key == str(key):
                return node
        return None

    def describe(self) -> str:
        """One line: what this is, and what is inside it."""
        inside = {t: n for t, n in self.counts().items() if t != self.table}
        if not inside:
            return f"{self.table} {self.label} · nothing inside it"
        parts = ", ".join(f"{n} {t}" for t, n in sorted(inside.items()))
        return f"{self.table} {self.label} · {parts}"


def _normalise(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Check one object table and give it string identity columns."""
    needed = list(schema.FIELD_KEY_COLUMNS) + [schema.OBJECT_LABEL_KEY]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise LineageError(
            f"the {table!r} table is missing {missing}, so its objects cannot "
            f"be named. A lineage needs "
            f"{needed} on every table it assembles.")
    out = frame.copy()
    for column in needed:
        out[column] = out[column].astype(str)
    return out


def build_forest(frames: Mapping[str, pd.DataFrame], *,
                 root: str = ROOT_TABLE,
                 typed: bool = True) -> Tuple[LineageNode, ...]:
    """Assemble object tables into one tree per root object.

    :param frames: ``{table name: rows}``. Only the tables named in
        :data:`LINEAGE_TABLES` are read; anything else is ignored, so a
        caller can hand over everything it loaded.
    :param root: the table whose rows become the roots.
    :param typed: put each node's object table into its shared key, so a
        cell's nucleus 1 and its pathogen 1 are two keys. ``False`` rebuilds
        the untyped keys spaCR wrote before object types existed — kept so
        the collapse can be *reproduced* rather than only remembered, which
        is what makes :meth:`LineageNode.key_collisions` a test with two
        sides to it.
    :returns: one :class:`LineageNode` per root row, in field order then
        label order.
    :raises LineageError: when the root table is absent or unusable. A
        missing *child* table is not an error — a run that measured cells and
        not pathogens produces a forest of childless cells, which is the
        truth about that run.

    Childless roots are kept. A cell with no pathogens in an infection assay
    is the negative control working, and a tree that dropped it would show
    the infected population as if it were the whole plate.
    """
    root = str(root)
    if root not in frames:
        raise LineageError(
            f"no {root!r} table to build a lineage from; got "
            f"{sorted(frames)}. Load the parent table as well as the "
            f"children — a tree with no roots is a list.")
    parents = _normalise(frames[root], root)
    # (field, label) -> the children hanging off it, table by table.
    by_parent: Dict[Tuple[str, str], List[LineageNode]] = {}
    for table in LINEAGE_TABLES:
        if table == root or table not in frames:
            continue
        children = _normalise(frames[table], table)
        parent_column = _parent_column(table, children)
        if parent_column is None:
            continue
        rows = children.to_dict("records")
        # Sorted by label so the tree is stable; `_object_label` normalises
        # '7', 7 and 7.0 to one thing, and 'onone'/'omulti' to nothing.
        rows.sort(key=lambda r: _sort_label(r[schema.OBJECT_LABEL_KEY]))
        for row in rows:
            parent_label = _object_label(row.get(parent_column))
            if not parent_label:
                continue
            node = _node(row, table, typed=typed)
            by_parent.setdefault((node.field, parent_label), []).append(node)

    roots: List[LineageNode] = []
    parent_rows = parents.to_dict("records")
    parent_rows.sort(key=lambda r: (field_key(r),
                                    _sort_label(r[schema.OBJECT_LABEL_KEY])))
    for row in parent_rows:
        node = _node(row, root, typed=typed)
        label = _object_label(row[schema.OBJECT_LABEL_KEY])
        children = tuple(by_parent.get((node.field, label), ()))
        roots.append(LineageNode(key=node.key, table=root, label=node.label,
                                 field=node.field, children=children,
                                 row=node.row))
    return tuple(roots)


def _parent_column(table: str, frame: pd.DataFrame) -> Optional[str]:
    """The column linking ``table``'s rows to their parent, if it is there."""
    contract = schema.OBJECT_TABLE_SCHEMAS.get(table)
    candidates = [contract.parent_column] if contract is not None else []
    candidates.append("cell_id")
    candidates.append(f"{table}_cell_id")
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return candidate
    return None


def _sort_label(value: Any) -> Tuple[int, Any]:
    """Sort key that puts real labels in numeric order and junk last."""
    text = _object_label(value)
    return (0, int(text)) if text else (1, str(value))


def _node(row: Mapping[str, Any], table: str, *,
          typed: bool = True) -> LineageNode:
    label = _object_label(row.get(schema.OBJECT_LABEL_KEY))
    return LineageNode(key=node_key(row, table if typed else None),
                       table=table,
                       label=int(label) if label else 0,
                       field=field_key(row), children=(), row=dict(row))


def tree_for(frames: Mapping[str, pd.DataFrame], key: str, *,
             root: str = ROOT_TABLE,
             typed: bool = True) -> Optional[LineageNode]:
    """The tree containing ``key``, or ``None``.

    ``key`` may name the root or anything inside it: asked about a pathogen,
    this returns the *cell* it lives in, because the useful view of a
    pathogen is the cell around it. A caller that wants only the subtree
    calls :meth:`LineageNode.find` on the result.
    """
    wanted = str(key)
    for node in build_forest(frames, root=root, typed=typed):
        if node.find(wanted) is not None:
            return node
    return None


def orphans(frames: Mapping[str, pd.DataFrame], *,
            root: str = ROOT_TABLE) -> pd.DataFrame:
    """Child rows whose parent link names no row in the root table.

    A finding, not an error. A nucleus whose ``cell_id`` is 12 in a field
    whose cell table has no object 12 means the two masks disagree — the
    nucleus segmentation found something the cell segmentation did not — and
    that is worth showing rather than dropping on the way into a tree.

    :returns: the offending rows with ``table`` and ``parent_id`` columns
        added, in table then field then label order. Empty when everything
        attaches, which is the healthy case.
    """
    root = str(root)
    if root not in frames:
        raise LineageError(f"no {root!r} table to check parents against")
    parents = _normalise(frames[root], root)
    known = {
        (field_key(row), _object_label(row[schema.OBJECT_LABEL_KEY]))
        for row in parents.to_dict("records")
    }
    loose: List[Dict[str, Any]] = []
    for table in LINEAGE_TABLES:
        if table == root or table not in frames:
            continue
        children = _normalise(frames[table], table)
        parent_column = _parent_column(table, children)
        if parent_column is None:
            continue
        for row in children.to_dict("records"):
            parent_label = _object_label(row.get(parent_column))
            if not parent_label:
                # No link at all is a different fact from a broken link: the
                # row never claimed a parent. Reported too, with an empty
                # parent_id, because a pathogen with no cell_id is also a
                # pathogen nothing will ever show.
                loose.append({**row, "table": table, "parent_id": ""})
            elif (field_key(row), parent_label) not in known:
                loose.append({**row, "table": table,
                              "parent_id": parent_label})
    if not loose:
        return pd.DataFrame(columns=["table", "parent_id"])
    out = pd.DataFrame(loose)
    return out.sort_values(
        ["table"] + list(schema.FIELD_KEY_COLUMNS) + [schema.OBJECT_LABEL_KEY],
        kind="stable").reset_index(drop=True)


def lineage_frame(forest: Sequence[LineageNode]) -> pd.DataFrame:
    """The forest flattened: one row per node, with its parent and depth.

    For export, for a table view, and for the tests — a tree is awkward to
    assert on and this is the same information in a shape pandas can compare.
    """
    rows: List[Dict[str, Any]] = []

    def visit(node: LineageNode, parent: str, depth: int) -> None:
        rows.append({"key": node.key, "table": node.table,
                     "label": node.label, "field": node.field,
                     "parent_key": parent, "depth": depth,
                     "n_children": len(node.children)})
        for child in node.children:
            visit(child, node.key, depth + 1)

    for root in forest:
        visit(root, "", 0)
    return pd.DataFrame(rows, columns=["key", "table", "label", "field",
                                       "parent_key", "depth", "n_children"])


def forest_key_collisions(forest: Sequence[LineageNode]
                          ) -> Dict[str, Tuple[str, ...]]:
    """Every shared key in ``forest`` that names more than one object.

    Collisions are counted *within a family*, not across the whole forest:
    two different cells' pathogens both labelled 1 in the same field cannot
    happen (the label is unique per mask), while a nucleus 1 and a pathogen 1
    inside one cell is the ordinary case. Merging the two would report a
    collision on every plate.
    """
    out: Dict[str, Tuple[str, ...]] = {}
    for root in forest:
        out.update(root.key_collisions())
    return out


def describe_forest(forest: Sequence[LineageNode]) -> str:
    """The shape of a whole forest in words.

    Says the thing a tree of ten thousand rows cannot: how many parents have
    nothing inside them. In an infection assay that number *is* the readout,
    and having to count it by scrolling is how it gets estimated instead.
    """
    if not forest:
        return "No parent objects, so there is no lineage to show."
    totals: Dict[str, int] = {}
    childless = 0
    for root in forest:
        if not root.children:
            childless += 1
        for table, count in root.counts().items():
            totals[table] = totals.get(table, 0) + count
    root_table = forest[0].table
    inside = ", ".join(f"{n} {t}" for t, n in sorted(totals.items())
                       if t != root_table) or "nothing"
    return (f"{len(forest)} {root_table}(s) holding {inside}. "
            f"{childless} of them ({childless / len(forest):.0%}) have "
            f"nothing inside.")


# ---------------------------------------------------------------------------
# Reading — sqlite, and no Qt
# ---------------------------------------------------------------------------

def read_object_tables(db_path: str,
                       tables: Optional[Sequence[str]] = None,
                       *, limit: int = 200_000) -> Dict[str, pd.DataFrame]:
    """Read the object tables a lineage needs. Safe on a worker thread.

    Missing tables are simply absent from the result — a run that measured
    cells and nuclei but no pathogens is a legitimate experiment, not a
    broken database.

    :param limit: per-table row cap, so a mis-aimed path cannot turn into a
        two-minute read behind a spinner.
    :raises LineageError: when there is no database at ``db_path``.
    """
    if not db_path or not os.path.isfile(db_path):
        raise LineageError(f"no measurements database at {db_path!r}")
    wanted = list(tables or LINEAGE_TABLES)
    out: Dict[str, pd.DataFrame] = {}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in wanted:
            if table not in present:
                continue
            out[table] = pd.read_sql_query(
                f'SELECT * FROM "{table}" LIMIT {int(limit)}', connection)
    finally:
        connection.close()
    return out
