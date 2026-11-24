"""One merged measurements frame from the plate rows of the input table.

A row of the regression input table is one plate: its score
CSV, its count CSV and now its measurements database. This module is the
headless half of that feature -- given ``{plate: database path}``, the tables
the user ticked and the anchor they chose, it returns the merged frame.

This module adds no merge logic of its own. spaCR already has four places
that join measurement tables, and one of them (``io._read_and_join_tables``)
aggregates every numeric column with ``mean``, which turns four pathogens'
total area into an average area and a minimum into a mean of minima. This
module composes existing functions instead:

* :func:`spacr.multi_database.describe_merge` / :func:`~spacr.multi_database.read_merged`
  stack the databases -- one table at a time, carrying ``source_database`` and
  ``screenID``, refusing a plate id repeated inside one screen;
* :func:`spacr.merge_tables.roll_up` aggregates each child table onto the
  anchor, one rule per column out of
  :data:`spacr.merge_tables.AGGREGATION_RULES`;
* :meth:`spacr.merge_tables.MergePolicy.how_for` decides the join PER TABLE,
  because the cardinality differs -- a cell has one cytoplasm and many
  pathogens, and an uninfected cell is still a cell.

There is no ``sum()`` or ``mean()`` of a measurement column written in this
file, and there must never be one.

Why the two existing functions do not compose by themselves
-----------------------------------------------------------
``read_merged`` is *many databases, one table*. ``merge_tables`` is *one
database, many tables* -- and it takes a PATH, so it cannot be handed a frame
that has already been stacked across databases. This feature needs both, so
the composition is: read each chosen table across every database, then roll
the children onto the anchor.

The roll-up keys are the identity columns plus ``screenID``,
``source_database`` and the child's anchor column. Without ``screenID``, two
screens that share ``plate1`` (which
:func:`~spacr.multi_database.describe_merge` permits) would collapse into one
parent, which is the pooling ``multi_database`` prevents.

What a caller must show the user
--------------------------------
A merge that silently changed how a measurement was combined produces a number
that is wrong and looks fine, so :class:`PlateMerge` carries what the panel has
to say: the anchor and the row count, the rows each source contributed, the
measurements ``columns='common'`` dropped, the plate ids shared across screens
-- and every column that fell through to
:data:`spacr.merge_tables.DEFAULT_AGGREGATION` because no rule matched it. That
last set is recomputed by re-walking ``AGGREGATION_RULES``
(:func:`default_aggregated_columns`) rather than listed anywhere, so it cannot
drift out of step with the rules.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import merge_tables as mt
from .merge_tables import (
    IDENTITY,
    OBJECT_COLUMN,
    OBJECT_TABLES,
    PNG_TABLE,
    MergePolicy,
    aggregation_plan,
    mergeable_tables,
    roll_up,
)
from .multi_database import (
    SCREEN_COLUMN,
    SOURCE_COLUMN,
    MergeCancelled,
    MergePlan,
    MergeRefused,
    describe_merge,
    read_merged,
    source_labels,
)
from .object_roles import ONE_ROW_PER_CELL, anchor_column, is_one_row_per_cell

LOG = logging.getLogger("spacr.plate_measurements")

#: Columns that name a row rather than measure it, and are therefore never
#: prefixed with the table they came from. ``screenID`` and
#: ``source_database`` are here because they are JOIN KEYS in this
#: composition: prefix them and the roll-up no longer meets the anchor.
_UNPREFIXED: Tuple[str, ...] = IDENTITY + (
    OBJECT_COLUMN, SCREEN_COLUMN, SOURCE_COLUMN, "timeID", "time_id")


@dataclass(frozen=True)
class PlateDatabase:
    """One plate row of the input table and the database attached to it.

    :param plate: input-table plate label used to map screen metadata and
        identify missing or duplicate attachments in diagnostics. A blank
        label normalised from an input row becomes ``"row N"``.
    :param path: filesystem path to the plate's measurements database. An
        empty value means no database is attached; a nonempty path is checked
        for existence before table discovery and merging.
    """

    plate: str
    path: str

    @property
    def exists(self) -> bool:
        """Whether the file is still where the input table says it is.

        Checked before the run rather than during it: the design asks
        that a database that has been moved is named up front, not four
        minutes into a regression.
        """
        return bool(self.path) and os.path.exists(self.path)


@dataclass(frozen=True)
class TableMerge:
    """What one chosen table contributed to the merge, and how.

    One of these per table the user ticked, including the anchor. It is the
    record a panel reads to answer "what happened to my numbers": which
    aggregation each column got, whether the table was rolled up or joined
    directly, and which join mode the merge policy selected.

    :param table: Chosen source object-table name, including the anchor table.
    :param plan: Multi-database read plan used to read this table.
    :param rows: Number of source rows read across the attached databases
        before joining or child roll-up.
    :param keys: Source identity, join, or grouping columns retained without a
        table prefix.
    :param how: Child-table join mode returned by
        :meth:`spacr.merge_tables.MergePolicy.how_for`; empty for the anchor
        or a skipped table.
    :param rolled_up: Whether this child table's rows were aggregated onto the
        anchor; false for the anchor, a directly joined one-row-per-cell table,
        or a skipped table.
    :param aggregations: Source-column-to-aggregation mapping actually applied
        during child roll-up; empty for the anchor, a direct join, or a skipped
        table.
    :param default_columns: Source columns in ``aggregations`` that used the
        default because no named aggregation rule matched and no override
        applied.
    :param dropped: Source columns absent from at least one attached database
        and therefore omitted by the common-column read.
    :param note: Explanation when the table was skipped and contributed no
        merged columns; empty otherwise.
    """

    table: str
    plan: MergePlan
    rows: int
    keys: Tuple[str, ...] = ()
    how: str = ""
    rolled_up: bool = False
    #: ``{column: aggregation}`` exactly as
    #: :func:`spacr.merge_tables.aggregation_plan` returned it -- keyed on the
    #: child's own column names, before the table prefix.
    aggregations: Mapping[str, str] = field(default_factory=dict)
    #: The subset of :attr:`aggregations` that no rule matched.
    default_columns: Tuple[str, ...] = ()
    #: What the read ACTUALLY dropped, off the frame's own ``attrs`` -- not
    #: :attr:`spacr.multi_database.MergePlan.dropped_columns`, which is what a
    #: ``columns='common'`` merge WOULD drop and is therefore wrong to report
    #: after a ``'union'`` one. A panel that names measurements as lost while
    #: they are sitting in the frame is a panel saying something untrue about
    #: the numbers.
    dropped: Tuple[str, ...] = ()
    #: Why this table contributed nothing, when it did not.
    note: str = ""

    def merged_column(self, column: str) -> str:
        """The name ``column`` carries in the merged frame.

        :param column: source-table column name to express in merged-frame
            form.

        The same rule :func:`spacr.merge_tables.roll_up` applies: a join key
        keeps its name, a column that already starts with the table's name is
        left alone -- ``nucleus_area`` must not become ``nucleus_nucleus_area``
        -- and everything else is prefixed with the table it measures.
        """
        name = str(column)
        if name in self.keys:
            return name
        if name in _UNPREFIXED and name != OBJECT_COLUMN:
            return name
        if name.startswith(f"{self.table}_"):
            return name
        return f"{self.table}_{name}"


@dataclass(frozen=True)
class PlateMerge:
    """The merged frame and the Measurements tab's audit information.

    :param frame: final measurements frame, with one row per surviving anchor
        object.
    :param anchor: normalized object-table name whose rows define the output
        cardinality.
    :param attachments: validated plate/database attachments included in the
        merge, in input order.
    :param tables: per-table audit records in merge order, beginning with the
        anchor and including any child table skipped because it could not be
        linked.
    """

    frame: pd.DataFrame
    anchor: str
    attachments: Tuple[PlateDatabase, ...]
    tables: Tuple[TableMerge, ...]

    @property
    def rows(self) -> int:
        """How many anchor objects survived the merge."""
        return int(len(self.frame))

    @property
    def sources(self) -> Tuple[str, ...]:
        """The database labels, as :mod:`spacr.multi_database` named them."""
        for entry in self.tables:
            if entry.table == self.anchor:
                return tuple(source.label for source in entry.plan.sources)
        return ()

    @property
    def rows_read_per_source(self) -> Dict[str, int]:
        """Anchor rows each database held, before any join removed rows."""
        for entry in self.tables:
            if entry.table == self.anchor:
                return {source.label: int(source.rows)
                        for source in entry.plan.sources}
        return {}

    @property
    def rows_per_source(self) -> Dict[str, int]:
        """Anchor rows each database contributed to the final frame.

        Less than :attr:`rows_read_per_source` wherever an inner join removed
        objects -- a nucleus-less cell, or a pathogen-less one when
        ``keep_uninfected=False``. The pair is what lets a panel say how many
        rows were dropped instead of only how many are left.
        """
        if SOURCE_COLUMN not in self.frame.columns:
            return {}
        counts = self.frame[SOURCE_COLUMN].value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    @property
    def dropped_columns(self) -> Tuple[str, ...]:
        """Measurements ``columns='common'`` left out, as merged names.

        A dropped measurement is a measurement the user came to compare, so it
        is reported rather than merely defaulted.
        """
        found: List[str] = []
        for entry in self.tables:
            for column in entry.dropped:
                name = entry.merged_column(column)
                if name not in found:
                    found.append(name)
        return tuple(sorted(found))

    @property
    def default_aggregation_columns(self) -> Tuple[str, ...]:
        """Merged columns whose aggregation is the default, no rule matching.

        :data:`spacr.merge_tables.DEFAULT_AGGREGATION` is ``mean``. Columns
        that no rule matched are listed because a mean may not be the
        appropriate aggregation for them.
        """
        return tuple(sorted(
            entry.merged_column(column)
            for entry in self.tables for column in entry.default_columns))

    @property
    def shared_plates_across_screens(self) -> Dict[str, Tuple[str, ...]]:
        """Plate ids that appear in more than one screen.

        Two screens sharing a guide library both have ``plate1``; with
        ``screenID`` in the frame those are two identities, so this is not a
        collision. They are reported so that a user can see that two screens
        were merged.
        """
        for entry in self.tables:
            if entry.table == self.anchor:
                return dict(entry.plan.shared_plates_across_screens)
        return {}

    def describe(self) -> str:
        """The disclosure, as lines -- what this merge did and what it cost."""
        lines = [
            f"{self.rows:,} {self.anchor} objects from "
            f"{len(self.attachments)} database(s), anchored on {self.anchor}",
        ]
        read = self.rows_read_per_source
        final = self.rows_per_source
        for label in self.sources:
            held, kept = read.get(label, 0), final.get(label, 0)
            line = f"  {label}: {kept:,} of {held:,} {self.anchor} rows"
            if kept < held:
                line += f" ({held - kept:,} dropped by an inner join)"
            lines.append(line)
        for entry in self.tables:
            if entry.table == self.anchor:
                continue
            if entry.note:
                lines.append(f"  {entry.table}: {entry.note}")
                continue
            lines.append(
                f"  {entry.table}: {'rolled up' if entry.rolled_up else 'joined'}"
                f" {entry.how}, {entry.rows:,} rows read")
        if self.dropped_columns:
            lines.append(
                f"  {len(self.dropped_columns)} measurement(s) present in only "
                "some databases were dropped: "
                + ", ".join(self.dropped_columns[:6])
                + (" ..." if len(self.dropped_columns) > 6 else ""))
        defaulted = self.default_aggregation_columns
        if defaulted:
            lines.append(
                f"  {len(defaulted)} column(s) matched no aggregation rule and "
                f"took the default ({mt.DEFAULT_AGGREGATION}): "
                + ", ".join(defaulted[:6])
                + (" ..." if len(defaulted) > 6 else ""))
        shared = self.shared_plates_across_screens
        if shared:
            lines.append(
                f"  {len(shared)} plate id(s) appear in more than one SCREEN, "
                "which is not a clash: " + ", ".join(sorted(shared)))
        return "\n".join(lines)


def _rows(attachments: Any) -> List[PlateDatabase]:
    """Normalise every shape the input table hands over into plate rows."""
    if isinstance(attachments, Mapping):
        pairs: List[Tuple[Any, Any]] = list(attachments.items())
    else:
        pairs = []
        for entry in attachments or ():
            if isinstance(entry, PlateDatabase):
                pairs.append((entry.plate, entry.path))
            elif isinstance(entry, Mapping):
                pairs.append((entry.get("plate") or "",
                              entry.get("database") or entry.get("path") or ""))
            else:
                plate, path = entry
                pairs.append((plate, path))
    return [PlateDatabase(plate=str(plate or f"row {index + 1}"),
                          path=str(path or "").strip())
            for index, (plate, path) in enumerate(pairs)]


def plate_databases(attachments: Any) -> Tuple[PlateDatabase, ...]:
    """Return the plate rows that have a database, in input-table order.

    :param attachments: ``{plate: path}``, a sequence of :class:`PlateDatabase`,
        a sequence of ``(plate, path)`` pairs, or the input table's own rows
        (mappings carrying ``plate`` and ``database``).
    :returns: one :class:`PlateDatabase` per attached row. Rows with no
        database are left out rather than carried as blanks -- see
        :func:`unattached_plates`, which is where they are named.
    """
    return tuple(row for row in _rows(attachments) if row.path)


def unattached_plates(attachments: Any) -> Tuple[str, ...]:
    """Return the plates that have no database attached.

    :param attachments: input-table attachment rows in any shape accepted by
        :func:`plate_databases`.

    The regression runs on scores and counts; the database is needed only for
    the Measurements tab. A plate without one is unavailable there but does
    not fail the run, and it is listed so that the omission is visible.
    """
    return tuple(row.plate for row in _rows(attachments) if not row.path)


def missing_databases(attachments: Any) -> Tuple[PlateDatabase, ...]:
    """Attached databases that are not on disk, named before the run starts.

    :param attachments: input-table attachment rows in any shape accepted by
        :func:`plate_databases`.
    """
    return tuple(row for row in plate_databases(attachments) if not row.exists)


def available_tables(attachments: Any) -> Tuple[str, ...]:
    """Return the object tables present in every attached database.

    :param attachments: input-table attachment rows whose existing databases
        define the table intersection.

    The intersection is returned, not the union, because
    :func:`spacr.multi_database.describe_merge` raises a bare
    ``sqlite3.OperationalError: no such table`` when one database lacks the
    chosen table.

    :returns: the tables in :data:`spacr.merge_tables.OBJECT_TABLES` order, so
        the picker's order is the registry's order rather than sqlite's, and
        so a sixth object kind reaches this list by being declared once.
        ``png_list`` is not offered: it holds one row per crop rather than per
        object, and a crop is not a measurement to aggregate.
    """
    attached = plate_databases(attachments)
    if not attached:
        return ()
    per_database = [set(mergeable_tables(row.path)) for row in attached]
    shared = set.intersection(*per_database)
    return tuple(table for table in OBJECT_TABLES if table in shared)


def default_aggregated_columns(
        plan: Mapping[str, str], *,
        overrides: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """Return the columns in ``plan`` that got the default because no rule matched.

    Recomputed by re-walking :data:`spacr.merge_tables.AGGREGATION_RULES`, so
    this cannot drift out of step with them: a rule added for a measurement
    tomorrow removes that measurement from this list the same day, and a rule
    deleted puts it back.

    A column the caller overrode is not reported however it was set -- they
    chose it. Nor is a text column, which takes
    :data:`spacr.merge_tables.TEXT_AGGREGATION` because text does not add up,
    not because nobody thought about it.

    :param plan: ``{column: aggregation}`` from
        :func:`spacr.merge_tables.aggregation_plan`.
    :param overrides: the caller's explicit choices, which are excluded.
    """
    named = []
    for column, how in plan.items():
        if overrides and column in overrides:
            continue
        if how != mt.DEFAULT_AGGREGATION:
            continue
        name = str(column).lower()
        if any(re.search(pattern, name)
               for pattern, _how in mt.AGGREGATION_RULES):
            continue
        named.append(str(column))
    return tuple(named)




#: What a text identifier gets when its value is the same for every child of
#: a parent -- it is one fact about the parent, carried through.
IDENTIFIER_CARRIED = "first"
#: ...and what it gets when it is not. Picking one of three file names for a
#: cell invents provenance, so the column is left out and NAMED.
IDENTIFIER_REFUSED = "refused"


def classify_default_columns(
        columns: Sequence[str],
        kinds: Optional[Mapping[str, str]] = None, *,
        overrides: Optional[Mapping[str, str]] = None
) -> Dict[str, Tuple[str, ...]]:
    """Split the columns no rule names by the aggregation they will receive.

    :param columns: the column names about to be aggregated.
    :param kinds: ``{column: 'numeric' | 'text' | 'unknown'}``, as
        :func:`spacr.multi_database.column_kinds` returns it. A column absent
        from the mapping is ``'unknown'``.
    :param overrides: the caller's explicit choices. A column they set is not
        a fall-through and appears in neither bucket -- they chose it.
    :returns: ``{'mean': (...), 'identifier': (...), 'unknown': (...)}``:

        ``mean``
            numeric, no rule matched, so
            :data:`spacr.merge_tables.DEFAULT_AGGREGATION` is what it takes.
        ``identifier``
            text. It takes :data:`spacr.merge_tables.TEXT_AGGREGATION`, not a
            mean, and whether it is carried or refused depends on the data
            -- see :func:`ambiguous_identifiers`, which reads the rows.
        ``unknown``
            the database declared no type, so neither aggregation can be
            stated in advance. Listed separately so that a column of unknown
            type is not reported as numeric or text.
    """
    chosen = dict(overrides or {})
    kinds = dict(kinds or {})
    out: Dict[str, List[str]] = {"mean": [], "identifier": [], "unknown": []}
    for column in columns:
        name = str(column)
        if name in chosen:
            continue
        lowered = name.lower()
        if any(re.search(pattern, lowered)
               for pattern, _how in mt.AGGREGATION_RULES):
            continue
        kind = str(kinds.get(name, "unknown"))
        if kind == "numeric":
            out["mean"].append(name)
        elif kind == "text":
            out["identifier"].append(name)
        else:
            out["unknown"].append(name)
    return {key: tuple(value) for key, value in out.items()}


def ambiguous_identifiers(child: pd.DataFrame, keys: Sequence[str], *,
                          plan: Optional[Mapping[str, str]] = None,
                          overrides: Optional[Mapping[str, str]] = None,
                          examples: int = 1) -> Dict[str, Dict[str, Any]]:
    """Return the text identifiers that vary within their roll-up group.

    A text identifier that takes ``first`` must have one value per group.
    When it has more, the column is reported rather than resolved silently.
    A cell with three pathogens has one ``path_name`` if all three came from
    the same image; if it has three, ``first`` would name one of them and the
    merged row would claim a provenance the data does not support.

    Only text columns are examined. ``object_label`` also takes ``first``, by
    the identity rule, and is expected to differ across the children:
    :data:`spacr.merge_tables.AGGREGATION_RULES` carries the label only so a
    row can be traced back. A numeric label that varies is the normal case; a
    text identifier that varies is ambiguous.

    :param child: the many-per-parent table, before the roll-up.
    :param keys: the group keys the roll-up will use.
    :param plan: the aggregation plan, from
        :func:`spacr.merge_tables.aggregation_plan`. Recomputed when omitted.
    :param overrides: the caller's explicit choices. A column set here is
        not examined.
    :param examples: how many offending group keys to record per column.
    :returns: ``{column: {'groups': n, 'examples': [(key, [values])]}}`` --
        empty when every identifier is constant, which is the normal case.
    """
    keys = [key for key in keys if key in child.columns]
    if not keys or child.empty:
        return {}
    plan = dict(plan if plan is not None
                else aggregation_plan(child, overrides=overrides,
                                      skip=list(keys)))
    chosen = dict(overrides or {})
    suspects = [column for column, how in plan.items()
                if how == mt.TEXT_AGGREGATION
                and column not in chosen
                and column in child.columns
                and not pd.api.types.is_numeric_dtype(child[column])]
    if not suspects:
        return {}
    grouped = child.groupby(list(keys), dropna=False)
    out: Dict[str, Dict[str, Any]] = {}
    for column in suspects:
        counts = grouped[column].nunique(dropna=False)
        offending = counts[counts > 1]
        if not len(offending):
            continue
        found = []
        for key in list(offending.index)[:max(int(examples), 1)]:
            rows = child
            key_values = key if isinstance(key, tuple) else (key,)
            for name, value in zip(keys, key_values):
                rows = rows[rows[name] == value]
            found.append((key_values,
                          sorted({str(value) for value in rows[column]})))
        out[str(column)] = {"groups": int(len(offending)), "examples": found}
    return out


def describe_identifier_refusal(table: str, column: str,
                                detail: Mapping[str, Any]) -> str:
    """One line saying why ``column`` was left out, with an example.

    :param table: child table whose identifier could not be carried safely.
    :param column: varying text-identifier column that was omitted.
    :param detail: ambiguity record from :func:`ambiguous_identifiers`,
        including the affected group count and optional examples.

    Written here rather than in the panel so the headless merge and the Qt one
    say the same sentence.
    """
    examples = list(detail.get("examples") or ())
    groups = int(detail.get("groups", 0))
    line = (f"  {table}: {column} is a text identifier that differs WITHIN "
            f"{groups:,} group(s), so it was left out rather than set to one "
            f"of its values — picking one invents provenance")
    if examples:
        key, values = examples[0]
        shown = ", ".join(str(value) for value in values[:3])
        if len(values) > 3:
            shown += f", … ({len(values)} values)"
        line += f" (e.g. {'/'.join(str(part) for part in key)}: {shown})"
    return line + "."


def _screen_labels(attached: Sequence[PlateDatabase], screens: Any) -> Any:
    """One screen label per database, from a per-PLATE mapping or a sequence.

    A mapping keyed by plate is the shape the input table has, and it is the
    recommended one: the label is written into
    :data:`spacr.multi_database.SCREEN_COLUMN`, where it stays a dimension you
    can block on -- rather than into the plate id, where ``on_collision``
    ``'qualify'`` would put it and where it becomes a string to be parsed back
    apart. This module never qualifies plate IDs.
    """
    if isinstance(screens, Mapping):
        return [screens.get(row.plate) for row in attached]
    return screens


def _refuse_missing_tables(attached: Sequence[PlateDatabase],
                           chosen: Sequence[str]) -> None:
    """Refuse a table that is not in every database, naming the ones without."""
    not_objects = [table for table in chosen if table not in OBJECT_TABLES]
    if not_objects:
        detail = (f" {PNG_TABLE} holds one row per CROP rather than per "
                  f"object, and is attached by `merge_tables` without being "
                  f"aggregated." if PNG_TABLE in not_objects else "")
        raise MergeRefused(
            f"{', '.join(repr(t) for t in not_objects)} is not an object "
            f"table; this merges {list(OBJECT_TABLES)}.{detail}")
    missing: Dict[str, List[str]] = {}
    for row in attached:
        present = set(mergeable_tables(row.path))
        for table in chosen:
            if table not in present:
                missing.setdefault(table, []).append(row.plate)
    if not missing:
        return
    detail = "; ".join(f"{table!r} is missing from {', '.join(plates)}"
                       for table, plates in sorted(missing.items()))
    raise MergeRefused(
        f"every attached database must have every chosen table, and {detail}. "
        f"Choose from the tables they all have: "
        f"{', '.join(available_tables(attached)) or 'none'}.")


def _prefixed_report(report: Optional[Callable[[str], None]],
                     table: str) -> Optional[Callable[[str], None]]:
    """``report``, with the table named -- one line per table, not per merge."""
    if report is None:
        return None
    return lambda line: report(f"{table}: {line}")


def merge_plate_databases(attachments: Any, tables: Sequence[str] = (), *,
                          anchor: Optional[str] = None,
                          policy: Optional[MergePolicy] = None,
                          screens: Any = None,
                          columns: str = "common",
                          report: Optional[Callable[[str], None]] = None,
                          on_ambiguous_identifier: str = "refuse",
                          ) -> PlateMerge:
    """Merge every attached plate database into one frame, one row per anchor.

    Calls :func:`spacr.multi_database.read_merged` once per chosen table and
    :func:`spacr.merge_tables.roll_up` once per child table. It performs no
    aggregation of its own, and the join type comes from
    :meth:`spacr.merge_tables.MergePolicy.how_for` per table rather than from a
    blanket ``how``.

    :param attachments: any shape :func:`plate_databases` accepts. Plates with
        no database are skipped, not fatal.
    :param tables: the object tables to join. The anchor is added if absent.
    :param anchor: the object type one row of the result represents.
        ``cell`` by default (:data:`spacr.merge_tables.DEFAULT_PRIMARY`),
        overriding ``policy.primary`` when both are given.
    :param policy: the aggregation and join policy -- ``na``, ``overrides``,
        ``consolidate_on_cell``, ``keep_uninfected``.
    :param screens: optional screen label per plate: a mapping keyed by plate,
        or a sequence parallel to the attached databases.
    :param columns: ``'common'`` or ``'union'``, passed to ``read_merged``.
    :param report: called with one line per thing the merge cost, the table
        named.
    :param on_ambiguous_identifier: ``'refuse'`` (default) leaves out a text
        identifier that differs within a roll-up group and names it on
        ``report``; ``'first'`` keeps the first value without a report. See
        :func:`ambiguous_identifiers`. The Measurements tab applies the same
        default, so the headless and Qt merges of one screen agree.
    :returns: the :class:`PlateMerge`.
    :raises MergeRefused: nothing attached, a database that has moved, one
        database attached to two plates, an anchor that is not one row per
        cell, a table missing from some database, or -- from ``read_merged``
        itself -- a plate id repeated inside one screen.
    """
    attached = plate_databases(attachments)
    if not attached:
        raise MergeRefused(
            "no plate has a measurements database attached, so there is "
            "nothing to merge. Drop a .db onto a plate row of the input "
            "table.")

    absent = missing_databases(attached)
    if absent:
        raise MergeRefused(
            "the input table names database(s) that are not on disk: "
            + "; ".join(f"{row.plate} -> {row.path}" for row in absent)
            + ". Re-attach them before the run rather than finding out during "
              "it.")

    by_path: Dict[str, List[str]] = {}
    for row in attached:
        by_path.setdefault(os.path.realpath(row.path), []).append(row.plate)
    doubled = {path: plates for path, plates in by_path.items()
               if len(plates) > 1}
    if doubled:
        raise MergeRefused(
            "one database is attached to more than one plate, so its rows "
            "would be counted twice: "
            + "; ".join(f"{path} on {', '.join(plates)}"
                        for path, plates in sorted(doubled.items()))
            + ". Each plate row carries its own database.")

    policy = policy or MergePolicy()
    if anchor is not None and str(anchor).strip().lower() != policy.primary:
        policy = replace(policy, primary=str(anchor).strip().lower())
    anchor_table = policy.primary
    if not is_one_row_per_cell(anchor_table):
        raise MergeRefused(
            f"{anchor_table!r} cannot be the anchor: every other object table "
            f"is keyed to the CELL, so anchoring on a many-per-cell table "
            f"joins a cell id to an object label. Choose one of "
            f"{list(ONE_ROW_PER_CELL)}.")

    paths = [row.path for row in attached]
    labels = _screen_labels(attached, screens)
    chosen = [table for table in dict.fromkeys([anchor_table, *tables])]
    _refuse_missing_tables(attached, chosen)

    if on_ambiguous_identifier not in ("refuse", "first"):
        raise MergeRefused(
            f"on_ambiguous_identifier must be 'refuse' or 'first', got "
            f"{on_ambiguous_identifier!r}. There is no option that averages a "
            f"file name -- text takes no mean.")

    merged: Optional[pd.DataFrame] = None
    records: List[TableMerge] = []
    refused_identifiers: Dict[str, Tuple[str, ...]] = {}
    for table in chosen:
        plan = describe_merge(paths, table, screens=labels)
        frame = read_merged(paths, table, plan=plan, columns=columns,
                            screens=labels,
                            report=_prefixed_report(report, table))
        dropped = tuple(frame.attrs.get("dropped_columns", ()))

        if table == anchor_table:
            merged = frame.rename(columns={
                column: f"{table}_{column}" for column in frame.columns
                if column not in _UNPREFIXED
                and not str(column).startswith(f"{table}_")})
            records.append(TableMerge(table=table, plan=plan, rows=len(frame),
                                      keys=tuple(_UNPREFIXED), dropped=dropped))
            continue

        link = anchor_column(table)
        if link not in frame.columns:
            note = (f"carries no {link}, so its rows cannot be matched to a "
                    f"{anchor_table}; re-run Measure with the parent mask set")
            LOG.info("%s %s", table, note)
            records.append(TableMerge(table=table, plan=plan, rows=len(frame),
                                      dropped=dropped, note=note))
            continue

        keys = tuple(mt._keys_in(frame)
                     + [column for column in (SCREEN_COLUMN, SOURCE_COLUMN)
                        if column in frame.columns]
                     + [link])
        if is_one_row_per_cell(table):
            skip = set(keys) | {"prcf", "prcfo"}
            rolled = frame.rename(columns={
                column: (column if str(column).startswith(f"{table}_")
                         else f"{table}_{column}")
                for column in frame.columns if column not in skip})
            plan_for_table: Dict[str, str] = {}
        else:
            table_overrides = mt.aggregation_overrides(policy, table)
            plan_for_table = aggregation_plan(frame, overrides=table_overrides,
                                              skip=keys, defaults=policy.defaults)
            ambiguous = ambiguous_identifiers(frame, keys,
                                              plan=plan_for_table,
                                              overrides=table_overrides)
            if ambiguous and on_ambiguous_identifier == "refuse":
                for column, detail in ambiguous.items():
                    line = describe_identifier_refusal(table, column, detail)
                    LOG.info("%s", line.strip())
                    if report is not None:
                        report(line.strip())
                frame = frame.drop(columns=list(ambiguous))
                plan_for_table = {name: how for name, how
                                  in plan_for_table.items()
                                  if name not in ambiguous}
                refused_identifiers[table] = tuple(sorted(ambiguous))
            rolled = roll_up(frame, keys, name=table, policy=policy)
        rolled = rolled.rename(columns={link: OBJECT_COLUMN})

        on = [column for column in list(keys[:-1]) + [OBJECT_COLUMN]
              if column in merged.columns and column in rolled.columns]
        mt._align_keys(merged, rolled, on)
        how = policy.how_for(table)
        before = len(merged)
        merged = merged.merge(rolled, on=on, how=how, validate="one_to_one")
        if how == "inner" and len(merged) < before:
            LOG.info("inner join on %s removed %d of %d %s objects "
                     "(consolidate_on_cell=%s, keep_uninfected=%s)",
                     table, before - len(merged), before, anchor_table,
                     policy.consolidate_on_cell, policy.keep_uninfected)
        records.append(TableMerge(
            table=table, plan=plan, rows=len(frame), keys=keys, how=how,
            rolled_up=not is_one_row_per_cell(table),
            aggregations=plan_for_table, dropped=dropped,
            default_columns=default_aggregated_columns(
                plan_for_table, overrides=policy.overrides)))

    merged = mt._apply_na_policy(merged, policy)
    result = PlateMerge(frame=merged, anchor=anchor_table,
                        attachments=attached, tables=tuple(records))
    merged.attrs["anchor"] = anchor_table
    merged.attrs["tables"] = tuple(record.table for record in records)
    merged.attrs["dropped_columns"] = result.dropped_columns
    merged.attrs["default_aggregation_columns"] = \
        result.default_aggregation_columns
    merged.attrs["screens"] = tuple(
        records[0].plan.screens) if records else ()
    merged.attrs["refused_identifiers"] = dict(refused_identifiers)
    return result


_MERGE_ROW = "merge_row"

_ALWAYS_KEPT: Tuple[str, ...] = _UNPREFIXED + (
    "object_key", "plate", "well", "wellID", "prcfo", "prcf", "prc", "file_name",
    "png_path", "cell_id", "nucleus_id", "pathogen_id", "parent_id")

_CHUNK_ROWS = 50_000

_PART_ROWS = 4096

_TABLE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _merge_sources(paths: Sequence[str], labels: Any) -> Tuple[List[str], List[str], List[str]]:
    """The projects of a merge, located, and one label for each.

    :param paths: project folders, run folders or ``measurements.db`` files.
    :param labels: None, a sequence parallel to ``paths``, or a mapping from
        path to label; a blank label takes the default.
    :returns: the located paths, the default labels and the chosen labels.
    :raises MergeRefused: no project, one project given twice, a label list
        of the wrong length, or one label for two projects.
    """
    from . import store

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    given = [str(path) for path in paths]
    if not given:
        raise MergeRefused("choose at least one project to merge")
    located = [store.locate(path) for path in given]
    seen: Dict[str, str] = {}
    for path, where in zip(given, located):
        real = os.path.realpath(where)
        if real in seen:
            raise MergeRefused(
                f"{seen[real]} and {path} are the same project. Each project "
                "can be merged once, or its rows would be counted twice.")
        seen[real] = path
    automatic = list(source_labels(located))
    if labels is None:
        chosen = list(automatic)
    elif isinstance(labels, Mapping):
        chosen = [str(labels.get(path) or labels.get(where) or "").strip()
                  or label for path, where, label
                  in zip(given, located, automatic)]
    else:
        listed = list(labels)
        if len(listed) != len(given):
            raise MergeRefused(
                f"{len(listed)} label(s) for {len(given)} project(s); give "
                "one label per project, in the same order.")
        chosen = [str(text or "").strip() or label
                  for text, label in zip(listed, automatic)]
    repeated = sorted({label for label in chosen if chosen.count(label) > 1})
    if repeated:
        raise MergeRefused(
            f"more than one project has the label {', '.join(repeated)}; "
            "give each project its own label.")
    return located, automatic, chosen


def _kept_apart(located: Sequence[str], table: str, automatic: Sequence[str],
                chosen: Sequence[str]) -> Tuple[Optional[List[str]], Dict[str, Tuple[str, ...]]]:
    """Whether two projects share a plate name, and the labels that part them.

    :param located: the located project paths.
    :param table: the table whose plate names are compared.
    :param automatic: the default label of each project.
    :param chosen: the label of each project.
    :returns: the screen label per project (None when no plate name repeats)
        and ``{plate: (labels of the projects that hold it)}``.
    """
    plan = describe_merge(list(located), table)
    names = dict(zip(automatic, chosen))
    clashes = {plate: tuple(names.get(label, label) for label in labels)
               for plate, labels in plan.colliding_plates.items()}
    return (list(chosen) if clashes else None), clashes


def _refuse_absent(located: Sequence[str], chosen: Sequence[str],
                   table: str) -> None:
    """Refuse a table that one of the projects does not hold.

    :param located: the located project paths.
    :param chosen: the label of each project.
    :param table: the table to read from every project.
    :raises MergeRefused: naming the projects without it.
    """
    from . import store

    missing = []
    for path, label in zip(located, chosen):
        with store.open(path, verify="none") as opened:
            if table not in opened.tables():
                missing.append(label)
    if missing:
        raise MergeRefused(
            f"{', '.join(missing)} has no table {table!r}; every project in "
            "a merge must hold the table.")


def _pruned(frame: pd.DataFrame, keep: Optional[Sequence[str]]) -> pd.DataFrame:
    """``frame`` with only the ``keep`` columns and the identity columns.

    :param frame: the merged rows.
    :param keep: the columns to keep, or None for every column.
    :returns: the narrowed frame, in its own column order.
    """
    if keep is None:
        return frame
    wanted = set(keep) | set(_ALWAYS_KEPT)
    return frame[[column for column in frame.columns if column in wanted]]


def _merge_keys(columns: Sequence[str]) -> List[str]:
    """The columns that together name one row of a merged object table.

    :param columns: the merged columns.
    :returns: the key columns present, in key order.
    """
    present = set(columns)
    return [column for column in (SOURCE_COLUMN, SCREEN_COLUMN, *IDENTITY,
                                  "timeID", "time_id", OBJECT_COLUMN)
            if column in present]


def _merge_summary(counts: Mapping[str, int], how: str, tables: Sequence[str],
                   chosen: Sequence[str],
                   clashes: Mapping[str, Tuple[str, ...]],
                   dropped: Sequence[str] = (), repeated: int = 0) -> List[str]:
    """What a merge did, in plain sentences for the person who asked for it.

    :param counts: the rows of each project label.
    :param how: ``'stack'`` or ``'join'``.
    :param tables: the merged tables.
    :param chosen: the project labels.
    :param clashes: ``{plate: labels}`` of plate names held by more than one
        project.
    :param dropped: columns left out because only some projects have them.
    :param repeated: rows whose key another row shares.
    :returns: the sentences.
    """
    total = sum(int(counts.get(label, 0)) for label in chosen)
    per = ", ".join(f"{label} {int(counts.get(label, 0)):,}" for label in chosen)
    way = ("rows of " + tables[0] + " stacked" if how == "stack" else
           ", ".join(tables) + " joined on object keys")
    lines = [f"{total:,} rows from {len(chosen)} project(s), {way}: {per}."]
    for plate, labels in sorted(clashes.items()):
        lines.append(
            f"Plate {plate} is in {' and '.join(labels)}. Their rows stay "
            f"apart: {SOURCE_COLUMN} names the project and {SCREEN_COLUMN} "
            "carries the same label, so no two rows share a key.")
    if dropped:
        lines.append(
            f"{len(dropped)} column(s) are in only some projects and were left "
            "out: " + ", ".join(dropped[:6]) + (" ..." if len(dropped) > 6 else ""))
    if repeated:
        lines.append(f"{int(repeated):,} rows repeat a key; the projects hold "
                     "repeated objects.")
    return lines


def _repeated_keys(frame: pd.DataFrame) -> int:
    """How many rows of a merged object table share their key with another.

    :param frame: merged rows.
    :returns: the count; 0 when the rows are not objects.
    """
    keys = _merge_keys(frame.columns)
    if OBJECT_COLUMN not in keys:
        return 0
    return int(frame.duplicated(keys, keep=False).sum())


def merge_projects(paths: Sequence[str], tables: Any, *, how: str = "stack",
                   anchor: Optional[str] = None, labels: Any = None,
                   columns: str = "union",
                   keep: Optional[Sequence[str]] = None,
                   policy: Optional[MergePolicy] = None,
                   limit_per_source: Optional[int] = None,
                   report: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Merge tables from one or more projects into one frame.

    Each project is a spaCR store project or an older ``measurements.db``.
    ``how='stack'`` puts the rows of one table from every project together
    (:func:`spacr.multi_database.read_merged`); ``how='join'`` joins the
    object tables of each project on their object keys, one row per anchor
    object, and stacks the projects (:func:`merge_plate_databases`).

    Every row carries its project label in
    :data:`spacr.multi_database.SOURCE_COLUMN`. When two projects hold a
    plate of the same name, the label is also written to
    :data:`spacr.multi_database.SCREEN_COLUMN`, so the two plates stay apart
    and every row key stays unique; ``frame.attrs['kept_apart']`` names
    those plates.

    :param paths: the projects: store project folders, run folders or
        ``measurements.db`` files.
    :param tables: for ``'stack'``, the one table to stack; for ``'join'``,
        the object tables to join.
    :param how: ``'stack'`` or ``'join'``.
    :param anchor: for ``'join'``, the table one row stands for; the first
        one-row-per-cell table chosen, or ``cell``, when None.
    :param labels: a label per project (a sequence parallel to ``paths`` or
        a mapping from path); None or a blank label uses
        :func:`spacr.multi_database.source_labels`.
    :param columns: ``'union'`` keeps a column any project has, with blanks
        where a project lacks it; ``'common'`` keeps only the columns every
        project has.
    :param keep: the columns to keep, or None for every column; the identity
        columns are always kept.
    :param policy: for ``'join'``, how child tables are rolled up.
    :param limit_per_source: for ``'stack'``, read at most this many rows of
        each project, for a preview.
    :param report: called with one line per thing the merge cost.
    :returns: the merged rows. ``attrs`` holds ``summary`` (plain sentences),
        ``kept_apart``, ``labels``, ``key_columns``, ``how`` and ``tables``.
    :raises MergeRefused: when the merge cannot be done as asked; the
        message says why.
    """
    located, automatic, chosen = _merge_sources(paths, labels)
    names = [tables] if isinstance(tables, str) else list(dict.fromkeys(tables or ()))
    if not names:
        raise MergeRefused("choose the table(s) to merge")
    if how == "stack":
        if len(names) != 1:
            raise MergeRefused(
                "stacking puts the rows of one table from each project "
                "together; choose one table, or join them instead.")
        table = names[0]
        _refuse_absent(located, chosen, table)
        screens, clashes = _kept_apart(located, table, automatic, chosen)
        plan = describe_merge(located, table, screens=screens)
        frame = read_merged(located, table, plan=plan, columns=columns,
                            screens=screens, report=report,
                            limit_per_source=limit_per_source)
        merged_tables = [table]
    elif how == "join":
        if anchor is None:
            anchor = next((t for t in names if is_one_row_per_cell(t)),
                          (policy or MergePolicy()).primary)
        screens, clashes = _kept_apart(located, anchor, automatic, chosen)
        result = merge_plate_databases(
            dict(zip(chosen, located)), [t for t in names if t != anchor],
            anchor=anchor, policy=policy,
            screens=dict(zip(chosen, chosen)) if screens else None,
            columns=columns, report=report)
        frame = result.frame
        merged_tables = [entry.table for entry in result.tables]
    else:
        raise MergeRefused(f"how must be 'stack' or 'join', got {how!r}")
    attrs = dict(frame.attrs)
    if SOURCE_COLUMN in frame.columns:
        frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].map(
            dict(zip(automatic, chosen))).fillna(frame[SOURCE_COLUMN])
    frame = _pruned(frame, keep)
    frame.attrs.update(attrs)
    frame.attrs.update({
        "how": how, "tables": tuple(merged_tables), "labels": tuple(chosen),
        "kept_apart": dict(clashes),
        "key_columns": tuple(_merge_keys(frame.columns)),
        "summary": tuple(_merge_summary(
            frame[SOURCE_COLUMN].value_counts().to_dict()
            if SOURCE_COLUMN in frame.columns else {}, how, merged_tables,
            chosen, clashes, tuple(attrs.get("dropped_columns", ())),
            _repeated_keys(frame)))})
    return frame


def _merge_target(project: Any, name: str) -> str:
    """The store project a merged table is saved in, checked.

    :param project: a store project folder, its ``spacr.duckdb``, or a new or
        empty folder.
    :param name: the new table name.
    :returns: the project folder.
    :raises MergeRefused: for a bad name, a table that exists, or a project
        that holds only an older ``measurements.db``.
    """
    from . import store

    if not _TABLE_NAME.match(str(name or "")):
        raise MergeRefused(
            f"{name!r} cannot be a table name; use letters, digits and _, "
            "starting with a letter.")
    folder = os.path.abspath(os.fspath(project))
    if os.path.basename(folder) == store.CATALOG:
        folder = os.path.dirname(folder)
    if os.path.isfile(os.path.join(folder, store.CATALOG)):
        with store.open(folder, verify="none") as opened:
            if name in opened.tables():
                raise MergeRefused(
                    f"{folder} already has a table named {name}; choose "
                    "another name.")
        return folder
    try:
        found = store.locate(folder)
    except FileNotFoundError:
        found = None
    if found is not None and not os.path.isdir(found):
        raise MergeRefused(
            f"{folder} holds an older measurements.db, which cannot take a "
            "new table. Save into a store project or an empty folder.")
    if found is not None and os.path.isfile(os.path.join(found, store.CATALOG)):
        return _merge_target(found, name)
    return folder


def _stacked_chunks(plan: MergePlan, table: str, columns: str,
                    keep: Optional[Sequence[str]], chunk_rows: int,
                    cancelled: Optional[Callable[[], bool]]):
    """The rows of a stack merge, a chunk at a time, as :func:`merge_projects` gives them.

    A store project is paged by key (:meth:`spacr.store.Store.page`); an
    older file by ``rowid``, through :func:`spacr.store.read_frame`. Only
    the columns the merge keeps are read.

    :param plan: the merge plan, its screens applied.
    :param table: the table.
    :param columns: ``'union'`` or ``'common'``.
    :param keep: the columns to keep, or None.
    :param chunk_rows: rows per chunk.
    :param cancelled: called before each chunk; True stops the merge.
    :yields: stamped chunks with canonical column names.
    :raises MergeCancelled: when ``cancelled()`` answered true.
    """
    from . import schema
    from .multi_database import _ALWAYS_KEPT as _PLAN_KEPT
    from .multi_database import _canonical_columns, _stamped
    from .store_browse import StoreDb

    held = (set(plan.common_columns) | set(_PLAN_KEPT)
            if columns == "common" else None)
    for source in plan.sources:
        db = StoreDb(source.path)
        names = _canonical_columns(source.columns)
        wanted = set(keep) | set(_ALWAYS_KEPT) if keep is not None else None
        read = [stored for stored, canonical in names.items()
                if (wanted is None or canonical in wanted)
                and (held is None or canonical in held)]
        for raw in _source_chunks(db, table, read, chunk_rows, cancelled):
            raw = raw[[c for c in source.columns if c in raw.columns]]
            frame = schema.canonicalise_frame(raw, report=None, warn=None,
                                              repair_plate_ids=True)
            yield _pruned(_stamped(frame, source, plan, held, "refuse"), keep)


def _source_chunks(db: Any, table: str, columns: List[str], chunk_rows: int,
                   cancelled: Optional[Callable[[], bool]]):
    """One table of one project, ``chunk_rows`` rows at a time, keyset paged.

    :param db: a :class:`spacr.store_browse.StoreDb` on the project.
    :param table: the table.
    :param columns: the stored columns to read.
    :param chunk_rows: rows per chunk.
    :param cancelled: called before each chunk; True stops the merge.
    :yields: DataFrames of at most ``chunk_rows`` rows.
    :raises MergeCancelled: when ``cancelled()`` answered true.
    """
    from . import store

    columns = list(columns) or db.columns(table)

    def _check():
        """Stop when the caller asked to."""
        if cancelled is not None and cancelled():
            raise MergeCancelled(f"stopped while reading {table}; nothing was saved.")

    if db.kind == "store":
        if not db.key_columns(table)[2:]:
            for frame in db.store._frames(table, chunksize=chunk_rows):
                _check()
                if len(frame):
                    yield frame[[c for c in frame.columns if c in set(columns)]]
            return
        cursor, first = None, True
        while first or cursor is not None:
            _check()
            frame, cursor = db.store.page(table, columns=columns, after=cursor,
                                          limit=chunk_rows)
            first = False
            if len(frame):
                yield frame
        return
    kind, keys = db.row_key(table)
    keyset = db.kind == "sqlite" and kind == "rowid" and len(keys) == 1
    after, offset = None, 0
    while True:
        _check()
        if keyset:
            sql = db.chunk_sql(table, columns, keys, after=after is not None)
            params = ([after] if after is not None else []) + [chunk_rows]
        else:
            sql = db.chunk_sql(table, columns, [], use_offset=True)
            params = [chunk_rows, offset]
        frame = store.read_frame(db.store, sql, params=params)
        if not len(frame):
            return
        if keyset:
            after = int(frame.iloc[-1, 0])
            frame = frame.iloc[:, 1:]
        offset += len(frame)
        yield frame
        if len(frame) < chunk_rows:
            return


def _partitions_of(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """The store plate and well of each merged row.

    A row read from a store project keeps the plate and well it was stored
    under (``plate`` and ``wellID`` once the column names are canonical); a
    row from an older file takes ``plateID`` and ``rowID`` + ``columnID``.
    Rows with neither go to ``_``.

    :param frame: merged rows.
    :returns: the plate and the well of each row, as text.
    """
    blank = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    plates = frame["plate"].astype(object) if "plate" in frame.columns else blank
    wells = frame["well"].astype(object) if "well" in frame.columns else blank
    if "wellID" in frame.columns:
        wells = wells.where(wells.notna(), frame["wellID"])
    if "plateID" in frame.columns:
        plates = plates.where(plates.notna(), frame["plateID"])
    if {"rowID", "columnID"} <= set(frame.columns):
        derived = frame["rowID"].astype(str) + frame["columnID"].astype(str)
        derived = derived.where(frame["rowID"].notna() & frame["columnID"].notna())
        wells = wells.where(wells.notna(), derived)
    from .measurement_store import _text

    return plates.map(_text).astype(str), wells.map(_text).astype(str)


def _prepared(frame: pd.DataFrame, start: int):
    """A merged chunk as the store writes it, and the partition of each row.

    :param frame: merged rows.
    :param start: the :data:`_MERGE_ROW` number of the first row.
    :returns: ``(Arrow table, plates, wells)``.
    """
    from .measurement_store import _arrow, _clean

    plates, wells = _partitions_of(frame)
    data = frame.drop(columns=[c for c in ("plate", "well", "crop_store",
                                           "crop_index") if c in frame.columns])
    data = data.reset_index(drop=True)
    data.attrs = {}
    data[_MERGE_ROW] = range(start, start + len(data))
    return _arrow(_clean(data)), plates.to_numpy(), wells.to_numpy()


def _merged_schema(schemas: Sequence[Any]):
    """One schema for every chunk: promoted where types differ, text where they clash.

    :param schemas: the Arrow schema of each chunk.
    :returns: the schema.
    """
    import pyarrow as pa

    order: Dict[str, List[Any]] = {}
    for found in schemas:
        for field_ in found:
            order.setdefault(field_.name, []).append(field_.type)
    fields = []
    for name, types in order.items():
        try:
            kind = pa.unify_schemas([pa.schema([pa.field(name, t)]) for t in types],
                                    promote_options="permissive").field(name).type
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            kind = pa.string()
        fields.append(pa.field(name, pa.string() if pa.types.is_null(kind) else kind))
    return pa.schema(fields)


def _write_merged(project: str, name: str, chunks: Callable[[], Any], *,
                  run_id: str, progress: Optional[Callable[[str, int, int], None]],
                  total: int) -> int:
    """Write merged chunks as one new store table, a part per plate and well.

    The chunks are read twice: once for the one schema every part shares,
    then to write. Each part is written by :func:`spacr.store.writer`;
    nothing is visible until every part is finished and registered.

    :param project: the store project folder.
    :param name: the new table.
    :param chunks: called with no argument, returns an iterator of chunks.
    :param run_id: the run the parts belong to.
    :param progress: called ``progress(stage, done, total)``.
    :param total: the expected rows, for ``progress``.
    :returns: the rows written.
    """
    import numpy as np

    from . import store
    from .measurement_store import _conform

    seen, start = [], 0
    for frame in chunks():
        seen.append(_prepared(frame, start)[0].schema)
        start += len(frame)
        if progress is not None:
            progress(f"checking the columns of {name}", start, total)
    if not seen:
        raise MergeRefused("the merge has no rows to save")
    layout = _merged_schema(seen)
    keys = tuple(c for c in (*_merge_keys(layout.names), "object_key", _MERGE_ROW)
                 if c in layout.names)
    writers: Dict[Tuple[str, str], Any] = {}
    written = 0
    try:
        for frame in chunks():
            data, plates, wells = _prepared(frame, written)
            data = _conform(data, layout)
            groups = pd.DataFrame({"plate": plates, "well": wells}).groupby(
                ["plate", "well"], sort=False).indices
            for (plate, well), rows in groups.items():
                part = writers.get((plate, well))
                if part is None:
                    part = store.writer(project, name, plate=plate, well=well,
                                        run_id=run_id, sort_by=(_MERGE_ROW,),
                                        key_columns=keys, schema=layout,
                                        row_group_rows=_PART_ROWS)
                    writers[(plate, well)] = part
                part.append(data.take(np.asarray(rows, dtype=np.int64)))
            written += len(frame)
            if progress is not None:
                progress(f"writing {name}", written, total)
        for part in writers.values():
            part.finish()
    except BaseException:
        for part in writers.values():
            if part.manifest is None:
                part.abort()
        raise
    return written


def save_merge(project: Any, name: str, paths: Sequence[str], tables: Any, *,
               how: str = "stack", anchor: Optional[str] = None,
               labels: Any = None, columns: str = "union",
               keep: Optional[Sequence[str]] = None,
               policy: Optional[MergePolicy] = None,
               chunk_rows: int = _CHUNK_ROWS,
               progress: Optional[Callable[[str, int, int], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Merge tables from one or more projects and save them as a new table.

    The merged table is the one :func:`merge_projects` returns for the same
    arguments, with two more columns: ``merge_row`` numbers the rows in the
    order they were written, and ``plate`` and ``well`` say which store
    part a row is in. A stack merge streams: each project is read
    ``chunk_rows`` rows at a time, keyset paged, reading only the kept
    columns, so a table larger than memory can be stacked. A join merge
    rolls each child table up onto its anchor and holds the tables while it
    does.

    :param project: the store project the new table goes in: a store
        project folder, its ``spacr.duckdb``, or a new or empty folder.
    :param name: the new table name: letters, digits and ``_``.
    :param paths: the projects to merge; see :func:`merge_projects`.
    :param tables: the table(s) to merge; see :func:`merge_projects`.
    :param how: ``'stack'`` or ``'join'``.
    :param anchor: for ``'join'``, the table one row stands for.
    :param labels: a label per project; see :func:`merge_projects`.
    :param columns: ``'union'`` or ``'common'``.
    :param keep: the columns to keep, or None for every column.
    :param policy: for ``'join'``, how child tables are rolled up.
    :param chunk_rows: rows read from a project at a time.
    :param progress: called ``progress(stage, done, total)`` from the
        thread this runs on.
    :param cancelled: called between chunks; True stops the merge and
        nothing is saved.
    :returns: ``{"project", "table", "rows", "rows_per_source",
        "kept_apart", "key_columns", "summary"}``.
    :raises MergeRefused: when the merge cannot be done or saved as asked.
    :raises MergeCancelled: when ``cancelled()`` answered true.
    """
    import json
    import uuid

    from . import store

    target = _merge_target(project, name)
    located, automatic, chosen = _merge_sources(paths, labels)
    names = [tables] if isinstance(tables, str) else list(dict.fromkeys(tables or ()))
    if how == "stack":
        if len(names) != 1:
            raise MergeRefused(
                "stacking puts the rows of one table from each project "
                "together; choose one table, or join them instead.")
        table = names[0]
        _refuse_absent(located, chosen, table)
        screens, clashes = _kept_apart(located, table, automatic, chosen)
        plan = describe_merge(located, table, screens=screens)
        relabel = dict(zip(automatic, chosen))
        total = plan.total_rows
        dropped = plan.dropped_columns if columns == "common" else ()

        def _chunks():
            """The stacked rows, relabelled, a chunk at a time."""
            for frame in _stacked_chunks(plan, table, columns, keep,
                                         int(chunk_rows), cancelled):
                frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].map(relabel)
                yield frame
        merged_tables = [table]
    else:
        merged = merge_projects(located, names, how=how, anchor=anchor,
                                labels=chosen, columns=columns, keep=keep,
                                policy=policy)
        clashes = dict(merged.attrs.get("kept_apart", {}))
        merged_tables = list(merged.attrs.get("tables", names))
        dropped = tuple(merged.attrs.get("dropped_columns", ()))
        total = len(merged)
        step = max(1, int(chunk_rows))

        def _chunks():
            """The joined rows, a slice at a time."""
            for begin in range(0, len(merged), step):
                if cancelled is not None and cancelled():
                    raise MergeCancelled("stopped while saving; nothing was saved.")
                yield merged.iloc[begin:begin + step]
    run_id = f"merge-{uuid.uuid4().hex[:12]}"
    rows = _write_merged(target, name, _chunks, run_id=run_id,
                         progress=progress, total=total)
    settings = {"merge": how, "tables": merged_tables, "labels": chosen,
                "projects": located, "columns": columns,
                "keep": list(keep) if keep is not None else None}
    store.register(target, run={"run_id": run_id,
                                "settings": json.loads(json.dumps(settings))})
    with store._SHARED_LOCK:
        held = store._SHARED.get(target)
        if held is not None:
            held["checked"] = float("-inf")
    with store.open(target, verify="none") as opened:
        columns_saved = list(opened.columns(name)["name"])
        found = opened.sql(f'SELECT {SOURCE_COLUMN} AS label, COUNT(*) AS n '
                           f'FROM "{name}" GROUP BY {SOURCE_COLUMN}')
        keys = _merge_keys(columns_saved)
        repeated = 0
        if OBJECT_COLUMN in keys:
            listed = ", ".join(f'"{key}"' for key in keys)
            repeated = int(opened.sql(
                f'SELECT COALESCE(SUM(n), 0) AS n FROM (SELECT COUNT(*) AS n '
                f'FROM "{name}" GROUP BY {listed} HAVING COUNT(*) > 1)')["n"].iloc[0])
    counts = {str(row.label): int(row.n) for row in found.itertuples()}
    summary = _merge_summary(counts, how, merged_tables, chosen, clashes,
                             tuple(dropped), repeated)
    summary.append(f"Saved as {name} in {target}.")
    return {"project": target, "table": name, "rows": int(rows),
            "rows_per_source": {label: counts.get(label, 0) for label in chosen},
            "kept_apart": dict(clashes),
            "key_columns": tuple(keys),
            "summary": tuple(summary)}
