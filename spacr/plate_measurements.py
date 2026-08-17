"""One merged measurements frame from the plate rows of the input table.

Instruction 130. A row of the regression input table is one PLATE: its score
CSV, its count CSV and now its measurements database. This module is the
headless half of that feature -- given ``{plate: database path}``, the tables
the user ticked and the anchor they chose, it returns the merged frame.

NOTHING HERE IS NEW MERGE LOGIC, AND THAT IS THE POINT. spaCR already has
four places that join measurement tables and one of them
(``io._read_and_join_tables``) aggregates every numeric column with ``mean``,
which turns four pathogens' total area into an average area and a MINIMUM
into a mean of minima. A fifth would be a fifth answer. So:

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

WHY THE TWO EXISTING FUNCTIONS DO NOT COMPOSE BY THEMSELVES
-----------------------------------------------------------
``read_merged`` is *many databases, one table*. ``merge_tables`` is *one
database, many tables* -- and it takes a PATH, so it cannot be handed a frame
that has already been stacked across databases. This feature needs both, so
the composition is: read each chosen table across every database, then roll
the children onto the anchor.

The roll-up keys are the load-bearing detail. They are the identity columns
PLUS ``screenID`` PLUS ``source_database`` PLUS the child's anchor column.
Leave ``screenID`` out and two screens that legitimately share ``plate1``
(instruction 122, which :func:`~spacr.multi_database.describe_merge`
deliberately permits) collapse into one parent -- reintroducing, one layer up,
exactly the pooling ``multi_database`` exists to prevent.

WHAT A CALLER MUST SHOW THE USER
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
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

import pandas as pd

from . import merge_tables as mt
# The structural constants are imported by value; AGGREGATION_RULES and
# DEFAULT_AGGREGATION are read through `mt.` at CALL time, deliberately. They
# are the maintainer's living decision about what each measurement means, and
# a module that snapshots them at import is a module that can disagree with
# them.
from .merge_tables import (IDENTITY, OBJECT_COLUMN, OBJECT_TABLES, PNG_TABLE,
                           MergePolicy, aggregation_plan, mergeable_tables,
                           roll_up)
from .multi_database import (SCREEN_COLUMN, SOURCE_COLUMN, MergePlan,
                             MergeRefused, describe_merge, read_merged)
from .object_roles import ONE_ROW_PER_CELL, anchor_column, is_one_row_per_cell

LOG = logging.getLogger("spacr.plate_measurements")

#: Columns that name a row rather than measure it, and are therefore never
#: prefixed with the table they came from. ``screenID`` and
#: ``source_database`` are here because they are JOIN KEYS in this
#: composition: prefix them and the roll-up no longer meets the anchor.
_UNPREFIXED: Tuple[str, ...] = IDENTITY + (
    OBJECT_COLUMN, SCREEN_COLUMN, SOURCE_COLUMN)


@dataclass(frozen=True)
class PlateDatabase:
    """One plate row of the input table and the database attached to it."""

    plate: str
    path: str

    @property
    def exists(self) -> bool:
        """Whether the file is still where the input table says it is.

        Checked before the run rather than during it: instruction 130 asks
        that a database that has been moved is named up front, not four
        minutes into a regression.
        """
        return bool(self.path) and os.path.isfile(self.path)


@dataclass(frozen=True)
class TableMerge:
    """What one chosen table contributed to the merge, and how.

    One of these per table the user ticked, including the anchor. It is the
    record a panel reads to answer "what happened to my numbers": which
    aggregation each column got, whether the table was rolled up or joined
    directly, and which join type its cardinality earned it.
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

        The same rule :func:`spacr.merge_tables.roll_up` applies: a join key
        keeps its name, a column that already starts with the table's name is
        left alone -- ``nucleus_area`` must not become ``nucleus_nucleus_area``
        -- and everything else is prefixed with the table it measures.
        """
        name = str(column)
        if name in self.keys or name in _UNPREFIXED:
            return name
        if name.startswith(f"{self.table}_"):
            return name
        return f"{self.table}_{name}"


@dataclass(frozen=True)
class PlateMerge:
    """The merged frame, and everything the Measurements tab has to say about it."""

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
        """Anchor rows each database HELD, before any join dropped any."""
        for entry in self.tables:
            if entry.table == self.anchor:
                return {source.label: int(source.rows)
                        for source in entry.plan.sources}
        return {}

    @property
    def rows_per_source(self) -> Dict[str, int]:
        """Anchor rows each database contributed to the FINAL frame.

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

        :data:`spacr.merge_tables.DEFAULT_AGGREGATION` is MEAN and the comment
        beside it says why -- but a measurement nobody thought about is exactly
        the one worth naming, because it is the one where the default is most
        likely to be answering a different question.
        """
        return tuple(sorted(
            entry.merged_column(column)
            for entry in self.tables for column in entry.default_columns))

    @property
    def shared_plates_across_screens(self) -> Dict[str, Tuple[str, ...]]:
        """Plate ids that appear in more than one screen. NOT a collision.

        Two screens sharing a guide library both have ``plate1``, and once
        ``screenID`` is in the frame those are two identities. Reported anyway,
        because a user who did not mean to run two screens still needs to see
        that they did.
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
        for index, entry in enumerate(attachments or ()):
            if isinstance(entry, PlateDatabase):
                pairs.append((entry.plate, entry.path))
            elif isinstance(entry, Mapping):
                # The input table's own row shape, so a caller can pass
                # `PairedFileTableWidget.get_value()` straight through.
                pairs.append((entry.get("plate") or "",
                              entry.get("database") or entry.get("path") or ""))
            else:
                plate, path = entry
                pairs.append((plate, path))
    return [PlateDatabase(plate=str(plate or f"row {index + 1}"),
                          path=str(path or "").strip())
            for index, (plate, path) in enumerate(pairs)]


def plate_databases(attachments: Any) -> Tuple[PlateDatabase, ...]:
    """The plate rows that HAVE a database, in the order the table lists them.

    :param attachments: ``{plate: path}``, a sequence of :class:`PlateDatabase`,
        a sequence of ``(plate, path)`` pairs, or the input table's own rows
        (mappings carrying ``plate`` and ``database``).
    :returns: one :class:`PlateDatabase` per attached row. Rows with no
        database are left out rather than carried as blanks -- see
        :func:`unattached_plates`, which is where they are named.
    """
    return tuple(row for row in _rows(attachments) if row.path)


def unattached_plates(attachments: Any) -> Tuple[str, ...]:
    """The plates with no database, which is legal and must be said out loud.

    The regression runs on scores and counts; the database is what makes the
    Measurements tab possible for that plate. Its absence disables that plate
    there rather than failing the run -- so the plate is listed, and the
    listing is what stops it looking like an omission.
    """
    return tuple(row.plate for row in _rows(attachments) if not row.path)


def missing_databases(attachments: Any) -> Tuple[PlateDatabase, ...]:
    """Attached databases that are not on disk, named BEFORE the run starts."""
    return tuple(row for row in plate_databases(attachments) if not row.exists)


def available_tables(attachments: Any) -> Tuple[str, ...]:
    """The object tables present in EVERY attached database.

    The intersection, not the union, and this is not a nicety:
    :func:`spacr.multi_database.describe_merge` raises a bare
    ``sqlite3.OperationalError: no such table`` when one database lacks the
    chosen table, which reaches a user as a crash rather than as a choice they
    were never offered.

    :returns: the tables in :data:`spacr.merge_tables.OBJECT_TABLES` order, so
        the picker's order is the registry's order rather than sqlite's, and
        so a sixth object kind reaches this list by being declared once.
        ``png_list`` is not offered: it holds one row per CROP rather than per
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
    """The columns in ``plan`` that got the default because NO rule matched.

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


def _screen_labels(attached: Sequence[PlateDatabase], screens: Any) -> Any:
    """One screen label per database, from a per-PLATE mapping or a sequence.

    A mapping keyed by plate is the shape the input table has, and it is the
    recommended one: the label is written into
    :data:`spacr.multi_database.SCREEN_COLUMN`, where it stays a dimension you
    can block on -- rather than into the plate id, where ``on_collision``
    ``'qualify'`` would put it and where it becomes a string to be parsed back
    apart (instruction 122). This module never qualifies.
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
    :param anchor: the object a row of the result MEANS. ``cell`` by default
        (:data:`spacr.merge_tables.DEFAULT_PRIMARY`), overriding
        ``policy.primary`` when both are given.
    :param policy: the aggregation and join policy -- ``na``, ``overrides``,
        ``consolidate_on_cell``, ``keep_uninfected``.
    :param screens: optional screen label per plate: a mapping keyed by plate,
        or a sequence parallel to the attached databases.
    :param columns: ``'common'`` or ``'union'``, passed to ``read_merged``.
    :param report: called with one line per thing the merge cost, the table
        named.
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
        # A nucleus carries its parent in `cell_id` and a pathogen carries its
        # own identity in `object_label`. Anchoring on a many-per-cell table
        # would join one to the other and match a cell id against a pathogen
        # label -- a join on a coincidence, which returns rows.
        raise MergeRefused(
            f"{anchor_table!r} cannot be the anchor: every other object table "
            f"is keyed to the CELL, so anchoring on a many-per-cell table "
            f"joins a cell id to an object label. Choose one of "
            f"{list(ONE_ROW_PER_CELL)}.")

    paths = [row.path for row in attached]
    labels = _screen_labels(attached, screens)
    chosen = [table for table in dict.fromkeys([anchor_table, *tables])]
    _refuse_missing_tables(attached, chosen)

    merged: Optional[pd.DataFrame] = None
    records: List[TableMerge] = []
    for table in chosen:
        plan = describe_merge(paths, table, screens=labels)
        # `on_collision` is deliberately left at 'refuse'. 'qualify' rewrites
        # plate1 to `<label>-plate1`, which makes the keys unique by hiding
        # the screen inside the plate id and stops it being analysable
        # (instruction 122). Two screens sharing a plate id do not reach this
        # branch at all once `screens=` is passed.
        frame = read_merged(paths, table, plan=plan, columns=columns,
                            screens=labels,
                            report=_prefixed_report(report, table))
        # What the read cost, from the read itself: `columns='union'` drops
        # nothing, and the plan's own list would say otherwise.
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
            # Measured without a parent mask: the roll-up is not empty, it is
            # UNDEFINED. Named and skipped, exactly as `merge_tables` does --
            # one unlinkable table must not cost the user the others.
            note = (f"carries no {link}, so its rows cannot be matched to a "
                    f"{anchor_table}; re-run Measure with the parent mask set")
            LOG.info("%s %s", table, note)
            records.append(TableMerge(table=table, plan=plan, rows=len(frame),
                                      dropped=dropped, note=note))
            continue

        keys = tuple([column for column in IDENTITY if column in frame.columns]
                     + [column for column in (SCREEN_COLUMN, SOURCE_COLUMN)
                        if column in frame.columns]
                     + [link])
        if is_one_row_per_cell(table):
            # One row per cell already: aggregating it is not wrong so much as
            # meaningless, and it would put the table's own measurements
            # through the sum/mean rules meant for a GROUP of children.
            skip = set(keys) | {"prcf", "prcfo"}
            rolled = frame.rename(columns={
                column: (column if str(column).startswith(f"{table}_")
                         else f"{table}_{column}")
                for column in frame.columns if column not in skip})
            plan_for_table: Dict[str, str] = {}
        else:
            # What the panel shows and what actually happens, from the same
            # function, so they cannot disagree.
            plan_for_table = aggregation_plan(frame, overrides=policy.overrides,
                                              skip=keys)
            rolled = roll_up(frame, keys, name=table, policy=policy)
        rolled = rolled.rename(columns={link: OBJECT_COLUMN})

        on = [column for column in list(keys[:-1]) + [OBJECT_COLUMN]
              if column in merged.columns and column in rolled.columns]
        # One dtype policy for join keys, not two: `_align_keys` is where
        # spaCR decided that a plate called `1` read as an integer from one
        # table and a string from another is the same plate.
        mt._align_keys(merged, rolled, on)
        how = policy.how_for(table)
        before = len(merged)
        merged = merged.merge(rolled, on=on, how=how)
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
    # Carried on the frame so they cannot be separated from the data they
    # describe -- the same reason `read_merged` does it.
    merged.attrs["anchor"] = anchor_table
    merged.attrs["tables"] = tuple(record.table for record in records)
    merged.attrs["dropped_columns"] = result.dropped_columns
    merged.attrs["default_aggregation_columns"] = \
        result.default_aggregation_columns
    merged.attrs["screens"] = tuple(
        records[0].plan.screens) if records else ()
    return result
