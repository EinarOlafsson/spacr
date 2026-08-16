"""Load several measurement databases as one frame, without pooling them.

Instruction 109. A screen acquired as three plates was three sessions in the
Gate Editor and three separate UMAPs, and the comparison a user actually
wants -- all of it in one embedding, one gate, one figure -- could not be made
inside spaCR at all.

WHY THIS IS ONE MODULE AND NOT TWO SCREENS' PRIVATE CODE. The hard part of
merging databases is not concatenation, it is the three ways it goes wrong
quietly, and each of them has exactly one right answer that both screens must
give:

* two plates named ``plate1`` in two files are two plates, and
  ``plate1_r1_c1_f1_o1`` is one key. Pool them and every per-well number
  afterwards is computed over two experiments at once, with nothing on screen
  to say so.
* databases written by different spaCR versions have different columns, and
  "just intersect" silently drops measurements the user came to compare.
* a row that has forgotten which file it came from cannot answer the single
  most valuable question a merged view can ask, which is whether the clusters
  are biology or batch.

Two private implementations would disagree about all three, and the third
module to need this would make a third set of answers.

THE DESIGN, in one line: describe the merge before performing it, refuse
ambiguity rather than resolving it silently, and never lose provenance.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import schema

__all__ = [
    "SOURCE_COLUMN",
    "MergePlan",
    "SourceSummary",
    "describe_merge",
    "read_merged",
    "MergeRefused",
]

#: The column every merged frame carries, naming the database a row came from.
#:
#: Not optional, and not a display detail: a merged UMAP whose clusters turn
#: out to be the three plates is the single most important thing this feature
#: can show, and it cannot show it if provenance was dropped at load time.
SOURCE_COLUMN = "source_database"


class MergeRefused(RuntimeError):
    """The merge would have silently changed what the numbers mean."""


@dataclass(frozen=True)
class SourceSummary:
    """What one database contributes to a merge."""

    path: str
    label: str
    table: str
    rows: int
    columns: Tuple[str, ...]
    plates: Tuple[str, ...]

    @property
    def name(self) -> str:
        """Short name for a legend or a chip."""
        return self.label


@dataclass(frozen=True)
class MergePlan:
    """What a merge WOULD do, computed before anything is concatenated.

    This exists so a user is told what they are about to lose. The column set
    a merge produces IS the analysis they are about to run, and finding out
    afterwards that half the measurements were dropped is finding out too
    late.
    """

    sources: Tuple[SourceSummary, ...]
    common_columns: Tuple[str, ...]
    #: Present in some sources and not others: ``{column: (labels that have it)}``
    partial_columns: Mapping[str, Tuple[str, ...]]
    #: Plate ids that appear in more than one source: ``{plate: (labels,)}``
    colliding_plates: Mapping[str, Tuple[str, ...]]

    @property
    def total_rows(self) -> int:
        return sum(source.rows for source in self.sources)

    @property
    def has_collisions(self) -> bool:
        return bool(self.colliding_plates)

    def describe(self) -> str:
        """A human-readable summary, for a dialog or a log line."""
        lines = [
            f"{len(self.sources)} databases, {self.total_rows:,} rows",
            f"  {len(self.common_columns)} columns in all of them",
        ]
        if self.partial_columns:
            lines.append(
                f"  {len(self.partial_columns)} columns in only some: "
                + ", ".join(sorted(self.partial_columns)[:6])
                + (" ..." if len(self.partial_columns) > 6 else ""))
        if self.colliding_plates:
            lines.append(
                f"  {len(self.colliding_plates)} plate id(s) appear in more "
                "than one database: "
                + ", ".join(sorted(self.colliding_plates)))
        return "\n".join(lines)


def _label_for(path: str, used: Sequence[str]) -> str:
    """A short, unique name for a database.

    The file's own stem, because that is what the user called it and what
    they will look for in a legend. Disambiguated with the parent directory
    when two files share a stem, which is common -- every plate's database is
    often ``measurements.db`` under a differently-named folder.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if stem not in used:
        return stem
    parent = os.path.basename(os.path.dirname(str(path))) or "?"
    candidate = f"{parent}/{stem}"
    if candidate not in used:
        return candidate
    index = 2
    while f"{candidate} ({index})" in used:
        index += 1
    return f"{candidate} ({index})"


def _table_columns(path: str, table: str) -> List[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in rows]


def _row_count(path: str, table: str) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        return int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _plates(path: str, table: str, columns: Sequence[str]) -> List[str]:
    if schema.PLATE_KEY not in columns:
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            f'SELECT DISTINCT "{schema.PLATE_KEY}" FROM "{table}"').fetchall()
    return sorted(str(row[0]) for row in rows if row[0] is not None)


def describe_merge(paths: Sequence[str], table: str) -> MergePlan:
    """Work out what merging ``paths`` would produce, WITHOUT reading rows.

    Reads only sqlite metadata and the distinct plate ids, so this is cheap
    enough to run while the user is still choosing files -- which is the
    point, because the answer has to arrive before they commit.

    :param paths: measurement databases, in the order the user added them.
    :param table: the table to merge, e.g. ``'cell'``.
    :returns: the :class:`MergePlan`.
    """
    summaries: List[SourceSummary] = []
    labels: List[str] = []
    for path in paths:
        label = _label_for(path, labels)
        labels.append(label)
        columns = _table_columns(path, table)
        summaries.append(SourceSummary(
            path=str(path), label=label, table=table,
            rows=_row_count(path, table),
            columns=tuple(columns),
            plates=tuple(_plates(path, table, columns)),
        ))

    if not summaries:
        return MergePlan((), (), {}, {})

    column_sets = [set(s.columns) for s in summaries]
    common = set.intersection(*column_sets) if column_sets else set()
    everything = set.union(*column_sets) if column_sets else set()
    partial = {
        column: tuple(s.label for s in summaries if column in s.columns)
        for column in sorted(everything - common)
    }

    seen: Dict[str, List[str]] = {}
    for source in summaries:
        for plate in source.plates:
            seen.setdefault(plate, []).append(source.label)
    colliding = {plate: tuple(labels_)
                 for plate, labels_ in sorted(seen.items())
                 if len(labels_) > 1}

    return MergePlan(tuple(summaries), tuple(sorted(common)), partial,
                     colliding)


def _qualified_plate(plate: Any, label: str) -> str:
    """``plate`` made unique to ``label``, reversibly.

    Uses the key escape, so the qualified id is still a legal plate token --
    it cannot introduce the separator and split the key into an extra
    component, which is the bug ``schema.KEY_ESCAPES`` exists for.
    """
    return (f"{schema.escape_filename_component(label)}"
            f"-{schema.escape_filename_component(str(plate))}")


def read_merged(paths: Sequence[str],
                table: str,
                *,
                plan: Optional[MergePlan] = None,
                columns: str = "common",
                on_collision: str = "refuse",
                limit_per_source: Optional[int] = None) -> pd.DataFrame:
    """Read ``table`` from every path and return one frame.

    :param paths: measurement databases.
    :param table: the table to read from each.
    :param plan: a plan from :func:`describe_merge`; recomputed if omitted.
    :param columns: ``'common'`` keeps only columns present in every source
        (safe, and drops); ``'union'`` keeps everything and leaves nulls where
        a source did not have it (keeps, and changes what "missing" means).
        There is no default that is right for both, so the caller states it
        and the plan tells the user which columns each choice costs.
    :param on_collision: what to do when a plate id appears in more than one
        database. ``'refuse'`` raises :class:`MergeRefused`; ``'qualify'``
        prefixes each colliding plate with its source label. There is
        deliberately no option that pools them.
    :param limit_per_source: row cap per database, for previews.
    :returns: one frame carrying :data:`SOURCE_COLUMN`.
    :raises MergeRefused: on colliding plates under ``on_collision='refuse'``,
        or an unknown option.
    """
    if columns not in ("common", "union"):
        raise MergeRefused(
            f"columns must be 'common' or 'union', got {columns!r}")
    if on_collision not in ("refuse", "qualify"):
        raise MergeRefused(
            f"on_collision must be 'refuse' or 'qualify', got "
            f"{on_collision!r}. There is no option that merges two plates of "
            f"the same name into one -- that is the failure this refuses.")

    plan = plan or describe_merge(paths, table)
    if not plan.sources:
        return pd.DataFrame()

    if plan.colliding_plates and on_collision == "refuse":
        detail = "; ".join(
            f"{plate!r} in {', '.join(labels)}"
            for plate, labels in sorted(plan.colliding_plates.items()))
        raise MergeRefused(
            "the same plate id appears in more than one database, so merging "
            "would compute every per-well number over two experiments at "
            f"once: {detail}. Rename the plates, drop one database, or pass "
            "on_collision='qualify' to prefix each with its source.")

    keep = set(plan.common_columns) if columns == "common" else None
    frames = []
    for source in plan.sources:
        query = f'SELECT * FROM "{source.table}"'
        if limit_per_source:
            query += f" LIMIT {int(limit_per_source)}"
        with sqlite3.connect(f"file:{source.path}?mode=ro", uri=True,
                             timeout=30) as db:
            frame = pd.read_sql_query(query, db)
        # One canonicaliser, so a column spelled differently across two
        # databases becomes one column rather than two. Case-folded, so this
        # cannot produce a frame SQLite will refuse (instruction 100, C1).
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
        if keep is not None:
            frame = frame[[c for c in frame.columns if c in keep]]
        if (on_collision == "qualify" and plan.colliding_plates
                and schema.PLATE_KEY in frame.columns):
            colliding = set(plan.colliding_plates)
            frame[schema.PLATE_KEY] = [
                _qualified_plate(value, source.label)
                if str(value) in colliding else value
                for value in frame[schema.PLATE_KEY]
            ]
        frame[SOURCE_COLUMN] = source.label
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged
