"""Load several measurement databases as one frame, without pooling them.

A screen acquired as three plates used to require three sessions in the
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

TWO SCREENS, AND A COLLISION THAT IS NOT ONE
--------------------------------------------
There is also a case where the first bullet above is *backwards*. Two
screens that share a guide library both have ``plate1``..``plate4``, and there
they are not two plates claiming one identity -- they are two *different*
plates whose identity was only ever partly written down. The missing part is
the screen.

So the check is no longer "does this plate id appear twice?" but **"does this
plate id appear twice INSIDE ONE SCREEN?"**. The first is normal and must be
silent about being fine while still *saying it happened*
(:attr:`MergePlan.shared_plates_across_screens`); the second is the original
error, unchanged and just as fatal.

``on_collision='qualify'`` still exists for callers who want it, but it is no
longer the recommended answer for two screens: rewriting ``plate1`` to
``kd-plate1`` makes the keys unique, which is all it was built to do, and
leaves the screen un-analysable -- you cannot block on it, test for a screen
effect, or colour by it without parsing a string back apart.
"""
from __future__ import annotations

import datetime as _datetime
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple)

import pandas as pd

from . import schema, tabular

__all__ = [
    "SOURCE_COLUMN",
    "SCREEN_COLUMN",
    "MergeCancelled",
    "MergePlan",
    "MergeDecision",
    "SourceSummary",
    "decision_for",
    "decision_log_path",
    "describe_merge",
    "read_merged",
    "record_decision",
    "source_labels",
    "column_kinds",
    "canonical_plate_id",
    "normalise_plate_ids",
    "PLATE_BEARING_COLUMNS",
    "MergeRefused",
]

#: The column every merged frame carries, naming the database a row came from.
#:
#: Not optional, and not a display detail: a merged UMAP whose clusters turn
#: out to be the three plates is the single most important thing this feature
#: can show, and it cannot show it if provenance was dropped at load time.
SOURCE_COLUMN = "source_database"

#: The experiment a row belongs to. ``source_database`` is the *file*;
#: this is the *screen*, and they answer different questions -- one screen is
#: routinely several files, and one file could carry two screens.
#:
#: Defined by :mod:`spacr.schema` and re-exported here so a caller reading a
#: merged frame does not have to know which module owns the name.
SCREEN_COLUMN = schema.SCREEN_KEY

#: Columns a ``columns='common'`` merge never drops, because they are the
#: frame's identity rather than measurements to be intersected. Dropping
#: ``screenID`` because only one source stored it would delete the dimension
#: the merge exists to create.
_ALWAYS_KEPT: Tuple[str, ...] = (SCREEN_COLUMN, SOURCE_COLUMN)


class MergeRefused(RuntimeError):
    """The merge would have silently changed what the numbers mean."""


class MergeCancelled(RuntimeError):
    """The user stopped the merge before it finished.

    NOT a :class:`MergeRefused`, and the distinction is the whole reason this
    is its own class. A refusal is an ANSWER about the data and the caller
    shows it; a cancellation is the user changing their mind and there is
    nothing to report about their databases. A caller that caught both as one
    would put "the merge was refused" in front of somebody who pressed Stop.

    Nothing is half-written when this is raised: every merge in this module
    builds a frame in memory and returns it at the end, so abandoning one
    leaves the previous result exactly where it was.
    """


@dataclass(frozen=True)
class SourceSummary:
    """What one database contributes to a merge."""

    path: str
    label: str
    table: str
    rows: int
    columns: Tuple[str, ...]
    plates: Tuple[str, ...]
    #: The screen label the caller assigned to this database, or ``None`` when
    #: they did not -- in which case the screen comes from a stored
    #: ``screenID`` column, or defaults.
    screen: Optional[str] = None
    #: Distinct ``(screenID, plateID)`` pairs this source will contribute
    #: **after** its screen is applied. This, not ``plates``, is what a
    #: collision is computed on.
    screen_plates: Tuple[Tuple[str, str], ...] = ()
    #: The plate ids EXACTLY AS STORED, before
    #: :func:`spacr.schema.canonical_plate_id` collapses a doubled ``pp``.
    #:
    #: ``plates`` is normalised, deliberately: a plan that said ``pplate1``
    #: while the merged frame says ``plate1`` would make the collision check
    #: compare two vocabularies. But that left NOTHING carrying the stored
    #: spelling, so a caller wanting to tell the user "your database is
    #: stamped ``pplate1``" had no source of truth to compare against and its
    #: report was silent on every database. Both are carried now: ``plates``
    #: is what the merge will key on, ``stored_plates`` is what is on disk,
    #: and the difference between them is the thing worth saying.
    stored_plates: Tuple[str, ...] = ()

    @property
    def odd_plates(self) -> Tuple[str, ...]:
        """Stored plate ids whose spelling is not the canonical one.

        Empty in the normal case, which is why a caller can print a line per
        entry and stay silent when there is nothing to say.
        """
        return tuple(plate for plate in self.stored_plates
                     if canonical_plate_id(plate) != str(plate))

    @property
    def name(self) -> str:
        """Short name for a legend or a chip."""
        return self.label

    @property
    def screens(self) -> Tuple[str, ...]:
        """Distinct screens this source contributes, in sorted order."""
        return tuple(sorted({pair[0] for pair in self.screen_plates}))


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
    #: A plate id duplicated **within one screen**: ``{plate: (labels,)}``.
    #: This is the real error, and the only thing ``has_collisions`` reports.
    colliding_plates: Mapping[str, Tuple[str, ...]]
    #: The same, keyed on the full identity: ``{(screen, plate): (labels,)}``.
    #: Needed to say *which* screen is duplicated when there are several.
    colliding_identities: Mapping[Tuple[str, str], Tuple[str, ...]] = field(
        default_factory=dict)
    #: A plate id that appears in more than one screen: ``{plate: (screens,)}``.
    #:
    #: **This is not a collision.** Two screens sharing a guide library both
    #: have ``plate1``, and once ``screenID`` is in the frame those are two
    #: distinct identities. It is reported anyway, because a user who did
    #: *not* mean to run two screens still needs to see that they did.
    shared_plates_across_screens: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(source.rows for source in self.sources)

    @property
    def has_collisions(self) -> bool:
        return bool(self.colliding_plates)

    @property
    def screens(self) -> Tuple[str, ...]:
        """Every screen this merge would produce, in sorted order."""
        found = set()
        for source in self.sources:
            found.update(source.screens)
        return tuple(sorted(found))

    @property
    def screens_were_named(self) -> bool:
        """Whether the caller is working in screens at all.

        True as soon as ONE database was given a screen label. It is not
        ``len(screens) > 1``: labelling two databases as the same screen is a
        deliberate statement (they are two halves of one experiment), and a
        refusal about them still has to say which screen, or the user cannot
        tell it apart from the two-screen case that is allowed.
        """
        return any(source.screen is not None for source in self.sources)

    @property
    def dropped_columns(self) -> Tuple[str, ...]:
        """Return measurements discarded by a ``columns='common'`` merge.

        Identity columns in :data:`_ALWAYS_KEPT` are excluded because the
        merge retains them even when only some sources contain them.
        """
        return tuple(sorted(
            name for name in self.partial_columns if name not in _ALWAYS_KEPT))

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
        screens = self.screens
        if len(screens) > 1:
            lines.append(f"  {len(screens)} screens: " + ", ".join(screens))
        if self.shared_plates_across_screens:
            lines.append(
                f"  {len(self.shared_plates_across_screens)} plate id(s) "
                "appear in more than one SCREEN, which is not a clash: "
                + ", ".join(sorted(self.shared_plates_across_screens)))
        if self.colliding_plates:
            lines.append(
                f"  {len(self.colliding_plates)} plate id(s) appear in more "
                "than one database within one screen: "
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


#: Folder names that say nothing about WHICH database is which.
#:
#: spaCR's own layout is ``<plate>/measurements/measurements.db``, so both the
#: file stem AND its immediate parent are the same string for every plate in a
#: screen. Disambiguating on the immediate parent therefore produced
#: ``measurements``, ``measurements/measurements`` and
#: ``measurements/measurements (2)`` for three plates -- three labels that are
#: distinct and tell the user nothing, in the column the merged embedding is
#: COLOURED BY.
_GENERIC_FOLDERS = frozenset({
    "measurements", "measurement", "db", "database", "databases",
    "data", "results", "result", "analysis", "output", "outputs",
})


def _meaningful_parent(path: str) -> str:
    """The nearest ancestor directory whose name says which source this is.

    Climbs past :data:`_GENERIC_FOLDERS`, which is what makes the plate folder
    -- ``plate1`` in ``plate1/measurements/measurements.db`` -- the name of the
    source, exactly as :func:`spacr.core._umap_source_label` already names it.
    """
    parent = os.path.dirname(str(path))
    while parent:
        name = os.path.basename(parent)
        if not name:
            break
        if name.casefold() not in _GENERIC_FOLDERS:
            return name
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    return ""


#: The plate id in the form the rest of spaCR keys on, and the columns that
#: carry it. Both are :mod:`spacr.schema`'s -- re-exported here, unchanged, so
#: the callers that import them from this module keep working.
#:
#: They used to be DEFINED here, which made three copies of one rule:
#: ``utils.correct_metadata`` repaired frames, this module repaired scalars
#: and database reads, and ``tests/test_multi_database.py`` had to pin the
#: two against each other by test precisely because they were two. One
#: definition cannot disagree with itself.
canonical_plate_id = schema.canonical_plate_id
PLATE_BEARING_COLUMNS = schema.PLATE_BEARING_COLUMNS


def normalise_plate_ids(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse a doubled ``p`` prefix in every column that carries a plate.

    WHY THIS IS NOT COSMETIC. A measurements database stamped ``pplate1``
    produces merged rows stamped ``pplate1``, while the score and count CSVs
    have already been normalised to ``plate1``. The two then do not meet.
    Every join INSIDE the merge is unaffected, because both sides read the
    same stored value -- which is exactly what makes it hard to see: the
    merge succeeds, the row counts are right, and the failure appears later
    and somewhere else, as a gene half that is missing for no visible reason.

    THE COMPOSED KEYS MATTER AS MUCH AS THE PLATE COLUMN. ``prc`` is
    ``<plate>_<row>_<column>``, so a doubled prefix rides in its first
    component; rewriting ``plateID`` alone would leave ``prc`` unjoinable and
    the two columns disagreeing about the same plate.

    Applied on READ, so nothing on disk is rewritten and an old database
    keeps working -- the standing rule is to correct the format going forward
    and migrate the content, and a measurements database is the user's data
    rather than ours to edit.

    A thin name over :func:`spacr.schema.normalise_plate_columns`, which is
    the one implementation.

    :param frame: any frame read from a measurements database.
    :returns: the same frame, with the plate-bearing columns normalised in
        place. Columns it does not have are skipped.
    """
    return schema.normalise_plate_columns(frame)


def source_labels(paths: Sequence[str]) -> Tuple[str, ...]:
    """One short, unique, human name per database, decided for the whole set.

    Public because a chip, a legend and the :data:`SOURCE_COLUMN` value have to
    be the SAME string: a screen that labels a chip ``plate1`` while the
    provenance column says ``measurements (2)`` has provenance the user cannot
    follow.

    Decided across all the paths at once rather than one at a time, because
    "is this name ambiguous?" is a question about the set. Three rules, in
    order:

    1. the file stems, when they differ -- that is what the user called them;
    2. the nearest MEANINGFUL parent folder (:func:`_meaningful_parent`), which
       is the plate folder in spaCR's own ``<plate>/measurements/
       measurements.db`` layout;
    3. the historical parent/stem rule with a numeric tail, for the case where
       even the folders repeat.
    """
    paths = [str(path) for path in paths]
    stems = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    if len(set(stems)) == len(stems):
        return tuple(stems)
    folders = [_meaningful_parent(path) for path in paths]
    if all(folders) and len(set(folders)) == len(folders):
        return tuple(folders)
    used: List[str] = []
    for path in paths:
        used.append(_label_for(path, used))
    return tuple(used)


def _table_columns(path: str, table: str) -> List[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in rows]


#: SQLite's own type-affinity rules, in the order the engine applies them.
#: Substring, first match wins -- ``VARCHAR`` is TEXT because it contains
#: ``CHAR``, ``FLOAT`` is REAL because it contains ``FLOA``.
_AFFINITY_RULES: Tuple[Tuple[str, str], ...] = (
    ("INT", "numeric"),
    ("CHAR", "text"), ("CLOB", "text"), ("TEXT", "text"),
    ("BLOB", "unknown"),
    ("REAL", "numeric"), ("FLOA", "numeric"), ("DOUB", "numeric"),
)


def column_kinds(path: str, table: str) -> Dict[str, str]:
    """``{column: 'numeric' | 'text' | 'unknown'}`` for one table, WITHOUT
    reading a row.

    WHY THIS EXISTS, and it is a correctness answer rather than a convenience.
    A pre-merge plan has to say how each column will be combined, and the
    merge decides that from the column's pandas dtype
    (:func:`spacr.merge_tables.aggregation_plan` asks
    ``is_numeric_dtype``). A plan that matched only on the column NAME told
    users that ``file_name`` and ``path_name`` "would take the default
    (mean)", which is not what happens and is not a thing that can happen to
    a string. The declared affinity is what predicts the dtype, and reading
    it costs one ``PRAGMA``.

    ``'unknown'`` is returned rather than guessed for a column declared with
    no type or as a BLOB -- an absent answer that reads as a definite one is
    the failure this module exists to avoid.

    :param path: the database.
    :param table: the table.
    :returns: one entry per column, in the table's own column order.
    """
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    out: Dict[str, str] = {}
    for row in rows:
        declared = str(row[2] or "").upper()
        kind = "unknown"
        for token, answer in _AFFINITY_RULES:
            if token in declared:
                kind = answer
                break
        else:
            # SQLite's fifth rule: anything else declared is NUMERIC. Nothing
            # declared at all stays unknown.
            if declared.strip():
                kind = "numeric"
        out[str(row[1])] = kind
    return out


def _row_count(path: str, table: str) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        return int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _canonical_columns(columns: Sequence[str]) -> Dict[str, str]:
    """``{stored name: the name the merged frame will use}``.

    The plan has to speak the merged frame's vocabulary, not the file's: a
    database spelling a column ``row`` and one spelling it ``rowID`` do not
    contribute two columns, and reporting them as two would tell the user a
    measurement is about to be dropped when it is not. This is the same
    mapping :func:`read_merged` applies, so the two cannot disagree.
    """
    mapping = schema.canonical_rename_plan(list(columns))
    return {name: mapping.get(name, name) for name in columns}


def _stored_plates(path: str, table: str,
                   plate_column: Optional[str]) -> List[str]:
    """The plate ids exactly as the database spells them.

    The un-normalised counterpart of :func:`_plates`. Somebody has to hold
    the stored spelling or the doubled-prefix report has nothing to compare
    against and is silent on every database, including the ones it exists
    for.
    """
    if not plate_column:
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            f'SELECT DISTINCT "{plate_column}" FROM "{table}"').fetchall()
    return sorted(str(row[0]) for row in rows if row[0] is not None)


def _plates(path: str, table: str, plate_column: Optional[str]) -> List[str]:
    if not plate_column:
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            f'SELECT DISTINCT "{plate_column}" FROM "{table}"').fetchall()
    # Normalised, so the PLAN names a plate the same way the DATA will after
    # `normalise_plate_ids`. A plan that says `pplate1` while the frame says
    # `plate1` would make the collision check compare two vocabularies.
    return sorted(canonical_plate_id(row[0])
                  for row in rows if row[0] is not None)


def _screen_plate_pairs(path: str, table: str, plate_column: Optional[str],
                        screen_column: Optional[str],
                        screen: Optional[str]) -> List[Tuple[str, str]]:
    """Distinct ``(screenID, plateID)`` this source will contribute.

    Three cases, in the order they are decided:

    * the caller named the screen -- their label wins over anything stored,
      because they are the one looking at the files;
    * the database carries a ``screenID`` column -- believe it, because a
      frame that already knows which experiment it came from must not be
      relabelled behind the user's back;
    * neither -- :data:`spacr.schema.DEFAULT_SCREEN`, i.e. a single-screen
      project, which is every project written before instruction 122.
    """
    if not plate_column:
        return []
    if screen is not None or not screen_column:
        label = schema.screen_id(screen)
        return [(label, plate) for plate in _plates(path, table, plate_column)]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            f'SELECT DISTINCT "{screen_column}", "{plate_column}" '
            f'FROM "{table}"').fetchall()
    return sorted({(schema.screen_id(row[0]), str(row[1]))
                   for row in rows if row[1] is not None})


def _resolve_screens(paths: Sequence[str],
                     screens: Any) -> List[Optional[str]]:
    """One screen label per path, or ``None`` where the caller gave none.

    Accepts a sequence parallel to ``paths`` (the shape a CLI has) or a
    mapping from path (the shape a GUI's file table has). A sequence of the
    wrong length is refused rather than zipped short: it would label the wrong
    database, and a mislabelled screen is the exact failure this column was
    added to prevent.
    """
    if screens is None:
        return [None] * len(paths)
    if isinstance(screens, Mapping):
        return [screens.get(str(path), screens.get(path)) for path in paths]
    labels = list(screens)
    if len(labels) != len(paths):
        raise MergeRefused(
            f"screens= has {len(labels)} label(s) for {len(paths)} "
            f"database(s). One label per database, in the same order, or a "
            f"mapping from path -- a short list would silently label the "
            f"wrong screen.")
    return labels


def describe_merge(paths: Sequence[str], table: str, *,
                   screens: Any = None) -> MergePlan:
    """Work out what merging ``paths`` would produce, WITHOUT reading rows.

    Reads only sqlite metadata and the distinct plate ids, so this is cheap
    enough to run while the user is still choosing files -- which is the
    point, because the answer has to arrive before they commit.

    :param paths: measurement databases, in the order the user added them.
    :param table: the table to merge, e.g. ``'cell'``.
    :param screens: optional screen label per database -- a sequence parallel
        to ``paths``, or a mapping from path. Two databases in **different**
        screens may share a plate id; two in the same screen may not.
    :returns: the :class:`MergePlan`.
    """
    assigned = _resolve_screens(paths, screens)
    summaries: List[SourceSummary] = []
    # Named for the whole set at once -- see :func:`source_labels` for why a
    # per-path rule gave every plate of a screen the same useless name.
    labels = list(source_labels(paths))
    for path, screen, label in zip(paths, assigned, labels):
        columns = _table_columns(path, table)
        canonical = _canonical_columns(columns)
        stored = {value: key for key, value in canonical.items()}
        plate_column = stored.get(schema.PLATE_KEY)
        screen_column = stored.get(schema.SCREEN_KEY)
        summaries.append(SourceSummary(
            path=str(path), label=label, table=table,
            rows=_row_count(path, table),
            columns=tuple(columns),
            plates=tuple(_plates(path, table, plate_column)),
            stored_plates=tuple(_stored_plates(path, table, plate_column)),
            screen=screen,
            screen_plates=tuple(_screen_plate_pairs(
                path, table, plate_column, screen_column, screen)),
        ))

    if not summaries:
        return MergePlan((), (), {}, {})

    column_sets = [set(_canonical_columns(s.columns).values())
                   for s in summaries]
    common = set.intersection(*column_sets) if column_sets else set()
    everything = set.union(*column_sets) if column_sets else set()
    partial = {
        column: tuple(summary.label
                      for summary, have in zip(summaries, column_sets)
                      if column in have)
        for column in sorted(everything - common)
    }

    # THE CHECK THAT CHANGED. Keyed on (screen, plate), not on plate: two
    # screens each owning a plate1 are two identities, and calling that a
    # clash makes stacking two screens impossible. A plate repeated inside one
    # screen is the original error and is untouched.
    seen: Dict[Tuple[str, str], List[str]] = {}
    plate_screens: Dict[str, List[str]] = {}
    for source in summaries:
        for screen_label, plate in source.screen_plates:
            seen.setdefault((screen_label, plate), []).append(source.label)
            if screen_label not in plate_screens.setdefault(plate, []):
                plate_screens[plate].append(screen_label)

    colliding_identities = {identity: tuple(labels_)
                            for identity, labels_ in sorted(seen.items())
                            if len(labels_) > 1}
    colliding: Dict[str, Tuple[str, ...]] = {}
    for (_screen_label, plate), labels_ in colliding_identities.items():
        colliding[plate] = tuple(
            dict.fromkeys(colliding.get(plate, ()) + labels_))
    shared = {plate: tuple(found)
              for plate, found in sorted(plate_screens.items())
              if len(found) > 1}

    return MergePlan(tuple(summaries), tuple(sorted(common)), partial,
                     colliding, colliding_identities, shared)


# --------------------------------------------------------------------------- #
#  What the user decided, written down
# --------------------------------------------------------------------------- #
#
# Instruction 109: "Two databases that both contain a plate called plate1 do
# NOT silently merge those plates. The user is TOLD, and what they choose is
# RECORDED."
#
# Telling them is the refusal and the plan. RECORDING is this: a merge that a
# user resolved by hand -- by dropping one of two colliding databases, say --
# leaves no trace in the result, and six months later the frame cannot say
# which of the two plate1s it is. One appended JSON line per merge, in one
# place, is the smallest thing that answers that question afterwards.


@dataclass(frozen=True)
class MergeDecision:
    """One merge, and what was decided about it.

    Deliberately flat and JSON-safe: this is written to a log that outlives
    the session, so it holds strings and numbers rather than objects whose
    class may not exist by the time somebody reads it back.
    """

    table: str
    sources: Tuple[str, ...]
    labels: Tuple[str, ...]
    #: Rows per source label, BEFORE the merge. The per-source count that a
    #: reader can compare against the merged frame to prove nothing pooled.
    rows: Mapping[str, int]
    columns: str
    dropped_columns: Tuple[str, ...]
    colliding_plates: Mapping[str, Tuple[str, ...]]
    #: ``'merged'`` or ``'refused'`` -- what actually happened.
    outcome: str
    #: What the user did about it, in their own terms. Empty when there was
    #: nothing to decide.
    resolution: str = ""
    when: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """The record as plain JSON-safe data."""
        return {
            "when": self.when,
            "table": self.table,
            "outcome": self.outcome,
            "resolution": self.resolution,
            "columns": self.columns,
            "sources": list(self.sources),
            "labels": list(self.labels),
            "rows": {str(k): int(v) for k, v in dict(self.rows).items()},
            "dropped_columns": list(self.dropped_columns),
            "colliding_plates": {
                str(plate): list(labels)
                for plate, labels in dict(self.colliding_plates).items()},
        }


def decision_for(plan: MergePlan, *, outcome: str, columns: str = "common",
                 resolution: str = "",
                 when: Optional[str] = None) -> MergeDecision:
    """Build the record for what ``plan`` was asked to do.

    :param plan: the plan the decision is about.
    :param outcome: ``'merged'`` or ``'refused'``.
    :param columns: the column rule that was used.
    :param resolution: what the user chose, in words.
    :param when: ISO timestamp; ``None`` takes the current local time.
    """
    return MergeDecision(
        table=plan.sources[0].table if plan.sources else "",
        sources=tuple(source.path for source in plan.sources),
        labels=tuple(source.label for source in plan.sources),
        rows={source.label: source.rows for source in plan.sources},
        columns=columns,
        dropped_columns=tuple(plan.dropped_columns) if columns == "common"
        else (),
        colliding_plates={plate: tuple(labels) for plate, labels
                          in dict(plan.colliding_plates).items()},
        outcome=outcome,
        resolution=resolution,
        when=when or _datetime.datetime.now().isoformat(timespec="seconds"),
    )


def decision_log_path() -> str:
    """Where merge decisions are appended.

    Beside ``~/.spacr/runs``, which is where this application already keeps
    the record of what it was asked to do.
    """
    return os.path.join(os.path.expanduser("~"), ".spacr",
                        "merge_decisions.jsonl")


def record_decision(decision: MergeDecision,
                    path: Optional[str] = None) -> str:
    """Append ``decision`` to the merge log and return the file it went to.

    JSON lines, appended: a merge decision is an event, and rewriting a whole
    document to add one would lose the others if two screens decided at once.

    Never raises for an unwritable log -- a read-only home directory must not
    take a screen down for the sake of an audit line -- but returns ``""`` so
    a caller that wants to say the record was not kept can.
    """
    target = path or decision_log_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision.as_dict(), sort_keys=True) + "\n")
    except OSError:
        return ""
    return target


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
                screens: Any = None,
                report: Optional[Callable[[str], None]] = None,
                limit_per_source: Optional[int] = None,
                progress: Optional[Callable[[str, int, int], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None,
                rows_done: int = 0,
                rows_total: Optional[int] = None) -> pd.DataFrame:
    """Read ``table`` from every path and return one frame.

    :param paths: measurement databases.
    :param table: the table to read from each.
    :param plan: a plan from :func:`describe_merge`; recomputed if omitted.
        Pass ``screens=`` to :func:`describe_merge` and here alike, or the
        plan and the read disagree about which screen a database is.
    :param columns: ``'common'`` keeps only columns present in every source
        (safe, and drops); ``'union'`` keeps everything and leaves nulls where
        a source did not have it (keeps, and changes what "missing" means).
        ``'common'`` is the default because measurement tables are wide and
        differ between spaCR versions -- but a dropped measurement is a
        measurement the user came to compare, so the set is reported rather
        than merely defaulted (see ``report`` and ``.attrs``).
    :param on_collision: what to do when a plate id is duplicated **within one
        screen**. ``'refuse'`` raises :class:`MergeRefused`; ``'qualify'``
        prefixes each colliding plate with its source label. There is
        deliberately no option that pools them. Two *different* screens
        sharing a plate id is not a collision and reaches neither branch.
    :param screens: optional screen label per database -- a sequence parallel
        to ``paths``, or a mapping from path. This is the recommended answer
        for two screens: the label is written into :data:`SCREEN_COLUMN`, so
        it stays a dimension you can block on, rather than into the plate id,
        where it becomes a string to be parsed back apart.
    :param report: called with one human-readable line per thing the merge
        cost -- currently the dropped measurements.
    :param limit_per_source: row cap per database, for previews.
    :param progress: called ``progress(stage, done, total)`` before and after
        each database is read. ``stage`` is a sentence naming the table and
        the database it is on; ``done`` and ``total`` are ROWS, counted
        against ``rows_total`` so a caller reading several tables can show one
        bar across all of them. Called from whatever thread this runs on, so a
        GUI caller must relay it rather than touch a widget in it.
    :param cancelled: called before each database; a true answer raises
        :class:`MergeCancelled` and nothing is returned. Checked between
        sources rather than inside one, because a half-read source is not a
        thing this function can hand back.
    :param rows_done: rows already counted by an earlier call, for the
        multi-table case.
    :param rows_total: the denominator for ``progress``. Defaults to this
        plan's own total, which is right for a single table and too small for
        a caller merging several -- so a caller merging several passes the
        grand total it computed from all their plans.
    :returns: one frame carrying :data:`SOURCE_COLUMN` and
        :data:`SCREEN_COLUMN`, with ``frame.attrs['dropped_columns']`` naming
        the measurements that did not survive.
    :raises MergeRefused: on a within-screen plate collision under
        ``on_collision='refuse'``, or an unknown option.
    :raises MergeCancelled: when ``cancelled()`` answered true.
    """
    if columns not in ("common", "union"):
        raise MergeRefused(
            f"columns must be 'common' or 'union', got {columns!r}")
    if on_collision not in ("refuse", "qualify"):
        raise MergeRefused(
            f"on_collision must be 'refuse' or 'qualify', got "
            f"{on_collision!r}. There is no option that merges two plates of "
            f"the same name into one -- that is the failure this refuses.")

    plan = plan or describe_merge(paths, table, screens=screens)
    if not plan.sources:
        return pd.DataFrame()

    if plan.colliding_plates and on_collision == "refuse":
        raise MergeRefused(_collision_message(plan))

    keep = (set(plan.common_columns) | set(_ALWAYS_KEPT)
            if columns == "common" else None)
    dropped = plan.dropped_columns if columns == "common" else ()
    frames = []
    source_rows: Dict[str, int] = {}
    done = int(rows_done)
    total = int(rows_total) if rows_total is not None else (
        int(rows_done) + plan.total_rows)
    for source in plan.sources:
        # BETWEEN SOURCES, NOT INSIDE ONE. A source is read by a single
        # `read_sql_query`; interrupting that would leave a partial frame
        # this function has no honest way to return, and the point of a
        # cancel is that nothing half-made survives it.
        if cancelled is not None and cancelled():
            raise MergeCancelled(
                f"stopped while reading {table} — {done:,} of {total:,} rows "
                f"had been read and none of them were kept.")
        if progress is not None:
            progress(f"reading {table} from {source.label}", done, total)
        # ONE READER. `tabular.read_database` is the door every spaCR read
        # goes through, and it is what applies the vocabulary: canonical
        # names, ONE column per metadata key (a `well` beside a `wellID` is
        # collapsed and the disagreement counted), and the `pplate1` plate
        # repair before anything keys on it. Case-folded, so it cannot
        # produce a frame SQLite will refuse.
        #
        # read_only, because a merge reads the user's measurement databases
        # and must not be able to write to one; migrate=False follows from
        # that and is what this call has always done.
        frame = tabular.read_database(
            source.path, [source.table],
            report=report, warn=report,
            migrate=False, read_only=True,
            limit=int(limit_per_source) if limit_per_source else None,
        )[0]
        # BEFORE the column filter, so a screen stored in only one source is
        # not intersected away, and so an explicitly named screen reaches
        # every row whether the database had the column or not.
        frame = schema.add_screen_column(
            frame, source.screen, overwrite=source.screen is not None)
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
        source_rows[source.label] = len(frame)
        done += len(frame)
        if progress is not None:
            progress(f"read {table} from {source.label}", done, total)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # HOW FAR THIS CALL GOT, carried on the frame so a caller stacking several
    # tables can continue the count without re-deriving it from row lengths
    # that the column filter may already have changed.
    merged.attrs["rows_done"] = done
    # The set a caller is about to analyse, and the set they are not. Carried
    # on the frame so it cannot be separated from the data it describes.
    merged.attrs["dropped_columns"] = dropped
    merged.attrs["screens"] = plan.screens
    # THE ANTI-POOLING EVIDENCE, carried with the data. Pooling two plates
    # that share a name is the one failure here with no symptom, so the counts
    # that would expose it travel on the frame rather than being recomputable
    # only by going back to the files.
    merged.attrs["source_rows"] = dict(source_rows)
    merged.attrs["labels"] = tuple(source.label for source in plan.sources)
    if dropped and report is not None:
        report(
            f"{len(dropped)} measurement(s) present in only some databases "
            f"were dropped by columns='common': " + ", ".join(dropped)
            + ". Pass columns='union' to keep them, with nulls where a "
              "database did not have them.")
    return merged


def _collision_message(plan: MergePlan) -> str:
    """Why the merge was refused, naming the screen when there is one to name.

    The wording changes on purpose. For a project that never mentioned a
    screen, "the same plate id appears in more than one database" is the whole
    story and is what instruction 109 has always said. Once screens are in
    play it is not: the user's next question is *which* screen, and without it
    they cannot tell a genuine duplicate from the two screens legitimately
    sharing that plate name.
    """
    identities = plan.colliding_identities
    if identities and (len(plan.screens) > 1 or plan.screens_were_named):
        detail = "; ".join(
            f"{plate!r} in screen {screen!r} in {', '.join(labels)}"
            for (screen, plate), labels in sorted(identities.items()))
        return (
            "the same plate id appears more than once WITHIN one screen, so "
            "merging would compute every per-well number over two plates of "
            f"one experiment at once: {detail}. Two screens sharing a plate "
            "id would be fine; this is the same screen twice. Rename the "
            "plates, drop one database, or correct the screen labels.")
    detail = "; ".join(
        f"{plate!r} in {', '.join(labels)}"
        for plate, labels in sorted(plan.colliding_plates.items()))
    # NO 'qualify' IN THIS SENTENCE. It is still available to a caller who
    # wants the plate id rewritten, and it is still the wrong thing to put in
    # front of a user: `plate1` becoming `runA-plate1` makes the keys unique
    # and hides which experiment a plate belongs to INSIDE its own id, where
    # it can no longer be blocked on, tested for or coloured by. This message
    # is what the Gate Editor and the Image UMAP show, so it names the
    # resolutions that keep the experiment analysable.
    return (
        "the same plate id appears in more than one database, so merging "
        "would compute every per-well number over two experiments at "
        f"once: {detail}. Remove one of those databases, rename the plates, "
        "or say these are separate screens so each keeps its own identity.")
