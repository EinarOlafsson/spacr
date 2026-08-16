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

TWO SCREENS, AND A COLLISION THAT IS NOT ONE
--------------------------------------------
Instruction 122 added the case the first bullet above gets *backwards*. Two
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

import os
import sqlite3
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple)

import pandas as pd

from . import schema

__all__ = [
    "SOURCE_COLUMN",
    "SCREEN_COLUMN",
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
        """The measurements a ``columns='common'`` merge would DROP.

        The instruction's words: "the user must be told which measurements
        were dropped -- that set IS the analysis they are about to run".
        Identity columns (:data:`_ALWAYS_KEPT`) are excluded because the merge
        keeps them however few sources stored them.
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


def _table_columns(path: str, table: str) -> List[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in rows]


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


def _plates(path: str, table: str, plate_column: Optional[str]) -> List[str]:
    if not plate_column:
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            f'SELECT DISTINCT "{plate_column}" FROM "{table}"').fetchall()
    return sorted(str(row[0]) for row in rows if row[0] is not None)


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
    labels: List[str] = []
    for path, screen in zip(paths, assigned):
        label = _label_for(path, labels)
        labels.append(label)
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
                limit_per_source: Optional[int] = None) -> pd.DataFrame:
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
    :returns: one frame carrying :data:`SOURCE_COLUMN` and
        :data:`SCREEN_COLUMN`, with ``frame.attrs['dropped_columns']`` naming
        the measurements that did not survive.
    :raises MergeRefused: on a within-screen plate collision under
        ``on_collision='refuse'``, or an unknown option.
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

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # The set a caller is about to analyse, and the set they are not. Carried
    # on the frame so it cannot be separated from the data it describes.
    merged.attrs["dropped_columns"] = dropped
    merged.attrs["screens"] = plan.screens
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
    return (
        "the same plate id appears in more than one database, so merging "
        "would compute every per-well number over two experiments at "
        f"once: {detail}. Rename the plates, drop one database, pass "
        "screens=[...] if these really are separate screens, or pass "
        "on_collision='qualify' to prefix each with its source.")
