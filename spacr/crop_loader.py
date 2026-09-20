"""Choose a set of crops, then read them a page at a time.

**WHAT IT IS FOR.** Every screen that shows objects has had its own way of
finding them, and the Embeddings screen had none at all --
:meth:`spacr.qt.screens.embeddings.EmbeddingsScreen.set_crops` existed and
nothing in ``spacr/`` called it, so the only route to a label-free vector was
to open Python. This module is the missing half: it turns a question about a
plate into a stack of pixels, and it does it in two steps so a GUI can stay
answerable between them.

**PLANNING IS NOT LOADING, and that separation is the whole design.**
:func:`plan_crops` asks the database or the folder *which objects match* and
reads no pixels at all: a ``COUNT`` and a ``SELECT`` of identifiers, or one
``scandir``. It is milliseconds on a plate of sixty thousand crops, so a
screen can run it on a click and tell the user what they asked for before
committing to reading any of it. :func:`load_crops` then reads the pixels in
pages, reporting after each one. A caller that runs the second half on a
worker thread gets a progress number per page and a stop that takes effect
within a page rather than at the end.

**WHAT IT NEEDS.** Either a ``measurements.db`` -- where the crops are chosen
by object class, by plate, or by a predicate over ``png_list`` -- or a folder
of crop PNGs. Nothing else: the pixels themselves come from
:func:`spacr.crops.resolve_crop_source`, so this module inherits the existing
answer to "are there pre-generated crops here, or do we cut from
``merged/*.npy``" rather than inventing a second one, and a legacy folder is
corrected on load exactly as it is everywhere else.

**WHAT IT PRODUCES.** One ``(objects, height, width, channels)`` uint8 array,
which is the layout :func:`spacr.embeddings.embed_array` documents and
:meth:`spacr.qt.screens.embeddings.EmbeddingsScreen.set_crops` requires, plus
a JSON-serialisable ``record`` of what was actually read -- how many matched,
how many were taken, the crop shape, and how many crops had to be conformed
to it.

**WHAT TO DO NEXT.** Hand the array to the Embeddings screen, or straight to
:func:`spacr.embeddings.embed_array`.

**A CAP IS PART OF THE ANSWER, NOT A FAILURE TO LOAD EVERYTHING.** Sixty
thousand crops of 96x96x3 is 1.6 GB before a backbone has seen one of them,
and a screen that tries it dies on a machine that could have embedded the
first two thousand happily. So :attr:`CropQuery.limit` is a real default and
the plan says what it left behind -- "the first 2,000 of 61,433" -- rather
than quietly returning a subset that looks like the whole plate.

**AND A CAP IS NOT THE ONLY THING THAT REMOVES A ROW, SO IT IS COUNTED ON ITS
OWN.** The merged route drops every row it cannot cut, and reporting that as
a cap is the same sentence read backwards: a complete answer dressed as a
subset, telling the user to raise a limit that never bit. So
:attr:`CropPlan.matched`, :attr:`CropPlan.selected` and
:attr:`CropPlan.count` are three numbers, :attr:`CropPlan.capped` is the gap
between the first two and :attr:`CropPlan.dropped` the gap between the last
two, and they reach the panel as separate clauses.

**AN EMPTY RESULT IS A SENTENCE, NEVER AN EMPTY ARRAY.** A plan that matched
nothing carries :attr:`CropPlan.empty_reason`, which names the table, the
class, the plate and the predicate that were asked for AND what the database
does hold, because "no crops" on its own sends a user to look for a broken
install when they have asked for plate 9 of an eight-plate screen.

**THE PREDICATE IS SQL AND THE DATABASE IS OPENED READ-ONLY.** ``where`` is a
fragment of the user's own query language against their own file; refusing it
would mean inventing a worse one. It is spliced into a ``WHERE`` clause, so
the connection is opened with ``readonly=True`` -- SQLite's ``query_only``,
which refuses every write at the engine -- and a predicate carrying a
statement separator is rejected before it is sent.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, List, Mapping, MutableMapping,
                    Optional, Sequence, Tuple)

import numpy as np

LOG = logging.getLogger("spacr.crop_loader")

__all__ = [
    "CropLoadError",
    "CROP_SOURCE_DATABASE",
    "CROP_SOURCE_FOLDER",
    "CROP_SOURCES",
    "CROP_TABLE",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_CROP_LIMIT",
    "CropQuery",
    "CropPlan",
    "object_classes",
    "plates",
    "plan_crops",
    "plan_from_database",
    "plan_from_folder",
    "conform_crop",
    "load_page",
    "load_crops",
]


class CropLoadError(ValueError):
    """A set of crops that cannot be loaded, and why.

    Carries a sentence fit for a status bar: every raise in this module names
    the path, the class or the predicate that produced it, because the caller
    is a screen and its only way to explain a failure is to repeat this text.
    """


#: Crops chosen out of a ``measurements.db``, by class, plate or predicate.
CROP_SOURCE_DATABASE = "database"

#: Crops read straight out of a folder of ``*.png``.
CROP_SOURCE_FOLDER = "folder"

#: ``(value, label)`` in the order a panel should offer them.
CROP_SOURCES: Tuple[Tuple[str, str], ...] = (
    (CROP_SOURCE_DATABASE, "measurements database"),
    (CROP_SOURCE_FOLDER, "crop folder"),
)

#: The table a measurement database records one row per crop in. The same
#: table :func:`spacr.png_list.crop_rows_from_png_list` joins, so a row from
#: here serves the PNG source and the merged one alike.
CROP_TABLE = "png_list"

#: Crops read per page. Small enough that a stop takes effect promptly and a
#: progress line moves; large enough that
#: :meth:`spacr.crops.MergedCropSource.get_many` still opens each ``.npy``
#: once for a useful number of objects rather than once per crop.
DEFAULT_PAGE_SIZE = 128

#: How many crops a load takes unless the caller raises it. 2,000 crops of
#: 96x96x3 is about 55 MB, which is a load any machine that can open the
#: screen can hold; a plate has thirty times that.
DEFAULT_CROP_LIMIT = 2000


@dataclass(frozen=True)
class CropQuery:
    """Which crops to load, and how much of the answer to take.

    :param source: :data:`CROP_SOURCE_DATABASE` or :data:`CROP_SOURCE_FOLDER`.
    :param path: the ``measurements.db``, or the folder of crop PNGs.
    :param object_type: which object class the crops are of -- ``'cell'``,
        ``'nucleus'``, ``'pathogen'``, ``'cytoplasm'`` or an organelle role.
        It picks the object-id column in ``png_list`` and the mask plane a
        streamed crop is cut by, so it is not cosmetic: asking for nuclei out
        of a cell ``png_list`` yields nothing rather than yielding cells.
        Ignored by the folder source, which has no classes.
    :param plate: one plate, or ``''`` for every plate.
    :param where: an SQL predicate over ``png_list``, or ``''``. Spliced into
        the ``WHERE`` clause of a read-only connection; see the module
        docstring for why it is accepted at all.
    :param limit: the most crops to take, ``0`` for no cap. The plan records
        what the cap left behind.
    :param page_size: crops read per page by :func:`load_crops`.
    :param prefer: ``'png'`` to read pre-generated crops, ``'merged'`` to cut
        them from ``merged/*.npy``, ``''`` to let
        :func:`spacr.crops.resolve_crop_source` decide. Database source only.
    """

    source: str = CROP_SOURCE_DATABASE
    path: str = ""
    object_type: str = "cell"
    plate: str = ""
    where: str = ""
    limit: int = DEFAULT_CROP_LIMIT
    page_size: int = DEFAULT_PAGE_SIZE
    prefer: str = ""

    def describe(self) -> str:
        """One line naming everything this query asked for.

        Used in the status bar and in every refusal, so a user reading "no
        crops" can see the question that produced it without reopening the
        panel.
        """
        if self.source == CROP_SOURCE_FOLDER:
            return f"crop folder {self.path}"
        parts = [f"{self.object_type} crops in {os.path.basename(self.path)}"]
        parts.append(f"plate {self.plate}" if self.plate else "every plate")
        if self.where:
            parts.append(f"where {self.where}")
        return ", ".join(parts)


@dataclass(frozen=True)
class CropPlan:
    """What a query matched, before any pixel was read.

    :param query: the :class:`CropQuery` this answers.
    :param rows: one opaque handle per crop, in the order they load -- a row
        mapping for the database source, a path string for the folder one.
        Never longer than ``query.limit``.
    :param matched: how many crops the query matched in total, before the
        limit and before the merged route's join. More than ``selected``
        exactly when the cap bit.
    :param selected: how many rows the selection actually took, which is
        ``min(matched, limit)``. Left at ``0`` it defaults to ``len(rows)``,
        which is right for every plan that has no join to lose rows to.
    :param source_label: which pixel route was chosen and why, from
        :meth:`spacr.crops.CropSource.describe`.
    :param empty_reason: the sentence to show when nothing matched, naming
        both what was asked and what is there. Empty when something did.

    **THREE NUMBERS, NOT TWO, BECAUSE TWO THINGS REMOVE ROWS AND THEY ASK
    THE READER FOR DIFFERENT THINGS.** ``matched`` is what the query found,
    ``selected`` is what the limit left of it, and ``count`` is what
    survived the merged route's join -- which drops every row with no single
    object label (``'omulti'`` / ``'onone'``) or no merged array recorded.
    Only the first gap is a cap, and only a cap is repaired by raising 'At
    most'; telling a user to raise a limit that never bit hands them an
    action that returns the same rows and reprints the same sentence.
    """

    query: CropQuery
    rows: Tuple[Any, ...] = ()
    matched: int = 0
    source_label: str = ""
    empty_reason: str = ""
    selected: int = 0
    _source: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Default ``selected`` to the rows, for a plan built without a join.

        A plan whose rows came straight out of the selection -- the folder
        route, and anything a test or a script builds by hand -- selected
        exactly what it carries, and should not have to say so.
        """
        if self.selected <= 0 and self.rows:
            object.__setattr__(self, "selected", len(self.rows))

    @property
    def count(self) -> int:
        """How many crops this plan will load."""
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """Whether the query matched nothing at all."""
        return not self.rows

    @property
    def capped(self) -> bool:
        """Whether the limit kept crops out of this plan.

        Against :attr:`selected`, never against :attr:`count`: a merged-route
        load that lost rows to the join has ``matched > count`` with no limit
        anywhere near it, and reporting that as a cap is this module's own
        promise -- that a cap is part of the answer -- running backwards.
        """
        return self.matched > self.selected

    @property
    def dropped(self) -> int:
        """How many selected rows could not be turned into a crop at all.

        Nonzero only on the merged route, and a complete answer rather than
        a capped one: the rows are gone because they cannot be cut, so there
        is nothing a larger limit would add.
        """
        return max(0, self.selected - self.count)

    def pages(self, page_size: int = 0) -> Tuple[Tuple[int, int], ...]:
        """``(start, stop)`` for each page, covering every row exactly once.

        :param page_size: crops per page; ``0`` takes the query's own.
        """
        size = int(page_size or self.query.page_size or DEFAULT_PAGE_SIZE)
        if size < 1:
            raise CropLoadError(f"page_size must be at least 1, got {size}")
        return tuple((start, min(start + size, self.count))
                     for start in range(0, self.count, size))

    def describe(self) -> str:
        """One line a status bar can show the moment planning returns.

        Says what was taken out of what, so a capped load reads as a
        deliberate subset rather than as the whole plate -- and names a
        dropped row as a dropped row, in its own clause, so the two never
        arrive as one number.
        """
        if self.is_empty:
            return self.empty_reason or f"no crops for {self.query.describe()}"
        asked = self.query.describe()
        if self.capped:
            head = (f"the first {self.count:,} of {self.matched:,} crops "
                    f"matching {asked}")
        elif self.dropped:
            head = f"{self.count:,} of the {self.matched:,} crops matching {asked}"
        else:
            head = f"{self.count:,} crops matching {asked}"
        parts = [head]
        if self.dropped:
            parts.append(f"{self.dropped:,} cannot be cut and are left out")
        if self.source_label:
            parts.append(self.source_label)
        return " -- ".join(parts)


def _require_file(path: str, what: str) -> str:
    """Return ``path`` as an absolute file path, or refuse by name."""
    if not path:
        raise CropLoadError(f"no {what} was chosen")
    absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(absolute):
        raise CropLoadError(f"{what} not found: {absolute}")
    return absolute


def _connect(db_path: str):
    """Open ``db_path`` read-only, refusing a path that is not a database."""
    from .database_concurrency import connect

    return connect(_require_file(db_path, "measurements database"),
                   readonly=True)


def _table_columns(conn, table: str) -> Tuple[str, ...]:
    """Column names of ``table``, or ``()`` when there is no such table."""
    import sqlite3

    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return ()
    return tuple(str(row[1]) for row in rows)


def object_classes(db_path: str, table: str = CROP_TABLE) -> Tuple[str, ...]:
    """Which object classes this database actually has crops for.

    A class is offered only when its id column is present AND holds at least
    one value, because a ``png_list`` written for cells carries an empty
    ``nucleus_id`` column on some schemas and offering it would put a choice
    in the panel that can only ever return nothing.

    :param db_path: the ``measurements.db``.
    :param table: the crop table; ``'png_list'``.
    :returns: class names in :data:`spacr.png_list.PNG_LIST_ID_COLUMNS`
        order. Empty when the database has no crop table.
    """
    import sqlite3

    from .png_list import PNG_LIST_ID_COLUMNS

    conn = _connect(db_path)
    try:
        columns = set(_table_columns(conn, table))
        found: List[str] = []
        for name, column in PNG_LIST_ID_COLUMNS.items():
            if column not in columns:
                continue
            try:
                row = conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL '
                    f'LIMIT 1').fetchone()
            except sqlite3.Error:
                continue
            if row is not None:
                found.append(name)
        return tuple(found)
    finally:
        conn.close()


def plates(db_path: str, table: str = CROP_TABLE) -> Tuple[str, ...]:
    """Every plate the crop table names, sorted.

    :param db_path: the ``measurements.db``.
    :param table: the crop table; ``'png_list'``.
    :returns: plate identifiers as strings. Empty when the table has no
        ``plateID`` column, which is how a database written before plate keys
        existed reads rather than an error.
    """
    import sqlite3

    conn = _connect(db_path)
    try:
        if "plateID" not in _table_columns(conn, table):
            return ()
        try:
            rows = conn.execute(
                f'SELECT DISTINCT plateID FROM "{table}" '
                f'WHERE plateID IS NOT NULL').fetchall()
        except sqlite3.Error:
            return ()
    finally:
        conn.close()
    return tuple(sorted({str(row[0]) for row in rows}))


def _checked_predicate(where: str) -> str:
    """Return ``where`` stripped, refusing anything but a single predicate.

    A statement separator is the one thing a predicate cannot legitimately
    contain, and it is what turns a filter box into a second statement. The
    connection is read-only besides, so this is the outer of two guards
    rather than the only one.
    """
    text = str(where or "").strip().rstrip(";").strip()
    if ";" in text:
        raise CropLoadError(
            f"the filter must be one condition, not several statements: "
            f"{where!r}")
    return text


def _database_conditions(query: CropQuery, columns: Sequence[str]
                         ) -> Tuple[List[str], List[Any]]:
    """The ``WHERE`` terms and their parameters for ``query``.

    The class and the plate are bound parameters; only the user's own
    predicate is spliced, and only after :func:`_checked_predicate`.

    **A CLASS COLUMN THAT IS NOT THERE MATCHES NOTHING, AND THAT IS THE
    POINT.** Skipping the term because the column is absent is what made
    asking for nuclei out of a cell crop table return every cell in it --
    the right count, the right-looking stack, and the wrong objects under
    the right name, which nothing downstream can catch. A table that names
    no class at all is the separate case: there the class cannot select,
    so it does not, and it still chooses the mask plane a streamed crop is
    cut by.
    """
    from .png_list import PNG_LIST_ID_COLUMNS

    terms: List[str] = []
    values: List[Any] = []
    id_column = PNG_LIST_ID_COLUMNS.get(str(query.object_type))
    classed = any(column in columns for column in PNG_LIST_ID_COLUMNS.values())
    if id_column and id_column in columns:
        terms.append(f'"{id_column}" IS NOT NULL')
    elif classed:
        terms.append("0 = 1")
    if query.plate:
        if "plateID" not in columns:
            raise CropLoadError(
                f'{CROP_TABLE} has no "plateID" column, so it cannot be '
                f"filtered to plate {query.plate!r}")
        terms.append('"plateID" = ?')
        values.append(str(query.plate))
    predicate = _checked_predicate(query.where)
    if predicate:
        terms.append(f"({predicate})")
    return terms, values


def _run(conn, sql: str, values: Sequence[Any]):
    """Execute ``sql``, turning SQLite's complaint into a readable refusal."""
    import sqlite3

    try:
        return conn.execute(sql, tuple(values))
    except sqlite3.Error as exc:
        raise CropLoadError(
            f"the database refused this selection ({exc}). The query was: "
            f"{sql}") from exc


def _empty_database_reason(conn, query: CropQuery,
                           columns: Sequence[str]) -> str:
    """Say what was asked for, and what the database does hold instead.

    Three shapes of nothing, and they send a reader to three different
    places: an empty table, a class with no rows, and a plate that is not in
    the screen. Collapsing them into "no crops" is what makes a user check
    their install when they have mistyped a plate name.
    """
    total = int(_run(conn, f'SELECT COUNT(*) FROM "{CROP_TABLE}"',
                     ()).fetchone()[0])
    asked = query.describe()
    if total == 0:
        return (f"No crops for {asked}: {CROP_TABLE} in "
                f"{os.path.basename(query.path)} is empty, so this database "
                f"has no crops of any class. Run Measure with crop output "
                f"turned on, or point this at a crop folder instead.")
    known = plates(query.path) if "plateID" in columns else ()
    if query.plate and known and str(query.plate) not in known:
        return (f"No crops for {asked}: {CROP_TABLE} holds {total:,} crops "
                f"but none on plate {query.plate!r}. It has "
                f"{', '.join(known)}.")
    classes = object_classes(query.path)
    if classes and str(query.object_type) not in classes:
        return (f"No crops for {asked}: {CROP_TABLE} holds {total:,} crops "
                f"but none labelled as {query.object_type!r}. It has "
                f"{', '.join(classes)}.")
    if query.where:
        return (f"No crops for {asked}: {CROP_TABLE} holds {total:,} crops "
                f"and none of them satisfies this filter.")
    return (f"No crops for {asked}: {CROP_TABLE} holds {total:,} crops and "
            f"none of them matches.")


def plan_from_database(query: CropQuery) -> CropPlan:
    """Choose crops out of a ``measurements.db``, reading no pixels.

    The rows come back carrying ``png_path`` -- what
    :class:`spacr.crops.PngCropSource` reads -- and, through
    :func:`spacr.png_list.crop_rows_from_png_list`, ``path_name`` and an
    integer ``object_label``, which is what
    :class:`spacr.crops.MergedCropSource` needs. So the plan is good for
    whichever route :func:`spacr.crops.resolve_crop_source` picks, and the
    route is picked here, once, rather than per crop.

    :param query: what to select. ``query.source`` is not re-checked.
    :returns: a :class:`CropPlan`, possibly an empty one carrying
        :attr:`CropPlan.empty_reason`.
    :raises CropLoadError: the database is missing, has no crop table, or
        refused the selection.
    """
    import pandas as pd

    db_path = _require_file(query.path, "measurements database")
    query = _replace_path(query, db_path)
    conn = _connect(db_path)
    try:
        columns = _table_columns(conn, CROP_TABLE)
        if not columns:
            raise CropLoadError(
                f"{os.path.basename(db_path)} has no {CROP_TABLE!r} table, so "
                f"it records no crops. Point this at a crop folder, or at a "
                f"database written by a Measure run that saved crops.")
        terms, values = _database_conditions(query, columns)
        clause = f" WHERE {' AND '.join(terms)}" if terms else ""
        matched = int(_run(conn, f'SELECT COUNT(*) FROM "{CROP_TABLE}"{clause}',
                           values).fetchone()[0])
        if matched == 0:
            return CropPlan(query=query,
                            empty_reason=_empty_database_reason(conn, query,
                                                                columns))
        limit = int(query.limit or 0)
        tail = f" LIMIT {max(0, limit)}" if limit > 0 else ""
        cursor = _run(conn, f'SELECT * FROM "{CROP_TABLE}"{clause}{tail}',
                      values)
        names = [str(d[0]) for d in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=names)
    finally:
        conn.close()

    frame = _normalised(frame)
    selected = int(len(frame))
    source = _pixel_source(query, db_path)
    rows = _crop_rows(db_path, frame, query.object_type,
                      streaming=source.kind != "png")
    if not rows:
        return CropPlan(
            query=query, matched=matched, selected=selected,
            source_label=source.describe(),
            empty_reason=_unusable_rows_reason(query, matched, source))
    return CropPlan(query=query, rows=tuple(rows), matched=matched,
                    selected=selected, source_label=source.describe(),
                    _source=source)


def _unusable_rows_reason(query: CropQuery, matched: int, source) -> str:
    """Why rows that matched cannot be turned into crops.

    The two causes are different problems with different repairs, and which
    one applies depends on the route, so both are named rather than merged
    into "unusable rows".
    """
    if source.kind == "png":
        return (f"No crops for {query.describe()}: {matched:,} rows matched, "
                f"and none records a 'png_path'. This database's "
                f"{CROP_TABLE} was written without crop paths, so the crops "
                f"have to be cut from merged/*.npy instead.")
    return (f"No crops for {query.describe()}: {matched:,} rows matched, and "
            f"none of them can be cut -- every one either has no single "
            f"object label ('omulti' / 'onone') or names a merged array this "
            f"database does not record.")


def _replace_path(query: CropQuery, path: str) -> CropQuery:
    """``query`` with ``path`` absolute, so every message names one spelling."""
    return replace(query, path=path)


def _normalised(frame):
    """Apply the metadata and plate-id corrections every reader applies.

    Both are best-effort in :func:`spacr.cell_montage.load_montage_objects`
    for the same reason they are here: a database old enough to lack them is
    still readable, and refusing it would close the screen to exactly the
    archived plates somebody wants a label-free vector for.
    """
    try:
        from .schema import correct_metadata_column_names

        frame = correct_metadata_column_names(frame)
    except Exception:                                            # noqa: BLE001
        LOG.debug("metadata column names were left as they are", exc_info=True)
    try:
        from .multi_database import normalise_plate_ids

        frame = normalise_plate_ids(frame)
    except Exception:                                            # noqa: BLE001
        LOG.debug("plate ids were left as they are", exc_info=True)
    return frame


def _crop_rows(db_path: str, frame, object_type: str,
               streaming: bool) -> List[Dict[str, Any]]:
    """Rows as plain dicts, ready for whichever route reads the pixels.

    Plain dicts rather than a frame because a :class:`CropPlan` is frozen and
    is handed between threads: a dict per row is what
    :meth:`spacr.crops.CropSource.get` reads anyway, and it cannot be mutated
    out from under a load in flight by whoever still holds the frame.

    **THE JOIN IS ONLY RUN FOR THE ROUTE THAT NEEDS IT.**
    :func:`spacr.png_list.crop_rows_from_png_list` recovers ``path_name`` and
    an integer ``object_label`` and DROPS every row it cannot recover them
    for. That is right for a streamed crop, which cannot be cut without them,
    and wrong for a pre-generated one, which needs only the ``png_path`` the
    row already carries -- running it for the PNG route silently discarded
    crops that were sitting on disk.

    :param streaming: whether the pixels are cut from ``merged/*.npy``.
    """
    if streaming:
        from .png_list import crop_rows_from_png_list

        try:
            frame = crop_rows_from_png_list(db_path, frame,
                                            object_type=object_type,
                                            verbose=False)
        except Exception:                                        # noqa: BLE001
            LOG.debug("png_list join failed; keeping the raw rows",
                      exc_info=True)
    elif "png_path" not in getattr(frame, "columns", ()):
        return []
    if frame is None or len(frame) == 0:
        return []
    return [_plain_row(row) for row in frame.to_dict("records")]


#: Row keys that :mod:`spacr.crops` reads as integers, and that a pandas
#: column carrying one missing value turns into floats.
_INTEGER_ROW_KEYS = ("object_label",)


def _plain_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """One row with its missing values dropped rather than left as ``NaN``.

    :func:`spacr.crops._row_get` treats a mapping's value as present when it
    is not ``None``, and pandas writes a missing number as ``NaN``, not
    ``None``. Left in, a missing ``bbox-0`` reads as a bounding box and a
    missing ``path_name`` reads as a path, and both fail much later with a
    message about something else.

    **AND THE LABEL IS PUT BACK TO AN INTEGER.**
    :func:`spacr.png_list.crop_rows_from_png_list` writes ``None`` for every
    row it is about to drop, which makes the whole ``object_label`` column
    float64; the rows that survive then carry ``3.0`` where they carried
    ``3``, and :func:`spacr.crops.object_label` refuses a float outright --
    ``int('3.0')`` raises. So one ``'omulti'`` anywhere in the selection took
    down the entire merged load, with a message about a label that is
    perfectly good. The dtype is pandas bookkeeping and it is undone here,
    where the row is made ready for :mod:`spacr.crops`, rather than by
    loosening what counts as a label everywhere else.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        name = str(key)
        if (name in _INTEGER_ROW_KEYS and isinstance(value, (float, np.floating))
                and float(value).is_integer()):
            value = int(value)
        out[name] = value
    return out


def _pixel_source(query: CropQuery, db_path: str):
    """The :class:`spacr.crops.CropSource` that reads this query's pixels.

    :func:`spacr.crops.resolve_crop_source` is asked first, because it is
    where "are there pre-generated crops under ``data/``" is already
    answered. A plate folder that holds neither ``data/`` nor ``merged/`` --
    a database copied away from its images, which is common enough to be
    worth not failing on -- falls back to reading the absolute ``png_path``
    each row carries.
    """
    from .crops import CropError, PngCropSource, resolve_crop_source

    root = _plate_root(db_path)
    prefer = str(query.prefer or "").strip().lower() or None
    try:
        return resolve_crop_source(root, object_type=str(query.object_type),
                                   prefer=prefer)
    except CropError as exc:
        LOG.debug("no crop source under %s (%s); reading recorded paths",
                  root, exc)
        return PngCropSource(
            root=None, db_path=db_path,
            reason=(f"the paths recorded in {CROP_TABLE}: {root} holds "
                    f"neither a crop folder nor merged/*.npy"))


def _plate_root(db_path: str) -> str:
    """The plate folder a ``measurements.db`` belongs to."""
    from .portable_paths import source_root_for_database

    return source_root_for_database(db_path) or os.path.dirname(db_path)


def plan_from_folder(query: CropQuery) -> CropPlan:
    """Choose crops out of a folder of ``*.png``, reading no pixels.

    :param query: what to select. Only ``path`` and ``limit`` apply: a folder
        records no class and no plate, and saying so in
        :attr:`CropPlan.source_label` is better than filtering on a filename
        convention that no run guarantees.
    :returns: a :class:`CropPlan`, possibly an empty one carrying
        :attr:`CropPlan.empty_reason`.
    :raises CropLoadError: no folder was chosen, or it is not a directory.
    """
    if not query.path:
        raise CropLoadError("no crop folder was chosen")
    folder = os.path.abspath(os.path.expanduser(os.fspath(query.path)))
    if not os.path.isdir(folder):
        raise CropLoadError(f"crop folder not found: {folder}")
    query = _replace_path(query, folder)

    names = sorted(entry.name for entry in os.scandir(folder)
                   if entry.is_file() and not entry.name.startswith(".")
                   and entry.name.lower().endswith(".png"))
    if not names:
        return CropPlan(query=query, empty_reason=_empty_folder_reason(folder))
    limit = int(query.limit or 0)
    taken = names[:limit] if limit > 0 else names
    from .crops import PngCropSource

    source = PngCropSource(root=None,
                           reason=f"the {len(names):,} PNGs in {folder}")
    return CropPlan(query=query,
                    rows=tuple(os.path.join(folder, n) for n in taken),
                    matched=len(names), source_label=source.describe(),
                    _source=source)


def _empty_folder_reason(folder: str) -> str:
    """Say whether the folder is empty, or holds something that is not a crop.

    A folder of ``.tif`` is the common miss -- a user points this at the
    images rather than at the crops cut from them -- and "no crops here" does
    not tell them that.
    """
    try:
        others = sorted({os.path.splitext(entry.name)[1].lower()
                         for entry in os.scandir(folder)
                         if entry.is_file() and not entry.name.startswith(".")
                         and os.path.splitext(entry.name)[1]})
    except OSError as exc:
        raise CropLoadError(f"cannot list crop folder {folder}: {exc}") from exc
    if others:
        return (f"No crops in {folder}: it holds no .png, only "
                f"{', '.join(others)}. A crop folder is the '*_png' folder "
                f"under data/, not the images the crops were cut from.")
    subfolders = sorted(entry.name for entry in os.scandir(folder)
                        if entry.is_dir() and not entry.name.startswith("."))
    if subfolders:
        return (f"No crops in {folder}: it holds no .png, only the folders "
                f"{', '.join(subfolders[:6])}. Choose one of those.")
    return f"No crops in {folder}: the folder is empty."


def plan_crops(query: CropQuery) -> CropPlan:
    """Answer ``query`` without reading a pixel.

    :param query: what to select.
    :returns: a :class:`CropPlan`. Check :attr:`CropPlan.is_empty` before
        loading -- an empty plan is an answer, not a failure, and it carries
        the sentence explaining itself.
    :raises CropLoadError: an unknown source, or a path that is not there.
    """
    source = str(query.source or "").strip().lower()
    if source == CROP_SOURCE_DATABASE:
        return plan_from_database(query)
    if source == CROP_SOURCE_FOLDER:
        return plan_from_folder(query)
    raise CropLoadError(
        f"crop source must be one of "
        f"{[value for value, _label in CROP_SOURCES]}, got {query.source!r}")


def conform_crop(crop: np.ndarray, height: int, width: int) -> np.ndarray:
    """Centre a crop in a ``height`` x ``width`` frame, padding or trimming.

    Padding rather than resizing, and that is the whole of the decision: a
    stack has to be rectangular before a backbone sees it, and rescaling a
    crop changes the apparent size of the object in it, which is a phenotype.
    Zero-padding changes the background, which is not. It is also what
    :func:`spacr.crop_source.crop_at` already does to a box that runs off the
    edge of a field, so two crops of the same object arrive the same way
    whichever route cut them.

    :param crop: ``(height, width, channels)``.
    :param height: the target first axis.
    :param width: the target second axis.
    :returns: ``crop`` itself when it already fits, so the common case copies
        nothing.
    """
    have_h, have_w = int(crop.shape[0]), int(crop.shape[1])
    if have_h == height and have_w == width:
        return crop
    top = max(0, (have_h - height) // 2)
    left = max(0, (have_w - width) // 2)
    cut = crop[top:top + min(have_h, height), left:left + min(have_w, width)]
    pad_h, pad_w = height - cut.shape[0], width - cut.shape[1]
    if pad_h or pad_w:
        cut = np.pad(cut,
                     ((pad_h // 2, pad_h - pad_h // 2),
                      (pad_w // 2, pad_w - pad_w // 2),
                      *((0, 0),) * (cut.ndim - 2)),
                     mode="constant")
    return cut


def load_page(plan: CropPlan, start: int, stop: int,
              shape: Optional[Tuple[int, int, int]] = None
              ) -> Tuple[np.ndarray, int]:
    """Read one page of ``plan``'s crops.

    :param plan: from :func:`plan_crops`.
    :param start: first row of the page.
    :param stop: one past its last row.
    :param shape: ``(height, width, channels)`` every crop must come back as.
        ``None`` takes the first crop of the page as the template, which is
        how :func:`load_crops` fixes the stack's shape from page one.
    :returns: ``(page, conformed)`` -- the ``(n, height, width, channels)``
        array, and how many of its crops had to be padded or trimmed to fit.
    :raises CropLoadError: a crop could not be read, or came back with a
        different number of channels than the page's template -- which is a
        real difference between two crops and is not silently padded away.
    """
    from .crops import CropError

    source = plan._source
    if source is None:
        raise CropLoadError(
            "this plan carries no crop source, so it cannot load anything; "
            "plans are made by plan_crops()")
    rows = plan.rows[start:stop]
    if not rows:
        raise CropLoadError(f"page {start}:{stop} of this plan is empty")
    try:
        crops = source.get_many(rows)
    except CropError as exc:
        raise CropLoadError(
            f"crops {start}-{stop} could not be read from the "
            f"{source.kind} source: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise CropLoadError(
            f"crops {start}-{stop} could not be read: {exc}") from exc

    first = np.asarray(crops[0])
    if first.ndim == 2:
        first = first[:, :, None]
    if shape is None:
        shape = (int(first.shape[0]), int(first.shape[1]),
                 int(first.shape[2]))
    height, width, channels = (int(v) for v in shape)
    page = np.empty((len(crops), height, width, channels), dtype=first.dtype)
    conformed = 0
    for index, crop in enumerate(crops):
        array = np.asarray(crop)
        if array.ndim == 2:
            array = array[:, :, None]
        if int(array.shape[2]) != channels:
            raise CropLoadError(
                f"crop {start + index} has {array.shape[2]} channels and the "
                f"first one has {channels}. Two crops with different channels "
                f"are not one stack; load them separately.")
        if array.shape[0] != height or array.shape[1] != width:
            conformed += 1
            array = conform_crop(array, height, width)
        page[index] = array
    return page, conformed


@contextmanager
def _quiet_crop_paths():
    """Stop :mod:`spacr.crops` announcing every crop path for one load.

    :data:`spacr.crops.PRINT_CROP_PATHS` defaults to on, and it is right to:
    one crop that failed to open is invisible otherwise. A bulk load is the
    case it was not written for -- sixty thousand paths is sixty thousand
    lines, and in the app they are sixty thousand appends to a console widget
    on the GUI thread, which is the freeze this module exists to avoid.

    ON THIS THREAD ONLY, through :func:`spacr.crops.quiet_crop_paths`. The
    global flag is left alone because a load here runs on a worker for
    minutes on a network mount, and clearing a process-wide switch for that
    long silences whatever other screen is reading crops beside it -- which
    is somebody else's diagnostic, turned off by a load they did not start.
    """
    from . import crops

    with crops.quiet_crop_paths():
        yield


def load_crops(plan: CropPlan, *,
               progress: Optional[Callable[[int, int], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None,
               record: Optional[MutableMapping[str, Any]] = None
               ) -> np.ndarray:
    """Read every crop in ``plan``, a page at a time.

    Nothing here touches a widget and nothing here is Qt: ``progress`` and
    ``cancelled`` are plain callables, so a screen passes a signal emit and a
    flag and the same function drives a script or a test.

    :param plan: from :func:`plan_crops`.
    :param progress: called ``(done, total)`` after each page, on whatever
        thread this runs on. A caller on a worker thread must not touch a
        widget from it -- emit a signal.
    :param cancelled: called before each page; a true answer stops the load
        and returns the pages already read, which is why the result can be
        shorter than ``plan.count``. Checked between pages, so a stop takes
        at most one page to take effect.
    :param record: filled in with what was read -- ``matched``, ``selected``,
        ``dropped``, ``loaded``, ``crop_shape``, ``conformed``, ``capped``,
        ``source``, ``stopped``. ``matched`` minus ``selected`` is the cap;
        ``selected`` minus ``loaded`` is rows that could not be cut, plus
        whatever a stop left unread. JSON-serialisable, so a caller can keep
        it beside the matrix.
    :returns: ``(objects, height, width, channels)``, the layout
        :func:`spacr.embeddings.embed_array` documents. Every crop is padded
        or trimmed to the first one's frame; see :func:`conform_crop`.
    :raises CropLoadError: the plan is empty, or a page could not be read.
    """
    if plan.is_empty:
        raise CropLoadError(plan.empty_reason
                            or f"nothing matched {plan.query.describe()}")
    total = plan.count
    out: Optional[np.ndarray] = None
    shape: Optional[Tuple[int, int, int]] = None
    conformed = 0
    done = 0
    stopped = False
    with _quiet_crop_paths():
        for start, stop in plan.pages():
            if cancelled is not None and cancelled():
                stopped = True
                break
            page, page_conformed = load_page(plan, start, stop, shape)
            conformed += page_conformed
            if out is None:
                shape = (int(page.shape[1]), int(page.shape[2]),
                         int(page.shape[3]))
                out = np.empty((total,) + shape, dtype=page.dtype)
            out[start:stop] = page
            done = stop
            if progress is not None:
                progress(done, total)
    if out is None or done == 0:
        raise CropLoadError(
            f"loading {plan.query.describe()} was stopped before the first "
            f"page finished, so there are no crops to show")
    result = out if done == total else out[:done]
    if record is not None:
        record["matched"] = int(plan.matched)
        record["selected"] = int(plan.selected)
        record["dropped"] = int(plan.dropped)
        record["loaded"] = int(done)
        record["crop_shape"] = [int(v) for v in result.shape[1:]]
        record["conformed"] = int(conformed)
        record["capped"] = bool(plan.capped)
        record["stopped"] = bool(stopped)
        record["source"] = str(plan.source_label)
        record["query"] = plan.query.describe()
    return result
