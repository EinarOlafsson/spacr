"""Where an OPS run's tables live, and the gate that says the objects exist.

The storage contract has two halves. In ``measurements.db``, which is
authoritative: ``ops_geometry``, ``ops_phenotype``, ``ops_objects``,
``ops_reads`` and ``ops_barcodes``. Beside it, as a cache: parquet copies of
``ops_reads`` and ``ops_barcodes``, written in the same step with their row
counts asserted equal, so the sidecar cannot drift from the authority
unnoticed.

SQLITE IS THE AUTHORITY AND PARQUET IS A CACHE, which is a decision and not a
detail. The two can disagree, so one of them has to be right by definition --
otherwise a reader that picked the faster one would sometimes be reading
yesterday's numbers with no way to tell. :func:`write_table` writes both in
one call and asserts the counts match before either is visible as complete;
:func:`read_table` prefers the cache and falls back without complaining,
because a missing cache is a performance question and not a correctness one.

READINESS IS A GATE, NOT A STEP. No sequencing or phenotype channel is read
until ``ops_objects`` is on disk and its row count checked.
:func:`objects_ready` is that sentence, and it returns a REASON when the
answer is no -- a gate that only says "not yet" makes the operator guess
which half failed.

Other measurement tables use the same object ids, so later steps need no
special case. The numbering is deterministic for that reason: those ids join
tables written at different times.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.ops_store")

__all__ = [
    "StoreError",
    "OPS_TABLES", "CACHED_TABLES", "OBJECTS_TABLE",
    "write_table", "read_table", "row_count", "objects_ready",
    "Readiness",
]


class StoreError(ValueError):
    """A store that cannot be trusted, with the way out in the text."""


#: The authoritative tables, in the order the phases write them.
#:
#: FIVE, AND 372's PHASE D NAMED FOUR. ``ops_phenotype`` is A4's output --
#: where each phenotype field's centre lies in the sequencing well frame and
#: which stitched tile covers it. Phase D was written before A4 was built and
#: assumed the placement would live in the phenotype run's own measurement
#: tables; it cannot, because the placement is what those tables are keyed
#: BY. Keeping it out of the contract would have left the one table a
#: phenotype measurement pass has to read as an undeclared file beside the
#: database.
OPS_TABLES = ("ops_geometry", "ops_phenotype", "ops_objects", "ops_reads",
              "ops_barcodes")

#: The two that also get a parquet sidecar. 372 names these specifically --
#: they are "the wide numeric tables where a columnar scan is worth an order
#: of magnitude". The other two are small and read whole, so a cache would be
#: two things to keep in step for no gain.
CACHED_TABLES = ("ops_reads", "ops_barcodes")

#: The table the readiness gate asks about.
OBJECTS_TABLE = "ops_objects"


def _resolved(db_path) -> str:
    """``db_path`` with ``~`` and ``$VARS`` expanded, as the funnel expands it.

    ONE RESOLUTION PER CALL, and it has to be this one. `write_table` now
    writes through :func:`spacr.tabular.write_database`, which resolves the
    path itself -- so a caller passing ``~/run/measurements.db`` had the
    database written to the expanded path while :func:`row_count` opened the
    literal one and :func:`_sidecar_path` put the parquet cache beside a
    directory named ``~``. The row-count check then compared a table that
    existed against one that did not, and the cache it wrote was never read.
    Nothing raised; the run simply lost its cache and its verification.
    """
    from .tabular import resolve_path

    return resolve_path(db_path)


def _sidecar_path(db_path: str, table: str) -> str:
    """Where ``table``'s parquet cache sits, beside the database."""
    directory = os.path.dirname(os.path.abspath(_resolved(db_path)))
    stem = os.path.splitext(os.path.basename(_resolved(db_path)))[0]
    return os.path.join(directory, f"{stem}.{table}.parquet")


def write_table(db_path: str, table: str, frame, *,
                if_exists: str = "replace") -> int:
    """Write one OPS table to sqlite, and its parquet cache if it has one.

    BOTH IN ONE CALL, because 372 asks for the row counts to be asserted
    equal and two calls could not do that -- a caller who wrote the database
    and then crashed would leave a sidecar describing a different run, and
    nothing would notice until a reader silently preferred it.

    :param db_path: the measurements database.
    :param table: one of :data:`OPS_TABLES`.
    :param frame: a DataFrame.
    :returns: rows written.
    :raises StoreError: on an unknown table; when the sidecar it just
        wrote does not have the same number of rows as the database; or when
        ``ops_objects`` or ``ops_barcodes`` would hold two rows for one
        object, inside the frame or between the frame and the table it is
        appended to. One row per object is a UNIQUE constraint in the schema,
        and nothing is written when it would be broken.
    """
    db_path = _resolved(db_path)
    if table not in OPS_TABLES:
        raise StoreError(
            f"{table!r} is not an OPS table; the storage contract names "
            f"{', '.join(OPS_TABLES)}. A table outside that list would not be "
            f"read by anything and would look like data that had been kept.")

    rows = int(len(frame))
    from .tabular import write_database

    key = _object_key(table, frame.columns)
    appending = if_exists == "append" and row_count(db_path, table) is not None
    if key:
        _refuse_repeated_keys(table, frame, key)
        if appending:
            _constrain(db_path, table, key, created=False)
    try:
        write_database(frame, db_path, table, if_exists=if_exists,
                       canonicalise=False, index=False)
    except Exception as failure:                       # noqa: BLE001
        refusal = _uniqueness_refusal(failure)
        if not key or refusal is None:
            raise
        raise StoreError(_collision_message(
            db_path, table, frame, key, refusal)) from failure
    if key:
        _constrain(db_path, table, key, created=not appending)
    stored = row_count(db_path, table)
    if stored is None:
        raise StoreError(
            f"{table} is not in {db_path} after writing it, which means the "
            f"write did not land. Nothing downstream should read a table "
            f"that is not there.")

    if table not in CACHED_TABLES:
        return stored

    path = _sidecar_path(db_path, table)
    try:
        frame.to_parquet(path, index=False)
    except Exception as failure:                       # noqa: BLE001
        _remove_stale(path)
        LOG.warning("no parquet cache for %s (%s); sqlite remains "
                    "authoritative", table, failure)
        return stored

    cached = _parquet_rows(path)
    if cached != stored:
        _remove_stale(path)
        raise StoreError(
            f"the parquet cache for {table} has {cached} rows and the "
            f"database has {stored}. The cache has been removed so reads "
            f"fall back to the authority rather than to the disagreement.")
    return stored


def _uniqueness_refusal(
        failure: BaseException) -> Optional[sqlite3.IntegrityError]:
    """SQLite's constraint refusal behind a failed write, or ``None``.

    pandas 2 lets ``to_sql`` raise :class:`sqlite3.IntegrityError` as it is;
    pandas 3 raises ``pandas.errors.DatabaseError("Execution failed")`` from
    it. Catching only the first let every refused append on pandas 3 escape
    as "Execution failed", where the store names the keys the append shared
    with the table.

    :param failure: what the write raised.
    :returns: the IntegrityError itself, the one ``failure`` was raised
        from, or ``None`` when the write failed for another reason.
    """
    if isinstance(failure, sqlite3.IntegrityError):
        return failure
    cause = failure.__cause__
    return cause if isinstance(cause, sqlite3.IntegrityError) else None


_ONE_ROW_PER_OBJECT = ("ops_objects", "ops_barcodes")


def _object_key(table: str, columns) -> Tuple[str, ...]:
    """The columns that name one object in ``table``, or ``()`` for none.

    372 PART 14-L, V11c: the storage contract says ``UNIQUE(plate, well,
    object_id)``, and the table on disk had plain columns and no index, so
    appending a well twice doubled it without an error. ``ops_objects`` and
    ``ops_barcodes`` hold one row per object; ``ops_reads`` holds many and
    ``ops_geometry`` none. The key is ``object_id`` together with whichever
    of ``plate`` and ``well`` the frame carries, because ids are numbered per
    well and two wells legitimately share them.

    :param table: the OPS table being written.
    :param columns: the frame's columns.
    :returns: the key columns, or ``()`` when the table has no one-row-per-
        object contract or the frame has no ``object_id`` to hold it to.
    """
    names = [str(column) for column in columns]
    if table not in _ONE_ROW_PER_OBJECT or "object_id" not in names:
        return ()
    return tuple(name for name in ("plate", "well") if name in names) + (
        "object_id",)


def _quoted(name: str) -> str:
    """A SQLite identifier in double quotes, embedded quotes doubled.

    :param name: a table, index or column name.
    :returns: the quoted identifier.
    """
    return '"' + str(name).replace('"', '""') + '"'


def _named(key: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Key values as ``plate='p1', well='A1', object_id=2``, joined by ``"; "``.

    :param key: the key columns.
    :param rows: one tuple of values per key.
    :returns: the readable list.
    """
    return "; ".join(
        ", ".join(f"{column}={value!r}" for column, value in zip(key, row))
        for row in rows)


def _refuse_repeated_keys(table: str, frame, key: Sequence[str]) -> None:
    """Refuse a frame that repeats a key, before anything is written.

    Checked in the frame first so a replace never drops a good table for a
    bad one: the write that would have replaced it does not start.

    :param table: the OPS table being written.
    :param frame: the DataFrame to be written.
    :param key: the key columns.
    :raises StoreError: naming how many rows share a key, and up to five of
        the keys.
    """
    columns = list(key)
    repeated = frame.duplicated(subset=columns, keep=False)
    if not bool(repeated.any()):
        return
    keys = frame.loc[repeated, columns].drop_duplicates()
    raise StoreError(
        f"{table} would hold {int(repeated.sum())} rows for {len(keys)} "
        f"object(s) on ({', '.join(columns)}), e.g. "
        f"{_named(columns, keys.head(5).itertuples(index=False, name=None))}. "
        f"One row per object is a constraint of the storage contract, and "
        f"every later join on object_id would count these objects more than "
        f"once. Nothing was written.")


def _constrain(db_path: str, table: str, key: Sequence[str], *,
               created: bool) -> None:
    """Put the one-row-per-object key into the schema as a UNIQUE index.

    A pandas ``replace`` drops the table and its indexes with it, so this
    runs after every write; ``IF NOT EXISTS`` makes it free when the index
    is there. Before an append it runs on the existing table, so the append
    itself fails inside its transaction and rolls back whole.

    :param db_path: the resolved database path.
    :param table: the OPS table.
    :param key: the key columns.
    :param created: whether this call's write created the table. A table
        created with keys SQLite considers equal (``1`` and ``1.0``, which a
        DataFrame keeps apart) is removed, so no table with duplicate ids is
        left for :func:`objects_ready` to pass; an existing table is the
        caller's data and is only reported.
    :raises StoreError: when the table already holds duplicate keys.
    """
    columns = ", ".join(_quoted(column) for column in key)
    index = _quoted(f"{table}_unique_{'_'.join(key)}")
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        present = {row[1] for row in conn.execute(
            f"PRAGMA table_info({_quoted(table)})")}
        if not set(key) <= present:
            return
        try:
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index} ON "
                         f"{_quoted(table)} ({columns})")
            return
        except sqlite3.IntegrityError:
            duplicates = conn.execute(
                f"SELECT {columns} FROM {_quoted(table)} GROUP BY {columns} "
                f"HAVING COUNT(*) > 1 LIMIT 5").fetchall()
            if created:
                conn.execute(f"DROP TABLE {_quoted(table)}")
                conn.commit()
    if created:
        _remove_stale(_sidecar_path(db_path, table))
        raise StoreError(
            f"{table} was written with more than one row for "
            f"{_named(key, duplicates)} -- keys the frame kept apart but "
            f"SQLite treats as equal. The table was removed rather than left "
            f"with duplicate ids for a later phase to join on.")
    raise StoreError(
        f"{table} in {db_path} already holds more than one row for "
        f"{_named(key, duplicates)}; it was written without the uniqueness "
        f"constraint. Rewrite it (if_exists='replace') before appending to "
        f"it.")


def _collision_message(db_path: str, table: str, frame, key: Sequence[str],
                       failure: Exception) -> str:
    """Name the keys an append shared with the table it was refused by.

    :param db_path: the resolved database path.
    :param table: the OPS table.
    :param frame: the frame whose append was refused.
    :param key: the key columns.
    :param failure: SQLite's own error, quoted when no shared key is found.
    :returns: the sentence for the :class:`StoreError`.
    """
    columns = list(key)
    incoming = ", ".join(_quoted(column) for column in columns)
    matched = " AND ".join(
        f"existing.{_quoted(column)} = incoming.{_quoted(column)}"
        for column in columns)
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        conn.execute(f"CREATE TEMP TABLE incoming_keys ({incoming})")
        conn.executemany(
            f"INSERT INTO incoming_keys VALUES "
            f"({', '.join('?' for _ in columns)})",
            frame[columns].itertuples(index=False, name=None))
        joined = (f"FROM {_quoted(table)} AS existing JOIN incoming_keys AS "
                  f"incoming ON {matched}")
        shared = conn.execute(f"SELECT COUNT(*) {joined}").fetchone()[0]
        examples = conn.execute(
            f"SELECT {', '.join('incoming.' + _quoted(c) for c in columns)} "
            f"{joined} LIMIT 5").fetchall()
    if not shared:
        return (f"appending to {table} broke its uniqueness constraint "
                f"({failure}); nothing was appended.")
    return (f"{table} already holds {shared} of the {len(frame)} object(s) "
            f"being appended, e.g. {_named(columns, examples)}. One object "
            f"would have two rows, so the append was rolled back and the "
            f"table is as it was. Write each well once, or replace it.")


def _remove_stale(path: str) -> None:
    """Delete a sidecar that must not be read. Never raises."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        LOG.warning("could not remove the stale parquet cache %s", path)


def _parquet_rows(path: str) -> int:
    """How many rows a parquet sidecar says it holds.

    Read from the file's footer rather than by loading it, because this
    runs on every write to check the cache against the database before
    either is trusted.

    :param path: the sidecar to count.
    :returns: the row count recorded in the file's metadata.
    """
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def read_table(db_path: str, table: str, *, prefer_cache: bool = True):
    """Read one OPS table, preferring its parquet cache.

    A missing or unreadable cache falls back to sqlite silently, because that
    is a speed question. A cache whose row count disagrees with the database
    is NOT silent -- it is removed and the authority is returned, since a
    disagreement is the one failure this arrangement can produce.

    :param db_path: the run's sqlite database, which is the authority.
    :param table: which of :data:`OPS_TABLES` to read.
    :raises StoreError: when the table is not in the database at all.
    """
    import pandas as pd

    db_path = _resolved(db_path)

    if prefer_cache and table in CACHED_TABLES:
        path = _sidecar_path(db_path, table)
        if os.path.exists(path):
            try:
                cached = pd.read_parquet(path)
            except Exception:                          # noqa: BLE001
                cached = None
            if cached is not None:
                stored = row_count(db_path, table)
                if stored is None or len(cached) == stored:
                    return cached
                _remove_stale(path)
                LOG.warning(
                    "the parquet cache for %s disagreed with the database "
                    "(%d vs %s) and was removed", table, len(cached), stored)

    from .tabular import _read_query

    with sqlite3.connect(str(db_path), timeout=30) as conn:
        try:
            return _read_query(conn, f"SELECT * FROM {table}",
                               canonicalise=False, report=None)
        except Exception as failure:                   # noqa: BLE001
            raise StoreError(
                f"{table} is not in {db_path}: {failure}") from failure


def row_count(db_path: str, table: str) -> Optional[int]:
    """Rows in one table, or ``None`` when the table is not there.

    :param db_path: the run's sqlite database.
    :param table: the table to count.
    """
    db_path = _resolved(db_path)
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            present = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()
            if not present:
                return None
            return int(conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return None


@dataclass(frozen=True)
class Readiness:
    """Whether object sampling may start, and why not when it may not.

    :param ready: the verdict. Also what ``bool(readiness)`` answers, so a
        caller can write ``if not objects_ready(db):`` and still reach
        :attr:`reason` when it needs to say why.
    :param rows: how many objects the table holds, or ``None`` when there is
        no table to count -- which is a different failure from a table with
        no rows in it, and the two are told apart here rather than by the
        caller.
    :param reason: one sentence naming which of the three things went wrong,
        empty when nothing did.
    """

    ready: bool
    rows: Optional[int]
    reason: str = ""

    def __bool__(self) -> bool:
        """The verdict alone, so the result reads as a condition.

        ``if not objects_ready(db):`` is the natural way to write the
        gate, and it must answer the verdict rather than "an object
        exists". :attr:`reason` is still there to say why the answer was
        no.

        :returns: :attr:`ready`.
        """
        return self.ready


def objects_ready(db_path: str, *, minimum: int = 1) -> Readiness:
    """May a sequencing or phenotype channel be read yet?

    The gate is one sentence -- "no sequencing or phenotype
    channel is read until `ops_objects` is on disk and its row count
    checked" -- and both halves matter. On disk without a count check would
    pass an empty table, and an empty ops_objects means every later phase
    silently produces nothing while appearing to run.

    IT RETURNS A REASON, not just a verdict. "Not yet" leaves the operator
    guessing which of three things went wrong: no database, no table, or a
    table with nothing in it. Those have different fixes.

    :param minimum: the fewest objects a real well can have. One is the
        honest floor -- a well with a single nucleus is a bad well, not an
        impossible one -- so this exists to be raised by a caller who knows
        their plate, not to encode a guess here.
    :param db_path: the run's sqlite database. Its ABSENCE is one of the
        three answers, so this is not required to exist.
    """
    db_path = _resolved(db_path)
    if not os.path.exists(str(db_path)):
        return Readiness(False, None,
                         f"there is no database at {db_path}. The object "
                         f"pass writes it; run that before reading "
                         f"channels.")
    rows = row_count(db_path, OBJECTS_TABLE)
    if rows is None:
        return Readiness(False, None,
                         f"{db_path} has no {OBJECTS_TABLE} table. The "
                         f"numbering pass assigns the object ids and the "
                         f"storage step writes them; neither has run for "
                         f"this well.")
    if rows < minimum:
        return Readiness(False, rows,
                         f"{OBJECTS_TABLE} has {rows} row(s), below the {minimum} "
                         f"required. Reading channels against an empty object "
                         f"list produces empty results that look like a "
                         f"finished run.")
    return Readiness(True, rows)
