"""Where an OPS run's tables live, and the gate that says B4 finished.

THE STORAGE CONTRACT for an OPS run, and the gate that says the object pass
finished. The contract is:

    IN measurements.db, authoritative:
        ops_geometry, ops_objects, ops_reads, ops_barcodes
    BESIDE IT, as a cache: parquet for `ops_reads` and `ops_barcodes` ...
    Written in the same step, with the row counts asserted equal, so the
    sidecar cannot silently drift from the authority. The reader prefers
    parquet and falls back to sqlite.

SQLITE IS THE AUTHORITY AND PARQUET IS A CACHE, which is a decision and not a
detail. The two can disagree, so one of them has to be right by definition --
otherwise a reader that picked the faster one would sometimes be reading
yesterday's numbers with no way to tell. :func:`write_table` writes both in
one call and asserts the counts match before either is visible as complete;
:func:`read_table` prefers the cache and falls back without complaining,
because a missing cache is a performance question and not a correctness one.

B5 IS A GATE, NOT A STEP. 372: "no sequencing or phenotype channel is read
until `ops_objects` is on disk and its row count checked". :func:`objects_ready`
is that sentence, and it returns a REASON when the answer is no -- a gate that
only says "not yet" makes the operator guess which half failed.

    THE ORDINARY MEASUREMENT TABLES ARE KEYED ON THE SAME OBJECT IDS, so
    nothing downstream needs to know the run was OPS. That is why B4's
    numbering had to be deterministic: these ids are a join key across
    tables written at different times by different phases.
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
OPS_TABLES = ("ops_geometry", "ops_objects", "ops_reads", "ops_barcodes")

#: The two that also get a parquet sidecar. 372 names these specifically --
#: they are "the wide numeric tables where a columnar scan is worth an order
#: of magnitude". The other two are small and read whole, so a cache would be
#: two things to keep in step for no gain.
CACHED_TABLES = ("ops_reads", "ops_barcodes")

#: The table B5 gates on.
OBJECTS_TABLE = "ops_objects"


def _sidecar_path(db_path: str, table: str) -> str:
    """Where ``table``'s parquet cache sits, beside the database."""
    directory = os.path.dirname(os.path.abspath(db_path))
    stem = os.path.splitext(os.path.basename(db_path))[0]
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
    :raises StoreError: on an unknown table, or when the sidecar it just
        wrote does not have the same number of rows as the database.
    """
    if table not in OPS_TABLES:
        raise StoreError(
            f"{table!r} is not an OPS table; 372's storage contract names "
            f"{', '.join(OPS_TABLES)}. A table outside that list would not be "
            f"read by anything and would look like data that had been kept.")

    rows = int(len(frame))
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        frame.to_sql(table, conn, if_exists=if_exists, index=False)
        stored = int(conn.execute(
            f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    if table not in CACHED_TABLES:
        return stored

    path = _sidecar_path(db_path, table)
    try:
        frame.to_parquet(path, index=False)
    except Exception as failure:                       # noqa: BLE001
        # A CACHE THAT CANNOT BE WRITTEN IS NOT A FAILED RUN. The authority
        # is already on disk and complete; losing the sidecar costs speed.
        # Removing a stale one matters more than creating a new one, because
        # a stale cache is the only way this design can hand back wrong data.
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


def _remove_stale(path: str) -> None:
    """Delete a sidecar that must not be read. Never raises."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        LOG.warning("could not remove the stale parquet cache %s", path)


def _parquet_rows(path: str) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def read_table(db_path: str, table: str, *, prefer_cache: bool = True):
    """Read one OPS table, preferring its parquet cache.

    A missing or unreadable cache falls back to sqlite silently, because that
    is a speed question. A cache whose row count disagrees with the database
    is NOT silent -- it is removed and the authority is returned, since a
    disagreement is the one failure this arrangement can produce.

    :raises StoreError: when the table is not in the database at all.
    """
    import pandas as pd

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

    with sqlite3.connect(str(db_path), timeout=30) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception as failure:                   # noqa: BLE001
            raise StoreError(
                f"{table} is not in {db_path}: {failure}") from failure


def row_count(db_path: str, table: str) -> Optional[int]:
    """Rows in one table, or ``None`` when the table is not there."""
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
    """Whether Phase C may start, and why not when it may not."""

    ready: bool
    rows: Optional[int]
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ready


def objects_ready(db_path: str, *, minimum: int = 1) -> Readiness:
    """B5: may a sequencing or phenotype channel be read yet?

    372 states the gate as one sentence -- "no sequencing or phenotype
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
    """
    if not os.path.exists(str(db_path)):
        return Readiness(False, None,
                         f"there is no database at {db_path}. Phase B writes "
                         f"it; run the object pass before reading channels.")
    rows = row_count(db_path, OBJECTS_TABLE)
    if rows is None:
        return Readiness(False, None,
                         f"{db_path} has no {OBJECTS_TABLE} table. B4 numbers "
                         f"the objects and Phase D writes them; neither has "
                         f"run for this well.")
    if rows < minimum:
        return Readiness(False, rows,
                         f"{OBJECTS_TABLE} has {rows} row(s), below the {minimum} "
                         f"required. Reading channels against an empty object "
                         f"list produces empty results that look like a "
                         f"finished run.")
    return Readiness(True, rows)
