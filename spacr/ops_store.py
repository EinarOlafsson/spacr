"""Storage of an OPS run's tables, and the check that its objects exist.

THE OPS TABLES LIVE IN THE SPACR STORE (item 0an). Every OPS table --
``ops_geometry``, ``ops_phenotype``, ``ops_objects``, ``ops_reads``,
``ops_barcodes``, ``ops_merge``, ``ops_cell_cycle`` and
``ops_cell_cycle_checks`` -- is written through the store's one writer
(:func:`spacr.store.writer`, one Parquet part per table per well, the
recorded dtype map applied) into the run's one store, the ``measurements``
folder beside the ``measurements.db`` path the phases pass around, and
read back through the store's one reader (:func:`spacr.store.open`). The
store table is the OPS name without its ``ops_`` prefix (``geometry``,
``merge``, ``barcodes``, ...), so ``ph_tables``' ``cell``, ``nucleus`` and
``cytoplasm`` and these tables share one catalog, ``spacr.duckdb``, that
the Database Browser and every other module read. No OPS table is written
to SQLite any more.

The ``measurements.db`` path is still what the phases hand each other: it
names the run (:func:`_store_project` maps ``<run>/measurements.db`` to
``<run>/measurements``). An OLDER RUN that kept its tables in that SQLite
file is read through the same reader, :func:`spacr.store.open` on the file,
when the store does not hold the table; nothing writes there.

Each well is its own part, so a phase that rewrites one well's rows
(:func:`_write_well`) writes only that well; the catalog supersedes the
well's earlier part in one transaction. Rows come back per plate and well,
in the order each well's rows were written.

Readiness is checked before any channel is read: no sequencing or phenotype
channel is read until ``ops_objects`` is in the store and its row count
checked. :func:`objects_ready` performs this check and returns a reason when
the answer is no, so the operator can see which condition failed.

Other measurement tables use the same object ids, so later steps need no
special case. The numbering is deterministic for that reason: those ids join
tables written at different times.
"""
from __future__ import annotations

import logging
import os
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
    """Raised for an inconsistent OPS store; the message says how to recover."""


#: The OPS tables, in the order the phases write them.
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
              "ops_barcodes", "ops_merge", "ops_cell_cycle", "ops_cell_cycle_checks")

#: The wide tables read back in the types they were written in. Until item
#: 0an these two had a parquet cache beside the SQLite authority, which
#: readers preferred, so their readers were written against the written
#: types (a bool stays a bool). The other tables were read from SQLite,
#: whose readers were written against its storage classes: integers and
#: booleans come back as int64 and floats as float64 (:func:`_as_read`).
CACHED_TABLES = ("ops_reads", "ops_barcodes")

#: The table the readiness gate asks about.
OBJECTS_TABLE = "ops_objects"

#: The columns each table's queries always return, never narrowed: the
#: store's key columns (Measure-style ``object_key`` where a row is one
#: phenotype cell, the sequencing ``object_id`` where it is one nucleus).
_KEY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "ops_geometry": ("cycle", "site"),
    "ops_phenotype": ("site",),
    "ops_objects": ("object_id",),
    "ops_reads": ("object_id",),
    "ops_barcodes": ("object_id",),
    "ops_merge": ("object_key", "site", "label", "object_id"),
    "ops_cell_cycle": ("object_key", "site", "label"),
    "ops_cell_cycle_checks": (),
}

#: The partition value of a table (or frame) without a ``plate`` or ``well``
#: column; it is dropped again on reading.
_WHOLE = "_all_"


def _resolved(db_path) -> str:
    """``db_path`` with ``~`` and ``$VARS`` expanded, as the funnel expands it.

    ONE RESOLUTION PER CALL, and it has to be this one: a caller passing
    ``~/run/measurements.db`` must reach the same store whichever function
    it calls.
    """
    from .tabular import resolve_path

    return resolve_path(db_path)


def _store_project(db_path) -> str:
    """The spaCR store folder of the run whose OPS database path is ``db_path``.

    ``<run>/measurements.db`` is ``<run>/measurements``, the store
    ``ph_tables`` writes the cell tables to; any other ``<dir>/<name>.<ext>``
    is ``<dir>/<name>``.
    """
    resolved = os.path.abspath(_resolved(db_path))
    stem, suffix = os.path.splitext(resolved)
    return stem if suffix else resolved + ".store"


def _store_name(table: str) -> str:
    """The store table of an OPS table: its name without ``ops_``."""
    return table[4:] if table.startswith("ops_") else table


def _checked(table: str) -> None:
    """Refuse a table outside :data:`OPS_TABLES`."""
    if table not in OPS_TABLES:
        raise StoreError(
            f"{table!r} is not an OPS table; the storage contract names "
            f"{', '.join(OPS_TABLES)}. A table outside that list would not be "
            f"read by anything and would look like data that had been kept.")


def _open(project: str):
    """The store at ``project`` through the shared reader, or None."""
    from . import store

    if not os.path.isfile(os.path.join(project, store.CATALOG)):
        return None
    return store.open(project)


def _stored(project: str, name: str) -> Optional[Dict[Tuple[str, str], int]]:
    """``(plate, well) -> rows`` of ``name`` in the store, or None if absent."""
    opened = _open(project)
    if opened is None:
        return None
    try:
        parts = [f for f in opened._files if f["table_name"] == name
                 and f["kind"] == "parquet"]
        if name not in opened.tables() or not parts:
            return None
        found: Dict[Tuple[str, str], int] = {}
        for entry in parts:
            key = (str(entry["plate"]), str(entry["well"]))
            found[key] = found.get(key, 0) + int(entry["rows"])
        return found
    finally:
        opened.close()


def _legacy(db_path) -> Optional[str]:
    """The older SQLite file at ``db_path`` when it is one, else None."""
    path = _resolved(db_path)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return path if handle.read(16) == b"SQLite format 3\x00" else None


def _legacy_tables(path: str) -> List[str]:
    """The tables of an older SQLite file, through the shared reader."""
    from . import store

    with store.open(path) as opened:
        return opened.tables()


def _exists(db_path) -> bool:
    """Whether the run at ``db_path`` has an OPS store, or an older database."""
    from . import store

    project = _store_project(db_path)
    return (os.path.isfile(os.path.join(project, store.CATALOG))
            or _legacy(db_path) is not None)


def _object_keys(frame, table: str, well: str):
    """``frame`` with the Measure-style ``object_key`` of each phenotype cell.

    The integer key of :func:`spacr.ops_run_infra.integer_cell_key` (row,
    column, field and label), as ``ph_tables`` and Measure key their rows,
    for the tables whose rows are phenotype cells (``site``, ``label``). A
    frame that already has it, lacks those columns, or whose well is not a
    letter and a number is returned as it is.
    """
    import numpy as np

    if ("object_key" not in _KEY_COLUMNS.get(table, ()) or "object_key"
            in frame.columns or not {"site", "label"} <= set(frame.columns)
            or not len(frame)):
        return frame
    from . import schema
    from .ops_run_infra import integer_cell_key

    try:
        row, column = schema.parse_well(well, strict=True)
        site = frame["site"].to_numpy()
        label = frame["label"].to_numpy()
        if not (np.issubdtype(site.dtype, np.integer)
                and np.issubdtype(label.dtype, np.integer)):
            return frame
        keys = integer_cell_key(int(row[1:]), int(column[1:]), site, label)
    except (ValueError, TypeError):
        return frame
    return frame.assign(object_key=keys)


def _partitions(frame) -> Dict[Tuple[str, str], Any]:
    """``frame`` split into its ``(plate, well)`` parts, in first-seen order.

    An empty frame is one empty part, so the table exists with no rows.
    """
    if not len(frame):
        return {(_WHOLE, _WHOLE): frame}
    plates = (frame["plate"].astype(str) if "plate" in frame.columns
              else None)
    wells = frame["well"].astype(str) if "well" in frame.columns else None
    if plates is None and wells is None:
        return {(_WHOLE, _WHOLE): frame}
    import pandas as pd

    plate = plates if plates is not None else pd.Series(
        _WHOLE, index=frame.index)
    well = wells if wells is not None else pd.Series(_WHOLE, index=frame.index)
    keys = list(zip(plate, well))
    order = list(dict.fromkeys(keys))
    if len(order) == 1:
        return {order[0]: frame}
    codes = pd.Series(keys, index=frame.index)
    return {key: frame[(codes == key).to_numpy()] for key in order}


def _write_part(project: str, table: str, frame, plate: str, well: str,
                schema=None, manifest: bool = False):
    """Write one well's rows of ``table`` through the store's one writer.

    :func:`spacr.store.write_table` with the recorded dtype map
    (``narrow``), the table's key columns and no sort: the rows keep the
    order the phase wrote them in. ``schema`` writes an empty part of that
    schema, which supersedes the well's earlier rows.
    """
    from . import store

    frame = _object_keys(frame, table, well)
    frame = frame.drop(columns=[c for c in ("plate", "well")
                                if c in frame.columns])
    keys = [c for c in _KEY_COLUMNS.get(table, ()) if c in frame.columns]
    if schema is not None:
        with store.writer(project, _store_name(table), plate=plate,
                          well=well, sort_by=(), key_columns=keys,
                          schema=schema) as part:
            part.append(schema.empty_table())
        return part.manifest if manifest else 0
    written = store.write_table(project, _store_name(table), frame,
                                plate=plate, well=well, sort_by=(),
                                key_columns=keys, narrow=True)
    return written if manifest else int(written["files"][0]["rows"])


def _write_parts(project: str, table: str, parts) -> None:
    """Write ``(plate, well) -> frame`` parts, then register them together.

    A part that fails takes the parts this call already wrote with it:
    their waiting manifests and files are removed, so a later registration
    cannot make half of the write visible.
    """
    from . import store

    written = []
    try:
        for (plate, well), part in parts.items():
            written.append(_write_part(project, table, part, plate, well,
                                       manifest=True))
    except BaseException:
        for manifest in written:
            for entry in manifest["files"]:
                store._remove(os.path.join(project, entry["path"]))
            store._remove(os.path.join(project, store._PENDING,
                                       f"{manifest['uuid']}.json"))
        raise
    store.register(project)


def _empty_schema(project: str, name: str):
    """The stored schema of ``name`` without its partition columns, or None."""
    opened = _open(project)
    if opened is None:
        return None
    try:
        if name not in opened.tables():
            return None
        from .store import _arrow_of, _ident

        schema = _arrow_of(opened._conn,
                           f"SELECT * FROM {_ident(name)} LIMIT 0").schema
    finally:
        opened.close()
    import pyarrow as pa

    return pa.schema([f for f in schema if f.name not in ("plate", "well")])


def _adopt_legacy(db_path: str, table: str) -> None:
    """Move an older run's SQLite rows of ``table`` into the store, once.

    A run from before item 0an kept its OPS tables in ``measurements.db``.
    The first write of a table to such a run's store copies that table's
    rows there, every well through the same writer, so a phase that
    rewrites one well does not hide the other wells the database held. The
    SQLite file is left as it was.

    :raises StoreError: when the older table holds two rows for one object;
        nothing is copied.
    """

    project = _store_project(db_path)
    if _stored(project, _store_name(table)) is not None:
        return
    legacy = _legacy(db_path)
    if legacy is None or table not in _legacy_tables(legacy):
        return
    frame = _read_rows(db_path, table)
    key = _object_key(table, frame.columns)
    if key:
        _refuse_repeated_keys(table, frame, key)
    os.makedirs(project, exist_ok=True)
    _write_parts(project, table, _partitions(frame))
    LOG.info("%s: %d rows of the older %s copied into %s", table, len(frame),
             legacy, project)


def write_table(db_path: str, table: str, frame, *,
                if_exists: str = "replace") -> int:
    """Write one OPS table to the run's spaCR store.

    Each ``(plate, well)`` of ``frame`` is written as its own part through
    :func:`spacr.store.writer` and the parts are registered in one catalog
    transaction. ``replace`` makes ``frame`` the whole table: a well it
    does not hold is removed. ``append`` adds its rows to the wells they
    belong to, which keep their earlier rows.

    :param db_path: the run's OPS database path (:func:`_store_project`).
    :param table: one of :data:`OPS_TABLES`.
    :param frame: a DataFrame.
    :param if_exists: ``"replace"`` or ``"append"``.
    :returns: the rows the table holds afterwards.
    :raises StoreError: on an unknown table; or when ``ops_objects``,
        ``ops_barcodes`` or ``ops_merge`` would hold two rows for one
        object, inside the frame or between the frame and the well it is
        appended to. One row per object is a constraint of the storage
        contract, and nothing is written when it would be broken.
    """
    import pandas as pd


    _checked(table)
    if if_exists not in ("replace", "append"):
        raise StoreError(f"if_exists must be 'replace' or 'append'; got "
                         f"{if_exists!r}")
    key = _object_key(table, frame.columns)
    if key:
        _refuse_repeated_keys(table, frame, key)
    if if_exists == "append":
        _adopt_legacy(db_path, table)
    project = _store_project(db_path)
    name = _store_name(table)
    present = _stored(project, name) or {}
    parts = _partitions(frame)
    if if_exists == "append":
        for where in [w for w in parts if w in present]:
            existing = _read(project, table, [where])
            combined = pd.concat([existing, parts[where]], ignore_index=True)
            if key:
                _refuse_appended_keys(table, existing, parts[where], key)
            parts[where] = combined
    else:
        for where in [w for w in present if w not in parts]:
            parts[where] = frame.iloc[:0]
    os.makedirs(project, exist_ok=True)
    _write_parts(project, table, parts)
    stored = row_count(db_path, table)
    if stored is None:
        raise StoreError(
            f"{table} is not in {project} after writing it, which means the "
            f"write did not land. Nothing downstream should read a table "
            f"that is not there.")
    return stored


def _write_well(db_path: str, table: str, frame, plate: str,
                well: str) -> int:
    """Write one well's rows of ``table``, replacing that well's earlier rows.

    Only the well's part is written; the catalog supersedes its earlier
    part, and the other wells are not read or rewritten.

    :param db_path: the run's OPS database path.
    :param table: an OPS table.
    :param frame: this well's rows.
    :param plate: the plate name.
    :param well: the well name.
    :returns: the rows the table holds afterwards.
    """
    from . import store

    _checked(table)
    key = _object_key(table, frame.columns)
    if key:
        _refuse_repeated_keys(table, frame, key)
    _adopt_legacy(db_path, table)
    project = _store_project(db_path)
    name = _store_name(table)
    if len(frame.columns):
        plate = str(plate) if "plate" in frame.columns else _WHOLE
        well = str(well) if "well" in frame.columns else _WHOLE
    schema = None
    if not len(frame.columns):
        schema = _empty_schema(project, name)
        if schema is None:
            return row_count(db_path, table) or 0
    os.makedirs(project, exist_ok=True)
    parts = {(plate, well): frame}
    if schema is None and _WHOLE not in (plate, well):
        for where in (_stored(project, name) or {}):
            if _WHOLE in where:
                parts[where] = frame.iloc[:0]
    if schema is not None:
        _write_part(project, table, frame, plate, well, schema=schema)
        store.register(project)
    else:
        _write_parts(project, table, parts)
    return row_count(db_path, table) or 0


def _object_key(table: str, columns) -> Tuple[str, ...]:
    """The columns that name one object in ``table``, or ``()`` for none.

    372 PART 14-L, V11c: the storage contract says ``UNIQUE(plate, well,
    object_id)``, and appending a well twice must not double it without an
    error. ``ops_objects`` and ``ops_barcodes`` hold one row per object;
    ``ops_reads`` holds many and ``ops_geometry`` none. The key is
    ``object_id`` together with whichever of ``plate`` and ``well`` the
    frame carries, because ids are numbered per well and two wells
    legitimately share them.

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


_ONE_ROW_PER_OBJECT = ("ops_objects", "ops_barcodes", "ops_merge")


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


def _refuse_appended_keys(table: str, existing, frame,
                          key: Sequence[str]) -> None:
    """Refuse an append whose objects the well already holds.

    :raises StoreError: naming how many objects the append shared with the
        table, and up to five of them; nothing is appended.
    """
    columns = [c for c in key if c in existing.columns]
    if not columns or not len(existing):
        return
    old = set(existing[columns].astype(str).itertuples(index=False,
                                                       name=None))
    new = frame[columns]
    hit = [tuple(map(str, row)) in old
           for row in new.itertuples(index=False, name=None)]
    shared = new[hit]
    if not len(shared):
        return
    raise StoreError(
        f"{table} already holds {len(shared)} of the {len(frame)} object(s) "
        f"being appended, e.g. "
        f"{_named(columns, shared.head(5).itertuples(index=False, name=None))}"
        f". One object would have two rows, so nothing was appended and the "
        f"table is as it was. Write each well once, or replace it.")


def _as_read(frame, table: str):
    """``frame`` as the OPS readers were written to receive ``table``.

    ``plate`` and ``well`` first (the store keeps them as partition
    columns at the end) and dropped where the table has none; for the
    tables outside :data:`CACHED_TABLES`, integers and booleans as int64 and
    floats as float64, the storage classes their SQLite readers received.
    The stored dtypes stay the compact ones of the dtype map.
    """
    import numpy as np

    for name in ("plate", "well"):
        if name in frame.columns and len(frame) and bool(
                (frame[name].astype(str) == _WHOLE).all()):
            frame = frame.drop(columns=[name])
    front = [c for c in ("plate", "well") if c in frame.columns]
    if front and list(frame.columns[:len(front)]) != front:
        frame = frame[front + [c for c in frame.columns if c not in front]]
    if table in CACHED_TABLES:
        return frame
    casts = {}
    for name, dtype in frame.dtypes.items():
        if dtype == bool or (np.issubdtype(dtype, np.integer)
                             and dtype != np.int64):
            casts[name] = np.int64
        elif np.issubdtype(dtype, np.floating) and dtype != np.float64:
            casts[name] = np.float64
    return frame.astype(casts) if casts else frame


def _where(wells: Optional[Sequence[Tuple[str, str]]]):
    """The SQL filter and parameters for ``(plate, well)`` pairs, or ``("", [])``."""
    if wells is None:
        return "", []
    pairs = list(dict.fromkeys((str(p), str(w)) for p, w in wells))
    if not pairs:
        return " WHERE FALSE", []
    clause = " OR ".join("(CAST(plate AS VARCHAR) = ? AND CAST(well AS "
                         "VARCHAR) = ?)" for _ in pairs)
    return f" WHERE {clause}", [v for pair in pairs for v in pair]


def _read(project: str, table: str,
          wells: Optional[Sequence[Tuple[str, str]]] = None):
    """Rows of ``table`` in the store at ``project``, for ``wells`` or all."""
    from . import store
    from .store import _ident

    clause, params = _where(wells)
    with store.open(project) as opened:
        return opened.sql(f"SELECT * FROM {_ident(_store_name(table))}"
                          f"{clause}", params or None)


def _read_rows(db_path: str, table: str, plate: Optional[str] = None,
               wells: Optional[Sequence[str]] = None):
    """Rows of one OPS table, for ``plate`` and ``wells`` or all; None if absent.

    The run's store when it holds the table, else an older run's SQLite
    file at ``db_path``; both through :func:`spacr.store.open`.
    """
    _checked(table)
    project = _store_project(db_path)
    present = _stored(project, _store_name(table))
    if present is not None:
        chosen = None
        if plate is not None:
            asked = {str(w) for w in (wells or ())}
            chosen = [(p, w) for p, w in present if p == str(plate)
                      and (wells is None or w in asked)]
        frame = _read(project, table, chosen)
        for position, name in enumerate(("plate", "well")):
            if name in frame.columns and all(key[position] == _WHOLE
                                             for key in present):
                frame = frame.drop(columns=[name])
        return _as_read(frame, table).reset_index(drop=True)
    legacy = _legacy(db_path)
    if legacy is None or table not in _legacy_tables(legacy):
        return None
    from . import store
    from .store import _ident

    with store.open(legacy) as opened:
        sql = f"SELECT * FROM {_ident(table)}"
        clause, params = "", []
        if plate is not None:
            names = set(opened.sql(f"{sql} LIMIT 0").columns)
            if wells is not None and {"plate", "well"} <= names:
                clause, params = _where([(plate, w) for w in wells])
            elif "plate" in names:
                clause, params = " WHERE CAST(plate AS VARCHAR) = ?", [
                    str(plate)]
        frame = opened.sql(sql + clause, params or None)
    return _as_read(frame, table).reset_index(drop=True)


def read_table(db_path: str, table: str, *, prefer_cache: bool = True):
    """Read one OPS table whole, from the run's spaCR store.

    An older run whose tables are in the SQLite file at ``db_path`` is read
    from it, through the same reader.

    :param db_path: the run's OPS database path.
    :param table: which of :data:`OPS_TABLES` to read.
    :param prefer_cache: kept for older callers; there is no cache to
        prefer any more.
    :raises StoreError: when neither holds the table.
    """
    frame = _read_rows(db_path, table)
    if frame is None:
        raise StoreError(f"{table} is not in {_store_project(db_path)} or "
                         f"{_resolved(db_path)}")
    return frame


def row_count(db_path: str, table: str) -> Optional[int]:
    """Rows in one OPS table, or ``None`` when the table is not there.

    :param db_path: the run's OPS database path.
    :param table: the table to count.
    """
    if table not in OPS_TABLES:
        return None
    present = _stored(_store_project(db_path), _store_name(table))
    if present is not None:
        return int(sum(present.values()))
    legacy = _legacy(db_path)
    if legacy is None:
        return None
    from . import store
    from .store import _ident

    try:
        with store.open(legacy) as opened:
            if table not in opened.tables():
                return None
            return int(opened.sql(f"SELECT COUNT(*) AS n FROM "
                                  f"{_ident(table)}")["n"].iloc[0])
    except Exception:                                  # noqa: BLE001
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
    """Check whether ``ops_objects`` is ready so that channels may be read.

    The gate is one sentence -- "no sequencing or phenotype
    channel is read until `ops_objects` is on disk and its row count
    checked" -- and both halves matter. On disk without a count check would
    pass an empty table, and an empty ops_objects means every later phase
    silently produces nothing while appearing to run.

    It returns a reason, not just a verdict. "Not yet" leaves the operator
    guessing which of three things went wrong: no store, no table, or a
    table with nothing in it. Those have different fixes.

    :param minimum: the fewest objects a real well can have. One is the
        lowest meaningful value -- a well with a single nucleus is a bad well, not an
        impossible one -- so this exists to be raised by a caller who knows
        their plate, not to encode a guess here.
    :param db_path: the run's OPS database path. Its absence is one of the
        three answers, so this is not required to exist.
    """
    if not _exists(db_path):
        return Readiness(False, None,
                         f"there is no database at {_resolved(db_path)} and "
                         f"no store at {_store_project(db_path)}. The object "
                         f"pass writes it; run that before reading "
                         f"channels.")
    rows = row_count(db_path, OBJECTS_TABLE)
    if rows is None:
        return Readiness(False, None,
                         f"{_store_project(db_path)} has no {OBJECTS_TABLE} "
                         f"table. The numbering pass assigns the object ids "
                         f"and the storage step writes them; neither has run "
                         f"for this well.")
    if rows < minimum:
        return Readiness(False, rows,
                         f"{OBJECTS_TABLE} has {rows} row(s), below the {minimum} "
                         f"required. Reading channels against an empty object "
                         f"list produces empty results that look like a "
                         f"finished run.")
    return Readiness(True, rows)
