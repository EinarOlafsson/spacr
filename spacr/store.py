"""The spaCR store: Parquet tables, Zarr crops and one DuckDB catalog.

A project folder holds one table per object class under
``tables/<class>/plate=<P>/well=<W>/part-<N>.parquet``, one crop store per
well under ``crops/plate=<P>/well=<W>.zarr`` and the catalog
``spacr.duckdb``, which registers every file with its row count and sha256
and defines one view per table over the registered files only.

Each well writes its own files with :func:`write_part` (a temporary name, an
atomic rename, then a manifest), so wells write in parallel without a lock.
:func:`register` adds every waiting manifest to the catalog in one
transaction. :func:`open` reads the registry and answers queries through an
in-memory DuckDB, so a reader never holds the catalog. A folder holding only
an older ``measurements.db`` opens through the same API.
"""
from __future__ import annotations

import builtins
import datetime
import glob
import hashlib
import json
import os
import time
import urllib.parse
import uuid
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["StoreError", "Store", "open", "write_part", "register",
           "cleanup", "CATALOG", "SCHEMA_VERSION"]

#: The catalog's file name at the project root.
CATALOG = "spacr.duckdb"

#: The layout version written into every registry row.
SCHEMA_VERSION = 1

#: Rows per Parquet row group, before the byte cap.
_ROW_GROUP_ROWS = 131072

#: The most uncompressed bytes one row group may hold.
_ROW_GROUP_BYTES = 256 * 2 ** 20

#: Cells per Zarr chunk.
_CROP_CHUNK = 256

_POINTER = ("crop_store", "crop_index")
_PARTITION = ("plate", "well")
_PENDING = "_pending"
_KEYS = "__keys"
_LOCATORS = ("_part", "_row_group", "_group_row")

#: The most rows a selection may hold to be read row group by row group.
_LOOKUP_ROWS = 20_000
_LEGACY_NAMES = ("measurements.db", os.path.join("measurements",
                                                 "measurements.db"))


class StoreError(ValueError):
    """A store that cannot be read or written as asked; the message says why."""


def _require(name: str):
    """Import ``name`` or refuse with the install hint."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as missing:
        raise StoreError(
            f"the spaCR store needs {name}, which is part of spaCR's core "
            f"install on Python 3.10 and later; install it with "
            f"'pip install {name}'") from missing


def _ident(name: str) -> str:
    """A SQL identifier in double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _literal(text: str) -> str:
    """A SQL string literal."""
    return "'" + str(text).replace("'", "''") + "'"


def _encoded(value: Any) -> str:
    """A partition value as it appears in a folder name."""
    return urllib.parse.quote(str(value), safe="")


def _now() -> str:
    """The current UTC time in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _sha256(path: str) -> str:
    """The sha256 of one file."""
    digest = hashlib.sha256()
    with builtins.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: str) -> Dict[str, Any]:
    """``sha256``, ``bytes`` and ``mtime_ns`` of a file or a Zarr folder.

    A folder's hash runs over every file's relative path and sha256 in
    sorted order; its size is the sum and its mtime the newest.
    """
    if os.path.isfile(path):
        stat = os.stat(path)
        return {"sha256": _sha256(path), "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns)}
    digest = hashlib.sha256()
    total, newest = 0, 0
    for name in sorted(_folder_files(path)):
        full = os.path.join(path, name)
        stat = os.stat(full)
        total += int(stat.st_size)
        newest = max(newest, int(stat.st_mtime_ns))
        digest.update(name.encode("utf-8") + b"\0" + _sha256(full).encode())
    return {"sha256": digest.hexdigest(), "bytes": total, "mtime_ns": newest}


def _folder_files(path: str) -> List[str]:
    """Every file under ``path``, relative, with ``/`` separators."""
    found = []
    for folder, _dirs, files in os.walk(path):
        for name in files:
            rel = os.path.relpath(os.path.join(folder, name), path)
            found.append(rel.replace(os.sep, "/"))
    return found


def _stat(path: str) -> Optional[Dict[str, int]]:
    """``bytes`` and ``mtime_ns`` of a file or folder, or None when absent."""
    if os.path.isfile(path):
        stat = os.stat(path)
        return {"bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if not os.path.isdir(path):
        return None
    total, newest = 0, 0
    for name in _folder_files(path):
        stat = os.stat(os.path.join(path, name))
        total += int(stat.st_size)
        newest = max(newest, int(stat.st_mtime_ns))
    return {"bytes": total, "mtime_ns": newest}



def _fsync(path: str) -> None:
    """Flush one file to disk; opened writable, as Windows requires."""
    with builtins.open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _next_name(folder: str, stem: str, suffix: str) -> str:
    """The first ``<stem><N><suffix>`` not yet in ``folder``."""
    used = set()
    for path in glob.glob(os.path.join(glob.escape(folder),
                                       f"{stem}*{suffix}")):
        middle = os.path.basename(path)[len(stem):-len(suffix)]
        if middle.isdigit():
            used.add(int(middle))
    number = 0
    while number in used:
        number += 1
    return f"{stem}{number}{suffix}"


def _row_group_rows(schema, requested: Optional[int]) -> int:
    """Rows per row group: ``requested`` or the default, under the byte cap."""
    rows = int(requested or _ROW_GROUP_ROWS)
    width = 0
    for field in schema:
        try:
            width += max(1, field.type.bit_width // 8)
        except ValueError:
            width += 16
    return max(1024, min(rows, _ROW_GROUP_BYTES // max(1, width)))


def _arrow_table(frame):
    """``frame`` as a pyarrow Table without its index."""
    pa = _require("pyarrow")
    if isinstance(frame, pa.Table):
        return frame
    return pa.Table.from_pandas(frame, preserve_index=False)


def _sorted_order(table, sort_by: Sequence[str]) -> np.ndarray:
    """The stable order of ``table`` by ``sort_by``, nulls last."""
    pa = _require("pyarrow")
    pc = _require("pyarrow.compute")
    names = [name for name in sort_by if name in table.column_names]
    if not names:
        return np.arange(table.num_rows)
    if int(str(pa.__version__).split(".")[0]) >= 25:
        return np.asarray(pc.sort_indices(table, sort_keys=[
            (name, "ascending", "at_end") for name in names]))
    return np.asarray(pc.sort_indices(
        table, sort_keys=[(name, "ascending") for name in names],
        null_placement="at_end"))


def _create_crop_array(path: str, shape, dtype, name: str = "crops"):
    """A Zarr format 2 array under ``path``/``name``, through zarr 2 or 3."""
    zarr = _require("zarr")
    numcodecs = _require("numcodecs")
    chunks = (min(_CROP_CHUNK, max(1, shape[0])),) + tuple(shape[1:])
    codec = numcodecs.Zstd(level=3)
    target = os.path.join(path, name)
    if int(str(zarr.__version__).split(".")[0]) >= 3:
        return zarr.create_array(store=target, shape=shape, chunks=chunks,
                                 dtype=dtype, compressors=codec,
                                 zarr_format=2, fill_value=0)
    return zarr.open_array(target, mode="w", shape=shape, chunks=chunks,
                           dtype=dtype, compressor=codec, fill_value=0)


def _write_crops(path: str, crops, order: np.ndarray, masks=None) -> None:
    """Write ``crops[order]`` (and ``masks[order]``) chunk by chunk."""
    os.makedirs(path, exist_ok=True)
    shape = (len(order),) + tuple(crops.shape[1:])
    array = _create_crop_array(path, shape, np.dtype(crops.dtype))
    mask_array = (_create_crop_array(path, (len(order),) + tuple(
        masks.shape[1:]), np.dtype(masks.dtype), "masks")
        if masks is not None else None)
    for start in range(0, len(order), _CROP_CHUNK):
        chosen = order[start:start + _CROP_CHUNK]
        array[start:start + len(chosen)] = np.asarray(crops[chosen])
        if mask_array is not None:
            mask_array[start:start + len(chosen)] = np.asarray(masks[chosen])
    with builtins.open(os.path.join(path, ".zgroup"), "w",
                       encoding="utf-8") as handle:
        json.dump({"zarr_format": 2}, handle)


def write_part(project, table: str, frame, *, plate, well,
               sort_by: Sequence[str] = ("gene", "cell_id"),
               key_columns: Sequence[str] = (), crops=None,
               crop_rows=None, masks=None, run_id: Optional[str] = None,
               row_group_rows: Optional[int] = None) -> Dict[str, Any]:
    """Write one well's rows of ``table`` (and its crops) and leave a manifest.

    The rows are sorted by ``sort_by`` and written under a temporary name,
    flushed and renamed into ``tables/<table>/plate=<P>/well=<W>/``. The
    crops go to ``crops/plate=<P>/well=<W>.zarr`` in the same row order and
    each row gets ``crop_store`` and ``crop_index``. The manifest in
    ``_pending/`` is written last; nothing is visible to readers until
    :func:`register` adds it to the catalog.

    :param project: the project folder.
    :param table: the object class, for example ``cells``.
    :param frame: a DataFrame or Arrow table, one row per object.
    :param plate: the plate name.
    :param well: the well name.
    :param sort_by: the columns the rows are sorted by, those present.
    :param key_columns: the columns every feature query returns.
    :param crops: an array ``(k, C, H, W)`` indexable by an integer array.
    :param crop_rows: the frame row of each crop; every row when None.
    :param masks: an array ``(k, H, W)`` beside the crops, or None.
    :param run_id: the run this write belongs to.
    :param row_group_rows: rows per row group, under the byte cap.
    :returns: the manifest.
    :raises StoreError: when the frame's plate or well disagree with the
        partition, or the crops do not match ``crop_rows``.
    """
    pa = _require("pyarrow")
    pq = _require("pyarrow.parquet")
    project = os.path.abspath(os.fspath(project))
    data = _arrow_table(frame)
    for name, value in (("plate", plate), ("well", well)):
        if name in data.column_names:
            found = set(map(str, data.column(name).to_pylist()))
            if found - {str(value)}:
                raise StoreError(
                    f"{table}: column {name} holds {sorted(found)[:3]}, not "
                    f"only {value!r}; one part holds one {name}")
            data = data.drop_columns([name])
    data = data.drop_columns([c for c in _POINTER if c in data.column_names])
    order = _sorted_order(data, sort_by)
    data = data.take(pa.array(order))
    uid = uuid.uuid4().hex
    files: List[Dict[str, Any]] = []
    crop_store = None
    if crops is not None:
        rows = (np.arange(len(order)) if crop_rows is None
                else np.asarray(crop_rows, dtype=np.int64))
        if len(rows) != int(crops.shape[0]):
            raise StoreError(
                f"{table}: {crops.shape[0]} crops for {len(rows)} crop rows")
        crop_of = np.full(len(order), -1, dtype=np.int64)
        crop_of[rows] = np.arange(len(rows))
        sorted_crop = crop_of[order]
        has = sorted_crop >= 0
        crop_order = sorted_crop[has]
        crop_dir = os.path.join(project, "crops", f"plate={_encoded(plate)}")
        os.makedirs(crop_dir, exist_ok=True)
        stem = f"well={_encoded(well)}"
        name = (f"{stem}.zarr" if not os.path.exists(
            os.path.join(crop_dir, f"{stem}.zarr"))
            else _next_name(crop_dir, f"{stem}.v", ".zarr"))
        final = os.path.join(crop_dir, name)
        temporary = f"{final}.tmp-{uid}"
        _write_crops(temporary, crops, crop_order, masks)
        os.replace(temporary, final)
        crop_store = os.path.relpath(final, project).replace(os.sep, "/")
        index = np.full(len(order), -1, dtype=np.int64)
        index[has] = np.arange(int(has.sum()))
        data = data.append_column("crop_store", pa.array(
            [crop_store if h else None for h in has], pa.string()))
        data = data.append_column("crop_index", pa.array(
            index, pa.int64(), mask=~has))
        files.append({"path": crop_store, "kind": "zarr",
                      "rows": int(has.sum()), **_fingerprint(final)})
    data = data.append_column("plate", pa.array(
        [str(plate)] * data.num_rows, pa.string()))
    data = data.append_column("well", pa.array(
        [str(well)] * data.num_rows, pa.string()))
    folder = os.path.join(project, "tables", table, f"plate={_encoded(plate)}",
                          f"well={_encoded(well)}")
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, _next_name(folder, "part-", ".parquet"))
    temporary = f"{final}.tmp-{uid}"
    floats = [f.name for f in data.schema if pa.types.is_floating(f.type)]
    others = [n for n in data.column_names if n not in set(floats)]
    group_rows = _row_group_rows(data.schema, row_group_rows)
    pq.write_table(data, temporary, compression="zstd",
                   row_group_size=group_rows,
                   write_statistics=True, use_dictionary=others or False,
                   use_byte_stream_split=floats or False)
    _fsync(temporary)
    os.replace(temporary, final)
    rel = os.path.relpath(final, project).replace(os.sep, "/")
    files.insert(0, {"path": rel, "kind": "parquet",
                     "rows": int(data.num_rows), **_fingerprint(final)})
    files.insert(1, _write_keys(project, table, data, rel, group_rows,
                                key_columns, plate, well, uid))
    manifest = {"uuid": uid, "table": table, "plate": str(plate),
                "well": str(well), "sort_by": list(sort_by),
                "key_columns": list(key_columns), "files": files,
                "schema_version": SCHEMA_VERSION, "written_at": _now(),
                "run_id": run_id}
    pending = os.path.join(project, _PENDING)
    os.makedirs(pending, exist_ok=True)
    path = os.path.join(pending, f"{uid}.json")
    with builtins.open(f"{path}.tmp", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(f"{path}.tmp", path)
    return manifest


def _keys_name(table: str) -> str:
    """The name of ``table``'s narrow key table."""
    return f"{table}{_KEYS}"


def _write_keys(project: str, table: str, data, part: str, group_rows: int,
                key_columns: Sequence[str], plate, well, uid: str):
    """Write the narrow key table of one part; returns its manifest entry.

    It holds every column that is not a float feature (and every key
    column), the crop pointer and, per row, where the row sits in the full
    part: ``_part``, ``_row_group`` and ``_group_row``. Key-only queries,
    crop batches and point lookups read it instead of the wide part.
    """
    pa = _require("pyarrow")
    pq = _require("pyarrow.parquet")
    keep = set(key_columns) | set(_POINTER) | set(_PARTITION)
    names = [f.name for f in data.schema if f.name in keep
             or not pa.types.is_floating(f.type)]
    rows = np.arange(data.num_rows, dtype=np.int64)
    keys = data.select(names)
    keys = keys.append_column("_part", pa.array([part] * data.num_rows,
                                                pa.string()))
    keys = keys.append_column("_row_group", pa.array(rows // group_rows,
                                                     pa.int32()))
    keys = keys.append_column("_group_row", pa.array(rows % group_rows,
                                                     pa.int32()))
    folder = os.path.join(project, "tables", _keys_name(table),
                          f"plate={_encoded(plate)}", f"well={_encoded(well)}")
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, os.path.basename(part))
    temporary = f"{final}.tmp-{uid}"
    pq.write_table(keys, temporary, compression="zstd",
                   row_group_size=_ROW_GROUP_ROWS, write_statistics=True)
    _fsync(temporary)
    os.replace(temporary, final)
    return {"path": os.path.relpath(final, project).replace(os.sep, "/"),
            "kind": "parquet", "table": _keys_name(table),
            "rows": int(keys.num_rows), **_fingerprint(final)}


_REGISTRY_DDL = (
    "CREATE TABLE IF NOT EXISTS spacr_schema (version INTEGER)",
    "CREATE TABLE IF NOT EXISTS spacr_tables (name VARCHAR PRIMARY KEY, "
    "sort_by VARCHAR, key_columns VARCHAR, created_at VARCHAR)",
    "CREATE TABLE IF NOT EXISTS spacr_files (path VARCHAR PRIMARY KEY, "
    "table_name VARCHAR, kind VARCHAR, plate VARCHAR, well VARCHAR, "
    "rows BIGINT, schema_version INTEGER, sha256 VARCHAR, bytes BIGINT, "
    "mtime_ns BIGINT, written_at VARCHAR, run_id VARCHAR)",
    "CREATE TABLE IF NOT EXISTS spacr_runs (run_id VARCHAR PRIMARY KEY, "
    "started_at VARCHAR, spacr_version VARCHAR, settings VARCHAR)",
)


def _source(paths: Sequence[str]) -> str:
    """The ``read_parquet`` expression over ``paths``."""
    listed = ", ".join(_literal(p) for p in paths)
    return (f"read_parquet([{listed}], hive_partitioning = true, "
            f"hive_types = {{'plate': VARCHAR, 'well': VARCHAR}}, "
            f"union_by_name = true)")


def _connect(path: Optional[str] = None, *, read_only: bool = False,
             threads: Optional[int] = None):
    """A DuckDB connection, in memory when ``path`` is None."""
    duckdb = _require("duckdb")
    conn = (duckdb.connect(path, read_only=read_only) if path
            else duckdb.connect())
    conn.execute("SET enable_progress_bar = false")
    if not path:
        conn.execute("SET parquet_metadata_cache = true")
    if threads:
        conn.execute(f"SET threads = {int(threads)}")
    return conn


def _rebuild_views(conn, project: str, tables: Sequence[str]) -> None:
    """Point each table's catalog view at its registered Parquet files."""
    for name in tables:
        paths = [os.path.join(project, row[0]) for row in conn.execute(
            "SELECT path FROM spacr_files WHERE table_name = ? AND kind = "
            "'parquet' ORDER BY plate, well, path", [name]).fetchall()]
        if paths:
            conn.execute(f"CREATE OR REPLACE VIEW {_ident(name)} AS "
                         f"SELECT * FROM {_source(paths)}")
        else:
            conn.execute(f"DROP VIEW IF EXISTS {_ident(name)}")


def register(project, *, run: Optional[Mapping[str, Any]] = None,
             accept_changes: bool = False) -> Dict[str, Any]:
    """Register every waiting manifest in one transaction.

    Each manifest's files are checked against their recorded size and
    mtime first. The well's earlier files of the same table are replaced in
    the registry, the views are rebuilt and, after the commit, the replaced
    files and the manifests are deleted.

    :param project: the project folder.
    :param run: a ``spacr_runs`` row (``run_id``, ``settings``), or None.
    :param accept_changes: rehash files whose size or mtime changed after
        the manifest was written and register their new hash.
    :returns: ``{"registered", "superseded", "tables"}``.
    :raises StoreError: when a manifest's file is missing or changed.
    """
    project = os.path.abspath(os.fspath(project))
    pending = sorted(glob.glob(os.path.join(glob.escape(project), _PENDING,
                                            "*.json")),
                     key=os.path.getmtime)
    manifests = []
    for path in pending:
        with builtins.open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for entry in manifest["files"]:
            found = _stat(os.path.join(project, entry["path"]))
            if found is None:
                raise StoreError(f"{entry['path']} of manifest {path} is "
                                 f"missing; write the well again")
            if (found["bytes"], found["mtime_ns"]) != (entry["bytes"],
                                                       entry["mtime_ns"]):
                if not accept_changes:
                    raise StoreError(
                        f"{entry['path']} changed after it was written; "
                        f"write the well again, or register with "
                        f"accept_changes=True to adopt the new bytes")
                entry.update(_fingerprint(os.path.join(project,
                                                       entry["path"])))
        manifests.append((path, manifest))
    superseded: List[str] = []
    tables = sorted({e.get("table", m["table"]) for _p, m in manifests
                     for e in m["files"]} | {m["table"] for _p, m in manifests})
    conn = _connect(os.path.join(project, CATALOG))
    try:
        conn.execute("BEGIN TRANSACTION")
        for ddl in _REGISTRY_DDL:
            conn.execute(ddl)
        if not conn.execute("SELECT COUNT(*) FROM spacr_schema").fetchone()[0]:
            conn.execute("INSERT INTO spacr_schema VALUES (?)",
                         [SCHEMA_VERSION])
        if run:
            from . import __version__ as version

            conn.execute(
                "INSERT OR REPLACE INTO spacr_runs VALUES (?, ?, ?, ?)",
                [str(run.get("run_id")), str(run.get("started_at") or _now()),
                 str(version), json.dumps(run.get("settings") or {},
                                          default=str, sort_keys=True)])
        for _path, manifest in manifests:
            conn.execute(
                "INSERT OR IGNORE INTO spacr_tables VALUES (?, ?, ?, ?)",
                [manifest["table"], json.dumps(manifest["sort_by"]),
                 json.dumps(manifest["key_columns"]), _now()])
            new = {entry["path"] for entry in manifest["files"]}
            names = sorted({manifest["table"], _keys_name(manifest["table"])})
            for name in names:
                for (old,) in conn.execute(
                        "SELECT path FROM spacr_files WHERE table_name = ? "
                        "AND plate = ? AND well = ?",
                        [name, manifest["plate"],
                         manifest["well"]]).fetchall():
                    if old not in new:
                        superseded.append(old)
                conn.execute("DELETE FROM spacr_files WHERE table_name = ? "
                             "AND plate = ? AND well = ?",
                             [name, manifest["plate"], manifest["well"]])
            for entry in manifest["files"]:
                conn.execute(
                    "INSERT OR REPLACE INTO spacr_files VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [entry["path"], entry.get("table", manifest["table"]),
                     entry["kind"],
                     manifest["plate"], manifest["well"], entry["rows"],
                     manifest["schema_version"], entry["sha256"],
                     entry["bytes"], entry["mtime_ns"],
                     manifest["written_at"], manifest.get("run_id")])
        _rebuild_views(conn, project, tables)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    still = _registered_paths(project)
    for rel in superseded:
        if rel not in still:
            _remove(os.path.join(project, rel))
    for path, _manifest in manifests:
        _remove(path)
    return {"registered": len(manifests), "superseded": len(superseded),
            "tables": tables}


def _registered_paths(project: str) -> set:
    """Every path the catalog registers."""
    catalog = os.path.join(project, CATALOG)
    if not os.path.isfile(catalog):
        return set()
    conn = _connect(catalog, read_only=True)
    try:
        return {row[0] for row in conn.execute(
            "SELECT path FROM spacr_files").fetchall()}
    finally:
        conn.close()


def _remove(path: str) -> None:
    """Delete a file or folder; a missing one is fine."""
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def cleanup(project, *, older_than: float = 3600.0) -> List[str]:
    """Remove temporary files older than ``older_than`` seconds.

    These are what a writer killed mid-write leaves behind. Registered files
    and waiting manifests are never touched.

    :param project: the project folder.
    :param older_than: the age in seconds below which a file is kept,
        since a writer may still be busy with it.
    :returns: the paths removed, relative to the project.
    """
    project = os.path.abspath(os.fspath(project))
    cutoff = time.time() - float(older_than)
    removed = []
    for folder, dirs, files in os.walk(project):
        for name in list(dirs) + files:
            if ".tmp-" not in name and not name.endswith(".json.tmp"):
                continue
            path = os.path.join(folder, name)
            if os.path.getmtime(path) <= cutoff:
                _remove(path)
                removed.append(os.path.relpath(path, project))
                if name in dirs:
                    dirs.remove(name)
    return removed


def _arrow_of(conn, sql: str, params=None):
    """The result of ``sql`` as an Arrow table, on DuckDB 1 or 2."""
    relation = conn.sql(sql, params=list(params)) if params else conn.sql(sql)
    for name in ("to_arrow_table", "fetch_arrow_table", "arrow"):
        method = getattr(relation, name, None)
        if method is not None:
            return method()
    raise StoreError("this DuckDB has no Arrow output")


def _reader_of(conn, sql: str, params, rows: int):
    """The result of ``sql`` as an Arrow record batch reader."""
    relation = conn.sql(sql, params=list(params)) if params else conn.sql(sql)
    for name in ("to_arrow_reader", "fetch_arrow_reader", "record_batch"):
        method = getattr(relation, name, None)
        if method is not None:
            return method(rows)
    raise StoreError("this DuckDB has no Arrow batch reader")


class Store:
    """An open spaCR store, or an older SQLite project, made by :func:`open`.

    :param project: the project folder, or a SQLite file.
    :param threads: DuckDB threads; None leaves DuckDB's default.
    :param verify: as for :meth:`refresh`.
    """

    def __init__(self, project: str, *, threads: Optional[int] = None,
                 verify: str = "quick"):
        """Read the registry of ``project``; see :func:`open`."""
        self.project = os.path.abspath(os.fspath(project))
        self.threads = threads
        self._conn = None
        self._files: List[Dict[str, Any]] = []
        self._tables: Dict[str, Dict[str, Any]] = {}
        self._arrays: Dict[str, Any] = {}
        self._views: List[str] = []
        self._described: Dict[str, List[str]] = {}
        self._parts: Dict[str, Any] = {}
        self.legacy: Optional[str] = None
        self.refresh(verify=verify)

    def refresh(self, *, verify: str = "quick") -> "Store":
        """Read the registry again and rebuild the views.

        :param verify: ``"quick"`` compares sizes and mtimes, ``"full"``
            rehashes every file, ``"none"`` checks nothing.
        :returns: the store.
        """
        self.close()
        self._arrays = {}
        self._conn = _connect(threads=self.threads)
        catalog = os.path.join(self.project, CATALOG)
        if os.path.isfile(self.project) or not os.path.isfile(catalog):
            self._attach_legacy()
            return self
        source = _connect(catalog, read_only=True)
        try:
            columns = [c[0] for c in source.execute(
                "SELECT * FROM spacr_files LIMIT 0").description]
            self._files = [dict(zip(columns, row)) for row in source.execute(
                "SELECT * FROM spacr_files ORDER BY plate, well, path"
            ).fetchall()]
            self._tables = {row[0]: {"sort_by": json.loads(row[1]),
                                     "key_columns": json.loads(row[2])}
                            for row in source.execute(
                                "SELECT name, sort_by, key_columns FROM "
                                "spacr_tables").fetchall()}
        finally:
            source.close()
        if verify != "none":
            self.verify(full=verify == "full")
        self._views = sorted({f["table_name"] for f in self._files
                              if f["kind"] == "parquet"})
        self._described = {}
        self._parts = {}
        for name in self._views:
            paths = [os.path.join(self.project, f["path"]) for f in self._files
                     if f["table_name"] == name and f["kind"] == "parquet"]
            if paths:
                self._conn.execute(f"CREATE OR REPLACE VIEW {_ident(name)} AS "
                                   f"SELECT * FROM {_source(paths)}")
        return self

    def _attach_legacy(self) -> None:
        """Open an older ``measurements.db`` through DuckDB's sqlite reader."""
        path = self.project
        if not os.path.isfile(path):
            path = next((os.path.join(self.project, n) for n in _LEGACY_NAMES
                         if os.path.isfile(os.path.join(self.project, n))),
                        None)
        if path is None:
            raise StoreError(
                f"{self.project} has neither {CATALOG} nor measurements.db")
        self.legacy = path
        names = _sqlite_tables(path)
        try:
            for statement in ("LOAD sqlite", "INSTALL sqlite"):
                try:
                    self._conn.execute(statement)
                    break
                except Exception:
                    continue
            self._conn.execute("LOAD sqlite")
            self._conn.execute(f"ATTACH {_literal(path)} AS legacy "
                               f"(TYPE sqlite, READ_ONLY)")
            for name in names:
                self._conn.execute(
                    f"CREATE VIEW {_ident(name)} AS SELECT * FROM "
                    f"legacy.{_ident(name)}")
        except Exception:
            for name in names:
                self._conn.register(name, _sqlite_arrow(path, name))
        self._tables = {name: {"sort_by": [], "key_columns": []}
                        for name in names}

    def tables(self) -> List[str]:
        """The tables this store answers queries on."""
        return sorted(self._tables)

    def verify(self, *, full: bool = False) -> int:
        """Check every registered file against the registry.

        :param full: rehash every file; otherwise compare sizes and mtimes.
        :returns: how many files were checked.
        :raises StoreError: naming each file that is missing or changed.
        """
        changed = []
        for entry in self._files:
            path = os.path.join(self.project, entry["path"])
            found = _stat(path)
            if found is None:
                changed.append(f"{entry['path']} is missing")
                continue
            if full:
                digest = _fingerprint(path)["sha256"]
                if digest != entry["sha256"]:
                    changed.append(f"{entry['path']}: sha256 {digest[:12]} "
                                   f"!= registered {entry['sha256'][:12]}")
            elif (found["bytes"], found["mtime_ns"]) != (entry["bytes"],
                                                         entry["mtime_ns"]):
                changed.append(f"{entry['path']}: size or mtime changed")
        if changed:
            raise StoreError(
                "files changed outside spaCR: " + "; ".join(changed[:10]) +
                ". Write those wells again, or register them with "
                "accept_changes=True to adopt the new bytes.")
        return len(self._files)

    def _columns(self, table: str) -> List[str]:
        """The columns of ``table``, read once."""
        if table not in self._described:
            self._described[table] = [row[0] for row in self._conn.execute(
                f"DESCRIBE SELECT * FROM {_ident(table)}").fetchall()]
        return self._described[table]

    def _wanted(self, table: str, source: str, features) -> List[str]:
        """The key columns, the crop pointer and ``features``, as present."""
        present = self._columns(source)
        keys = list(self._tables[table].get("key_columns") or [])
        wanted = [c for c in keys + ["plate", "well", "cell_id", *_POINTER]
                  if c in present]
        missing = [f for f in features if f not in present]
        if missing:
            raise StoreError(f"{table} has no column {missing[:5]}")
        return wanted + [f for f in features if f not in wanted]

    def _routed(self, where, table: str, features):
        """A query answered from the narrow key table, or None.

        Key-only selections (``features`` a list of key columns, or empty)
        are read from the key table alone. Other selections of at most
        :data:`_LOOKUP_ROWS` rows (or 2 % of the table) are located there
        and only their row groups of the wide parts are read. Anything else,
        or a filter on a column the key table lacks, returns None and runs
        on the wide view.
        """
        keys = _keys_name(table)
        if (self.legacy or keys not in self._views or table not in self._tables
                or (isinstance(where, str) and where.lstrip().lower(
                ).startswith(("select", "with")))):
            return None
        duckdb = _require("duckdb")
        key_columns = set(self._columns(keys)) - set(_LOCATORS)
        if features is not None and set(features) <= key_columns:
            sql, params = self._select(where, table, features, source=keys)
            try:
                return _arrow_of(self._conn, sql, params)
            except duckdb.Error:
                return None
        sql, params = self._select(where, table, list(_LOCATORS), source=keys)
        sql = sql.replace("SELECT ", "SELECT " + ", ".join(
            _ident(c) for c in _LOCATORS) + " FROM (SELECT ", 1) + ")"
        total = sum(int(f["rows"]) for f in self._files
                    if f["table_name"] == keys)
        limit = max(_LOOKUP_ROWS, int(total * 0.02))
        try:
            located = _arrow_of(self._conn, sql + f" LIMIT {limit + 1}", params)
        except duckdb.Error:
            return None
        if located.num_rows > limit:
            return None
        columns = (self._columns(table) if features is None
                   else self._wanted(table, table, features))
        if not located.num_rows:
            return _arrow_of(self._conn, "SELECT " + ", ".join(
                _ident(c) for c in columns) + f" FROM {_ident(table)} LIMIT 0")
        return self._read_located(located, columns)

    def _part(self, rel: str):
        """A Parquet file of the store, opened once."""
        if rel not in self._parts:
            pq = _require("pyarrow.parquet")
            self._parts[rel] = pq.ParquetFile(os.path.join(self.project, rel))
        return self._parts[rel]

    def _read_located(self, located, columns: List[str]):
        """The located rows, reading only their row groups and ``columns``."""
        pa = _require("pyarrow")
        parts = located.column("_part").to_pylist()
        groups = located.column("_row_group").to_numpy()
        rows = located.column("_group_row").to_numpy()
        pieces, positions = [], []
        for rel in sorted(set(parts)):
            mine = np.flatnonzero(np.asarray(parts, dtype=object) == rel)
            handle = self._part(rel)
            names = set(handle.schema_arrow.names)
            chosen = [c for c in columns if c in names]
            taken, where = [], []
            for group in sorted(set(int(g) for g in groups[mine])):
                hit = mine[groups[mine] == group]
                read = handle.read_row_group(group, columns=chosen,
                                             use_threads=True)
                taken.append(read.take(pa.array(rows[hit].astype(np.int64))))
                where.append(hit)
            mine = np.concatenate(where)
            piece = pa.concat_tables(taken)
            for name in columns:
                if name not in piece.column_names:
                    piece = piece.append_column(name, pa.nulls(piece.num_rows))
            pieces.append(piece.select(columns))
            positions.append(mine)
        try:
            whole = pa.concat_tables(pieces, promote_options="default")
        except TypeError:
            whole = pa.concat_tables(pieces, promote=True)
        order = np.argsort(np.concatenate(positions), kind="stable")
        return whole.take(pa.array(order))

    def _select(self, where, table: str, features,
                source: Optional[str] = None) -> tuple:
        """The SQL and parameters of one :meth:`query`, run on ``source``."""
        if isinstance(where, str) and where.lstrip().lower().startswith(
                ("select", "with")):
            if features is not None:
                raise StoreError("features= applies to a table query, not to "
                                 "a full SELECT")
            return where, None
        if table not in self._tables:
            raise StoreError(f"no table {table!r}; this store holds "
                             f"{self.tables()}")
        source = source or table
        columns = "*"
        if features is not None:
            columns = ", ".join(_ident(c) for c in self._wanted(
                table, source, features))
        sql = f"SELECT {columns} FROM {_ident(source)}"
        params: List[Any] = []
        if isinstance(where, Mapping):
            clauses = []
            for name, value in where.items():
                if value is None:
                    clauses.append(f"{_ident(name)} IS NULL")
                elif isinstance(value, (list, tuple, set, np.ndarray)):
                    values = list(value)
                    clauses.append(f"{_ident(name)} IN (" + ", ".join(
                        "?" for _ in values) + ")")
                    params.extend(_python(v) for v in values)
                else:
                    clauses.append(f"{_ident(name)} = ?")
                    params.append(_python(value))
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
        elif where:
            sql += f" WHERE {where}"
        return sql, params or None

    def query(self, where=None, *, table: str = "cells", features=None,
              crops: bool = False, as_: str = "pandas"):
        """Rows of one table, optionally with their crops in row order.

        :param where: a SQL ``WHERE`` clause, a full ``SELECT``, a mapping
            ``{column: value or list}``, or None for every row.
        :param table: the table.
        :param features: the feature columns to return beside the key
            columns and the crop pointer; None returns every column.
        :param crops: also return the crops, ``(n, C, H, W)`` in row order;
            a row without a crop gets zeros.
        :param as_: ``"pandas"``, ``"arrow"`` or ``"numpy"``, the last
            giving ``(metadata DataFrame, float32 feature matrix)``.
        :returns: the result, or ``(result, crops)`` when ``crops``.
        """
        result = self._routed(where, table, features)
        if result is None:
            sql, params = self._select(where, table, features)
            result = _arrow_of(self._conn, sql, params)
        return self._shaped(result, table, features, crops, as_)

    def sql(self, statement: str, params=None, *, as_: str = "pandas"):
        """Run raw SQL against the views.

        :param statement: the SQL.
        :param params: its ``?`` parameters.
        :param as_: ``"pandas"`` or ``"arrow"``.
        :returns: the result.
        """
        result = _arrow_of(self._conn, statement, params)
        return result if as_ == "arrow" else result.to_pandas()

    def batches(self, where=None, *, table: str = "cells", features=None,
                crops: bool = False, as_: str = "numpy",
                batch_size: int = 1024, shuffle: bool = False,
                seed: int = 0, buffer: int = 8) -> Iterator[Any]:
        """Iterate over a query in batches, for training.

        Rows stream from DuckDB as Arrow record batches, so the whole
        result is never held at once. With ``shuffle`` the files are read in
        a random order and the rows are shuffled within ``buffer`` batches;
        the same seed gives the same order.

        :param where: as for :meth:`query`, but not a full ``SELECT`` when
            shuffling.
        :param table: as for :meth:`query`.
        :param features: as for :meth:`query`.
        :param crops: as for :meth:`query`.
        :param as_: as for :meth:`query`; ``"numpy"`` by default.
        :param batch_size: rows per batch.
        :param shuffle: shuffle the files and the rows within the buffer.
        :param seed: the shuffle seed.
        :param buffer: how many batches are shuffled together.
        :returns: an iterator over results shaped as :meth:`query` shapes
            them.
        """
        sql, params = self._select(where, table, features)
        if not shuffle:
            reader = _reader_of(self._conn, sql, params, int(batch_size))
            for batch in reader:
                yield self._shaped(_table_of([batch]), table, features, crops,
                                   as_)
            return
        if isinstance(where, str) and where.lstrip().lower().startswith(
                ("select", "with")):
            raise StoreError("shuffle needs a WHERE clause or a mapping, "
                             "not a full SELECT")
        rng = np.random.default_rng(seed)
        paths = [os.path.join(self.project, f["path"]) for f in self._files
                 if f["table_name"] == table and f["kind"] == "parquet"]
        pending: List[Any] = []
        held = 0
        for position in rng.permutation(len(paths)):
            one = sql.replace(f"FROM {_ident(table)}",
                              f"FROM {_source([paths[position]])}", 1)
            for batch in _reader_of(self._conn, one, params, int(batch_size)):
                pending.append(batch)
                held += batch.num_rows
                if held >= batch_size * buffer:
                    yield from self._drain(pending, rng, batch_size, table,
                                           features, crops, as_)
                    pending, held = [], 0
        if pending:
            yield from self._drain(pending, rng, batch_size, table, features,
                                   crops, as_)

    def _drain(self, pending, rng, batch_size, table, features, crops, as_):
        """Shuffle the buffered rows and yield them in batches."""
        pa = _require("pyarrow")
        whole = _table_of(pending)
        whole = whole.take(pa.array(rng.permutation(whole.num_rows)))
        for start in range(0, whole.num_rows, batch_size):
            yield self._shaped(whole.slice(start, batch_size), table,
                               features, crops, as_)

    def _shaped(self, result, table, features, crops, as_):
        """Convert an Arrow result to ``as_`` and attach the crops."""
        stack = self._crops_for(result) if crops else None
        if as_ == "arrow":
            shaped = result
        elif as_ == "pandas":
            shaped = result.to_pandas(split_blocks=True)
        elif as_ == "numpy":
            shaped = self._as_numpy(result, table, features)
        else:
            raise StoreError(f"as_ must be pandas, arrow or numpy; got {as_!r}")
        return (shaped, stack) if crops else shaped

    def _as_numpy(self, result, table, features):
        """``(metadata DataFrame, float32 matrix)`` of an Arrow result."""
        import pyarrow.types as types

        keys = set(self._tables.get(table, {}).get("key_columns") or [])
        keys |= {"plate", "well", "cell_id", *_POINTER}
        fields = list(result.schema)
        if features is not None:
            present = {f.name: i for i, f in enumerate(fields)}
            chosen = [present[f] for f in features if f in present]
        else:
            chosen = [i for i, f in enumerate(fields) if f.name not in keys and (
                types.is_floating(f.type) or types.is_integer(f.type))]
        matrix = np.empty((result.num_rows, len(chosen)), dtype=np.float32,
                          order="F")
        for position, index in enumerate(chosen):
            column = result.column(index)
            matrix[:, position] = (column.to_numpy() if column.null_count == 0
                                   else np.asarray(column.to_pandas(),
                                                   dtype=np.float64))
        taken = set(chosen)
        meta = result.select([i for i in range(len(fields))
                              if i not in taken]).to_pandas()
        return meta, np.ascontiguousarray(matrix)

    def _array(self, rel: str):
        """The Zarr crop array of one registered store, opened once."""
        if rel not in self._arrays:
            path = os.path.join(self.project, rel, "crops")
            self._arrays[rel] = (_ChunkReader.open(path, self.threads)
                                 or _require("zarr").open_array(path, mode="r"))
        return self._arrays[rel]

    def _crops_for(self, result) -> np.ndarray:
        """The crops of an Arrow result's rows, in row order."""
        if not all(name in result.column_names for name in _POINTER):
            raise StoreError("crops=True needs crop_store and crop_index in "
                             "the result")
        stores = result.column("crop_store").to_pylist()
        index = result.column("crop_index").to_pandas().to_numpy()
        present = sorted({s for s in stores if s})
        if not present:
            return np.zeros((len(stores), 0, 0, 0), dtype=np.uint16)
        first = self._array(present[0])
        out = np.zeros((len(stores),) + tuple(first.shape[1:]),
                       dtype=first.dtype)
        stores = np.asarray(stores, dtype=object)
        for rel in present:
            array = self._array(rel)
            if tuple(array.shape[1:]) != out.shape[1:]:
                raise StoreError(f"{rel} holds crops of shape "
                                 f"{array.shape[1:]}, not {out.shape[1:]}")
            rows = np.flatnonzero(stores == rel)
            out[rows] = _read_rows(array, index[rows].astype(np.int64))
        return out

    def close(self) -> None:
        """Close the in-memory connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Store":
        """The store itself, for ``with``."""
        return self

    def __exit__(self, *exc) -> None:
        """Close on leaving ``with``."""
        self.close()


def _python(value):
    """A NumPy scalar as a Python one, for DuckDB parameters."""
    return value.item() if isinstance(value, np.generic) else value


def _table_of(batches):
    """One Arrow table from record batches."""
    pa = _require("pyarrow")
    return pa.Table.from_batches(batches)


class _ChunkReader:
    """A Zarr format 2 array read chunk by chunk with numcodecs, in threads.

    Used for the stores :func:`write_part` writes (zstd, no filters, C
    order); zarr's own reader is used for anything else.
    """

    def __init__(self, path, meta, threads):
        """Hold the array's metadata."""
        self.path = path
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.separator = meta.get("dimension_separator") or "."
        self.threads = max(1, int(threads or 4))
        self.codec = _require("numcodecs").Zstd()

    @classmethod
    def open(cls, path, threads):
        """The reader of ``path``, or None when the array is not plain zstd."""
        try:
            with builtins.open(os.path.join(path, ".zarray"),
                               encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            return None
        compressor = meta.get("compressor") or {}
        if (meta.get("zarr_format") != 2 or meta.get("filters")
                or meta.get("order", "C") != "C"
                or compressor.get("id") != "zstd"
                or any(c != s for c, s in zip(meta["chunks"][1:],
                                              meta["shape"][1:]))):
            return None
        return cls(path, meta, threads)

    def _chunk(self, number: int) -> np.ndarray:
        """One decoded chunk."""
        name = self.separator.join([str(number)] + ["0"] * (len(self.shape) - 1))
        try:
            with builtins.open(os.path.join(self.path, name), "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return np.full(self.chunks, self.fill, dtype=self.dtype)
        return np.frombuffer(self.codec.decode(raw),
                             dtype=self.dtype).reshape(self.chunks)

    def __getitem__(self, span: slice) -> np.ndarray:
        """Rows ``span.start`` to ``span.stop``."""
        from concurrent.futures import ThreadPoolExecutor

        size = self.chunks[0]
        first, last = span.start // size, (span.stop - 1) // size
        numbers = list(range(first, last + 1))
        if len(numbers) > 1:
            with ThreadPoolExecutor(min(self.threads, len(numbers))) as pool:
                blocks = list(pool.map(self._chunk, numbers))
        else:
            blocks = [self._chunk(numbers[0])]
        whole = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        return whole[span.start - first * size:span.stop - first * size]


def _read_rows(array, indices: np.ndarray) -> np.ndarray:
    """``array[indices]`` reading each needed chunk span once."""
    out = np.empty((len(indices),) + tuple(array.shape[1:]), dtype=array.dtype)
    if not len(indices):
        return out
    chunk = int(array.chunks[0])
    order = np.argsort(indices, kind="stable")
    wanted = indices[order]
    chunk_of = wanted // chunk
    spans = np.split(np.arange(len(wanted)),
                     np.flatnonzero(np.diff(chunk_of) > 1) + 1)
    for span in spans:
        lo = int(chunk_of[span[0]]) * chunk
        hi = min(int(array.shape[0]), (int(chunk_of[span[-1]]) + 1) * chunk)
        block = np.asarray(array[lo:hi])
        out[order[span]] = block[wanted[span] - lo]
    return out


def _sqlite_tables(path: str) -> List[str]:
    """The table names of a SQLite file."""
    import sqlite3

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                         timeout=30) as conn:
        return [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name "
            "NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]


def _sqlite_arrow(path: str, table: str):
    """One SQLite table as an Arrow table, when DuckDB's reader is absent."""
    import sqlite3

    pa = _require("pyarrow")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                         timeout=30) as conn:
        cursor = conn.execute(f"SELECT * FROM {_ident(table)}")
        names = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    return pa.table({name: [row[i] for row in rows]
                     for i, name in enumerate(names)})


def open(project, *, threads: Optional[int] = None,
         verify: str = "quick") -> Store:
    """Open a spaCR store, or an older SQLite project, for reading.

    :param project: a project folder with ``spacr.duckdb``, or a folder or
        file holding ``measurements.db``.
    :param threads: DuckDB threads; None leaves DuckDB's default.
    :param verify: ``"quick"`` (sizes and mtimes), ``"full"`` (sha256) or
        ``"none"``.
    :returns: the :class:`Store`.
    :raises StoreError: when there is nothing to open, or a registered file
        changed.
    """
    return Store(project, threads=threads, verify=verify)
