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
    pq.write_table(data, temporary, compression="zstd",
                   row_group_size=_row_group_rows(data.schema, row_group_rows),
                   write_statistics=True, use_dictionary=others or False,
                   use_byte_stream_split=floats or False)
    _fsync(temporary)
    os.replace(temporary, final)
    files.insert(0, {"path": os.path.relpath(final, project).replace(
        os.sep, "/"), "kind": "parquet", "rows": int(data.num_rows),
        **_fingerprint(final)})
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
    tables = sorted({m["table"] for _p, m in manifests})
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
            for (old,) in conn.execute(
                    "SELECT path FROM spacr_files WHERE table_name = ? AND "
                    "plate = ? AND well = ?",
                    [manifest["table"], manifest["plate"],
                     manifest["well"]]).fetchall():
                if old not in new:
                    superseded.append(old)
            conn.execute("DELETE FROM spacr_files WHERE table_name = ? AND "
                         "plate = ? AND well = ?",
                         [manifest["table"], manifest["plate"],
                          manifest["well"]])
            for entry in manifest["files"]:
                conn.execute(
                    "INSERT OR REPLACE INTO spacr_files VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [entry["path"], manifest["table"], entry["kind"],
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
        for name in self._tables:
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
        """The columns of ``table``."""
        return [row[0] for row in self._conn.execute(
            f"DESCRIBE SELECT * FROM {_ident(table)}").fetchall()]

    def _select(self, where, table: str, features) -> tuple:
        """The SQL and parameters of one :meth:`query`."""
        if isinstance(where, str) and where.lstrip().lower().startswith(
                ("select", "with")):
            if features is not None:
                raise StoreError("features= applies to a table query, not to "
                                 "a full SELECT")
            return where, None
        if table not in self._tables:
            raise StoreError(f"no table {table!r}; this store holds "
                             f"{self.tables()}")
        columns = "*"
        if features is not None:
            present = self._columns(table)
            keys = list(self._tables[table].get("key_columns") or [])
            wanted = [c for c in keys + ["plate", "well", "cell_id",
                                         *_POINTER] if c in present]
            missing = [f for f in features if f not in present]
            if missing:
                raise StoreError(f"{table} has no column {missing[:5]}")
            wanted += [f for f in features if f not in wanted]
            columns = ", ".join(_ident(c) for c in wanted)
        sql = f"SELECT {columns} FROM {_ident(table)}"
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
            zarr = _require("zarr")
            self._arrays[rel] = zarr.open_array(
                os.path.join(self.project, rel, "crops"), mode="r")
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
