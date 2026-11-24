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
import functools
import glob
import hashlib
import json
import os
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["StoreError", "Store", "open", "writer", "WellWriter",
           "write_table", "write_part", "register", "cleanup", "CATALOG",
           "SCHEMA_VERSION"]

#: The catalog's file name at the project root.
CATALOG = "spacr.duckdb"

#: The layout version written into every registry row.
SCHEMA_VERSION = 1

#: Rows per Parquet row group, before the byte cap. A scattered lookup reads
#: whole row groups, so 8,192 rows keep a point read of one cell at
#: 1/16 of the 131,072-row group it replaced; the page index and the column
#: statistics keep full scans as fast (benchmark, 100M x 300: a single-cell
#: read touched one 50,000-row group, 0.40 s).
_ROW_GROUP_ROWS = 8192

#: The most uncompressed bytes one row group may hold.
_ROW_GROUP_BYTES = 64 * 2 ** 20

#: Rows per row group of a key table, which holds only narrow columns.
_KEYS_GROUP_ROWS = 131072

#: The most bytes :meth:`Store.query` holds from its fallback scan before it
#: stops and names :meth:`Store.batches`.
_QUERY_BYTES = 8 * 2 ** 30

#: Rows per Arrow batch the fallback scan of :meth:`Store.query` reads.
_QUERY_BATCH = 8192

#: The file at the project root that holds the store's layout version.
STORE_VERSION_FILE = "STORE_VERSION"

#: The lock file :func:`register` holds while it writes the catalog.
_LOCK = ".spacr-register.lock"

#: Seconds :func:`register` waits for another registering process.
_LOCK_WAIT = 120.0

#: Age in seconds after which a lock file whose process is gone is stale.
_LOCK_STALE = 600.0

#: Cells per Zarr chunk.
_CROP_CHUNK = 256

_POINTER = ("crop_store", "crop_index")
_PARTITION = ("plate", "well")
_PENDING = "_pending"
_KEYS = "__keys"
_ROWS = "__rows"

#: Rows per chunk of a row-major copy. Item 974's micro-benchmark (20,000
#: rows x 10,000 float32, 1,000 random rows read, 4 threads): 4 rows 0.37 s,
#: 8 rows 0.42 s, 16 rows 0.61 s, 32 rows 0.70 s, 64 rows 1.08 s; 16 keeps
#: 6,250 chunk files per 100,000-cell well at a read time within 2x of the
#: smallest chunks.
_ROW_CHUNK = 16
_LOCATORS = ("_part", "_row_group", "_group_row")

#: The most rows (or 2 % of the table) a selection may hold to be read from
#: the row-major copy, or row group by row group when it asks for at most
#: :data:`_NARROW_COLUMNS` columns.
_LOOKUP_ROWS = 20_000

#: The most columns a scattered selection may ask for on the row-group path.
_NARROW_COLUMNS = 64

#: The largest share of the table's row groups a wide selection may touch
#: and still be read row group by row group; above it DuckDB scans the view.
_TOUCHED_GROUPS = 0.25

#: The most rows the key table is asked to locate before DuckDB takes over.
_LOCATE_MAX = 2_000_000
_LEGACY_NAMES = ("measurements.db", os.path.join("measurements",
                                                 "measurements.db"))

#: The recorded storage dtype of each measurement column, each measured on
#: its own by ``tools/measure_store_dtypes.py`` (the map names the plate,
#: well and cells it was measured on).
_DTYPE_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "resources", "data", "store_dtypes.json")

#: The object classes a column name may start with; any module's.
_COMPARTMENTS = ("cell", "nucleus", "cytoplasm", "pathogen")

#: The integer types a column may be stored in, narrowest first.
_INT_LADDER = ("uint8", "int8", "uint16", "int16", "uint32", "int32",
               "int64")


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


def _writer_options(schema) -> Dict[str, Any]:
    """The Parquet encodings of a part: zstd, dictionaries, byte-stream split.

    Dictionaries go to the non-float columns; byte-stream split to float32
    and float64 only, because DuckDB 2.0 reads byte-stream-split data only
    for FLOAT, DOUBLE, INT32 and INT64. Statistics are on.
    """
    pa = _require("pyarrow")
    floats = {f.name for f in schema if pa.types.is_floating(f.type)}
    split = [f.name for f in schema if f.name in floats
             and not pa.types.is_float16(f.type)]
    others = [n for n in schema.names if n not in floats]
    return {"compression": "zstd", "write_statistics": True,
            "use_dictionary": others or False,
            "use_byte_stream_split": split or False}


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


def _create_crop_array(path: str, shape, dtype, name: str = "crops",
                       chunk_rows: int = _CROP_CHUNK):
    """A Zarr format 2 array under ``path``/``name``, through zarr 2 or 3."""
    zarr = _require("zarr")
    numcodecs = _require("numcodecs")
    chunks = (min(chunk_rows, max(1, shape[0])),) + tuple(shape[1:])
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


def write_table(project, table: str, frame, *, plate, well,
                run_id: Optional[str] = None,
                sort_by: Optional[Sequence[str]] = ("object_key", "timeID"),
                key_columns: Optional[Sequence[str]] = None, crops=None,
                crop_rows=None, masks=None,
                row_group_rows: Optional[int] = None,
                row_copy: bool = False, narrow: bool = False,
                channels: Sequence[str] = ()) -> Dict[str, Any]:
    """Write one well's rows of ``table``, held whole, and leave a manifest.

    The one-shot form of :func:`writer`: the rows are sorted by
    ``sort_by`` (those present, nulls last) and handed to one
    :class:`WellWriter`, which writes them under a temporary name, flushes
    and renames them into ``tables/<table>/plate=<P>/well=<W>/``, puts the
    crops in ``crops/plate=<P>/well=<W>.zarr`` in the same row order (each
    row gets ``crop_store`` and ``crop_index``) and writes the manifest in
    ``_pending/`` last; nothing is visible to readers until
    :func:`register` adds it to the catalog.

    :param project: the project folder.
    :param table: the object class or table, for example ``cell``.
    :param frame: a DataFrame or Arrow table, one row per object.
    :param plate: the plate name.
    :param well: the well name.
    :param run_id: the run this write belongs to.
    :param sort_by: the columns the rows are sorted by, those present.
    :param key_columns: the columns every feature query returns.
    :param crops: an array ``(k, C, H, W)`` indexable by an integer array.
    :param crop_rows: the frame row of each crop; every row when None.
    :param masks: an array ``(k, H, W)`` beside the crops, or None.
    :param row_group_rows: rows per row group, under the byte cap.
    :param row_copy: also write the float columns as a row-major Zarr copy,
        which point lookups of every column read instead of the Parquet part.
    :param narrow: store numeric columns in the recorded dtypes
        (:func:`_stored_types`), guarded by the frame's own ranges.
    :param channels: the run's channel names, for the dtype families.
    :returns: the manifest.
    :raises StoreError: when the frame's plate or well disagree with the
        partition, or the crops do not match ``crop_rows``.
    """
    pa = _require("pyarrow")
    data = _arrow_table(frame)
    order = _sorted_order(data, sort_by or ())
    data = data.take(pa.array(order))
    position = None
    if crops is not None:
        rows = (np.arange(len(order)) if crop_rows is None
                else np.asarray(crop_rows, dtype=np.int64))
        if len(rows) != int(crops.shape[0]):
            raise StoreError(
                f"{table}: {crops.shape[0]} crops for {len(rows)} crop rows")
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        position = inverse[rows]
    schema = ranges = None
    if narrow:
        schema = data.schema.remove_metadata()
        ranges = _table_ranges(data)
    part = WellWriter(project, table, plate=plate, well=well, schema=schema,
                      ranges=ranges, narrow=narrow, channels=channels,
                      sort_by=sort_by or (), key_columns=key_columns or (),
                      crops=crops, crop_rows=position, masks=masks,
                      run_id=run_id, row_group_rows=row_group_rows,
                      row_copy=row_copy)
    part._sorted_here = True
    with part:
        part.append(data)
    return part.manifest


def write_part(project, table: str, frame, *, plate, well,
               sort_by: Sequence[str] = ("gene", "cell_id"),
               key_columns: Sequence[str] = (), crops=None,
               crop_rows=None, masks=None, run_id: Optional[str] = None,
               row_group_rows: Optional[int] = None,
               row_copy: bool = False) -> Dict[str, Any]:
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
    :param row_copy: also write the float columns as a row-major Zarr copy,
        which point lookups of every column read instead of the Parquet part.
    :returns: the manifest.
    :raises StoreError: when the frame's plate or well disagree with the
        partition, or the crops do not match ``crop_rows``.
    """
    return write_table(project, table, frame, plate=plate, well=well,
                       run_id=run_id, sort_by=sort_by,
                       key_columns=key_columns, crops=crops,
                       crop_rows=crop_rows, masks=masks,
                       row_group_rows=row_group_rows, row_copy=row_copy)


def writer(project, table: str, *, plate, well, run_id: Optional[str] = None,
           sort_by: Sequence[str] = ("object_key", "timeID"),
           key_columns: Optional[Sequence[str]] = None,
           row_copy: bool = False, **options) -> "WellWriter":
    """The streamed writer of one well's part of ``table``: every module's.

    Measure, OPS and every other module that saves a table too large to
    hold at once appends it here a field (or a batch) at a time and
    finishes it once; :func:`write_table` is the same writer for a frame
    held whole. Use it as a context manager: it finishes on success and
    aborts on an error, so nothing half-written becomes visible.

    :param project: the project folder (a run's ``measurements``).
    :param table: the object class or table.
    :param plate: the plate name.
    :param well: the well name.
    :param run_id: the run this write belongs to.
    :param sort_by: the order the rows are appended in (recorded only when
        they did arrive in it).
    :param key_columns: the columns every feature query returns.
    :param row_copy: also write the row-major copy of the float columns.
    :param options: :class:`WellWriter`'s other arguments: ``schema``,
        ``ranges``, ``narrow``, ``channels``, ``dtype_map``, ``crops``,
        ``crop_rows``, ``masks`` and ``row_group_rows``.
    :returns: the :class:`WellWriter`.
    """
    return WellWriter(project, table, plate=plate, well=well, run_id=run_id,
                      sort_by=sort_by, key_columns=key_columns or (),
                      row_copy=row_copy, **options)


def _table_ranges(data) -> Dict[str, tuple]:
    """``column -> (min, max)`` of an Arrow table's numeric columns, NaN skipped."""
    pa = _require("pyarrow")
    pc = _require("pyarrow.compute")
    ranges: Dict[str, tuple] = {}
    for field in data.schema:
        if not (pa.types.is_floating(field.type)
                or pa.types.is_integer(field.type)):
            continue
        column = data.column(field.name)
        if pa.types.is_floating(field.type):
            column = pc.if_else(pc.is_nan(column), pa.scalar(None, field.type),
                                column)
        found = pc.min_max(column)
        low, high = found["min"].as_py(), found["max"].as_py()
        ranges[field.name] = ((np.nan, np.nan) if low is None
                              else (float(low), float(high)))
    return ranges


class _CropPieces:
    """Crops appended a batch at a time, indexable as one array."""

    def __init__(self):
        """No pieces yet."""
        self.pieces: List[tuple] = []
        self.count = 0

    def add(self, crops, masks, order) -> None:
        """Keep ``crops[order]`` (and ``masks[order]``) as the next crops."""
        self.pieces.append((crops, masks, np.asarray(order, dtype=np.int64)))
        self.count += len(order)

    @property
    def shape(self):
        """``(k, C, H, W)`` of every crop held."""
        return (self.count,) + tuple(self.pieces[0][0].shape[1:])

    @property
    def dtype(self):
        """The crops' dtype."""
        return np.dtype(self.pieces[0][0].dtype)

    def _take(self, rows, masks: bool) -> np.ndarray:
        """Crops (or masks) of ``rows``, counted over every piece."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = np.cumsum([0] + [len(p[2]) for p in self.pieces])
        first = self.pieces[0][1 if masks else 0]
        out = np.empty((len(rows),) + tuple(first.shape[1:]), dtype=first.dtype)
        piece = np.searchsorted(starts, rows, side="right") - 1
        for number in np.unique(piece):
            crops, mask, order = self.pieces[int(number)]
            mine = np.flatnonzero(piece == number)
            source = mask if masks else crops
            out[mine] = np.asarray(source[order[rows[mine] - starts[number]]])
        return out

    def __getitem__(self, rows) -> np.ndarray:
        """The crops of ``rows``."""
        return self._take(rows, False)

    def masks(self):
        """The masks as an indexable view, or None when no piece has any."""
        if any(p[1] is None for p in self.pieces):
            return None
        holder = self

        class _Masks:
            """The masks of the held pieces."""

            shape = (holder.count,) + tuple(holder.pieces[0][1].shape[1:])
            dtype = np.dtype(holder.pieces[0][1].dtype)

            def __getitem__(self, rows):
                """The masks of ``rows``."""
                return holder._take(rows, True)

        return _Masks()


class WellWriter:
    """Write one well's part of a table in batches, then leave its manifest.

    The one writer every module saves tables with (made by :func:`writer`;
    :func:`write_table` hands it a frame held whole). Batches -- a well's
    cells one field at a time -- are appended in order, so a well needs
    the memory of one row group, not of the well, and the part is one file
    per table per well: nothing is spooled and nothing needs compacting.
    :meth:`finish` writes the key table, the crops and the row-major copy
    from the finished part, and the manifest last.

    Values: every batch is cast to the part's schema, which with
    ``narrow`` is the recorded dtype map's (:func:`_stored_types`, keys
    untouched), widened where ``ranges`` show a column does not fit. A
    value the stored type cannot hold is refused, never wrapped or turned
    infinite; NaN is stored as null (DuckDB orders NaN above every number
    and averages it to NaN), and inf stays inf. float16 columns are not
    byte-stream split, which DuckDB 2.0 cannot read.

    :param project: the project folder.
    :param table: the object class or table.
    :param plate: the plate name.
    :param well: the well name.
    :param schema: the part's Arrow schema without ``plate``, ``well`` and
        the crop pointer, as the source holds it; a column a batch lacks is
        null. None takes the first batch's schema.
    :param ranges: ``column -> (min, max)`` over the whole part
        (:func:`_column_ranges` of the source), for the dtype guard; without
        them a float16 family is stored as float32.
    :param narrow: apply the recorded dtype map to ``schema``.
    :param channels: the run's channel names, for the dtype families.
    :param dtype_map: the recorded dtype map; None is the shipped one
        (``spacr/resources/data/store_dtypes.json``).
    :param sort_by: the order the caller appends the rows in. Nothing is
        sorted here: the catalog records it only when the rows did arrive
        non-decreasing in its first column, and records no order otherwise.
    :param key_columns: the columns every feature query returns; never
        narrowed.
    :param crops: the well's crops, ``(k, C, H, W)`` indexable by an integer
        array, given at once (or per batch to :meth:`append`, not both).
    :param crop_rows: the part row of each of those crops; every row when
        None.
    :param masks: an array ``(k, H, W)`` beside them, or None.
    :param run_id: the run this write belongs to.
    :param row_group_rows: rows per row group, under the byte cap.
    :param row_copy: also write the row-major copy of the float columns.
    :raises StoreError: from :meth:`append` when a batch's plate or well
        disagree with the partition, a value does not fit its column's type
        or a column is not in the schema; from :meth:`finish` when the crops
        do not match the rows written.
    """

    def __init__(self, project, table: str, *, plate, well, schema=None,
                 ranges=None, narrow: bool = False,
                 channels: Sequence[str] = (),
                 dtype_map: Optional[str] = None,
                 sort_by: Sequence[str] = (),
                 key_columns: Sequence[str] = (), crops=None,
                 crop_rows=None, masks=None, run_id: Optional[str] = None,
                 row_group_rows: Optional[int] = None,
                 row_copy: bool = False):
        """Reserve the part's names; nothing is written before :meth:`append`."""
        pa = _require("pyarrow")
        self.project = os.path.abspath(os.fspath(project))
        self.table = table
        self.plate, self.well = plate, well
        self.sort_by = list(sort_by)
        self.key_columns = list(key_columns)
        self.crops, self.masks = crops, masks
        self.run_id = run_id
        self.row_copy = row_copy
        self.rows = 0
        self.manifest: Optional[Dict[str, Any]] = None
        self.promoted: List[Dict[str, Any]] = []
        self._ordered = bool(self.sort_by)
        self._sorted_here = False
        self._last = None
        self.uid = uuid.uuid4().hex
        self._requested = row_group_rows
        if schema is not None:
            schema = schema.remove_metadata()
            if narrow:
                types, self.promoted = _stored_types(
                    pa.schema([f for f in schema
                               if f.name not in set(self.key_columns)]),
                    ranges or {}, channels, path=dtype_map)
                schema = pa.schema([pa.field(f.name, types.get(f.name, f.type))
                                    for f in schema])
        self._schema = schema
        self.dtype_map_version = (
            int(_dtype_map(dtype_map).get("version") or 0) or None
            if schema is not None and narrow else None)
        self._full = None
        self._writer = None
        self._pending: List[Any] = []
        self._held = 0
        self._group_rows = 0
        self._crop_rows = None
        self._ranked = None
        self._crop_store = None
        self._appended = None
        if crops is not None:
            count = int(crops.shape[0])
            if crop_rows is not None:
                self._crop_rows = np.asarray(crop_rows, dtype=np.int64)
                if len(self._crop_rows) != count:
                    raise StoreError(f"{table}: {count} crops for "
                                     f"{len(self._crop_rows)} crop rows")
                self._ranked = np.sort(self._crop_rows)
                if len(self._ranked) and (
                        self._ranked[0] < 0
                        or np.any(np.diff(self._ranked) == 0)):
                    raise StoreError(f"{table}: crop rows must be distinct "
                                     f"and not negative")
            self._reserve_crops()
        folder = os.path.join(self.project, "tables", table,
                              f"plate={_encoded(plate)}",
                              f"well={_encoded(well)}")
        os.makedirs(folder, exist_ok=True)
        self._final = os.path.join(folder, _next_name(folder, "part-",
                                                      ".parquet"))
        self._temporary = f"{self._final}.tmp-{self.uid}"

    def _reserve_crops(self) -> None:
        """Choose the well's crop store name."""
        crop_dir = os.path.join(self.project, "crops",
                                f"plate={_encoded(self.plate)}")
        os.makedirs(crop_dir, exist_ok=True)
        stem = f"well={_encoded(self.well)}"
        name = (f"{stem}.zarr" if not os.path.exists(
            os.path.join(crop_dir, f"{stem}.zarr"))
            else _next_name(crop_dir, f"{stem}.v", ".zarr"))
        self._crop_final = os.path.join(crop_dir, name)
        self._crop_store = os.path.relpath(
            self._crop_final, self.project).replace(os.sep, "/")

    def __enter__(self) -> "WellWriter":
        """The writer itself, for ``with``."""
        return self

    def __exit__(self, kind, value, trace) -> None:
        """Finish on success, abort on an error."""
        if kind is None:
            self.finish()
        else:
            self.abort()

    def _conform(self, data):
        """``data`` cast to the part's schema, each value checked to fit."""
        pa = _require("pyarrow")
        pc = _require("pyarrow.compute")
        if self._schema is None:
            self._schema = data.schema.remove_metadata()
        names = set(self._schema.names)
        extra = [n for n in data.column_names if n not in names]
        if extra:
            raise StoreError(f"{self.table}: column {extra[:5]} is not in "
                             f"the part's schema")
        present = {n: i for i, n in enumerate(data.column_names)}
        arrays = []
        for field in self._schema:
            if field.name not in present:
                arrays.append(pa.nulls(data.num_rows, field.type))
                continue
            column = data.column(present[field.name])
            if pa.types.is_floating(column.type):
                nan = pc.is_nan(column)
                if pc.any(nan).as_py():
                    column = pc.if_else(nan, pa.scalar(None, column.type),
                                        column)
            if column.type == field.type:
                arrays.append(column)
                continue
            try:
                cast = pc.cast(column, field.type, safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                raise StoreError(f"{self.table}: column {field.name} does "
                                 f"not fit {field.type}: {error}") from error
            if (pa.types.is_floating(field.type)
                    and pa.types.is_floating(column.type)
                    and _overflows(column, cast)):
                raise StoreError(
                    f"{self.table}: column {field.name} holds finite values "
                    f"beyond the range of {field.type}")
            arrays.append(cast)
        return pa.Table.from_arrays(arrays, schema=self._schema)

    def append(self, frame, *, field=None, crops=None, crop_rows=None,
               masks=None) -> int:
        """Append the rows of ``frame``, a DataFrame or Arrow table.

        :param frame: the rows, in the part's order.
        :param field: the field they come from, for messages only.
        :param crops: these rows' crops, ``(k, C, H, W)``, when the crops
            were not given to the writer.
        :param crop_rows: the frame row of each crop; every row when None.
        :param masks: their masks, or None.
        :returns: the rows written so far.
        """
        pa = _require("pyarrow")
        data = _arrow_table(frame)
        for name, value in (("plate", self.plate), ("well", self.well)):
            if name in data.column_names:
                found = set(map(str, data.column(name).to_pylist()))
                if found - {str(value)}:
                    raise StoreError(
                        f"{self.table}: column {name} holds "
                        f"{sorted(found)[:3]}, not only {value!r}; one part "
                        f"holds one {name}")
                data = data.drop_columns([name])
        data = data.drop_columns([c for c in _POINTER
                                  if c in data.column_names])
        try:
            data = self._conform(data)
        except StoreError as error:
            if field is None:
                raise
            raise StoreError(f"field {field}: {error}") from error
        count = data.num_rows
        if self._ordered and count and not self._sorted_here:
            self._check_order(data)
        if crops is not None:
            if self.crops is not None:
                raise StoreError(f"{self.table}: crops were given to the "
                                 f"writer and to append")
            local = (np.arange(count) if crop_rows is None
                     else np.asarray(crop_rows, dtype=np.int64))
            if len(local) != int(crops.shape[0]):
                raise StoreError(f"{self.table}: {crops.shape[0]} crops for "
                                 f"{len(local)} crop rows")
            if self._appended is None:
                self._appended = _CropPieces()
                self._reserve_crops()
            order = np.argsort(local, kind="stable")
            index = np.full(count, -1, dtype=np.int64)
            index[local[order]] = self._appended.count + np.arange(len(local))
            self._appended.add(crops, masks, order)
            has = index >= 0
        elif self.crops is not None:
            here = np.arange(self.rows, self.rows + count, dtype=np.int64)
            if self._ranked is None:
                index = here
                has = here < int(self.crops.shape[0])
            else:
                index = np.searchsorted(self._ranked, here)
                inside = index < len(self._ranked)
                has = np.zeros(count, bool)
                has[inside] = self._ranked[index[inside]] == here[inside]
        else:
            has = None
        if has is not None or self._appended is not None:
            if has is None:
                has = np.zeros(count, bool)
                index = np.full(count, -1, dtype=np.int64)
            data = data.append_column("crop_store", pa.array(
                [self._crop_store if h else None for h in has], pa.string()))
            data = data.append_column("crop_index", pa.array(
                np.where(has, index, -1), pa.int64(), mask=~has))
        data = data.append_column("plate", pa.array(
            [str(self.plate)] * count, pa.string()))
        data = data.append_column("well", pa.array(
            [str(self.well)] * count, pa.string()))
        if self._writer is None:
            self._open(data.schema)
        elif data.schema != self._full:
            raise StoreError(f"{self.table}: crops must come with every "
                             f"batch or with none")
        self.rows += count
        self._pending.append(data)
        self._held += count
        if self._held >= self._group_rows:
            self._flush(final=False)
        return self.rows

    def _check_order(self, data) -> None:
        """Keep :attr:`sort_by` only while its first column never decreases."""
        name = self.sort_by[0]
        if name not in data.column_names or data.column(name).null_count:
            self._ordered = False
            return
        values = data.column(name).to_numpy(zero_copy_only=False)
        try:
            ordered = bool(np.all(values[1:] >= values[:-1])) and (
                self._last is None or values[0] >= self._last)
        except TypeError:
            ordered = False
        self._ordered = ordered
        self._last = values[-1]

    def _open(self, schema) -> None:
        """Open the temporary part with the store's encodings.

        zstd; dictionaries for the non-float columns; byte-stream split for
        float32 and float64 only, because DuckDB 2.0 reads byte-stream-split
        data only for FLOAT, DOUBLE, INT32 and INT64.
        """
        pq = _require("pyarrow.parquet")
        self._full = schema
        self._group_rows = _row_group_rows(schema, self._requested)
        self._writer = pq.ParquetWriter(self._temporary, schema,
                                        **_writer_options(schema))

    def _flush(self, *, final: bool) -> None:
        """Write every whole row group held, and the remainder when ``final``."""
        pa = _require("pyarrow")
        if not self._pending:
            return
        held = (pa.concat_tables(self._pending) if len(self._pending) > 1
                else self._pending[0])
        size = self._group_rows
        whole = held.num_rows if final else held.num_rows // size * size
        for start in range(0, whole, size):
            self._writer.write_table(held.slice(start, min(size,
                                                           whole - start)),
                                     row_group_size=size)
        rest = held.slice(whole)
        self._pending = [rest] if rest.num_rows else []
        self._held = rest.num_rows

    def finish(self) -> Dict[str, Any]:
        """Finish the part, write its key table, crops and copy, then the manifest.

        :returns: the manifest.
        """
        pa = _require("pyarrow")
        pq = _require("pyarrow.parquet")
        if self.manifest is not None:
            return self.manifest
        if self._writer is None:
            if self._schema is None:
                raise StoreError(f"{self.table}: no rows and no schema to "
                                 f"write")
            self.append(self._schema.empty_table())
        self._flush(final=True)
        self._writer.close()
        self._writer = None
        files: List[Dict[str, Any]] = []
        if self.crops is not None or self._appended is not None:
            if self._appended is not None:
                source, masks = self._appended, self._appended.masks()
                order = np.arange(self._appended.count)
            else:
                count = int(self.crops.shape[0])
                rows = (np.arange(count) if self._crop_rows is None
                        else self._crop_rows)
                if len(rows) and int(rows.max()) >= self.rows:
                    raise StoreError(f"{self.table}: a crop names row "
                                     f"{int(rows.max())} of {self.rows}")
                source, masks = self.crops, self.masks
                order = np.argsort(rows, kind="stable")
            temporary = f"{self._crop_final}.tmp-{self.uid}"
            _write_crops(temporary, source, order, masks)
            os.replace(temporary, self._crop_final)
            files.append({"path": self._crop_store, "kind": "zarr",
                          "rows": int(len(order)),
                          **_fingerprint(self._crop_final)})
        _fsync(self._temporary)
        os.replace(self._temporary, self._final)
        rel = os.path.relpath(self._final, self.project).replace(os.sep, "/")
        files.insert(0, {"path": rel, "kind": "parquet", "rows": self.rows,
                         **_fingerprint(self._final)})
        part = pq.ParquetFile(self._final)
        keep = set(self.key_columns) | set(_POINTER) | set(_PARTITION)
        narrow = [f.name for f in self._full if f.name in keep
                  or not pa.types.is_floating(f.type)]
        files.insert(1, _write_keys(self.project, self.table,
                                    part.read(columns=narrow), rel,
                                    self._group_rows, self.key_columns,
                                    self.plate, self.well, self.uid))
        if files[1].get("key_digest"):
            files[0]["key_digest"] = files[1]["key_digest"]
        if self.row_copy:
            files.append(_write_row_copy(self.project, self.table, part, rel,
                                         self.key_columns, self.uid))
        manifest = {"uuid": self.uid, "table": self.table,
                    "plate": str(self.plate), "well": str(self.well),
                    "sort_by": self.sort_by if self._ordered else [],
                    "key_columns": self.key_columns, "files": files,
                    "schema_version": SCHEMA_VERSION, "written_at": _now(),
                    "run_id": self.run_id, "promoted": self.promoted,
                    "dtype_map_version": self.dtype_map_version}
        pending = os.path.join(self.project, _PENDING)
        os.makedirs(pending, exist_ok=True)
        path = os.path.join(pending, f"{self.uid}.json")
        with builtins.open(f"{path}.tmp", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(f"{path}.tmp", path)
        self.manifest = manifest
        return manifest

    def abort(self) -> None:
        """Drop the unfinished part; nothing of it reaches the catalog."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        _remove(self._temporary)


def _rebuild(project, tables: Sequence[str]) -> List[str]:
    """Clear ``tables`` and an older ``measurements.db`` before a run rebuilds them.

    For a phase that rewrites its tables on every run, so a well this run
    does not write cannot survive from an older one (:func:`_drop_tables`).

    :returns: the paths removed, relative to the project.
    """
    project = os.path.abspath(os.fspath(project))
    for suffix in ("", "-wal", "-shm", "-journal"):
        _remove(os.path.join(project, _LEGACY_NAMES[0] + suffix))
    return _drop_tables(project, tables)


def _overflows(original, cast) -> bool:
    """Whether a float cast turned a finite value infinite."""
    pc = _require("pyarrow.compute")
    before = pc.sum(pc.is_finite(original)).as_py() or 0
    after = pc.sum(pc.is_finite(cast)).as_py() or 0
    return after < before


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
                   row_group_size=_KEYS_GROUP_ROWS, write_statistics=True)
    _fsync(temporary)
    os.replace(temporary, final)
    entry = {"path": os.path.relpath(final, project).replace(os.sep, "/"),
             "kind": "parquet", "table": _keys_name(table),
             "rows": int(keys.num_rows), **_fingerprint(final),
             "schema_digest": _schema_digest(keys.schema)}
    if "object_key" in keys.column_names:
        entry["key_digest"] = _column_digest(keys.column("object_key"))
    return entry


def _rows_name(table: str) -> str:
    """The registry name of ``table``'s row-major copies."""
    return f"{table}{_ROWS}"


def _row_copy_path(part: str) -> str:
    """Where the row-major copy of a Parquet part goes, relative."""
    head, _, rest = part.partition("/")
    return "rows/" + rest[:-len(".parquet")] + ".zarr"


def _write_row_copy(project: str, table: str, data, part: str,
                    key_columns: Sequence[str], uid: str):
    """Write the float columns of one part as a row-major Zarr array.

    The array ``features`` is ``(rows, float columns)`` in the part's row
    order, in the widest float type of those columns (float32 at least),
    chunked :data:`_ROW_CHUNK` rows by every column; its column names are
    in ``columns.json``. Nulls are stored as NaN. ``data`` is an Arrow
    table or the written part as a ``pyarrow.parquet.ParquetFile``, which is
    read a block of rows at a time.
    """
    pa = _require("pyarrow")
    schema = getattr(data, "schema_arrow", None) or data.schema
    total = (int(data.metadata.num_rows) if hasattr(data, "schema_arrow")
             else data.num_rows)
    keep = set(key_columns) | set(_POINTER) | set(_PARTITION)
    names = [f.name for f in schema if f.name not in keep
             and pa.types.is_floating(f.type)]
    wide = any(schema.field(n).type == pa.float64() for n in names)
    dtype = np.dtype(np.float64 if wide else np.float32)
    final = os.path.join(project, _row_copy_path(part))
    temporary = f"{final}.tmp-{uid}"
    os.makedirs(temporary, exist_ok=True)
    array = _create_crop_array(temporary, (total, len(names)), dtype,
                               "features", chunk_rows=_ROW_CHUNK)
    step = _ROW_CHUNK * 64
    if hasattr(data, "schema_arrow"):
        blocks = (_table_of([b]) for b in data.iter_batches(
            batch_size=step, columns=names))
    else:
        blocks = (data.slice(s, step) for s in range(0, total, step))
    start = 0
    for block in blocks:
        matrix = np.empty((block.num_rows, len(names)), dtype=dtype)
        for position, name in enumerate(names):
            matrix[:, position] = block.column(name).to_numpy(
                zero_copy_only=False)
        array[start:start + block.num_rows] = matrix
        start += block.num_rows
    with builtins.open(os.path.join(temporary, "columns.json"), "w",
                       encoding="utf-8") as handle:
        json.dump(names, handle)
    with builtins.open(os.path.join(temporary, ".zgroup"), "w",
                       encoding="utf-8") as handle:
        json.dump({"zarr_format": 2}, handle)
    os.replace(temporary, final)
    return {"path": os.path.relpath(final, project).replace(os.sep, "/"),
            "kind": "zarr", "table": _rows_name(table),
            "rows": int(total), **_fingerprint(final)}


_FILE_COLUMNS = (
    ("path", "VARCHAR"), ("table_name", "VARCHAR"), ("kind", "VARCHAR"),
    ("plate", "VARCHAR"), ("well", "VARCHAR"), ("rows", "BIGINT"),
    ("schema_version", "INTEGER"), ("sha256", "VARCHAR"),
    ("bytes", "BIGINT"), ("mtime_ns", "BIGINT"), ("written_at", "VARCHAR"),
    ("run_id", "VARCHAR"), ("schema_digest", "VARCHAR"),
    ("key_digest", "VARCHAR"), ("content_digest", "VARCHAR"),
    ("dtype_map_version", "INTEGER"))

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
    "CREATE TABLE IF NOT EXISTS spacr_columns (table_name VARCHAR, "
    "ordinal INTEGER, column_name VARCHAR, family VARCHAR, "
    "stored_type VARCHAR, compute_type VARCHAR, nullable BOOLEAN, "
    "dtype_map_version INTEGER)",
) + tuple(f"ALTER TABLE spacr_files ADD COLUMN IF NOT EXISTS {name} {kind}"
          for name, kind in _FILE_COLUMNS[12:])


def _message(text: str, **values: Any) -> str:
    """``text`` in the interface language, its ``{name}`` fields filled."""
    from .qt.i18n import tr

    return tr(text, **values)


def _schema_digest(schema) -> str:
    """The sha256 of a schema's column names and types, in order."""
    digest = hashlib.sha256()
    for field in schema:
        digest.update(f"{field.name}\0{field.type}\n".encode("utf-8"))
    return digest.hexdigest()


def _column_digest(column) -> str:
    """The sha256 of one Arrow column's values, independent of chunking."""
    pa = _require("pyarrow")
    one = pa.table({"v": pa.chunked_array(column.chunks).combine_chunks()
                    if hasattr(column, "chunks") else column})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, one.schema) as stream:
        stream.write_table(one)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _source(paths: Sequence[str], *, uniform: bool = False) -> str:
    """The ``read_parquet`` expression over ``paths``.

    :param uniform: every file has one schema (the catalog guarantees it),
        so DuckDB binds the first footer instead of unifying every file's
        on each query.
    """
    listed = ", ".join(_literal(p) for p in paths)
    return (f"read_parquet([{listed}], hive_partitioning = true, "
            f"hive_types = {{'plate': VARCHAR, 'well': VARCHAR}}, "
            f"union_by_name = {'false' if uniform else 'true'})")


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


def _uniform(digests: Sequence[Optional[str]]) -> bool:
    """Whether every file of a table has the same recorded schema."""
    return bool(digests) and None not in digests and len(set(digests)) == 1


def _rebuild_views(conn, project: str, tables: Sequence[str]) -> None:
    """Point each table's catalog view at its registered Parquet files.

    The paths are relative to the project, so a copied project opens in
    the DuckDB CLI started in its folder. A table whose files all record
    one schema is read without ``union_by_name``.
    """
    conn.execute(f"SET file_search_path = {_literal(project)}")
    for name in tables:
        rows = conn.execute(
            "SELECT path, schema_digest FROM spacr_files WHERE table_name = ? "
            "AND kind = 'parquet' ORDER BY plate, well, path",
            [name]).fetchall()
        if rows:
            same = _uniform([r[1] for r in rows])
            conn.execute(
                f"CREATE OR REPLACE VIEW {_ident(name)} AS SELECT * FROM "
                f"{_source([r[0] for r in rows], uniform=same)}")
        else:
            conn.execute(f"DROP VIEW IF EXISTS {_ident(name)}")


def _alive(pid: int) -> bool:
    """Whether a process with this id runs."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


@contextmanager
def _register_lock(project: str, wait: float = _LOCK_WAIT):
    """Hold the project's register lock file; retry while another holds it.

    The file is made with ``O_EXCL`` and names the holder's pid. One whose
    process has gone and that is older than :data:`_LOCK_STALE` is taken
    over.

    :raises StoreError: when the lock stays held for ``wait`` seconds.
    """
    path = os.path.join(project, _LOCK)
    deadline = time.monotonic() + wait
    delay = 0.05
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode("ascii"))
            os.close(handle)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(path)
                with builtins.open(path, encoding="ascii") as held:
                    holder = int(held.read().strip() or 0)
                if age > _LOCK_STALE and not _alive(holder):
                    os.remove(path)
                    continue
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise StoreError(_message(
                    "another process is registering {project}; its lock "
                    "file {lock} stayed for {seconds} s", project=project,
                    lock=path, seconds=int(wait))) from None
            time.sleep(delay)
            delay = min(delay * 2, 1.0)
    try:
        yield
    finally:
        _remove(path)


def _write_version(project: str) -> None:
    """Write the layout version into ``STORE_VERSION`` when it differs."""
    path = os.path.join(project, STORE_VERSION_FILE)
    text = f"{SCHEMA_VERSION}\n"
    try:
        with builtins.open(path, encoding="ascii") as handle:
            if handle.read() == text:
                return
    except OSError:
        pass
    with builtins.open(f"{path}.tmp", "w", encoding="ascii") as handle:
        handle.write(text)
    os.replace(f"{path}.tmp", path)


def _check_version(project: str) -> None:
    """Refuse a store a newer spaCR wrote."""
    try:
        with builtins.open(os.path.join(project, STORE_VERSION_FILE),
                           encoding="ascii") as handle:
            found = int(handle.read().strip())
    except (OSError, ValueError):
        return
    if found > SCHEMA_VERSION:
        raise StoreError(_message(
            "{project} is store version {found}; this spaCR reads up to "
            "version {known}. Update spaCR to open it.", project=project,
            found=found, known=SCHEMA_VERSION))


def _same_columns(schema, reference) -> bool:
    """Whether two schemas have the same column names and types in order."""
    return ([(f.name, f.type) for f in schema]
            == [(f.name, f.type) for f in reference])


def _conformed_part(path: str, reference, table: str) -> None:
    """Rewrite one part so its columns are ``reference``'s, losslessly.

    A column the part lacks becomes null, a column of another type is cast
    exactly (a value that does not fit refuses), and the columns are put in
    the reference's order. Row groups, order and statistics are kept.

    :raises StoreError: when the part holds a column the reference lacks,
        or a value does not fit.
    """
    pa = _require("pyarrow")
    pc = _require("pyarrow.compute")
    pq = _require("pyarrow.parquet")
    part = pq.ParquetFile(path)
    schema = part.schema_arrow.remove_metadata()
    extra = [n for n in schema.names if n not in set(reference.names)]
    if extra:
        raise StoreError(_message(
            "{path}: column(s) {names} are not in the other parts of "
            "{table}; one table has one schema. Write every well with the "
            "same columns.", path=path, names=", ".join(extra[:5]),
            table=table))
    temporary = f"{path}.tmp-{uuid.uuid4().hex}"
    out = pq.ParquetWriter(temporary, reference, **_writer_options(reference))
    try:
        for index in range(part.metadata.num_row_groups):
            data = part.read_row_group(index)
            arrays = []
            for field in reference:
                if field.name not in data.column_names:
                    arrays.append(pa.nulls(data.num_rows, field.type))
                    continue
                column = data.column(field.name)
                if column.type != field.type:
                    try:
                        column = pc.cast(column, field.type, safe=True)
                    except (pa.ArrowInvalid,
                            pa.ArrowNotImplementedError) as error:
                        raise StoreError(_message(
                            "{path}: column {column} ({found}) does not fit "
                            "{expected}, the type of the other parts of "
                            "{table}", path=path, column=field.name,
                            found=str(column.type), expected=str(field.type),
                            table=table)) from error
                arrays.append(column)
            out.write_table(pa.Table.from_arrays(arrays, schema=reference),
                            row_group_size=max(1, data.num_rows))
        out.close()
    except BaseException:
        try:
            out.close()
        except Exception:
            pass
        _remove(temporary)
        raise
    _fsync(temporary)
    os.replace(temporary, path)


def _columns_rows(table: str, schema, version) -> List[tuple]:
    """The ``spacr_columns`` rows of one table's schema."""
    pa = _require("pyarrow")
    rows = []
    for position, field in enumerate(schema):
        compute = ("float32" if pa.types.is_float16(field.type)
                   else str(field.type))
        rows.append((table, position, field.name, _dtype_family(field.name),
                     str(field.type), compute, bool(field.nullable),
                     version))
    return rows


def _reconcile(conn, project: str, manifests: List[tuple]) -> Dict[str, Any]:
    """Make every new Parquet part of a table have the table's one schema.

    The reference is the first registered part of the table that this
    batch does not replace, or else the first new part. A new part with
    another schema is rewritten to it when that is lossless and refused
    otherwise. Each entry gets its ``schema_digest``; the reference schema
    of every table touched is returned.
    """
    pq = _require("pyarrow.parquet")
    replaced = set()
    for _path, manifest in manifests:
        for name in (manifest["table"], _keys_name(manifest["table"]),
                     _rows_name(manifest["table"])):
            replaced.add((name, manifest["plate"], manifest["well"]))
    reference: Dict[str, Any] = {}
    for _path, manifest in manifests:
        for entry in manifest["files"]:
            if entry["kind"] != "parquet":
                continue
            name = entry.get("table", manifest["table"])
            full = os.path.join(project, entry["path"])
            schema = pq.read_schema(full).remove_metadata()
            if name not in reference:
                other = [r[0] for r in conn.execute(
                    "SELECT path, plate, well FROM spacr_files WHERE "
                    "table_name = ? AND kind = 'parquet' ORDER BY plate, "
                    "well, path", [name]).fetchall()
                    if (name, r[1], r[2]) not in replaced]
                reference[name] = (
                    pq.read_schema(os.path.join(project, other[0]))
                    .remove_metadata() if other else schema)
            if not _same_columns(schema, reference[name]):
                _conformed_part(full, reference[name], name)
                entry.update(_fingerprint(full))
                schema = reference[name]
            entry["schema_digest"] = _schema_digest(schema)
    return reference


def _stage(manifests: List[tuple]) -> tuple:
    """The manifests that register and the paths of those they replace.

    Of several manifests for one well, the last wins; the earlier ones'
    files are superseded.
    """
    last: Dict[tuple, tuple] = {}
    dropped: List[str] = []
    for path, manifest in manifests:
        key = (manifest["table"], manifest["plate"], manifest["well"])
        if key in last:
            dropped.extend(e["path"] for e in last[key][1]["files"])
        last[key] = (path, manifest)
    keep = list(last.values())
    kept = {e["path"] for _p, m in keep for e in m["files"]}
    return keep, [p for p in dropped if p not in kept]


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
    os.makedirs(project, exist_ok=True)
    with _register_lock(project):
        return _register_locked(project, run, accept_changes)


def _register_locked(project: str, run, accept_changes: bool) -> Dict[str, Any]:
    """:func:`register` while the lock file is held."""
    pa = _require("pyarrow")
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
    tables = sorted({e.get("table", m["table"]) for _p, m in manifests
                     for e in m["files"]} | {m["table"] for _p, m in manifests})
    conn = _connect(os.path.join(project, CATALOG))
    superseded: List[str] = []
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
        reference = _reconcile(conn, project, manifests)
        keep, dropped = _stage(manifests)
        superseded.extend(dropped)
        rows = []
        wells = []
        for _path, manifest in keep:
            conn.execute(
                "INSERT OR IGNORE INTO spacr_tables VALUES (?, ?, ?, ?)",
                [manifest["table"], json.dumps(manifest["sort_by"]),
                 json.dumps(manifest["key_columns"]), _now()])
            for name in sorted({manifest["table"],
                                _keys_name(manifest["table"]),
                                _rows_name(manifest["table"])}):
                wells.append((name, manifest["plate"], manifest["well"]))
            for entry in manifest["files"]:
                rows.append({
                    "path": entry["path"],
                    "table_name": entry.get("table", manifest["table"]),
                    "kind": entry["kind"], "plate": manifest["plate"],
                    "well": manifest["well"], "rows": int(entry["rows"]),
                    "schema_version": int(manifest["schema_version"]),
                    "sha256": entry["sha256"], "bytes": int(entry["bytes"]),
                    "mtime_ns": int(entry["mtime_ns"]),
                    "written_at": manifest["written_at"],
                    "run_id": manifest.get("run_id"),
                    "schema_digest": entry.get("schema_digest"),
                    "key_digest": entry.get("key_digest"),
                    "content_digest": entry["sha256"],
                    "dtype_map_version": manifest.get("dtype_map_version")})
        if rows:
            types = {"VARCHAR": pa.string(), "BIGINT": pa.int64(),
                     "INTEGER": pa.int32()}
            conn.register("spacr_stage_files", pa.table(
                {name: pa.array([r[name] for r in rows], types[kind])
                 for name, kind in _FILE_COLUMNS}))
            conn.register("spacr_stage_wells", pa.table({
                "table_name": pa.array([w[0] for w in wells], pa.string()),
                "plate": pa.array([w[1] for w in wells], pa.string()),
                "well": pa.array([w[2] for w in wells], pa.string())}))
            superseded.extend(r[0] for r in conn.execute(
                "SELECT f.path FROM spacr_files f JOIN spacr_stage_wells s "
                "USING (table_name, plate, well) WHERE f.path NOT IN "
                "(SELECT path FROM spacr_stage_files)").fetchall())
            conn.execute(
                "DELETE FROM spacr_files WHERE EXISTS (SELECT 1 FROM "
                "spacr_stage_wells s WHERE s.table_name = "
                "spacr_files.table_name AND s.plate = spacr_files.plate "
                "AND s.well = spacr_files.well)")
            names = ", ".join(_ident(n) for n, _k in _FILE_COLUMNS)
            conn.execute(f"INSERT OR REPLACE INTO spacr_files ({names}) "
                         f"SELECT {names} FROM spacr_stage_files")
            conn.unregister("spacr_stage_files")
            conn.unregister("spacr_stage_wells")
        for name, schema in reference.items():
            if name.endswith(_KEYS) or name.endswith(_ROWS):
                continue
            version = next((m.get("dtype_map_version") for _p, m in keep
                            if m["table"] == name), None)
            conn.execute("DELETE FROM spacr_columns WHERE table_name = ?",
                         [name])
            conn.executemany(
                "INSERT INTO spacr_columns VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                _columns_rows(name, schema, version))
        _rebuild_views(conn, project, tables)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    _write_version(project)
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


def _arrow_bounded(conn, sql: str, params=None):
    """The result of ``sql`` read in bounded batches, as one Arrow table.

    The scan streams :data:`_QUERY_BATCH` rows at a time, so DuckDB never
    holds the whole result beside the copy kept here; the copy itself is
    capped at :data:`_QUERY_BYTES`.

    :raises StoreError: when the result outgrows the cap (use
        :meth:`Store.batches`).
    """
    pa = _require("pyarrow")
    reader = _reader_of(conn, sql, params, _QUERY_BATCH)
    kept, held = [], 0
    for batch in reader:
        kept.append(batch)
        held += batch.nbytes
        if held > _QUERY_BYTES:
            raise StoreError(_message(
                "the query result is larger than {gigabytes} GB; read it "
                "with Store.batches() or narrow it with features= or where=",
                gigabytes=_QUERY_BYTES // 2 ** 30))
    if not kept:
        return reader.schema.empty_table()
    return pa.Table.from_batches(kept)


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
        self._lazy_legacy: Optional[List[str]] = None
        self._conn = None
        self._files: List[Dict[str, Any]] = []
        self._tables: Dict[str, Dict[str, Any]] = {}
        self._arrays: Dict[str, Any] = {}
        self._views: List[str] = []
        self._described: Dict[str, List[str]] = {}
        self._parts: Dict[str, Any] = {}
        self._column_rows: List[tuple] = []
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
        catalog = os.path.join(self.project, CATALOG)
        if os.path.isfile(self.project) or not os.path.isfile(catalog):
            self._attach_legacy()
            return self
        _check_version(self.project)
        self._conn = _connect(threads=self.threads)
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
            try:
                self._column_rows = source.execute(
                    "SELECT table_name, ordinal, column_name, family, "
                    "stored_type, compute_type, nullable, dtype_map_version "
                    "FROM spacr_columns ORDER BY table_name, ordinal"
                ).fetchall()
            except Exception:
                self._column_rows = []
        finally:
            source.close()
        if verify != "none":
            self.verify(full=verify == "full")
        self._views = sorted({f["table_name"] for f in self._files
                              if f["kind"] == "parquet"})
        self._described = {}
        self._parts = {}
        for name in self._views:
            held = [f for f in self._files
                    if f["table_name"] == name and f["kind"] == "parquet"]
            paths = [os.path.join(self.project, f["path"]) for f in held]
            if paths:
                same = _uniform([f.get("schema_digest") for f in held])
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW {_ident(name)} AS "
                    f"SELECT * FROM {_source(paths, uniform=same)}")
        return self

    def _attach_legacy(self) -> None:
        """Open an older ``measurements.db``, ``.duckdb`` or ``.parquetdb``.

        A SQLite file gets its DuckDB views on first SQL use.
        """
        if _legacy_attach(self):
            return
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
        self._tables = {name: {"sort_by": [], "key_columns": []}
                        for name in names}
        self._lazy_legacy = names

    def _legacy_views(self, names: List[str]) -> None:
        """One DuckDB view per table of the attached SQLite file."""
        path = self.legacy
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

    def tables(self) -> List[str]:
        """The tables this store answers queries on."""
        return sorted(self._tables)

    def columns(self, table: str):
        """The columns of ``table`` as the catalog records them.

        Read from ``spacr_columns``, so no Parquet footer is opened.

        :param table: the table.
        :returns: a DataFrame with ``name``, ``family``, ``stored_type``,
            ``compute_type`` (what a reader computes in: float16 is float32),
            ``nullable`` and ``dtype_map_version``, in stored order; an
            older project without the record gets ``name`` and
            ``stored_type`` read from the view.
        :raises StoreError: when the store has no such table.
        """
        import pandas as pd

        self._known(table)
        rows = [r[2:] for r in self._column_rows if r[0] == table]
        names = ["name", "family", "stored_type", "compute_type", "nullable",
                 "dtype_map_version"]
        if rows:
            return pd.DataFrame(rows, columns=names)
        found = self._table_columns(table)
        frame = pd.DataFrame({"name": found["name"], "family": None,
                              "stored_type": found["type"],
                              "compute_type": found["type"], "nullable": True,
                              "dtype_map_version": None})
        return frame[names]

    def count(self, table: str, where=None) -> int:
        """The number of rows of ``table``, optionally matching ``where``.

        Without ``where`` the registry's row counts are added, so nothing
        is read; with it the narrow key table answers when it holds every
        column the filter names, and the wide table otherwise.

        :param table: the table.
        :param where: a SQL ``WHERE`` clause or a mapping, as for
            :meth:`query`; None counts every row.
        :returns: the row count.
        :raises StoreError: when the store has no such table.
        """
        self._known(table)
        if where is None and not self.legacy:
            held = [f for f in self._files if f["table_name"] == table
                    and f["kind"] == "parquet"]
            if held:
                return int(sum(int(f["rows"]) for f in held))
        duckdb = _require("duckdb")
        clause, params = _where_clause(where)
        keys = _keys_name(table)
        sources = [table]
        if (where is not None and not self.legacy and keys in self._views):
            sources.insert(0, keys)
        for source in sources:
            try:
                return int(self._conn.execute(
                    f"SELECT COUNT(*) FROM {_ident(source)}{clause}",
                    params or None).fetchall()[0][0])
            except duckdb.Error:
                if source == sources[-1]:
                    raise
        return 0

    def page(self, table: str, *, columns=None, where=None, order_by=None,
             after=None, limit: int = 200):
        """One page of ``table`` by keyset, not by offset.

        A page is the rows after a cursor, in key order, so page 1,000 costs
        what page 1 does. Unsorted, the order is plate, well, then the
        table's sort keys, and a page reads one well's files (a well whose
        rows run out continues in the next well). With ``order_by`` the
        order is that column, then plate, well and the sort keys; rows where
        the column is null are not paged. The sort keys must identify a
        row within a well.

        :param table: the table.
        :param columns: the columns to return beside the keys (``plate``,
            ``well`` and the sort keys always come); None returns every
            column.
        :param where: a SQL ``WHERE`` clause or a mapping, as for
            :meth:`query`.
        :param order_by: a column name, or ``(column, "asc" or "desc")``;
            None walks the wells in key order.
        :param after: the cursor a previous call returned; None starts at
            the first row.
        :param limit: the most rows in the page.
        :returns: ``(DataFrame, cursor)``; the cursor is None when no row
            follows.
        :raises StoreError: for an older SQLite project, a column the
            table lacks or a cursor of another table.
        """
        self._known(table)
        if self.legacy:
            raise StoreError(_message(
                "page needs a spaCR store; read an older project with "
                "query()"))
        present = self._columns(table)
        sort = [c for c in (self._tables[table].get("sort_by") or [])
                if c in present] or [c for c in ("object_key",)
                                     if c in present] or [
            c for c in (self._tables[table].get("key_columns") or [])
            if c in present][:1]
        if not sort:
            raise StoreError(_message(
                "{table} has no sort key to page by", table=table))
        base = ["plate", "well", *sort]
        wanted = (list(present) if columns is None
                  else [c for c in columns])
        missing = [c for c in wanted if c not in present]
        if missing:
            raise StoreError(f"{table} has no column {missing[:5]}")
        selected = list(dict.fromkeys([*base, *wanted]))
        text = ", ".join(_ident(c) for c in selected)
        clause, params = _where_clause(where)
        clause = clause.replace(" WHERE ", " AND (", 1) + (")" if clause else "")
        limit = max(int(limit), 0)
        if order_by is None:
            wells = sorted({(f["plate"], f["well"]) for f in self._files
                            if f["table_name"] == table
                            and f["kind"] == "parquet"})
            start, resume = 0, None
            if after is not None:
                try:
                    start = wells.index((after["plate"], after["well"]))
                    resume = list(after["after"])
                except (KeyError, ValueError, TypeError) as error:
                    raise StoreError(_message(
                        "that cursor is not from {table}", table=table)
                    ) from error
            tables, need, last = [], limit, None
            for plate, well in wells[start:]:
                if need <= 0:
                    break
                where_sql, held = "plate = ? AND well = ?", [plate, well]
                if resume is not None and (plate, well) == wells[start]:
                    marks = ", ".join("?" for _ in sort)
                    where_sql += f" AND ({', '.join(_ident(c) for c in sort)}) > ({marks})"
                    held += [_python(v) for v in resume]
                sql = (f"SELECT {text} FROM {_ident(table)} WHERE {where_sql}"
                       f"{clause} ORDER BY "
                       f"{', '.join(_ident(c) for c in sort)} LIMIT {need}")
                got = _arrow_of(self._conn, sql, held + params)
                if got.num_rows:
                    tables.append(got)
                    need -= got.num_rows
                    last = got
            if need > 0 or last is None:
                cursor = None
            else:
                tail = last.slice(last.num_rows - 1, 1).to_pylist()[0]
                cursor = {"plate": tail["plate"], "well": tail["well"],
                          "after": [_python(tail[c]) for c in sort]}
        else:
            column, direction = (order_by if isinstance(order_by, (tuple, list))
                                  else (order_by, "asc"))
            if column not in present:
                raise StoreError(f"{table} has no column {column!r}")
            descending = str(direction).lower() == "desc"
            order = [column, *base]
            selected = list(dict.fromkeys([*order, *wanted]))
            text = ", ".join(_ident(c) for c in selected)
            sign = "<" if descending else ">"
            tail_sql = " DESC" if descending else ""
            where_sql = f"{_ident(column)} IS NOT NULL"
            held: List[Any] = []
            if after is not None:
                try:
                    values = list(after["after"])
                    assert len(values) == len(order)
                except (KeyError, TypeError, AssertionError) as error:
                    raise StoreError(_message(
                        "that cursor is not from {table}", table=table)
                    ) from error
                marks = ", ".join("?" for _ in order)
                where_sql += (f" AND ({', '.join(_ident(c) for c in order)}) "
                              f"{sign} ({marks})")
                held = [_python(v) for v in values]
            sql = (f"SELECT {text} FROM {_ident(table)} WHERE {where_sql}"
                   f"{clause} ORDER BY " + ", ".join(
                       _ident(c) + tail_sql for c in order) +
                   f" LIMIT {limit}")
            got = _arrow_of(self._conn, sql, held + params)
            tables = [got]
            cursor = None
            if got.num_rows >= limit and got.num_rows:
                tail = got.slice(got.num_rows - 1, 1).to_pylist()[0]
                cursor = {"after": [_python(tail[c]) for c in order]}
        pa = _require("pyarrow")
        whole = (pa.concat_tables(tables) if tables
                 else _arrow_of(self._conn, f"SELECT {text} FROM "
                                f"{_ident(table)} LIMIT 0"))
        return _upcast_half(whole).to_pandas(split_blocks=True), cursor

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
        return list(dict.fromkeys(wanted + list(features)))

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
            located = _arrow_of(self._conn,
                                sql + f" LIMIT {_LOCATE_MAX + 1}", params)
        except duckdb.Error:
            return None
        if located.num_rows > _LOCATE_MAX:
            return None
        columns = (self._columns(table) if features is None
                   else self._wanted(table, table, features))
        copies = {f["path"] for f in self._files
                  if f["table_name"] == _rows_name(table)}
        small = located.num_rows <= limit
        if (features is None and small and located.num_rows and all(
                _row_copy_path(p) in copies
                for p in set(located.column("_part").to_pylist()))):
            sql, params = self._select(where, table, None, source=keys)
            keyed = _arrow_of(self._conn, sql, params)
            return self._read_row_copies(keyed, columns)
        if not located.num_rows:
            return _arrow_of(self._conn, "SELECT " + ", ".join(
                _ident(c) for c in columns) + f" FROM {_ident(table)} LIMIT 0")
        if small and len(columns) <= _NARROW_COLUMNS:
            return self._read_located(located, columns)
        parts = located.column("_part").to_pylist()
        touched = len(set(zip(parts, located.column("_row_group").to_pylist())))
        per_group = max(1, self._part(parts[0]).metadata.row_group(0).num_rows)
        if touched > _TOUCHED_GROUPS * max(1.0, total / per_group):
            return None
        return self._read_located(located, columns)

    def _read_row_copies(self, keyed, columns: List[str]):
        """Every column of the located rows: keys from the key table and the
        float columns from the row-major copies, cast back to their types."""
        pa = _require("pyarrow")
        parts = np.asarray(keyed.column("_part").to_pylist(), dtype=object)
        groups = keyed.column("_row_group").to_numpy().astype(np.int64)
        rows = keyed.column("_group_row").to_numpy().astype(np.int64)
        float_columns: Dict[str, Any] = {}
        types: Dict[str, Any] = {}
        for rel in sorted(set(parts)):
            mine = np.flatnonzero(parts == rel)
            handle = self._part(rel)
            starts = np.concatenate([[0], np.cumsum([
                handle.metadata.row_group(g).num_rows
                for g in range(handle.metadata.num_row_groups)])])
            copy = os.path.join(self.project, _row_copy_path(rel))
            if copy not in self._arrays:
                with builtins.open(os.path.join(copy, "columns.json"),
                                   encoding="utf-8") as handle_names:
                    names = json.load(handle_names)
                array = (_ChunkReader.open(os.path.join(copy, "features"),
                                           self.threads)
                         or _require("zarr").open_array(
                             os.path.join(copy, "features"), mode="r"))
                self._arrays[copy] = (names, array)
            names, array = self._arrays[copy]
            matrix = _read_rows(array, starts[groups[mine]] + rows[mine])
            schema = handle.schema_arrow
            for position, name in enumerate(names):
                types.setdefault(name, schema.field(name).type)
                column = float_columns.setdefault(
                    name, np.full(len(parts), np.nan, dtype=matrix.dtype))
                column[mine] = matrix[:, position]
        present = {name: i for i, name in enumerate(keyed.column_names)}
        arrays = []
        for name in columns:
            if name in float_columns:
                arrays.append(pa.array(float_columns[name], from_pandas=True
                                       ).cast(types[name]))
            elif name in present:
                arrays.append(keyed.column(present[name]))
            else:
                arrays.append(pa.nulls(keyed.num_rows))
        return pa.Table.from_arrays(arrays, names=list(columns))

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
            index = {name: i for i, name in enumerate(piece.column_names)}
            arrays = [piece.column(index[name]) if name in index
                      else pa.nulls(piece.num_rows) for name in columns]
            pieces.append(pa.Table.from_arrays(arrays, names=list(columns)))
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
        clause, params = _where_clause(where)
        sql += clause
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
            result = _arrow_bounded(self._conn, sql, params)
        return self._shaped(result, table, features, crops, as_)

    def sql(self, statement: str, params=None, *, as_: str = "pandas"):
        """Run raw SQL against the views.

        :param statement: the SQL.
        :param params: its ``?`` parameters.
        :param as_: ``"pandas"`` or ``"arrow"``.
        :returns: the result.
        """
        result = _upcast_half(_arrow_of(self._conn, statement, params))
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
        """Convert an Arrow result to ``as_`` and attach the crops.

        A float16 column is returned as float32, so nothing computes on
        half precision.
        """
        result = _upcast_half(result)
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
        self._lazy_legacy = None
        if self._duck is not None:
            self._duck.close()
            self._duck = None

    def __enter__(self) -> "Store":
        """The store itself, for ``with``."""
        return self

    def __exit__(self, *exc) -> None:
        """Close on leaving ``with``."""
        self.close()

    @property
    def _conn(self):
        """The in-memory DuckDB connection.

        An older SQLite file gets it, and its views, on first use, so a
        caller that only reads whole tables never starts DuckDB.
        """
        if self._duck is None and self._lazy_legacy is not None:
            names, self._lazy_legacy = self._lazy_legacy, None
            self._duck = _connect(threads=self.threads)
            self._legacy_views(names)
        return self._duck

    @_conn.setter
    def _conn(self, value) -> None:
        """Hold ``value`` as the DuckDB connection."""
        self._duck = value

    def _known(self, table: str) -> None:
        """Refuse a table this store does not hold."""
        if table not in self._tables:
            raise StoreError(f"no table {table!r}; this store holds "
                             f"{self.tables()}")

    def _table_columns(self, table: str):
        """The columns of one table, in stored order.

        :param table: the table.
        :returns: a DataFrame with ``name`` and ``type`` (the stored type).
        :raises StoreError: when the store has no such table.
        """
        import pandas as pd

        self._known(table)
        if _legacy_is_sqlite(self):
            rows = _legacy_sqlite_columns(self.legacy, table)
        else:
            rows = [(row[0], row[1]) for row in self._conn.execute(
                f"DESCRIBE SELECT * FROM {_ident(table)}").fetchall()]
        return pd.DataFrame(rows, columns=["name", "type"])

    def _frames(self, table: str, *, limit: Optional[int] = None,
                chunksize: int = 100_000, **sqlite_kwargs) -> Iterator[Any]:
        """One table as stored, in pandas frames of at most ``chunksize`` rows.

        At least one frame is yielded, empty with the table's columns when
        there are no rows. An older SQLite file is read with SQLite's own
        value types, as ``pandas.read_sql_query`` gives them.

        :param table: the table.
        :param limit: read at most this many rows; None reads every row.
        :param chunksize: rows per frame.
        :param sqlite_kwargs: passed to ``pandas.read_sql_query`` for an
            older SQLite file; ignored otherwise.
        :returns: an iterator over DataFrames.
        :raises StoreError: when the store has no such table.
        """
        self._known(table)
        if _legacy_is_sqlite(self):
            yield from _legacy_sqlite_frames(
                self.legacy, table, limit=limit, chunksize=chunksize,
                **sqlite_kwargs)
            return
        sql = f"SELECT * FROM {_ident(table)}"
        if limit is not None:
            sql += f" LIMIT {max(int(limit), 0)}"
        reader = _reader_of(self._conn, sql, None, int(chunksize))
        yielded = False
        for batch in reader:
            yielded = True
            yield _upcast_half(_table_of([batch])).to_pandas(
                split_blocks=True)
        if not yielded:
            yield reader.schema.empty_table().to_pandas()


def _upcast_half(result):
    """``result`` with every float16 column cast to float32."""
    pa = _require("pyarrow")
    if not any(pa.types.is_float16(f.type) for f in result.schema):
        return result
    fields = [pa.field(f.name, pa.float32(), f.nullable, f.metadata)
              if pa.types.is_float16(f.type) else f for f in result.schema]
    return result.cast(pa.schema(fields, metadata=result.schema.metadata))


def _drop_tables(project, tables: Sequence[str]) -> List[str]:
    """Remove ``tables`` from a store: their catalog rows, views and files.

    For a writer that rebuilds its tables on every run, so the rows of a
    well the new run does not write cannot survive from an older run, and
    a waiting manifest of a killed run cannot be registered later. Their
    key tables and row-major copies go too; other tables are untouched.

    :param project: the project folder.
    :param tables: the table names.
    :returns: the paths removed, relative to the project.
    """
    project = os.path.abspath(os.fspath(project))
    names = sorted({n for t in tables
                    for n in (t, _keys_name(t), _rows_name(t))})
    removed: List[str] = []
    catalog = os.path.join(project, CATALOG)
    if os.path.isfile(catalog):
        conn = _connect(catalog)
        try:
            marks = ", ".join("?" for _ in names)
            conn.execute("BEGIN TRANSACTION")
            removed = [row[0] for row in conn.execute(
                f"SELECT path FROM spacr_files WHERE table_name IN ({marks})",
                names).fetchall()]
            conn.execute(f"DELETE FROM spacr_files WHERE table_name IN "
                         f"({marks})", names)
            conn.execute(f"DELETE FROM spacr_tables WHERE name IN ({marks})",
                         names)
            for name in names:
                conn.execute(f"DROP VIEW IF EXISTS {_ident(name)}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
    for path in sorted(glob.glob(os.path.join(glob.escape(project), _PENDING,
                                              "*.json"))):
        with builtins.open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("table") in names:
            removed.extend(entry["path"] for entry in manifest["files"])
            _remove(path)
    for rel in removed:
        _remove(os.path.join(project, rel))
    for name in names:
        _remove(os.path.join(project, "tables", name))
        _remove(os.path.join(project, "rows", name))
    return removed


def _dtype_family(column: str, channels: Sequence[str] = (), *,
                  literal_numbers: bool = False) -> str:
    """The column family of a measurement column, for reporting dtype evidence.

    The dtype map is per column (:func:`_stored_types`); the family only
    groups the evidence and the equivalence report. The name is split at
    ``_``; a leading object class (``cell``, ``nucleus``, ``cytoplasm``,
    ``pathogen``) becomes ``{compartment}``, a channel of ``channels``
    becomes ``{channel}`` and, unless ``literal_numbers``, a whole number
    becomes ``{n}``: so ``nucleus_gfp_percentile_95`` is
    ``{compartment}_{channel}_percentile_{n}``.

    :param column: the column name.
    :param channels: the run's channel names, any case.
    :param literal_numbers: keep whole numbers as they are.
    :returns: the family.
    """
    names = {str(c).lower() for c in channels}
    tokens = []
    for position, token in enumerate(str(column).lower().split("_")):
        if position == 0 and token in _COMPARTMENTS:
            tokens.append("{compartment}")
        elif token in names:
            tokens.append("{channel}")
        elif token.isdigit() and not literal_numbers:
            tokens.append("{n}")
        else:
            tokens.append(token)
    return "_".join(tokens)


def _dtype_map(path: Optional[str] = None) -> Dict[str, Any]:
    """A recorded dtype map, read once: ``{"columns": {column: entry}}``.

    Each entry names the dtype that column is stored in, the source dtype
    it was measured from and the evidence of the four criteria for it
    (``tools/measure_store_dtypes.py``). None reads the shipped map
    (:data:`_DTYPE_MAP_PATH`, looked up at the call).
    """
    return _read_dtype_map(os.path.abspath(path or _DTYPE_MAP_PATH))


@functools.lru_cache(maxsize=8)
def _read_dtype_map(path: str) -> Dict[str, Any]:
    """The dtype map at ``path``, parsed once."""
    with builtins.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _fitted_dtype(dtype: str, kind: str, low, high) -> str:
    """``dtype``, or the next one up its ladder that holds ``[low, high]``.

    :param dtype: the recorded dtype.
    :param kind: ``"float"`` or ``"int"``.
    :param low: the column's smallest value, None when no statistics were
        written, NaN when every value is NaN.
    :param high: its largest.
    :returns: the dtype to store the column in.
    """
    if kind == "float":
        if dtype != "float16":
            return dtype
        if low is None or high is None:
            return "float32"
        if np.isnan(low) and np.isnan(high):
            return dtype
        limit = max(abs(float(low)), abs(float(high)))
        return dtype if limit <= float(np.finfo(np.float16).max) else "float32"
    if low is None or high is None:
        return "int64"
    if np.isnan(low) and np.isnan(high):
        return dtype
    start = _INT_LADDER.index(dtype) if dtype in _INT_LADDER else 0
    for name in _INT_LADDER[start:]:
        info = np.iinfo(name)
        if info.min <= low and high <= info.max:
            return name
    return "int64"


def _column_ranges(meta) -> Dict[str, tuple]:
    """``column -> (min, max)`` over a Parquet file's row-group statistics.

    NaN is not counted; a column whose every value is NaN gives
    ``(nan, nan)`` and one without statistics ``(None, None)``.

    :param meta: a ``pyarrow.parquet.FileMetaData``.
    """
    ranges: Dict[str, tuple] = {}
    for group in range(meta.num_row_groups):
        row_group = meta.row_group(group)
        for index in range(row_group.num_columns):
            chunk = row_group.column(index)
            name = chunk.path_in_schema
            stats = chunk.statistics
            low, high = ranges.get(name, (np.nan, np.nan))
            if stats is None:
                ranges[name] = (None, None)
                continue
            if low is None:
                continue
            if stats.has_min_max:
                if not isinstance(stats.min, (int, float, np.number)):
                    continue
                low = float(np.fmin(low, float(stats.min)))
                high = float(np.fmax(high, float(stats.max)))
            ranges[name] = (low, high)
    return ranges


def _stored_types(schema, ranges, channels: Sequence[str] = (), *,
                  path: Optional[str] = None):
    """The Arrow type each numeric column of a table is stored in.

    PER COLUMN: the dtype the recorded map holds for that column name, each
    column measured on its own (``tools/measure_store_dtypes.py``), never
    wider than the column already is, moved up its ladder when the
    table's own range does not fit (:func:`_fitted_dtype`). A column the
    map does not hold keeps its type; an integer column the map measured
    as floats takes the narrowest integer type its range fits, and a float
    column it measured as integers keeps its type. Readers get float16
    columns back as float32 (:meth:`Store.query`).

    :param schema: the source table's Arrow schema.
    :param ranges: :func:`_column_ranges` of the source.
    :param channels: the run's channel names (unused; kept for callers).
    :param path: the dtype map; None is the shipped one.
    :returns: ``({column: pyarrow type}, promotions)``; each promotion
        names the column, the recorded dtype and the one used.
    """
    pa = _require("pyarrow")
    recorded = _dtype_map(path).get("columns") or {}
    types: Dict[str, Any] = {}
    promoted: List[Dict[str, Any]] = []
    for field in schema:
        if pa.types.is_floating(field.type):
            kind = "float"
        elif pa.types.is_integer(field.type):
            kind = "int"
        else:
            continue
        source = np.dtype(field.type.to_pandas_dtype()).name
        entry = recorded.get(field.name)
        chosen = source
        if entry and entry.get("kind") == kind:
            chosen = entry["dtype"]
        elif entry and kind == "int":
            chosen = _INT_LADDER[0]
        if np.dtype(chosen).itemsize > np.dtype(source).itemsize:
            chosen = source
        low, high = ranges.get(field.name, (None, None))
        fitted = _fitted_dtype(chosen, kind, low, high)
        if fitted != chosen:
            promoted.append({"column": field.name, "recorded": chosen,
                             "stored": fitted})
        types[field.name] = pa.from_numpy_dtype(np.dtype(fitted))
    return types, promoted


#: The sequence stamp of the edit log is two columns, ``seq_ns`` (int64
#: nanoseconds since the epoch, about 1.7e18, under the int64 maximum 9.2e18)
#: and ``seq_writer`` (int32), ordered lexicographically. A single
#: ``ns * 1000 + writer`` would be about 1.7e21 and overflow int64.
_EDIT_COLUMNS = (
    ("table", "string"), ("object_key", "int64"), ("timeID", "int64"),
    ("column", "string"), ("value_num", "float64"),
    ("value_text", "string"), ("op", "string"), ("seq_ns", "int64"),
    ("seq_writer", "int32"), ("author", "string"), ("source", "string"),
    ("session", "string"), ("run_id", "string"))

_EDIT_LAST: Dict[int, int] = {}


def _edit_schema():
    """The Arrow schema of one edit-log file (see :data:`_EDIT_COLUMNS`)."""
    pa = _require("pyarrow")
    return pa.schema([pa.field(name, getattr(pa, kind)())
                      for name, kind in _EDIT_COLUMNS])


def _edit_stamp(writer: int = 0) -> tuple:
    """The next ``(seq_ns, seq_writer)`` of ``writer``, strictly increasing.

    Order edits by ``seq_ns`` then ``seq_writer``; a later edit of one
    writer always sorts after its earlier ones, even when the clock stalls
    or steps back.
    """
    writer = int(writer)
    now = time.time_ns()
    last = _EDIT_LAST.get(writer, 0)
    stamp = now if now > last else last + 1
    _EDIT_LAST[writer] = stamp
    return stamp, writer


def _where_clause(where) -> tuple:
    """`` WHERE ...`` and its parameters for a mapping or SQL ``where``."""
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
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params
    if where:
        return f" WHERE {where}", params
    return "", params


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
    return _legacy_sqlite_tables(path)


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



#: Suffixes of a DuckDB measurement copy.
_LEGACY_DUCKDB_SUFFIXES: tuple[str, ...] = (".duckdb", ".ddb")

#: Suffix of a Parquet-folder measurement copy.
_LEGACY_PARQUET_SUFFIX = ".parquetdb"


def _legacy_kind(path: Any) -> str:
    """Which older format a path holds: ``'parquet'``, ``'duckdb'`` or
    ``'sqlite'``.

    :param path: a file or folder path.
    :returns: the format name; anything that is neither a ``.parquetdb``
        folder nor a ``.duckdb``/``.ddb`` file is SQLite.
    """
    text = os.fspath(path).rstrip("/\\")
    suffix = os.path.splitext(text)[1].lower()
    if suffix == _LEGACY_PARQUET_SUFFIX:
        return "parquet"
    if (suffix in _LEGACY_DUCKDB_SUFFIXES and os.path.basename(text) != CATALOG
            and not os.path.isdir(text)):
        return "duckdb"
    return "sqlite"


def _legacy_is_sqlite(store) -> bool:
    """Whether an open :class:`spacr.store.Store` reads an older SQLite file.

    :param store: the store.
    :returns: True for a SQLite file, False for a store project or another
        older format.
    """
    return store.legacy is not None and _legacy_kind(store.legacy) == "sqlite"


def _legacy_sqlite_connection(path: Any, *, read_only: bool = True):
    """Open a SQLite file the way spaCR reads and writes measurements.

    Inside an atomic write packet for the same file, the packet's own
    connection is returned, so the caller sees its uncommitted rows.

    :param path: the SQLite file.
    :param read_only: open through ``file:...?mode=ro``.
    :returns: a :class:`sqlite3.Connection`.
    """
    import sqlite3

    from .database_concurrency import _inside_write_packet, connect

    if read_only:
        return connect(path, readonly=True)
    if _inside_write_packet(path):
        return connect(path)
    return sqlite3.connect(path, timeout=30)


@contextmanager
def _legacy_reading(path: str):
    """A read-only connection, closed afterwards unless a write packet owns it."""
    from .database_concurrency import _inside_write_packet

    conn = _legacy_sqlite_connection(path, read_only=True)
    try:
        yield conn
    finally:
        if not _inside_write_packet(path):
            conn.close()


def _legacy_quoted(name: str) -> str:
    """A SQLite identifier in double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _legacy_sqlite_tables(path: str) -> List[str]:
    """The table names of a SQLite file, without SQLite's own tables.

    :param path: the SQLite file.
    :returns: the names, sorted.
    """
    with _legacy_reading(path) as conn:
        return [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name "
            "NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]


def _legacy_sqlite_columns(path: str, table: str) -> List[tuple[str, str]]:
    """The columns of one SQLite table, with their declared types.

    :param path: the SQLite file.
    :param table: the table.
    :returns: ``(name, declared type)`` pairs in stored order.
    """
    with _legacy_reading(path) as conn:
        return [(str(row[1]), str(row[2] or "")) for row in conn.execute(
            f"PRAGMA table_info({_legacy_quoted(table)})").fetchall()]


def _legacy_sqlite_frames(path: str, table: str, *, limit: Optional[int] = None,
                  chunksize: int = 100_000, **kwargs) -> Iterator[Any]:
    """One SQLite table in pandas frames, with SQLite's own value types.

    :param path: the SQLite file.
    :param table: the table.
    :param limit: read at most this many rows; None reads every row.
    :param chunksize: rows per frame.
    :param kwargs: passed to :func:`pandas.read_sql_query`.
    :returns: an iterator yielding at least one frame.
    """
    import pandas as pd

    quoted = _legacy_quoted(table)
    query = f"SELECT * FROM {quoted}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _legacy_reading(path) as conn:
        yielded = False
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize,
                                       **kwargs):
            yielded = True
            yield chunk
        if not yielded:
            yield pd.read_sql_query(f"SELECT * FROM {quoted} LIMIT 0", conn,
                                    **kwargs)


def _legacy_parquet_tables(folder: str) -> List[str]:
    """The tables of a ``.parquetdb`` folder that hold at least one part.

    A folder is read as an escaped table name only when it is exactly the
    Windows spelling of a name Windows forbids.

    :param folder: the ``.parquetdb`` folder.
    :returns: the names, sorted.
    """
    from .tabular import (_parquet_parts, _parquet_table_name,
                          _parquet_verbatim_folder)

    if not os.path.isdir(folder):
        return []
    names = set()
    for entry in os.listdir(folder):
        table = _parquet_table_name(entry)
        if (table != entry and not _parquet_verbatim_folder(folder, table)
                and _parquet_parts(folder, table)):
            names.add(table)
        elif _parquet_parts(folder, entry):
            names.add(entry)
    return sorted(names)


def _legacy_attach(store) -> bool:
    """Give an open store the views of a ``.duckdb`` file or a ``.parquetdb``
    folder.

    :param store: the :class:`spacr.store.Store` being refreshed.
    :returns: True when ``store.project`` is one of those formats (attached,
        or absent and so empty), False for SQLite, which the store attaches
        itself.
    """
    path = store.project
    found = _legacy_kind(path)
    if found == "sqlite":
        return False
    store.legacy = path
    names: List[str] = []
    if found == "parquet":
        from .tabular import _parquet_parts

        names = _legacy_parquet_tables(path)
        store._conn = _connect(threads=store.threads)
        for name in names:
            files = ", ".join(_literal(part)
                              for part in _parquet_parts(path, name))
            store._conn.execute(
                f"CREATE VIEW {_ident(name)} AS SELECT * FROM "
                f"read_parquet([{files}], union_by_name = true)")
    else:
        store._conn = _connect(threads=store.threads)
    if found == "duckdb" and os.path.isfile(path):
        store._conn.execute(f"ATTACH {_literal(path)} AS legacy (READ_ONLY)")
        names = sorted(row[0] for row in store._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = 'legacy' AND table_schema = 'main' "
            "AND table_type = 'BASE TABLE'").fetchall())
        for name in names:
            store._conn.execute(
                f"CREATE VIEW {_ident(name)} AS SELECT * FROM "
                f"legacy.{_ident(name)}")
    store._tables = {name: {"sort_by": [], "key_columns": []}
                     for name in names}
    return True


import threading as _threading

#: The read side for the GUI (unified-store step S6) starts here, one block
#: apart from the edit side: the shared Store per project, a per-thread twin,
#: the project locator, distinct values for filter lists and the export.
_SHARED_VERSION = 1

__all__ += ["locate", "shared", "release_shared", "fork", "distinct",
            "legacy_connection", "export", "interrupt"]

#: The open store of each project this process reads, keyed by its path.
_SHARED: Dict[str, Dict[str, Any]] = {}

#: Guards :data:`_SHARED`.
_SHARED_LOCK = _threading.Lock()

#: Seconds between two checks of a shared project's catalog on disk.
_SHARED_CHECK = 2.0

#: The most columns SQLite takes in one table; an export splits wider ones.
_SQLITE_COLUMNS = 2000


def locate(path) -> str:
    """The path :func:`open` takes, for a file or folder a user picked.

    Accepted are a store project folder (holding ``spacr.duckdb``), the
    ``spacr.duckdb`` file itself, a run folder whose ``measurements``
    folder is a store project, an older ``measurements.db`` (or
    ``.duckdb``, ``.parquetdb``) file, and a run or ``measurements``
    folder holding ``measurements.db``. A store wins over an older file
    in the same folder.

    :param path: the file or folder.
    :returns: the absolute path of the project folder or the older file.
    :raises FileNotFoundError: when nothing there opens.
    """
    if path is None or not str(path).strip():
        raise FileNotFoundError("no project given")
    found = os.path.abspath(os.path.expanduser(str(path).strip()))
    if os.path.isfile(found):
        if os.path.basename(found) == CATALOG:
            return os.path.dirname(found)
        return found
    if os.path.isdir(found):
        if found.rstrip("/\\").lower().endswith(_LEGACY_PARQUET_SUFFIX):
            return found
        for candidate in (found, os.path.join(found, "measurements")):
            if os.path.isfile(os.path.join(candidate, CATALOG)):
                return candidate
        for candidate in (os.path.join(found, "measurements",
                                       "measurements.db"),
                          os.path.join(found, "measurements.db")):
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"{found} holds neither {CATALOG} nor measurements.db")
    raise FileNotFoundError(f"no such file or folder: {found}")


def _shared_stamp(project: str) -> Optional[tuple]:
    """Size and mtime of what a shared store was opened from."""
    target = project
    if os.path.isdir(project):
        catalog = os.path.join(project, CATALOG)
        target = catalog if os.path.isfile(catalog) else project
    found = _stat(target)
    return None if found is None else (found["bytes"], found["mtime_ns"])


def shared(project, *, verify: str = "quick") -> Store:
    """The one open :class:`Store` of ``project`` in this process.

    Every screen reading the same project gets the same store, so the
    registry, the Parquet footers and the views are read once. The catalog
    (or the older file) is checked at most every two seconds; when it
    changed, the next call opens the project again. The store replaced
    then is not closed, so a query still running on it finishes.

    The store's connection belongs to the thread that asked first; a
    worker thread reads through :func:`fork`.

    :param project: anything :func:`locate` accepts.
    :param verify: as for :func:`open`, used when the store is opened.
    :returns: the store.
    :raises FileNotFoundError: when nothing there opens.
    :raises StoreError: when the project does not open.
    """
    key = locate(project)
    now = time.monotonic()
    with _SHARED_LOCK:
        held = _SHARED.get(key)
        if held is not None and now - held["checked"] < _SHARED_CHECK:
            return held["store"]
        stamp = _shared_stamp(key)
        if held is not None and held["stamp"] == stamp:
            held["checked"] = now
            return held["store"]
        opened = Store(key, verify=verify)
        _SHARED[key] = {"store": opened, "stamp": stamp, "checked": now}
        return opened


def release_shared(project=None) -> None:
    """Close the shared store of ``project``, or of every project.

    :param project: anything :func:`locate` accepts; None closes all.
    """
    with _SHARED_LOCK:
        if project is None:
            keys = list(_SHARED)
        else:
            try:
                keys = [locate(project)]
            except FileNotFoundError:
                keys = [os.path.abspath(os.fspath(project))]
        for key in keys:
            held = _SHARED.pop(key, None)
            if held is not None:
                held["store"].close()


def fork(store: Store) -> Store:
    """A twin of ``store`` with its own DuckDB cursor, for another thread.

    The twin shares the registry, the views and the cached file handles,
    so it costs a cursor, not an open. Closing it closes only its cursor;
    ``twin.interrupt()`` stops the query it is running.

    :param store: an open store.
    :returns: the twin.
    """
    import copy

    twin = copy.copy(store)
    twin._lazy_legacy = None
    base = store._conn
    twin._duck = base.cursor() if base is not None else None
    return twin


def distinct(store: Store, table: str, column: str, *, where=None,
             limit: int = 1000) -> List[Any]:
    """The distinct values of one column, for a filter's drop-down list.

    A store reads them from the narrow key table when it holds the
    column, so a million-row table answers in milliseconds; an older
    SQLite file answers with SQLite itself. Nulls are left out.

    :param store: an open store.
    :param table: the table.
    :param column: the column.
    :param where: a SQL ``WHERE`` clause or a mapping, as for
        :meth:`Store.query`.
    :param limit: the most values returned.
    :returns: the values, sorted.
    :raises StoreError: when the table or the column is unknown.
    """
    store._known(table)
    names = list(store._table_columns(table)["name"])
    if column not in names:
        raise StoreError(f"{table} has no column {column!r}")
    limit = max(int(limit), 0)
    if _legacy_is_sqlite(store):
        quoted = _legacy_quoted(column)
        sql = (f"SELECT DISTINCT {quoted} FROM {_legacy_quoted(table)} "
               f"WHERE {quoted} IS NOT NULL")
        clause, params = _where_clause(where)
        if clause:
            sql += " AND (" + clause[len(" WHERE "):] + ")"
        sql += f" ORDER BY {quoted} LIMIT {limit}"
        with _legacy_reading(store.legacy) as conn:
            return [row[0] for row in conn.execute(sql, params).fetchall()]
    source = table
    keys = _keys_name(table)
    clause, params = _where_clause(where)
    if not store.legacy and keys in store._views:
        held = set(store._columns(keys))
        if column in held and (where is None or (
                isinstance(where, Mapping) and set(where) <= held)):
            source = keys
    sql = (f"SELECT DISTINCT {_ident(column)} FROM {_ident(source)} WHERE "
           f"{_ident(column)} IS NOT NULL")
    if clause:
        sql += " AND (" + clause[len(" WHERE "):] + ")"
    sql += f" ORDER BY 1 LIMIT {limit}"
    result = _upcast_half(_arrow_of(store._conn, sql, params or None))
    return [_python(v) for v in result.column(0).to_pylist()]


@contextmanager
def legacy_connection(store: Store):
    """A read-only SQLite connection to an older project's ``measurements.db``.

    For readers that need SQLite's own dialect and ``rowid`` (the
    Database Browser pages an older file by it). The connection is
    opened ``mode=ro`` and closed on leaving the block.

    :param store: a store opened on an older SQLite file.
    :returns: a context manager yielding the connection.
    :raises StoreError: when the store is not an older SQLite file.
    """
    if not _legacy_is_sqlite(store):
        raise StoreError("legacy_connection needs an older SQLite project")
    with _legacy_reading(store.legacy) as conn:
        yield conn


def _export_select(store: Store, table: str, columns, where, params,
                   order_by) -> tuple:
    """The SELECT of one export, in the dialect of the store's reader."""
    present = list(store._table_columns(table)["name"])
    wanted = present if not columns else list(columns)
    missing = [c for c in wanted if c not in present]
    if missing:
        raise StoreError(f"{table} has no column {missing[:5]}")
    sqlite = _legacy_is_sqlite(store)
    quote = _legacy_quoted if sqlite else _ident
    sql = (f"SELECT {', '.join(quote(c) for c in wanted)} FROM "
           f"{quote(table)}")
    clause, held = _where_clause(where)
    sql += clause
    order = []
    if order_by is not None:
        column, direction = (order_by if isinstance(order_by, (tuple, list))
                             else (order_by, "asc"))
        if column not in present:
            raise StoreError(f"{table} has no column {column!r}")
        order.append(quote(column) + (" DESC" if str(direction).lower()
                                      == "desc" else ""))
    if not store.legacy:
        sort = [c for c in (store._tables[table].get("sort_by") or [])
                if c in present]
        order += [quote(c) for c in ("plate", "well", *sort) if c in present]
    if order:
        sql += " ORDER BY " + ", ".join(order)
    return wanted, sql, list(held) + list(params or ())


def _export_batches(store: Store, table: str, columns, where, params,
                    order_by, batch_size: int):
    """The rows of one export as Arrow tables, never the whole result."""
    pa = _require("pyarrow")
    wanted, sql, held = _export_select(store, table, columns, where, params,
                                       order_by)
    if _legacy_is_sqlite(store):
        import pandas as pd

        declared = dict(_legacy_sqlite_columns(store.legacy, table))
        types = {"INTEGER": pa.int64(), "REAL": pa.float64(),
                 "TEXT": pa.string(), "BLOB": pa.binary()}
        schema = pa.schema([(c, types.get(_affinity(declared.get(c, "")),
                                          pa.string())) for c in wanted])
        with _legacy_reading(store.legacy) as conn:
            for frame in pd.read_sql_query(sql, conn, params=held or None,
                                           chunksize=batch_size):
                yield _conformed(frame, schema)
        return
    reader = _reader_of(store._conn, sql, held or None, int(batch_size))
    for batch in reader:
        yield _upcast_half(pa.Table.from_batches([batch]))


def _affinity(declared: str) -> str:
    """SQLite's type affinity of a declared column type."""
    text = str(declared or "").upper()
    if "INT" in text:
        return "INTEGER"
    if "CHAR" in text or "CLOB" in text or "TEXT" in text:
        return "TEXT"
    if "BLOB" in text:
        return "BLOB"
    if "REAL" in text or "FLOA" in text or "DOUB" in text:
        return "REAL"
    return "NUMERIC"


def _conformed(frame, schema):
    """A pandas chunk of an older file as Arrow in one schema.

    A column whose values do not fit its declared type (SQLite stores
    what it is given) is written as text, never dropped.
    """
    pa = _require("pyarrow")
    arrays = []
    for field in schema:
        values = frame[field.name] if field.name in frame else None
        try:
            arrays.append(pa.array(values, type=field.type,
                                   from_pandas=True))
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
            arrays.append(pa.array(
                [None if v is None or v != v else str(v) for v in values],
                type=pa.string()))
    return pa.table(arrays, names=[f.name for f in schema])


def _csv_header(path: str, names) -> None:
    """Write a CSV header line the way Python's csv module quotes it."""
    import csv

    with builtins.open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(list(names))


def _sqlite_type(kind) -> str:
    """The declared SQLite type an Arrow type is exported as."""
    pa = _require("pyarrow")
    if pa.types.is_boolean(kind) or pa.types.is_integer(kind):
        return "INTEGER"
    if pa.types.is_floating(kind) or pa.types.is_decimal(kind):
        return "REAL"
    if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
        return "BLOB"
    return "TEXT"


def _export_group(name: str, family: Optional[str], table: str) -> str:
    """The family group a column is exported under when a table is split."""
    text = str(family or name)
    text = text.replace("{compartment}_", "")
    if text.startswith(f"{table}_"):
        text = text[len(table) + 1:]
    head = text.split("_", 1)[0].strip("{}") or "other"
    return "".join(ch if ch.isalnum() else "_" for ch in head.lower())


def _export_split(store: Store, table: str, wanted: List[str]) -> List[tuple]:
    """``(sqlite table, columns)`` pairs for one exported table.

    One table when its columns fit SQLite; otherwise one table per feature
    family (the channel or measurement named first), each repeating the
    key columns, so every split table joins back on them.
    """
    if len(wanted) < _SQLITE_COLUMNS:
        return [(table, wanted)]
    keys = [c for c in ["plate", "well",
                        *(store._tables.get(table, {}).get("key_columns")
                          or [])] if c in wanted]
    if not keys and "rowid" not in wanted:
        keys = wanted[:1]
    families = {}
    try:
        recorded = store.columns(table)
        families = dict(zip(recorded["name"], recorded["family"]))
    except StoreError:
        pass
    groups: Dict[str, List[str]] = {}
    for column in wanted:
        if column in keys:
            continue
        groups.setdefault(_export_group(column, families.get(column), table),
                          []).append(column)
    room = _SQLITE_COLUMNS - 1 - len(keys)
    pieces = []
    for group, members in groups.items():
        chunks = [members[i:i + room] for i in range(0, len(members), room)]
        for number, chunk in enumerate(chunks, 1):
            suffix = group if len(chunks) == 1 else f"{group}_{number}"
            pieces.append((f"{table}__{suffix}", keys + chunk))
    return pieces


def export(project, target, *, fmt: str, tables=None, where=None,
           params=None, features=None, order_by=None,
           batch_size: int = 65536) -> Dict[str, Any]:
    """Stream tables of a project to a CSV, Parquet or SQLite file.

    Rows are read in batches of ``batch_size`` and written as they come,
    so memory stays bounded whatever the table's size. CSV and Parquet
    take one table; Parquet keeps the stored types. SQLite takes any
    number of tables; a table with 2,000 columns or more is split by
    feature family into ``<table>__<family>`` tables, each repeating the
    key columns, and the table ``spacr_export_tables`` lists the split.
    The file appears complete or not at all (written aside, then
    renamed).

    :param project: an open :class:`Store`, or anything :func:`locate`
        accepts.
    :param target: the file to write.
    :param fmt: ``"csv"``, ``"parquet"`` or ``"sqlite"``.
    :param tables: a table name or a list of them; None exports every
        table (SQLite only).
    :param where: a SQL ``WHERE`` clause or a mapping, as for
        :meth:`Store.query`, in the reader's dialect (SQLite's for an
        older SQLite file).
    :param params: the ``?`` parameters of ``where``.
    :param features: the columns to export; None exports every column.
    :param order_by: a column, or ``(column, "asc" or "desc")``; the
        rows otherwise come in the store's key order.
    :param batch_size: rows per batch.
    :returns: ``{"path", "rows", "tables"}``; ``tables`` maps each written
        table to its row count.
    :raises StoreError: for an unknown format, table or column, or more
        than one table to CSV or Parquet.
    """
    fmt = str(fmt).lower()
    if fmt not in ("csv", "parquet", "sqlite"):
        raise StoreError(f"fmt must be csv, parquet or sqlite; got {fmt!r}")
    held = project if isinstance(project, Store) else Store(locate(project))
    try:
        names = (held.tables() if tables is None else
                 [tables] if isinstance(tables, str) else list(tables))
        for name in names:
            held._known(name)
        if fmt != "sqlite" and len(names) != 1:
            raise StoreError(f"{fmt} takes one table; got {len(names)}")
        target = os.path.abspath(os.fspath(target))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        pending = target + ".partial"
        if os.path.exists(pending):
            os.remove(pending)
        try:
            total, counts = _export_into(held, pending, fmt, names, where,
                                         params, features, order_by,
                                         int(batch_size))
            os.replace(pending, target)
        except BaseException:
            if os.path.exists(pending):
                os.remove(pending)
            raise
    finally:
        if held is not project:
            held.close()
    return {"path": target, "rows": int(total), "tables": counts}


def _export_into(store: Store, path: str, fmt: str, names: List[str], where,
                 params, features, order_by, batch_size: int) -> tuple:
    """Write the export to ``path``.

    :returns: ``(rows exported, {written table: its row count})``.
    """
    pa = _require("pyarrow")
    counts: Dict[str, int] = {}
    if fmt in ("csv", "parquet"):
        table = names[0]
        wanted = _export_select(store, table, features, where, params,
                                order_by)[0]
        writer, sink, rows = None, None, 0
        try:
            for chunk in _export_batches(store, table, wanted, where, params,
                                         order_by, batch_size):
                if writer is None:
                    if fmt == "csv":
                        from pyarrow import csv as pa_csv
                        _csv_header(path, chunk.schema.names)
                        sink = builtins.open(path, "ab")
                        writer = pa_csv.CSVWriter(
                            sink, chunk.schema,
                            write_options=pa_csv.WriteOptions(
                                include_header=False,
                                quoting_style="needed"))
                    else:
                        import pyarrow.parquet as pq
                        writer = pq.ParquetWriter(path, chunk.schema)
                writer.write_table(chunk.cast(writer.schema)
                                   if fmt == "parquet" else chunk)
                rows += chunk.num_rows
            if writer is None:
                empty = pa.table({c: pa.array([], pa.string())
                                  for c in wanted})
                if fmt == "csv":
                    _csv_header(path, wanted)
                else:
                    import pyarrow.parquet as pq
                    pq.write_table(empty, path)
        finally:
            if writer is not None:
                writer.close()
            if sink is not None:
                sink.close()
        counts[table] = rows
        return rows, counts
    import sqlite3

    conn = sqlite3.connect(path)
    total = 0
    try:
        conn.execute("CREATE TABLE spacr_export_tables (table_name TEXT, "
                     "source_table TEXT, columns INTEGER)")
        for table in names:
            wanted = _export_select(store, table, features, where, params,
                                    order_by)[0]
            pieces = _export_split(store, table, wanted)
            created = False
            rows = 0
            for chunk in _export_batches(store, table, wanted, where, params,
                                         order_by, batch_size):
                if not created:
                    for name, members in pieces:
                        kinds = ", ".join(
                            f"{_legacy_quoted(c)} "
                            f"{_sqlite_type(chunk.schema.field(c).type)}"
                            for c in members)
                        conn.execute(f"CREATE TABLE {_legacy_quoted(name)} "
                                     f"({kinds})")
                    created = True
                for name, members in pieces:
                    marks = ", ".join("?" for _ in members)
                    columns = [chunk.column(c).to_pylist() for c in members]
                    conn.executemany(
                        f"INSERT INTO {_legacy_quoted(name)} VALUES ({marks})",
                        zip(*columns))
                rows += chunk.num_rows
            if not created:
                for name, members in pieces:
                    conn.execute(f"CREATE TABLE {_legacy_quoted(name)} (" +
                                 ", ".join(_legacy_quoted(c)
                                           for c in members) + ")")
            conn.executemany("INSERT INTO spacr_export_tables VALUES (?, ?, ?)",
                             [(name, table, len(members))
                              for name, members in pieces])
            for name, _members in pieces:
                counts[name] = rows
            total += rows
        conn.commit()
    finally:
        conn.close()
    return total, counts


def interrupt(store: Store) -> None:
    """Stop the query a store (or a :func:`fork` of one) is running.

    The query raises in the thread that runs it; nothing else is touched.

    :param store: the store or its twin.
    """
    held = store._duck
    if held is not None:
        try:
            held.interrupt()
        except Exception:
            pass
