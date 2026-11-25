"""Ragged object crops in per-well Zarr pixel pools, with the ssCell_Crops table (items 983 and 984, alpha).

Each crop is its object's bounding box plus a margin, stored at its own size.
A well's crops of one object class go into flat pixel pools inside the
project's crop store, ``crops.ome.zarr/plate=<P>/well=<W>`` (item 0aj; older
projects have ``crops/plate=<P>/well=<W>.zarr``, which still reads):
``<class>/pool/<k>`` holds the pixels ``(C, n)``, ``<class>/mask/<k>`` the
object's own mask pixels ``(n,)`` and ``<class>/labels/<other class>/<k>``
EVERY object class's labels cut to the same box ``(n,)`` (uint16 or uint32
by the shard's largest label), one pool array per shard of
``crops_per_shard`` crops, written whole by the one writer. The crop table
gives each crop its pool, offset, height and width, so a crop is one
contiguous read; its ``object_label`` with ``labels/<own class>`` rebuilds
the own-object mask exactly. The table is the ``crop_index`` table of the
one store of the project, ``<project>/measurements`` (:func:`_write_crop_index`
and :func:`_crop_index_writer` write it, :func:`_read_crop_index` and
:func:`crop_rows` read it); the root catalog of an older project is still read.

This is spaCR's own crop format, not an OME-Zarr image: viewers open the
images, not the pools. :func:`export_crops` writes crops out as TIFF, PNG
or npy for other tools.
"""
from __future__ import annotations

import collections
import functools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ports import measurements_db_path

__all__ = ["CROP_COLUMNS", "CropPoolWriter", "CropSink", "cut_crops",
           "read_crop", "export_crops", "crop_relationships",
           "open_measure_sink", "receive_measure_crops", "close_measure_sink",
           "record_measure_crops", "record_measure_field_crops",
           "OmeZarrCropSource", "ome_zarr_crops_exist", "ROLE_COLOURS",
           "composite_crop", "load_channel_presets", "save_channel_preset",
           "parse_channel_roles", "format_channel_roles", "legacy_png_view",
           "read_commit", "discard_uncommitted", "pool_attributes", "committed_rows",
           "class_dir", "read_crops"]

#: The ssCell_Crops columns, in order (item 984).
CROP_COLUMNS: Tuple[str, ...] = (
    "object_id", "object_type", "plate", "row", "column", "well", "field",
    "timepoint", "object_label",
    "crop_store", "crop_array", "crop_index", "offset", "height", "width",
    "mask_offset",
    "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1", "margin_px", "clipped_at_edge",
    "png_path")

#: Index columns item 0aj adds beside CROP_COLUMNS: the classes cut per crop,
#: how to cut a crop again from fov.ome.zarr, and its 0.6 transformations.
INDEX_EXTRA_COLUMNS: Tuple[str, ...] = (
    "label_classes", "source_field", "source_channels", "crop_dtype",
    "zero_outside", "coordinate_transformations")

#: Crops per pool shard; each shard is one array written whole (16 chunks of 256).
DEFAULT_CROPS_PER_SHARD = 4096

#: The project's crop store: ``<project>/crop.ome.zarr``,
#: written only when Measure saves crops; otherwise crops are streamed from
#: ``fov.ome.zarr`` (:func:`read_crop`).
CROP_STORE_NAME = "crop.ome.zarr"

#: The older projects' crop root, ``<project>/crops/plate=<P>/well=<W>.zarr``.
_LEGACY_CROP_ROOT = "crops"

#: Every crop root a project may hold, newest first, with the suffix of its
#: well folders: ``crop.ome.zarr`` (0aj), ``crops.ome.zarr`` (the first 0aj
#: commits) and ``crops/`` (items 983/984). Readers look in all of them.
_CROP_ROOTS: Tuple[Tuple[str, str], ...] = (
    (CROP_STORE_NAME, ""), ("crops.ome.zarr", ""), (_LEGACY_CROP_ROOT, ".zarr"))


#: Crops per chunk inside a shard.
CROPS_PER_CHUNK = 256


def cut_crops(image: np.ndarray, labels: np.ndarray, *, margin: int = 0,
              zero_outside: bool = False,
              objects: Optional[Sequence[int]] = None,
              all_labels: Optional[Mapping[str, np.ndarray]] = None
              ) -> List[Dict[str, Any]]:
    """Cut each object's bounding box plus ``margin`` out of a field.

    :param image: the field ``(C, Y, X)``, or None to give each object's
        box only (the crop index of a run that streams its crops, item 0aj).
    :param labels: the object labels ``(Y, X)``.
    :param margin: pixels added on every side, clipped at the field edge.
    :param zero_outside: set pixels outside the object to 0 in the crop.
    :param objects: the labels to cut, ascending; every label when None.
    :param all_labels: ``{object class: labels (Y, X)}``: every class's
        labels cut to the same box go into the record's ``labels`` (item
        0aj), as integers, unchanged.
    :returns: one record per object: ``object_label``, ``pixels``
        ``(C, h, w)``, ``mask`` ``(h, w)`` bool, ``bbox_y0/x0/y1/x1`` (the
        cut box, end-exclusive), ``margin_px`` and ``clipped_at_edge``; with
        ``all_labels`` also ``labels`` ``{class: (h, w)}``.
    """
    from scipy import ndimage

    labels = np.asarray(labels)
    if image is not None:
        image = np.asarray(image)
    if image is not None and (image.ndim != 3 or image.shape[1:] != labels.shape):
        raise ValueError(f"image {image.shape} must be (C, Y, X) over labels {labels.shape}")
    found = np.unique(labels[labels > 0])
    wanted = found if objects is None else np.asarray(sorted(objects))
    boxes = ndimage.find_objects(labels.astype(np.int64, copy=False))
    out = []
    height, width = labels.shape
    for label in wanted:
        box = boxes[int(label) - 1] if int(label) - 1 < len(boxes) else None
        if box is None:
            continue
        y0 = box[0].start - margin
        x0 = box[1].start - margin
        y1 = box[0].stop + margin
        x1 = box[1].stop + margin
        clipped = y0 < 0 or x0 < 0 or y1 > height or x1 > width
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, height), min(x1, width)
        record = {"object_label": int(label),
                  "bbox_y0": int(y0), "bbox_x0": int(x0), "bbox_y1": int(y1),
                  "bbox_x1": int(x1), "margin_px": int(margin),
                  "clipped_at_edge": bool(clipped)}
        if image is not None:
            record.update(_cut_box(image, labels, int(label), (y0, x0, y1, x1),
                                   zero_outside=zero_outside, all_labels=all_labels))
        out.append(record)
    return out


def _cut_box(image: np.ndarray, own_labels: np.ndarray, label: int,
             box: Tuple[int, int, int, int], *, zero_outside: bool = False,
             all_labels: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, Any]:
    """The pixels, own mask and labels of one crop box: the ONE cut (item 0aj).

    Measure's stored crops (:func:`cut_crops`) and the crops streamed from
    ``fov.ome.zarr`` (:func:`read_crop` with no stored pool) both cut here,
    so a stored crop equals its streamed cut byte for byte.

    :param image: the field ``(C, Y, X)`` (or any array indexable that way).
    :param own_labels: the object's own class labels ``(Y, X)``.
    :param label: the object's label.
    :param box: ``(y0, x0, y1, x1)``, end-exclusive, inside the field.
    :param zero_outside: set the pixels outside the object to 0.
    :param all_labels: ``{class: labels (Y, X)}`` cut to the same box.
    :returns: ``pixels`` ``(C, h, w)``, ``mask`` ``(h, w)`` bool and, with
        ``all_labels``, ``labels`` ``{class: (h, w)}``.
    """
    y0, x0, y1, x1 = (int(v) for v in box)
    mask = np.asarray(own_labels[y0:y1, x0:x1]) == label
    pixels = np.array(image[:, y0:y1, x0:x1])
    if zero_outside:
        pixels[:, ~mask] = 0
    out: Dict[str, Any] = {"pixels": pixels, "mask": mask}
    if all_labels:
        out["labels"] = {str(name): np.array(np.asarray(plane)[y0:y1, x0:x1])
                         for name, plane in all_labels.items()}
    return out


def _writer_pool(workers: int, shard_bytes: int):
    """The writer processes of :meth:`CropPoolWriter._flush`, as many as free RAM holds.

    :param workers: writer processes asked for.
    :param shard_bytes: bytes of one full shard, which each writer compresses.
    :returns: a spawn :class:`~concurrent.futures.ProcessPoolExecutor` and
        its process count.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .resource_log import _guard_workers

    count = max(1, int(_guard_workers('crop_writer', workers, shard_bytes) or 1))
    return ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("spawn")), count


class CropPoolWriter:
    """The one writer of a well's ragged crop pools (item 984).

    :param project: the project folder (the 974 store's root).
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class, such as ``'cell'``.
    :param channels: the number of channels.
    :param dtype: the pixel dtype.
    :param mask_dtype: ``bool`` by default, or ``uint8`` to preserve integer
        mask labels from 0 to 255. Zero is background; nonzero labels are
        foreground when reading without context. A class cannot mix mask
        dtypes across resumed writes.
    :param crops_per_shard: crops per pool shard (default 4096). Each shard
        is buffered until full (or :meth:`close`) and written whole, so no
        shard is ever rewritten.
    :param attributes: metadata of this class's pools, stored on the class
        group under ``spacr`` (for example ``channel_names``,
        ``pixel_size_um``, ``settings_hash``, ``relationship_columns``,
        ``source``). ``dtype``, ``channels``, ``compression``,
        ``crops_per_shard`` and ``spacr_version`` are added. ``mask_dtype``
        is also stored for non-Boolean masks; its absence means Boolean.
        A key that is already stored with another value is refused, so pools of one class
        never mix channel orders or settings.
    :param store_name: the store folder inside ``project``:
        ``crop.ome.zarr`` (Measure), or ``datasets/<name>.ome.zarr`` for a
        training dataset (:func:`write_training_dataset`).

    Pools are written whole as they fill. :meth:`commit` then records them,
    with their table rows, as the class's committed set; a stopped run
    leaves pools that no commit lists, and :meth:`discard_uncommitted`
    removes them before a rerun. :meth:`read_commit` gives back what was
    committed, so a converter or a resumed run can skip the wells it has.
    """

    def __init__(self, project: Any, plate: str, well: str, object_type: str,
                 channels: int, dtype: Any = np.uint16, *,
                 mask_dtype: Any = np.bool_,
                 crops_per_shard: int = DEFAULT_CROPS_PER_SHARD,
                 attributes: Optional[Mapping[str, Any]] = None,
                 store_name: Any = CROP_STORE_NAME):
        """Open the well's crop store and start an empty shard buffer."""
        from .ome_zarr import require_zarr
        from .store import _encoded

        self.zarr = require_zarr()
        self.project = Path(project)
        self.plate, self.well = str(plate), str(well)
        self.object_type = str(object_type)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        self.mask_dtype = np.dtype(mask_dtype)
        if self.mask_dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            from .qt.i18n import tr
            raise ValueError(tr("Crop mask_dtype must be bool or uint8."))
        self.crops_per_shard = max(1, int(crops_per_shard))
        self.relative = (Path(store_name) / f"plate={_encoded(plate)}"
                         / f"well={_encoded(well)}")
        self.path = self.project / self.relative
        root = self.project / store_name
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "zarr.json").is_file():
            self.zarr.open_group(str(root), mode="w", zarr_format=3).attrs[
                "spacr"] = {"crop_store": 2, "layout": "plate=<P>/well=<W>/<class>"}
        if not (self.path.parent / "zarr.json").is_file():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.zarr.open_group(str(self.path.parent), mode="w", zarr_format=3)
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / "zarr.json").is_file():
            self.zarr.open_group(str(self.path), mode="w", zarr_format=3)
        self._shard = 0
        while (self.path / self.object_type / "pool" / str(self._shard)).exists():
            self._shard += 1
        self._buffer: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []
        self.pools: List[str] = []
        self.keep_rows = True
        self.workers = 1
        self._procs = None
        self._inflight: "collections.deque" = collections.deque()
        self._groups: set = set()
        self._set_attributes(dict(attributes or {}))
        base = self.path / self.object_type
        self._previous = _read_commit_at(base, rows=False)
        self._stream_path = base / ".rows.jsonl.tmp"
        self._stream = None
        self._streamed = 0
        self._index: Optional["_IndexStream"] = None

    def _set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Store the class's pool attributes; refuse a conflicting value."""
        from ._version import __version__

        group_path = self.path / self.object_type
        group_path.mkdir(parents=True, exist_ok=True)
        group = self.zarr.open_group(str(group_path), mode="a", zarr_format=3)
        stored = dict(group.attrs.get("spacr") or {})
        if ((stored and np.dtype(stored.get("mask_dtype", bool)) != self.mask_dtype)
                or ("mask_dtype" in attributes
                    and np.dtype(attributes["mask_dtype"]) != self.mask_dtype)):
            from .qt.i18n import tr
            raise ValueError(tr(
                "Crop mask_dtype conflicts with the stored pools or supplied metadata; "
                "write this class to a new store."))
        attributes.pop("mask_dtype", None)
        wanted = {"dtype": self.dtype.str, "channels": self.channels,
                  "coordinate_systems": {"crop": ["y", "x"], "fov": ["y", "x"],
                                         "ngff_version": "0.6"},
                  "compression": "zstd-3", "crops_per_shard": self.crops_per_shard,
                  **json.loads(json.dumps(attributes, default=str))}
        if self.mask_dtype != np.dtype(bool):
            wanted["mask_dtype"] = self.mask_dtype.str
        fixed = ("dtype", "channels", "compression", "channel_names",
                 "pixel_size_um", "settings_hash", "relationship_columns")
        for key, value in wanted.items():
            if key in fixed and key in stored and stored[key] != value:
                raise ValueError(
                    f"{group_path}: {key} is {stored[key]!r} in the stored pools, "
                    f"not {value!r}; write this class to a new store")
        stored.update(wanted)
        stored.setdefault("spacr_version", str(__version__))
        group.attrs["spacr"] = stored
        self.attributes = stored

    def commit(self) -> Dict[str, Any]:
        """Close the writer and record its pools and rows as committed.

        The rows go to ``<class>/rows.jsonl`` and the record to
        ``<class>/spacr_commit.json``, each under a temporary name and then
        renamed, record last. Earlier commits of the class are kept: the
        record lists every committed pool and row.

        :returns: the commit record, with its ``rows``.
        """
        rows = self.close()
        base = self.path / self.object_type
        previous = self._previous
        pools = list(previous["pools"]) if previous else []
        pools += [p for p in self.pools if p not in pools]
        if self._index is not None:
            return self._commit_to_index(base, pools)
        if self.keep_rows:
            stream = self._open_stream()
            for row in rows:
                stream.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
            self._streamed += len(rows)
        n_rows = (int(previous["n_rows"]) if previous else 0) + self._streamed
        self._open_stream().close()
        os.replace(self._stream_path, base / "rows.jsonl")
        record = {"version": 1, "plate": self.plate, "well": self.well,
                  "object_type": self.object_type, "pools": pools,
                  "n_rows": n_rows, "attributes": self.attributes}
        _atomic_write(base / "spacr_commit.json",
                      json.dumps(record, indent=1, default=_json_default) + "\n")
        self.rows = []
        self.pools = []
        self._stream, self._streamed, self._previous = None, 0, record
        if not self.keep_rows:
            return record
        prior = (_read_commit_at(base)["rows"][:int(previous["n_rows"])]
                 if previous else [])
        return {**record, "rows": prior + rows}

    def _commit_to_index(self, base: Path, pools: List[str]) -> Dict[str, Any]:
        """Commit with the rows in the crop index of the store, not ``rows.jsonl``.

        The index part is finished and registered first, then the record
        is written, naming the store of the index; so a record on disk always
        has its rows in the catalog.
        """
        index = self._index
        index.finish()
        stale = base / "rows.jsonl"
        if stale.is_file():
            stale.unlink()
        record = {"version": 1, "plate": self.plate, "well": self.well,
                  "object_type": self.object_type, "pools": pools,
                  "n_rows": index.rows, "attributes": self.attributes,
                  "index": {"table": CROP_INDEX_TABLE,
                            "store": _relative_to(index.project, base)}}
        _atomic_write(base / "spacr_commit.json",
                      json.dumps(record, indent=1, default=_json_default) + "\n")
        self.rows, self.pools, self._previous = [], [], record
        self._index = None
        return record

    def _streaming(self, *, workers: int = 1, index: Any = None,
                   index_schema: Any = None) -> "CropPoolWriter":
        """Keep only counts, not rows, and stream each shard's rows to the row file (item 0ap).

        A well of a million crops then needs no rows in memory, and
        :meth:`commit` only renames the row file it streamed; it returns
        the record without ``rows`` (:func:`_read_commit_record` reads it
        back). ``workers`` above 1 hands each full shard, put together in
        shared memory, to one of that many writer processes, which compress
        and write it while this process fills the next; the shards, pools
        and rows are the same for every value.

        :param workers: shards written at once.
        :param index: a store folder: the rows go to its ``crop_index``
            table (:class:`_IndexStream`) instead of the row file.
        :param index_schema: the Arrow schema of the index part (see
            :class:`_IndexStream`).
        :returns: the writer.
        """
        if self._buffer or self.rows or self._streamed:
            raise RuntimeError("choose streaming before adding crops")
        self.keep_rows = False
        self.workers = max(1, int(workers))
        if index is not None:
            if self._previous:
                raise ValueError(f"{self.path / self.object_type}: a committed class "
                                 f"cannot take more crops through the store index; "
                                 f"remove it first")
            self._index = _IndexStream(index, self.plate, self.well, schema=index_schema)
        return self

    def _add_prepared(self, crops: Iterable[Mapping[str, Any]]) -> None:
        """Queue crops :func:`_prepare_crops` checked and indexed where they were cut (item 0ap).

        :param crops: the prepared crops, in the order they are stored.
        """
        for crop in crops:
            self._buffer.append(crop)
            if len(self._buffer) >= self.crops_per_shard:
                self._flush()

    def add(self, crops: Iterable[Mapping[str, Any]], **identity: Any) -> None:
        """Queue crops from :func:`cut_crops`, with their identity columns.

        :param identity: ``field``, ``timepoint`` and any other columns every
            crop of this call shares.

        :param crops: crops from :func:`cut_crops`, in the order they are stored.
            With ``mask_dtype=uint8``, masks must contain integers from 0 to 255.
        """
        for crop in crops:
            _check_crop(crop, self.channels, self.mask_dtype)
            self._buffer.append({**identity, **crop})
            if len(self._buffer) >= self.crops_per_shard:
                self._flush()

    def _add_index_lines(self, lines: Iterable[str]) -> None:
        """Commit rows of crops that are not stored (saving crops off, item 0ap).

        :param lines: finished crop-index rows as JSON lines (from
            :func:`_index_lines`), in order; they name no pool, so the crop
            reader cuts each crop from its source field.
        """
        if self._buffer:
            self._flush()
        self._drain(0)
        if self.keep_rows:
            self.rows.extend(json.loads(line) for line in lines)
            return
        if self._index is not None:
            self._index.add(lines)
            return
        stream = self._open_stream()
        for line in lines:
            stream.write(line)
            self._streamed += 1

    def close(self) -> List[Dict[str, Any]]:
        """Write the last, partly filled shard; return every crop's table row."""
        if self._buffer:
            self._flush()
        self._drain(0)
        return self.rows

    def _flush(self) -> None:
        """Write the buffered crops as one pool shard and one mask shard.

        With ``workers`` above 1 (:meth:`_streaming`) the shard is put
        together here in shared memory and compressed and written by a
        writer process while this process fills the next one; at most
        ``workers`` shards are in flight, and their rows reach the row file
        in shard order.
        """
        crops, self._buffer = self._buffer, []
        name = str(self._shard)
        self._shard += 1
        self.pools.append(name)
        classes = sorted({c for crop in crops for c in (crop.get("labels") or {})})
        self._ensure_groups(["pool", "mask"] + [f"labels/{c}" for c in classes])
        if self.workers <= 1:
            arrays, offsets = _shard_arrays(self.channels, self.dtype, self.mask_dtype,
                                            crops, classes, np.empty)
            for kind, array, chunks, shards in arrays:
                _write_pool_array(self.zarr, self.path / self.object_type / kind, name,
                                  array, chunks, shards)
            del arrays
            self._take(self._shard_rows(name, crops, offsets))
            return
        held: List[Any] = []

        def shared(shape, dtype):
            """An array in a new shared-memory block, which the writer process attaches."""
            from multiprocessing import shared_memory

            block = shared_memory.SharedMemory(
                create=True, size=max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize))
            held.append(block)
            return np.ndarray(shape, dtype, buffer=block.buf)

        arrays, offsets = _shard_arrays(self.channels, self.dtype, self.mask_dtype,
                                        crops, classes, shared)
        specs = [(kind, held[index].name, array.shape, array.dtype.str, chunks, shards)
                 for index, (kind, array, chunks, shards) in enumerate(arrays)]
        shard_bytes = sum(int(array.nbytes) for _k, array, _c, _s in arrays)
        del arrays
        try:
            if self._procs is None:
                self._procs, self.workers = _writer_pool(self.workers, shard_bytes)
            future = self._procs.submit(_write_shared_shard,
                                        str(self.path / self.object_type), name, specs)
        except BaseException:
            for block in held:
                block.close()
                block.unlink()
            raise
        self._inflight.append((future, held, name, crops, offsets))
        self._drain(self.workers)

    def _drain(self, keep: int) -> None:
        """Take finished shards in order until at most ``keep`` are in flight."""
        while len(self._inflight) > keep:
            future, held, name, crops, offsets = self._inflight.popleft()
            try:
                future.result()
            finally:
                for block in held:
                    block.close()
                    block.unlink()
            self._take(self._shard_rows(name, crops, offsets))
        if not self._inflight and self._procs is not None and keep == 0:
            self._procs.shutdown()
            self._procs = None

    def _take(self, written: Tuple[str, int, Optional[List[Dict[str, Any]]]]) -> None:
        """Append one written shard's rows to the row file (and the kept rows)."""
        text, count, rows = written
        if self.keep_rows:
            self.rows.extend(rows)
            return
        if self._index is not None:
            self._index.add(text.splitlines(True))
            return
        self._open_stream().write(text)
        self._streamed += count

    def _open_stream(self):
        """The temporary row file, opened on first use after the earlier commit's rows."""
        if self._stream is None:
            self._stream = open(self._stream_path, "w", encoding="utf-8")
            if self._previous:
                with open(self.path / self.object_type / "rows.jsonl",
                          encoding="utf-8") as earlier:
                    shutil.copyfileobj(earlier, self._stream, 1 << 22)
        return self._stream

    def _ensure_groups(self, kinds: Sequence[str]) -> None:
        """Create each kind's parent group once, before any writer uses it."""
        for kind in kinds:
            if kind in self._groups:
                continue
            parent = self.path / self.object_type / kind
            parent.mkdir(parents=True, exist_ok=True)
            self.zarr.open_group(str(parent), mode="a", zarr_format=3)
            self._groups.add(kind)

    def _shard_rows(self, name: str, crops: List[Mapping[str, Any]], offsets: Sequence[int]
                    ) -> Tuple[str, int, Optional[List[Dict[str, Any]]]]:
        """One written shard's rows as JSON lines, and as dicts when the writer keeps rows.

        :returns: ``(lines, count, rows)``.
        """
        store = self.relative.as_posix()
        array_name = f"{self.object_type}/pool/{name}"
        lines, rows = [], ([] if self.keep_rows else None)
        for index, (crop, off) in enumerate(zip(crops, offsets)):
            location = {"crop_store": store, "crop_array": array_name,
                        "crop_index": index, "offset": off, "mask_offset": off}
            template = crop.get("_line")
            if template is not None:
                line = _fill_line(template, location)
                if rows is not None:
                    rows.append(json.loads(line))
            else:
                row = _index_row(self.plate, self.well, self.object_type, crop, location)
                line = json.dumps(row, default=_json_default, sort_keys=True) + "\n"
                if rows is not None:
                    rows.append(row)
            lines.append(line)
        return "".join(lines), len(lines), rows


def _shard_arrays(channels: int, dtype: Any, mask_dtype: Any, crops: List[Dict[str, Any]],
                  classes: Sequence[str], allocate) -> Tuple[List[Tuple[str, np.ndarray, Any, Any]],
                                                             List[int]]:
    """One shard's pool, mask and label arrays with their chunking, and each crop's offset.

    The crops give up their pixels once they are in the arrays, so a shard is
    held once while it is written (item 0ap).

    :param allocate: ``(shape, dtype) -> array`` for the pool and mask
        (``np.empty``, or shared memory for a writer process).
    :returns: ``([(kind, array, chunks, shards)], offsets)``.
    """
    sizes = [int(c["pixels"].shape[1] * c["pixels"].shape[2]) for c in crops]
    total = int(sum(sizes))
    pool = allocate((channels, total), np.dtype(dtype))
    mask = allocate((total,), np.dtype(mask_dtype))
    offset = 0
    offsets = []
    for crop, size in zip(crops, sizes):
        pool[:, offset:offset + size] = crop["pixels"].reshape(channels, size)
        mask[offset:offset + size] = np.asarray(crop["mask"]).reshape(size)
        offsets.append(offset)
        offset += size
    chunk = max(1, -(-total // 16)) if total else 1
    arrays = [("pool", pool, (channels, chunk), (channels, chunk * 16)),
              ("mask", mask, (chunk,), (chunk * 16,))]
    for cls in classes:
        planes = [np.asarray((crop.get("labels") or {}).get(
            cls, np.zeros(crop["pixels"].shape[1:], np.uint16))).reshape(size)
            for crop, size in zip(crops, sizes)]
        from .image_store import _compact_labels
        joined = (_compact_labels(np.concatenate(planes)) if planes
                  else np.zeros(0, np.uint16))
        placed = allocate(joined.shape, joined.dtype)
        placed[...] = joined
        arrays.append((f"labels/{cls}", placed, (chunk,), (chunk * 16,)))
    for crop in crops:
        crop.pop("pixels", None)
        crop.pop("mask", None)
        if crop.get("labels"):
            crop["labels"] = dict.fromkeys(crop["labels"], True)
    return arrays, offsets


def _write_pool_array(zarr: Any, parent: Path, name: str, array: np.ndarray,
                      chunks: Any, shards: Any) -> None:
    """Write one pool array whole under a temporary name, then rename it into place."""
    from zarr.codecs import ZstdCodec

    tmp = parent / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    group = zarr.open_group(str(tmp.parent), mode="a", zarr_format=3)
    out = group.create_array(
        tmp.name, shape=array.shape, dtype=array.dtype,
        chunks=chunks, shards=shards,
        compressors=[ZstdCodec(level=3)], fill_value=0,
        overwrite=True)
    if array.size:
        out[...] = array
    if (parent / name).exists():
        shutil.rmtree(parent / name)
    os.replace(tmp, parent / name)


def _write_shared_shard(class_path: str, name: str, specs) -> None:
    """A writer process: compress and write one shard's arrays from shared memory (item 0ap).

    :param class_path: the class folder (``.../<class>``).
    :param name: the shard's name.
    :param specs: ``(kind, block, shape, dtype, chunks, shards)`` per array.
    """
    from multiprocessing import shared_memory

    from .ome_zarr import require_zarr

    zarr = require_zarr()
    for kind, block_name, shape, dtype, chunks, shards in specs:
        block = shared_memory.SharedMemory(name=block_name, track=False)
        try:
            array = np.ndarray(tuple(shape), np.dtype(dtype), buffer=block.buf)
            _write_pool_array(zarr, Path(class_path) / kind, name, array,
                              tuple(chunks), tuple(shards))
            del array
        finally:
            block.close()
#: Where a crop's pixels are: the row columns only the writer knows (item 0ap).
_LOCATION_KEYS: Tuple[str, ...] = ("crop_store", "crop_array", "crop_index", "offset",
                                   "mask_offset")

#: Stand-ins for the location columns in a row built before its shard is
#: known; each is unique in a row's JSON and replaced by :func:`_fill_line`.
_LOCATION_TOKENS: Dict[str, str] = {
    key: json.dumps(f"\x00spacr-crop-location:{key}") for key in _LOCATION_KEYS}


def _fill_line(template: str, location: Mapping[str, Any]) -> str:
    """A prepared row's JSON line with its pool location filled in.

    The result is byte for byte the line the writer would dump from the
    finished row (``sort_keys``), because only the stand-in values change.
    """
    for key, token in _LOCATION_TOKENS.items():
        template = template.replace(token, json.dumps(location[key], default=_json_default), 1)
    return template


def _check_crop(crop: Mapping[str, Any], channels: int, mask_dtype: Any) -> None:
    """Refuse a crop the pool cannot hold: the writer's checks (item 984)."""
    pixels = np.asarray(crop["pixels"])
    if pixels.shape[0] != channels:
        raise ValueError(f"crop has {pixels.shape[0]} channels, the pool {channels}")
    if np.dtype(mask_dtype) == np.dtype(np.uint8):
        mask = np.asarray(crop["mask"])
        if mask.dtype.kind not in "biu" or np.any(mask < 0) or np.any(mask > 255):
            from .qt.i18n import tr
            raise ValueError(tr("Crop uint8 masks must contain integers from 0 to 255."))


def _prepare_crops(plate: str, well: str, object_type: str, crops: Iterable[Mapping[str, Any]],
                  *, channels: int, mask_dtype: Any = np.bool_,
                  **identity: Any) -> List[Dict[str, Any]]:
    """Check crops and build their rows where they are cut, for :meth:`CropPoolWriter.add`.

    The per-crop work of :meth:`CropPoolWriter.add` and of its row builder,
    done in the process that cut the crops (a worker, item 0ap), so the
    writing process only routes finished crops into shards. Pass the result
    to :meth:`CropPoolWriter._add_prepared`; the rows are the same as
    :meth:`CropPoolWriter.add` gives the same crops.

    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class.
    :param crops: crops from :func:`cut_crops` (or the same keys).
    :param channels: the pool's channel count.
    :param mask_dtype: the pool's mask dtype.
    :param identity: columns every crop shares (``field``, ``timepoint``).
    :returns: one entry per crop: its ``pixels``, ``mask`` and ``labels``
        and its row as a JSON line with the pool location left open.
    """
    placeholder = {key: json.loads(token) for key, token in _LOCATION_TOKENS.items()}
    out = []
    for crop in crops:
        _check_crop(crop, channels, mask_dtype)
        merged = {**identity, **crop}
        row = _index_row(str(plate), str(well), str(object_type), merged, placeholder)
        entry = {"pixels": merged["pixels"], "mask": merged["mask"],
                 "_line": json.dumps(row, default=_json_default, sort_keys=True) + "\n"}
        if merged.get("labels"):
            entry["labels"] = merged["labels"]
        out.append(entry)
    return out


def _index_lines(plate: str, well: str, object_type: str,
                crops: Iterable[Mapping[str, Any]], **identity: Any) -> List[str]:
    """Crop-index rows of crops that are not stored, as JSON lines (item 0ap).

    :param crops: crop records with their box (no pixels) and the columns
        the reader needs to cut them again (``source_field`` and the rest).
    :returns: the rows as :meth:`CropPoolWriter._add_index_lines` takes them.
    """
    return [json.dumps(_index_row(str(plate), str(well), str(object_type),
                                  {**identity, **crop}, {}),
                       default=_json_default, sort_keys=True) + "\n"
            for crop in crops]
def _index_row(plate: str, well: str, object_type: str, crop: Mapping[str, Any],
               location: Mapping[str, Any]) -> Dict[str, Any]:
    """One crop-index row: the object's keys, its box and where its pixels are (item 0aj).

    The one row builder of the pool writer (``location`` names the pool) and
    of a run that streams its crops (``location`` empty: the crop is cut
    from ``fov.ome.zarr`` on read, from the box and ``source_*`` columns).
    """
    row_letter = "".join(ch for ch in well if ch.isalpha())
    column = "".join(ch for ch in well if ch.isdigit())
    height = int(crop["bbox_y1"]) - int(crop["bbox_y0"])
    width = int(crop["bbox_x1"]) - int(crop["bbox_x0"])
    classes = (",".join(sorted(crop["labels"])) if crop.get("labels")
               else crop.get("label_classes"))
    transform = crop.get("coordinate_transformations") or crop_transformations(
        plate, well, crop.get("field"), crop["bbox_y0"], crop["bbox_x0"])
    return {
        "coordinate_transformations": transform,
        "object_type": object_type, "plate": plate,
        "row": row_letter, "column": int(column) if column else None,
        "well": well, "field": crop.get("field"),
        "timepoint": crop.get("timepoint"),
        "object_label": crop["object_label"],
        "object_id": crop.get("object_id") or (
            f"{plate}_{well}_{crop.get('field')}_{object_type}_{crop['object_label']}"),
        "crop_store": location.get("crop_store"),
        "crop_array": location.get("crop_array"),
        "crop_index": location.get("crop_index"), "offset": location.get("offset"),
        "height": height, "width": width, "mask_offset": location.get("mask_offset"),
        **{k: crop[k] for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
                                 "margin_px", "clipped_at_edge")},
        "png_path": None,
        **({"label_classes": classes} if classes else {}),
        **{k: v for k, v in crop.items() if k not in (
            "pixels", "mask", "labels", "field", "timepoint", "object_label",
            "object_id", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
            "margin_px", "clipped_at_edge", "label_classes",
            "coordinate_transformations")}}


def crop_transformations(plate: str, well: str, field: Any, y0: int, x0: int, *,
                         scale: Sequence[float] = (1.0, 1.0)) -> str:
    """Where a crop came from, as OME-Zarr 0.6 coordinate transformations (item 0aj).

    A crop's pixel grid (coordinate system ``crop``, axes y, x) maps into
    its source field's pixel grid (coordinate system ``fov:<row>/<col>/<field>``
    of ``fov.ome.zarr``) by ``scale`` (1, 1 unless the crop was resized)
    then a translation to the box's top-left corner (bbox y0, x0, which
    already include the margin). Crops sit in pooled arrays, not one
    OME-Zarr image each, and 0.6 has no reference from one store's
    coordinate system into another store, so the transformation is kept
    with each crop's index row (JSON), naming the source store and field
    key beside it. :func:`crop_to_fov` applies it.

    :returns: the JSON text of ``{"input", "output", "source_store",
        "source_key", "transformations": [scale, translation]}``.
    """
    from .image_store import MERGED_STORE_NAME, field_key

    try:
        key = field_key(well, field) if field is not None else None
    except (TypeError, ValueError):
        key = None
    record = {"input": "crop", "output": f"fov:{key}", "source_store": MERGED_STORE_NAME,
              "source_key": key,
              "transformations": [
                  {"type": "scale", "scale": [float(v) for v in scale]},
                  {"type": "translation", "translation": [float(y0), float(x0)]}]}
    return json.dumps(record, sort_keys=True)


def crop_to_fov(row: Mapping[str, Any], ys: Any, xs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Map crop pixel coordinates into the source field's pixel grid (item 0aj).

    :param row: the crop's index row (its ``coordinate_transformations``).
    :param ys: crop row coordinates.
    :param xs: crop column coordinates.
    :returns: ``(fov_ys, fov_xs)``: scale, then translation, as 0.6 orders a sequence.
    """
    record = json.loads(str(_row_field(row, "coordinate_transformations")))
    ys, xs = np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)
    for step in record["transformations"]:
        if step["type"] == "scale":
            ys, xs = ys * step["scale"][0], xs * step["scale"][1]
        elif step["type"] == "translation":
            ys, xs = ys + step["translation"][0], xs + step["translation"][1]
        else:
            raise ValueError(f"crop transformation {step['type']!r} is not one spaCR writes")
    return ys, xs


def _json_default(value: Any) -> Any:
    """JSON for numpy scalars and paths in crop rows and records."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot store {type(value).__name__} in a crop record")


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` under a temporary name, then rename it into place."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def class_dir(project: Any, plate: str, well: str, object_type: str) -> Path:
    """The folder of one well's crops of one class: ``<project>/crops/plate=<P>/well=<W>.zarr/<class>``.

    New projects keep it at ``<project>/crops.ome.zarr/plate=<P>/well=<W>/<class>``
    (item 0aj); the older ``crops/`` folder is returned while only it exists.

    :param project: the project folder.
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class.
    """
    from .store import _encoded

    found = [Path(project) / root / f"plate={_encoded(plate)}"
             / f"well={_encoded(well)}{suffix}" / str(object_type)
             for root, suffix in _CROP_ROOTS]
    return next((path for path in found if path.exists()), found[0])


#: The former private name, kept for existing callers.
_class_dir = class_dir


def read_commit(project: Any, plate: str, well: str,
                object_type: str) -> Optional[Dict[str, Any]]:
    """What :meth:`CropPoolWriter.commit` recorded for one well and class.

    :param project: the project folder.
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class.
    :returns: the record with its ``rows``, or None when nothing is committed.
    """
    return _read_commit_at(_class_dir(project, plate, well, object_type))


def _read_commit_record(project: Any, plate: str, well: str,
                        object_type: str) -> Optional[Dict[str, Any]]:
    """:func:`read_commit` without the rows: the record only (item 0ap).

    Pools, ``n_rows`` and attributes, with the row file checked to hold
    ``n_rows`` lines without parsing them: what a resume or a count needs,
    in constant memory.
    """
    return _read_commit_at(_class_dir(project, plate, well, object_type), rows=False)


def _count_lines(path: Path) -> int:
    """The lines of a row file, counted in blocks without parsing (item 0ap).

    The writer puts one row per line, each ending in a newline, and never
    an empty line.
    """
    count, last = 0, b""
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (1 if last not in (b"\n", b"") else 0)


def _read_commit_at(base: Path, *, rows: bool = True) -> Optional[Dict[str, Any]]:
    """The commit record (and rows) of one class folder, or None (this writer's own store)."""
    record_path = base / "spacr_commit.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("index"):
        if not rows:
            return record
        folder = Path(os.path.normpath(base / str(record["index"]["store"])))
        found = _index_rows(folder, record["plate"], record["well"], record["object_type"])
        if len(found) != int(record["n_rows"]):
            raise ValueError(f"{base}: the commit lists {record['n_rows']} rows, "
                             f"the crop index of {folder} holds {len(found)}")
        return {**record, "rows": found}
    if not rows:
        held = _count_lines(base / "rows.jsonl")
        if held != int(record["n_rows"]):
            raise ValueError(f"{base}: the commit lists {record['n_rows']} rows, "
                             f"rows.jsonl holds {held}")
        return record
    lines = (base / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    found = [json.loads(line) for line in lines if line.strip()]
    if len(found) != int(record["n_rows"]):
        raise ValueError(f"{base}: the commit lists {record['n_rows']} rows, "
                         f"rows.jsonl holds {len(found)}")
    return {**record, "rows": found}


#: Row columns that say where a crop is stored, rebuilt by every write.
_LAYOUT_COLUMNS = ("object_type", "plate", "row", "column", "well", "crop_store",
                   "crop_array", "crop_index", "offset", "height", "width",
                   "mask_offset", "png_path", "field", "timepoint")


def _crop_from_row(project: Any, row: Mapping[str, Any]
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A committed crop read back as ``(crop, identity)`` for a new writer."""
    pixels, mask = read_crop(project, row)
    crop = {k: v for k, v in row.items() if k not in _LAYOUT_COLUMNS + ("label_classes",)}
    crop.update(pixels=pixels, mask=mask)
    labels = read_crop_labels(project, row)
    if labels:
        crop["labels"] = labels
    return crop, {"field": row.get("field"), "timepoint": row.get("timepoint")}


def _crop_wells(project: Any, plate: str) -> List[str]:
    """The wells of one plate that hold a crop store (item 0aj), sorted.

    The one listing every reader uses: ``crops.ome.zarr/plate=<P>/well=<W>``
    and, for older projects, ``crops/plate=<P>/well=<W>.zarr``.

    :param project: the project folder.
    :param plate: the plate name.
    """
    from urllib.parse import unquote

    from .store import _encoded

    wells = set()
    for root, suffix in _CROP_ROOTS:
        folder = Path(project) / root / f"plate={_encoded(plate)}"
        for path in folder.glob(f"well=*{suffix}"):
            if path.is_dir() and path.name.endswith(suffix):
                name = path.name[len("well="):]
                wells.add(unquote(name[:-len(suffix)] if suffix else name))
    return sorted(wells)


def committed_rows(project: Any) -> List[Dict[str, Any]]:
    """Every committed crop row of the project, by plate, well and class.

    :param project: the project folder holding ``crops.ome.zarr`` (or an
        older project's ``crops/``).
    :returns: the rows :meth:`CropPoolWriter.commit` recorded, in path order.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()
    records = [record for root, suffix in _CROP_ROOTS
               for record in sorted((Path(project) / root).glob(
                   f"plate=*/well=*{suffix}/*/spacr_commit.json"))]
    for record in records:
        meta = json.loads(record.read_text(encoding="utf-8"))
        key = (meta["plate"], meta["well"], meta["object_type"])
        if key in seen:
            continue
        seen.add(key)
        found = read_commit(project, *key)
        rows.extend(found["rows"] if found else [])
    return rows


def discard_uncommitted(project: Any, plate: str, well: str,
                        object_type: str) -> List[str]:
    """Remove the pools of one well and class that no commit lists.

    A stopped run leaves whole pools it never committed, and temporary
    folders; a rerun calls this first, so the class holds only committed
    pools and new ones are numbered after them.

    :param project: the project folder.
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class.
    :returns: the pool names removed.
    """
    base = _class_dir(project, plate, well, object_type)
    record = _read_commit_record(project, plate, well, object_type)
    keep = set(record["pools"]) if record else set()
    removed = []
    stale_rows = base / ".rows.jsonl.tmp"
    if stale_rows.is_file():
        stale_rows.unlink()
    kinds = ["pool", "mask"] + [f"labels/{entry.name}" for entry in sorted(
        (base / "labels").iterdir()) if entry.is_dir()] if (base / "labels").is_dir() else ["pool", "mask"]
    for kind in kinds:
        folder = base / kind
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            if entry.name == "zarr.json":
                continue
            if entry.name.startswith(".") or entry.name not in keep:
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                if kind == "pool" and not entry.name.startswith("."):
                    removed.append(entry.name)
    return removed


def pool_attributes(project: Any, plate: str, well: str,
                    object_type: str) -> Dict[str, Any]:
    """The attributes stored on one well's class group (see :class:`CropPoolWriter`).

    :param project: the project folder.
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class.
    :returns: the ``spacr`` attributes; empty when the class has none.
    """
    from .ome_zarr import require_zarr

    base = _class_dir(project, plate, well, object_type)
    if not (base / "zarr.json").is_file():
        return {}
    group = require_zarr().open_group(str(base), mode="r")
    return dict(group.attrs.get("spacr") or {})


def read_crop(project: Any, row: Mapping[str, Any], *, context: bool = True,
              size: Optional[Tuple[int, int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """One crop and its mask from its crop-index row: THE crop reader (item 0aj).

    A row with a stored pool (Measure saved crops to ``crop.ome.zarr``, or
    an older project's crop store) is read from it. A row without one (the
    run streamed its crops) is cut on the fly from ``fov.ome.zarr`` and its
    labels through the shared field reader, by the same cut the writer
    uses, so the two give the same pixels, mask and labels.

    :param project: the project folder.
    :param row: the crop's table row (``crop_store``, ``crop_array``,
        ``offset``, ``height``, ``width``, ``mask_offset``).
    :param context: keep the pixels around the object; False zeroes them.
        A row with ``has_mask=False`` keeps its pixels because its stored
        zero mask is a placeholder, which is still returned unchanged.
    :param size: ``(h, w)`` to resize to, nearest-neighbour and therefore
        deterministic; the stored size when None.
    :returns: ``(pixels (C, h, w), mask (h, w))``, preserving both stored
        dtypes and mask labels.
    """
    if not _has_pool(row):
        crop = _stream_crop(project, row)
        return _finish_crop(crop["pixels"], crop["mask"],
                            context or _mask_is_absent(row), size)
    store = Path(project) / str(row["crop_store"])
    height, width = int(row["height"]), int(row["width"])
    n = height * width
    off = int(row["offset"])
    pool = _open_pool(store / str(row["crop_array"]))
    pixels = np.asarray(pool[:, off:off + n]).reshape(pool.shape[0], height, width)
    mask_array = str(row["crop_array"]).replace("/pool/", "/mask/")
    moff = int(row["mask_offset"])
    mask = np.asarray(_open_pool(store / mask_array)[
        moff:moff + n]).reshape(height, width)
    return _finish_crop(pixels, mask, context or _mask_is_absent(row), size)


@functools.lru_cache(maxsize=256)
def _open_pool_cached(path: str, modified: int):
    """One opened pool array, kept while its metadata file is unchanged."""
    from .ome_zarr import require_zarr

    return require_zarr().open_array(path, mode="r")


def _open_pool(path: Any):
    """Open a crop pool array once per process (item 0aj: random crop reads).

    Pools are written whole and renamed into place, so an array whose
    ``zarr.json`` is unchanged holds the same bytes; a rewritten pool gets a
    new modification time and is opened afresh.
    """
    path = os.fspath(path)
    try:
        modified = os.stat(os.path.join(path, "zarr.json")).st_mtime_ns
    except OSError:
        from .ome_zarr import require_zarr

        return require_zarr().open_array(path, mode="r")
    return _open_pool_cached(path, modified)


def _label_classes(row: Mapping[str, Any]) -> List[str]:
    """The object classes whose labels a crop row's shard stores (item 0aj)."""
    value = row.get("label_classes")
    if not isinstance(value, str) or not value:
        return []
    return [name for name in value.split(",") if name]


def read_crop_labels(project: Any, row: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Every object class's labels cut to a crop's box (item 0aj).

    :param project: the project folder.
    :param row: the crop's table row; ``label_classes`` names the classes.
    :returns: ``{class: labels (h, w)}`` in the stored integer dtype; empty
        for a crop of an older project, which stored its own mask only.
        ``labels[row['object_type']] == row['object_label']`` is the crop's
        own-object mask.
    """
    classes = _label_classes(row)
    if not classes:
        return {}
    if not _has_pool(row):
        return _stream_crop(project, row, labels=True)["labels"]
    store = Path(project) / str(row["crop_store"])
    height, width = int(row["height"]), int(row["width"])
    n = height * width
    off = int(row["mask_offset"])
    out = {}
    for cls in classes:
        array = str(row["crop_array"]).replace("/pool/", f"/labels/{cls}/")
        out[cls] = np.asarray(_open_pool(store / array)[off:off + n]).reshape(height, width)
    return out


def _has_pool(row: Mapping[str, Any]) -> bool:
    """Whether a crop-index row names a stored pool (not a streamed crop)."""
    value = _row_field(row, "crop_array")
    return isinstance(value, str) and bool(value)


def _source_field_path(project: Any, row: Mapping[str, Any]) -> str:
    """The field a streamed crop is cut from: ``fov.ome.zarr/<name>`` or ``merged/<name>``."""
    from .image_store import _merged_store_path, _store_entries

    name = _row_field(row, "source_field")
    if not isinstance(name, str) or not name:
        from .qt.i18n import tr
        raise ValueError(tr(
            "crop {object_id} has no stored pixels and no source field to cut "
            "it from; run Measure again", object_id=_row_field(row, "object_id")))
    store = _merged_store_path(project)
    if name in _store_entries(store):
        return os.fspath(store / name)
    return os.fspath(Path(project) / "merged" / name)


@functools.lru_cache(maxsize=16)
def _measured_record(path: str, stamp: Any) -> Dict[Any, np.ndarray]:
    """The measured-mask planes of one fov.ome.zarr field, read once per record (item 0aj)."""
    from .crops import _read_measured_masks
    from .image_store import _read_field

    if stamp is None:
        return {}
    return _read_measured_masks(path, _read_field(path, mmap_mode="r"))


#: Open field readers of the crop streamer, by (field path, ledger entry, record stamp).
_WINDOWS: "collections.OrderedDict[Any, _StoreFieldWindows]" = collections.OrderedDict()


def _field_windows(path: str, store: Path, entry: Mapping[str, Any]) -> "_StoreFieldWindows":
    """One cached :class:`_StoreFieldWindows` per field while its entry and record are unchanged."""
    from .crops import _measured_masks_stamp

    key = (path, json.dumps(dict(entry), sort_keys=True, default=str), _measured_masks_stamp(path))
    found = _WINDOWS.get(key)
    if found is None:
        found = _WINDOWS[key] = _StoreFieldWindows(path, store, entry)
        while len(_WINDOWS) > 32:
            _WINDOWS.popitem(last=False)
    else:
        _WINDOWS.move_to_end(key)
    return found


class _StoreFieldWindows:
    """Box reads of one fov.ome.zarr field: channels and labels, chunk by chunk (item 0aj).

    The crop streamer reads only the chunks a crop's box covers, through
    cached array handles, and applies the labels Measure measured with (the
    field's measured-mask record), as :class:`spacr.crops.MergedField` does
    for a whole field.
    """

    def __init__(self, path: str, store: Path, entry: Mapping[str, Any]):
        """Open the field's image and label arrays and its measured-mask record.

        :param path: the field's name inside the store, as Measure lists it.
        :param store: the ``fov.ome.zarr`` folder.
        :param entry: the field's ledger entry in that store.
        """
        from .crops import _measured_masks_stamp
        from .image_store import _image_planes, _store_layout

        layout = _store_layout(store) or {}
        self.entry = dict(entry)
        self.group = store / entry["key"]
        self.n_intensity = len(layout.get("intensity_channels") or ())
        self.planes = _image_planes(entry, layout)
        self.mask_dims = {str(k): int(v) for k, v in (layout.get("mask_dims") or {}).items()}
        self.dtype = np.dtype(entry["dtype"])
        self.measured = _measured_record(path, _measured_masks_stamp(path))
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._cache: Dict[str, np.ndarray] = {}

    def prefetch(self, y0: int, y1: int, x0: int, x1: int) -> None:
        """Read one region once for many crops inside it (a batch of crops of one field)."""
        self._region = (int(y0), int(y1), int(x0), int(x1))
        self._cache = {}

    def release(self) -> None:
        """Forget the prefetched region."""
        self._region, self._cache = None, {}

    def _box(self, name: str, array_path: Path, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """One box of a stored array: ``array[..., y0:y1, x0:x1]`` of a 2-D field, or
        the field frame with z reduced on read (item 0aj P3, the shared reader)."""
        from .image_store import _is_layered, read_frame_channels, read_frame_labels

        if not _is_layered(self.entry):
            return np.asarray(_open_pool(array_path)[..., y0:y1, x0:x1])
        if name == "0":
            return read_frame_channels(self.group, self.entry, window=(y0, y1, x0, x1))
        return read_frame_labels(self.group, self.entry, name.split("/", 1)[1],
                                 window=(y0, y1, x0, x1))

    def _read(self, name: str, array_path: Path, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """``array[..., y0:y1, x0:x1]``, from the prefetched region when it holds the box."""
        region = self._region
        if region is not None and region[0] <= y0 and y1 <= region[1] and region[2] <= x0 and x1 <= region[3]:
            block = self._cache.get(name)
            if block is None:
                block = self._cache[name] = self._box(name, array_path, *region)
            return block[..., y0 - region[0]:y1 - region[0], x0 - region[2]:x1 - region[2]]
        return self._box(name, array_path, y0, y1, x0, x1)

    def channels(self, channels: Sequence[int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """``(C, h, w)`` of the given merged planes (intensity planes only)."""
        if any(c >= self.n_intensity for c in channels):
            raise ValueError(f"{self.group}: crop channels {list(channels)} include a mask plane")
        block = self._read("0", self.group / "0", y0, y1, x0, x1)
        block = np.array(block[[self.planes[c] for c in channels]])
        if block.shape[-2:] != (y1 - y0, x1 - x0):
            padded = np.zeros(block.shape[:-2] + (y1 - y0, x1 - x0), block.dtype)
            padded[..., :block.shape[-2], :block.shape[-1]] = block
            block = padded
        return block

    def labels(self, cls: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """One class's labels over the box: measured where Measure changed them."""
        if cls == "cytoplasm":
            if "cytoplasm" in self.measured:
                return np.asarray(self.measured["cytoplasm"][y0:y1, x0:x1])
            cell = self.labels("cell", y0, y1, x0, x1)
            interior = np.zeros(cell.shape, bool)
            for other in self.mask_dims:
                if other != "cell":
                    interior |= self.labels(other, y0, y1, x0, x1) != 0
            return np.where(interior, 0, cell)
        dim = self.mask_dims.get(cls)
        if dim is not None and dim in self.measured:
            return np.asarray(self.measured[dim][y0:y1, x0:x1]).astype(self.dtype, copy=False)
        return np.array(self._read(f"labels/{cls}", self.group / "labels" / cls / "0",
                                   y0, y1, x0, x1), dtype=self.dtype)


def _stream_crop(project: Any, row: Mapping[str, Any], *, labels: bool = False
                 ) -> Dict[str, Any]:
    """Cut one crop from its field on the fly (item 0aj; saving crops off).

    The field and its labels come through the shared reader
    (:func:`spacr.crops.open_merged_field`, which also applies the labels
    Measure measured with), and the cut is :func:`_cut_box`, the one the
    writer uses.

    :param project: the project folder.
    :param row: a crop-index row with its box, ``object_type``,
        ``object_label``, ``source_field``, ``source_channels``,
        ``crop_dtype``, ``zero_outside`` and ``label_classes``.
    :param labels: also cut every class in ``label_classes``.
    :returns: ``pixels`` ``(C, h, w)``, ``mask`` ``(h, w)`` and, with
        ``labels``, ``labels``.
    """
    if _is_ops_source(row):
        from .ops_engine import _ph_stream_crop

        pixels, mask = _ph_stream_crop(project, row)
        return {"pixels": pixels, "mask": mask, **({"labels": {}} if labels else {})}
    from .crops import open_merged_field
    from .image_store import _compact_labels, _store_field

    path = _source_field_path(project, row)
    y0, x0, y1, x1 = (int(_row_field(row, k)) for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
    channels = [int(c) for c in str(_row_field(row, "source_channels")).split(",") if c != ""]
    local = (0, 0, y1 - y0, x1 - x0)
    object_type = str(_row_field(row, "object_type"))
    found = _store_field(path)
    if found is not None:
        window_of = _field_windows(path, found[0], found[1])
        dtype = np.dtype(_row_field(row, "crop_dtype") or window_of.dtype)
        window = window_of.channels(channels, y0, y1, x0, x1).astype(dtype, copy=False)
        own = window_of.labels(object_type, y0, y1, x0, x1)
        planes = ({cls: _compact_labels(window_of.labels(cls, y0, y1, x0, x1))
                   for cls in _label_classes(row)} if labels else None)
    else:
        field = open_merged_field(path)
        dtype = np.dtype(_row_field(row, "crop_dtype") or field.crop_dtype)
        window = np.moveaxis(field.read_window(y0, y1, x0, x1, channels, dtype=dtype), -1, 0)
        own = field.mask_plane(object_type)[y0:y1, x0:x1]
        planes = ({cls: _compact_labels(field.mask_plane(cls)[y0:y1, x0:x1])
                   for cls in _label_classes(row)} if labels else None)
    zero = _row_field(row, "zero_outside")
    return _cut_box(window, own, int(_row_field(row, "object_label")), local,
                    zero_outside=bool(zero) and str(zero) not in ("0", "False", "nan"),
                    all_labels=planes)


#: ``source_field`` of a streamed OPS crop of an older run:
#: ``ops:<site>``, cut from the phenotype tiles of that site. Newer OPS rows name
#: their field in ``fov.ome.zarr``, like Measure rows, and carry ``ops_crop``.
#: The reader cuts both through :func:`spacr.ops_engine._ph_stream_crop`.
OPS_SOURCE_PREFIX = "ops:"


def _is_ops_source(row: Mapping[str, Any]) -> bool:
    """Whether a streamed crop is an OPS crop: ``ops_crop`` set, or an older ``ops:<site>`` row."""
    value = _row_field(row, "source_field")
    if isinstance(value, str) and value.startswith(OPS_SOURCE_PREFIX):
        return True
    flag = _row_field(row, "ops_crop")
    return isinstance(flag, (bool, np.bool_)) and bool(flag)


def _mask_is_absent(row: Mapping[str, Any]) -> bool:
    """Recognize an explicit absent-mask flag, including its NumPy and SQLite forms."""
    value = row.get("has_mask")
    return isinstance(value, (bool, np.bool_, int, np.integer)) and value == 0


def _finish_crop(pixels: np.ndarray, mask: np.ndarray, context: bool,
                 size: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply :func:`read_crop`'s ``context`` and ``size`` to one stored crop."""
    if not context:
        pixels = pixels.copy()
        pixels[:, mask == 0] = 0
    if size is not None:
        height, width = mask.shape
        ys = (np.arange(int(size[0])) * height // int(size[0])).astype(np.int64)
        xs = (np.arange(int(size[1])) * width // int(size[1])).astype(np.int64)
        pixels = pixels[:, ys][:, :, xs]
        mask = mask[ys][:, xs]
    return pixels, mask


def _crop_ranges(starts: Sequence[int], sizes: Sequence[int], chunk_size: int
                 ) -> List[Tuple[int, int, List[int]]]:
    """Group requests sharing or touching inner chunks, without spanning unused chunks."""
    ranges: List[Tuple[int, int, List[int]]] = []
    for index in sorted(range(len(starts)), key=starts.__getitem__):
        start, stop = starts[index], starts[index] + sizes[index]
        if ranges and start // chunk_size <= (ranges[-1][1] - 1) // chunk_size + 1:
            lo, hi, members = ranges[-1]
            members.append(index)
            ranges[-1] = (lo, max(hi, stop), members)
        else:
            ranges.append((start, stop, [index]))
    return ranges


def read_crops(project: Any, rows: Iterable[Mapping[str, Any]], *, context: bool = True,
               size: Optional[Tuple[int, int]] = None
               ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Many crops at once, each pool opened once and requested chunk ranges read together.

    The same crops as :func:`read_crop` row by row, in the order given.
    Reading crop by crop opens the pool and decodes its chunk again for every
    crop; here ranges sharing or touching chunks are read together, while
    unrequested chunks between groups are left unread. Contiguous selections
    use one slice. Pixel and mask ranges use their own offsets and chunk sizes.

    :param project: the project folder.
    :param rows: ``ssCell_Crops`` rows.
    :param context: keep the pixels around each object; False zeroes them.
        Rows with ``has_mask=False`` keep their pixels and return the stored
        zero placeholder mask, as :func:`read_crop` does.
    :param size: ``(h, w)`` to resize every crop to (nearest-neighbour).
    :returns: one ``(pixels (C, h, w), mask (h, w))`` per row, preserving
        both stored dtypes and mask labels.
    """
    from .ome_zarr import require_zarr

    zarr = require_zarr()
    rows = list(rows)
    groups: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
    out: List[Any] = [None] * len(rows)
    streamed: Dict[str, List[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if _has_pool(row):
            groups[(str(row["crop_store"]), str(row["crop_array"]))].append(index)
        else:
            streamed[str(_row_field(row, "source_field"))].append(index)
    for _name, members in streamed.items():
        if _is_ops_source(rows[members[0]]):
            from .ops_engine import _PH_WHOLE_FIELD_CROPS, _ph_stream_crop

            whole = len(members) >= _PH_WHOLE_FIELD_CROPS
            for index in members:
                pixels, mask = _ph_stream_crop(project, rows[index], whole=whole)
                out[index] = _finish_crop(pixels, mask, context or _mask_is_absent(rows[index]),
                                          size)
            continue
        windows = None
        if len(members) > 1:
            from .image_store import _store_field

            path = _source_field_path(project, rows[members[0]])
            found = _store_field(path)
            if found is not None:
                windows = _field_windows(path, *found)
                boxes = np.array([[int(_row_field(rows[i], k)) for k in
                                   ("bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1")] for i in members])
                windows.prefetch(boxes[:, 0].min(), boxes[:, 1].max(),
                                 boxes[:, 2].min(), boxes[:, 3].max())
        try:
            for index in members:
                out[index] = read_crop(project, rows[index], context=context, size=size)
        finally:
            if windows is not None:
                windows.release()
    for (store, array), indices in groups.items():
        base = Path(project) / store
        sizes = [int(rows[i]["height"]) * int(rows[i]["width"]) for i in indices]
        starts = [int(rows[i]["offset"]) for i in indices]
        mstarts = [int(rows[i]["mask_offset"]) for i in indices]
        pool = zarr.open_array(str(base / array), mode="r")
        pixels: List[Any] = [None] * len(indices)
        for lo, hi, members in _crop_ranges(starts, sizes, pool.chunks[-1]):
            block = np.asarray(pool[:, lo:hi])
            for member in members:
                i = indices[member]
                height, width = int(rows[i]["height"]), int(rows[i]["width"])
                start, n = starts[member] - lo, sizes[member]
                pixels[member] = block[:, start:start + n].reshape(pool.shape[0], height, width)
        masks = zarr.open_array(str(base / array.replace("/pool/", "/mask/")), mode="r")
        for lo, hi, members in _crop_ranges(mstarts, sizes, masks.chunks[-1]):
            block = np.asarray(masks[lo:hi])
            for member in members:
                i = indices[member]
                height, width = int(rows[i]["height"]), int(rows[i]["width"])
                start, n = mstarts[member] - lo, sizes[member]
                mask = block[start:start + n].reshape(height, width).copy()
                out[i] = _finish_crop(
                    pixels[member], mask, context or _mask_is_absent(rows[i]), size)
    return out


def export_crops(project: Any, rows: Iterable[Mapping[str, Any]], out_dir: Any,
                 *, fmt: str = "tiff", context: bool = True) -> List[str]:
    """Write crops out of the pools as files other tools read (item 984).

    :param rows: ssCell_Crops rows.
    :param out_dir: the folder for the files.
    :param fmt: ``'tiff'`` (all channels, the mask as a second file),
        ``'npy'`` (``(C, h, w)``) or ``'png'`` (first three channels, 8-bit
        only, as PNG (Legacy) crops are).
    :returns: the files written.
    :param project: the project folder holding ``crops/``.
    """
    fmt = str(fmt).lower()
    if fmt not in ("tiff", "npy", "png"):
        raise ValueError(f"fmt must be 'tiff', 'npy' or 'png'; got {fmt!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        pixels, mask = read_crop(project, row, context=context)
        stem = (f"{row['plate']}_{row['well']}_{row.get('field')}_"
                f"{row['object_type']}_{row['object_label']}")
        if fmt == "npy":
            target = out / f"{stem}.npy"
            np.save(target, pixels)
        elif fmt == "tiff":
            from .tiff_io import write_tiff

            target = out / f"{stem}.tif"
            write_tiff(str(target), pixels, metadata={"axes": "CYX"})
            write_tiff(str(out / f"{stem}_mask.tif"), mask.astype(np.uint8))
        else:
            from PIL import Image

            if pixels.dtype != np.uint8:
                raise ValueError("png export takes 8-bit crops; use tiff or npy")
            rgb = np.moveaxis(pixels[:3], 0, -1)
            target = out / f"{stem}.png"
            Image.fromarray(rgb if rgb.shape[-1] == 3 else rgb[..., 0]).save(target)
        written.append(str(target))
    return written


#: The open Measure crop sinks, by project folder, in the writing process.
_SINKS: Dict[str, "CropSink"] = {}


def _labels_under(masks: Mapping[str, Any], name: str,
                  box: Tuple[int, int, int, int], inside: np.ndarray) -> np.ndarray:
    """The nonzero labels of mask ``name`` under the object's own pixels."""
    mask = masks.get(name)
    if mask is None:
        return np.zeros(0, np.int64)
    y0, x0, y1, x1 = box
    values = np.asarray(mask)[y0:y1, x0:x1][inside]
    return values[values > 0]


def _most_common(values: np.ndarray) -> Optional[int]:
    """The most frequent label in ``values``; None when there is none."""
    return int(np.bincount(values.astype(np.int64)).argmax()) if values.size else None


def crop_relationships(crop: Mapping[str, Any], masks: Mapping[str, np.ndarray]
                       ) -> Dict[str, Any]:
    """The per-cell relationship columns of one crop (item 984).

    :param crop: a record from :func:`cut_crops`.
    :param masks: the field's label images by class (``cell``, ``nucleus``,
        ``pathogen`` and organelle classes).
    :returns: ``parent_cell_id``, ``nucleus_id``, ``has_nucleus``,
        ``n_nuclei``, ``n_pathogens``, ``infected`` and one
        ``n_<organelle class>`` per organelle class, counted over the object's
        own pixels; ``annotation``, ``class`` and ``class_score`` empty.
    """
    y0, x0, y1, x1 = (crop[k] for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
    inside = np.asarray(crop["mask"])

    box = (y0, x0, y1, x1)
    cells, nuclei, pathogens = (_labels_under(masks, n, box, inside)
                                for n in ("cell", "nucleus", "pathogen"))
    out = {"parent_cell_id": _most_common(cells), "nucleus_id": _most_common(nuclei),
           "has_nucleus": bool(nuclei.size), "n_nuclei": int(np.unique(nuclei).size),
           "n_pathogens": int(np.unique(pathogens).size),
           "infected": bool(pathogens.size)}
    for name in masks:
        if name not in ("cell", "nucleus", "pathogen", "cytoplasm"):
            out[f"n_{name}"] = int(np.unique(_labels_under(masks, name, box, inside)).size)
    out.update({"annotation": None, "class": None, "class_score": None})
    return out


def _plan_key(name: Any) -> str:
    """A field's key in a sink's plan: its file stem, with or without ``.npy`` or a folder."""
    return os.path.splitext(os.path.basename(str(name)))[0]


class CropSink:
    """The parent's crop writer for one Measure run (item 984).

    :param project: the project folder (the measurements folder's parent).
    :param plan: the run's field files, in the order crops are committed.
    :param crops_per_shard: crops per pool shard.

    Fields can finish in any order; their crops wait in a reorder buffer and
    are written in plan order, so a rerun writes the same shards.
    """

    def __init__(self, project: Any, plan: Sequence[str], *,
                 crops_per_shard: int = DEFAULT_CROPS_PER_SHARD):
        """Index the plan and start with no open writers."""
        self.project = Path(project)
        self.index = {_plan_key(f): i for i, f in enumerate(plan)}
        self.plan_fields = {_field_parts(name) for name in self.index}
        self.crops_per_shard = int(crops_per_shard)
        self.next = 0
        self.pending: Dict[int, List[Mapping[str, Any]]] = {}
        self.writers: Dict[Tuple[str, str, str], CropPoolWriter] = {}
        #: Rows of crops this run streams instead of storing (item 0aj).
        self.index_rows: List[Dict[str, Any]] = []

    def receive(self, file: str, payloads: Sequence[Mapping[str, Any]]) -> None:
        """Take one field's crops; write every field now due in plan order.

        Measure sends a field's classes in one call. A later call for a field
        already written (a further class) is written at once: Measure commits
        one field's packet as a unit, so no other field came between.

        :param file: the field's file name, as in the plan.
        :param payloads: one entry per object class, as :func:`record_measure_field_crops` builds them.
        """
        position = self.index.get(_plan_key(file))
        if position is None:
            position = len(self.index) + len(self.pending)
        if position < self.next:
            self._write(payloads)
            return
        self.pending.setdefault(position, []).extend(payloads)
        while self.next in self.pending:
            self._write(self.pending.pop(self.next))
            self.next += 1

    def _write(self, payloads: Sequence[Mapping[str, Any]]) -> None:
        """Hand one field's crops to their well's writers."""
        for payload in payloads:
            key = (str(payload["plate"]), str(payload["well"]), str(payload["object_type"]))
            if payload.get("index_only"):
                self.index_rows.extend(
                    _index_row(key[0], key[1], key[2], {**crop, **payload["identity"]}, {})
                    for crop in payload["crops"])
                continue
            writer = self.writers.get(key)
            if writer is None:
                writer = self.writers[key] = self._fresh_writer(
                    key, int(payload["channels"]), payload["dtype"])
            writer.add(payload["crops"], **payload["identity"])

    def _fresh_writer(self, key: Tuple[str, str, str], channels: int,
                      dtype: Any) -> "CropPoolWriter":
        """A writer that replaces this run's fields of one well and class.

        A rerun must not append: the class's committed crops of fields this
        run measures again are dropped, crops of the well's other fields (a
        resumed run measures only some) are carried over first, in their
        committed order, and the class is written afresh, so a rerun gives
        the pools and rows of a clean run.
        """
        plate, well, object_type = key
        redo = {fld for (p, w, fld) in self.plan_fields if (p, w) == (plate, well)}
        discard_uncommitted(self.project, plate, well, object_type)
        prior = read_commit(self.project, plate, well, object_type)
        kept = [_crop_from_row(self.project, row) for row in (prior or {}).get("rows", [])
                if row.get("field") not in redo]
        class_dir = _class_dir(self.project, plate, well, object_type)
        if class_dir.exists():
            shutil.rmtree(class_dir)
        writer = CropPoolWriter(self.project, plate, well, object_type, channels,
                                dtype, crops_per_shard=self.crops_per_shard,
                                mask_dtype=(prior or {}).get("attributes", {}).get(
                                    "mask_dtype", bool))
        for crop, identity in kept:
            writer.add([crop], **identity)
        return writer

    def close(self) -> List[Dict[str, Any]]:
        """Write the fields still waiting, in plan order, then commit every writer.

        A class of a planned well that this run found no objects of is
        rewritten too, so its crops of the measured fields go.
        """
        for position in sorted(self.pending):
            self._write(self.pending.pop(position))
        for plate, well in sorted({(p, w) for (p, w, _f) in self.plan_fields}):
            store = _class_dir(self.project, plate, well, "x").parent
            for class_dir in sorted(store.glob("*/spacr_commit.json")):
                key = (plate, well, class_dir.parent.name)
                if key not in self.writers:
                    attrs = pool_attributes(self.project, *key)
                    self.writers[key] = self._fresh_writer(
                        key, int(attrs["channels"]), np.dtype(attrs["dtype"]))
        rows: List[Dict[str, Any]] = []
        for key in sorted(self.writers):
            rows.extend(self.writers[key].commit()["rows"])
        return rows


def open_measure_sink(project: Any, plan: Sequence[str], *,
                      crops_per_shard: int = DEFAULT_CROPS_PER_SHARD) -> CropSink:
    """Start the crop sink of one Measure run in this (the writing) process.

    :param project: the project folder (the measurements folder's parent).
    :param plan: the run's field files, in the order they are committed.
    """
    sink = CropSink(project, plan, crops_per_shard=crops_per_shard)
    _SINKS[os.path.abspath(os.fspath(project))] = sink
    return sink


def receive_measure_crops(project: Any, file: str,
                          payloads: Sequence[Mapping[str, Any]]) -> None:
    """Pass one field's crops to the run's sink (the parent's write handler).

    :param project: the project folder the sink was opened for.
    :param file: the field's file name.
    :param payloads: the field's crops, one entry per object class.
    """
    sink = _SINKS.get(os.path.abspath(os.fspath(project)))
    if sink is None:
        if all(payload.get("index_only") for payload in payloads):
            return
        raise RuntimeError(f"no crop sink is open for {project}")
    sink.receive(file, payloads)


def close_measure_sink(project: Any, db_path: Any) -> int:
    """Close the run's sink and write its rows as the ``ssCell_Crops`` table.

    The table lists every committed crop of the project, so the wells a
    resumed run did not touch stay in it.

    :returns: the number of crops this run's wells and classes now hold.
    :param project: the project folder the sink was opened for.
    :param db_path: the ``measurements.db`` that receives ``ssCell_Crops``.
    """
    import pandas as pd

    from .tabular import write_database

    sink = _SINKS.pop(os.path.abspath(os.fspath(project)), None)
    if sink is None:
        return 0
    written = len(sink.close()) + len(sink.index_rows)
    rows = committed_rows(project)
    prior = _read_crop_index(project)
    if prior is not None and len(prior):
        redo = sink.plan_fields
        for row in prior.to_dict("records"):
            if not _has_pool(row) and (str(row.get("plate")), str(row.get("well")),
                                       _plain_field(row.get("field"))) not in redo:
                rows.append(row)
    rows.extend(sink.index_rows)
    if rows:
        frame = pd.DataFrame(rows)
        ordered = [c for c in CROP_COLUMNS if c in frame.columns]
        frame = frame[ordered + [c for c in frame.columns if c not in ordered]]
        _write_crop_index(db_path, frame, project=project)
    return written


def _plain_field(value: Any) -> Optional[int]:
    """A field number as the sink's plan keeps it (an int or None)."""
    try:
        return None if value is None or value != value else int(value)
    except (TypeError, ValueError):
        return None


#: The crop index's table in spacr.store (item 0aj): object_key -> crop location.
CROP_INDEX_TABLE = "crop_index"

#: spacr.store keeps its own ``crop_store``/``crop_index`` pointer columns,
#: so the crop index stores its pool location under these names.
_INDEX_RENAMES = {"crop_store": "pool_store", "crop_index": "pool_index"}


#: The training datasets of a project (item 0aj): ``datasets/<name>.ome.zarr``.
DATASETS_FOLDER = "datasets"


def dataset_store(project: Any, name: str) -> Path:
    """``<project>/datasets/<name>.ome.zarr``, one store per training dataset (item 0aj)."""
    return Path(project) / DATASETS_FOLDER / f"{name}.ome.zarr"


def write_training_dataset(project: Any, name: str, rows: Sequence[Mapping[str, Any]]
                           ) -> List[Dict[str, Any]]:
    """Save a training dataset as its own OME-Zarr crop store (item 0aj).

    Each selected crop is read through the one crop reader (stored or
    streamed) and written, with every class's labels, through the shared
    crop writer into ``<project>/datasets/<name>.ome.zarr``, by plate, well
    and object class. Every row keeps the keys back to the measured object
    (``source_object_id``, ``plate``, ``well``, ``field``, ``object_type``,
    ``object_label``) and its ``class`` and ``split``. A dataset written
    again under the same name replaces the earlier one.

    :param project: the project folder the crops come from.
    :param name: the dataset name (the generated dataset folder's name).
    :param rows: crop-index rows with ``class`` and ``split`` columns.
    :returns: the dataset's rows, pointing at its own pools (their
        ``crop_store`` is relative to ``project``), in the order given.
    """
    target = dataset_store(project, name)
    if target.exists():
        shutil.rmtree(target)
    relative = Path(DATASETS_FOLDER) / f"{name}.ome.zarr"
    groups: Dict[Tuple[str, str, str], List[int]] = collections.defaultdict(list)
    rows = [dict(r) for r in rows]
    for index, row in enumerate(rows):
        groups[(str(row["plate"]), str(row["well"]), str(row["object_type"]))].append(index)
    out: List[Any] = [None] * len(rows)
    for (plate, well, object_type), members in sorted(groups.items()):
        chosen = [rows[i] for i in members]
        crops = read_crops(project, chosen)
        writer = None
        for row, (pixels, mask) in zip(chosen, crops):
            if writer is None:
                writer = CropPoolWriter(project, plate, well, object_type, pixels.shape[0],
                                        pixels.dtype, store_name=relative,
                                        mask_dtype=mask.dtype if mask.dtype == np.uint8 else bool,
                                        attributes={"dataset": str(name)})
            crop = {k: v for k, v in row.items() if k not in _LAYOUT_COLUMNS + (
                "label_classes", "source_field", "source_channels", "crop_dtype")}
            crop.update(pixels=pixels, mask=mask, source_object_id=row.get("object_id"))
            labels = read_crop_labels(project, row)
            if labels:
                crop["labels"] = labels
            writer.add([crop], field=row.get("field"), timepoint=row.get("timepoint"))
        written = writer.commit()["rows"]
        for i, new in zip(members, written):
            out[i] = new
    meta = {"dataset": str(name), "n_crops": len(rows),
            "classes": sorted({str(r.get("class")) for r in rows}),
            "splits": dict(collections.Counter(str(r.get("split")) for r in rows))}
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write(target / "spacr_dataset.json", json.dumps(meta, indent=1) + "\n")
    return out


def read_training_dataset(project: Any, name: str) -> List[Dict[str, Any]]:
    """The rows of a training dataset saved by :func:`write_training_dataset`.

    :param project: the project folder.
    :param name: the dataset name.
    :returns: every committed row, ready for :func:`read_crop`.
    """
    target = dataset_store(project, name)
    rows: List[Dict[str, Any]] = []
    for record in sorted(target.glob("plate=*/well=*/*/spacr_commit.json")):
        meta = json.loads(record.read_text(encoding="utf-8"))
        lines = (record.parent / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        found = [json.loads(line) for line in lines if line.strip()]
        if len(found) != int(meta["n_rows"]):
            raise ValueError(f"{record.parent}: the commit lists {meta['n_rows']} rows, "
                             f"rows.jsonl holds {len(found)}")
        rows.extend(found)
    return rows


#: The key columns of every crop-index part, and the order its rows are kept in.
_INDEX_KEYS = ("object_id", "object_type", "field", "object_label")
_INDEX_ORDER = ("object_type", "field", "object_label")

#: Index columns held as nullable integers.
_INDEX_INTEGERS = ("field", "pool_index", "offset", "mask_offset", "column", "timepoint")


def _index_frame(frame: Any) -> Any:
    """Crop rows as the crop index stores them: pool columns renamed, integers nullable."""
    index = frame.rename(columns=_INDEX_RENAMES)
    for column in _INDEX_INTEGERS:
        if column in index.columns:
            index[column] = index[column].astype("Int64")
    return index


def _index_folder(project: Any, *, write: bool = False) -> Path:
    """The store folder that holds the crop index of a project.

    New projects keep it in their one store, ``<project>/measurements``;
    an older project whose root catalog ``<project>/spacr.duckdb`` lists
    ``crop_index`` (and whose measurements store does not) is read there,
    and never rewritten. A ``measurements`` folder is its own index folder.

    :param project: the project folder (or its measurements store).
    :param write: the folder a new index is written to: always the
        measurements store.
    """
    project = Path(project)
    if project.name == "measurements":
        return project
    new = project / "measurements"
    if write:
        return new
    for folder in (new, project):
        if _lists_index(folder):
            return folder
    return new


def _lists_index(folder: Path) -> bool:
    """Whether the catalog of a store folder lists the crop-index table."""
    from . import store

    catalog = Path(folder) / store.CATALOG
    try:
        stat = os.stat(catalog)
    except OSError:
        return False
    return _catalog_lists_index(os.path.abspath(catalog), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=64)
def _catalog_lists_index(catalog: str, modified: int, size: int) -> bool:
    """:func:`_lists_index` for one catalog file, read once while it is unchanged."""
    from . import store

    try:
        with store.open(os.path.dirname(catalog), verify="none") as opened:
            return CROP_INDEX_TABLE in opened.tables()
    except store.StoreError:
        return False


def _relative_to(path: Any, base: Any) -> str:
    """``path`` relative to the folder ``base`` (``..`` allowed), posix; absolute on another drive."""
    try:
        return Path(os.path.relpath(os.path.abspath(path), os.path.abspath(base))).as_posix()
    except ValueError:
        return Path(os.path.abspath(path)).as_posix()


def _crop_index_writer(folder: Any, plate: str, well: str, *, schema: Any = None):
    """THE writer of the crop-index part of one well, for Measure and OPS alike.

    :param folder: the store folder (:func:`_index_folder` with ``write``).
    :param plate: the plate name.
    :param well: the well name.
    :param schema: the Arrow schema of the part; None takes that of the first batch.
    :returns: a :class:`spacr.store.WellWriter`; append frames made by
        :func:`_index_frame`, then finish it and register the folder.
    """
    from . import store

    return store.writer(str(folder), CROP_INDEX_TABLE, plate=plate, well=well,
                        sort_by=_INDEX_ORDER, key_columns=_INDEX_KEYS, schema=schema)


class _IndexStream:
    """Crop-index rows streamed into the part of one well, a batch at a time.

    :param folder: the store folder the index goes to.
    :param plate: the plate name.
    :param well: the well name.
    :param schema: the Arrow schema of the part (without ``plate``, ``well``),
        so a batch whose optional column is all empty still has its type.
    :param batch: rows per appended batch.

    The rows arrive as the JSON lines the pool writer builds (in their
    stored order), so the rows of a well are never all held at once.
    """

    def __init__(self, folder: Any, plate: str, well: str, *, schema: Any = None,
                 batch: int = 8192):
        """Open the part of the well; nothing is visible before :meth:`finish`."""
        self.project = Path(folder)
        self.plate, self.well = str(plate), str(well)
        self.writer = _crop_index_writer(folder, plate, well, schema=schema)
        self.batch = int(batch)
        self.lines: List[str] = []
        self.rows = 0

    def add(self, lines: Iterable[str]) -> None:
        """Queue rows (JSON lines), appending each full batch."""
        for line in lines:
            if line.strip():
                self.lines.append(line)
        while len(self.lines) >= self.batch:
            self._append(self.lines[:self.batch])
            self.lines = self.lines[self.batch:]

    def _append(self, lines: Sequence[str]) -> None:
        """Append one batch of rows to the part."""
        import pandas as pd

        frame = _index_frame(pd.DataFrame([json.loads(line) for line in lines]))
        self.writer.append(frame)
        self.rows += len(lines)

    def finish(self) -> Dict[str, Any]:
        """Append the last rows, finish the part and register it in the store."""
        from . import store

        if self.lines:
            self._append(self.lines)
            self.lines = []
        manifest = self.writer.finish()
        store.register(str(self.project))
        return manifest


def _index_rows(folder: Any, plate: str, well: str, object_type: str) -> List[Dict[str, Any]]:
    """One well and class of the crop index of a store folder, in stored order."""
    from . import store

    with store.open(str(folder), verify="none") as opened:
        if CROP_INDEX_TABLE not in opened.tables():
            return []
        found = opened.sql(
            f"SELECT * FROM {CROP_INDEX_TABLE} WHERE plate = ? AND well = ? AND "
            f"object_type = ? ORDER BY {', '.join(_INDEX_ORDER)}",
            [str(plate), str(well), str(object_type)])
    found = found.rename(columns={v: k for k, v in _INDEX_RENAMES.items()})
    return found.astype(object).where(found.notna(), None).to_dict("records")


def _write_crop_index(db_path: Any, frame: Any, *, project: Any = None) -> None:
    """The crop-index adapter's writer (item 0aj): one place that stores the crop rows.

    The index goes through spacr.store, into the one store of the project
    (``<project>/measurements``): one ``crop_index`` part per
    plate and well (:func:`_crop_index_writer`), keyed like every table
    (``plate``, ``well``, ``field``, ``object_type``, ``object_label``,
    ``object_id``), registered in its catalog. The same rows are mirrored
    to ``measurements.db:ssCell_Crops``, because Annotate and Classify write
    their annotation columns into that table by ``object_id`` and old
    readers read it; it is the annotation table, not a second index.

    :param db_path: the run's ``measurements.db``.
    :param frame: the crop rows, as :func:`committed_rows` gives them.
    :param project: the project folder; the parent of ``db_path``'s folder
        when None.
    """
    from . import store
    from .tabular import write_database

    project = Path(project) if project is not None else Path(db_path).parent.parent
    folder = _index_folder(project, write=True)
    index = _index_frame(frame)
    for (plate, well), part in index.groupby(["plate", "well"], sort=True):
        part = part.sort_values(list(_INDEX_ORDER), kind="stable", na_position="last")
        with _crop_index_writer(folder, plate, well) as writer:
            writer.append(part.reset_index(drop=True))
    store.register(str(folder))
    from .measurement_store import drop_tables, store_native, write_rows
    if store_native(db_path):
        drop_tables(db_path, ["ssCell_Crops"])
        write_rows(os.path.dirname(os.path.abspath(os.fspath(db_path))),
                   "ssCell_Crops", frame, narrow=False)
        return
    write_database(frame, os.fspath(db_path), "ssCell_Crops",
                   if_exists="replace", canonicalise=False)


#: Opened spacr stores holding a crop index, by (catalog path, mtime, size).
_INDEX_STORES: "collections.OrderedDict[Tuple[str, int, int], Any]" = collections.OrderedDict()


def _index_store(project: Any):
    """The opened store holding the ``crop_index`` of the project; kept while its catalog is unchanged."""
    from . import store

    folder = _index_folder(project)
    catalog = folder / store.CATALOG
    try:
        stat = os.stat(catalog)
    except OSError:
        return None
    key = (os.path.abspath(catalog), stat.st_mtime_ns, stat.st_size)
    opened = _INDEX_STORES.get(key)
    if opened is None:
        try:
            opened = store.open(str(folder))
        except store.StoreError:
            return None
        if CROP_INDEX_TABLE not in opened.tables():
            opened.close()
            return None
        for old in [k for k in _INDEX_STORES if k[0] == key[0]]:
            _INDEX_STORES.pop(old).close()
        _INDEX_STORES[key] = opened
        while len(_INDEX_STORES) > 8:
            _INDEX_STORES.popitem(last=False)[1].close()
    return opened


def crop_rows(project: Any, object_ids: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    """The crop-index rows of some objects, by key (item 0aj): no table scan.

    The lookup goes through spacr.store's key table, which locates the rows
    and reads only their row groups, so one crop or a batch of crops costs
    the same whatever the project's size.

    :param project: the project folder.
    :param object_ids: the objects' ``object_id`` keys.
    :returns: the rows in the order asked (missing keys left out), or None
        when the project's index is not in the store (an older project; use
        :func:`_read_crop_index`).
    """
    from . import store

    opened = _index_store(project)
    if opened is None:
        return None
    try:
        found = opened.query({"object_id": [str(i) for i in object_ids]},
                             table=CROP_INDEX_TABLE)
    except store.StoreError:
        return None
    found = found.rename(columns={v: k for k, v in _INDEX_RENAMES.items()})
    found = found.astype(object).where(found.notna(), None)
    by_id = {row["object_id"]: row for row in found.to_dict("records")}
    return [by_id[i] for i in map(str, object_ids) if i in by_id]


def _read_crop_index(project: Any) -> Optional[Any]:
    """The crop-index adapter's reader (item 0aj): the project's crop rows.

    Reads the ``crop_index`` table through the shared spaCR data reader,
    :func:`spacr.store.open`, from the store that holds it
    (:func:`_index_folder`: the measurements store of the project, or the
    root catalog of an older project). An older project without it is read from
    ``measurements.db:ssCell_Crops`` (through the same reader, or directly
    when its catalog does not list the table).

    :param project: the project folder (holding ``measurements/``).
    :returns: a DataFrame of crop rows, or None when the project has none.
    """
    from . import store
    from .tabular import read_database

    db = Path(measurements_db_path(project))
    folder = _index_folder(project)
    if _lists_index(folder):
        try:
            with store.open(str(folder)) as opened:
                found = opened.query(table=CROP_INDEX_TABLE)
                found = found.rename(columns={v: k for k, v in _INDEX_RENAMES.items()})
                return found.astype(object).where(found.notna(), None)
        except store.StoreError:
            pass
    from .measurement_store import exists as _measurements_exist
    from .measurement_store import read as _store_read, tables as _store_tables
    if not _measurements_exist(str(db)):
        return None
    if "ssCell_Crops" not in _store_tables(str(db)):
        return None
    if not db.is_file():
        return _store_read(str(db), "ssCell_Crops")
    return read_database(str(db), "ssCell_Crops", canonicalise=False,
                         report=None, migrate=False, read_only=True)[0]


def record_measure_crops(settings: Mapping[str, Any], file_name: str,
                         object_type: str, crops: Sequence[Mapping[str, Any]],
                         channels: int, dtype: Any) -> None:
    """Send one field's crops of one class to the run's writer (item 984).

    A field with several classes goes through
    :func:`record_measure_field_crops` in one call instead, so the sink
    takes the whole field at once.

    :param settings: the Measure settings (``src`` names the merged folder).
    :param file_name: the field's file name, ``<plate>_<well>_<field>``.
    :param object_type: the object class, e.g. ``'cell'``.
    :param crops: crops from :func:`cut_crops`.
    :param channels: the number of stored channels.
    :param dtype: the stored dtype.
    """
    record_measure_field_crops(settings, file_name,
                               [(object_type, crops, channels, dtype)])


def record_measure_field_crops(settings: Mapping[str, Any], file_name: str,
                               classes: Sequence[Tuple[str, Sequence[Mapping[str, Any]],
                                                       int, Any]], *,
                               index_only: bool = False) -> None:
    """Send every class of one field's crops to the run's writer (item 984).

    :param classes: ``(object_type, crops, channels, dtype)`` per class.

    Inside a Measure worker the crops join the field's write packet and reach
    the parent's sink through the write queue; in a single-process run they go
    to the sink directly. Either way only the parent writes the store, and it
    takes the field whole, so fields commit in plan order.

    :param settings: the Measure settings (``src`` names the merged folder).
    :param file_name: the field's file name, ``<plate>_<well>_<field>``.
    :param index_only: the crops carry their boxes only (saving crops is
        off, item 0aj): they enter the crop index with no pool and are cut
        from ``fov.ome.zarr`` when read.

    Every row records how to cut its crop again (``source_field``,
    ``source_channels``, ``crop_dtype``, ``zero_outside``), so a stored crop
    and its streamed cut are the same.
    """
    from .database_concurrency import _capture_write_operation

    plate, well, field = _field_parts(file_name)
    identity = {"field": field, "timepoint": None}
    project = os.path.dirname(os.fspath(settings["src"]))
    source = {"source_field": f"{_plan_key(file_name)}.npy",
              "source_channels": ",".join(str(int(c)) for c in settings["channels"]),
              "zero_outside": bool(settings.get("crop_zero_outside", False))}
    payload = []
    for object_type, crops, channels, dtype in classes:
        packed = [{**c, **source, "crop_dtype": np.dtype(dtype).str, "object_id":
                   f"{plate}_{well}_{field}_{object_type}_{c['object_label']}"}
                  for c in crops]
        payload.append({"plate": plate, "well": well, "object_type": object_type,
                        "channels": int(channels), "dtype": np.dtype(dtype).str,
                        "identity": identity, "crops": packed,
                        "index_only": bool(index_only)})
    if not _capture_write_operation("crops", (project, file_name, payload)):
        receive_measure_crops(project, file_name, payload)


def _field_parts(file_name: str) -> Tuple[str, str, Optional[int]]:
    """``'plate1_B03_2.npy'`` -> ``('plate1', 'B03', 2)``, as :func:`spacr.image_store.split_field_name`.

    A name of another shape keeps its stem as the plate, well ``A01`` and no
    field, so its crops are still stored.
    """
    from .image_store import split_field_name

    try:
        return split_field_name(file_name)
    except ValueError:
        return os.path.splitext(os.path.basename(str(file_name)))[0], "A01", None


def ome_zarr_crops_exist(root: Any) -> bool:
    """Whether ``root`` holds OME-Zarr crops: stored or streamed, with their index.

    :param root: the project folder (the one holding ``crop.ome.zarr`` or an
        older crop store, ``fov.ome.zarr`` and ``measurements/measurements.db``).
        A project that streams its crops (saving crops off, item 0aj) has no
        crop store, only the index and ``fov.ome.zarr``.
    """
    from .image_store import _is_merged_store, _merged_store_path

    root = Path(root)
    db = Path(measurements_db_path(root))
    stored = any(any((root / name).glob(f"plate=*/well=*{suffix}"))
                 for name, suffix in _CROP_ROOTS)
    from .measurement_store import connect as _connect_measurements, exists
    from .measurement_store import tables as _measurement_tables
    if not exists(str(db)) or not (stored or _is_merged_store(_merged_store_path(root))):
        return False
    if "ssCell_Crops" not in _measurement_tables(str(db)):
        return False
    with _connect_measurements(str(db), readonly=True) as conn:
        columns = [r[1] for r in conn.execute(
            'PRAGMA table_info("ssCell_Crops")').fetchall()]
    return bool(columns) and (stored or "source_field" in columns)


def _row_field(row: Any, name: str) -> Any:
    """Field ``name`` of a mapping, a pandas Series or an object; None when absent."""
    if isinstance(row, Mapping):
        return row.get(name)
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return getattr(row, name, None)


def _lazy_crop_source_base():
    """:class:`spacr.crops.CropSource`, imported here so this module loads without it first."""
    from .crops import CropSource
    return CropSource


class OmeZarrCropSource(_lazy_crop_source_base()):
    """Crop source that reads Measure's OME-Zarr crops (items 983 and 984).

    Same interface as :class:`spacr.crops.PngCropSource`: :meth:`get` returns
    a ``(H, W, 3)`` uint8 RGB array, :meth:`get_image` a PIL image,
    :meth:`get_many` one array per row in order, :meth:`resolve` a string
    locator. :meth:`get_raw` adds the stored pixels and mask, in their own
    dtype and every channel, for training.

    A row is an ``ssCell_Crops`` row (``crop_store``, ``crop_array``,
    ``offset``, ...) or any row carrying ``object_id``, or ``plate``,
    ``well``, ``field``, ``object_type`` and ``object_label``; the crop
    columns are then looked up in ``ssCell_Crops``. The RGB view maps
    ``channels`` to red, green, blue and stretches each to its 0.1-99.9
    percentile inside the crop, for display only: :meth:`get_raw` and the
    stored pixels are never stretched.
    """

    kind = "ome-zarr"

    def __init__(self, root: Optional[str] = None, reason: str = "",
                 db_path: Optional[str] = None, *,
                 channels: Optional[Sequence[int]] = None,
                 size: Optional[Tuple[int, int]] = None,
                 context: bool = True):
        """Open the crop table of ``root`` (``measurements/measurements.db``).

        :param root: the project folder.
        :param reason: why this source was picked, for :meth:`describe`.
        :param db_path: the database holding ``ssCell_Crops``.
        :param channels: stored channels shown as red, green, blue (up to
            three); the first three when None.
        :param size: ``(h, w)`` every crop is resized to (nearest-neighbour);
            the stored size when None.
        :param context: keep the pixels around the object; False zeroes them.
        """
        self.root = root
        self.reason = reason
        if db_path is None and root:
            db_path = measurements_db_path(root)
        self.db_path = db_path
        self.channels = None if channels is None else [int(c) for c in channels]
        self.size = None if size is None else (int(size[0]), int(size[1]))
        self.context = bool(context)
        self._table = None

    def _rows_table(self):
        """The ``ssCell_Crops`` table, read once and indexed by ``object_id``."""
        if self._table is None:
            from .tabular import read_database

            table = (_read_crop_index(self.root) if self.root else None)
            if table is None:
                table = read_database(self.db_path, "ssCell_Crops", canonicalise=False,
                                      report=None, migrate=False, read_only=True)[0]
            self._table = table.set_index("object_id", drop=False)
        return self._table

    def _crop_row(self, row: Any) -> Mapping[str, Any]:
        """``row``'s ``ssCell_Crops`` columns, looked up when it lacks them."""
        if all(_row_field(row, k) is not None for k in ("crop_store", "crop_array", "offset")):
            return {k: _row_field(row, k) for k in ("crop_store", "crop_array", "offset",
                                          "height", "width", "mask_offset",
                                          "object_id", "has_mask")}
        object_id = _row_field(row, "object_id")
        if object_id is None:
            parts = [_row_field(row, k) for k in ("plate", "well", "field", "object_type",
                                        "object_label")]
            if any(p is None for p in parts):
                raise ValueError("an OME-Zarr crop row needs the ssCell_Crops "
                                 "columns, object_id, or plate, well, field, "
                                 "object_type and object_label")
            object_id = "_".join(str(int(p)) if isinstance(p, (int, np.integer))
                                 else str(p) for p in parts)
        if self.root and self._table is None:
            found = crop_rows(self.root, [object_id])
            if found is not None:
                if not found:
                    raise KeyError(f"no OME-Zarr crop {object_id!r} in {self.root}")
                return found[0]
        table = self._rows_table()
        if object_id not in table.index:
            raise KeyError(f"no OME-Zarr crop {object_id!r} in {self.db_path}")
        return table.loc[object_id].to_dict()

    def resolve(self, row: Any) -> str:
        """``crops/plate=P/well=W.zarr::<class>/pool/<k>@<offset>`` for ``row``.

        :param row: as for :meth:`get`.
        """
        crop = self._crop_row(row)
        return f"{crop['crop_store']}::{crop['crop_array']}@{int(crop['offset'])}"

    def get_raw(self, row: Any) -> Tuple[np.ndarray, np.ndarray]:
        """The stored crop: ``(pixels (C, h, w), mask (h, w))`` in their stored dtypes.

        :param row: as for :meth:`get`.
        """
        return read_crop(self.root, self._crop_row(row), context=self.context,
                         size=self.size)

    def get_many(self, rows: Iterable[Any]) -> List[np.ndarray]:
        """The RGB views of many rows, read pool by pool (see :func:`read_crops`).

        :param rows: rows as for :meth:`get`; one array per row, in order.
        """
        crops = read_crops(self.root, [self._crop_row(r) for r in rows],
                           context=self.context, size=self.size)
        return [self._rgb(pixels) for pixels, _mask in crops]

    def get(self, row: Any) -> np.ndarray:
        """The crop for ``row`` as a ``(H, W, 3)`` uint8 RGB array (display stretch).

        :param row: an ``ssCell_Crops`` row, or a row naming the object (see the class).
        """
        pixels, _mask = self.get_raw(row)
        return self._rgb(pixels)

    def _rgb(self, pixels: np.ndarray) -> np.ndarray:
        """``channels`` as red, green, blue, each stretched to its 0.1-99.9 percentile."""
        picked = self.channels if self.channels is not None else list(
            range(min(3, pixels.shape[0])))
        rgb = np.zeros(pixels.shape[1:] + (3,), np.uint8)
        for slot, channel in enumerate(picked[:3]):
            plane = pixels[channel].astype(np.float64)
            lo, hi = np.percentile(plane, (0.1, 99.9))
            if hi > lo:
                rgb[..., slot] = np.round(
                    np.clip((plane - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return rgb


#: Named colours a channel role may use, as RGB in 0-1.
ROLE_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0), "grey": (1.0, 1.0, 1.0), "gray": (1.0, 1.0, 1.0),
    "orange": (1.0, 0.5, 0.0), "white": (1.0, 1.0, 1.0)}


def _role_colour(value: Any) -> np.ndarray:
    """A role's colour as RGB in 0-1: a name from :data:`ROLE_COLOURS`, ``'#rrggbb'`` or a triple."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ROLE_COLOURS:
            return np.asarray(ROLE_COLOURS[text])
        if text.startswith("#") and len(text) == 7:
            return np.asarray([int(text[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        raise ValueError(f"unknown colour {value!r}; use a name from ROLE_COLOURS "
                         f"or '#rrggbb'")
    colour = np.asarray(value, dtype=np.float64).reshape(3)
    return colour / 255 if colour.max() > 1 else colour


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    """The ``width``-pixel inner rim of ``mask``; the crop's edge counts as outside."""
    from scipy import ndimage

    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    inner = ndimage.binary_erosion(mask, iterations=max(1, int(width)),
                                   border_value=0)
    return mask & ~inner


def composite_crop(pixels: np.ndarray, roles: Sequence[Mapping[str, Any]], *,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Render a crop of any number of channels to ``(H, W, 3)`` uint8 RGB.

    :param pixels: ``(C, H, W)``, in the stored dtype.
    :param roles: one entry per channel shown, in drawing order:

        - ``{"channel": c, "role": "image", "colour": "green",
          "range": (lo, hi), "gamma": 1.0}``: additive blend of the channel,
          scaled to ``range`` (stored units; the 0.1-99.9 percentile of the
          crop when omitted) and raised to ``1 / gamma``.
        - ``{"channel": c, "role": "outline", "colour": "yellow",
          "width": 1, "threshold": t}``: the object's outline drawn over the
          image, from ``mask`` (``channel`` ``"mask"``), from a label channel
          (pixels > 0) or from a threshold on the channel.

    :param mask: the crop's object mask, used by ``channel: "mask"`` outlines.
    """
    pixels = np.asarray(pixels)
    if pixels.ndim != 3:
        raise ValueError(f"pixels must be (C, H, W); got shape {pixels.shape}")
    rgb = np.zeros(pixels.shape[1:] + (3,), np.float64)
    outlines = []
    for role in roles:
        kind = str(role.get("role", "image")).lower()
        colour = _role_colour(role.get("colour", role.get("color", "grey")))
        channel = role.get("channel", 0)
        if kind == "outline":
            if channel == "mask":
                if mask is None:
                    raise ValueError("an outline on 'mask' needs the crop's mask")
                region = np.asarray(mask, bool)
            elif role.get("threshold") is not None:
                region = pixels[int(channel)] > float(role["threshold"])
            else:
                region = pixels[int(channel)] > 0
            outlines.append((_outline(region, int(role.get("width", 1))), colour))
            continue
        if kind != "image":
            raise ValueError(f"channel role must be 'image' or 'outline'; got {kind!r}")
        plane = pixels[int(channel)].astype(np.float64)
        lo_hi = role.get("range")
        lo, hi = (np.percentile(plane, (0.1, 99.9)) if lo_hi is None
                  else (float(lo_hi[0]), float(lo_hi[1])))
        scaled = np.clip((plane - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(plane)
        gamma = float(role.get("gamma", 1.0) or 1.0)
        if gamma != 1.0:
            scaled = scaled ** (1.0 / gamma)
        rgb += scaled[..., None] * colour
    out = np.clip(rgb, 0, 1)
    for line, colour in outlines:
        out[line] = colour
    return np.round(out * 255).astype(np.uint8)


#: Where Annotate keeps a project's channel-role presets.
PRESETS_FILE = Path("settings") / "annotate_channel_presets.json"


def load_channel_presets(project: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The project's saved channel-role presets, ``{name: roles}``; empty when none.

    :param project: the project folder.
    """
    path = Path(project) / PRESETS_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): list(v) for k, v in (data.get("presets") or {}).items()}


def save_channel_preset(project: Any, name: str,
                        roles: Sequence[Mapping[str, Any]]) -> Path:
    """Save ``roles`` as preset ``name`` for this project (checked by a test render).

    :param project: the project folder.
    :param name: the preset's name; an existing preset of that name is replaced.
    :param roles: channel roles, as :func:`composite_crop` takes them.
    """
    roles = [dict(r) for r in roles]
    channels = [r.get("channel", 0) for r in roles if r.get("channel") != "mask"]
    probe = np.zeros((max([int(c) for c in channels] + [0]) + 1, 4, 4), np.uint16)
    composite_crop(probe, roles, mask=np.zeros((4, 4), bool))
    presets = load_channel_presets(project)
    presets[str(name)] = roles
    path = Path(project) / PRESETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": 1, "presets": presets}, indent=1,
                              default=list), encoding="utf-8")
    os.replace(tmp, path)
    return path


def parse_channel_roles(text: str) -> List[Dict[str, Any]]:
    """Channel roles from Annotate's one-line form; empty text gives ``[]``.

    Roles are separated by ``;``, fields by ``:``: ``channel:role:colour``
    then ``key=value`` options. For example
    ``0:image:blue; 1:image:green:gamma=1.5:range=100-4000;
    mask:outline:yellow:width=2; 3:outline:magenta:threshold=900``.

    :param text: the roles in the one-line form.
    """
    roles: List[Dict[str, Any]] = []
    for part in str(text or "").split(";"):
        fields = [f.strip() for f in part.split(":") if f.strip()]
        if not fields:
            continue
        channel: Any = fields[0]
        if channel != "mask":
            try:
                channel = int(channel)
            except ValueError:
                raise ValueError(f"channel {channel!r} must be a number or 'mask'") from None
        role: Dict[str, Any] = {"channel": channel,
                                "role": fields[1] if len(fields) > 1 else "image",
                                "colour": fields[2] if len(fields) > 2 else "grey"}
        for option in fields[3:]:
            key, _eq, value = option.partition("=")
            key = key.strip().lower()
            if key == "range":
                lo, _dash, hi = value.partition("-")
                role["range"] = (float(lo), float(hi))
            elif key in ("gamma", "threshold"):
                role[key] = float(value)
            elif key == "width":
                role[key] = int(value)
            else:
                raise ValueError(f"unknown channel-role option {key!r}; use range, "
                                 f"gamma, width or threshold")
        roles.append(role)
    composite_crop(np.zeros((max([r["channel"] for r in roles
                                  if r["channel"] != "mask"] + [0]) + 1, 2, 2)),
                   roles, mask=np.zeros((2, 2), bool))
    return roles


def format_channel_roles(roles: Sequence[Mapping[str, Any]]) -> str:
    """The one-line form of ``roles``; :func:`parse_channel_roles` reads it back.

    :param roles: channel roles, as :func:`composite_crop` takes them.
    """
    parts = []
    for role in roles:
        fields = [str(role.get("channel", 0)), str(role.get("role", "image")),
                  str(role.get("colour", role.get("color", "grey")))]
        if role.get("range") is not None:
            lo, hi = role["range"]
            fields.append(f"range={lo:g}-{hi:g}")
        for key in ("gamma", "threshold"):
            if role.get(key) is not None:
                fields.append(f"{key}={float(role[key]):g}")
        if role.get("width") is not None:
            fields.append(f"width={int(role['width'])}")
        parts.append(":".join(fields))
    return "; ".join(parts)


def legacy_png_view(project: Any, row: Mapping[str, Any], png_size: Sequence[int],
                    planes: Sequence[int]) -> np.ndarray:
    """The crop Measure's PNG (Legacy) path writes for this object, from its stored crop.

    PNG (Legacy) keeps the object's own pixels, zeroes the rest and cuts a
    ``png_size`` window centred on the object's centroid (no resize), with
    ``crop_intensity='raw'`` at the source bit depth. The OME-Zarr crop holds
    the box around the object and its mask, so the same window is rebuilt
    exactly; this is what the 986 round trip compares.

    :param png_size: ``(width, height)``, as the ``png_size`` setting.
    :param planes: the stored channels in PNG file order (red, green, blue),
        positions in the crop's own channel list.
    :returns: ``(height, width, len(planes))``, in the stored dtype.
    :param project: the project folder holding ``crops/``.
    :param row: the crop's ``ssCell_Crops`` row.
    """
    from scipy import ndimage

    pixels, mask = read_crop(project, row, context=False)
    width, height = int(png_size[0]), int(png_size[1])
    y0, x0 = int(row["bbox_y0"]), int(row["bbox_x0"])
    cy, cx = ndimage.center_of_mass(mask.astype(np.uint8))
    start_y = int(np.round(cy + y0)) - height // 2
    start_x = int(np.round(cx + x0)) - width // 2
    out = np.zeros((height, width, len(planes)), pixels.dtype)
    stack = np.moveaxis(pixels[list(planes)], 0, -1)
    ys = slice(max(0, y0 - start_y), min(height, y0 - start_y + mask.shape[0]))
    xs = slice(max(0, x0 - start_x), min(width, x0 - start_x + mask.shape[1]))
    src_y = slice(ys.start - (y0 - start_y), ys.stop - (y0 - start_y))
    src_x = slice(xs.start - (x0 - start_x), xs.stop - (x0 - start_x))
    out[ys, xs] = stack[src_y, src_x]
    return out
