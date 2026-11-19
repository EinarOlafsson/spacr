"""spaCR's images as OME-Zarr 0.5 plates, and one reader for every image format (item 983, alpha).

A plate is one store, ``<root>/<plate>.ome.zarr``, laid out the HCS way:
row, column (the well) and field groups, each field an OME-NGFF 0.5 image
(zarr-format 3, one zstd shard per field) with its masks under ``labels/``.

ONE WRITER PER FILE. Field workers never write the store. They return their
arrays through the process pool's result pipe; the parent process, the only
writer, writes each field whole, in plan order, and records it in the
store's commit ledger only after all of the field's arrays are on disk. A
rerun skips committed fields and overwrites the rest, so a stopped run
resumes and a repeated run is byte-identical.

Readers take any of npy, npz, TIFF, PNG or an OME-Zarr field, and find the
format from what is on disk, never from a setting.
"""
from __future__ import annotations

import collections
import json
import logging
import multiprocessing
import os
import re
import shutil
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Sequence, Tuple)

import numpy as np

__all__ = ["IMAGE_FORMATS", "PlateWriter", "read_image", "read_labels",
           "run_fields"]

_LOG = logging.getLogger(__name__)

#: The image formats spaCR writes; ``'npy'`` is today's files exactly.
IMAGE_FORMATS: Tuple[str, ...] = ("npy", "tiff", "ome-zarr")

#: The commit ledger inside a plate store: one JSON line per field.
_LEDGER = "spacr_commits.jsonl"

_WELL = re.compile(r"^([A-Za-z]+)0*(\d+)$")


def _split_well(well: str) -> Tuple[str, str]:
    """``'B03'`` -> ``('B', '3')``: the HCS row and column names."""
    match = _WELL.match(str(well))
    if not match:
        raise ValueError(f"well {well!r} is not a row letter and a column number")
    return match.group(1).upper(), str(int(match.group(2)))


def _field_key(well: str, field: Any) -> str:
    """The short store path of one field: ``B/3/0``."""
    row, column = _split_well(well)
    return f"{row}/{column}/{int(field)}"


class PlateWriter:
    """The only writer of one plate's OME-Zarr store (item 983).

    :param root: the folder that holds ``<plate>.ome.zarr``.
    :param plate: the plate name.
    :param channel_names: the image channels' names, written to ``omero``.
    :param spacing: a :class:`spacr.layers.Spacing` for the pixel size, or
        None to record pixels with no physical unit.

    Fields are written whole with :meth:`write_field`; the plate and well
    metadata are rewritten from the ledger after each field, so a reader
    opening the store mid-run sees only committed fields.
    """

    def __init__(self, root: Any, plate: str, *,
                 channel_names: Optional[Sequence[str]] = None,
                 spacing=None):
        """Open (or start) the plate store and read its commit ledger."""
        from .ome_zarr import require_zarr

        self.zarr = require_zarr()
        self.plate = str(plate)
        self.path = Path(root) / f"{self.plate}.ome.zarr"
        self.channel_names = list(channel_names) if channel_names else None
        self.spacing = spacing
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / "zarr.json").is_file():
            self.zarr.open_group(str(self.path), mode="w", zarr_format=3)
        self._committed: Dict[str, Dict[str, Any]] = {}
        ledger = self.path / _LEDGER
        if ledger.is_file():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._committed[entry["key"]] = entry

    def committed(self, well: str, field: Any) -> bool:
        """Whether the field is written and recorded."""
        return _field_key(well, field) in self._committed

    def write_field(self, well: str, field: Any, image: Optional[np.ndarray],
                    labels: Optional[Mapping[str, np.ndarray]] = None) -> str:
        """Write one field and its label images, then commit it.

        :param image: the field ``(C, Y, X)`` or ``(Y, X)``, or None to write
            labels only.
        :param labels: ``{object class: label image (Y, X)}``.
        :returns: the field's store key.

        Each array is written to a temporary group, closed and renamed into
        place, and the field enters the ledger last, so a crash leaves an
        uncommitted field that the next run rewrites.
        """
        from .ome_zarr import write_ome_zarr

        key = _field_key(well, field)
        target = self.path / key
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if image is not None:
            data = np.asarray(image)
            axes = ["c", "y", "x"] if data.ndim == 3 else ["y", "x"]
            self._atomic(target, lambda tmp: write_ome_zarr(
                tmp, data, axes=axes, spacing=self.spacing,
                channel_names=self.channel_names if data.ndim == 3 else None,
                ngff_version="0.5", name=key))
        else:
            self.zarr.open_group(str(target), mode="w", zarr_format=3)
        names = []
        for name, mask in (labels or {}).items():
            mask = np.asarray(mask)
            label_dir = target / "labels" / str(name)
            label_dir.parent.mkdir(parents=True, exist_ok=True)
            self._atomic(label_dir, lambda tmp, m=mask: write_ome_zarr(
                tmp, m, axes=["y", "x"], spacing=self.spacing,
                downsample="stride", ngff_version="0.5", name=str(name)))
            self._set_attrs(label_dir, {"image-label": {"version": "0.5"}})
            names.append(str(name))
        if names:
            self.zarr.open_group(str(target / "labels"), mode="a",
                                 zarr_format=3).attrs.update(
                {"ome": {"version": "0.5", "labels": names}})
        self._commit(key, well, field, names, image is not None)
        return key

    def _atomic(self, target: Path, write: Callable[[Path], Any]) -> None:
        """Write to a temporary name beside ``target``, then rename it in."""
        tmp = target.parent / f".{target.name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        write(tmp)
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)

    def _set_attrs(self, group_path: Path, extra: Mapping[str, Any]) -> None:
        """Merge ``extra`` into a 0.5 group's ``ome`` attributes."""
        group = self.zarr.open_group(str(group_path), mode="a", zarr_format=3)
        ome = dict(group.attrs.get("ome", {}))
        ome.update(extra)
        group.attrs.update({"ome": ome})

    def _commit(self, key: str, well: str, field: Any, labels: List[str],
                has_image: bool) -> None:
        """Record the field in the ledger and rewrite the plate metadata."""
        entry = {"key": key, "well": str(well), "field": int(field),
                 "labels": labels, "image": bool(has_image)}
        self._committed[key] = entry
        ledger = self.path / _LEDGER
        with open(ledger, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._write_hcs_metadata()

    def _write_hcs_metadata(self) -> None:
        """The plate and well groups' HCS metadata, from committed fields."""
        wells: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
        for entry in self._committed.values():
            row, column, fld = entry["key"].split("/")
            wells[(row, column)].append(int(fld))
        rows = sorted({r for r, _c in wells})
        columns = sorted({c for _r, c in wells}, key=int)
        for (row, column), fields in wells.items():
            group = self.zarr.open_group(str(self.path / row / column),
                                         mode="a", zarr_format=3)
            group.attrs.update({"ome": {"version": "0.5", "well": {
                "images": [{"path": str(f)} for f in sorted(fields)]}}})
            self.zarr.open_group(str(self.path / row), mode="a",
                                 zarr_format=3)
        root = self.zarr.open_group(str(self.path), mode="a", zarr_format=3)
        root.attrs.update({"ome": {"version": "0.5", "plate": {
            "name": self.plate,
            "rows": [{"name": r} for r in rows],
            "columns": [{"name": c} for c in columns],
            "wells": [{"path": f"{r}/{c}", "rowIndex": rows.index(r),
                       "columnIndex": columns.index(c)}
                      for (r, c) in sorted(wells, key=lambda k: (k[0], int(k[1])))],
            "field_count": max((len(v) for v in wells.values()), default=0)}}})


def run_fields(plan: Sequence[Tuple[str, Any, Any]],
               work: Callable[[Any], Tuple[Optional[np.ndarray],
                                           Mapping[str, np.ndarray]]],
               writer: PlateWriter, *, workers: int = 1,
               ahead: Optional[int] = None,
               start_method: Optional[str] = None) -> Dict[str, Any]:
    """Run ``work`` over the fields of ``plan`` and write each through ``writer``.

    :param plan: ``(well, field, task)`` in the order fields are committed.
    :param work: a top-level, spawn-safe function of ``task`` returning
        ``(image or None, {object class: labels})``.
    :param writer: the plate's single writer.
    :param workers: worker processes; 1 runs in this process.
    :param ahead: how many fields may be in flight beyond the next one to
        commit (the reorder buffer); default ``2 * workers``. Memory stays at
        most ``ahead`` fields of results.
    :param start_method: ``'spawn'``, ``'forkserver'`` or ``'fork'``; the
        platform default when None.
    :returns: ``written``, ``skipped`` (already committed) and ``keys``.

    Results come back through the pool's result pipe; nothing is shared
    between processes but those pickled arrays. Fields are committed in plan
    order whatever order they finish in, so a rerun writes the same bytes.
    """
    todo = [(w, f, t) for w, f, t in plan if not writer.committed(w, f)]
    report: Dict[str, Any] = {"written": 0, "skipped": len(plan) - len(todo),
                              "keys": []}
    if not todo:
        return report
    if int(workers) <= 1:
        for well, field, task in todo:
            image, labels = work(task)
            report["keys"].append(writer.write_field(well, field, image, labels))
            report["written"] += 1
        return report
    context = multiprocessing.get_context(start_method)
    window = int(ahead or 2 * int(workers))
    with context.Pool(int(workers)) as pool:
        pending: collections.deque = collections.deque()
        queue = iter(todo)
        for item in queue:
            pending.append((item, pool.apply_async(work, (item[2],))))
            if len(pending) > window:
                break
        while pending:
            (well, field, _task), result = pending.popleft()
            image, labels = result.get()
            report["keys"].append(writer.write_field(well, field, image, labels))
            report["written"] += 1
            nxt = next(queue, None)
            if nxt is not None:
                pending.append((nxt, pool.apply_async(work, (nxt[2],))))
    return report


def _is_zarr_group(path: Path) -> bool:
    """Whether ``path`` is a zarr group or array (format 2 or 3)."""
    return path.is_dir() and any((path / name).is_file()
                                 for name in ("zarr.json", ".zgroup", ".zarray"))


def read_image(path: Any, *, key: Optional[str] = None) -> np.ndarray:
    """One image from any format spaCR writes (item 983).

    :param path: an ``.npy``, ``.npz`` (its first array), ``.tif``/``.tiff``
        or ``.png`` file, an OME-Zarr image, or a plate store with ``key``.
    :param key: a field inside a plate store, as ``'B/3/0'`` or
        ``('B03', 0)``.
    :returns: the array with the dtype, axes and values it was written with.
    """
    target = Path(path)
    if key is not None:
        if not isinstance(key, str):
            key = _field_key(*key)
        target = target / key
    if _is_zarr_group(target):
        from .ome_zarr import read_ome_zarr

        return np.asarray(read_ome_zarr(target).read())
    suffix = target.suffix.lower()
    if suffix == ".npy":
        return np.load(target, allow_pickle=False)
    if suffix == ".npz":
        with np.load(target, allow_pickle=False) as saved:
            return saved[saved.files[0]]
    if suffix in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(target))
    if suffix == ".png":
        from PIL import Image

        with Image.open(target) as handle:
            return np.asarray(handle)
    raise ValueError(f"{target} is not an image spaCR reads (npy, npz, tiff, "
                     f"png or OME-Zarr)")


def read_labels(path: Any, name: str, *, key: Optional[str] = None
                ) -> np.ndarray:
    """One label image: an OME-Zarr field's ``labels/<name>``, or a mask file.

    :param path: a plate store (with ``key``), an OME-Zarr field with a
        ``labels/<name>`` image, a label image itself, or a mask file in any
        format :func:`read_image` reads.
    :param name: the object class, such as ``'cell'``.
    """
    target = Path(path)
    if key is not None:
        target = target / (key if isinstance(key, str) else _field_key(*key))
    if _is_zarr_group(target / "labels" / str(name)):
        return read_image(target / "labels" / str(name))
    return read_image(target)
