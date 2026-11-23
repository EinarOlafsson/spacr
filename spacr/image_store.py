"""spaCR's images as OME-Zarr 0.5 plates, and one reader for every image format (item 983, alpha).

A plate is one store, ``<root>/<plate>.ome.zarr``, laid out the HCS way:
row, column (the well) and field groups, each field an OME-NGFF 0.5 image
(zarr-format 3, unsharded Blosc LZ4 chunks) with its masks under ``labels/``.

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
import functools
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

__all__ = ["IMAGE_FORMATS", "PlateWriter", "export_merged", "read_image",
           "read_labels", "run_fields", "split_well", "field_key",
           "split_field_name"]

_LOG = logging.getLogger(__name__)

#: The image formats spaCR writes; ``'npy'`` is today's files exactly.
IMAGE_FORMATS: Tuple[str, ...] = ("npy", "tiff", "ome-zarr")

#: The commit ledger inside a plate store: one JSON line per field.
_LEDGER = "spacr_commits.jsonl"

_WELL = re.compile(r"^([A-Za-z]+)0*(\d+)$")


def split_well(well: str) -> Tuple[str, str]:
    """``'B03'`` -> ``('B', '3')``: the HCS row and column names.

    :param well: the well name, a row letter and a column number.
    :returns: ``(row, column)`` as the plate store names them.
    :raises ValueError: the name does not start with a row letter.
    """
    match = _WELL.match(str(well))
    if not match:
        raise ValueError(f"well {well!r} is not a row letter and a column number")
    return match.group(1).upper(), str(int(match.group(2)))


#: The former private name, kept for existing callers.
_split_well = split_well


def field_key(well: str, field: Any) -> str:
    """The short store path of one field: ``B/3/0``.

    :param well: the well name, e.g. ``'B03'``.
    :param field: the field number.
    :returns: ``<row>/<column>/<field>`` inside the plate store.
    """
    row, column = _split_well(well)
    return f"{row}/{column}/{int(field)}"


#: The former private name, kept for existing callers.
_field_key = field_key


#: The codec of every field and label array (item 983). Blosc LZ4 with byte
#: shuffle read a whole 1994 x 1994 four-channel field with three masks about
#: twice as fast as zstd level 3 in 256 x 256 tiles, at the same size (about a
#: third of npy), in the 2026-11-19 layout test on the Measure example.
FIELD_CODEC = "lz4"

#: Blosc's compression level for :data:`FIELD_CODEC`.
FIELD_CODEC_LEVEL = 5


class PlateWriter:
    """The only writer of one plate's OME-Zarr store (item 983).

    :param root: the folder that holds ``<plate>.ome.zarr``.
    :param plate: the plate name.
    :param channel_names: the image channels' names, written to ``omero``.
    :param spacing: a :class:`spacr.layers.Spacing` for the pixel size, or
        None to record pixels with no physical unit.
    :param ngff_version: ``'0.5'`` (the default) or ``'0.6'`` (item 1007):
        the OME-Zarr version of every group in the store. One store holds
        one version, so a store started in the other one is refused.
    :param registration: optional field-number to OPS registration-row mapping.
        Requires version 0.6 and pixel coordinates (spacing=None). Existing
        stores must use the same registration mode.
    :param store_path: the store folder itself, instead of
        ``<root>/<plate>.ome.zarr`` (item 0aj: ``<project>/merged.ome.zarr``).
    :param shards: write each field image and each label image as ONE
        zarr v3 shard file holding its 1 x 256 x 256 chunks (item 0aj),
        instead of one file per chunk.
    :param attributes: JSON values merged into the store root's ``spacr``
        attributes, such as the merged plane layout; a key stored with
        another value is refused.

    Fields are written whole with :meth:`write_field`; the plate and well
    metadata are rewritten from the ledger after each field, so a reader
    opening the store mid-run sees only committed fields. Image and label
    arrays share the OME-Zarr default chunk policy: one channel and up to
    256 x 256 spatial pixels per chunk, clipped to the array shape. These
    128 KiB uint16 tiles align with 64- and 128-pixel cell crops and bound
    the amount decoded for a region read. Chunks remain unsharded and use
    Blosc LZ4 with byte shuffle.
    """

    def __init__(self, root: Any, plate: str, *,
                 channel_names: Optional[Sequence[str]] = None,
                 spacing=None, ngff_version: str = "0.5",
                 registration: Optional[Mapping[Any, Mapping[str, Any]]] = None,
                 store_path: Any = None, shards: bool = False,
                 attributes: Optional[Mapping[str, Any]] = None):
        """Open (or start) the plate store and read its commit ledger."""
        from .ome_zarr import require_zarr
        from .qt.i18n import tr

        self.zarr = require_zarr()
        self.plate = str(plate)
        self.path = (Path(store_path) if store_path is not None
                     else Path(root) / f"{self.plate}.ome.zarr")
        self.shards = bool(shards)
        self.channel_names = list(channel_names) if channel_names else None
        self.spacing = spacing
        self.ngff_version = str(ngff_version)
        if self.ngff_version not in ("0.5", "0.6"):
            raise ValueError(tr(
                "The plate store writes OME-Zarr 0.5 or 0.6, not {value!r}.",
                value=ngff_version))
        self.registration = (None if registration is None else
                             {str(key): value for key, value in registration.items()})
        if self.registration is not None and (self.ngff_version != "0.6" or spacing is not None):
            raise ValueError(tr(
                "Stored registration requires OME-Zarr 0.6 and pixel coordinates; "
                "use ngff_version='0.6' and spacing=None, or disable registration."))
        self.path.mkdir(parents=True, exist_ok=True)
        root_json = self.path / "zarr.json"
        if not root_json.is_file():
            self.zarr.open_group(str(self.path), mode="w", zarr_format=3)
        else:
            stored_attrs = (json.loads(root_json.read_text(encoding="utf-8"))
                            .get("attributes", {}))
            stored = stored_attrs.get("ome", {}).get("version")
            stored_plate = stored_attrs.get("ome", {}).get("plate", {}).get("name")
            if stored_plate and str(stored_plate) != self.plate:
                raise ValueError(tr(
                    "{path} holds plate {stored}; it cannot take fields of plate "
                    "{wanted}, because one plate store holds one plate.",
                    path=self.path, stored=stored_plate, wanted=self.plate))
            if stored and str(stored) != self.ngff_version:
                raise ValueError(tr(
                    "{path} is an OME-Zarr {stored} store; it cannot take "
                    "OME-Zarr {wanted} fields, because one store holds one "
                    "version. Write to a new folder or convert the store.",
                    path=self.path, stored=stored, wanted=self.ngff_version))
        if attributes:
            self._set_spacr_attrs(dict(attributes))
        self._committed: Dict[str, Dict[str, Any]] = {}
        ledger = self.path / _LEDGER
        if ledger.is_file():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._committed[entry["key"]] = entry
        if any(bool(entry.get("registration", False)) != (self.registration is not None)
               for entry in self._committed.values()):
            raise ValueError(tr(
                "This plate store uses another registration mode; keep "
                "ome_zarr_store_registration unchanged or write to a new folder."))

    def committed(self, well: str, field: Any) -> bool:
        """Whether the field is written and recorded.

        :param well: the well name, e.g. ``'B03'``.
        :param field: the field number.
        """
        return _field_key(well, field) in self._committed

    def write_field(self, well: str, field: Any, image: Optional[np.ndarray],
                    labels: Optional[Mapping[str, np.ndarray]] = None, *,
                    record: Optional[Mapping[str, Any]] = None) -> str:
        """Write one field and its label images, then commit it.

        :param image: the field ``(C, Y, X)`` or ``(Y, X)``, or None to write
            labels only.
        :param labels: ``{object class: label image (Y, X)}``.
        :param record: extra JSON values kept with the field's ledger entry,
            such as the field's file ``name`` (item 0aj).
        :returns: the field's store key.

        Each array is written to a temporary group, closed and renamed into
        place, and the field enters the ledger last, so a crash leaves an
        uncommitted field that the next run rewrites.

        Only ``(C, Y, X)`` and ``(Y, X)`` are written. A field with time or
        depth axes is refused before anything is written, never flattened
        into channels: the HCS layout here has no ``t`` or ``z`` axis yet.

        :param well: the well name, e.g. ``'B03'``.
        :param field: the field number.
        :raises ValueError: the image is not 2-D or 3-D, or a label image is
            not 2-D.
        """
        from .ome_zarr import write_ome_zarr

        if image is not None and np.ndim(image) not in (2, 3):
            raise ValueError(
                f"field {well}/{field}: the image has {np.ndim(image)} axes; only "
                f"(C, Y, X) or (Y, X) are written (time and depth are not "
                f"flattened into channels)")
        for name, mask in (labels or {}).items():
            if np.ndim(mask) != 2:
                raise ValueError(f"field {well}/{field}: label image {name!r} has "
                                 f"{np.ndim(mask)} axes; only (Y, X) is written")
        metadata, label_metadata = {}, {}
        if self.registration is not None:
            from .ome_zarr_registration import registration_metadata
            from .qt.i18n import tr

            record = self.registration.get(str(field))
            if record is None:
                raise ValueError(tr(
                    "Field {field} has no registration table row; complete the "
                    "OPS registration before exporting.", field=field))
            axes = ["c", "y", "x"] if image is not None and np.ndim(image) == 3 else ["y", "x"]
            metadata = registration_metadata(record, axes=axes, ngff_version=self.ngff_version)
            label_metadata = registration_metadata(record, axes=["y", "x"], ngff_version=self.ngff_version)
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
                ngff_version=self.ngff_version, name=key,
                shards=None if self.shards else False,
                compressor=FIELD_CODEC, compression_level=FIELD_CODEC_LEVEL, **metadata))
        else:
            self.zarr.open_group(str(target), mode="w", zarr_format=3)
        names = []
        for name, mask in (labels or {}).items():
            mask = np.asarray(mask)
            label_dir = target / "labels" / str(name)
            label_dir.parent.mkdir(parents=True, exist_ok=True)
            self._atomic(label_dir, lambda tmp, m=mask: write_ome_zarr(
                tmp, m, axes=["y", "x"], spacing=self.spacing,
                downsample="stride", ngff_version=self.ngff_version, name=str(name),
                shards=None if self.shards else False, compressor=FIELD_CODEC,
                compression_level=FIELD_CODEC_LEVEL, **label_metadata))
            self._set_attrs(label_dir, {"image-label": (
                {"version": "0.5"} if self.ngff_version == "0.5" else {})})
            names.append(str(name))
        if names:
            self.zarr.open_group(str(target / "labels"), mode="a",
                                 zarr_format=3).attrs.update(
                {"ome": {"version": self.ngff_version, "labels": names}})
        self._commit(key, well, field, names, image is not None, record)
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
        """Merge ``extra`` into a 0.5 or 0.6 group's ``ome`` attributes."""
        group = self.zarr.open_group(str(group_path), mode="a", zarr_format=3)
        ome = dict(group.attrs.get("ome", {}))
        ome.update(extra)
        group.attrs.update({"ome": ome})

    def _set_spacr_attrs(self, extra: Mapping[str, Any]) -> None:
        """Merge ``extra`` into the store root's ``spacr`` attributes; refuse a change.

        A key already stored with another value is refused, so one store
        never mixes plane layouts (item 0aj).
        """
        group = self.zarr.open_group(str(self.path), mode="a", zarr_format=3)
        stored = dict(group.attrs.get("spacr") or {})
        wanted = json.loads(json.dumps(dict(extra), default=str))
        for name, value in wanted.items():
            if name in stored and stored[name] != value:
                raise ValueError(
                    f"{self.path}: {name} is {stored[name]!r} in the store, not "
                    f"{value!r}; write to a new store")
        stored.update(wanted)
        group.attrs.update({"spacr": stored})

    def _commit(self, key: str, well: str, field: Any, labels: List[str],
                has_image: bool, record: Optional[Mapping[str, Any]] = None) -> None:
        """Record the field in the ledger and rewrite the plate metadata."""
        entry = {**json.loads(json.dumps(dict(record or {}), default=str)),
                 "key": key, "well": str(well), "field": int(field),
                 "labels": labels, "image": bool(has_image)}
        if self.registration is not None:
            entry["registration"] = True
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
            group.attrs.update({"ome": {"version": self.ngff_version, "well": {
                "images": [{"path": str(f)} for f in sorted(fields)]}}})
            self.zarr.open_group(str(self.path / row), mode="a",
                                 zarr_format=3)
        root = self.zarr.open_group(str(self.path), mode="a", zarr_format=3)
        root.attrs.update({"ome": {"version": self.ngff_version, "plate": {
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
    :param workers: worker processes; 1 runs in this process. With more,
        the first field is computed here and its size sets how many workers
        the free RAM can hold (the RAM guard, ``'image_store'``).
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
    if int(workers) > 1:
        from .resource_log import _guard_workers

        well, field, task = todo.pop(0)
        image, labels = work(task)
        unit = int(getattr(image, "nbytes", 0)) + sum(
            int(getattr(m, "nbytes", 0)) for m in (labels or {}).values())
        report["keys"].append(writer.write_field(well, field, image, labels))
        report["written"] += 1
        workers = max(1, int(_guard_workers("image_store", int(workers), unit)))
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


def split_field_name(name: str) -> Tuple[str, str, int]:
    """``'plate1_B03_2.npy'`` -> ``('plate1', 'B03', 2)``, spaCR's field name.

    The name is read as :func:`spacr.schema.parse_field_stem` reads it: the
    plate (with its underscores escaped), the well and the field, then the
    timepoint Mask Generation appends to every stack
    (``plate1_E01_10_1.npy`` is plate1, E01, field 10), which a single
    field's identity leaves out.

    :param name: a field file name or path, ``<plate>_<well>_<field>[_<time>]``.
    :returns: ``(plate, well, field)``.
    :raises ValueError: the name has another shape.
    """
    from .schema import KEY_SEPARATOR, parse_int_token, unescape_filename_component

    stem = os.path.splitext(os.path.basename(str(name)))[0]
    parts = stem.split(KEY_SEPARATOR)
    if len(parts) == 4 and (parse_int_token(parts[3]) is not None or parts[3] == "None"):
        parts = parts[:3]
    if len(parts) != 3 or parse_int_token(parts[2]) is None:
        raise ValueError(f"cannot identify a field from {stem!r}: expected "
                         f"<plate>_<well>_<field>[_<time>]")
    return unescape_filename_component(parts[0]), parts[1], int(parts[2])


#: The former private name, kept for existing callers.
_split_field_name = split_field_name


def _merged_field(task: Tuple[str, Sequence[int], Mapping[str, int]]
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """One merged stack split into its image ``(C, Y, X)`` and labels (spawn-safe).

    :param task: ``(path, intensity planes, {object class: mask plane})``.
    """
    path, intensity, mask_dims = task
    array = np.load(path, mmap_mode="r")
    image = np.moveaxis(np.array(array[..., list(intensity)]), -1, 0)
    labels = {role: np.array(array[..., dim]) for role, dim in mask_dims.items()
              if dim < array.shape[-1]}
    return image, labels


def _merged_layout(src: Any) -> Tuple[Path, List[int], Dict[str, int], List[str]]:
    """``merged/``, its intensity planes, mask planes and field files."""
    from .crops import MERGED_LAYOUT_SIDECAR

    merged = Path(src) / "merged"
    layout_path = merged / MERGED_LAYOUT_SIDECAR
    if not layout_path.is_file():
        raise ValueError(f"{merged} has no plane layout ({MERGED_LAYOUT_SIDECAR}); "
                         f"run Mask Generation to the merge step first")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    intensity = list(range(len(layout.get("intensity_channels") or ())))
    mask_dims = {str(r): int(d) for r, d in (layout.get("mask_dims") or {}).items()}
    names = sorted(p.name for p in merged.glob("*.npy")
                   if not p.name.startswith("."))
    return merged, intensity, mask_dims, names


def export_merged(src: Any, fmt: str, *, workers: int = 1,
                  channel_names: Optional[Sequence[str]] = None,
                  start_method: Optional[str] = None) -> Dict[str, Any]:
    """Write every ``<src>/merged`` field in ``fmt`` (Mask Generation, item 983).

    :param src: the run root holding ``merged/`` and its plane layout.
    :param fmt: ``'ome-zarr'``: the project's merged store,
        ``<src>/merged.ome.zarr`` (item 0aj), one shard file per field image
        and label image, with the masks as labels and the field names and
        plane layout recorded, so Measure reads it in place of the stacks.
        ``'tiff'``: ``<src>/merged_tiff/<field>.tif`` (the image, planes
        first) and ``<field>_<object>_mask.tif`` beside it, each in its own
        dtype. ``'npy'`` writes nothing.
    :param workers: processes reading the merged stacks; the parent writes.
    :param channel_names: names for the intensity channels.
    :returns: per plate, what :func:`run_fields` reports; or ``tiff``, the
        files written.

    The merged ``.npy`` stacks are left in place. Mask Generation itself
    writes the merged store field by field and no stacks (item 0aj); this
    converts an older project's ``merged/`` folder. One store holds one
    plate, so a folder of several plates is refused.
    """
    fmt = str(fmt or "npy").strip().lower()
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"mask_image_format must be one of {list(IMAGE_FORMATS)}; "
                         f"got {fmt!r}")
    if fmt == "npy":
        return {}
    merged, intensity, mask_dims, names = _merged_layout(src)
    if fmt == "tiff":
        from .tiff_io import write_tiff

        out_dir = Path(src) / "merged_tiff"
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        for name in names:
            image, labels = _merged_field((str(merged / name), intensity, mask_dims))
            stem = Path(name).stem
            arrays = [(stem, image, "CYX")] + [
                (f"{stem}_{role}_mask", mask, "YX") for role, mask in labels.items()]
            for file_stem, array, axes in arrays:
                target = out_dir / f"{file_stem}.tif"
                tmp = out_dir / f".{file_stem}.tif.tmp"
                write_tiff(str(tmp), array, metadata={"axes": axes})
                os.replace(tmp, target)
                written.append(str(target))
        return {"tiff": written}
    from .crops import MERGED_LAYOUT_SIDECAR

    layout = json.loads((merged / MERGED_LAYOUT_SIDECAR).read_text(encoding="utf-8"))
    by_plate: Dict[str, List[Tuple[str, int, Any]]] = collections.defaultdict(list)
    seen: Dict[Tuple[str, str, int], str] = {}
    for name in names:
        plate, well, field = split_field_name(name)
        if (plate, well, field) in seen:
            raise ValueError(
                f"{seen[(plate, well, field)]} and {name} are the same field "
                f"({plate} {well} {field}) at two timepoints; the plate store "
                f"has no time axis yet, so time-lapse fields are not exported")
        seen[(plate, well, field)] = name
        by_plate[plate].append((well, field, name))
    if len(by_plate) > 1:
        raise ValueError(
            f"{merged} holds fields of {len(by_plate)} plates "
            f"({', '.join(sorted(by_plate))}); one merged store holds one plate. "
            f"Give each plate its own project folder.")
    report: Dict[str, Any] = {}
    for plate, plan in sorted(by_plate.items()):
        plan.sort(key=lambda item: (item[0], item[1]))
        writer = _merged_writer(src, plate, layout, channel_names=channel_names)
        done = {entry.get("name") for entry in writer._committed.values()}
        todo = [name for _w, _f, name in plan if name not in done]
        for name in todo:
            _write_merged_field(writer, name, np.load(merged / name), layout)
        report[plate] = {"written": len(todo), "skipped": len(plan) - len(todo),
                         "keys": [field_key(w, f) for w, f, n in plan if n in todo]}
    return report


# ---------------------------------------------------------------------------
# The project's merged store (item 0aj): <project>/merged.ome.zarr
# ---------------------------------------------------------------------------

#: The merged plate store of a project folder: the field channels with every
#: object mask embedded as labels, one shard file per field image and label
#: image. Measure reads it in place of ``merged/*.npy`` (item 0aj).
MERGED_STORE_NAME = "merged.ome.zarr"


def _merged_store_path(project: Any) -> Path:
    """``<project>/merged.ome.zarr``."""
    return Path(project) / MERGED_STORE_NAME


def _compact_labels(mask: np.ndarray) -> np.ndarray:
    """A label image as uint16 or uint32, chosen by its largest label; exact.

    Integer labels from 0 to 65535 become uint16, larger ones uint32. A
    label image with negative or non-integer values, or labels above the
    uint32 range, keeps its dtype, so the stored labels always equal the
    input exactly.

    :param mask: the label image.
    :returns: the label image in its compact dtype.
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.astype(np.uint16)
    if mask.dtype.kind == "b":
        return mask.astype(np.uint16)
    if mask.dtype.kind not in "iuf":
        return mask
    low, high = mask.min(), mask.max()
    if low < 0:
        return mask
    if mask.dtype.kind == "f" and not np.array_equal(mask, np.floor(mask)):
        return mask
    if high <= np.iinfo(np.uint16).max:
        return mask.astype(np.uint16)
    if high <= np.iinfo(np.uint32).max:
        return mask.astype(np.uint32)
    return mask


def _merged_writer(project: Any, plate: str, layout: Mapping[str, Any], *,
                   channel_names: Optional[Sequence[str]] = None) -> PlateWriter:
    """The single writer of ``<project>/merged.ome.zarr`` for one plate.

    :param project: the project folder.
    :param plate: the plate name; one store holds one plate.
    :param layout: the merged plane layout (``intensity_channels``,
        ``mask_plane_order``, ``mask_dims``), kept in the store's ``spacr``
        attributes as ``merged_layout`` so readers rebuild the merged array.
    :param channel_names: names for the intensity channels.
    """
    return PlateWriter(project, plate, store_path=_merged_store_path(project),
                       shards=True, channel_names=channel_names,
                       attributes={"merged_layout": dict(layout)})


def _write_merged_field(writer: PlateWriter, name: str, stack: np.ndarray,
                        layout: Mapping[str, Any], *,
                        image_dtype: Any = None) -> str:
    """Write one merged field ``(Y, X, planes)`` into the merged store, then commit it.

    The intensity planes are stored as one ``(C, Y, X)`` image in
    ``image_dtype`` (the acquired dtype; the merged array's own dtype when
    None or when the cast would change a value) and every mask plane as a
    label image in :func:`_compact_labels`'s dtype. The ledger entry keeps
    the field's file ``name``, the merged ``dtype`` and ``shape``, so
    :func:`_read_field` rebuilds the array ``merged/<name>`` would have held,
    value for value.

    :param writer: the store's writer, from :func:`_merged_writer`.
    :param name: the field's file name, ``<plate>_<well>_<field>[_<time>].npy``.
    :param stack: the merged field.
    :param layout: the store's plane layout.
    :param image_dtype: the dtype to keep the intensity planes in.
    :returns: the field's store key.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"{name}: a merged field is (Y, X, planes), not {stack.shape}")
    n_intensity = len(layout.get("intensity_channels") or ())
    image = np.moveaxis(stack[..., :n_intensity], -1, 0)
    if image_dtype is not None and np.dtype(image_dtype) != image.dtype:
        narrowed = image.astype(image_dtype)
        if np.array_equal(narrowed.astype(image.dtype), image):
            image = narrowed
    labels = {role: _compact_labels(stack[..., int(dim)])
              for role, dim in (layout.get("mask_dims") or {}).items()
              if int(dim) < stack.shape[-1]}
    _plate, well, field = split_field_name(name)
    if not str(name).endswith(".npy"):
        name = f"{os.path.splitext(os.path.basename(str(name)))[0]}.npy"
    return writer.write_field(well, field, np.ascontiguousarray(image), labels,
                              record={"name": os.path.basename(str(name)),
                                      "dtype": stack.dtype.str,
                                      "shape": [int(v) for v in stack.shape]})


@functools.lru_cache(maxsize=64)
def _ledger_entries(ledger: str, modified: int, size: int) -> Dict[str, Dict[str, Any]]:
    """The last ledger entry of every named field, cached while the ledger is unchanged."""
    entries: Dict[str, Dict[str, Any]] = {}
    with open(ledger, encoding="utf-8") as handle:
        by_key = {}
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                by_key[entry["key"]] = entry
    for entry in by_key.values():
        if entry.get("name"):
            entries[str(entry["name"])] = entry
    return entries


def _store_entries(store: Any) -> Dict[str, Dict[str, Any]]:
    """``{field file name: ledger entry}`` of a merged store; empty when it has none."""
    ledger = Path(store) / _LEDGER
    try:
        stat = os.stat(ledger)
    except OSError:
        return {}
    return _ledger_entries(str(ledger), stat.st_mtime_ns, stat.st_size)


def _is_merged_store(path: Any) -> bool:
    """Whether ``path`` is a merged store (item 0aj): a plate store whose fields carry names."""
    target = Path(os.fspath(path))
    return (target.name.endswith(".ome.zarr") and _is_zarr_group(target)
            and bool(_store_entries(target)))


def _store_layout(store: Any) -> Optional[Dict[str, Any]]:
    """The merged plane layout kept in a merged store's ``spacr`` attributes, or None."""
    root_json = Path(store) / "zarr.json"
    try:
        attrs = json.loads(root_json.read_text(encoding="utf-8")).get("attributes", {})
    except (OSError, ValueError):
        return None
    layout = (attrs.get("spacr") or {}).get("merged_layout")
    return dict(layout) if isinstance(layout, dict) else None


def _list_fields(folder: Any) -> List[str]:
    """The field file names Measure and the overlays read from ``folder``.

    :param folder: a merged store (its committed field names, sorted) or a
        folder of ``.npy`` fields (its visible ``.npy`` names, in
        :func:`os.listdir` order, as before).
    """
    if _is_merged_store(folder):
        return sorted(_store_entries(folder))
    try:
        return [name for name in os.listdir(os.fspath(folder))
                if name.endswith(".npy") and not name.startswith(".")]
    except OSError:
        return []


def _store_field(path: Any) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """``(store, ledger entry)`` when ``path`` names a field inside a merged store."""
    target = Path(os.fspath(path))
    if target.is_file():
        return None
    entry = _store_entries(target.parent).get(target.name)
    if entry is None:
        return None
    return target.parent, entry


def _field_header(path: Any) -> Tuple[Tuple[int, ...], np.dtype]:
    """``(shape, dtype)`` of one field, from an ``.npy`` header or a store ledger entry."""
    found = _store_field(path)
    if found is not None:
        _store, entry = found
        return tuple(int(v) for v in entry["shape"]), np.dtype(entry["dtype"])
    header = np.load(os.fspath(path), mmap_mode="r")
    return tuple(header.shape), header.dtype


def _read_field(path: Any, *, mmap_mode: Optional[str] = None) -> np.ndarray:
    """One merged field ``(Y, X, planes)``: an ``.npy`` file or a merged-store field.

    A store field is rebuilt from its image and labels exactly as
    ``merged/<name>.npy`` held it: same shape, dtype, plane order and values.

    :param path: ``merged/<name>.npy``, or ``<project>/merged.ome.zarr/<name>.npy``
        (the field's name inside the store, which is not a file).
    :param mmap_mode: passed to :func:`numpy.load` for an ``.npy`` file.
    """
    found = _store_field(path)
    if found is None:
        return np.load(os.fspath(path), mmap_mode=mmap_mode)
    store, entry = found
    layout = _store_layout(store) or {}
    shape = tuple(int(v) for v in entry["shape"])
    data = np.empty(shape, dtype=np.dtype(entry["dtype"]))
    n_intensity = len(layout.get("intensity_channels") or ())
    group = store / entry["key"]
    if n_intensity:
        data[..., :n_intensity] = np.moveaxis(read_image(group), 0, -1)
    stored = set(entry.get("labels") or ())
    for role, dim in (layout.get("mask_dims") or {}).items():
        if int(dim) < shape[-1]:
            if role not in stored:
                raise ValueError(f"{store}: field {entry['name']} has no {role} labels")
            data[..., int(dim)] = read_labels(group, role)
    return data


def _field_stat(path: Any) -> os.stat_result:
    """``os.stat`` of one field: the ``.npy`` file, or a merged-store field's image shard.

    For a store field the size is the bytes of its image and label shard
    files and the time the newest of them, so a rewritten field reads as
    changed, as a rewritten ``.npy`` does.
    """
    found = _store_field(path)
    if found is None:
        return os.stat(os.fspath(path))
    store, entry = found
    group = store / entry["key"]
    stats = [os.stat(os.path.join(base, name))
             for base, _dirs, names in os.walk(group) for name in names]
    newest = max(stats, key=lambda item: item.st_mtime_ns)
    values = list(newest)
    values[6] = sum(item.st_size for item in stats)
    result = os.stat_result(values, {"st_mtime_ns": newest.st_mtime_ns})
    return result
