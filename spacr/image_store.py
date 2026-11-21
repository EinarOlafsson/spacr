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


def _field_chunks(shape: Sequence[int]) -> Tuple[int, ...]:
    """One chunk per channel, the whole field: spaCR reads fields whole."""
    return (1,) + tuple(int(v) for v in shape[1:]) if len(shape) == 3 else tuple(
        int(v) for v in shape)


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

    Fields are written whole with :meth:`write_field`; the plate and well
    metadata are rewritten from the ledger after each field, so a reader
    opening the store mid-run sees only committed fields. Each image channel
    and label image uses one unsharded whole-field chunk, compressed with
    Blosc LZ4 and byte shuffle.
    """

    def __init__(self, root: Any, plate: str, *,
                 channel_names: Optional[Sequence[str]] = None,
                 spacing=None, ngff_version: str = "0.5",
                 registration: Optional[Mapping[Any, Mapping[str, Any]]] = None):
        """Open (or start) the plate store and read its commit ledger."""
        from .ome_zarr import require_zarr
        from .qt.i18n import tr

        self.zarr = require_zarr()
        self.plate = str(plate)
        self.path = Path(root) / f"{self.plate}.ome.zarr"
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
            stored = (json.loads(root_json.read_text(encoding="utf-8"))
                      .get("attributes", {}).get("ome", {}).get("version"))
            if stored and str(stored) != self.ngff_version:
                raise ValueError(tr(
                    "{path} is an OME-Zarr {stored} store; it cannot take "
                    "OME-Zarr {wanted} fields, because one store holds one "
                    "version. Write to a new folder or convert the store.",
                    path=self.path, stored=stored, wanted=self.ngff_version))
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
                    labels: Optional[Mapping[str, np.ndarray]] = None) -> str:
        """Write one field and its label images, then commit it.

        :param image: the field ``(C, Y, X)`` or ``(Y, X)``, or None to write
            labels only.
        :param labels: ``{object class: label image (Y, X)}``.
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
                chunks=_field_chunks(data.shape), shards=False,
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
                chunks=_field_chunks(m.shape), shards=False, compressor=FIELD_CODEC,
                compression_level=FIELD_CODEC_LEVEL, **label_metadata))
            self._set_attrs(label_dir, {"image-label": (
                {"version": "0.5"} if self.ngff_version == "0.5" else {})})
            names.append(str(name))
        if names:
            self.zarr.open_group(str(target / "labels"), mode="a",
                                 zarr_format=3).attrs.update(
                {"ome": {"version": self.ngff_version, "labels": names}})
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
        """Merge ``extra`` into a 0.5 or 0.6 group's ``ome`` attributes."""
        group = self.zarr.open_group(str(group_path), mode="a", zarr_format=3)
        ome = dict(group.attrs.get("ome", {}))
        ome.update(extra)
        group.attrs.update({"ome": ome})

    def _commit(self, key: str, well: str, field: Any, labels: List[str],
                has_image: bool) -> None:
        """Record the field in the ledger and rewrite the plate metadata."""
        entry = {"key": key, "well": str(well), "field": int(field),
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
    :param fmt: ``'ome-zarr'``: one plate store per plate under
        ``<src>/images/``, with the masks as labels. ``'tiff'``:
        ``<src>/merged_tiff/<field>.tif`` (the image, planes first) and
        ``<field>_<object>_mask.tif`` beside it, each in its own dtype.
        ``'npy'`` writes nothing.
    :param workers: processes reading the merged stacks; the parent writes.
    :param channel_names: names for the intensity channels.
    :returns: per plate, what :func:`run_fields` reports; or ``tiff``, the
        files written.

    The merged ``.npy`` stacks and plane layout stay in place for field
    discovery and legacy readers. Measure reads committed spatial OME-Zarr
    fields through :func:`read_image` when selected, with the retained
    stacks as the fallback for missing or unsupported field layouts.
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
        by_plate[plate].append((well, field, (str(merged / name), intensity, mask_dims)))
    report: Dict[str, Any] = {}
    for plate, plan in sorted(by_plate.items()):
        plan.sort(key=lambda item: (item[0], item[1]))
        writer = PlateWriter(Path(src) / "images", plate,
                             channel_names=channel_names)
        report[plate] = run_fields(plan, _merged_field, writer,
                                   workers=workers, start_method=start_method)
    return report
