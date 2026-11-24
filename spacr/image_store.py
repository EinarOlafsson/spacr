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


#: The OME-Zarr version of every new store spaCR writes (item 0aj,
#: maintainer 2026-11-23: "0.6 is stable and more optimized"). Stores of
#: 0.4 to 0.6 are still read; an existing 0.5 store keeps its version.
DEFAULT_NGFF_VERSION = "0.6"


#: NGFF's axis order for images and for labels (item 0aj P3).
IMAGE_AXIS_ORDER = "tczyx"
LABEL_AXIS_ORDER = "tzyx"

#: The z reductions the shared reader computes on read (item 0aj P3).
#: ``'max'`` keeps the stored dtype and is what Mask has always segmented.
Z_PROJECTIONS: Tuple[str, ...] = ("max", "min", "mean", "sum")

#: The z reduction a reader applies when it names none (item 0aj P3).
DEFAULT_Z_PROJECTION = "max"


def _stored_ngff_version(path: Any) -> Optional[str]:
    """The ``ome`` version a store's root ``zarr.json`` declares, or None."""
    try:
        attrs = json.loads((Path(path) / "zarr.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = ((attrs.get("attributes") or {}).get("ome") or {}).get("version")
    return str(version) if version else None


class PlateWriter:
    """The only writer of one plate's OME-Zarr store (item 983).

    :param root: the folder that holds ``<plate>.ome.zarr``.
    :param plate: the plate name.
    :param channel_names: the image channels' names, written to ``omero``.
    :param spacing: a :class:`spacr.layers.Spacing` for the pixel size, or
        None to record pixels with no physical unit.
    :param ngff_version: ``'0.6'`` or ``'0.5'`` (item 1007): the OME-Zarr
        version of every group in the store. None (the default) takes the
        version of a store that already exists, so an older 0.5 store
        resumes as 0.5, and writes a new store as 0.6 (item 0aj: 0.6 is
        the default of every new store). One store holds one version, so a
        store started in the other one is refused when a version is given.
    :param registration: optional field-number to OPS registration-row mapping.
        Requires version 0.6 and pixel coordinates (spacing=None). Existing
        stores must use the same registration mode.
    :param store_path: the store folder itself, instead of
        ``<root>/<plate>.ome.zarr`` (item 0aj: ``<project>/fov.ome.zarr``).
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
                 spacing=None, ngff_version: Optional[str] = None,
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
        if ngff_version is None:
            ngff_version = _stored_ngff_version(self.path) or DEFAULT_NGFF_VERSION
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
                    record: Optional[Mapping[str, Any]] = None,
                    axes: Optional[str] = None,
                    label_axes: Optional[str] = None) -> str:
        """Write one field and its label images, then commit it.

        :param image: the field ``(C, Y, X)`` or ``(Y, X)``, or with ``axes``
            any of NGFF's ``t, c, z, y, x`` in that order (item 0aj P3), or
            None to write labels only.
        :param labels: ``{object class: label image}``, ``(Y, X)`` or with
            ``label_axes`` any of ``t, z, y, x``.
        :param record: extra JSON values kept with the field's ledger entry,
            such as the field's file ``name`` (item 0aj).
        :param axes: the image's axes, such as ``'tczyx'`` or ``'czyx'``;
            by default ``'cyx'`` (3-D) or ``'yx'`` (2-D), the layout of every
            2-D project, which records no axes in the ledger.
        :param label_axes: the label images' axes, such as ``'tyx'``; by
            default ``'yx'``.
        :returns: the field's store key.

        Each array is written to a temporary group, closed and renamed into
        place, and the field enters the ledger last, so a crash leaves an
        uncommitted field that the next run rewrites.

        Time and depth are their own NGFF axes, never flattened into
        channels; a field whose rank does not match its axes is refused
        before anything is written.

        :param well: the well name, e.g. ``'B03'``.
        :param field: the field number.
        :raises ValueError: the image or a label image does not have the
            rank of its axes, or the axes are not NGFF's order.
        """
        from .ome_zarr import write_ome_zarr

        image_axes = (None if image is None else
                      field_axes(np.ndim(image), axes, kind="image", where=f"field {well}/{field}"))
        names_axes = {name: field_axes(np.ndim(mask), label_axes, kind="labels",
                                       where=f"field {well}/{field}: label image {name!r}")
                      for name, mask in (labels or {}).items()}
        layered = (image_axes not in (None, "cyx", "yx")
                   or any(value != "yx" for value in names_axes.values()))
        metadata, label_metadata = {}, {}
        if self.registration is not None and layered:
            raise ValueError(f"field {well}/{field}: stored registration is written for "
                             f"(C, Y, X) fields only, not {image_axes}")
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
            self._atomic(target, lambda tmp: write_ome_zarr(
                tmp, data, axes=list(image_axes), spacing=self._spacing_for(image_axes),
                channel_names=self.channel_names if "c" in image_axes else None,
                ngff_version=self.ngff_version, name=key,
                shards=None if self.shards else False,
                compressor=FIELD_CODEC, compression_level=FIELD_CODEC_LEVEL, **metadata))
        else:
            self.zarr.open_group(str(target), mode="w", zarr_format=3)
        names = self._write_label_arrays(target, labels, label_metadata, label_axes)
        if layered:
            record = dict(record or {})
            if image_axes not in (None, "cyx", "yx"):
                record["axes"] = image_axes
            if any(value != "yx" for value in names_axes.values()):
                record["label_axes"] = next(v for v in names_axes.values() if v != "yx")
        self._commit(key, well, field, names, image is not None, record)
        return key

    def _spacing_for(self, axes: str):
        """The writer's spacing for an image with these axes (item 0aj P3).

        A 2-D field keeps the writer's spacing as before. A field with a
        ``z`` axis and no spacing is written on a pixel grid over its space
        axes; a spacing over ``(y, x)`` only gets ``z`` added at 1 pixel.
        """
        if axes in ("cyx", "yx"):
            return self.spacing
        from .layers import Spacing
        from .ome_zarr import PIXEL_UNITS

        space = [a for a in axes if a in "zyx"]
        if self.spacing is None:
            return Spacing.isotropic(len(space), 1.0, units=PIXEL_UNITS)
        if len(space) == getattr(self.spacing, "ndim", len(space)):
            return self.spacing
        values = {a: 1.0 for a in space}
        values.update({a: float(v) for a, v in zip(self.spacing.axes, self.spacing.scale)})
        return Spacing.from_map({a: values[a] for a in space}, units=self.spacing.units)

    def _write_label_arrays(self, target: Path, labels: Optional[Mapping[str, np.ndarray]],
                            label_metadata: Mapping[str, Any],
                            label_axes: Optional[str] = None) -> List[str]:
        """Write each label image under ``target/labels`` atomically; returns their names."""
        from .ome_zarr import write_ome_zarr

        names = []
        for name, mask in (labels or {}).items():
            mask = np.asarray(mask)
            axes = field_axes(mask.ndim, label_axes, kind="labels", where=str(name))
            label_dir = target / "labels" / str(name)
            label_dir.parent.mkdir(parents=True, exist_ok=True)
            self._atomic(label_dir, lambda tmp, m=mask, a=axes: write_ome_zarr(
                tmp, m, axes=list(a), spacing=self._spacing_for(a),
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
        return names

    def write_labels(self, well: str, field: Any, labels: Mapping[str, np.ndarray], *,
                     record: Optional[Mapping[str, Any]] = None,
                     keep_others: bool = False,
                     unchanged: Iterable[str] = (),
                     label_axes: Optional[str] = None) -> str:
        """Add label images to a committed field, never touching its channels (item 0aj P1c).

        The field's image arrays are left as they are, so their recorded
        checksums still hold; labels of a class not in ``labels`` (left by an
        earlier run) are removed. The field is committed again with its
        earlier ledger values updated by ``record``.

        With ``keep_others`` (item 0aj P2, one object mask as a Mask stage
        writes it) the field's other labels stay, and the ledger is left
        alone: each label image is renamed into place whole, so its presence
        is its commit, and several processes may add different classes to
        one store at once. The merge commits the field's final labels.

        :param well: the well name.
        :param field: the field number.
        :param labels: ``{object class: label image}``, ``(Y, X)``, or with
            ``label_axes`` (item 0aj P3) any of ``t, z, y, x``, such as one
            ``(T, Y, X)`` array for a time-lapse field.
        :param record: ledger values to add, such as the merged ``shape``.
        :param keep_others: keep the other labels and do not commit.
        :param unchanged: classes of ``labels`` the field already holds with
            these exact values (item 0aj P2); their committed label images
            are kept instead of written again.
        :param label_axes: the label images' axes; ``'yx'`` by default.
        :returns: the field's store key.
        :raises ValueError: the field has no committed image.
        """
        key = _field_key(well, field)
        entry = self._committed.get(key)
        if not entry or not entry.get("image"):
            raise ValueError(f"field {well}/{field} has no committed channels to add labels to")
        axes_seen = {field_axes(np.ndim(mask), label_axes, kind="labels",
                                where=f"field {well}/{field}: label image {name!r}")
                     for name, mask in labels.items()}
        target = self.path / key
        if keep_others:
            self._write_label_arrays(target, labels, {}, label_axes)
            present = sorted(group.name for group in (target / "labels").iterdir()
                             if (group / "zarr.json").is_file()
                             and not group.name.startswith("."))
            self.zarr.open_group(str(target / "labels"), mode="a",
                                 zarr_format=3).attrs.update(
                {"ome": {"version": self.ngff_version, "labels": present}})
            return key
        stale = target / "labels"
        if stale.is_dir():
            for old in stale.iterdir():
                if old.is_dir() and old.name not in {str(n) for n in labels}:
                    shutil.rmtree(old)
        kept_labels = {str(n) for n in unchanged
                       if str(n) in {str(m) for m in labels}
                       and (stale / str(n) / "zarr.json").is_file()}
        names = self._write_label_arrays(
            target, {n: m for n, m in labels.items() if str(n) not in kept_labels}, {},
            label_axes)
        if kept_labels:
            names = [str(n) for n in labels]
            self.zarr.open_group(str(target / "labels"), mode="a",
                                 zarr_format=3).attrs.update(
                {"ome": {"version": self.ngff_version, "labels": names}})
        kept = {k: v for k, v in entry.items()
                if k not in ("key", "well", "field", "labels", "image", "label_axes")}
        if axes_seen - {"yx"}:
            kept["label_axes"] = sorted(axes_seen - {"yx"})[0]
        kept.update(json.loads(json.dumps(dict(record or {}), default=str)))
        self._commit(key, well, field, names, True, kept)
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
        ``<src>/fov.ome.zarr`` (item 0aj), one shard file per field image
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
    plate, so a folder of several plates is refused. The timepoints of a
    time-lapse field become one field with a t axis (item 0aj P3).
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
    frames_of: Dict[Tuple[str, str, int], List[str]] = collections.defaultdict(list)
    for name in names:
        plate, well, field = split_field_name(name)
        if not frames_of[(plate, well, field)]:
            by_plate[plate].append((well, field, name))
        frames_of[(plate, well, field)].append(name)
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
        done |= {f for entry in writer._committed.values() for f in entry.get("frames") or ()}
        todo = [name for _w, _f, name in plan if name not in done]
        for name in todo:
            series = frames_of[split_field_name(name)]
            if len(series) > 1:
                # A time-lapse field (item 0aj P3): its frames on one t axis.
                _write_merged_series(writer, series, [np.load(merged / f) for f in series],
                                     layout)
            else:
                _write_merged_field(writer, name, np.load(merged / name), layout)
        report[plate] = {"written": len(todo), "skipped": len(plan) - len(todo),
                         "keys": [field_key(w, f) for w, f, n in plan if n in todo]}
    return report



#: The merged plate store of a project folder: the field channels with every
#: object mask embedded as labels, one shard file per field image and label
#: image. Measure reads it in place of ``merged/*.npy`` (item 0aj).
MERGED_STORE_NAME = "fov.ome.zarr"


def _merged_store_path(project: Any) -> Path:
    """``<project>/fov.ome.zarr``."""
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
    """The single writer of ``<project>/fov.ome.zarr`` for one plate.

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
                        image_dtype: Any = None,
                        stored_labels: Iterable[str] = ()) -> str:
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
    :param stored_labels: classes whose label the field already holds with
        exactly these values (item 0aj P2: the merge read them from there);
        they are kept, not written again.
    :returns: the field's store key (None for a time-lapse frame whose
        field still waits for other frames, item 0aj P3).
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
    series = _layered_entry(writer, name)
    if series is not None:
        # A time-lapse field (item 0aj P3): its frames' masks become (T, Y, X) labels.
        return _write_merged_frame(writer, name, stack, layout, series)
    stored = writer._committed.get(field_key(well, field)) or {}
    if (stored.get("image_planes") == STACK_PLANES
            and stored.get("name") == os.path.basename(str(name))):
        return writer.write_labels(well, field, labels, record={
            "dtype": stack.dtype.str, "shape": [int(v) for v in stack.shape]},
            unchanged=stored_labels)
    image = np.ascontiguousarray(image)
    return writer.write_field(well, field, image, labels,
                              record={"name": os.path.basename(str(name)),
                                      "dtype": stack.dtype.str,
                                      "shape": [int(v) for v in stack.shape],
                                      "image_dtype": image.dtype.str,
                                      "plane_sha256": _plane_hashes(image)})


def _plane_hashes(image: np.ndarray) -> List[str]:
    """The sha256 of each channel plane's raw bytes (C-order), recorded at write (item 0aj).

    :param image: the field's channels ``(C, Y, X)``; a field with t or z
        axes (item 0aj P3) is hashed plane by plane over every t, c and z,
        in C order of its leading axes.
    """
    import hashlib

    image = np.asarray(image)
    planes = image if image.ndim == 3 else _as_planes(image)
    return [hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest()
            for plane in planes]


def verify_store(store: Any, originals: Optional[Mapping[str, Sequence[np.ndarray]]] = None
                 ) -> Dict[str, Any]:
    """Re-hash every committed field's channel planes against what was recorded (item 0aj).

    The channels of ``fov.ome.zarr`` are the acquired pixels, byte for byte;
    this is the check every run ends with and that
    ``keep_original_images=False`` passes before it deletes an original.

    :param store: the project's ``fov.ome.zarr`` (or the project folder).
    :param originals: optional ``{field file name: [plane, ...]}`` to compare
        the stored planes with as well (value, dtype and shape); a field with
        z planes lists them channel by channel, each channel's planes in z
        order (item 0aj P3).
    :returns: ``{"fields": n, "planes": n, "mismatches": [...], "unrecorded": [...]}``;
        a field written before hashes were recorded is listed as unrecorded.
    """
    target = Path(store)
    if not (target.name.endswith(".ome.zarr") and _image_entries(target)):
        target = _merged_store_path(target)
    report: Dict[str, Any] = {"fields": 0, "planes": 0, "mismatches": [], "unrecorded": []}
    for name, entry in sorted(_image_entries(target).items()):
        if not entry.get("image"):
            continue
        # Every stored plane: each frame's channels with their raw z planes (item 0aj P3).
        image = _as_planes(read_frame_channels(target / entry["key"], entry, z_projection=None))
        report["fields"] += 1
        recorded = entry.get("plane_sha256")
        if not recorded:
            report["unrecorded"].append(name)
        else:
            for index, (want, got) in enumerate(zip(recorded, _plane_hashes(image))):
                report["planes"] += 1
                if want != got:
                    report["mismatches"].append(f"{name} plane {index}: checksum")
            if len(recorded) != image.shape[0]:
                report["mismatches"].append(f"{name}: {image.shape[0]} planes, {len(recorded)} recorded")
        for index, plane in enumerate((originals or {}).get(name, ())):
            plane = np.asarray(plane)
            if (index >= image.shape[0] or plane.dtype != image.dtype
                    or plane.shape != image.shape[1:] or not np.array_equal(plane, image[index])):
                report["mismatches"].append(f"{name} plane {index}: differs from the original")
    return report


def _mask_settings_of(project: Path) -> Dict[str, Any]:
    """The Mask settings a project recorded (``settings/gen_mask_settings``), or ``{}``."""
    folder = project / "settings"
    for name in ("gen_mask_settings.csv", "gen_mask_settings.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                return dict(json.loads(path.read_text(encoding="utf-8")))
            from .utils import load_settings

            return dict(load_settings(str(path)))
        except Exception as exc:
            _LOG.info("could not read %s: %s", path, exc)
    return {}


def original_planes(project: Any, settings: Optional[Mapping[str, Any]] = None
                    ) -> Tuple[Dict[str, List[Callable[[], np.ndarray]]], List[str]]:
    """The acquired images of a project's fields, plane by plane as fov.ome.zarr keeps them (item 0aj).

    The raw images in the project folder and in ``orig/`` are grouped with
    the filename pattern of the recorded Mask settings, by the same
    function Mask stacking uses, and ordered as the store keeps its
    channels (the sorted channel names, then the run's ``channels``).

    :param project: the project folder.
    :param settings: Mask settings naming ``metadata_type`` and
        ``custom_regex``; by default the ones the project recorded.
    :returns: ``({field file name: [loader of each plane]}, notes)``. A
        loader reads its file only when called. A field stored with its z
        planes (item 0aj P3) lists each channel's slices in z order; a field
        whose channel was projected from several files (an older store) has
        no single original plane, so it is left out with a note.
    """
    project = Path(project)
    fields, sorted_channels, notes = _original_files(project, settings)
    store = _merged_store_path(project)
    layout = _store_layout(store) or {}
    entries = _image_entries(store)
    planes: Dict[str, List[Callable[[], np.ndarray]]] = {}
    for name, by_channel in sorted(fields.items()):
        entry = entries.get(name) or {}
        depth = _frame_depth(entry)
        if any(len(paths) != depth for paths in by_channel.values()):
            notes.append(f"{name[:-4]}: a channel is projected from several files; "
                         f"no single original plane to compare")
            continue
        order = [c for c in _source_channels(store, sorted_channels, entry) if c in by_channel]
        if entry.get("image_planes") != "stack" and layout.get("intensity_channels") is not None:
            order = [order[int(i)] for i in layout["intensity_channels"] if int(i) < len(order)]
        # Raw z planes (item 0aj P3): each channel's files, in z order.
        planes[name] = [functools.partial(read_image, path)
                        for c in order for path in by_channel[c]]
    return planes, notes


def _original_files(project: Any, settings: Optional[Mapping[str, Any]] = None,
                    source_plate: Optional[str] = None
                    ) -> Tuple[Dict[str, Dict[str, List[str]]], List[str], List[str]]:
    """The acquired image files of a project's fields, by field and channel (item 0aj).

    The raw images in the project folder and in ``orig/`` are grouped with
    the filename pattern of the Mask settings, by the same function Mask
    stacking uses (:func:`spacr.io._raw_images_by_key`).

    :param project: the project folder.
    :param settings: Mask settings naming ``metadata_type`` and
        ``custom_regex``; by default the ones the project recorded.
    :param source_plate: the plate name when it is not the folder name (a
        test-mode sample folder).
    :returns: ``({field file name: {channel id: [paths]}}, the sorted
        channel ids, notes)``; channel ids are strings.
    """
    import re

    from .io import _RAW_IMAGE_SUFFIXES, _escaped_field_stem, _raw_image_names, _raw_images_by_key
    from .utils import _get_regex

    project = Path(project)
    settings = dict(settings if settings is not None else _mask_settings_of(project))
    names = (_raw_image_names(project, _RAW_IMAGE_SUFFIXES)
             + _raw_image_names(project / "orig", _RAW_IMAGE_SUFFIXES))
    if not names:
        return {}, [], []
    if not settings.get("metadata_type"):
        return {}, [], [f"{project}: no recorded Mask settings name the file pattern; "
                        f"the originals were not compared"]
    ext = collections.Counter(n.rsplit(".", 1)[-1].lower() for n in names).most_common(1)[0][0]
    regex = re.compile(_get_regex(settings["metadata_type"], ext, settings.get("custom_regex")))
    by_key = _raw_images_by_key(str(project), regex, settings["metadata_type"],
                                list(_RAW_IMAGE_SUFFIXES),
                                metadata_well_map=settings.get("metadata_well_map"),
                                source_plate=source_plate)
    sorted_channels = [str(c) for c in sorted({key[3] for key in by_key})]
    fields: Dict[str, Dict[str, List[str]]] = collections.defaultdict(dict)
    # Each channel's files in z order, as Mask stores the planes (item 0aj P3).
    for key, paths in sorted(by_key.items(), key=lambda item: z_order(item[0])):
        stem = _escaped_field_stem(key[0], key[1], key[2], key[4])
        fields[f"{stem}.npy"].setdefault(str(key[3]), []).extend(paths)
    return dict(fields), sorted_channels, []


def _source_channels(store: Path, found: Sequence[str],
                     entry: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The channel ids of a field's stored planes, in plane order (item 0aj).

    A field written by Mask preprocessing records its own (``source_channels``
    in its ledger entry). Otherwise the plate's sorted channel ids are used,
    recorded in the store the first time they are needed, so that a field
    whose originals were partly deleted still maps each remaining file to
    its plane.

    :param store: the project's ``fov.ome.zarr``.
    :param found: the sorted channel ids of the originals on disk now.
    :param entry: the field's ledger entry.
    """
    recorded = (entry or {}).get("source_channels")
    if recorded:
        return [str(c) for c in recorded]
    stored = _store_spacr_attrs(store).get("source_channels")
    if stored:
        return [str(c) for c in stored]
    if found and _is_zarr_group(Path(store)):
        _update_store_spacr_attrs(store, {"source_channels": list(found)})
    return list(found)


def _store_spacr_attrs(store: Any) -> Dict[str, Any]:
    """A store root's ``spacr`` attributes; empty when it has none."""
    try:
        attrs = json.loads((Path(store) / "zarr.json").read_text(encoding="utf-8")).get(
            "attributes", {})
    except (OSError, ValueError):
        return {}
    return dict(attrs.get("spacr") or {})


def _update_store_spacr_attrs(store: Any, values: Mapping[str, Any]) -> None:
    """Merge ``values`` into a store root's ``spacr`` attributes."""
    from .ome_zarr import require_zarr

    group = require_zarr().open_group(str(store), mode="a", zarr_format=3)
    stored = dict(group.attrs.get("spacr") or {})
    stored.update(json.loads(json.dumps(dict(values), default=str)))
    group.attrs.update({"spacr": stored})



def _same_bytes(first: Path, second: Path, chunk: int = 1 << 20) -> bool:
    """Whether two files hold the same bytes (read now, never from a cache)."""
    if first.stat().st_size != second.stat().st_size:
        return False
    with open(first, "rb") as a, open(second, "rb") as b:
        while True:
            x, y = a.read(chunk), b.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def _move_without_overwrite(source: Any, target: Any) -> str:
    """Move one file to ``target``, never replacing a file that is already there.

    The move is a hard link plus an unlink where the file system allows it,
    so an existing ``target`` makes it fail instead of being replaced, and a
    move cut short leaves the file under both names, never under neither.
    A ``target`` that is already the same file (a move interrupted between
    the link and the unlink) is finished. A different file at ``target``
    with the same bytes counts as already moved, and ``source`` is removed.

    :param source: the file to move.
    :param target: its new path; the folder is created.
    :returns: ``'moved'`` or ``'already there'``.
    :raises FileExistsError: a different file of that name is at ``target``.
    """
    import errno

    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target) or _same_bytes(source, target):
            os.remove(source)
            return "already there"
        raise FileExistsError(f"{target} already exists and differs from {source}; "
                              f"nothing was moved")
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError(f"{target} appeared while {source} was moved; "
                                  f"nothing was replaced") from exc
        os.rename(source, target)
        return "moved"
    os.remove(source)
    return "moved"


def _field_checksum_problems(name: str, entry: Mapping[str, Any], image: np.ndarray) -> List[str]:
    """The planes of one stored field whose bytes differ from the recorded sha256."""
    recorded = entry.get("plane_sha256")
    if not recorded:
        return [f"{name}: no checksums were recorded for its planes"]
    problems = [f"{name} plane {i}: checksum" for i, (want, got)
                in enumerate(zip(recorded, _plane_hashes(image))) if want != got]
    if len(recorded) != image.shape[0]:
        problems.append(f"{name}: {image.shape[0]} planes, {len(recorded)} recorded")
    return problems


def _stored_plane(entry: Mapping[str, Any], layout: Mapping[str, Any],
                  channels: Sequence[str], channel: str) -> Optional[int]:
    """The plane of a field's stored image that holds ``channel``, or None.

    :param entry: the field's ledger entry.
    :param layout: the store's merged plane layout.
    :param channels: the field's channel ids in stack order.
    :param channel: one channel id.
    """
    if str(channel) not in channels:
        return None
    position = channels.index(str(channel))
    if entry.get("image_planes") == "stack":
        return position
    intensity = [int(i) for i in (layout.get("intensity_channels") or ())]
    return intensity.index(position) if position in intensity else None


def _delete_original(path: Any) -> None:
    """Delete one original image whose field was verified (the one place spaCR does)."""
    os.remove(path)


def _settle_originals(project: Any, *, keep: bool = True,
                     settings: Optional[Mapping[str, Any]] = None,
                     names: Optional[Iterable[str]] = None,
                     printer: Callable[[str], Any] = print,
                     source_plate: Optional[str] = None) -> Dict[str, Any]:
    """Move or delete each field's original images once its stored copy is verified (item 0aj).

    For every field committed to ``fov.ome.zarr`` (or only ``names``), the
    stored planes are re-hashed against the sha256 recorded at write, and
    every original image of the field is read and compared with its stored
    plane (values, dtype and shape). Only a field that passes both is
    settled:

    - ``keep=True`` (``keep_original_images``, the default): its originals
      are moved into ``<project>/orig/``, never over another file.
    - ``keep=False``: its originals are deleted, one file at a time, each
      only after that check.

    A field that fails is left exactly as it is and named in ``problems``.
    A field whose channel was projected from several files (z slices), or
    whose channel is not in the store, has no single plane to compare an
    original with, so its originals are moved to ``orig/`` even when
    ``keep`` is False, and named in ``kept``. Every step can be cut short
    and run again: a deleted original is never needed again, because its
    plane was verified first, and the others are checked anew.

    :param project: the project folder.
    :param keep: move (True) or delete (False) the verified originals.
    :param settings: the Mask settings naming the filename pattern; by
        default the ones the project recorded.
    :param names: only these field file names; every committed field by default.
    :param printer: where the one summary line goes.
    :param source_plate: the plate name when it is not the folder name.
    :returns: ``{"moved": [...], "deleted": [...], "kept": [...], "problems": [...]}``.
    """
    from .qt.i18n import tr

    project = Path(project)
    store = _merged_store_path(project)
    report: Dict[str, Any] = {"moved": [], "deleted": [], "kept": [], "problems": []}
    entries = _image_entries(store)
    if not entries:
        return report
    wanted = None if names is None else {os.path.basename(str(n)) for n in names}
    files, found_channels, _notes = _original_files(project, settings, source_plate)
    layout = _store_layout(store) or {}
    orig = project / "orig"
    for name, entry in sorted(entries.items()):
        if wanted is not None and name not in wanted:
            continue
        by_channel = files.get(name)
        if not by_channel:
            continue
        raw = read_frame_channels(store / entry["key"], entry, z_projection=None)
        problems = _field_checksum_problems(name, entry, _as_planes(raw))
        # The stored planes of each channel: one, or its raw z planes (item 0aj P3).
        image = raw if "z" in _entry_axes(entry) else raw[:, None]
        reason = None
        channels = _source_channels(store, found_channels, entry)
        if not problems:
            for channel, paths in sorted(by_channel.items()):
                plane = _stored_plane(entry, layout, channels, channel)
                if plane is None or plane >= image.shape[0]:
                    reason = tr("channel {channel} is not stored", channel=channel)
                    continue
                if len(paths) != image.shape[1]:
                    reason = tr("channel {channel} was projected from {count} files",
                                channel=channel, count=len(paths))
                    continue
                for depth, path in enumerate(paths):
                    original = np.asarray(read_image(path))
                    stored = image[plane, depth]
                    if (original.dtype != stored.dtype or original.shape != stored.shape
                            or not np.array_equal(original, stored)):
                        problems.append(f"{name} channel {channel}: {path} differs "
                                        f"from stored plane {plane}"
                                        + (f" z {depth}" if image.shape[1] > 1 else ""))
        if problems:
            report["problems"].extend(problems)
            continue
        paths = sorted(p for ps in by_channel.values() for p in ps)
        if keep or reason is not None:
            for path in paths:
                if Path(path).parent != orig:
                    _move_without_overwrite(path, orig / Path(path).name)
                    report["moved"].append(path)
            if reason is not None and not keep:
                report["kept"].append(f"{name}: {reason}")
        else:
            for path in paths:
                _delete_original(path)
                report["deleted"].append(path)
    if report["moved"] or report["deleted"] or report["kept"]:
        printer(tr("Original images: {moved} moved to orig/, {deleted} deleted, "
                   "after each field was verified in fov.ome.zarr; {kept} field(s) "
                   "kept because a plane could not be compared.",
                   moved=len(report["moved"]), deleted=len(report["deleted"]),
                   kept=len(report["kept"])))
    for line in report["kept"]:
        printer(tr("kept in orig/: {line}", line=line))
    return report


class _LazyPlanes:
    """Planes read one at a time from their loaders, each time it is iterated."""

    def __init__(self, loaders: Sequence[Callable[[], np.ndarray]]):
        """Keep the loaders; nothing is read until the planes are iterated.

        :param loaders: one callable per plane, each returning that plane.
        """
        self._loaders = list(loaders)

    def __len__(self) -> int:
        """The number of planes."""
        return len(self._loaders)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Each plane in order, read from its loader as it is reached."""
        return (load() for load in self._loaders)


def verify_project(project: Any, *, originals: bool = True) -> Dict[str, Any]:
    """``spacr store verify``: the recorded checksums and, when present, the originals (item 0aj).

    :param project: the project folder or its ``fov.ome.zarr``.
    :param originals: also compare every stored plane with its original
        image, when the originals are in the project folder or ``orig/``.
    :returns: :func:`verify_store`'s report plus ``originals_compared``
        (planes compared with an original) and ``notes``.
    """
    target = Path(project)
    if target.name.endswith(".ome.zarr") and _image_entries(target):
        target = target.parent
    store = _merged_store_path(target)
    if not _image_entries(store):
        raise FileNotFoundError(f"{store} is not a spaCR fov.ome.zarr with committed fields")
    loaders, notes = original_planes(target) if originals else ({}, [])
    entries = _image_entries(store)
    compared = {name: _LazyPlanes(planes) for name, planes in loaders.items()
                if name in entries}
    for name in sorted(set(loaders) - set(compared)):
        notes.append(f"{name}: an original with no field in the store")
    report = verify_store(store, compared)
    report["originals_compared"] = sum(len(v) for v in compared.values())
    report["notes"] = notes
    return report


#: ``{ledger path: (mtime_ns, size, entries)}``: ONE parsed copy per ledger.
#: An lru_cache keyed by the modification time kept a copy per commit, so a
#: Mask run held memory that grew with the plate (test 622).
_LEDGER_CACHE: "collections.OrderedDict[str, Tuple[int, int, Dict[str, Dict[str, Any]]]]" = (
    collections.OrderedDict())
_LEDGER_CACHE_SIZE = 16


def _ledger_entries(ledger: str, modified: int, size: int) -> Dict[str, Dict[str, Any]]:
    """The last ledger entry of every named field, cached while the ledger is unchanged."""
    cached = _LEDGER_CACHE.get(ledger)
    if cached is not None and cached[:2] == (modified, size):
        _LEDGER_CACHE.move_to_end(ledger)
        return cached[2]
    entries = _parse_ledger(ledger)
    _LEDGER_CACHE[ledger] = (modified, size, entries)
    _LEDGER_CACHE.move_to_end(ledger)
    while len(_LEDGER_CACHE) > _LEDGER_CACHE_SIZE:
        _LEDGER_CACHE.popitem(last=False)
    return entries


def _parse_ledger(ledger: str) -> Dict[str, Dict[str, Any]]:
    """``{field file name: last ledger entry}`` read from a commit ledger."""
    entries: Dict[str, Dict[str, Any]] = {}
    with open(ledger, encoding="utf-8") as handle:
        by_key = {}
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                by_key[entry["key"]] = entry
    for entry in by_key.values():
        if entry.get("frames"):
            # A time-lapse field (item 0aj P3): one view per frame name.
            entries.update(_frame_views(entry))
        elif entry.get("name"):
            entries[str(entry["name"])] = entry
    return entries


def _entry_views(store: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{"merged": {...}, "image": {...}}`` of a store's ledger, cached while it is unchanged.

    A field Mask preprocessing wrote (item 0aj P1c) holds its channels
    only until its masks are added; it is an ``image`` entry, and becomes a
    ``merged`` one (with the merged ``shape``) once its labels are committed.
    """
    ledger = Path(store) / _LEDGER
    try:
        stat = os.stat(ledger)
    except OSError:
        return {"merged": {}, "image": {}}
    entries = _ledger_entries(str(ledger), stat.st_mtime_ns, stat.st_size)
    cached = _VIEW_CACHE.get(str(ledger))
    if cached is not None and cached[0] is entries:
        return cached[1]
    views = {"merged": {n: e for n, e in entries.items() if "shape" in e},
             "image": {n: e for n, e in entries.items() if e.get("image")}}
    _VIEW_CACHE[str(ledger)] = (entries, views)
    while len(_VIEW_CACHE) > _LEDGER_CACHE_SIZE:
        _VIEW_CACHE.pop(next(iter(_VIEW_CACHE)))
    return views


#: ``{ledger path: (parsed entries, their views)}``, beside :data:`_LEDGER_CACHE`.
_VIEW_CACHE: Dict[str, Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]] = {}


def _store_entries(store: Any) -> Dict[str, Dict[str, Any]]:
    """``{field file name: ledger entry}`` of a store's MERGED fields; empty when it has none.

    A field whose channels are stored but whose masks are not yet (item 0aj
    P1c) is not listed: Measure, the overlays and resume see a field only
    once its labels are committed.
    """
    return _entry_views(store)["merged"]


def _image_entries(store: Any) -> Dict[str, Dict[str, Any]]:
    """``{field file name: ledger entry}`` of every store field with stored channels (item 0aj)."""
    return _entry_views(store)["image"]


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
        :func:`os.listdir` order, as before); a project's ``stack`` path
        without ``.npy`` files lists the field stacks in fov.ome.zarr.
    """
    if _is_merged_store(folder):
        return sorted(_store_entries(folder))
    try:
        names = [name for name in os.listdir(os.fspath(folder))
                 if name.endswith(".npy") and not name.startswith(".")]
    except OSError:
        names = []
    if not names and Path(os.fspath(folder)).name == "stack":
        return _stack_names(folder)
    return names


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


def _read_field(path: Any, *, mmap_mode: Optional[str] = None,
                z_projection: Any = DEFAULT_Z_PROJECTION) -> np.ndarray:
    """One merged field ``(Y, X, planes)``: an ``.npy`` file or a merged-store field.

    A store field is rebuilt from its image and labels exactly as
    ``merged/<name>.npy`` held it: same shape, dtype, plane order and values.
    A time-lapse frame is read by its own name; a field stored with its z
    planes is reduced on read (item 0aj P3).

    :param path: ``merged/<name>.npy``, or ``<project>/fov.ome.zarr/<name>.npy``
        (the field's name inside the store, which is not a file).
    :param mmap_mode: passed to :func:`numpy.load` for an ``.npy`` file.
    :param z_projection: the z reduction of the channels (see
        :func:`project_z`); ``'max'``, the default, is the projection Mask
        segmented, so the merged field equals the one Mask used to store.
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
        image = read_frame_channels(group, entry, z_projection=z_projection)
        planes = np.moveaxis(image[_image_planes(entry, layout)], 0, -1)
        height, width = planes.shape[:2]
        if (height, width) != shape[:2]:
            data[..., :n_intensity] = 0
        data[:height, :width, :n_intensity] = planes
    stored = set(entry.get("labels") or ())
    for role, dim in (layout.get("mask_dims") or {}).items():
        if int(dim) < shape[-1]:
            if role not in stored:
                raise ValueError(f"{store}: field {entry['name']} has no {role} labels")
            data[..., int(dim)] = read_frame_labels(group, entry, role)
    return data


def _image_planes(entry: Mapping[str, Any], layout: Mapping[str, Any]) -> List[int]:
    """The stored image planes that are the merged field's intensity planes, in order (item 0aj).

    A field Mask preprocessing stored keeps every acquired channel
    (``image_planes='stack'``), so the run's ``channels`` select from it; a
    field written whole by the merge holds just those planes.
    """
    n_intensity = len(layout.get("intensity_channels") or ())
    if entry.get("image_planes") == STACK_PLANES:
        return [int(c) for c in layout.get("intensity_channels") or ()]
    return list(range(n_intensity))


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



#: The ledger value of ``image_planes`` for a field whose stored image is
#: the whole acquired stack (every channel, in sorted channel order), as
#: Mask preprocessing writes it (item 0aj P1c).
STACK_PLANES = "stack"


def _stack_store(stack_dir: Any) -> Path:
    """The fov.ome.zarr beside a project's ``stack/`` folder."""
    return _merged_store_path(Path(os.fspath(stack_dir)).parent)


def _stack_entries(stack_dir: Any) -> Dict[str, Dict[str, Any]]:
    """``{field file name: entry}`` of the whole-stack fields in the project's fov.ome.zarr."""
    return {name: entry for name, entry in _image_entries(_stack_store(stack_dir)).items()
            if entry.get("image_planes") == STACK_PLANES}


def _stack_folder_names(stack_dir: Any) -> List[str]:
    """The visible ``.npy`` names of a real ``stack/`` folder, sorted."""
    try:
        names = os.listdir(os.fspath(stack_dir))
    except OSError:
        return []
    return sorted(n for n in names if n.endswith(".npy") and not n.startswith("."))


def _stack_names(stack_dir: Any) -> List[str]:
    """The field stacks of a project, ``<plate>_<well>_<field>_<time>.npy``, sorted (item 0aj).

    THE one list of a project's field stacks for every Mask stage. An older
    project's ``stack/*.npy`` files are listed when the folder holds any;
    otherwise the fields Mask preprocessing committed to ``fov.ome.zarr``.

    :param stack_dir: the project's ``stack/`` path (it need not exist).
    """
    names = _stack_folder_names(stack_dir)
    return names if names else sorted(_stack_entries(stack_dir))


def _stack_exists(stack_dir: Any) -> bool:
    """Whether a project has field stacks: a ``stack/`` folder, or stack fields in fov.ome.zarr."""
    return os.path.isdir(os.fspath(stack_dir)) or bool(_stack_entries(stack_dir))


def _stack_entry(path: Any) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """``(store, entry)`` when ``stack/<name>`` is not a file but a field in fov.ome.zarr."""
    target = Path(os.fspath(path))
    if target.is_file():
        return None
    entry = _stack_entries(target.parent).get(target.name)
    return None if entry is None else (_stack_store(target.parent), entry)


def _read_stack(path: Any, *, mmap_mode: Optional[str] = None,
                z_projection: Any = DEFAULT_Z_PROJECTION) -> np.ndarray:
    """One field stack ``(Y, X, channels)``: ``stack/<name>.npy`` or its fov.ome.zarr field (item 0aj).

    THE reader of field stacks for every Mask stage (normalisation, the
    in-memory batches, image QC, illumination and PSF preparation, the
    merge). A store field is returned exactly as ``stack/<name>.npy`` held
    it: the acquired channels in sorted channel order, same dtype and values.
    A time-lapse frame is read by its own name, and a field stored with its
    raw z planes is reduced on read (item 0aj P3), so the stack equals the
    maximum projection Mask preprocessing used to store.

    :param path: ``<project>/stack/<name>.npy`` (the file need not exist).
    :param mmap_mode: passed to :func:`numpy.load` for an ``.npy`` file.
    :param z_projection: the z reduction (see :func:`project_z`); None
        returns ``(Y, X, channels, Z)`` for a field with z planes.
    """
    found = _stack_entry(path)
    if found is None:
        return np.load(os.fspath(path), mmap_mode=mmap_mode, allow_pickle=False)
    store, entry = found
    image = read_frame_channels(store / entry["key"], entry, z_projection=z_projection)
    if image.ndim == 4:
        return np.ascontiguousarray(np.moveaxis(image, (0, 1), (2, 3)))
    return np.ascontiguousarray(np.moveaxis(image, 0, -1))


def _stack_header(path: Any) -> Tuple[Tuple[int, ...], np.dtype]:
    """``(shape, dtype)`` of one field stack without reading its pixels."""
    found = _stack_entry(path)
    if found is None:
        header = np.load(os.fspath(path), mmap_mode="r", allow_pickle=False)
        return tuple(header.shape), header.dtype
    _store, entry = found
    return tuple(int(v) for v in entry["stack_shape"]), np.dtype(entry["stack_dtype"])


def _stack_stat(path: Any) -> os.stat_result:
    """``os.stat`` of one field stack: the ``.npy``, or its image shard in fov.ome.zarr."""
    found = _stack_entry(path)
    if found is None:
        return os.stat(os.fspath(path))
    store, entry = found
    group = store / entry["key"] / "0"
    stats = [os.stat(os.path.join(base, name))
             for base, _dirs, names in os.walk(group) for name in names]
    newest = max(stats, key=lambda item: item.st_mtime_ns)
    values = list(newest)
    values[6] = sum(item.st_size for item in stats)
    return os.stat_result(values, {"st_mtime_ns": newest.st_mtime_ns})


def _stack_writer(project: Any, plate: str) -> PlateWriter:
    """The single writer of ``<project>/fov.ome.zarr`` while Mask preprocessing fills it."""
    return PlateWriter(project, plate, store_path=_merged_store_path(project), shards=True)


def _write_stack_field(writer: PlateWriter, name: str, stack: np.ndarray,
                       channels: Sequence[Any]) -> str:
    """Write one field stack ``(Y, X, C)`` into fov.ome.zarr and commit it (item 0aj P1c).

    The channels are stored as one ``(C, Y, X)`` image exactly as acquired
    (no cast), each plane's sha256 recorded, with the channel ids, the
    stack's dtype and shape and ``image_planes='stack'``. The field becomes
    a merged field once :func:`_write_merged_field` adds its labels.

    :param writer: from :func:`_stack_writer`.
    :param name: ``<plate>_<well>_<field>_<time>.npy``.
    :param stack: the field's channels ``(Y, X, C)``.
    :param channels: the channel id of each plane.
    :returns: the field's store key.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"{name}: a field stack is (Y, X, channels), not {stack.shape}")
    image = np.ascontiguousarray(np.moveaxis(stack, -1, 0))
    _plate, well, field = split_field_name(name)
    return writer.write_field(well, field, image, None, record={
        "name": os.path.basename(str(name)), "image_planes": STACK_PLANES,
        "stack_dtype": stack.dtype.str, "stack_shape": [int(v) for v in stack.shape],
        "image_dtype": image.dtype.str, "plane_sha256": _plane_hashes(image),
        "source_channels": [str(c) for c in channels]})


# ---------------------------------------------------------------------------
# The object masks (item 0aj P2): no masks/<class>_mask_stack/*.npy, the
# masks are labels of their field in fov.ome.zarr
# ---------------------------------------------------------------------------
#
# A project folder whose field stacks live in fov.ome.zarr (item 0aj P1c)
# keeps its object masks there too: every Mask stage that used to write
# ``masks/<class>_mask_stack/<field>.npy`` writes ``labels/<class>`` of that
# field through :func:`_write_mask` (PlateWriter.write_labels), and reads it
# back through :func:`_read_mask`. The mask-folder paths stay the names the
# stages pass around; the folders themselves are never created. An older
# project, or a run whose stacks are ``stack/*.npy`` (mask_image_format
# 'npy' or 'tiff', time-lapse), keeps its ``.npy`` files, read by the same
# functions.
#
# Commit rule (the 430 rule, per field and class): a label image is written
# to a temporary group and renamed into place, so ``labels/<class>`` either
# is absent or holds the complete mask. A field+class is done only once that
# rename has happened; the merge then records the field's final labels in
# the store ledger.

#: The suffix of the folder Mask Generation names after each object class.
MASK_STACK_SUFFIX = "_mask_stack"

#: The file types a mask folder holds in an older project.
_MASK_FILE_SUFFIXES = (".npy", ".tif", ".tiff")


def _mask_label_name(folder: Any) -> Optional[str]:
    """The label class a project's mask folder stands for, or None (item 0aj P2).

    ``<project>/masks/cell_mask_stack`` -> ``'cell'``,
    ``.../masks/adjusted_cell_mask_stack`` -> ``'adjusted_cell'``,
    ``.../masks/nucleus_territory_seed_stack`` -> ``'nucleus_territory_seed'``.
    A folder that is not directly inside a ``masks`` folder, or whose name
    does not end in ``_stack``, is not a mask folder.

    :param folder: the mask folder path (it need not exist).
    """
    target = Path(os.path.normpath(os.fspath(folder)))
    if target.parent.name != "masks":
        return None
    name = target.name
    if name.endswith(MASK_STACK_SUFFIX):
        name = name[:-len(MASK_STACK_SUFFIX)]
    elif name.endswith("_stack") and name != "_stack":
        name = name[:-len("_stack")]
    else:
        return None
    return name or None


def _mask_project_store(folder: Any) -> Path:
    """The fov.ome.zarr of the project that holds ``<project>/masks/<folder>``."""
    target = Path(os.path.normpath(os.fspath(folder)))
    return _merged_store_path(target.parent.parent)


def _mask_folder_files(folder: Any) -> List[str]:
    """The visible mask files (npy/tif) of a real mask folder, sorted."""
    try:
        names = os.listdir(os.fspath(folder))
    except OSError:
        return []
    return sorted(n for n in names if not n.startswith(".")
                  and n.lower().endswith(_MASK_FILE_SUFFIXES))


def _mask_store_entries(folder: Any) -> Dict[str, Dict[str, Any]]:
    """``{field file name: entry}`` of the fields whose masks ``folder`` keeps in fov.ome.zarr.

    Empty unless ``folder`` is a project mask folder whose project stores its
    fields' channels in fov.ome.zarr and the folder itself holds no mask
    files (an older project's files always win).
    """
    if _mask_label_name(folder) is None or _mask_folder_files(folder):
        return {}
    store = _mask_project_store(folder)
    if not (store / _LEDGER).is_file():
        return {}
    return _image_entries(store)


def _masks_in_store(folder: Any) -> bool:
    """Whether the masks of ``folder`` are labels in fov.ome.zarr (item 0aj P2)."""
    return bool(_mask_store_entries(folder))


def _mask_label_dir(store: Path, entry: Mapping[str, Any], label: str) -> Path:
    """``<store>/<row>/<col>/<field>/labels/<label>``."""
    return Path(store) / entry["key"] / "labels" / str(label)


#: The ledger value of ``masks`` for a merged field whose labels no longer
#: count as working masks: the run ended without keeping intermediates, where
#: an older project deleted ``masks/`` (item 0aj P2).
MASKS_CONSUMED = "consumed"


def _label_committed(store: Path, entry: Mapping[str, Any], label: str) -> bool:
    """Whether the field's ``labels/<label>`` is a committed working mask.

    Committed: renamed into place whole. A field whose masks were consumed
    at the end of a run (:func:`_consume_working_masks`) has none.
    """
    if entry.get("masks") == MASKS_CONSUMED:
        return False
    return (_mask_label_dir(store, entry, label) / "zarr.json").is_file()


def _mask_location(path: Any) -> Optional[Tuple[Path, Dict[str, Any], str]]:
    """``(store, entry, label)`` when ``masks/<class>_mask_stack/<name>`` lives in fov.ome.zarr.

    None for a real file and for a path outside a store-backed mask folder.
    The name is matched by its stem, so ``<name>.tif`` finds the field too.
    """
    target = Path(os.fspath(path))
    if target.is_file():
        return None
    folder = target.parent
    entries = _mask_store_entries(folder)
    if not entries:
        return None
    name = f"{os.path.splitext(target.name)[0]}.npy"
    entry = entries.get(name)
    if entry is None or (folder.parent.parent / "stack" / name).is_file():
        # A field whose stack is a stack/*.npy file keeps .npy masks.
        return None
    return _mask_project_store(folder), entry, str(_mask_label_name(folder))


def _mask_names(folder: Any) -> List[str]:
    """THE list of a mask folder's masks, ``<plate>_<well>_<field>_<time>.npy``, sorted (item 0aj P2).

    The files of a real folder (an older project); otherwise the fields of
    fov.ome.zarr whose ``labels/<class>`` is committed.

    :param folder: ``<project>/masks/<class>_mask_stack`` (it need not exist).
    """
    files = _mask_folder_files(folder)
    if files:
        return files
    entries = _mask_store_entries(folder)
    if not entries:
        return []
    store, label = _mask_project_store(folder), str(_mask_label_name(folder))
    return sorted(name for name, entry in entries.items()
                  if _label_committed(store, entry, label))


def _mask_folder_exists(folder: Any) -> bool:
    """Whether a project has masks of this folder's class: the folder, or committed labels."""
    return os.path.isdir(os.fspath(folder)) or bool(_mask_names(folder))


def _mask_exists(path: Any) -> bool:
    """Whether one field's mask is there: the file, or its committed label (item 0aj P2)."""
    if os.path.isfile(os.fspath(path)):
        return True
    found = _mask_location(path)
    return found is not None and _label_committed(*found)


def _read_mask(path: Any, *, mmap_mode: Optional[str] = None) -> np.ndarray:
    """THE reader of one field's object mask (item 0aj P2).

    ``masks/<class>_mask_stack/<name>.npy`` (or ``.tif``) when that file
    exists, as before; otherwise the field's ``labels/<class>`` in the
    project's fov.ome.zarr, value for value (uint16, or uint32 above 65535).

    :param path: the mask path a Mask stage names (the file need not exist).
    :param mmap_mode: passed to :func:`numpy.load` for an ``.npy`` file.
    :raises FileNotFoundError: neither the file nor a committed label exists.
    """
    target = os.fspath(path)
    if os.path.isfile(target):
        if target.lower().endswith((".tif", ".tiff")):
            import tifffile

            return tifffile.imread(target)
        return np.load(target, mmap_mode=mmap_mode, allow_pickle=False)
    found = _mask_location(path)
    if found is None or not _label_committed(*found):
        raise FileNotFoundError(target)
    store, entry, label = found
    return read_labels(store / entry["key"], label)


def _mask_header(path: Any) -> Tuple[Tuple[int, ...], np.dtype]:
    """``(shape, dtype)`` of one field's mask without reading its pixels (item 0aj P2)."""
    found = _mask_location(path)
    if found is None:
        target = os.fspath(path)
        if target.lower().endswith((".tif", ".tiff")):
            image = _read_mask(target)
            return tuple(image.shape), image.dtype
        header = np.load(target, mmap_mode="r", allow_pickle=False)
        return tuple(header.shape), header.dtype
    meta = json.loads((_mask_label_dir(*found) / "0" / "zarr.json")
                      .read_text(encoding="utf-8"))
    return tuple(int(v) for v in meta["shape"]), np.dtype(meta["data_type"])


def _mask_valid(path: Any) -> Tuple[bool, str]:
    """``(ok, reason)`` for one field's mask, as :func:`spacr.resume.validate_merged_field` (item 0aj P2).

    A mask file is checked by its header and length; a label in fov.ome.zarr
    is complete once committed, so it is ``(True, 'done')`` or
    ``(False, 'missing')``.
    """
    from .resume import REASON_DONE, REASON_MISSING, validate_merged_field

    found = _mask_location(path)
    if found is None:
        return validate_merged_field(os.fspath(path))
    return (True, REASON_DONE) if _label_committed(*found) else (False, REASON_MISSING)


def _mask_stat(path: Any) -> os.stat_result:
    """``os.stat`` of one mask: the file, or its committed label's files (size summed, newest time)."""
    found = _mask_location(path)
    if found is None:
        return os.stat(os.fspath(path))
    group = _mask_label_dir(*found)
    stats = [os.stat(os.path.join(base, name))
             for base, _dirs, names in os.walk(group) for name in names]
    if not stats:
        raise FileNotFoundError(os.fspath(path))
    newest = max(stats, key=lambda item: item.st_mtime_ns)
    values = list(newest)
    values[6] = sum(item.st_size for item in stats)
    return os.stat_result(values, {"st_mtime_ns": newest.st_mtime_ns})


#: ``{(store path, process id): PlateWriter}``: one label writer per store and process.
_LABEL_WRITERS: Dict[Tuple[str, int], PlateWriter] = {}


def _label_writer(store: Path, entry: Mapping[str, Any]) -> PlateWriter:
    """The project's fov.ome.zarr writer for masks, kept open per process."""
    cache_key = (str(store), os.getpid())
    writer = _LABEL_WRITERS.get(cache_key)
    if writer is None or not writer.path.is_dir():
        plate = split_field_name(entry["name"])[0]
        writer = PlateWriter(store.parent, plate, store_path=store, shards=True)
        _LABEL_WRITERS[cache_key] = writer
    if entry["key"] not in writer._committed:
        writer._committed[entry["key"]] = dict(entry)
    return writer


def _write_mask(path: Any, mask: np.ndarray) -> str:
    """THE writer of one field's object mask for every Mask stage (item 0aj P2).

    When the project keeps its fields in fov.ome.zarr, the mask becomes the
    field's ``labels/<class>`` through :meth:`PlateWriter.write_labels`
    (other classes kept, the channels never touched), as uint16, or uint32
    above 65535, exactly. Otherwise (an older project, or a run whose stacks
    are ``.npy``) it is written to ``path`` as ``.npy`` atomically, its
    folder created when missing, as before.

    :param path: ``<project>/masks/<class>_mask_stack/<name>.npy``.
    :param mask: the field's label image ``(Y, X)``.
    :returns: ``path``.
    """
    found = _mask_location(path)
    if found is None:
        from .io import _save_array_atomic

        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        return _save_array_atomic(os.fspath(path), mask)
    store, entry, label = found
    _plate, well, field = split_field_name(entry["name"])
    writer = _label_writer(store, entry)
    if "shape" in entry or entry.get("masks") == MASKS_CONSUMED:
        _reopen_merged_field(writer, store, entry)
    writer.write_labels(well, field, {label: _compact_labels(mask)}, keep_others=True)
    return os.fspath(path)


def _reopen_merged_field(writer: PlateWriter, store: Path, entry: Mapping[str, Any]) -> None:
    """A new mask for a merged field: the field is not merged any more (item 0aj P2).

    The field leaves the merged fields (Measure, resume) until the merge
    commits it again with the new mask, so a merged field never mixes masks
    of two runs. When its masks were consumed, its other labels go too, as
    an older project's deleted ``masks/`` held none.
    """
    import fcntl

    key = entry["key"]
    labels_dir = Path(store) / key / "labels"
    if entry.get("masks") == MASKS_CONSUMED and labels_dir.is_dir():
        for group in labels_dir.iterdir():
            if group.is_dir():
                shutil.rmtree(group)
    kept = {k: v for k, v in entry.items()
            if k not in ("key", "well", "field", "labels", "image", "shape",
                         "dtype", "masks")}
    present = sorted(g.name for g in labels_dir.iterdir()
                     if (g / "zarr.json").is_file()) if labels_dir.is_dir() else []
    _plate, well, field = split_field_name(entry["name"])
    with open(Path(store) / f".{_LEDGER}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            writer._commit(key, well, field, present, True, kept)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _consume_working_masks(project: Any) -> int:
    """End of a run that keeps no intermediates: the merged labels stop being working masks (item 0aj P2).

    An older project deleted ``masks/`` here. The labels of fov.ome.zarr
    are the run's output and stay; each merged field is marked
    ``masks='consumed'`` in the ledger, so a later run segments it again
    (its first new mask reopens the field) instead of reusing the merged
    labels as raw masks.

    :param project: the project folder.
    :returns: the fields marked.
    """
    import fcntl

    store = _merged_store_path(project)
    ledger = store / _LEDGER
    if not ledger.is_file():
        return 0
    entries = [dict(entry, masks=MASKS_CONSUMED) for entry in _store_entries(store).values()
               if entry.get("masks") != MASKS_CONSUMED]
    if not entries:
        return 0
    with open(store / f".{_LEDGER}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(ledger, "a", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    for writer in _LABEL_WRITERS.values():
        if writer.path == store:
            for entry in entries:
                writer._committed[entry["key"]] = entry
    return len(entries)


def _project_keeps_masks_in_store(project: Any) -> bool:
    """Whether a project's fields, and so its masks, live in fov.ome.zarr (item 0aj P2/P4).

    True once the store has committed field channels and the project has no
    ``stack/*.npy`` (a run whose stacks are files keeps file masks).
    """
    store = _merged_store_path(project)
    return bool((store / _LEDGER).is_file() and _image_entries(store)
                and not _stack_folder_names(Path(project) / "stack"))


#: The one folder of small Mask run records (the batch plan with its
#: normalisation percentiles and calibration, the adjusted-cell ledger) of a
#: project whose fields live in fov.ome.zarr (item 0aj P4). It sits inside the
#: store beside the commit ledger and is removed with the run's working
#: records at the end of a run that keeps no intermediates.
MASK_RUN_RECORDS = ".spacr_mask_run"


def _mask_run_dir(masks_dir: Any) -> str:
    """THE folder of a Mask run's records for a ``masks`` folder (item 0aj P4).

    ``<project>/fov.ome.zarr/.spacr_mask_run`` once a run keeps its records
    there, and for a project that keeps its fields in fov.ome.zarr, so no
    ``masks/`` folder is ever created; ``<project>/masks`` itself for an
    older project (a real ``masks/`` folder and no record folder, so its
    ``*.npz`` archives and plan still read) and for runs whose stacks are
    files. A record folder that exists always wins, so the choice does not
    change in the middle of a run when something else creates ``masks/``.
    Nothing is created here.

    :param masks_dir: ``<project>/masks`` (it need not exist).
    :returns: the folder path, as a string.
    """
    masks_dir = os.fspath(masks_dir)
    target = Path(os.path.normpath(masks_dir))
    if target.name != "masks":
        return masks_dir
    records = _merged_store_path(target.parent) / MASK_RUN_RECORDS
    if records.is_dir():
        return os.fspath(records)
    if os.path.isdir(masks_dir) or not _project_keeps_masks_in_store(target.parent):
        return masks_dir
    return os.fspath(records)


def _remove_mask_run_records(project: Any) -> bool:
    """Remove a project's ``fov.ome.zarr/.spacr_mask_run`` (item 0aj P4).

    :returns: whether it existed.
    """
    folder = _merged_store_path(project) / MASK_RUN_RECORDS
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True


def _prepare_mask_folder(folder: Any) -> None:
    """Create a mask folder only when its masks are files (item 0aj P2).

    A project whose fields live in fov.ome.zarr keeps its masks as labels
    there, so ``masks/<class>_mask_stack`` is never created.
    """
    if _mask_label_name(folder) is not None:
        project = Path(os.path.normpath(os.fspath(folder))).parent.parent
        if _project_keeps_masks_in_store(project):
            return
    os.makedirs(os.fspath(folder), exist_ok=True)


def _remove_mask_folder(folder: Any) -> bool:
    """Remove a class's masks: the folder, and its labels in fov.ome.zarr (item 0aj P2).

    It runs where a stage redraws that class (a fresh run, superseded
    inputs), so the label goes from every field, merged or not.

    :returns: whether anything was removed.
    """
    removed = False
    target = os.fspath(folder)
    if os.path.isdir(target):
        shutil.rmtree(target)
        removed = True
    label = _mask_label_name(folder)
    if label is None:
        return removed
    store = _mask_project_store(folder)
    if not (store / _LEDGER).is_file():
        return removed
    for entry in _image_entries(store).values():
        group = _mask_label_dir(store, entry, label)
        if group.is_dir():
            shutil.rmtree(group)
            removed = True
    return removed


def _mask_folders(masks_dir: Any) -> Dict[str, str]:
    """``{folder name: path}`` of every ``*_mask_stack`` of a ``masks`` folder (item 0aj P2).

    Real folders, plus one (never created) folder per label class committed
    on a field of the project's fov.ome.zarr.
    """
    masks_dir = os.fspath(masks_dir)
    found: Dict[str, str] = {}
    try:
        for name in sorted(os.listdir(masks_dir)):
            path = os.path.join(masks_dir, name)
            if name.endswith(MASK_STACK_SUFFIX) and os.path.isdir(path):
                found[name] = path
    except OSError:
        pass
    store = _merged_store_path(Path(masks_dir).parent)
    if Path(masks_dir).name == "masks" and (store / _LEDGER).is_file():
        for entry in _image_entries(store).values():
            labels = Path(store) / entry["key"] / "labels"
            if entry.get("masks") == MASKS_CONSUMED or not labels.is_dir():
                continue
            for group in labels.iterdir():
                if (group / "zarr.json").is_file() and not group.name.startswith("."):
                    folder = f"{group.name}{MASK_STACK_SUFFIX}"
                    found.setdefault(folder, os.path.join(masks_dir, folder))
    return dict(sorted(found.items()))


def _mask_digest(path: Any) -> Optional[str]:
    """The SHA-256 of one mask (item 0aj P2), or None when it cannot be read.

    A mask file: the sha256 of its bytes, as before. A label in fov.ome.zarr:
    the sha256 of its dtype, shape and C-order pixel bytes, so an unchanged
    label always gives the same digest and a rewritten one another.
    """
    import hashlib

    digest = hashlib.sha256()
    target = os.fspath(path)
    if os.path.isfile(target):
        try:
            with open(target, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            return None
        return digest.hexdigest()
    try:
        mask = _read_mask(path)
    except (OSError, ValueError):
        return None
    digest.update(f"{mask.dtype.str}{tuple(mask.shape)}".encode("ascii"))
    digest.update(np.ascontiguousarray(mask).tobytes())
    return digest.hexdigest()


def _mask_sidecar_folder(folder: Any) -> str:
    """Where a record kept beside a mask folder's masks goes (item 0aj P2).

    The folder itself for ``.npy`` masks; the run's record folder
    (:func:`_mask_run_dir`) when the masks are labels in fov.ome.zarr (the
    mask folder, and since P4 the ``masks`` folder, is never created).
    """
    folder = os.fspath(folder)
    if os.path.isdir(folder) or not _masks_in_store(folder):
        return folder
    return _mask_run_dir(os.path.dirname(os.path.normpath(folder)))


# ---------------------------------------------------------------------------
# Time and depth (item 0aj P3): TCZYX images, TZYX labels, z reduced on read
# ---------------------------------------------------------------------------
#
# A field with a time or depth axis is ONE NGFF 0.6 image with those axes:
# the image ``(T, C, Z, Y, X)`` with only the axes it has, its labels
# ``(T, Z, Y, X)`` likewise. The raw z planes are stored; any z reduction a
# module needs (the maximum projection Mask segments and Measure measures)
# is computed when the field is read, here, with a ``z_projection``
# parameter. Every frame of a time-lapse field keeps its own field file name
# (``<plate>_<well>_<field>_<time>.npy``) as a view of the one stored field,
# so each reader asks for one frame by name, as it asked for one ``.npy``.
# A 2-D, single-time field records no axes and is stored and read exactly
# as before.

def field_axes(ndim: int, axes: Optional[str] = None, *, kind: str = "image",
               where: str = "field") -> str:
    """The checked axes of one stored array, as a string such as ``'tczyx'`` (item 0aj P3).

    :param ndim: the array's rank.
    :param axes: its axes; by default ``'cyx'`` for a 3-D image and ``'yx'``
        for a 2-D image or label image, the 2-D project layout.
    :param kind: ``'image'`` (``t, c, z, y, x``) or ``'labels'`` (``t, z, y, x``).
    :param where: the array, named in a refusal.
    :returns: the axes, lower case, in NGFF order.
    :raises ValueError: the axes are missing for a rank beyond the default,
        are not in NGFF order, do not end in ``y, x`` or do not match the rank.
    """
    order = IMAGE_AXIS_ORDER if kind == "image" else LABEL_AXIS_ORDER
    if axes is None:
        if int(ndim) == 2:
            return "yx"
        if kind == "image" and int(ndim) == 3:
            return "cyx"
        raise ValueError(
            f"{where} has {ndim} axes; name them (axes such as "
            f"{order[-int(ndim):]!r}), because time and depth are stored as "
            f"their own axes and never flattened into channels"
            if int(ndim) <= len(order) else
            f"{where} has {ndim} axes; only (Y, X) up to {order!r} is written")
    axes = str(axes).lower()
    positions = [order.find(a) for a in axes]
    if (len(axes) != int(ndim) or -1 in positions or positions != sorted(set(positions))
            or not axes.endswith("yx")):
        raise ValueError(f"{where}: axes {axes!r} do not fit a {ndim}-D array in "
                         f"NGFF order {order!r} ending in y, x")
    return axes


def project_z(array: np.ndarray, axis: int, projection: Any = DEFAULT_Z_PROJECTION
              ) -> np.ndarray:
    """Reduce one z axis on read (item 0aj P3).

    :param array: the planes.
    :param axis: the z axis of ``array``.
    :param projection: ``'max'`` (the stored dtype; the maximum projection
        Mask preprocessing used to store), ``'min'`` (the stored dtype),
        ``'mean'`` (float32), ``'sum'`` (float64 for float input, else
        int64/uint64), an int (that one plane), or None (keep the axis).
    :returns: the array without the z axis (or unchanged for None).
    :raises ValueError: an unknown projection.
    """
    array = np.asarray(array)
    if projection is None:
        return array
    if isinstance(projection, (int, np.integer)) and not isinstance(projection, bool):
        return np.take(array, int(projection), axis=axis)
    name = str(projection).lower()
    if name == "max":
        return array.max(axis=axis)
    if name == "min":
        return array.min(axis=axis)
    if name == "mean":
        return array.mean(axis=axis, dtype=np.float64).astype(np.float32)
    if name == "sum":
        return array.sum(axis=axis)
    raise ValueError(f"z_projection must be one of {list(Z_PROJECTIONS)}, a plane "
                     f"index or None; got {projection!r}")


def select_planes(array: Any, axes: str, *, t: Optional[int] = None,
                  z_projection: Any = DEFAULT_Z_PROJECTION,
                  window: Optional[Tuple[int, int, int, int]] = None
                  ) -> Tuple[np.ndarray, str]:
    """One frame of a stored array with z reduced, read lazily (item 0aj P3).

    THE selection every reader of a t/z field shares: the frame and the box
    are taken from the (zarr or numpy) array before anything is decoded,
    then z is reduced with :func:`project_z`.

    :param array: a zarr array (or ndarray) with ``axes``.
    :param axes: its axes, such as ``'tczyx'``.
    :param t: the frame; None keeps the t axis (or takes the only frame of
        a one-frame field).
    :param z_projection: see :func:`project_z`; None keeps z.
    :param window: ``(y0, y1, x0, x1)``, or None for the whole plane.
    :returns: ``(array, its remaining axes)``.
    """
    index: List[Any] = []
    left = ""
    shape = tuple(array.shape)
    for position, axis in enumerate(axes):
        if axis == "t" and (t is not None or shape[position] == 1):
            index.append(int(t or 0))
            continue
        if axis == "z" and isinstance(z_projection, (int, np.integer)) \
                and not isinstance(z_projection, bool):
            index.append(int(z_projection))
            continue
        if axis == "y" and window is not None:
            index.append(slice(int(window[0]), int(window[1])))
        elif axis == "x" and window is not None:
            index.append(slice(int(window[2]), int(window[3])))
        else:
            index.append(slice(None))
        left += axis
    data = np.asarray(array[tuple(index)])
    if "z" in left and z_projection is not None:
        data = project_z(data, left.index("z"), z_projection)
        left = left.replace("z", "")
    return data, left


def _as_planes(image: np.ndarray) -> np.ndarray:
    """A stored image as its 2-D planes ``(n, Y, X)``, in C order of its leading axes."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image[None]
    return image.reshape((-1,) + image.shape[-2:])


def _entry_axes(entry: Mapping[str, Any]) -> str:
    """The stored image axes of a ledger entry; ``'cyx'`` for a 2-D field."""
    return str(entry.get("axes") or "cyx")


def _entry_label_axes(entry: Mapping[str, Any]) -> str:
    """The stored label axes of a ledger entry; ``'yx'`` for a 2-D field."""
    return str(entry.get("label_axes") or "yx")


def _is_layered(entry: Mapping[str, Any]) -> bool:
    """Whether a ledger entry is a field with a t or z axis (item 0aj P3)."""
    return bool(entry.get("axes") or entry.get("label_axes"))


def _open_field_array(path: Any):
    """One stored array of a field, opened (not read) through zarr."""
    from .ome_zarr import require_zarr

    return require_zarr().open_array(os.fspath(path), mode="r")


def _entry_frame(entry: Mapping[str, Any]) -> Optional[int]:
    """The frame a per-frame ledger view names, or None."""
    value = entry.get("t")
    return None if value is None else int(value)


def read_frame_channels(group: Any, entry: Mapping[str, Any], *,
                        z_projection: Any = DEFAULT_Z_PROJECTION,
                        window: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """THE reader of one stored field frame's channels (item 0aj P3).

    :param group: the field group in the store.
    :param entry: its ledger entry (a frame view for a time-lapse field).
    :param z_projection: the z reduction (see :func:`project_z`); None
        keeps the z planes ``(C, Z, Y, X)``.
    :param window: ``(y0, y1, x0, x1)``.
    :returns: ``(C, Y, X)`` (``(C, Z, Y, X)`` without projection); a 2-D
        field is read exactly as :func:`read_image` reads it.
    """
    group = Path(os.fspath(group))
    if not _is_layered(entry):
        if window is None:
            image = read_image(group)
        else:
            image = np.asarray(_open_field_array(group / "0")[..., window[0]:window[1],
                                                             window[2]:window[3]])
        return image if image.ndim == 3 else image[None]
    axes = _entry_axes(entry)
    data, left = select_planes(_open_field_array(group / "0"), axes,
                               t=_entry_frame(entry), z_projection=z_projection,
                               window=window)
    if "c" not in left:
        data, left = data[None], "c" + left
    return data


def read_frame_labels(group: Any, entry: Mapping[str, Any], name: str, *,
                      z_projection: Any = None,
                      window: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """THE reader of one stored field frame's label image (item 0aj P3).

    :param group: the field group in the store.
    :param entry: its ledger entry (a frame view for a time-lapse field).
    :param name: the object class.
    :param z_projection: for labels with a z axis: a plane index, or
        ``'max'``; None keeps z (labels are never averaged).
    :param window: ``(y0, y1, x0, x1)``.
    :returns: ``(Y, X)`` (``(Z, Y, X)`` for volumetric labels kept whole).
    """
    group = Path(os.fspath(group))
    path = group / "labels" / str(name)
    if not _is_layered(entry):
        if window is None:
            return read_labels(group, name)
        return np.asarray(_open_field_array(path / "0")[..., window[0]:window[1],
                                                        window[2]:window[3]])
    data, _left = select_planes(_open_field_array(path / "0"), _entry_label_axes(entry),
                                t=_entry_frame(entry), z_projection=z_projection,
                                window=window)
    return data


def _frame_views(entry: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{frame file name: ledger view}`` of one time-lapse field entry (item 0aj P3).

    Each frame is listed under its own field file name, with ``t`` its
    index and ``plane_sha256`` the checksums of its own planes, so the
    readers, ``verify_store`` and the originals check treat it as one field.
    """
    frames = list(entry.get("frames") or ())
    views: Dict[str, Dict[str, Any]] = {}
    hashes = list(entry.get("plane_sha256") or ())
    per = len(hashes) // len(frames) if frames and hashes else 0
    merged = set(int(t) for t in entry.get("merged_frames") or range(len(frames))) \
        if "shape" in entry else set()
    for index, name in enumerate(frames):
        view = {k: v for k, v in entry.items() if k not in ("frames", "merged_frames")}
        view["name"] = str(name)
        view["t"] = index
        if per:
            view["plane_sha256"] = hashes[index * per:(index + 1) * per]
        if "shape" in entry and index not in merged:
            view.pop("shape", None)
        views[str(name)] = view
    return views


def _frame_planes_count(entry: Mapping[str, Any]) -> int:
    """How many stored 2-D planes one frame of an entry has (C x Z)."""
    shape = entry.get("image_shape")
    axes = _entry_axes(entry)
    if not shape:
        return 0
    n = 1
    for axis, size in zip(axes[:-2], shape[:-2]):
        if axis != "t":
            n *= int(size)
    return n


def _write_stack_series(writer: PlateWriter, names: Sequence[str], frames: Sequence[np.ndarray],
                        channels: Sequence[Any], *, z_planes: bool) -> str:
    """Write one field's frames (and raw z planes) into fov.ome.zarr and commit it (item 0aj P3).

    The one writer of a field with a t or z axis, as :func:`_write_stack_field`
    is for a 2-D field: the channels are stored exactly as acquired, every
    plane's sha256 recorded, and the 2-D view each Mask stage reads (the
    maximum projection over z, ``stack_shape``) is computed on read.

    :param writer: from :func:`_stack_writer`.
    :param names: the field file name of each frame, in time order.
    :param frames: each frame ``(C, Z, Y, X)`` when ``z_planes`` else ``(C, Y, X)``.
    :param channels: the channel id of each plane.
    :param z_planes: whether the frames carry a z axis.
    :returns: the field's store key.
    """
    if not names or len(names) != len(frames):
        raise ValueError("a field series needs one name per frame")
    series = len(names) > 1
    image = np.stack([np.asarray(f) for f in frames]) if series else np.asarray(frames[0])
    axes = ("t" if series else "") + ("czyx" if z_planes else "cyx")
    image = np.ascontiguousarray(image)
    view, _left = select_planes(image, axes, t=0 if series else None)
    stack_shape = [int(view.shape[-2]), int(view.shape[-1]), int(view.shape[0])]
    plates_wells = {split_field_name(n)[1:] for n in names}
    if len(plates_wells) != 1:
        raise ValueError(f"the frames {list(names)} are not one field")
    well, field = next(iter(plates_wells))
    record: Dict[str, Any] = {
        "image_planes": STACK_PLANES, "stack_dtype": image.dtype.str,
        "stack_shape": stack_shape, "image_dtype": image.dtype.str,
        "image_shape": [int(v) for v in image.shape],
        "plane_sha256": _plane_hashes(image),
        "source_channels": [str(c) for c in channels]}
    if series:
        record["frames"] = [os.path.basename(str(n)) for n in names]
    else:
        record["name"] = os.path.basename(str(names[0]))
    return writer.write_field(well, field, image, None, record=record, axes=axes)


def _write_merged_frame(writer: PlateWriter, name: str, stack: np.ndarray,
                        layout: Mapping[str, Any], entry: Mapping[str, Any]) -> Optional[str]:
    """Add one merged frame's masks to a time-lapse field (item 0aj P3).

    The masks of each frame are held until the field's last frame arrives;
    then every class is written as one ``(T, Y, X)`` label image and the
    field is committed (the 430 rule: a field counts as merged only once
    all its frames are on disk). The channels are never touched.

    :param writer: the merged store's writer.
    :param name: the frame's field file name.
    :param stack: the merged frame ``(Y, X, planes)``.
    :param layout: the store's plane layout.
    :param entry: the field's committed ledger entry (with ``frames``).
    :returns: the field key once committed, else None.
    """
    frames = [str(f) for f in entry["frames"]]
    index = frames.index(os.path.basename(str(name)))
    labels = {role: _compact_labels(stack[..., int(dim)])
              for role, dim in (layout.get("mask_dims") or {}).items()
              if int(dim) < stack.shape[-1]}
    # The frames wait on the writer itself, so a new run never sees an old run's frames.
    waiting = writer.__dict__.setdefault("_pending_frames", {})
    pending = waiting.setdefault(entry["key"], {})
    pending[index] = (labels, [int(v) for v in stack.shape], stack.dtype.str)
    if len(pending) < len(frames):
        return None
    del waiting[entry["key"]]
    roles = list(pending[0][0])
    series = {}
    for role in roles:
        planes = [pending[t][0][role] for t in range(len(frames))]
        wide = np.result_type(*[p.dtype for p in planes])
        series[role] = _compact_labels(np.stack([p.astype(wide, copy=False) for p in planes]))
    _plate, well, field = split_field_name(frames[0])
    label_axes = "t" + ("zyx" if "z" in _entry_label_axes(entry) else "yx")
    return writer.write_labels(well, field, series, label_axes=label_axes, record={
        "dtype": pending[0][2], "shape": pending[0][1],
        "merged_frames": list(range(len(frames)))})


def _layered_entry(writer: PlateWriter, name: str) -> Optional[Dict[str, Any]]:
    """The committed time-lapse field entry a frame name belongs to, or None."""
    try:
        _plate, well, field = split_field_name(name)
    except ValueError:
        return None
    entry = writer._committed.get(field_key(well, field))
    if entry and os.path.basename(str(name)) in (entry.get("frames") or ()):
        return entry
    return None


def _frame_depth(entry: Mapping[str, Any]) -> int:
    """How many z planes each channel of a stored field frame has (1 without a z axis)."""
    axes = _entry_axes(entry)
    shape = entry.get("image_shape")
    if "z" not in axes or not shape:
        return 1
    return int(shape[axes.index("z")])


def z_order(key: Sequence[Any]) -> Tuple[Any, ...]:
    """The sort key that puts one channel's raw files in z order (item 0aj P3).

    :param key: a raw-image key ``(plate, well, field, channel, time, slice)``
        from :func:`spacr.io._raw_images_by_key`.
    :returns: a key ordering by the slice number (numeric when it is one),
        the one order Mask stores z planes in and the originals check reads.
    """
    value = key[5] if len(key) > 5 else None
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return (1, str(value))
    return (0, number)


def merged_source(project: Any, image_format: str = "ome-zarr") -> str:
    """The merged fields of a project folder: ``fov.ome.zarr`` or ``merged/`` (item 0aj).

    THE one choice every reader of merged fields makes (Mask's overlays,
    the time-lapse readers, motility); list it with :func:`_list_fields` and
    read each field with :func:`_read_field`.

    :param project: the project (plate) folder.
    :param image_format: ``'ome-zarr'`` reads the store when it holds
        fields; any other format reads ``merged/`` when it exists. Either
        falls back to the one that is there.
    :returns: the store or folder path.
    """
    store = os.fspath(_merged_store_path(project))
    folder = os.path.join(os.fspath(project), "merged")
    if _is_merged_store(store) and (image_format == "ome-zarr" or not os.path.isdir(folder)):
        return store
    return folder


def merged_field_path(source: Any, name: str) -> Optional[str]:
    """The path :func:`_read_field` reads one merged field by name from, or None (item 0aj P3).

    :param source: ``merged/`` or the project's ``fov.ome.zarr``.
    :param name: the field file name, with or without ``.npy``.
    """
    base = os.path.splitext(os.path.basename(str(name)))[0]
    for candidate in (base + ".npy", str(name), str(name) + ".npy"):
        path = os.path.join(os.fspath(source), candidate)
        if os.path.isfile(path) or _store_field(path) is not None:
            return path
    return None


def _time_key(name: str) -> Tuple[Any, ...]:
    """Sort key of a field file name by its time point (numeric when it is one)."""
    from .schema import KEY_SEPARATOR

    parts = os.path.splitext(os.path.basename(str(name)))[0].split(KEY_SEPARATOR)
    token = parts[3] if len(parts) > 3 else ""
    try:
        return (0, int(token))
    except ValueError:
        return (1, token)


def _write_merged_series(writer: PlateWriter, names: Sequence[str],
                         stacks: Sequence[np.ndarray], layout: Mapping[str, Any]) -> str:
    """Write an older project's merged time-lapse frames as one field with a t axis (item 0aj P3).

    :param writer: the merged store's writer.
    :param names: the frames' field file names.
    :param stacks: each merged frame ``(Y, X, planes)``, same shape and dtype.
    :param layout: the store's plane layout.
    :returns: the field's store key.
    """
    order = sorted(range(len(names)), key=lambda i: _time_key(names[i]))
    names = [os.path.basename(str(names[i])) for i in order]
    stacks = [np.asarray(stacks[i]) for i in order]
    if len({(s.shape, s.dtype.str) for s in stacks}) != 1:
        raise ValueError(f"{names[0]}: the time points differ in shape or dtype")
    n_intensity = len(layout.get("intensity_channels") or ())
    image = np.ascontiguousarray(np.stack([np.moveaxis(s[..., :n_intensity], -1, 0)
                                           for s in stacks]))
    labels = {role: _compact_labels(np.stack([s[..., int(dim)] for s in stacks]))
              for role, dim in (layout.get("mask_dims") or {}).items()
              if int(dim) < stacks[0].shape[-1]}
    _plate, well, field = split_field_name(names[0])
    return writer.write_field(well, field, image, labels, axes="tcyx", label_axes="tyx", record={
        "frames": names, "dtype": stacks[0].dtype.str,
        "shape": [int(v) for v in stacks[0].shape], "image_dtype": image.dtype.str,
        "image_shape": [int(v) for v in image.shape], "plane_sha256": _plane_hashes(image),
        "merged_frames": list(range(len(names)))})
