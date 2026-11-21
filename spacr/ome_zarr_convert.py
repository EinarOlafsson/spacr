"""Resumable, opt-in conversion of legacy projects through the shared stores.

``convert_project`` keeps source files by default. The JSON ledger owns the
destination, fingerprints the complete source snapshot and records decoded
array digests independently of the image-store commit ledger. One coordinator
writes fields and publishes crops using field-sized buffers and one crop
shard. No settings, GUI or default format is changed.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import tempfile

import numpy as np

from . import image_store
from .qt.i18n import tr

_SCHEMA = 1
_LEDGER = "conversion.json"
_ASSETS = ("merged", "masks", "crops")
_IMAGE_EXTENSIONS = {".npy", ".tif", ".tiff"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def _path(value):
    """Resolve a local path, accepting separator-only Windows spellings.

    :param value: local project path; foreign drives and UNC roots require
        a native mount instead of an invented mapping.
    :returns: absolute Path without traversals or symlink aliases.
    """
    text = os.fspath(value)
    if os.name != "nt" and PureWindowsPath(text).drive:
        raise ValueError(tr("Mount the Windows project locally before conversion: {path}").format(path=text))
    path = Path(text.replace("\\", os.sep)).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(tr("Conversion paths must not contain traversal or symlinks: {path}").format(path=text))
    return path


def _portable_name(name):
    """Reject components that collide or cannot be represented on Windows."""
    if (not name or name in {".", ".."} or name.endswith((" ", "."))
            or name.split(".")[0].upper() in _RESERVED
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)):
        raise ValueError(tr("Rename this nonportable project component before conversion: {name}").format(name=name))


def _file_digest(path):
    """Fingerprint file bytes while rejecting a mutation during the read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(tr("Source changed while reading; close the project and retry: {path}").format(path=path))
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def _inventory(root, *, fingerprints=True):
    """Inventory a quiescent local tree without following links or opening tables."""
    files, names = {}, set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        _portable_name(path.name)
        if path.is_symlink() or relative.casefold() in names:
            raise ValueError(tr("Resolve symlinks or case-insensitive path collisions before conversion: {path}").format(path=path))
        names.add(relative.casefold())
        if path.is_file():
            disposable = not fingerprints and relative in {
                ".conversion-crops/measurements/measurements.db" + suffix
                for suffix in ("-wal", "-shm", "-journal")}
            if not disposable and (path.name.endswith("-journal")
                    or (path.name.endswith("-wal") and path.stat().st_size)):
                raise ValueError(tr("Close database writers and checkpoint the project before conversion: {path}").format(path=path))
            files[relative] = _file_digest(path) if fingerprints else None
        elif not path.is_dir():
            raise ValueError(tr("Conversion requires ordinary local files: {path}").format(path=path))
    return files


def _array_digest(array):
    """Return shape, dtype and an MD5 of exact C-order decoded bytes.

    NaN payloads and signed zero are retained. This detects accidental
    corruption; it is not an authentication mechanism.
    """
    array = np.asarray(array)
    header = {"shape": list(array.shape), "dtype": array.dtype.str}
    digest = hashlib.md5(json.dumps(header, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return {**header, "md5": digest.hexdigest()}


def _storage_array(array):
    """Use native byte order for Zarr while preserving scalar bits and source dtype metadata."""
    array = np.asarray(array)
    return array if array.dtype.isnative else array.astype(array.dtype.newbyteorder("="))


def _identity(name):
    """Parse the existing field naming contract and retain a canonical well key."""
    plate, well, field = image_store.split_field_name(name)
    _portable_name(plate)
    if field < 0:
        raise ValueError(tr("Field numbers must be nonnegative: {name}").format(name=name))
    key = image_store.field_key(well, field)
    return {"id": f"{plate}/{key}", "plate": plate, "well": well,
            "field": field, "key": key, "merged": None, "mask_files": {}}


def _mask_role(relative):
    """Recognize existing mask directories without guessing roles from pixels."""
    path = Path(relative)
    folder = path.parent.name
    if folder.endswith("_mask_stack"):
        return folder[:-11]
    if folder.endswith("_mask"):
        return folder[:-5]
    if len(path.parts) == 3 and path.parts[0] == "masks":
        return folder
    return None


def _load_field(src, entry):
    """Read native intensity and effective masks through shared image readers.

    :returns: CYX image or None, label mapping, original merged HWC or None.
    """
    from .crops import MergedField, measured_masks_state

    raw, image, labels = None, None, {}
    if entry["merged"]:
        path = src / entry["merged"]
        raw = image_store.read_image(path)
        layout = entry["layout"]
        count = len(layout["intensity_channels"])
        if raw.ndim != 3 or raw.shape[-1] != count + len(layout["mask_dims"]) or not count:
            raise ValueError(tr("Merged shape does not match the saved plane layout: {path}").format(path=path))
        field = MergedField(str(path), array=raw, mask_dims=layout["mask_dims"])
        if measured_masks_state(str(path)) == "stale":
            raise ValueError(tr("Measured-mask provenance is stale; resolve it before conversion: {path}").format(path=path))
        image = np.moveaxis(raw[..., :count], -1, 0)
        for role in layout["mask_dims"]:
            labels[role] = field.mask_plane(role)
    for role, relative in entry["mask_files"].items():
        mask = image_store.read_labels(src / relative, role)
        if mask.ndim != 2 or (image is not None and mask.shape != image.shape[1:]):
            raise ValueError(tr("Mask shape does not match its field: {path}").format(path=src / relative))
        if role in labels and not np.array_equal(mask, labels[role]):
            raise ValueError(tr("Separate and effective merged masks conflict; resolve their provenance: {path}").format(path=src / relative))
        labels[role] = mask
    if labels and len({tuple(mask.shape) for mask in labels.values()}) != 1:
        raise ValueError(tr("All label images in a field must have the same shape: {field}").format(field=entry["id"]))
    return (None if image is None else _storage_array(image),
            {role: _storage_array(mask) for role, mask in labels.items()}, raw)


def _stored_arrays(src, entry, what):
    """Select store arrays and retain original embedded planes needed for reversal."""
    image, labels, raw = _load_field(src, entry)
    if "merged" not in what:
        image = None
    if "masks" not in what:
        labels = {role: mask for role, mask in labels.items()
                  if raw is not None and "merged" in what and role in entry["layout"]["mask_dims"]}
    if raw is not None and "merged" in what:
        for role, dim in entry["layout"]["mask_dims"].items():
            original = _storage_array(raw[..., dim])
            if _array_digest(original) != _array_digest(labels[role]):
                if f"legacy_{role}" in labels:
                    raise ValueError(tr("A mask role conflicts with a reserved legacy-plane name: {role}").format(role=role))
                labels[f"legacy_{role}"] = original
    return image, labels, raw


def _recorded_settings(path):
    """Read settings through shared readers without creating SQLite sidecars."""
    from .resume import read_recorded_settings

    path = Path(path)
    if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        return read_recorded_settings(str(path))
    if not path.is_file():
        return {}
    try:
        rows = _crop_table(path, "settings")
    except ValueError as exc:
        if "Table not found in database: settings" not in str(exc):
            raise
        return {}
    return {str(row["setting_key"]): _crop_scalar(row["setting_value"])
            for row in rows.to_dict(orient="records")}


def _saved_image_metadata(src):
    """Read consistent saved calibration and channel names through the shared settings reader."""
    from .resume import _is_inert

    paths = [src / "measurements" / "measurements.db"]
    paths.extend(src / "settings" / name for name in (
        "gen_mask_settings.csv", "measure_crop_settings.csv", "measure_settings.csv"))
    collected = {"pixel_size_um": [], "channel_names": []}
    for path in paths:
        settings = _recorded_settings(path)
        for key, values in collected.items():
            value = settings.get(key)
            if _is_inert(value):
                continue
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            if value is not None:
                if value not in values:
                    values.append(value)
    if any(len(values) > 1 for values in collected.values()):
        raise ValueError(tr("Saved image calibration or channel names conflict between runs; resolve the metadata before conversion."))
    result = {key: values[0] if values else None for key, values in collected.items()}
    size = result["pixel_size_um"]
    if size is not None and (isinstance(size, bool) or not isinstance(size, (int, float))
                             or not np.isfinite(size) or size <= 0):
        raise ValueError(tr("Saved pixel_size_um must be a positive finite number; correct the source metadata."))
    names = result["channel_names"]
    if names is not None and (not isinstance(names, (tuple, list))
                              or any(not isinstance(name, str) for name in names)):
        raise ValueError(tr("Saved channel_names must be a list of channel labels; correct the source metadata."))
    return result


def _field_channel_names(entry, metadata):
    """Map original channel labels through the recorded merged channel selection."""
    names = metadata["channel_names"]
    channels = (entry.get("layout") or {}).get("intensity_channels", [])
    if not channels:
        return None
    if names is None:
        return [f"channel_{index}" for index in range(len(channels))]
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 or c >= len(names) for c in channels):
        raise ValueError(tr("Saved channel names do not cover the merged channel selection: {field}").format(field=entry["id"]))
    return [names[index] for index in channels]


def _versions():
    """Bind resume to the converter, store source and dependency versions."""
    from . import crop_store, crops, ome_zarr

    return {"schema": _SCHEMA, "modules": {
        **{Path(module.__file__).name: _file_digest(Path(module.__file__))["sha256"]
           for module in (image_store, crop_store, crops, ome_zarr)},
        "ome_zarr_convert.py": _file_digest(Path(__file__))["sha256"]},
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "zarr", "numcodecs")}}


def _plan(src, dst, what):
    """Build a deterministic immutable plan, collecting blockers before any writes."""
    from . import schema
    from .crops import CropError, read_merged_plane_layout
    from .ome_zarr import require_zarr

    files = _inventory(src)
    plan = {"source": str(src), "destination": str(dst), "what": list(what),
            "files": files, "fields": [], "crops": [], "problems": [],
            "notes": [tr("Physical pixel size is unknown; unitless coordinates are used.")],
            "source_bytes": sum(item["size"] for item in files.values())}
    if _LEDGER in files:
        plan["problems"].append(tr("This source already has a conversion ledger; use the original legacy project."))
    try:
        plan["image_metadata"] = _saved_image_metadata(src)
        if plan["image_metadata"]["pixel_size_um"] is not None:
            plan["notes"] = []
    except ValueError as exc:
        plan["problems"].append(str(exc))
        plan["image_metadata"] = {"pixel_size_um": None, "channel_names": None}
    try:
        zarr = require_zarr()
        if int(zarr.__version__.split(".")[0]) < 3:
            raise ValueError(tr("OME-Zarr conversion requires the Zarr 3 dependency."))
        plan["versions"] = _versions()
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        plan["problems"].append(str(exc))
    fields, field_times = {}, {}
    needs_merged = "merged" in what or "crops" in what
    for relative in files:
        path = Path(relative)
        is_merged = path.parent.as_posix() == "merged" and path.suffix.lower() in _IMAGE_EXTENSIONS
        role = _mask_role(relative) if "masks" in what or "crops" in what else None
        if not (is_merged or role and path.suffix.lower() in _IMAGE_EXTENSIONS):
            continue
        if is_merged and not (needs_merged or "masks" in what):
            continue
        try:
            item = _identity(path.name)
            try:
                frame = schema.parse_field_stem(path.name, timelapse=True)
            except schema.KeyParseError:
                frame = schema.parse_field_stem(path.name)
            time = schema.parse_int_token(frame.timeID)
            if time is not None:
                previous = field_times.setdefault(item["id"], (time, relative))
                if previous[0] != time:
                    raise ValueError(tr(
                        "One field contains different timepoints: {first} and {second}; "
                        "retain originals until time-aware conversion is supported.").format(
                            first=previous[1], second=relative))
            existing = fields.setdefault(item["id"], item)
            if existing["well"] != item["well"]:
                raise ValueError(tr("Field names collide after well normalization: {path}").format(path=relative))
            if is_merged:
                if path.suffix.lower() != ".npy" or existing["merged"]:
                    raise ValueError(tr("Merged conversion needs one NPY field and its plane layout: {path}").format(path=relative))
                existing["merged"] = relative
                existing["layout"] = read_merged_plane_layout(str(src / "merged"))
                if existing["layout"] is None:
                    raise ValueError(tr("Merged conversion needs .spacr_plane_layout.json; recover the saved channel layout first."))
            else:
                _portable_name(role)
                if role in existing["mask_files"]:
                    raise ValueError(tr("Multiple mask files have the same field and role: {path}").format(path=relative))
                existing["mask_files"][role] = relative
        except (ValueError, OSError, CropError) as exc:
            if str(exc) not in plan["problems"]:
                plan["problems"].append(str(exc))
    keys = set()
    for entry in sorted(fields.values(), key=lambda item: item["id"]):
        if entry.get("merged") and entry.get("layout") is None:
            continue
        try:
            if entry["id"].casefold() in keys:
                raise ValueError(tr("Plate or field names collide without case: {field}").format(field=entry["id"]))
            keys.add(entry["id"].casefold())
            image, labels, raw = _stored_arrays(src, entry, what)
            entry["channel_names"] = _field_channel_names(entry, plan["image_metadata"])
            entry["pixel_size_um"] = plan["image_metadata"]["pixel_size_um"]
            entry["image"] = _array_digest(image) if image is not None else None
            entry["labels"] = {role: _array_digest(mask) for role, mask in sorted(labels.items())}
            entry["original"] = _array_digest(raw) if raw is not None else None
            entry["legacy_planes"] = {role: f"legacy_{role}" if f"legacy_{role}" in labels else role
                                      for role in entry.get("layout", {}).get("mask_dims", {})}
            entry["mask_arrays"] = {role: _array_digest(image_store.read_labels(src / relative, role))
                                    for role, relative in entry["mask_files"].items()}
            if image is not None or labels:
                plan["fields"].append(entry)
        except (ValueError, OSError, KeyError, TypeError, EOFError, CropError) as exc:
            plan["problems"].append(str(exc))
    if "crops" in what and not plan["problems"]:
        _plan_crops(src, plan, fields)
    if not plan["fields"] and not plan["crops"] and not plan["problems"]:
        plan["problems"].append(tr("No selected legacy assets were found in this project."))
    if _inventory(src) != files:
        plan["problems"].append(tr("Source changed during planning; close the project and retry."))
    return json.loads(json.dumps(plan, allow_nan=False))


def _atomic_json(path, data):
    """Publish a flushed JSON ledger atomically on the destination filesystem."""
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _sync_directory(path.parent)


def _sync_directory(path):
    """Flush directory entries on platforms that expose directory descriptors."""
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sync_tree(path):
    """Flush completed store files before publishing converter commit records."""
    for child in sorted(path.rglob("*")):
        if child.is_file():
            with child.open("r+b") as handle:
                os.fsync(handle.fileno())
    for folder in sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True):
        _sync_directory(folder)
    _sync_directory(path)


@contextmanager
def _lock(dst):
    """Hold a process-lifetime destination lock; dead processes leave no stale lock."""
    with (dst / ".conversion.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _check_sources(src, ledger):
    """Require the same complete source generation, allowing journaled deletions."""
    expected = ledger["plan"]["files"]
    actual = _inventory(src)
    for relative, state in ledger["deletions"].items():
        if state in {"pending", "deleted"} and relative not in actual:
            actual[relative] = expected[relative]
    if actual != expected:
        raise ValueError(tr("Source changed since planning; use a new destination for this project generation."))


def _validate_resume_paths(plan):
    """Validate stored locators before a resumed writer or deletion can use them."""
    for relative in plan["files"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(tr("The conversion ledger contains an unsafe source path; preserve it and use a new destination."))
    entries = {entry["id"]: entry for entry in (*plan["fields"], *plan.get("crop_fields", []))}
    for entry in entries.values():
        _portable_name(entry["plate"])
        if (entry["key"] != image_store.field_key(entry["well"], entry["field"])
                or entry["id"] != f"{entry['plate']}/{entry['key']}"):
            raise ValueError(tr("The conversion ledger has conflicting field locators; preserve it and use a new destination."))
        sources = list(entry["mask_files"].values())
        if entry["merged"]:
            sources.append(entry["merged"])
        if any(relative not in plan["files"] for relative in sources):
            raise ValueError(tr("The conversion ledger references an unplanned source file; use a new destination."))
        for role in (*entry["mask_files"], *entry.get("labels", {})):
            _portable_name(role)


def _check_field_sources(src, entry, plan):
    """Check just one field during writes, avoiding a full-project scan per field."""
    from .crops import MERGED_LAYOUT_SIDECAR, measured_masks_path

    paths = set(entry["mask_files"].values())
    if entry["merged"]:
        paths.add(entry["merged"])
        paths.add(f"merged/{MERGED_LAYOUT_SIDECAR}")
        measured = Path(measured_masks_path(str(src / entry["merged"])))
        relative = measured.relative_to(src).as_posix()
        if relative in plan["files"]:
            paths.add(relative)
        elif measured.exists():
            raise ValueError(tr("Measured masks appeared after planning; close the project and restart conversion."))
    for relative in paths:
        path = src / relative
        if _path(path) != path or _file_digest(path) != plan["files"][relative]:
            raise ValueError(tr("Source changed since planning: {path}").format(path=path))


def _copy_support_files(src, dst, ledger):
    """Keep database, saved settings and layout metadata byte-exact in the new project."""
    files = ledger["plan"]["files"]
    copies = ledger.setdefault("support_files", {})
    for relative, expected in files.items():
        path = Path(relative)
        selected = (path.parts[0] == "settings"
                    or relative == "merged/.spacr_plane_layout.json"
                    or (len(path.parts) == 1 and path.suffix.lower() == ".json"
                        and relative not in {_LEDGER, "conversion_report.json"})
                    or (path.parts[0] == "measurements" and path.suffix.lower() == ".db"))
        if not selected or (ledger["plan"]["crops"] and relative == "measurements/measurements.db"):
            continue
        target = dst / relative
        if relative not in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.conversion-tmp")
            shutil.copyfile(src / relative, tmp)
            with tmp.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        if _file_digest(target) != expected:
            raise ValueError(tr("Copied project metadata failed verification: {path}").format(path=relative))
        copies[relative] = expected
        _atomic_json(dst / _LEDGER, ledger)


def _assert_array(array, expected, name):
    """Reject shape, dtype or pixel corruption, including NaN bit changes."""
    if _array_digest(array) != expected:
        raise ValueError(tr("Pixel verification failed for {name}; keep originals and inspect the destination.").format(name=name))


def _round_trip(array, relative, expected, temporary):
    """Encode in the legacy format through shared writers, then decode and compare."""
    from .io import _save_array_atomic
    from .tiff_io import write_tiff

    target = temporary / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array).astype(expected["dtype"], copy=False)
    if target.suffix.lower() == ".npy":
        _save_array_atomic(str(target), array)
    else:
        write_tiff(target, array)
    _assert_array(image_store.read_image(target), expected, relative)


def _verify_spacing(metadata, entry):
    """Check that unitless pixels stay unitless and saved calibration stays exact."""
    size = entry["pixel_size_um"]
    if size is None:
        valid = (not metadata.units_declared and
                 all(axis.scale == 1.0 and axis.translate == 0.0 for axis in metadata.space_axes))
    else:
        spacing = metadata.spacing
        valid = (spacing.units == "um" and all(scale == size for scale in spacing.scale)
                 and all(offset == 0.0 for offset in spacing.translate))
    if not valid:
        raise ValueError(tr("Physical scale metadata verification failed: {field}").format(field=entry["id"]))


def _verify_field(dst, entry, *, round_trip=False, ledger=None):
    """Check store arrays and reconstruct original legacy arrays with inverse axes."""
    from .ome_zarr import read_ome_zarr

    store = dst / "merged" / f"{entry['plate']}.ome.zarr"
    image = None
    if entry["image"] is not None:
        metadata = read_ome_zarr(store / entry["key"])
        if metadata.axis_names != ("c", "y", "x") or metadata.ngff_version not in ("0.5", "0.6") or metadata.name != entry["key"]:
            raise ValueError(tr("Image metadata verification failed: {field}").format(field=entry["id"]))
        if entry["channel_names"] is not None and metadata.channel_names != tuple(entry["channel_names"]):
            raise ValueError(tr("Channel metadata verification failed: {field}").format(field=entry["id"]))
        _verify_spacing(metadata, entry)
        image = image_store.read_image(store, key=entry["key"])
        _assert_array(image, entry["image"], entry["id"])
    labels = {}
    for role, expected in entry["labels"].items():
        metadata = read_ome_zarr(store / entry["key"] / "labels" / role)
        if metadata.axis_names != ("y", "x") or metadata.ngff_version not in ("0.5", "0.6") or metadata.name != role:
            raise ValueError(tr("Label metadata verification failed: {field}/{role}").format(field=entry["id"], role=role))
        _verify_spacing(metadata, entry)
        labels[role] = image_store.read_labels(store, role, key=entry["key"])
        _assert_array(labels[role], expected, f"{entry['id']}/{role}")
    if round_trip:
        with tempfile.TemporaryDirectory(prefix=".conversion-verify-", dir=dst) as folder:
            temporary = Path(folder)
            if image is not None and entry["merged"]:
                original = np.empty(entry["original"]["shape"], dtype=entry["original"]["dtype"])
                original[..., :image.shape[0]] = np.moveaxis(image, 0, -1)
                for role, dim in entry["layout"]["mask_dims"].items():
                    original[..., dim] = labels[entry["legacy_planes"][role]]
                _assert_array(original, entry["original"], entry["merged"])
                _round_trip(original, entry["merged"], entry["original"], temporary)
            for role, relative in entry["mask_files"].items():
                if role in labels:
                    _round_trip(labels[role], relative, entry["mask_arrays"][role], temporary)
            if ledger is not None:
                ledger.setdefault("verification", {})[entry["id"]] = {
                    "image": entry["image"], "labels": entry["labels"],
                    "legacy_round_trip": True}
                _atomic_json(dst / _LEDGER, ledger)


def _report(plan, status, ledger=None):
    """Return a JSON-compatible public report with explicit exclusions and blockers."""
    crop_commit = (ledger or {}).get("crops") or {}
    return {"source": plan["source"], "destination": plan["destination"],
            "status": status, "fields": plan["fields"], "problems": plan["problems"],
            "counts": {"fields": len(plan["fields"]), "crops": len(plan["crops"])},
            "excluded": [asset for asset in _ASSETS if asset not in plan["what"]],
            "notes": plan["notes"], "source_bytes": plan["source_bytes"],
            "space_estimate": tr("Source bytes only; compression and temporary export space are not predicted."),
            "crop_verification": {name: crop_commit.get(name, False)
                                  for name in ("raw_verified", "png_verified")},
            "legacy_pngs_retained": [record["legacy_png_path"] for record in plan["crops"]
                                      if (Path(plan["source"]) / record["legacy_png_path"]).exists()],
            "deletions": (ledger or {}).get("deletions", {})}


def convert_project(src, dst=None, *, what=("merged", "masks", "crops"),
                    dry_run=False, verify=True, delete_old=False):
    """Convert a closed legacy project, preserving originals unless explicitly deleted.

    :param src: local project directory containing merged images, mask files
        and optionally measurements/measurements.db with png_list rows.
    :param dst: a separate destination; defaults to the sibling suffix
        ``_ome_zarr``. Existing destinations require this converter ledger.
    :param what: nonempty subset of merged, masks and crops.
    :param dry_run: validate and report without creating files or locks.
    :param verify: export and decode legacy formats for exact verification.
    :param delete_old: delete only eligible verified source files in this
        invocation; requires verify. Databases and metadata always remain.
    :returns: JSON-compatible report. ``verified`` means legacy-format exports
        match exactly; ``raw_verified`` means native crops verify but some
        original PNG rendering cannot be reproduced and deletion is blocked.
        ``converted`` omits export verification. Dry runs return ``planned``
        or ``blocked``. Repeated calls resume compatible work after checking
        sources and every committed output. Only untimed 2-D fields are
        currently supported; one coordinator owns all writers.
    :raises ValueError: ambiguous inputs, incompatible resume, changed source,
        corruption, unsupported crop provenance or unsafe deletion.
    """
    src = _path(src)
    dst = _path(dst) if dst is not None else src.with_name(src.name + "_ome_zarr")
    _portable_name(src.name)
    _portable_name(dst.name)
    if (not src.is_dir() or str(src).casefold() == str(dst).casefold()
            or str(dst).casefold().startswith(str(src).casefold() + os.sep)
            or str(src).casefold().startswith(str(dst).casefold() + os.sep)
            or src in dst.parents or dst in src.parents):
        raise ValueError(tr("Choose an existing source and a separate, nonoverlapping destination."))
    if isinstance(what, str) or not what or set(what) - set(_ASSETS):
        raise ValueError(tr("Choose a nonempty subset of merged, masks and crops."))
    what = tuple(asset for asset in _ASSETS if asset in what)
    if delete_old and not verify:
        raise ValueError(tr("Deleting originals requires verification in the same invocation."))
    ledger_path = dst / _LEDGER
    if dst.exists():
        if not ledger_path.is_file():
            raise ValueError(tr("Destination belongs to another conversion; choose a new directory."))
        _inventory(dst, fingerprints=False)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        plan = ledger["plan"]
        _validate_resume_paths(plan)
        if (ledger.get("schema") != _SCHEMA or plan["source"] != str(src)
                or plan["destination"] != str(dst) or plan["what"] != list(what)
                or plan["versions"] != _versions()):
            raise ValueError(tr("Resume plan is incompatible; use the original inputs and options or a new destination."))
        _check_sources(src, ledger)
    else:
        plan = _plan(src, dst, what)
        ledger = {"schema": _SCHEMA, "plan": plan, "fields": {},
                  "crops": None, "deletions": {}, "status": "planned"}
    if dry_run:
        return _report(plan, "blocked" if plan["problems"] else "planned", ledger)
    if plan["problems"]:
        raise ValueError(tr("Conversion is blocked: {problems}").format(problems="; ".join(plan["problems"])))
    if not dst.exists():
        dst.mkdir(parents=True)
        _atomic_json(ledger_path, ledger)
    with _lock(dst):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("schema") != _SCHEMA or ledger.get("plan") != plan:
            raise ValueError(tr("Destination plan changed while waiting for the writer lock; inspect the existing conversion before retrying."))
        _check_sources(src, ledger)
        writers = {}
        for entry in plan["fields"]:
            if entry["id"] not in ledger["fields"]:
                _check_field_sources(src, entry, plan)
                writer = writers.get(entry["plate"])
                if writer is None:
                    writer = image_store.PlateWriter(dst / "merged", entry["plate"])
                    writers[entry["plate"]] = writer
                writer.channel_names = entry["channel_names"]
                if entry["pixel_size_um"] is not None:
                    from .layers import Spacing

                    writer.spacing = Spacing.isotropic(2, entry["pixel_size_um"], units="um")
                image, labels, _raw = _stored_arrays(src, entry, what)
                writer.write_field(entry["well"], entry["field"], image, labels)
                _sync_tree(writer.path)
                _verify_field(dst, entry)
                _check_field_sources(src, entry, plan)
                ledger["fields"][entry["id"]] = "committed"
                _atomic_json(ledger_path, ledger)
            _verify_field(dst, entry, round_trip=verify, ledger=ledger)
        _copy_support_files(src, dst, ledger)
        if plan["crops"]:
            _convert_crops(src, dst, ledger, verify=verify)
        _check_sources(src, ledger)
        ledger["status"] = "verified" if verify else "converted"
        if verify and plan["crops"] and not ledger["crops"].get("png_verified"):
            ledger["status"] = "raw_verified"
        _atomic_json(ledger_path, ledger)
        if delete_old:
            _delete_sources(src, dst, ledger)
        report = _report(plan, ledger["status"], ledger)
        _atomic_json(dst / "conversion_report.json", report)
        return report


def _delete_sources(src, dst, ledger):
    """Journal individually verified deletions; retain all unconverted data and tables."""
    plan = ledger["plan"]
    if plan["crops"] and not ledger["crops"].get("png_verified"):
        raise ValueError(tr("Legacy crop transforms could not be verified; keep the PNG files and originals."))
    eligible = set()
    for entry in plan["fields"]:
        if "merged" in plan["what"] and entry["merged"]:
            eligible.add(entry["merged"])
        if "masks" in plan["what"]:
            eligible.update(entry["mask_files"].values())
    if "crops" not in plan["what"] and any(Path(name).suffix.lower() == ".png" for name in plan["files"]):
        raise ValueError(tr("Unconverted crops still depend on the legacy project; keep source images and masks."))
    if "crops" not in plan["what"] and "measurements/measurements.db" in plan["files"]:
        raise ValueError(tr("Measurement rows may still need legacy crop sources; include and verify crops before deleting images or masks."))
    if plan["crops"]:
        eligible.update(record["legacy_png_path"] for record in plan["crops"])
    for relative in sorted(eligible):
        path = src / relative
        if not path.exists() and ledger["deletions"].get(relative) in {"pending", "deleted"}:
            ledger["deletions"][relative] = "deleted"
            _atomic_json(dst / _LEDGER, ledger)
            continue
        if _path(path) != path or _file_digest(path) != plan["files"][relative]:
            raise ValueError(tr("Source changed before deletion; keep this file: {path}").format(path=path))
        ledger["deletions"][relative] = "pending"
        _atomic_json(dst / _LEDGER, ledger)
        if _file_digest(path) != plan["files"][relative]:
            raise ValueError(tr("Source changed before deletion; keep this file: {path}").format(path=path))
        path.unlink()
        _sync_directory(path.parent)
        ledger["deletions"][relative] = "deleted"
        _atomic_json(dst / _LEDGER, ledger)


def _crop_scalar(value):
    """Keep nullable SQLite scalar values JSON-compatible without string coercion."""
    import pandas as pd

    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(tr("Crop metadata contains an unsupported scalar type; preserve the original database and resolve that column first."))


def _crop_table(db, table):
    """Read a source table without migrations, plate repairs or nullable-int coercion."""
    import sqlite3
    from .tabular import _quote_identifier, _read_query

    db = Path(db)
    wal = db.with_name(db.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise ValueError(tr("Close database writers and checkpoint the project before conversion: {path}").format(path=wal))
    uri = db.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as connection:
        present = _read_query(
            connection, "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            params=(table,), canonicalise=False, report=None, repair_plate_ids=False)
        if present.empty:
            raise ValueError(tr("Table not found in database: {table}").format(table=table))
        return _read_query(connection, f"SELECT * FROM {_quote_identifier(table)}",
                           canonicalise=False, report=None, repair_plate_ids=False,
                           dtype_backend="numpy_nullable", coerce_float=False)


def _crop_metadata(row):
    """Preserve column names and supported scalar values from one source row."""
    return {str(key): _crop_scalar(value) for key, value in row.items()}


def _crop_row_key(row):
    """Resolve complete untimed field identity through the shared schema helpers."""
    from . import schema

    canonical = {}
    for column, value in row.items():
        name = schema.canonical_column_name(column)
        if name not in {"plateID", "rowID", "columnID", "fieldID", "timeID"}:
            continue
        if name in canonical and canonical[name] != value:
            raise ValueError(tr("Crop identity aliases conflict; resolve the saved table keys before conversion."))
        canonical[name] = value
    if canonical.get("timeID") is not None:
        raise ValueError(tr("Timed crop identities require a time-aware field store; retain this project until that shared API is available."))
    required = ("plateID", "rowID", "columnID", "fieldID")
    if any(canonical.get(name) is None for name in required):
        raise ValueError(tr("Crop rows need plate, row, column and field identities; restore the original acquisition keys."))
    return (str(canonical["plateID"]), schema.row_id(canonical["rowID"], strict=True),
            schema.column_id(canonical["columnID"], strict=True),
            schema.field_id(canonical["fieldID"], strict=True))


def _crop_role(row):
    """Require an explicit crop role or exactly one populated object-ID column."""
    from .png_list import PNG_LIST_ID_COLUMNS

    explicit = {str(row[name]) for name in ("object_type", "crop_mode")
                if row.get(name) is not None}
    if len(explicit) > 1:
        raise ValueError(tr("Crop mode columns conflict; restore one documented object role per PNG row."))
    if explicit:
        role = next(iter(explicit))
    else:
        present = [role for role, column in PNG_LIST_ID_COLUMNS.items()
                   if row.get(column) is not None]
        if len(present) != 1:
            raise ValueError(tr("Crop mode is ambiguous; record the original object role instead of choosing an overlapping object."))
        role = present[0]
    if role not in PNG_LIST_ID_COLUMNS:
        raise ValueError(tr("Crop role is unsupported by the shared crop reader: {role}").format(role=role))
    return role


def _crop_relative_png(src, value, files):
    """Require the recorded PNG to identify one existing inventoried source file."""
    from .crops import normalise_separators

    if not isinstance(value, str) or not value.strip():
        raise ValueError(tr("A crop row has no PNG path; restore its original source record."))
    normalized = normalise_separators(value)
    if ".." in Path(normalized).parts:
        raise ValueError(tr("Crop paths contain parent traversal; restore a direct path inside the source project."))
    windows = PureWindowsPath(value)
    if os.name != "nt" and (windows.drive or normalized.startswith("//")):
        raise ValueError(tr("Reanchor the saved Windows crop paths to this local project before conversion."))
    candidate = Path(normalized)
    candidate = candidate if candidate.is_absolute() else src / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(src).as_posix()
    except ValueError as exc:
        raise ValueError(tr("Crop PNG paths must stay inside the selected source project.")) from exc
    if relative not in files or candidate.suffix.lower() != ".png":
        raise ValueError(tr("A recorded crop PNG is missing from the source inventory: {path}").format(path=relative))
    return relative


def _crop_saved_policy(src):
    """Resolve saved crop policies without guessing between conflicting run histories."""
    import ast

    values = {"crop_margin_px": [], "crop_zero_outside": []}
    paths = [src / "measurements" / "measurements.db"]
    paths.extend(src / "settings" / name for name in (
        "measure_crop_settings.csv", "measure_settings.csv"))
    for path in paths:
        settings = _recorded_settings(path)
        for key, known in values.items():
            if key not in settings:
                continue
            value = settings[key]
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            if not any(type(value) is type(old) and value == old for old in known):
                known.append(value)
    if any(len(known) > 1 for known in values.values()):
        raise ValueError(tr("Saved crop margin or outside-pixel policies conflict between runs; resolve their provenance before conversion."))
    return {key: known[0] for key, known in values.items() if known}


def _crop_policy(row, saved):
    """Prefer explicit row policies, then saved policies, then the shared defaults."""
    margins = [row[name] for name in ("margin_px", "crop_margin_px") if name in row]
    zeros = [row[name] for name in ("zero_outside", "crop_zero_outside") if name in row]
    if ((len(margins) > 1 and margins[0] != margins[1])
            or (len(zeros) > 1 and zeros[0] != zeros[1])):
        raise ValueError(tr("Crop policy aliases conflict within a source row; restore the recorded margin and outside-pixel policy."))
    margin = margins[0] if margins else saved.get("crop_margin_px", 0)
    zero = zeros[0] if zeros else saved.get("crop_zero_outside", False)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError(tr("Recorded crop margin must be a nonnegative integer; resolve the saved crop policy."))
    if isinstance(zero, int) and not isinstance(zero, bool) and zero in (0, 1):
        zero = bool(zero)
    if not isinstance(zero, bool):
        raise ValueError(tr("Recorded crop zero_outside must be true or false; resolve the saved crop policy."))
    return margin, zero


def _crop_cut(image, labels, label, margin=0, zero_outside=False):
    """Use the shared cutter on the exact object mask without a max-label allocation."""
    from .crop_store import cut_crops

    inside = np.equal(labels, label)
    if not inside.any():
        raise ValueError(tr("A crop label is absent from its effective mask: {label}").format(label=label))
    crops = cut_crops(image, inside.astype(np.uint8), margin=margin,
                      zero_outside=zero_outside, objects=[1])
    if len(crops) != 1:
        raise ValueError(tr("The shared crop cutter did not return the requested object; retain the original project."))
    crop = crops[0]
    crop["object_label"] = label
    return crop


def _crop_role_mask(src, entry, labels, raw, role):
    """Read a crop mask, including shared effective cytoplasm derivation when needed."""
    if role in labels:
        return labels[role]
    if role == "cytoplasm" and raw is not None and entry["merged"]:
        from .crops import MergedField

        field = MergedField(str(src / entry["merged"]), array=raw,
                            mask_dims=entry["layout"]["mask_dims"])
        return _storage_array(field.mask_plane(role))
    raise ValueError(tr("A crop needs its native merged intensities and effective object mask; retain this PNG as legacy-only."))


def _plan_crops(src, plan, fields):
    """Validate every legacy crop identity and retain source rows before any writes."""
    import pandas as pd
    import sqlite3
    from decimal import Decimal, InvalidOperation
    from . import schema
    from .crop_store import CROP_COLUMNS
    from .crops import CropError, read_crop_array
    from .png_list import PNG_LIST_ID_COLUMNS, _object_id_int, crop_rows_from_png_list

    db_relative = "measurements/measurements.db"
    crop_pngs = {name for name in plan["files"] if name.lower().endswith(".png")
                 and any(part.endswith("_png") for part in Path(name).parts[:-1])}
    if db_relative not in plan["files"]:
        if crop_pngs:
            plan["problems"].append(tr("Crop PNGs have no measurements database; restore their original metadata before conversion."))
        return
    db = src / db_relative
    try:
        png_table = _crop_table(db, "png_list")
    except (ValueError, OSError, sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
        if "Table not found in database: png_list" not in str(exc):
            plan["problems"].append(str(exc))
        elif crop_pngs:
            plan["problems"].append(tr("Crop PNGs have no png_list metadata; restore the saved crop records before conversion."))
        return
    try:
        existing = _crop_table(db, "ssCell_Crops")
    except (ValueError, OSError, sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
        if "Table not found in database: ssCell_Crops" not in str(exc):
            plan["problems"].append(str(exc))
            return
    else:
        if len(existing):
            plan["problems"].append(tr("This project already has a crop-store index; use its existing store instead of replacing its locators."))
            return
    try:
        saved_policy = _crop_saved_policy(src)
        _crop_policy({}, saved_policy)
    except (ValueError, OSError, TypeError) as exc:
        plan["problems"].append(str(exc))
        return
    by_key = {}
    for entry in fields.values():
        if not entry.get("merged"):
            continue
        parsed = schema.parse_field_stem(Path(entry["merged"]).name)
        key = (parsed.plateID, parsed.rowID, parsed.columnID, parsed.fieldID)
        if key in by_key:
            plan["problems"].append(tr("Multiple merged fields share a crop identity; resolve the original field records."))
            return
        by_key[key] = entry
    object_tables, records, identities, paths, object_ids = {}, [], set(), set(), set()
    for position, values in enumerate(png_table.to_dict(orient="records")):
        try:
            metadata = _crop_metadata(values)
            role = _crop_role(metadata)
            key = _crop_row_key(metadata)
            entry = by_key.get(key)
            if entry is None:
                raise ValueError(tr("A crop has no matching merged field with its complete identity; restore that field before conversion."))
            label = _object_id_int(metadata.get(PNG_LIST_ID_COLUMNS[role]))
            if label is None or label <= 0:
                raise ValueError(tr("A crop has no single positive object label; multi-object and missing-label PNGs must remain legacy-only."))
            identity = (*key, None, role, label)
            if identity in identities:
                raise ValueError(tr("Duplicate crop identities cannot be assigned one locator; resolve duplicate source rows first."))
            png = _crop_relative_png(src, metadata.get("png_path"), plan["files"])
            if png in paths:
                raise ValueError(tr("Multiple crop records reuse one PNG path; resolve the original crop identities."))
            supplied = pd.DataFrame([metadata], dtype=object)
            supplied["path_name"] = str(src / entry["merged"])
            resolved = crop_rows_from_png_list(db, supplied, role, verbose=False)
            if (len(resolved) != 1 or resolved.iloc[0]["object_label_type"] != role
                    or int(resolved.iloc[0]["object_label"]) != label):
                raise ValueError(tr("The shared crop reader cannot resolve this exact source row; restore its original role and label."))
            if role not in object_tables:
                try:
                    objects = _crop_table(db, role)
                except (ValueError, OSError, sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
                    if f"Table not found in database: {role}" not in str(exc):
                        raise
                    objects = None
                mapping = {}
                if objects is not None:
                    for object_values in objects.to_dict(orient="records"):
                        object_metadata = _crop_metadata(object_values)
                        object_key = _crop_row_key(object_metadata)
                        object_label = _object_id_int(object_metadata.get("object_label"))
                        full_key = (*object_key, object_label)
                        if object_label is None or object_label <= 0 or full_key in mapping:
                            raise ValueError(tr("The object measurement table has ambiguous labels or duplicate identities; resolve it before conversion."))
                        mapping[full_key] = object_metadata
                object_tables[role] = (objects is not None, mapping)
            table_exists, mapping = object_tables[role]
            object_metadata = mapping.get((*key, label))
            if table_exists and object_metadata is None:
                raise ValueError(tr("The crop label is absent from its object measurement table; restore matching source records."))
            combined = dict(metadata)
            for name, value in (object_metadata or {}).items():
                if name not in combined:
                    combined[name] = value
                elif combined[name] != value and name not in {
                        "file_name", "path_name", "prcf", "prcfo", "object_label",
                        "plateID", "rowID", "columnID", "fieldID", "timeID",
                        *PNG_LIST_ID_COLUMNS.values()}:
                    raise ValueError(tr("Crop and object metadata disagree in {column}; resolve the saved annotation or relationship before conversion.").format(column=name))
            source_id = combined.get("object_id")
            if source_id is not None and (not isinstance(source_id, str) or not source_id):
                raise ValueError(tr("An existing crop object_id must be a nonempty unique string; preserve and repair its source identity first."))
            object_id = source_id or f"{entry['plate']}_{entry['well']}_{entry['field']}_{role}_{label}"
            if object_id in object_ids:
                raise ValueError(tr("Crop object_id values are not unique; restore complete object identities before conversion."))
            reserved = set(CROP_COLUMNS) - {"png_path", "object_id", "object_type", "object_label", "timepoint", "margin_px"}
            if any(name in combined for name in reserved):
                raise ValueError(tr("Legacy crop metadata already uses reserved crop-store columns; resolve those conflicting fields before conversion."))
            if combined.get("timepoint") is not None:
                raise ValueError(tr("Timed crop identities require a time-aware field store; retain this project until that shared API is available."))
            margin, zero_outside = _crop_policy(combined, saved_policy)
            relationships = ["parent_cell_id", "nucleus_id", "has_nucleus", "n_nuclei",
                             "n_pathogens", "infected"]
            relationships.extend(f"n_{name}" for name in schema.ORGANELLE_ROLES
                                 if name == "organelle" or f"n_{name}" in combined
                                 or name in entry["layout"]["mask_dims"]
                                 or name in entry["mask_files"])
            relationships.extend(["annotation", "class", "class_score"])
            extra = {name: combined.get(name) for name in relationships}
            for name, value in extra.items():
                if name in {"parent_cell_id", "nucleus_id"} and value is not None:
                    number = _object_id_int(value)
                    if number is not None:
                        extra[name] = number
                        continue
                if (not isinstance(value, str) or name in {"annotation", "class"}):
                    continue
                try:
                    number = Decimal(value.strip())
                    if not number.is_finite():
                        raise ValueError
                    if name == "class_score":
                        converted = float(number)
                        if not np.isfinite(converted) or Decimal(str(converted)) != number:
                            raise ValueError
                    else:
                        if number != number.to_integral_value():
                            raise ValueError
                        converted = int(number)
                except (InvalidOperation, ValueError, OverflowError):
                    raise ValueError(tr("Crop index column {column} does not match its native scalar type; restore that column before conversion.").format(column=name)) from None
                extra[name] = converted
            normalized = _crop_index_frame([{"object_id": object_id, **extra}])
            extra = {name: _crop_scalar(normalized.iloc[0][name]) for name in extra}
            records.append({"source_row": position, "field_id": entry["id"],
                            "object_id": object_id, "object_type": role,
                            "object_label": label, "timepoint": None,
                            "legacy_png_path": png, "metadata": extra,
                            "source_metadata": metadata, "margin_px": margin,
                            "zero_outside": zero_outside,
                            "legacy_png_contract": _legacy_png_contract(src, metadata, role, entry)})
            identities.add(identity)
            paths.add(png)
            object_ids.add(object_id)
        except (ValueError, OSError, KeyError, TypeError, sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
            plan["problems"].append(tr("Crop row {row}: {problem}").format(row=position, problem=exc))
    if crop_pngs - paths:
        plan["problems"].append(tr("Some crop PNGs have no validated metadata row; restore those records instead of silently omitting files."))
    layouts, metadata_types = {}, {}
    active_field, loaded = None, None
    for record in sorted(records, key=lambda item: (item["field_id"], item["object_type"], item["object_label"])):
        try:
            entry = fields[record["field_id"]]
            if active_field != entry["id"]:
                loaded = _load_field(src, entry)
                active_field = entry["id"]
            image, labels, _raw = loaded
            role = record["object_type"]
            if image is None:
                raise ValueError(tr("A crop needs its native merged intensities and effective object mask; retain this PNG as legacy-only."))
            mask = _crop_role_mask(src, entry, labels, _raw, role)
            if (mask.dtype.kind not in "buif" or not np.isfinite(mask).all()
                    or np.any(mask < 0) or np.any(mask != np.floor(mask))):
                raise ValueError(tr("Crop masks must contain finite nonnegative integer labels; recover the original labelled mask."))
            crop = _crop_cut(image, mask, record["object_label"],
                             record["margin_px"], record["zero_outside"])
            record["pixels"] = _array_digest(crop["pixels"])
            record["mask"] = _array_digest(crop["mask"])
            record["geometry"] = {name: crop[name] for name in (
                "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1", "margin_px", "clipped_at_edge")}
            record["png_container_view"] = _array_digest(image_store.read_image(src / record["legacy_png_path"]))
            record["png"] = _array_digest(read_crop_array(src / record["legacy_png_path"]))
            group = (entry["plate"], entry["well"], role)
            layout = (image.dtype.str, tuple(entry["layout"]["intensity_channels"]))
            if group in layouts and layouts[group] != layout:
                raise ValueError(tr("A crop pool has conflicting native dtypes or channel maps; split the source into compatible projects."))
            layouts[group] = layout
            for name, value in record["metadata"].items():
                if value is not None:
                    metadata_types.setdefault(name, set()).add(type(value).__name__)
        except (ValueError, OSError, KeyError, TypeError, CropError) as exc:
            plan["problems"].append(tr("Crop row {row}: {problem}").format(row=record["source_row"], problem=exc))
    if any(len(types) > 1 and not types <= {"int", "float"}
           for types in metadata_types.values()):
        plan["problems"].append(tr("Crop metadata mixes scalar types within a column; preserve the source database and resolve that column before conversion."))
    plan["crops"] = sorted(records, key=lambda item: (
        item["field_id"], item["object_type"], item["object_label"]))
    plan["crop_fields"] = [fields[key] for key in sorted({record["field_id"] for record in records})]
    plan["crop_pool_layouts"] = {}
    if records and not plan["problems"]:
        import psutil

        budget = min(256 * 1024 * 1024, int(psutil.virtual_memory().available) // 8)
        maximum_bytes = {}
        for record in records:
            entry = fields[record["field_id"]]
            group = json.dumps([entry["plate"], entry["well"], record["object_type"]])
            pixels = record["pixels"]
            size = int(np.prod(pixels["shape"])) * np.dtype(pixels["dtype"]).itemsize
            size += int(np.prod(record["mask"]["shape"]))
            maximum_bytes[group] = max(maximum_bytes.get(group, 0), size)
        for group, size in maximum_bytes.items():
            if size * 3 > budget:
                plan["problems"].append(tr("One raw crop exceeds the bounded conversion buffer; use a host with a suitable shared crop writer before conversion."))
            count = 4096
            while count > 1 and count * size * 3 > budget:
                count //= 2
            plan["crop_pool_layouts"][group] = {
                "crops_per_shard": count, "buffer_budget_bytes": budget,
                "maximum_crop_bytes": size,
                "reason": "Bound buffered crops and writer copies to the available memory budget."}
    if records:
        plan["notes"].extend([
            tr("Raw crops use all native intensity channels and the recorded crop policy; only absent policies use margin 0 and zero_outside=False."),
            tr("Crop conversion supports untimed 2-D objects; unresolved identities, missing source masks and unsupported metadata remain blocked."),
            tr("Legacy PNG rendering is verified separately; a raw_verified result retains PNGs that cannot be exported exactly.")])


def _crop_index_frame(rows):
    """Build native Measure index columns with explicit nullable scalar dtypes."""
    import pandas as pd
    from .crop_store import CROP_COLUMNS
    from .schema import ORGANELLE_ROLES

    relationships = ["parent_cell_id", "nucleus_id", "has_nucleus", "n_nuclei",
                     "n_pathogens", "infected", *(f"n_{name}" for name in ORGANELLE_ROLES),
                     "annotation", "class", "class_score"]
    present = {name for row in rows for name in row}
    if present - set(CROP_COLUMNS) - set(relationships):
        raise ValueError(tr("Crop index columns differ from the native Measure schema; retain originals."))
    columns = [*CROP_COLUMNS, *(name for name in relationships if name in present)]
    frame = pd.DataFrame(rows, dtype=object).reindex(columns=columns)
    integers = {"column", "field", "timepoint", "object_label", "crop_index", "offset",
                "height", "width", "mask_offset", "bbox_y0", "bbox_x0", "bbox_y1",
                "bbox_x1", "margin_px", "parent_cell_id", "nucleus_id", "n_nuclei",
                "n_pathogens", *(f"n_{name}" for name in ORGANELLE_ROLES)}
    booleans = {"clipped_at_edge", "has_nucleus", "infected"}
    for name in columns:
        values = [_crop_scalar(value) for value in frame[name]]
        if name in integers or name in booleans or name == "class_score":
            for value in values:
                if value is None:
                    continue
                if (not isinstance(value, (bool, int, float))
                        or name in integers and int(value) != value
                        or name in booleans and value not in (False, True)):
                    raise ValueError(tr("Crop index column {column} does not match its native scalar type; restore that column before conversion.").format(column=name))
            dtype = "boolean" if name in booleans else "Float64" if name == "class_score" else "Int64"
            frame[name] = pd.array(values, dtype=dtype)
        else:
            frame[name] = pd.Series(values, dtype=object).convert_dtypes()
    return frame


def _crop_table_rows(rows):
    """Normalize index rows for exact null, scalar-type and column comparisons."""
    result = {}
    for raw in _crop_index_frame(rows).to_dict(orient="records"):
        row = _crop_metadata(raw)
        object_id = row["object_id"]
        if object_id in result:
            raise ValueError(tr("The converted crop index contains duplicate object_id values; retain originals."))
        result[object_id] = row
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verify_crop_outputs(project, records, rows, *, round_trip):
    """Verify raw pixels, masks, index metadata and optional legacy-format exports."""
    from .crop_store import OmeZarrCropSource, export_crops
    from .crops import read_crop_array

    stored = _crop_table(project / "measurements" / "measurements.db", "ssCell_Crops")
    if (list(stored.columns) != list(_crop_index_frame(rows).columns)
            or _crop_table_rows(stored.to_dict(orient="records")) != _crop_table_rows(rows)):
        raise ValueError(tr("Converted crop metadata differs from the committed source rows; retain originals."))
    source = OmeZarrCropSource(str(project), context=True, size=None)
    by_id = {row["object_id"]: row for row in rows}
    png_verified = True
    checks = []
    with tempfile.TemporaryDirectory(prefix=".crop-verification-", dir=project) as temporary:
        for index, record in enumerate(records):
            row = by_id[record["object_id"]]
            if row["png_path"] is not None:
                raise ValueError(tr("Converted crop rows must use store locators instead of legacy PNG paths."))
            if (row["offset"] < 0 or row["mask_offset"] != row["offset"]
                    or [row["height"], row["width"]] != record["mask"]["shape"]
                    or min(row["height"], row["width"]) <= 0):
                raise ValueError(tr("Crop locator bounds do not match the source object; retain originals."))
            pixels, mask = source.get_raw({"object_id": record["object_id"]})
            _assert_array(pixels, record["pixels"], record["object_id"])
            _assert_array(mask, record["mask"], record["object_id"] + " mask")
            for name, expected in record["geometry"].items():
                if row[name] != expected:
                    raise ValueError(tr("Crop geometry differs from its source object; retain originals."))
            if not round_trip:
                png_verified = False
                continue
            folder = Path(temporary) / str(index)
            npy = export_crops(project, [row], folder / "npy", fmt="npy", context=True)[0]
            _assert_array(image_store.read_image(npy), record["pixels"], npy)
            tiff = export_crops(project, [row], folder / "tiff", fmt="tiff", context=True)[0]
            _assert_array(image_store.read_image(tiff), record["pixels"], tiff)
            mask_path = Path(tiff).with_name(Path(tiff).stem + "_mask.tif")
            exported_mask = image_store.read_image(mask_path)
            if (exported_mask.dtype != np.uint8 or exported_mask.shape != mask.shape
                    or not np.array_equal(exported_mask, mask.astype(np.uint8))):
                raise ValueError(tr("Exported crop mask differs from the source object; retain originals."))
            if pixels.dtype == np.uint8:
                png = export_crops(project, [row], folder / "png", fmt="png", context=True)[0]
                exact_png = (_array_digest(read_crop_array(png)) == record["png"]
                             and _array_digest(image_store.read_image(png)) == record["png_container_view"])
            else:
                exact_png = False
            if not exact_png and record.get("legacy_png_contract") is not None:
                exact_png = _verify_legacy_png(project, row, record, folder / "legacy_png")
            png_verified = exact_png and png_verified
            checks.append({"object_id": record["object_id"], "pixels": record["pixels"],
                           "mask": record["mask"], "png_verified": exact_png})
        proof = {"raw_verified": bool(round_trip), "png_verified": bool(png_verified),
                 "objects": checks}
        if round_trip:
            receipt = project / "crop_verification.json"
            if not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8")) != proof:
                _atomic_json(receipt, proof)
    return proof


def _crop_group_attributes(plan, entry, records, pool_layout):
    """Describe one pool class through the shared writer metadata contract."""
    policies = sorted({(record["margin_px"], record["zero_outside"]) for record in records})
    settings = {"image_metadata": plan["image_metadata"],
                "channel_map": entry["layout"]["intensity_channels"],
                "policies": policies, "pool_layout": pool_layout}
    return json.loads(json.dumps({
        "schema": 1, "axes": ["c", "y", "x"], "pool_axes": ["c", "pixel"],
        "mask_axes": ["pixel"], "channel_map": settings["channel_map"],
        "channel_names": entry.get("channel_names"),
        "pixel_size_um": entry.get("pixel_size_um"),
        "physical_scale_unit": "micrometer" if entry.get("pixel_size_um") is not None else None,
        "store_versions": plan["versions"], "pool_layout": pool_layout,
        "settings_hash": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "policies": policies,
        "metadata_columns": sorted({name for record in records for name in record["metadata"]}),
        "relationship_columns": sorted({name for record in records for name in record["metadata"]
                                        if name.startswith("n_") or name in {
                                            "parent_cell_id", "nucleus_id", "has_nucleus", "infected"}}),
        "source": {"database": plan["files"]["measurements/measurements.db"],
                   "fields": sorted({record["field_id"] for record in records})}}))


def _verify_crop_group(project, fields, records, commit, attributes):
    """Recheck a shared well/class commit before reusing its rows or publishing them."""
    from .crop_store import OmeZarrCropSource, class_dir, pool_attributes

    first = fields[records[0]["field_id"]]
    plate, well, role = first["plate"], first["well"], records[0]["object_type"]
    actual = pool_attributes(project, plate, well, role)
    required = {**attributes, "dtype": records[0]["pixels"]["dtype"],
                "channels": records[0]["pixels"]["shape"][0], "compression": "zstd-3",
                "crops_per_shard": attributes["pool_layout"]["crops_per_shard"]}
    if (commit["plate"] != plate or commit["well"] != well or commit["object_type"] != role
            or actual != commit["attributes"]
            or any(actual.get(key) != value for key, value in required.items())):
        raise ValueError(tr("Committed crop attributes differ from the conversion plan; retain originals."))
    row_map = {row["object_id"]: row for row in commit["rows"]}
    if len(row_map) != len(commit["rows"]) or set(row_map) != {record["object_id"] for record in records}:
        raise ValueError(tr("Committed crop membership differs from the conversion plan; retain originals."))
    reader = OmeZarrCropSource(str(project), context=True, size=None)
    store = class_dir(project, plate, well, role).parent.relative_to(project).as_posix()
    row_letter, column = image_store.split_well(well)
    for record in records:
        row = row_map[record["object_id"]]
        entry = fields[record["field_id"]]
        expected = {"plate": plate, "well": well, "row": row_letter, "column": int(column),
                    "field": entry["field"], "timepoint": record["timepoint"],
                    "object_id": record["object_id"], "object_type": role,
                    "object_label": record["object_label"], "png_path": None,
                    **record["geometry"], **record["metadata"]}
        if json.dumps({key: row.get(key) for key in expected}, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(tr("Committed crop rows differ from the conversion plan; retain originals."))
        locator = str(row["crop_array"])
        if (row["crop_store"] != store or locator not in {f"{role}/pool/{pool}" for pool in commit["pools"]}
                or row["offset"] < 0 or row["mask_offset"] != row["offset"]
                or [row["height"], row["width"]] != record["mask"]["shape"]):
            raise ValueError(tr("Committed crop locators differ from the conversion plan; retain originals."))
        pixels, mask = reader.get_raw(row)
        _assert_array(pixels, record["pixels"], record["object_id"])
        _assert_array(mask, record["mask"], record["object_id"] + " mask")


def _convert_crops(src, dst, ledger, verify):
    """Stage a complete crop phase and publish its tree and copied database together."""
    import contextlib
    from .crop_store import CropPoolWriter, discard_uncommitted, read_commit
    from .database_concurrency import connect
    from .tabular import write_database

    plan = ledger["plan"]
    records = plan["crops"]
    if not records:
        return
    committed = ledger.get("crops")
    if committed is not None:
        if _inventory(dst / "crops") != committed["files"]:
            raise ValueError(tr("Committed crop files changed; keep originals and inspect the destination."))
        if _file_digest(dst / "measurements" / "measurements.db") != committed["database"]:
            raise ValueError(tr("Committed crop metadata changed; keep originals and inspect the destination."))
        proof = _verify_crop_outputs(dst, records, committed["rows"], round_trip=verify)
        if verify and not committed["raw_verified"]:
            committed["raw_verified"] = True
            committed["png_verified"] = proof["png_verified"]
            committed["verification"] = proof
            _atomic_json(dst / _LEDGER, ledger)
        return
    stage = dst / ".conversion-crops"
    if stage.is_symlink():
        raise ValueError(tr("Crop staging must be an ordinary converter-owned directory."))
    phase = ledger.get("crop_phase")
    if stage.exists():
        if phase not in {"building", "publishing"}:
            raise ValueError(tr("Crop staging has no converter ownership record; choose a clean destination."))
        if phase == "publishing":
            shutil.rmtree(stage)
    if phase == "publishing":
        if (dst / "crops").exists():
            shutil.rmtree(dst / "crops")
        (dst / "measurements" / "measurements.db").unlink(missing_ok=True)
        (dst / "crop_verification.json").unlink(missing_ok=True)
    elif (dst / "crops").exists() or (dst / "measurements" / "measurements.db").exists():
        raise ValueError(tr("Uncommitted crop output already exists outside staging; choose a clean destination."))
    ledger["crop_phase"] = "building"
    _atomic_json(dst / _LEDGER, ledger)
    stage.mkdir(exist_ok=True)
    fields = {entry["id"]: entry for entry in plan["crop_fields"]}
    groups = {}
    for record in records:
        entry = fields[record["field_id"]]
        groups.setdefault((entry["plate"], entry["well"], record["object_type"]), []).append(record)
    rows = []
    for (plate, well, role), group in sorted(groups.items()):
        entry = fields[group[0]["field_id"]]
        pool_layout = plan["crop_pool_layouts"][json.dumps([plate, well, role])]
        attributes = _crop_group_attributes(plan, entry, group, pool_layout)
        previous = read_commit(stage, plate, well, role)
        if previous is not None:
            _verify_crop_group(stage, fields, group, previous, attributes)
            discard_uncommitted(stage, plate, well, role)
            rows.extend(previous["rows"])
            continue
        discard_uncommitted(stage, plate, well, role)
        writer, loaded, active_field = None, None, None
        for record in group:
            entry = fields[record["field_id"]]
            if active_field != entry["id"]:
                loaded = _load_field(src, entry)
                active_field = entry["id"]
            image, labels, _raw = loaded
            mask = _crop_role_mask(src, entry, labels, _raw, role)
            crop = _crop_cut(image, mask, record["object_label"],
                             record["margin_px"], record["zero_outside"])
            _assert_array(crop["pixels"], record["pixels"], record["object_id"])
            _assert_array(crop["mask"], record["mask"], record["object_id"] + " mask")
            if writer is None:
                writer = CropPoolWriter(stage, plate, well, role, image.shape[0],
                                        dtype=image.dtype,
                                        crops_per_shard=pool_layout["crops_per_shard"],
                                        attributes=attributes)
            if image.dtype != writer.dtype or image.shape[0] != writer.channels:
                raise ValueError(tr("Crop pool dtype or channel count changed after planning; retain originals."))
            crop["object_id"] = record["object_id"]
            writer.add([crop], field=entry["field"], timepoint=record["timepoint"])
        group_rows = writer.close()
        metadata = {record["object_id"]: record["metadata"] for record in group}
        for row in group_rows:
            row.update(metadata[row["object_id"]])
        committed_group = writer.commit()
        _sync_tree(writer.path)
        _verify_crop_group(stage, fields, group, committed_group, attributes)
        rows.extend(committed_group["rows"])
    frame = _crop_index_frame(rows)
    rows = [_crop_metadata(row) for row in frame.to_dict(orient="records")]
    copied_db = stage / "measurements" / "measurements.db"
    copied_db.parent.mkdir(exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(copied_db) + suffix).unlink(missing_ok=True)
    shutil.copy2(src / "measurements" / "measurements.db", copied_db)
    with contextlib.closing(connect(copied_db, journal_mode="DELETE")):
        pass
    write_database(frame, copied_db, "ssCell_Crops",
                   if_exists="replace", canonicalise=False, migrate=False)
    copied_png_rows = [_crop_metadata(row) for row in
                       _crop_table(copied_db, "png_list").to_dict(orient="records")]
    original_png_rows = [record["source_metadata"] for record in
                         sorted(records, key=lambda item: item["source_row"])]
    if copied_png_rows != original_png_rows:
        raise ValueError(tr("Copied legacy crop metadata differs from the source database; retain originals."))
    proof = _verify_crop_outputs(stage, records, rows, round_trip=verify)
    _check_sources(src, ledger)
    _sync_tree(stage)
    crop_files = _inventory(stage / "crops")
    database = _file_digest(copied_db)
    ledger["crop_phase"] = "publishing"
    _atomic_json(dst / _LEDGER, ledger)
    os.replace(stage / "crops", dst / "crops")
    (dst / "measurements").mkdir(exist_ok=True)
    os.replace(copied_db, dst / "measurements" / "measurements.db")
    if (stage / "crop_verification.json").exists():
        os.replace(stage / "crop_verification.json", dst / "crop_verification.json")
    _sync_directory(dst / "measurements")
    _sync_directory(dst)
    ledger["crops"] = {"rows": rows, "files": crop_files, "database": database,
                       "raw_verified": bool(verify), "png_verified": proof["png_verified"],
                       "verification": proof}
    ledger["crop_phase"] = "committed"
    _atomic_json(dst / _LEDGER, ledger)
    shutil.rmtree(stage)
    _sync_directory(dst)


def _legacy_png_settings(src):
    """Read consistent recorded PNG settings without guessing across run histories."""
    import ast

    paths = [src / "measurements" / "measurements.db"]
    paths.extend(src / "settings" / name for name in (
        "measure_crop_settings.csv", "measure_settings.csv"))
    names = {"png_size", "png_dims", "png_channel_mapping", "crop_intensity",
             "crop_mode", "use_bounding_box", "dialate_pngs",
             "illumination_correction", "flat_field_sha256"}
    result, conflicting = {}, False
    for path in paths:
        for name, value in _recorded_settings(path).items():
            if name not in names:
                continue
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            if name in result and (type(result[name]) is not type(value) or result[name] != value):
                conflicting = True
            result[name] = value
    if conflicting:
        return None
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError):
        return None


def _legacy_png_role_value(settings, name, role):
    """Resolve explicitly recorded per-role values without a first-role fallback."""
    value = settings.get(name)
    per_role = (isinstance(value, (list, tuple)) and
                (name != "png_size" or value and isinstance(value[0], (list, tuple))))
    if not per_role:
        return value
    modes = settings.get("crop_mode")
    if isinstance(modes, str):
        modes = [modes]
    if (not isinstance(modes, (list, tuple)) or len(modes) != len(value)
            or len(set(modes)) != len(modes) or role not in modes):
        return None
    return value[list(modes).index(role)]


def _legacy_png_contract(src, source_metadata, role, field_entry):
    """Prove a raw masked-centroid PNG recipe from explicit saved provenance."""
    from .crops import (CROP_FORMAT_SIDECAR, CropError, normalise_separators,
                        read_crop_folder_marker, resolve_png_channel_mapping)

    try:
        if "legacy_png_settings" not in field_entry:
            field_entry["legacy_png_settings"] = _legacy_png_settings(src)
        settings = field_entry["legacy_png_settings"]
        if settings is None:
            return None
        settings = dict(settings)
        for name in ("png_size", "png_dims", "png_channel_mapping", "crop_intensity",
                     "crop_mode", "use_bounding_box", "dialate_pngs",
                     "illumination_correction", "flat_field_sha256"):
            if name in source_metadata:
                value = source_metadata[name]
                if name in settings and settings[name] != value:
                    return None
                settings[name] = value
        if settings.get("crop_intensity") != "raw":
            return None
        if settings.get("illumination_correction") not in (None, False, 0):
            return None
        if settings.get("flat_field_sha256"):
            return None
        raw_path = source_metadata.get("png_path")
        if not isinstance(raw_path, str):
            return None
        path = Path(normalise_separators(raw_path))
        path = path if path.is_absolute() else src / path
        marker = read_crop_folder_marker(str(path.parent), use_cache=False)
        if (path.parent / CROP_FORMAT_SIDECAR).exists() and marker is None:
            return None
        if marker and (marker.get("flat_field_sha256") or marker.get("migration")
                       or marker.get("unconverted") or marker.get("migrated_from")):
            return None
        formats = {value for value in (source_metadata.get("crop_format"),
                   (marker or {}).get("spacr_crop_format")) if value is not None}
        if formats != {4}:
            return None
        mask_mode = (marker or {}).get("mask_mode")
        boxed = settings.get("use_bounding_box")
        if boxed not in (None, False, 0) or mask_mode not in (None, "masked"):
            return None
        if boxed is None and mask_mode != "masked":
            return None
        dilated = False if role == "cytoplasm" else _legacy_png_role_value(settings, "dialate_pngs", role)
        if dilated is None or dilated not in (False, 0):
            return None
        size = _legacy_png_role_value(settings, "png_size", role)
        if isinstance(size, int) and not isinstance(size, bool):
            size = [size, size]
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in size)):
            return None
        raw_mapping = settings.get("png_channel_mapping")
        dims = settings.get("png_dims")
        if raw_mapping is not None:
            if not isinstance(raw_mapping, dict):
                return None
            values = list(raw_mapping.values())
        elif isinstance(dims, (list, tuple)) and 1 <= len(dims) <= 3:
            values = list(dims)
        else:
            return None
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int)) for value in values):
            return None
        channel_map = field_entry.get("layout", {}).get("intensity_channels")
        if (not isinstance(channel_map, list) or not channel_map
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in channel_map)
                or len(set(channel_map)) != len(channel_map)):
            return None
        mapping = resolve_png_channel_mapping(settings)
        if any(index is not None and not 0 <= index < len(channel_map) for index in mapping.values()):
            return None
        original = field_entry.get("image") or field_entry.get("original")
        if original is None:
            return None
        dtype = np.dtype(original["dtype"]).newbyteorder("=")
        if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            return None
        grey = len(set(mapping.values())) == 1 and mapping["r"] is not None
        return {"method": "masked_centroid_raw", "png_size": list(size),
                "planes": list(range(len(channel_map))), "channel_map": channel_map,
                "mapping": mapping, "dtype": dtype.str,
                "shape": [size[1], size[0], 1 if grey else 3], "crop_format": 4,
                "mask_mode": "masked", "dilated": False,
                "centroid": "round_center_of_mass", "padding": "zero"}
    except (ValueError, TypeError, KeyError, OSError, CropError):
        return None


def _verify_legacy_png(project, row, record, folder):
    """Regenerate a proven native PNG through shared crop and PNG APIs and compare all pixels."""
    from .crop_store import legacy_png_view
    from .crops import CropError, read_crop_array, build_png_channels

    contract = record.get("legacy_png_contract")
    expected = record["png"]
    if (not contract or contract.get("method") != "masked_centroid_raw"
            or contract["shape"] != expected["shape"]
            or contract["dtype"] != expected["dtype"]):
        return False
    try:
        native = legacy_png_view(project, row, contract["png_size"], contract["planes"])
        pixels = build_png_channels(native, contract["mapping"])
        if _array_digest(pixels) != expected:
            return False
        from .measure import save_and_add_image_to_grid

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "legacy.png"
        save_and_add_image_to_grid(pixels, str(target), [], plot=False,
                                   crop_format=4, marker={"mask_mode": "masked"})
        return (target.is_file() and _array_digest(read_crop_array(target)) == expected
                and _array_digest(image_store.read_image(target)) == record["png_container_view"])
    except (ValueError, TypeError, KeyError, IndexError, OSError, CropError):
        return False
