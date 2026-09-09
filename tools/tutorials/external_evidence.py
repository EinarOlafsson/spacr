"""Read-only, independent evidence for the bounded External Masks tutorial.

No spaCR calculation, reader, naming or merge helper is imported.  This checks
the two neutral TIFF/mask pairs against their recorded original array planes,
then checks every output pixel and every measured cell.  Demonstration IDs are
deliberately NOT interpreted as the original acquired plate/well metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import tifffile


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path, root=None):
    path = Path(path)
    _require(path.is_absolute(), f"Expected an absolute file path: {path}")
    _require(path.is_file(), f"Missing evidence file: {path}")
    _require(not any(part.is_symlink() for part in (path, *path.parents)),
             f"Evidence paths must not use symlinks: {path}")
    resolved = path.resolve()
    if root is not None:
        _require(resolved.is_relative_to(root), f"Output escapes private destination: {path}")
        _require(path.stat().st_nlink == 1, f"Output must not be hard-linked: {path}")
    return resolved


def _integer(value, context):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value == int(value) and value > 0,
             f"{context}: expected a positive integral value, got {value!r}")
    return int(value)


def _same_pixels(actual, expected, context):
    _require(actual.dtype == expected.dtype and actual.shape == expected.shape,
             f"{context}: shape/dtype mismatch")
    _require(np.array_equal(actual, expected), f"{context}: pixel mismatch")


def _settings(settings):
    if settings is None:
        return
    required = {
        "channels": [0], "png_dims": [0], "normalize": False,
        "cell_min_size": 0, "cytoplasm": False, "uninfected": True,
        "merge_edge_pathogen_cells": False, "n_jobs": 1,
        "save_measurements": True, "save_png": True, "crop_mode": ["cell"],
        "layout": "flat", "z_handling": "first", "plate_naming": "index",
        "radial_dist": False, "spatial_measurements": False,
        "object_distances": False, "object_distance_maxima": False,
        "object_distance_intensity": False, "calculate_correlation": False,
        "homogeneity": False,
    }
    for key, expected in required.items():
        _require(key in settings and settings[key] == expected,
                 f"Unsupported setting {key}: expected {expected!r}, got {settings.get(key)!r}")
    for key in ("cell_max_size",):
        _require(settings.get(key) is None, f"Unsupported restrictive setting {key}")
    for key in ("timelapse", "dry_run", "test_mode", "overwrite"):
        _require(settings.get(key, False) is False, f"Unsupported setting {key}")


def _rows(connection, table, columns, limit):
    available = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    _require(set(columns) <= available,
             f"{table}: missing columns {sorted(set(columns) - available)}")
    names = ", ".join(f'"{name}"' for name in columns)
    return [dict(row) for row in connection.execute(
        f'SELECT {names} FROM "{table}" LIMIT ?', (limit + 1,))]


def verify_external_project(destination, records, settings=None, *,
                            expected_shape=(1994, 1994), expected_labels=(44, 59)):
    """Return JSON-safe accepted evidence, or raise ValueError on a mismatch.

    ``records`` is the unchanged ``foreign_release_v2/inputs.json['records']``
    list. Default cardinalities pin the real two-field/103-cell example. The
    keyword overrides exist only for deliberately small independent fixtures.
    All files are read-only; the SQLite connection uses ``mode=ro`` and
    ``query_only``. PNGs are checked for readability and exact object linkage,
    NOT claimed to preserve raw intensities (display crops may be rescaled).
    """
    try:
        return _verify(destination, records, settings, expected_shape, expected_labels)
    except ValueError:
        raise
    except (OSError, KeyError, TypeError, sqlite3.Error, csv.Error) as error:
        raise ValueError(f"External Masks evidence could not be verified: {error}") from error


def _verify(destination, records, settings, expected_shape, expected_labels):
    _settings(settings)
    root = Path(destination).absolute()
    _require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
             "Private destination must be an existing non-symlink directory")
    root = root.resolve()
    shape = tuple(expected_shape)
    _require(len(shape) == 2 and all(isinstance(n, int) and n > 0 for n in shape),
             "Expected shape must be two positive dimensions")
    _require(len(expected_labels) == 2 and len(records) == 2,
             "This evidence contract requires exactly two recorded fields")
    counts = tuple(_integer(n, "Expected object count") for n in expected_labels)
    records = sorted(records, key=lambda row: row["neutral_stem"])
    _require([r["neutral_stem"] for r in records] == ["fov01", "fov02"],
             "Recorded neutral field identities must be fov01 and fov02")
    source_paths, sources = set(), []
    for index, record in enumerate(records):
        _require(record["image_plane"] == 1 and record["mask_plane"] == 4,
                 "Original source planes must be image index 1 and mask index 4")
        _require(tuple(record["shape"]) == shape and record["objects"] == counts[index],
                 "Recorded shape/object counts differ from the pinned example")
        files = {}
        for kind in ("source", "image", "mask"):
            path = _file(record[kind])
            _require(not path.is_relative_to(root) and not root.is_relative_to(path.parent),
                     "Private destination must be separate from every source directory")
            _require(path not in source_paths, "Duplicate recorded source path")
            source_paths.add(path)
            _require(_digest(path) == record[f"{kind}_sha256"],
                     f"{kind}: recorded source SHA256 mismatch")
            files[kind] = path
        _require(files["source"].stem == record["original_stem"],
                 "Original source stem disagrees with its recorded path")
        _require(files["image"].name == f'{record["neutral_stem"]}_C1.tif'
                 and files["mask"].name == f'{record["neutral_stem"]}_cell_mask.tif',
                 "Source TIFF/mask names disagree with their neutral field identities")
        original = np.load(files["source"], mmap_mode="r", allow_pickle=False)
        _require(original.shape == (*shape, 7) and original.dtype == np.uint16,
                 "Original source must be the recorded seven-plane uint16 array")
        intensity = tifffile.imread(files["image"])
        labels = tifffile.imread(files["mask"])
        _same_pixels(intensity, original[..., 1], "Source intensity extraction")
        _same_pixels(labels, original[..., 4], "Source label extraction")
        label_counts = np.bincount(labels.ravel(), minlength=1)
        sums = np.bincount(labels.ravel(), weights=intensity.ravel(), minlength=1)
        label_ids = np.flatnonzero(label_counts[1:]) + 1
        _require(len(label_ids) == counts[index], "Source mask object count mismatch")
        sources.append((record, files, intensity, labels, label_ids, label_counts, sums))

    map_path = _file(root / "images" / "conversion_map.csv", root)
    with map_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None
                 and len(reader.fieldnames) == len(set(reader.fieldnames)),
                 "Conversion map has missing or duplicate column names")
        mappings = list(reader)
    _require(len(mappings) == 2, "Conversion map must contain exactly two rows")
    _require(len({row.get("source") for row in mappings}) == 2,
             "Conversion map contains duplicate source identities")
    mapping_by_source = {row["source"]: row for row in mappings}
    _require(set(mapping_by_source) == {str(item[1]["image"]) for item in sources},
             "Conversion map source identities differ from the recorded input TIFFs")
    layout = json.loads(_file(root / "merged" / ".spacr_plane_layout.json", root).read_text())
    _require(layout == {"version": 1, "intensity_channels": [0],
                        "mask_plane_order": ["cell"], "mask_dims": {"cell": 1}},
             "Merged plane layout must be intensity [0], cell mask 1")

    expected_cells, expected_pngs, fields = {}, {}, []
    for field, (record, files, intensity, labels, ids, areas, sums) in enumerate(sources, 1):
        stem = f"plate1_A01_{field}"
        target = f"plate1_A01_T0001F{field:03d}L01A01Z01C01.tif"
        expected_map = {
            "target": target, "target_path": str(root / "images" / target),
            "source": str(files["image"]), "source_relpath": files["image"].name,
            "plate": "plate1", "well": "A01", "field": str(field),
            "channel": "1", "z": "1", "t": "1",
            "source_plate": files["image"].parent.name, "source_well": "A01",
            "source_field": record["neutral_stem"], "source_channel": "C1",
            "source_z": "1", "source_t": "1", "plateID": "plate1",
            "rowID": "r1", "columnID": "c1", "fieldID": f"f{field}",
            "prc": "plate1_r1_c1", "prcf": f"plate1_r1_c1_f{field}",
            "z_handling": "first", "n_z_planes": "1", "n_timepoints": "1",
            "status": "converted",
        }
        mapping = mapping_by_source[str(files["image"])]
        for column, expected in expected_map.items():
            _require(mapping.get(column) == expected,
                     f"Conversion map {record['neutral_stem']} {column}: "
                     f"expected {expected!r}, got {mapping.get(column)!r}")
        meta = json.loads(mapping["meta_json"])
        _require(meta.get("axes") == "YX" and meta.get("reader") == "tifffile"
                 and meta.get("series") == 0,
                 "Conversion map must describe TIFF series 0 with axes YX")
        converted = tifffile.imread(_file(root / "images" / target, root))
        _same_pixels(converted, intensity, f"{stem} converted TIFF")
        stack = np.load(_file(root / "stack" / f"{stem}.npy", root), allow_pickle=False)
        _same_pixels(stack, intensity[..., None], f"{stem} stack")
        merged = np.load(_file(root / "merged" / f"{stem}.npy", root), allow_pickle=False)
        _same_pixels(merged, np.stack((intensity, labels), axis=-1), f"{stem} merged")
        mask = np.load(_file(root / "masks" / "cell_mask_stack" / f"{stem}.npy", root),
                       allow_pickle=False)
        _same_pixels(mask, labels, f"{stem} cell mask")
        for label in ids:
            label = int(label)
            key = (stem, "plate1", "r1", "c1", f"f{field}", label)
            expected_cells[key] = (int(areas[label]), float(sums[label] / areas[label]))
            prcfo = f"plate1_r1_c1_f{field}_o{label}"
            expected_pngs[prcfo] = (*key[1:5], f"o{label}", f"{stem}_{label}.png")
        fields.append({"original_stem": record["original_stem"],
                       "neutral_stem": record["neutral_stem"], "assigned_stem": stem,
                       "objects": len(ids), "shape": list(shape),
                       "source_image_plane": 1, "source_mask_plane": 4,
                       "all_output_pixels_equal": True})

    db_path = _file(root / "measurements" / "measurements.db", root)
    count = sum(counts)
    connection = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        cells = _rows(connection, "cell", ["file_name", "plateID", "rowID", "columnID",
                      "fieldID", "object_label", "cell_area", "cell_channel_0_mean_intensity"], count)
        pngs = _rows(connection, "png_list", ["png_path", "file_name", "plateID", "rowID",
                     "columnID", "fieldID", "prcfo", "cell_id"], count)
    finally:
        connection.close()
    _require(len(cells) == count, f"Expected exactly {count} measured cell rows")
    actual_keys, max_error = set(), 0.0
    for row in cells:
        key = tuple(row[name] for name in ("file_name", "plateID", "rowID", "columnID", "fieldID"))
        key += (_integer(row["object_label"], "Cell object_label"),)
        _require(key not in actual_keys, f"Duplicate measured cell identity: {key}")
        actual_keys.add(key)
        _require(key in expected_cells, f"Unexpected measured cell identity: {key}")
        area, mean = expected_cells[key]
        _require(_integer(row["cell_area"], "Cell area") == area,
                 f"Cell area mismatch: {key}")
        actual_mean = row["cell_channel_0_mean_intensity"]
        _require(isinstance(actual_mean, (int, float)) and math.isfinite(actual_mean)
                 and math.isclose(actual_mean, mean, rel_tol=1e-12, abs_tol=1e-9),
                 f"Cell mean intensity mismatch: {key}")
        max_error = max(max_error, abs(actual_mean - mean))
    _require(actual_keys == set(expected_cells), "Measured cell keyset mismatch")
    _require(len(pngs) == count, f"Expected exactly {count} PNG links")
    png_keys, png_paths = set(), set()
    for row in pngs:
        prcfo = row["prcfo"]
        _require(prcfo in expected_pngs and prcfo not in png_keys,
                 f"Unexpected or duplicate PNG object identity: {prcfo}")
        png_keys.add(prcfo)
        actual = tuple(row[name] for name in ("plateID", "rowID", "columnID", "fieldID",
                                             "cell_id", "file_name"))
        _require(actual == expected_pngs[prcfo], f"PNG full identity mismatch: {prcfo}")
        path = _file(row["png_path"], root)
        _require(path.name == row["file_name"] and path.suffix == ".png"
                 and "cell_png" in path.relative_to(root).parts,
                 f"PNG filename/crop-role mismatch: {prcfo}")
        _require(path not in png_paths, f"Repeated PNG path: {path}")
        png_paths.add(path)
        with Image.open(path) as png:
            _require(png.format == "PNG" and all(n > 0 for n in png.size),
                     f"Unreadable PNG: {path}")
            png.verify()
    _require(png_keys == set(expected_pngs), "PNG object keyset mismatch")
    for record, files, *_rest in sources:
        for kind, path in files.items():
            _require(_digest(path) == record[f"{kind}_sha256"],
                     f"{kind}: recorded source changed during verification")
    return {"accepted": True, "reason": "Exact source, pixel, full cell identity and measurement checks passed",
            "fields": fields, "field_count": 2, "cell_rows": count, "png_rows": count,
            "cell_keys": [list(key) for key in sorted(actual_keys)],
            "all_output_pixels_equal": True, "all_sources_unchanged": True,
            "max_cell_mean_absolute_error": max_error, "area_units": "pixels",
            "mean_intensity_units": "unscaled source uint16 counts",
            "layout": layout, "identities_are_demonstration_coordinates": True,
            "png_evidence": "Readable files and full per-cell links; not raw-pixel equivalence"}
