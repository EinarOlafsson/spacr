"""Independent tiny fixtures: no spaCR import, pipeline or real-source writes."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from external_evidence import verify_external_project


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_map(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "preserved"
    root = tmp_path / "new_project"
    for folder in (source / "original", source / "images", source / "cell_masks",
                   root / "images", root / "stack", root / "merged",
                   root / "masks" / "cell_mask_stack", root / "measurements",
                   root / "data" / "no_nucleus" / "uninfected" / "plate1_A01" / "cell_png"):
        folder.mkdir(parents=True)
    settings = {
        "channels": [0], "png_dims": [0], "normalize": False,
        "cell_min_size": 0, "cytoplasm": False, "uninfected": True,
        "merge_edge_pathogen_cells": False, "n_jobs": 1,
        "save_measurements": True, "save_png": True, "crop_mode": ["cell"],
        "layout": "flat", "z_handling": "first", "plate_naming": "index",
        "radial_dist": False, "spatial_measurements": False, "object_distances": False,
        "object_distance_maxima": False, "object_distance_intensity": False,
        "calculate_correlation": False, "homogeneity": False, "cell_max_size": None,
        "timelapse": False, "dry_run": False, "test_mode": False, "overwrite": False,
    }
    # Repeated label 1 in different fields, non-contiguous labels, unequal areas
    # and known means. The expected measurements below are hand-calculated,
    # not produced by the verifier's bincount algorithm or an app helper.
    images = [np.array([[10, 20, 0, 1], [65535, 65535, 65535, 2],
                        [3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.uint16),
              np.array([[0, 10, 20, 0], [2, 4, 0, 0],
                        [65535, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)]
    masks = [np.array([[1, 1, 0, 0], [7, 7, 7, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16),
             np.array([[1, 1, 1, 0], [2, 2, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)]
    measurements = [[(1, 2, 15.0), (7, 3, 65535.0)],
                    [(1, 3, 10.0), (2, 2, 3.0), (9, 1, 65535.0)]]
    records, maps, cells, pngs = [], [], [], []
    for field, (intensity, labels, expected) in enumerate(zip(images, masks, measurements), 1):
        original_stem = ("acquired_E01_7_1", "acquired_L02_9_1")[field - 1]
        neutral = f"fov{field:02d}"
        paths = {"source": source / "original" / f"{original_stem}.npy",
                 "image": source / "images" / f"{neutral}_C1.tif",
                 "mask": source / "cell_masks" / f"{neutral}_cell_mask.tif"}
        original = np.full((4, 4, 7), 1234, dtype=np.uint16)
        original[..., 1], original[..., 4] = intensity, labels
        np.save(paths["source"], original)
        tifffile.imwrite(paths["image"], intensity)
        tifffile.imwrite(paths["mask"], labels)
        record = {kind: str(path) for kind, path in paths.items()}
        record.update({f"{kind}_sha256": digest(path) for kind, path in paths.items()})
        record.update(neutral_stem=neutral, original_stem=original_stem, image_plane=1,
                      mask_plane=4, objects=len(expected), shape=[4, 4])
        records.append(record)
        stem = f"plate1_A01_{field}"
        target = f"plate1_A01_T0001F{field:03d}L01A01Z01C01.tif"
        maps.append({
            "target": target, "target_path": str(root / "images" / target),
            "source": str(paths["image"]), "source_relpath": paths["image"].name,
            "plate": "plate1", "well": "A01", "field": str(field), "channel": "1",
            "z": "1", "t": "1", "source_plate": "images", "source_well": "A01",
            "source_field": neutral, "source_channel": "C1", "source_z": "1", "source_t": "1",
            "plateID": "plate1", "rowID": "r1", "columnID": "c1", "fieldID": f"f{field}",
            "prc": "plate1_r1_c1", "prcf": f"plate1_r1_c1_f{field}",
            "z_handling": "first", "n_z_planes": "1", "n_timepoints": "1", "status": "converted",
            "meta_json": json.dumps({"axes": "YX", "reader": "tifffile", "series": 0}),
        })
        tifffile.imwrite(root / "images" / target, intensity)
        np.save(root / "stack" / f"{stem}.npy", intensity[..., None])
        np.save(root / "merged" / f"{stem}.npy", np.stack((intensity, labels), axis=-1))
        np.save(root / "masks" / "cell_mask_stack" / f"{stem}.npy", labels)
        for label, area, mean in expected:
            cells.append((stem, "plate1", "r1", "c1", f"f{field}", float(label), float(area), mean))
            name = f"{stem}_{label}.png"
            path = root / "data" / "no_nucleus" / "uninfected" / "plate1_A01" / "cell_png" / name
            Image.new("RGB", (8, 8), (label, field, 0)).save(path)
            pngs.append((str(path), name, "plate1", "r1", "c1", f"f{field}",
                         f"plate1_r1_c1_f{field}_o{label}", f"o{label}"))
    write_map(root / "images" / "conversion_map.csv", maps)
    (root / "merged" / ".spacr_plane_layout.json").write_text(json.dumps({
        "version": 1, "intensity_channels": [0], "mask_plane_order": ["cell"], "mask_dims": {"cell": 1}}))
    db = root / "measurements" / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE cell(file_name TEXT, plateID TEXT, rowID TEXT, columnID TEXT, "
                           "fieldID TEXT, object_label REAL, cell_area REAL, cell_channel_0_mean_intensity REAL)")
        connection.executemany("INSERT INTO cell VALUES(?,?,?,?,?,?,?,?)", cells)
        connection.execute("CREATE TABLE png_list(png_path TEXT, file_name TEXT, plateID TEXT, rowID TEXT, "
                           "columnID TEXT, fieldID TEXT, prcfo TEXT, cell_id TEXT)")
        connection.executemany("INSERT INTO png_list VALUES(?,?,?,?,?,?,?,?)", pngs)
    return {"root": root, "records": records, "settings": settings, "maps": maps, "db": db}


def verify(example):
    return verify_external_project(example["root"], example["records"], example["settings"],
                                   expected_shape=(4, 4), expected_labels=(2, 3))


def sql(example, statement, parameters=()):
    with sqlite3.connect(example["db"]) as connection:
        cursor = connection.execute(statement, parameters)
        assert cursor.rowcount > 0, "Corruption fixture must actually change a row"


def test_positive_exact_pixels_identities_math_pngs_and_read_only(example):
    files = list(example["root"].rglob("*"))
    files += [Path(row[kind]) for row in example["records"] for kind in ("source", "image", "mask")]
    before = {path: digest(path) for path in files if path.is_file()}
    evidence = verify(example)
    assert evidence["accepted"] is True
    assert evidence["field_count"] == 2
    assert evidence["cell_rows"] == evidence["png_rows"] == 5
    assert evidence["max_cell_mean_absolute_error"] == 0
    assert evidence["all_output_pixels_equal"] is True
    assert evidence["all_sources_unchanged"] is True
    assert [row[-1] for row in evidence["cell_keys"]] == [1, 7, 1, 2, 9]
    assert {path: digest(path) for path in before} == before
    json.dumps(evidence, allow_nan=False)


def test_two_argument_api_and_record_order(example):
    assert verify_external_project(example["root"], list(reversed(example["records"])),
                                   expected_shape=(4, 4), expected_labels=(2, 3))["accepted"]


def test_production_defaults_reject_tiny_fixture(example):
    with pytest.raises(ValueError, match="Recorded shape/object counts"):
        verify_external_project(example["root"], example["records"])


@pytest.mark.parametrize("column,value", [
    ("file_name", "plate1_A01_2"), ("plateID", "acquired"), ("rowID", "r5"),
    ("columnID", "c2"), ("fieldID", "f2"), ("object_label", 2),
    ("object_label", 1.5), ("object_label", 0), ("object_label", None),
    ("cell_area", 3), ("cell_area", 2.1), ("cell_area", None),
    ("cell_channel_0_mean_intensity", 10), ("cell_channel_0_mean_intensity", float("inf")),
    ("cell_channel_0_mean_intensity", float("nan")),
])
def test_reject_cell_identity_area_or_mean(example, column, value):
    sql(example, f'UPDATE cell SET "{column}"=? WHERE _rowid_=1', (value,))
    with pytest.raises(ValueError, match="identity|integral|area|intensity"):
        verify(example)


@pytest.mark.parametrize("statement", [
    "DELETE FROM cell WHERE _rowid_=1", "INSERT INTO cell SELECT * FROM cell WHERE _rowid_=1",
    "UPDATE cell SET object_label=1 WHERE file_name='plate1_A01_1'",
    "DELETE FROM png_list WHERE _rowid_=1", "INSERT INTO png_list SELECT * FROM png_list WHERE _rowid_=1",
])
def test_reject_row_loss_or_duplicates(example, statement):
    sql(example, statement)
    with pytest.raises(ValueError, match="exactly|identity"):
        verify(example)


@pytest.mark.parametrize("kind", ["converted", "stack", "merged_image", "merged_mask", "mask"])
def test_reject_one_changed_output_pixel(example, kind):
    root = example["root"]
    if kind == "converted":
        path = root / "images" / example["maps"][0]["target"]
        data = tifffile.imread(path)
        data[0, 0] += 1
        tifffile.imwrite(path, data)
    else:
        folder = {"stack": "stack", "merged_image": "merged", "merged_mask": "merged",
                  "mask": "masks/cell_mask_stack"}[kind]
        path = root / folder / "plate1_A01_1.npy"
        data = np.load(path)
        index = (0, 0) if kind == "mask" else (0, 0, 1 if kind == "merged_mask" else 0)
        data[index] += 1
        np.save(path, data)
    with pytest.raises(ValueError, match="pixel mismatch"):
        verify(example)


@pytest.mark.parametrize("kind", ["source", "image", "mask"])
def test_reject_source_hash_mismatch(example, kind):
    example["records"][0][f"{kind}_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        verify(example)


@pytest.mark.parametrize("kind", ["image", "mask"])
def test_reject_wrong_original_plane_even_with_updated_tiff_hash(example, kind):
    record = example["records"][0]
    path = Path(record[kind])
    data = tifffile.imread(path)
    data[0, 0] += 1
    tifffile.imwrite(path, data)
    record[f"{kind}_sha256"] = digest(path)
    with pytest.raises(ValueError, match="Source .* extraction: pixel mismatch"):
        verify(example)


@pytest.mark.parametrize("column,value", [
    ("source", "wrong.tif"), ("source_field", "fov02"), ("source_channel", "C2"),
    ("source_z", "2"), ("source_t", "2"), ("source_plate", "acquired"),
    ("source_well", "E01"), ("plate", "acquired"), ("well", "E01"),
    ("field", "2"), ("channel", "0"), ("z", "2"), ("t", "2"),
    ("plateID", "acquired"), ("rowID", "r5"), ("columnID", "c2"), ("fieldID", "f2"),
    ("prcf", "plate1_r1_c1_f2"), ("prc", "plate1_r5_c1"),
    ("target", "wrong.tif"), ("target_path", "/wrong.tif"),
    ("status", "planned"), ("n_z_planes", "2"), ("z_handling", "max"),
    ("meta_json", '{"axes":"CYX","reader":"tifffile","series":0}'),
])
def test_reject_map_identity_and_plane_claims(example, column, value):
    example["maps"][0][column] = value
    write_map(example["root"] / "images" / "conversion_map.csv", example["maps"])
    with pytest.raises(ValueError, match="Conversion map"):
        verify(example)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra"])
def test_reject_conversion_map_cardinality(example, mutation):
    rows = example["maps"]
    rows = [rows[0], rows[0]] if mutation == "duplicate" else rows[:1] if mutation == "missing" else rows + rows[:1]
    write_map(example["root"] / "images" / "conversion_map.csv", rows)
    with pytest.raises(ValueError, match="Conversion map"):
        verify(example)


@pytest.mark.parametrize("column,value", [
    ("plateID", "acquired"), ("rowID", "r5"), ("columnID", "c2"), ("fieldID", "f2"),
    ("prcfo", "plate1_r1_c1_f2_o1"), ("cell_id", "o7"), ("file_name", "plate1_A01_2_1.png"),
    ("png_path", "/absent.png"),
])
def test_reject_png_identity_and_link_mismatch(example, column, value):
    sql(example, f'UPDATE png_list SET "{column}"=? WHERE _rowid_=1', (value,))
    with pytest.raises(ValueError, match="PNG|Missing evidence"):
        verify(example)


def test_reject_unreadable_png(example):
    path = next((example["root"] / "data").rglob("*.png"))
    path.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="could not be verified"):
        verify(example)


@pytest.mark.parametrize("layout", [
    {"version": 1, "intensity_channels": [1], "mask_plane_order": ["cell"], "mask_dims": {"cell": 0}},
    {"version": 1, "intensity_channels": [0], "mask_plane_order": ["nucleus"], "mask_dims": {"nucleus": 1}},
])
def test_reject_wrong_layout(example, layout):
    (example["root"] / "merged" / ".spacr_plane_layout.json").write_text(json.dumps(layout))
    with pytest.raises(ValueError, match="plane layout"):
        verify(example)


@pytest.mark.parametrize("key,value", [
    ("channels", [1]), ("png_dims", [1]), ("normalize", True), ("cell_min_size", 3),
    ("cell_max_size", 5000), ("cytoplasm", True), ("uninfected", False),
    ("merge_edge_pathogen_cells", True), ("n_jobs", 2), ("crop_mode", ["nucleus"]),
    ("save_png", False), ("save_measurements", False), ("timelapse", True),
    ("z_handling", "max"), ("plate_naming", "original"), ("layout", "well"),
    ("object_distances", True), ("test_mode", True),
])
def test_reject_unsupported_settings(example, key, value):
    example["settings"][key] = value
    with pytest.raises(ValueError, match="Unsupported"):
        verify(example)


def test_reject_missing_required_setting(example):
    del example["settings"]["normalize"]
    with pytest.raises(ValueError, match="Unsupported setting normalize"):
        verify(example)


def test_reject_output_link_to_source(example):
    target = example["root"] / "images" / example["maps"][0]["target"]
    target.unlink()
    target.symlink_to(example["records"][0]["image"])
    with pytest.raises(ValueError, match="symlinks"):
        verify(example)


def test_reject_linked_database(example):
    (example["root"].parent / "another.db").hardlink_to(example["db"])
    with pytest.raises(ValueError, match="hard-linked"):
        verify(example)
