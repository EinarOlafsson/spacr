"""Project conversion preserves scientific arrays and refuses unsafe reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spacr import image_store, ome_zarr_convert
from spacr.crops import MERGED_LAYOUT_SIDECAR
from spacr.tiff_io import write_tiff


def _project(root: Path):
    """Write tiny legacy fields with native intensities and separate label TIFFs."""
    merged = root / "merged"
    masks = root / "pathogen_mask"
    merged.mkdir(parents=True)
    masks.mkdir()
    layout = {
        "version": 1,
        "intensity_channels": [0, 1],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 2, "nucleus": 3},
    }
    (merged / MERGED_LAYOUT_SIDECAR).write_text(
        json.dumps(layout), encoding="utf-8")
    expected = {}
    for index, identity in enumerate([
            ("plate1", "A01", 1), ("plate1", "A01", 2),
            ("plate2", "B03", 1)]):
        plate, well, field = identity
        name = f"{plate}_{well}_{field}"
        stack = np.zeros((9, 11, 4), dtype=np.uint16)
        stack[..., 0] = np.arange(99, dtype=np.uint16).reshape(9, 11) + index
        stack[..., 1] = 60000 - stack[..., 0]
        if index != 2:
            stack[:3, :4, 2] = 7
            stack[5:8, 7:10, 2] = 31
            stack[1:3, 1:3, 3] = 4
        pathogen = np.zeros((9, 11), dtype=np.uint32)
        if index == 0:
            pathogen[1:2, 2:3] = 70001
        np.save(merged / f"{name}.npy", stack)
        write_tiff(masks / f"{name}.tif", pathogen, compression=None)
        expected[identity] = (
            np.moveaxis(stack[..., :2], -1, 0),
            {"cell": stack[..., 2], "nucleus": stack[..., 3],
             "pathogen": pathogen},
        )
    (root / "keep.txt").write_text("unrelated acquisition notes\n", encoding="utf-8")
    return expected


def _tree_contents(root: Path):
    """Record every directory and file so dry runs cannot leave hidden artifacts."""
    return {
        str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _assert_round_trip(destination: Path, expected):
    """Read every converted image and role through the shared storage readers."""
    for (plate, well, field), (image, labels) in expected.items():
        store = destination / "merged" / f"{plate}.ome.zarr"
        actual = image_store.read_image(store, key=(well, field))
        assert actual.dtype == image.dtype
        assert actual.shape == image.shape
        assert actual.tobytes() == image.tobytes()
        for role, mask in labels.items():
            actual_mask = image_store.read_labels(store, role, key=(well, field))
            assert actual_mask.dtype == mask.dtype
            assert actual_mask.shape == mask.shape
            assert actual_mask.tobytes() == mask.tobytes()


def test_dry_run_reports_the_plan_without_creating_destination_or_changing_source(tmp_path):
    """Discovery is read-only even when every selected output can be written."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    before = _tree_contents(tmp_path)

    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"), dry_run=True)

    assert report["status"] == "planned"
    assert Path(report["source"]) == source
    assert Path(report["destination"]) == destination
    assert len(report["fields"]) == 3
    assert isinstance(report["counts"], dict)
    assert isinstance(report["problems"], list)
    json.dumps(report, allow_nan=False)
    assert _tree_contents(tmp_path) == before
    assert not destination.exists()


def test_conversion_round_trips_native_arrays_and_retains_all_originals(tmp_path):
    """Labels above uint16 and empty fields remain exact in the default sibling."""
    source = tmp_path / "project"
    expected = _project(source)
    before = _tree_contents(source)

    report = ome_zarr_convert.convert_project(source, what=("merged", "masks"))

    destination = tmp_path / "project_ome_zarr"
    assert report["status"] == "verified"
    assert Path(report["destination"]) == destination
    assert len(report["fields"]) == len(expected)
    json.dumps(report, allow_nan=False)
    assert (destination / "conversion.json").is_file()
    assert isinstance(json.loads((destination / "conversion.json").read_text()), dict)
    _assert_round_trip(destination, expected)
    assert _tree_contents(source) == before


def test_an_interrupted_conversion_resumes_only_uncommitted_fields(tmp_path, monkeypatch):
    """A failed second field leaves the first field usable and immutable on resume."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    expected = _project(source)
    original = image_store.PlateWriter.write_field
    attempted = []

    def stop_on_second(writer, well, field, image, labels=None):
        """Interrupt after exactly one complete shared writer transaction."""
        attempted.append((writer.plate, well, field))
        if len(attempted) == 2:
            raise RuntimeError("synthetic interrupted second field")
        return original(writer, well, field, image, labels)

    monkeypatch.setattr(image_store.PlateWriter, "write_field", stop_on_second)
    with pytest.raises(RuntimeError, match="synthetic interrupted second field"):
        ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))
    assert len(attempted) == 2
    first = attempted[0]
    first_path = (destination / "merged" / f"{first[0]}.ome.zarr"
                  / image_store._field_key(first[1], first[2]))
    first_contents = _tree_contents(first_path)
    assert first_contents
    resumed = []

    def record_resume(writer, well, field, image, labels=None):
        """Record each field that resume passes to the real shared writer."""
        resumed.append((writer.plate, well, field))
        return original(writer, well, field, image, labels)

    monkeypatch.setattr(image_store.PlateWriter, "write_field", record_resume)
    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"))

    assert report["status"] == "verified"
    assert set(resumed) == set(expected) - {first}
    assert len(resumed) == 2
    assert _tree_contents(first_path) == first_contents
    _assert_round_trip(destination, expected)


def test_a_verified_rerun_never_rewrites_a_field(tmp_path, monkeypatch):
    """An unchanged source and verified destination reuse the completed stores."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))
    before = _tree_contents(destination)

    def refuse_write(*args, **kwargs):
        """Make any attempted field rewrite fail independently of the result."""
        raise AssertionError("a verified field was rewritten")

    monkeypatch.setattr(image_store.PlateWriter, "write_field", refuse_write)
    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"))

    assert report["status"] == "verified"
    assert _tree_contents(destination) == before


def test_committed_pixel_corruption_is_rejected_without_overwriting_it(tmp_path):
    """A commit record cannot hide changed scientific pixels on a later run."""
    import zarr

    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))
    pixels = zarr.open_array(
        str(destination / "merged" / "plate1.ome.zarr" / "A" / "1" / "1" / "0"),
        mode="r+")
    pixels[0, 0, 0] = int(pixels[0, 0, 0]) + 1
    del pixels
    before_source = _tree_contents(source)
    before_destination = _tree_contents(destination)

    with pytest.raises(ValueError, match="(?i)corrupt|verif|digest|checksum|changed"):
        ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))

    assert _tree_contents(source) == before_source
    assert _tree_contents(destination) == before_destination


def test_changed_source_is_rejected_before_any_destination_rewrite(tmp_path, monkeypatch):
    """Resume cannot silently combine fields from different source snapshots."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))
    changed_path = source / "merged" / "plate1_A01_1.npy"
    changed = image_store.read_image(changed_path)
    changed[0, 0, 0] += 1
    np.save(changed_path, changed)
    before_source = _tree_contents(source)
    before_destination = _tree_contents(destination)

    def refuse_write(*args, **kwargs):
        """Prove fingerprint refusal occurs before the shared writer runs."""
        raise AssertionError("changed source reached the destination writer")

    monkeypatch.setattr(image_store.PlateWriter, "write_field", refuse_write)
    with pytest.raises(ValueError, match="(?i)source|fingerprint|changed|snapshot"):
        ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))

    assert _tree_contents(source) == before_source
    assert _tree_contents(destination) == before_destination


def test_unverified_conversion_retains_originals_and_reports_converted(tmp_path):
    """Disabling verification cannot imply the stronger verified status."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    expected = _project(source)
    before = _tree_contents(source)

    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"), verify=False)

    assert report["status"] == "converted"
    assert _tree_contents(source) == before
    _assert_round_trip(destination, expected)


def test_deleting_originals_requires_verification_before_any_write(tmp_path):
    """An invalid destructive request leaves the complete project untouched."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    before = _tree_contents(tmp_path)

    with pytest.raises(ValueError, match="(?i)verif"):
        ome_zarr_convert.convert_project(
            source, destination, what=("merged", "masks"),
            verify=False, delete_old=True)

    assert _tree_contents(tmp_path) == before


def test_explicit_verified_deletion_removes_only_manifest_image_files(tmp_path):
    """Deletion keeps source metadata and unrelated files after exact verification."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    expected = _project(source)
    layout = source / "merged" / MERGED_LAYOUT_SIDECAR
    layout_bytes = layout.read_bytes()
    notes_bytes = (source / "keep.txt").read_bytes()

    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"), delete_old=True)

    assert report["status"] == "verified"
    _assert_round_trip(destination, expected)
    assert not list((source / "merged").glob("*.npy"))
    assert not list((source / "pathogen_mask").glob("*.tif"))
    assert layout.read_bytes() == layout_bytes
    assert (source / "keep.txt").read_bytes() == notes_bytes
    assert (source / "merged").is_dir()
    assert (source / "pathogen_mask").is_dir()


def test_backslash_paths_and_spaces_resolve_to_the_native_project(tmp_path):
    """Copied Windows separators do not create a second misspelled project root."""
    source = tmp_path / "source project"
    destination = tmp_path / "converted project"
    expected = _project(source)

    report = ome_zarr_convert.convert_project(
        str(source).replace("/", "\\"), str(destination).replace("/", "\\"),
        what=("merged", "masks"))

    assert report["status"] == "verified"
    assert Path(report["source"]) == source
    assert Path(report["destination"]) == destination
    _assert_round_trip(destination, expected)


def test_missing_layout_is_blocked_in_dry_run_and_refused_without_writes(tmp_path):
    """A merged stack without proven plane roles cannot be interpreted by guessing."""
    source = tmp_path / "project"
    destination = tmp_path / "converted"
    _project(source)
    (source / "merged" / MERGED_LAYOUT_SIDECAR).unlink()
    before = _tree_contents(tmp_path)

    report = ome_zarr_convert.convert_project(
        source, destination, what=("merged", "masks"), dry_run=True)

    assert report["status"] == "blocked"
    assert report["problems"]
    with pytest.raises(ValueError):
        ome_zarr_convert.convert_project(source, destination, what=("merged", "masks"))
    assert _tree_contents(tmp_path) == before


def _crop_project(root, *, exact_png=False):
    """Add three indexed legacy crops with nullable metadata to a tiny project."""
    import pandas as pd
    from PIL import Image
    from spacr.tabular import write_database

    _project(root)
    if exact_png:
        for path in (root / "merged").glob("*.npy"):
            np.save(path, image_store.read_image(path).astype(np.uint8))
    folder = root / "data" / "cell_png"
    folder.mkdir(parents=True)
    db = root / "measurements" / "measurements.db"
    records, objects, expected = [], [], {}
    for index, (field, label) in enumerate([(1, 7), (1, 31), (2, 7)]):
        field_path = root / "merged" / f"plate1_A01_{field}.npy"
        stack = image_store.read_image(field_path)
        inside = stack[..., 2] == label
        ys, xs = np.where(inside)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pixels = np.moveaxis(stack[y0:y1, x0:x1, :2], -1, 0)
        mask = inside[y0:y1, x0:x1]
        png = folder / f"plate1_A01_{field}_{label}.png"
        image = pixels[0] if exact_png else np.full((5, 6), index + 11, np.uint8)
        Image.fromarray(image).save(png)
        record = {
            "plateID": "plate1", "rowID": "r1", "columnID": "c1",
            "fieldID": f"f{field}", "cell_id": f"o{label}", "object_type": "cell",
            "png_path": str(png), "annotation": ["reviewed", None, ""][index],
            "class": ["positive", None, "negative"][index],
            "class_score": [0.75, None, 0.0][index],
            "infected": [None, 0, 1][index], "n_pathogens": [None, 0, 2][index],
            "nucleus_id": None, "has_nucleus": None, "n_nuclei": None,
            "custom_annotation": [17, None, 19][index],
        }
        records.append(record)
        objects.append({
            "plateID": "plate1", "rowID": "r1", "columnID": "c1",
            "fieldID": f"f{field}", "object_label": label,
            "path_name": str(field_path), "file_name": field_path.name,
            "n_organelle_1": [None, 0, 3][index],
        })
        expected[(field, label)] = (pixels, mask)
    write_database(pd.DataFrame(records, dtype=object), db, "png_list",
                   if_exists="replace", canonicalise=False, migrate=False)
    write_database(pd.DataFrame(objects, dtype=object), db, "cell",
                   if_exists="replace", canonicalise=False, migrate=False)
    write_database(pd.DataFrame({"note": ["keep", None]}, dtype=object), db, "notes",
                   if_exists="replace", canonicalise=False, migrate=False)
    return db, records, expected


def _read_crop_table(db, table="ssCell_Crops"):
    """Read crop test records with the same nullable, read-only table contract."""
    from spacr.tabular import read_database

    return read_database(db, table, canonicalise=False, repair_plate_ids=False,
                         report=None, migrate=False, read_only=True,
                         dtype_backend="numpy_nullable", coerce_float=False)[0]


def test_raw_crop_conversion_preserves_native_pixels_nulls_and_source_database(tmp_path):
    """Transformed PNGs stay intact while raw ragged crops and nullable metadata verify."""
    import pandas as pd
    from spacr.crop_store import OmeZarrCropSource

    source, destination = tmp_path / "project", tmp_path / "converted"
    source_db, records, expected = _crop_project(source)
    before = _tree_contents(source)

    report = ome_zarr_convert.convert_project(source, destination)

    assert report["status"] == "raw_verified"
    table = _read_crop_table(destination / "measurements" / "measurements.db")
    assert len(table) == len(records)
    assert table["object_id"].is_unique
    assert table["png_path"].isna().all()
    reader = OmeZarrCropSource(str(destination), context=True, size=None)
    for row in table.to_dict(orient="records"):
        pixels, mask = reader.get_raw({"object_id": row["object_id"]})
        expected_pixels, expected_mask = expected[(int(row["field"]), int(row["object_label"]))]
        assert pixels.dtype == expected_pixels.dtype
        assert pixels.tobytes() == expected_pixels.tobytes()
        assert np.array_equal(mask, expected_mask)
        by_identity, _mask = reader.get_raw({name: row[name] for name in (
            "plate", "well", "field", "object_type", "object_label")})
        assert by_identity.tobytes() == pixels.tobytes()
    by_object = table.set_index(["field", "object_label"])
    assert pd.isna(by_object.loc[(1, 7), "infected"])
    assert by_object.loc[(1, 31), "infected"] == 0
    assert by_object.loc[(2, 7), "infected"] == 1
    assert pd.isna(by_object.loc[(1, 7), "n_pathogens"])
    assert by_object.loc[(1, 31), "n_pathogens"] == 0
    assert by_object.loc[(2, 7), "annotation"] == ""
    assert pd.isna(by_object.loc[(1, 31), "annotation"])
    assert by_object.loc[(1, 7), "custom_annotation"] == 17
    assert pd.isna(by_object.loc[(1, 31), "custom_annotation"])
    assert by_object.loc[(2, 7), "n_organelle_1"] == 3
    for name in ("png_list", "cell", "notes"):
        pd.testing.assert_frame_equal(
            _read_crop_table(source_db, name),
            _read_crop_table(destination / "measurements" / "measurements.db", name))
    ledger = json.loads((destination / "conversion.json").read_text())
    assert ledger["crops"]["raw_verified"] is True
    assert ledger["crops"]["png_verified"] is False
    import zarr

    first = table.iloc[0]
    attributes = zarr.open_group(str(destination / first["crop_store"] / "cell"), mode="r").attrs["spacr_conversion"]
    assert attributes["pool_axes"] == ["c", "pixel"]
    assert attributes["physical_scale_unit"] is None
    assert attributes["channel_map"] == [0, 1]
    assert attributes["store_versions"] == ledger["plan"]["versions"]
    assert len(attributes["settings_hash"]) == 64
    assert "infected" in attributes["relationship_columns"]
    assert "n_organelle_1" in attributes["relationship_columns"]
    assert (destination / "crop_verification.json").is_file()
    assert _tree_contents(source) == before


def test_crop_only_selection_writes_no_field_store_and_retains_raw_sources(tmp_path):
    """Crop-only requests still resolve their native fields without publishing image copies."""
    source, destination = tmp_path / "project", tmp_path / "converted"
    _crop_project(source)
    before = _tree_contents(source)

    report = ome_zarr_convert.convert_project(source, destination, what=("crops",))

    assert report["status"] == "raw_verified"
    assert report["counts"]["crops"] == 3
    assert not list(destination.rglob("*.ome.zarr"))
    assert len(_read_crop_table(destination / "measurements" / "measurements.db")) == 3
    assert _tree_contents(source) == before


def test_exact_png_export_allows_only_verified_crop_files_to_be_deleted(tmp_path):
    """Exact shared PNG export authorizes deletion while source tables and notes remain."""
    from spacr.crop_store import OmeZarrCropSource

    source, destination = tmp_path / "project", tmp_path / "converted"
    source_db, records, expected = _crop_project(source, exact_png=True)
    database_bytes = source_db.read_bytes()

    report = ome_zarr_convert.convert_project(source, destination, delete_old=True)

    assert report["status"] == "verified"
    assert all(not Path(row["png_path"]).exists() for row in records)
    assert source_db.read_bytes() == database_bytes
    assert (source / "keep.txt").is_file()
    table = _read_crop_table(destination / "measurements" / "measurements.db")
    reader = OmeZarrCropSource(str(destination), context=True, size=None)
    for row in table.to_dict(orient="records"):
        pixels, mask = reader.get_raw({"object_id": row["object_id"]})
        expected_pixels, expected_mask = expected[(int(row["field"]), int(row["object_label"]))]
        assert pixels.dtype == np.uint8
        assert np.array_equal(pixels, expected_pixels)
        assert np.array_equal(mask, expected_mask)


def test_nonmatching_png_export_refuses_deletion_and_keeps_every_source_file(tmp_path):
    """Raw-crop verification cannot authorize deleting a transformed legacy PNG."""
    source, destination = tmp_path / "project", tmp_path / "converted"
    _crop_project(source)
    before = _tree_contents(source)

    with pytest.raises(ValueError, match="(?i)legacy crop|PNG|transform"):
        ome_zarr_convert.convert_project(source, destination, delete_old=True)

    assert _tree_contents(source) == before
    ledger = json.loads((destination / "conversion.json").read_text())
    assert ledger["crops"]["raw_verified"] is True
    assert ledger["crops"]["png_verified"] is False
    assert not ledger["deletions"]


def test_crop_phase_resumes_after_tree_publication_before_index_publication(tmp_path, monkeypatch):
    """A crop tree without its published index is rebuilt from the same immutable plan."""
    source, destination = tmp_path / "project", tmp_path / "converted"
    _crop_project(source)
    before = _tree_contents(source)
    replace = ome_zarr_convert.os.replace
    stopped = []

    def stop_after_crop_tree(old, new):
        """Interrupt only the completed crop-tree rename, after it has happened."""
        result = replace(old, new)
        if Path(old).name == "crops" and Path(new) == destination / "crops":
            stopped.append(str(new))
            raise RuntimeError("synthetic stop after crop tree publication")
        return result

    monkeypatch.setattr(ome_zarr_convert.os, "replace", stop_after_crop_tree)
    with pytest.raises(RuntimeError, match="synthetic stop after crop tree publication"):
        ome_zarr_convert.convert_project(source, destination)
    assert len(stopped) == 1
    ledger = json.loads((destination / "conversion.json").read_text())
    assert ledger["crops"] is None
    assert ledger["crop_phase"] == "publishing"
    monkeypatch.setattr(ome_zarr_convert.os, "replace", replace)

    report = ome_zarr_convert.convert_project(source, destination)

    assert report["status"] == "raw_verified"
    assert len(_read_crop_table(destination / "measurements" / "measurements.db")) == 3
    assert not (destination / ".conversion-crops").exists()
    assert _tree_contents(source) == before
    clean = tmp_path / "clean"
    ome_zarr_convert.convert_project(source, clean)
    import pandas as pd

    pd.testing.assert_frame_equal(
        _read_crop_table(destination / "measurements" / "measurements.db"),
        _read_crop_table(clean / "measurements" / "measurements.db"))


def test_committed_crop_pool_corruption_is_refused_without_repair(tmp_path):
    """A damaged crop shard fails validation before any source deletion or rewriting."""
    import zarr

    source, destination = tmp_path / "project", tmp_path / "converted"
    _crop_project(source)
    ome_zarr_convert.convert_project(source, destination)
    row = _read_crop_table(destination / "measurements" / "measurements.db").iloc[0]
    pool = zarr.open_array(str(destination / row["crop_store"] / row["crop_array"]), mode="r+")
    offset = int(row["offset"])
    pool[0, offset] = int(pool[0, offset]) + 1
    del pool
    before_source, before_destination = _tree_contents(source), _tree_contents(destination)

    with pytest.raises(ValueError, match="(?i)crop.*changed|corrupt|verif"):
        ome_zarr_convert.convert_project(source, destination, delete_old=True)

    assert _tree_contents(source) == before_source
    assert _tree_contents(destination) == before_destination


@pytest.mark.parametrize("per_row", [pytest.param(False, id="saved"), pytest.param(True, id="per_row")])
def test_saved_and_per_row_crop_policies_control_margin_and_outside_pixels(tmp_path, per_row):
    """Per-row policies override the recorded default without changing masks or labels."""
    import pandas as pd
    from spacr.crop_store import OmeZarrCropSource
    from spacr.tabular import write_database

    source, destination = tmp_path / "project", tmp_path / "converted"
    db, records, _expected = _crop_project(source)
    write_database(pd.DataFrame({"setting_key": ["crop_margin_px", "crop_zero_outside"],
                                 "setting_value": ["1", "True"]}), db, "settings",
                   if_exists="replace", canonicalise=False, migrate=False)
    if per_row:
        for index, record in enumerate(records):
            record["margin_px"] = 2 if index == 0 else 1
            record["zero_outside"] = 0 if index == 1 else 1
        write_database(pd.DataFrame(records, dtype=object), db, "png_list",
                       if_exists="replace", canonicalise=False, migrate=False)

    report = ome_zarr_convert.convert_project(source, destination)

    assert report["status"] == "raw_verified"
    table = _read_crop_table(destination / "measurements" / "measurements.db")
    reader = OmeZarrCropSource(str(destination), context=True, size=None)
    for row in table.to_dict(orient="records"):
        stack = image_store.read_image(source / "merged" / f"plate1_A01_{row['field']}.npy")
        y0, y1 = int(row["bbox_y0"]), int(row["bbox_y1"])
        x0, x1 = int(row["bbox_x0"]), int(row["bbox_x1"])
        expected = np.moveaxis(stack[y0:y1, x0:x1, :2], -1, 0).copy()
        expected_mask = stack[y0:y1, x0:x1, 2] == row["object_label"]
        if not per_row or row["zero_outside"]:
            expected[:, ~expected_mask] = 0
        pixels, mask = reader.get_raw({"object_id": row["object_id"]})
        assert np.array_equal(pixels, expected)
        assert np.array_equal(mask, expected_mask)
    first = table[(table["field"] == 1) & (table["object_label"] == 7)].iloc[0]
    assert first["margin_px"] == (2 if per_row else 1)
    assert first["clipped_at_edge"] == 1
    expected_box = (0, 0, 5, 6) if per_row else (0, 0, 4, 5)
    assert (first["bbox_y0"], first["bbox_x0"], first["bbox_y1"], first["bbox_x1"]) == expected_box


@pytest.mark.parametrize("problem", ["duplicate", "multi", "timed", "ambiguous", "traversal", "missing_merged"])
def test_unresolved_crop_records_block_planning_without_writes(tmp_path, problem):
    """Unproven identity, role or provenance cannot silently drop a legacy crop."""
    import pandas as pd
    from spacr.tabular import write_database

    source, destination = tmp_path / "project", tmp_path / "converted"
    db, records, _expected = _crop_project(source)
    if problem == "duplicate":
        records.append(dict(records[0]))
    elif problem == "multi":
        records[0]["cell_id"] = "omulti"
    elif problem == "timed":
        for record in records:
            record["timeID"] = "t1"
    elif problem == "ambiguous":
        for record in records:
            record.pop("object_type")
        records[0]["nucleus_id"] = "o4"
    elif problem == "traversal":
        records[0]["png_path"] = str(
            source / "data" / "cell_png" / ".." / "cell_png" / Path(records[0]["png_path"]).name)
    else:
        (source / "merged" / "plate1_A01_1.npy").unlink()
    write_database(pd.DataFrame(records, dtype=object), db, "png_list",
                   if_exists="replace", canonicalise=False, migrate=False)
    before = _tree_contents(tmp_path)

    report = ome_zarr_convert.convert_project(source, destination, dry_run=True)

    assert report["status"] == "blocked"
    assert report["problems"]
    with pytest.raises(ValueError):
        ome_zarr_convert.convert_project(source, destination)
    assert _tree_contents(tmp_path) == before


def test_sparse_uint32_crop_labels_reach_the_shared_cutter_as_binary_masks(tmp_path, monkeypatch):
    """A maximal sparse label keeps its identity without a max-label-sized allocation."""
    import pandas as pd
    from spacr import crop_store
    from spacr.tabular import write_database

    source, destination = tmp_path / "project", tmp_path / "converted"
    db, records, _expected = _crop_project(source)
    large_label = int(np.iinfo(np.uint32).max)
    for path in (source / "merged").glob("*.npy"):
        stack = image_store.read_image(path).astype(np.uint32)
        if path.name == "plate1_A01_1.npy":
            stack[..., 2][stack[..., 2] == 7] = large_label
        np.save(path, stack)
    records[0]["cell_id"] = f"o{large_label}"
    write_database(pd.DataFrame(records, dtype=object), db, "png_list",
                   if_exists="replace", canonicalise=False, migrate=False)
    objects = _read_crop_table(db, "cell")
    objects.loc[(objects["fieldID"] == "f1") & (objects["object_label"] == 7), "object_label"] = large_label
    write_database(objects, db, "cell", if_exists="replace", canonicalise=False, migrate=False)
    original = crop_store.cut_crops
    seen = []

    def checked_cutter(image, labels, **kwargs):
        """Refuse a dangerous allocation before entering the real shared cutter."""
        assert labels.dtype == np.uint8
        assert set(np.unique(labels)) <= {0, 1}
        assert kwargs["objects"] == [1]
        seen.append(int(labels.sum()))
        return original(image, labels, **kwargs)

    monkeypatch.setattr(crop_store, "cut_crops", checked_cutter)
    report = ome_zarr_convert.convert_project(source, destination)

    assert report["status"] == "raw_verified"
    assert len(seen) == 6
    table = _read_crop_table(destination / "measurements" / "measurements.db")
    row = table[table["object_label"] == large_label].iloc[0].to_dict()
    pixels, mask = crop_store.OmeZarrCropSource(str(destination)).get_raw({"object_id": row["object_id"]})
    assert pixels.dtype == np.uint32
    assert mask.shape == (3, 4)
    assert mask.all()
    assert row["object_label"] == large_label
