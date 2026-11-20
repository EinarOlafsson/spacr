"""OME crop defaults retain native pixels, identities and legacy source choices."""
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from spacr.crop_store import CropPoolWriter, cut_crops
from spacr.crops import resolve_crop_source
from spacr.tabular import read_database, read_table, write_database, write_table


@pytest.fixture
def native_project(tmp_path):
    """Two wells with committed native crops and no PNG or merged intermediates."""
    rows = []
    for well_index, well in enumerate(("A01", "A02")):
        image = (np.arange(4 * 12 * 12).reshape(4, 12, 12)
                 + 1000 + 1500 * well_index).astype(np.uint16)
        labels = np.zeros((12, 12), np.uint16)
        labels[1:5, 1:5] = 1
        labels[7:11, 7:11] = 2
        writer = CropPoolWriter(tmp_path, "plate1", well, "cell", 4)
        writer.add(cut_crops(image, labels), field=1)
        for row in writer.commit()["rows"]:
            row["annotate"] = int(row["object_label"])
            rows.append(row)
    database = tmp_path / "measurements" / "measurements.db"
    database.parent.mkdir()
    write_database(pd.DataFrame(rows), database, "ssCell_Crops",
                   if_exists="fail", canonicalise=False)
    return tmp_path, database, rows


def test_native_source_preference_preserves_inherited_auto(native_project):
    """Explicit OME and inherited auto select pools while explicit PNG stays available."""
    root, _database, _rows = native_project
    folder = root / "data" / "cell_png"
    folder.mkdir(parents=True)
    assert resolve_crop_source({"src": str(root), "crop_source": "ome-zarr"}).kind == "ome-zarr"
    assert resolve_crop_source({"src": str(root), "crop_source": "auto"}).kind == "ome-zarr"
    assert resolve_crop_source({"src": str(root), "crop_source": "png"}).kind == "png"


def test_missing_ome_crops_reports_legacy_fallback(tmp_path):
    """Old PNG projects remain readable with the new explicit default."""
    (tmp_path / "data" / "cell_png").mkdir(parents=True)
    source = resolve_crop_source({"src": str(tmp_path), "crop_source": "ome-zarr"})
    assert source.kind == "png"
    assert "OME-Zarr crops are unavailable" in source.reason


def test_native_crop_plan_and_duplicate_order_preserve_dtype(native_project):
    """Database selection and repeated requests retain raw four-channel uint16 crops."""
    from spacr.crop_loader import CropQuery, load_page, object_classes, plan_crops, plates

    root, database, rows = native_project
    plan = plan_crops(CropQuery(path=str(database), where="object_label = 2", limit=0))
    assert plan._source.kind == "ome-zarr"
    assert plan.matched == plan.selected == plan.count == 2
    assert [row["object_id"] for row in plan.rows] == [rows[1]["object_id"], rows[3]["object_id"]]
    assert object_classes(str(database)) == ("cell",)
    assert plates(str(database)) == ("plate1",)
    requested = (plan.rows[1], plan.rows[0], plan.rows[1])
    repeated = replace(plan, rows=requested, selected=3)
    pixels, conformed = load_page(repeated, 0, 3)
    assert pixels.dtype == np.uint16
    assert pixels.shape == (3, 4, 4, 4)
    assert conformed == 0
    for actual, row in zip(pixels, requested):
        expected, _mask = plan._source.get_raw(row)
        np.testing.assert_array_equal(actual, np.moveaxis(expected, 0, -1))
    assert not (root / "data").exists()


@pytest.mark.parametrize("native_state", ["committed", "missing_store", "empty_table"])
def test_crop_choices_include_legacy_metadata_beside_native_records(native_project, native_state):
    """Native metadata cannot hide PNG choices or change explicit legacy pixels."""
    import sqlite3
    from PIL import Image
    from spacr.crop_loader import CropQuery, load_page, object_classes, plan_crops, plates

    root, database, _rows = native_project
    folder = root / "data" / "nucleus_png"
    folder.mkdir(parents=True)
    path = folder / "plate9_A01_1_7.png"
    pixels = np.full((5, 6, 3), 37, dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    legacy = pd.DataFrame([{
        "png_path": str(path), "plateID": "plate9", "rowID": "r1",
        "columnID": "c1", "fieldID": "f1", "nucleus_id": "o7", "cell_id": None,
    }])
    write_database(legacy, database, "png_list", if_exists="fail", canonicalise=False)
    write_database(legacy, database, "legacy_only", if_exists="fail", canonicalise=False)
    if native_state == "missing_store":
        shutil.rmtree(root / "crops")
    elif native_state == "empty_table":
        with sqlite3.connect(database) as connection:
            connection.execute('DELETE FROM "ssCell_Crops"')
    expected_classes = ("nucleus",) if native_state == "empty_table" else ("cell", "nucleus")
    expected_plates = ("plate9",) if native_state == "empty_table" else ("plate1", "plate9")
    assert object_classes(str(database)) == expected_classes
    assert plates(str(database)) == expected_plates
    assert object_classes(str(database), table="legacy_only") == ("nucleus",)
    assert plates(str(database), table="legacy_only") == ("plate9",)
    preferences = ("png", "ome-zarr") if native_state == "missing_store" else ("png",)
    for prefer in preferences:
        plan = plan_crops(CropQuery(path=str(database), object_type="nucleus",
                                   plate="plate9", prefer=prefer, limit=0))
        assert plan._source.kind == "png"
        assert plan.matched == plan.selected == plan.count == 1
        loaded, conformed = load_page(plan, 0, 1)
        np.testing.assert_array_equal(loaded[0], pixels)
        assert conformed == 0


def test_native_source_overrides_use_shared_raw_reader(native_project):
    """Explicit size/context apply through the shared reader and do not alter its dtype."""
    root, _database, rows = native_project
    source = resolve_crop_source({"src": str(root), "crop_source": "ome-zarr",
                                  "channels": [3, 1], "crop_size": [7, 9],
                                  "context": False})
    pixels, mask = source.get_raw(rows[0])
    assert pixels.shape == (4, 7, 9)
    assert pixels.dtype == np.uint16
    assert mask.shape == (7, 9)
    assert source.channels == [3, 1]
    assert source.context is False


def test_native_lazy_export_retains_real_object_identity(native_project):
    """The explicit RGB export adapter carries native identities without fake PNG paths."""
    from spacr.io import _ome_crop_rows, crop_refs_for_rows, open_crop_source

    root, database, _rows = native_project
    source = open_crop_source({"src": str(root), "crop_source": "ome-zarr"}, verbose=False)
    frame = _ome_crop_rows(str(database), "cell")
    selected = frame.iloc[[3, 0, 3]]
    refs = crop_refs_for_rows(source, selected)
    assert [item.row["object_id"] for item in refs] == selected["object_id"].tolist()
    assert all(item.row.get("png_path") is None for item in refs)
    assert [item.name for item in refs] == ["plate1_A02_1_2.png", "plate1_A01_1_1.png", "plate1_A02_1_2.png"]
    for item in refs:
        np.testing.assert_array_equal(item.array(), source.get(item.row))
    assert read_database(database, "ssCell_Crops", canonicalise=False,
                         migrate=False, read_only=True, report=None)[0]["annotate"].tolist() == [1, 2, 1, 2]


@pytest.fixture
def training_project(tmp_path):
    """Four real native crop wells supply both labels without any exported images."""
    root = tmp_path / "project"
    rows = []
    for well_index, well in enumerate(("A01", "A02", "A03", "A04")):
        image = (np.arange(4 * 12 * 12).reshape(4, 12, 12)
                 + 1000 + 2000 * well_index).astype(np.uint16)
        labels = np.zeros((12, 12), np.uint16)
        labels[1:5, 1:5] = 1
        labels[7:11, 7:11] = 2
        writer = CropPoolWriter(root, "plate1", well, "cell", 4)
        writer.add(cut_crops(image, labels), field=1)
        for row in writer.commit()["rows"]:
            row["annotate"] = int(row["object_label"])
            rows.append(row)
    database = root / "measurements" / "measurements.db"
    database.parent.mkdir()
    write_database(pd.DataFrame(rows), database, "ssCell_Crops",
                   if_exists="fail", canonicalise=False)
    return root, database, rows


@pytest.fixture
def native_training_dataset(training_project):
    """Generate the actual grouped training dataset with one shared tabular index."""
    from spacr.io import generate_training_dataset

    root, database, rows = training_project
    settings = dict(src=str(root), crop_source="ome-zarr", dataset_mode="annotation",
                    annotation_column="annotate", annotation_columns=["annotate"],
                    test_split=.25, cv_group_by="well", random_seed=42,
                    balance_to_smallest=False, path_string="cell_png")
    train, test = generate_training_dataset(settings)
    dataset = Path(train).parent
    assert Path(test).parent == dataset
    return root, database, rows, dataset


def test_real_generation_loaders_and_cv_use_one_native_index(native_training_dataset):
    """Empty class folders are discovered by the real loader, CV and leakage paths."""
    from spacr.classification_pixels import _native_crop_reference
    from spacr.classifier_evaluation import audit_dataset_splits
    from spacr.deep_spacr import dataset_class_balance
    from spacr.io import dataset_filenames, generate_cv_loaders, generate_loaders
    from spacr.crop_source import validate
    from spacr.crops import _stretched_view

    root, _database, rows, dataset = native_training_dataset
    assert validate({"crop_source": "ome-zarr"}) == "ome-zarr"
    classes = ["annotate_1", "annotate_2"]
    frame = read_table(dataset / "crop_index.csv", canonicalise=False, report=None)
    assert len(frame) == len(rows) == 8
    assert frame["object_id"].tolist() == list(dict.fromkeys(frame["object_id"]))
    assert set(frame["object_id"]) == {row["object_id"] for row in rows}
    assert all(not Path(value).is_absolute() for value in frame["source"])
    assert not list(dataset.rglob("*.png"))
    assert not list(dataset.rglob("*.omecrop"))
    assert not (root / "data").exists()
    assert all(not list((dataset / split / name).iterdir())
               for split in ("train", "test") for name in classes)
    assert set(frame.loc[frame["split"] == "train", "well"]).isdisjoint(
        frame.loc[frame["split"] == "test", "well"])
    assert audit_dataset_splits(dataset, hash_content=True).passed
    assert dataset_class_balance(dataset, classes) == {
        "train": {name: 3 for name in classes}, "test": {name: 1 for name in classes}}
    loader, _, _ = generate_loaders(str(dataset), classes=classes, image_size=4,
                                     batch_size=2, n_jobs=0, channels=["r", "b"])
    batch, labels, names = next(iter(loader))
    assert tuple(batch.shape) == (2, 3, 4, 4)
    assert set(labels.tolist()) <= {0, 1}
    assert all(_native_crop_reference(name).raw().dtype == np.uint16 for name in names)
    for actual, name in zip(batch.numpy(), names):
        expected = _stretched_view(_native_crop_reference(name).raw())[..., :3]
        expected = np.moveaxis(expected, -1, 0).astype(np.float32) / 255
        expected[1] = 0
        np.testing.assert_array_equal(actual, expected)
    folds, _info = generate_cv_loaders(str(dataset), 3, classes=classes,
                                      image_size=4, batch_size=2, n_jobs=0,
                                      channels=["r", "b"], group_by="well")
    validation = []
    for train, val in folds:
        train_names, val_names = dataset_filenames(train.dataset), dataset_filenames(val.dataset)
        assert {name.row["well"] for name in train_names}.isdisjoint(
            name.row["well"] for name in val_names)
        validation.extend(map(str, val_names))
        val_batch, _labels, val_sample_names = next(iter(val))
        assert tuple(val_batch.shape[1:]) == (3, 4, 4)
        for actual, name in zip(val_batch.numpy(), val_sample_names):
            expected = _stretched_view(_native_crop_reference(name).raw())[..., :3]
            expected = np.moveaxis(expected, -1, 0).astype(np.float32) / 255
            expected[1] = 0
            np.testing.assert_array_equal(actual, expected)
    assert sorted(validation) == sorted(map(str, dataset_filenames(loader.dataset)))


def test_fixed_scale_loader_receives_native_values_before_model_transform(native_training_dataset):
    """The normal fixed-scale loader receives all native channels before model selection."""
    from spacr.classification_pixels import _native_crop_reference
    from spacr.io import generate_loaders

    _root, _database, _rows, dataset = native_training_dataset
    loader, _, _ = generate_loaders(str(dataset), classes=["annotate_1", "annotate_2"],
                                     image_size=4, batch_size=2, n_jobs=0,
                                     crop_scaling="fixed", channels=["r", "b"])
    pixels, _labels, names = next(iter(loader))
    assert tuple(pixels.shape) == (2, 4, 4, 4)
    assert hasattr(loader, "input_transform")
    for actual, name in zip(pixels.numpy(), names):
        reference = _native_crop_reference(name)
        np.testing.assert_array_equal(actual, np.moveaxis(reference.raw(), -1, 0).astype(np.float32))
        assert actual.max() > 255


def test_native_index_preserves_duplicates_and_relocates_with_project(native_training_dataset, tmp_path):
    """One relative index preserves duplicate order and follows a moved complete project."""
    from spacr.io import spacrDataset

    root, _database, _rows, dataset = native_training_dataset
    index = dataset / "crop_index.csv"
    frame = read_table(index, canonicalise=False, report=None)
    selected = frame.loc[frame["split"] == "train"].iloc[[1, 0, 1]].copy()
    write_table(selected, index, canonicalise=False, index=False)
    original = spacrDataset(str(dataset / "train"), ["annotate_1", "annotate_2"], shuffle=False)
    expected = [reference.raw() for reference in original.filenames]
    assert [reference.row["object_id"] for reference in original.filenames] == selected["object_id"].tolist()
    relative = dataset.relative_to(root)
    moved = tmp_path / "moved"
    shutil.move(str(root), moved)
    restored = spacrDataset(str(moved / relative / "train"), ["annotate_1", "annotate_2"], shuffle=False)
    for reference, pixels in zip(restored.filenames, expected):
        np.testing.assert_array_equal(reference.raw(), pixels)
    missing = Path(restored.filenames[0].root) / restored.filenames[0].row["crop_store"]
    missing.rename(missing.with_name(missing.name + ".moved"))
    with pytest.raises(FileNotFoundError, match="indexed OME-Zarr crop store is missing"):
        restored[0]


def test_native_content_leakage_uses_pixels_not_locator_text(native_training_dataset):
    """Renaming acquisition identity cannot hide identical native pixel content."""
    from spacr.classification_pixels import _NativeCropReference, _indexed_crop_samples
    from spacr.classifier_evaluation import audit_split_leakage

    _root, _database, _rows, dataset = native_training_dataset
    reference = _indexed_crop_samples(dataset / "train")[0][0]
    other = _NativeCropReference("plate9_Z99_7_99.png", reference.root, reference.row)
    report = audit_split_leakage([reference], [other], group_by="well",
                                 hash_content=True, require_identity=True)
    assert report.overlap_counts["content_sha256"] == 1
    assert report.overlap_counts["well"] == 0
    assert not report.passed


@pytest.mark.parametrize("absent_flag,present_flag", [(False, True), (0, 1)],
                         ids=["boolean", "integer"])
def test_mixed_index_restores_native_mask_flags_without_changing_legacy_rows(
        tmp_path, absent_flag, present_flag):
    """Blank legacy cells cannot turn native absent-mask flags into truthy CSV strings."""
    from PIL import Image
    from spacr.classification_pixels import _native_crop_reference, _read_crop_index
    from spacr.crop_store import OmeZarrCropSource
    from spacr.io import spacrDataset

    project = tmp_path / "project"
    image = (np.arange(3 * 6 * 6).reshape(3, 6, 6) + 1000).astype(np.uint16)
    labels = np.zeros((6, 6), np.uint8)
    labels[1:5, 1:5] = 1
    crop = next(iter(cut_crops(image, labels, margin=1)))
    crop.update(mask=np.zeros_like(crop["mask"], dtype=np.uint8), has_mask=False)
    writer = CropPoolWriter(project, "plate1", "A01", "cell", 3, mask_dtype=np.uint8)
    writer.add([crop], field=1)
    row = writer.commit()["rows"][0]
    dataset = tmp_path / "dataset"
    (dataset / "train" / "cells").mkdir(parents=True)
    legacy = np.full((6, 6, 3), 37, np.uint8)
    Image.fromarray(legacy).save(dataset / "legacy.png")
    native = dict(row, source="../project", source_kind="ome-zarr", split="train",
                  crop_name="plate1_A01_1_1.png", **{"class": "cells"})
    png = dict(source="../project", source_kind="png", split="train",
               crop_name="legacy.png", legacy_path="legacy.png", has_mask="",
               **{"class": "cells"})
    index = dataset / "crop_index.csv"
    write_table(pd.DataFrame([dict(native, has_mask=absent_flag), png,
                              dict(native, has_mask=present_flag),
                              dict(native, has_mask=absent_flag)]),
                index, canonicalise=False, index=False)
    restored = _read_crop_index(index)
    assert restored.loc[1, "has_mask"] == ""
    assert restored.loc[0, "has_mask"] is False
    assert restored.loc[2, "has_mask"] is True
    for column in ("offset", "height", "width", "mask_offset", "crop_index"):
        assert restored.loc[1, column] == ""
        assert restored.loc[0, column] == row[column]
        assert isinstance(restored.loc[0, column], (int, np.integer))
    samples = spacrDataset(str(dataset / "train"), ["cells"], shuffle=False)
    assert len(samples) == 4
    assert [reference.name if hasattr(reference, "name") else Path(reference).name
            for reference in samples.filenames] == [native["crop_name"], "legacy.png",
                                                    native["crop_name"], native["crop_name"]]
    source = OmeZarrCropSource(str(project), context=False)
    for position in (0, 2, 3):
        reference = _native_crop_reference(str(samples.filenames[position]))
        pixels, mask = source.get_raw(reference.row)
        expected = np.zeros_like(crop["pixels"]) if position == 2 else crop["pixels"]
        assert pixels.dtype == np.uint16
        np.testing.assert_array_equal(pixels, expected)
        assert not mask.any()
        np.testing.assert_array_equal(reference.raw(), np.moveaxis(crop["pixels"], 0, -1))
    np.testing.assert_array_equal(np.asarray(samples[1][0]), legacy)


def test_native_index_integer_metadata_is_exact_and_rejects_invalid_values(tmp_path):
    """Large locator integers stay exact and corrupt decimal metadata is never truncated."""
    from spacr.classification_pixels import _read_crop_index

    offset = 2 ** 53 + 1
    native = dict(source="project", source_kind="ome-zarr", split="train",
                  crop_name="plate1_A01_1_1.png", crop_store="crops/pool.zarr",
                  crop_array="cell/pool/0", offset=offset, mask_offset=offset,
                  height="6.0", width="6.0", crop_index="0.0", **{"class": "cell"})
    legacy = dict(source=".", source_kind="png", split="train",
                  crop_name="legacy.png", legacy_path="legacy.png", **{"class": "cell"})
    index = tmp_path / "crop_index.csv"
    write_table(pd.DataFrame([native, legacy], dtype=object), index,
                canonicalise=False, index=False)
    restored = _read_crop_index(index)
    assert restored.loc[0, "offset"] == offset
    assert restored.loc[0, "mask_offset"] == offset
    assert restored.loc[0, "height"] == restored.loc[0, "width"] == 6
    for bad_value in ("6.25", "NaN", "inf"):
        write_table(pd.DataFrame([dict(native, height=bad_value), legacy], dtype=object),
                    index, canonicalise=False, index=False)
        with pytest.raises(ValueError, match="invalid integer in height"):
            _read_crop_index(index)


def test_cnn_preflight_uses_selected_native_annotations(training_project):
    """A legacy PNG table cannot hide the labels selected from the native crop table."""
    from spacr.validate import _check_annotation_column_in_png_list, validate_settings

    root, database, _rows = training_project
    write_database(pd.DataFrame({"png_path": ["old.png"]}), database, "png_list",
                   if_exists="fail", canonicalise=False)
    (root / "data" / "cell_png").mkdir(parents=True)
    settings = dict(src=str(root), crop_source="ome-zarr", dataset_mode="annotation",
                    annotation_column="annotate")
    assert _check_annotation_column_in_png_list(settings, native=True) == []
    assert not [problem for problem in validate_settings(settings, "classify")
                if problem.setting == "annotation_column"]
    assert _check_annotation_column_in_png_list(settings)
    assert _check_annotation_column_in_png_list({**settings, "crop_source": "png"}, native=True)
    assert _check_annotation_column_in_png_list({**settings, "image_source": "load_images"}, native=True)
    problems = _check_annotation_column_in_png_list({**settings, "annotation_column": "missing"}, native=True)
    assert len(problems) == 1
    assert "ssCell_Crops" in problems[0].message


def test_native_random_annotation_is_written_by_object_identity(training_project):
    """The optional random-class annotation updates the native table, not invented PNG rows."""
    from spacr.io import generate_training_dataset

    root, database, _rows = training_project
    frame = read_database(database, "ssCell_Crops", canonicalise=False,
                          migrate=False, read_only=True, report=None)[0]
    frame.loc[frame["object_label"] == 2, "annotate"] = np.nan
    write_database(frame, database, "ssCell_Crops", if_exists="replace", canonicalise=False)
    generate_training_dataset(dict(src=str(root), crop_source="ome-zarr",
                                    dataset_mode="annotation", annotation_column="annotate",
                                    test_split=.25, cv_group_by="well", random_seed=42,
                                    write_random_annotation_column=True, balance_to_smallest=False))
    saved = read_database(database, "ssCell_Crops", canonicalise=False,
                          migrate=False, read_only=True, report=None)[0]
    assert saved.loc[saved["object_label"] == 2, "annotate_random"].tolist() == [1, 1, 1, 1]
    assert saved.loc[saved["object_label"] == 1, "annotate_random"].isna().all()
