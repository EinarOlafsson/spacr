"""Item 790: Measure imports label images made by another tool.

The BBBC021 swap test measured CellProfiler's label TIFFs with spaCR by
building ``merged/`` arrays by hand. These tests hold the import to that
driver's arrays: raw planes in chanID order, then the cell and nucleus
labels, the same plane layout, and spaCR's own escaped file names.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import external_masks as em
from spacr.errors import ConfigurationError
from spacr.settings import ALPHA_FEATURES, get_measure_crop_settings

PLATE = "Week1_22123"
IMAGE_REGEX = r"(?:^|_)(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w(?P<chanID>\d)"
CELL_REGEX = r"(?:^|_)(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w\d.*_Cell\.tiff$"
NUCLEUS_REGEX = (
    r"(?:^|_)(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w\d.*_Nuclei\.tiff$")
SHAPE = (48, 64)


def _labels():
    """Two cells, each holding one nucleus."""
    cell = np.zeros(SHAPE, np.uint16)
    cell[4:24, 4:30] = 1
    cell[26:46, 34:62] = 2
    nucleus = np.zeros(SHAPE, np.uint16)
    nucleus[8:18, 8:20] = 1
    nucleus[30:40, 40:52] = 2
    return cell, nucleus


def _drop(tmp_path, fields=(("B02", 1), ("B02", 2), ("D04", 1))):
    """BBBC021-named raw images and CellProfiler-named labels.

    :returns: ``(image folder, label root, {(well, site): driver array})``.
    """
    images = tmp_path / "images" / PLATE
    images.mkdir(parents=True)
    labels = tmp_path / "analysis" / PLATE / "c0" / "labels" / PLATE
    labels.mkdir(parents=True)
    rng = np.random.default_rng(790)
    expected = {}
    cell, nucleus = _labels()
    for well, site in fields:
        planes = []
        for channel in (1, 2, 4):
            plane = rng.integers(0, 4000, SHAPE).astype(np.uint16)
            tifffile.imwrite(
                images / f"Week1_150607_{well}_s{site}_w{channel}AB-{channel}.tif",
                plane)
            planes.append(plane)
        stem = f"Week1_150607_{well}_s{site}_w1AB-1"
        for suffix, array in (("Cell", cell), ("Nuclei", nucleus),
                              ("Cytoplasm", cell)):
            tifffile.imwrite(labels / f"{stem}_{suffix}.tiff", array)
        expected[(well, site)] = np.stack([*planes, cell, nucleus], axis=-1)
    return images, tmp_path / "analysis" / PLATE, expected


def _settings(tmp_path, images, labels, **extra):
    """Measure settings that import the drop into ``work/<plate>``."""
    src = tmp_path / "work" / PLATE
    src.mkdir(parents=True, exist_ok=True)
    values = dict(
        src=str(src), label_import=True, label_import_images=str(images),
        label_import_image_regex=IMAGE_REGEX,
        cell_label_folder=str(labels), nucleus_label_folder=str(labels),
        cell_label_regex=CELL_REGEX, nucleus_label_regex=NUCLEUS_REGEX)
    values.update(extra)
    return get_measure_crop_settings(values)


def test_the_import_writes_the_arrays_the_swap_driver_built_by_hand(tmp_path):
    images, labels, expected = _drop(tmp_path)
    resolved, manifest = em._import_label_images_for_measure(
        _settings(tmp_path, images, labels))
    merged = os.path.join(resolved["src"], "merged")
    for (well, site), array in expected.items():
        built = np.load(os.path.join(merged, f"Week1%5F22123_{well}_{site}.npy"))
        assert built.dtype == np.uint16
        np.testing.assert_array_equal(built, array)
    with open(os.path.join(merged, ".spacr_plane_layout.json")) as handle:
        assert json.load(handle) == {
            "intensity_channels": [0, 1, 2],
            "mask_dims": {"cell": 3, "nucleus": 4},
            "mask_plane_order": ["cell", "nucleus"], "version": 1}
    assert (resolved["cell_mask_dim"], resolved["nucleus_mask_dim"],
            resolved["pathogen_mask_dim"]) == (3, 4, None)
    assert resolved["channels"] == [0, 1, 2]
    assert manifest["channels"] == ["1", "2", "4"]
    record = os.path.join(resolved["src"], em._LABEL_IMPORT_MANIFEST)
    with open(record) as handle:
        saved = json.load(handle)
    stem = "Week1%5F22123_B02_1"
    assert saved["signature"]["fields"][stem]["labels"]["cell"][0].endswith(
        "Week1_150607_B02_s1_w1AB-1_Cell.tiff")
    assert saved["plane_layout"]["mask_dims"] == {"cell": 3, "nucleus": 4}


def test_an_unchanged_import_is_reused_and_a_changed_one_rebuilt(tmp_path):
    images, labels, _expected = _drop(tmp_path)
    settings = _settings(tmp_path, images, labels)
    resolved, _manifest = em._import_label_images_for_measure(settings)
    merged = os.path.join(resolved["src"], "merged", "Week1%5F22123_B02_1.npy")
    before = os.stat(merged).st_mtime_ns
    em._import_label_images_for_measure(settings)
    assert os.stat(merged).st_mtime_ns == before
    em._import_label_images_for_measure(dict(settings, label_import_channels=[4, 1]))
    rebuilt = np.load(merged)
    assert rebuilt.shape[-1] == 4
    first = tifffile.imread(next(images.glob("*_B02_s1_w4*.tif")))
    np.testing.assert_array_equal(rebuilt[..., 0], first)


def test_a_field_without_labels_is_skipped_and_said(tmp_path, capsys):
    images, labels, _expected = _drop(tmp_path)
    next(labels.rglob("*_D04_s1_*_Nuclei.tiff")).unlink()
    plan = em._plan_label_import(_settings(tmp_path, images, labels))
    assert [stem for stem, _i, _l in plan["fields"]] == [
        "Week1%5F22123_B02_1", "Week1%5F22123_B02_2"]
    assert plan["skipped"] == ["Week1%5F22123_D04_1: no nucleus"]
    assert any("lack a channel or a label" in text for text in plan["warnings"])


@pytest.mark.parametrize("bad, message", [
    (np.full(SHAPE, 0.5, np.float32), "not an integer label image"),
    (np.zeros(SHAPE, bool), "not an integer label image"),
    (np.zeros((SHAPE[0] + 1, SHAPE[1]), np.uint16), "does not match intensity shape"),
    (np.full(SHAPE, 70000, np.uint32), "exceeds the maximum 65535"),
])
def test_labels_that_break_the_measure_contract_are_refused(tmp_path, bad, message):
    images, labels, _expected = _drop(tmp_path, fields=(("B02", 1),))
    tifffile.imwrite(next(labels.rglob("*_Cell.tiff")), bad)
    with pytest.raises(ConfigurationError, match=message):
        em._import_label_images_for_measure(_settings(tmp_path, images, labels))


def test_npy_labels_import_like_tiffs(tmp_path):
    images, labels, expected = _drop(tmp_path, fields=(("B02", 1),))
    cell, _nucleus = _labels()
    folder = tmp_path / "cellpose"
    folder.mkdir()
    np.save(folder / "B02_s1_masks.npy", cell.astype(np.int32))
    resolved, _manifest = em._import_label_images_for_measure(_settings(
        tmp_path, images, labels, cell_label_folder=str(folder),
        cell_label_regex=r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_masks"))
    built = np.load(os.path.join(resolved["src"], "merged",
                                 "Week1%5F22123_B02_1.npy"))
    np.testing.assert_array_equal(built, expected[("B02", 1)])


def test_a_plate_folder_holding_other_arrays_is_refused(tmp_path):
    images, labels, _expected = _drop(tmp_path, fields=(("B02", 1),))
    settings = _settings(tmp_path, images, labels)
    stack = os.path.join(settings["src"], "stack")
    os.makedirs(stack)
    np.save(os.path.join(stack, "other_A01_1.npy"), np.zeros((2, 2)))
    with pytest.raises(ConfigurationError, match="did not write"):
        em._import_label_images_for_measure(settings)
    assert not os.path.isdir(os.path.join(settings["src"], "merged"))


def test_a_pattern_without_its_groups_is_refused_before_anything_is_read(tmp_path):
    images, labels, _expected = _drop(tmp_path, fields=(("B02", 1),))
    with pytest.raises(ConfigurationError, match=r"\(\?P<chanID>\.\.\.\)"):
        em._plan_label_import(_settings(
            tmp_path, images, labels,
            label_import_image_regex=r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)"))


def test_blank_patterns_are_inferred_from_plain_file_names(tmp_path):
    images = tmp_path / "raw" / "plate1"
    images.mkdir(parents=True)
    masks = tmp_path / "nuclei"
    masks.mkdir()
    cell, nucleus = _labels()
    for well in ("A01", "A02", "B01"):
        for field in (1, 2):
            for channel in (1, 2):
                tifffile.imwrite(images / f"{well}F{field:03d}C{channel}.tif",
                                 np.full(SHAPE, channel, np.uint16))
            tifffile.imwrite(masks / f"{well}F{field:03d}.tif", nucleus)
    src = tmp_path / "work"
    src.mkdir()
    plan = em._plan_label_import(get_measure_crop_settings(dict(
        src=str(src), label_import=True, label_import_images=str(images),
        nucleus_label_folder=str(masks))))
    assert not plan["errors"], plan["errors"]
    assert plan["image_pattern_inferred"]
    assert plan["label_patterns"]["nucleus"]["inferred"]
    assert len(plan["fields"]) == 6
    assert plan["fields"][0][0] == "plate1_A01_1"


def test_dry_run_reports_the_plan_and_writes_nothing(tmp_path):
    from spacr.measure import measure_crop

    images, labels, _expected = _drop(tmp_path, fields=(("B02", 1),))
    settings = _settings(tmp_path, images, labels, dry_run=True,
                         nucleus_label_folder=str(tmp_path / "missing"))
    with pytest.raises(ConfigurationError, match="is not a folder"):
        measure_crop(settings)
    problems = measure_crop(dict(settings, nucleus_label_folder=""))
    assert problems == []
    assert os.listdir(settings["src"]) == []


def test_validation_does_not_ask_for_merged_arrays_the_import_will_write(tmp_path):
    from spacr.validate import validate_settings

    images, labels, _expected = _drop(tmp_path, fields=(("B02", 1),))
    settings = _settings(tmp_path, images, labels)
    merged_problems = [problem for problem in validate_settings(settings, "measure")
                       if "merged" in problem.message and problem.is_error]
    assert merged_problems == []
    off = [problem for problem in validate_settings(
        dict(settings, label_import=False), "measure")
        if "merged" in problem.message and problem.is_error]
    assert off


def test_measure_settings_alone_measure_the_imported_labels(tmp_path):
    """The swap half A, end to end on three small fields."""
    from spacr.measure import measure_crop

    images, labels, _expected = _drop(tmp_path)
    settings = _settings(tmp_path, images, labels, save_png=False,
                         cell_min_size=0, nucleus_min_size=0,
                         cytoplasm_min_size=0, n_jobs=1, plot=False,
                         verbose=False)
    measure_crop(settings)
    database = os.path.join(settings["src"], "measurements", "measurements.db")
    with sqlite3.connect(database) as connection:
        cells = connection.execute(
            "SELECT plateID, COUNT(*) FROM cell GROUP BY plateID").fetchall()
        nuclei = connection.execute("SELECT COUNT(*) FROM nucleus").fetchone()
    assert cells == [(PLATE, 6)]
    assert nuclei == (6,)
    from spacr.resume import read_recorded_settings
    saved = read_recorded_settings(database)
    assert str(saved["cell_label_folder"]) == str(labels)
    assert saved["label_import"] in (True, "True")


def test_the_label_import_settings_are_alpha_and_described():
    from spacr.settings import categories, expected_types, tooltips

    keys = set(em._LABEL_IMPORT_SETTINGS)
    assert set(ALPHA_FEATURES[790]["settings"]) == keys
    assert set(categories["Label Image Import α"]) == keys
    assert keys <= set(get_measure_crop_settings({}))
    assert keys <= set(expected_types) and keys <= set(tooltips)
    assert not keys & set(em.default_settings())
