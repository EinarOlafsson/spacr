"""Label-image import from other tools: planning refusals and write checks.

The plan pairs raw channel images with label images by filename. These tests
hold each way a plan can be wrong to a named error or warning, and the write
step's refusals when files disagree or a rebuild finds its old files gone.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import external_masks as em
from spacr.errors import ConfigurationError

IMAGE_REGEX = r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_w(?P<chanID>\d)"
CELL_REGEX = r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_Cell\.tiff$"


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


def _plan(tmp_path, images, labels=None, **extra):
    src = tmp_path / "work" / "plate"
    src.mkdir(parents=True, exist_ok=True)
    values = {"src": str(src), "label_import_images": str(images),
              "label_import_image_regex": IMAGE_REGEX}
    if labels is not None:
        values.update(cell_label_folder=str(labels),
                      cell_label_regex=CELL_REGEX)
    values.update(extra)
    return em._plan_label_import(values)


def test_dot_files_and_other_extensions_are_not_importable(tmp_path):
    folder = _touch(tmp_path / "f", "a.tif", ".hidden.tif", "notes.txt")
    found = em._label_import_files(str(folder), recursive=False)
    assert [os.path.basename(path) for path in found] == ["a.tif"]


def test_a_bad_regular_expression_is_refused_with_its_setting_name(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif")
    with pytest.raises(ConfigurationError, match="not a regular expression"):
        _plan(tmp_path, images, label_import_image_regex="(unclosed")


def test_an_uninferable_pattern_names_the_setting_and_a_file(tmp_path):
    images = _touch(tmp_path / "img", "x.tif", "y.tif")
    with pytest.raises(ConfigurationError, match="could be inferred") as caught:
        _plan(tmp_path, images, label_import_image_regex="")
    assert "such as x.tif" in str(caught.value)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigurationError, match=r"from 0 file\(s\)"):
        _plan(tmp_path, empty, label_import_image_regex="")


def test_src_must_name_one_plate_folder(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif")
    with pytest.raises(ConfigurationError, match="one plate folder at a time"):
        em._plan_label_import({"src": ["a", "b"],
                               "label_import_images": str(images)})
    with pytest.raises(ConfigurationError, match="needs src"):
        em._plan_label_import({"src": "  "})
    single = tmp_path / "work" / "plate"
    single.mkdir(parents=True)
    plan = em._plan_label_import({
        "src": [str(single)], "label_import_images": str(images),
        "label_import_image_regex": IMAGE_REGEX})
    assert plan["src"] == str(single)


def test_a_merged_src_is_read_as_its_plate_folder(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif")
    plate = tmp_path / "work" / "plate"
    (plate / "merged").mkdir(parents=True)
    plan = em._plan_label_import({
        "src": str(plate / "merged"), "label_import_images": str(images),
        "label_import_image_regex": IMAGE_REGEX})
    assert plan["src"] == str(plate)


def test_image_problems_are_named(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif", "B02_s1_w1_copy.tif",
                    "unrelated.tif")
    labels = _touch(tmp_path / "lab", "B02_s1_Cell.tiff")
    plan = _plan(tmp_path, images, labels)
    assert any("are both channel 1" in text for text in plan["errors"])
    assert any("1 file(s)" in text and "do not match the image pattern" in text
               for text in plan["warnings"])
    none = _touch(tmp_path / "none", "unrelated.tif")
    plan = _plan(tmp_path, none, labels)
    assert any("No file in" in text and "matches label_import_image_regex" in text
               or "label_import_image_regex" in text for text in plan["errors"])


def test_unknown_channels_and_missing_label_folder_are_errors(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif")
    plan = _plan(tmp_path, images, label_import_channels=[1, 9])
    assert any("names channel(s) 9" in text for text in plan["errors"])
    assert any("no label folder is set" in text for text in plan["errors"])
    assert plan["objects"] == []


def test_label_files_that_fit_no_or_many_or_repeat_a_field_are_said(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_t1_w1.tif", "B02_s1_t2_w1.tif",
                    "C03_s1_t1_w1.tif")
    labels = _touch(tmp_path / "lab", "B02_s1_Cell.tiff", "P99_s1_Cell.tiff",
                    "C03_s1_Cell.tiff", "other_C03_s1_Cell.tiff", "junk.tiff")
    plan = _plan(
        tmp_path, images, labels,
        label_import_image_regex=(
            r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)_t(?P<timeID>\d+)"
            r"_w(?P<chanID>\d)"))
    errors = "\n".join(plan["errors"])
    assert "fits 2 image fields; add plateID or timeID to cell_label_regex" in errors
    assert "are both cell labels of one field" in errors
    assert any("1 cell label file(s) match the pattern but no image field"
               in text for text in plan["warnings"])


def test_a_label_folder_with_no_matching_file_is_an_error(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif")
    labels = _touch(tmp_path / "lab", "nothing.tiff")
    plan = _plan(tmp_path, images, labels)
    assert any("matches cell_label_regex" in text or "cell_label_regex" in text
               for text in plan["errors"])


def test_fields_each_missing_something_leave_nothing_ready(tmp_path):
    images = _touch(tmp_path / "img", "B02_s1_w1.tif", "C03_s1_w2.tif")
    labels = _touch(tmp_path / "lab", "B02_s1_Cell.tiff", "C03_s1_Cell.tiff")
    plan = _plan(tmp_path, images, labels)
    assert plan["fields"] == []
    assert len(plan["skipped"]) == 2
    assert plan["errors"] == ["No field has every channel and every label."]


def _real_drop(tmp_path, shapes):
    images = tmp_path / "img"
    images.mkdir(parents=True, exist_ok=True)
    labels = tmp_path / "lab"
    labels.mkdir(parents=True, exist_ok=True)
    for channel, shape in enumerate(shapes, start=1):
        tifffile.imwrite(images / f"B02_s1_w{channel}.tif",
                         np.ones(shape, np.uint16))
    tifffile.imwrite(labels / "B02_s1_Cell.tiff", np.ones((8, 8), np.uint16))
    src = tmp_path / "work" / "plate"
    src.mkdir(parents=True, exist_ok=True)
    return {"src": str(src), "label_import_images": str(images),
            "label_import_image_regex": IMAGE_REGEX,
            "cell_label_folder": str(labels), "cell_label_regex": CELL_REGEX}


def test_a_multi_plane_file_is_refused(tmp_path):
    settings = _real_drop(tmp_path, [(8, 8)])
    tifffile.imwrite(tmp_path / "lab" / "B02_s1_Cell.tiff",
                     np.ones((2, 8, 8), np.uint16))
    with pytest.raises(ConfigurationError, match="must hold one 2-D plane"):
        em._import_label_images_for_measure(settings)


def test_channels_of_different_sizes_are_refused(tmp_path):
    settings = _real_drop(tmp_path, [(8, 8), (9, 8)])
    with pytest.raises(ConfigurationError, match="inconsistent shapes"):
        em._import_label_images_for_measure(settings)


def test_a_plan_with_errors_writes_nothing(tmp_path):
    settings = _real_drop(tmp_path, [(8, 8)])
    settings["label_import_channels"] = [5]
    with pytest.raises(ConfigurationError, match="refused; nothing was written"):
        em._import_label_images_for_measure(settings)
    assert os.listdir(settings["src"]) == []


def test_a_merge_that_writes_nothing_is_refused(tmp_path, monkeypatch):
    from spacr import io

    settings = _real_drop(tmp_path, [(8, 8)])
    monkeypatch.setattr(io, "_load_and_concatenate_arrays",
                        lambda *args, **kwargs: None)
    with pytest.raises(ConfigurationError, match="wrote no merged array"):
        em._import_label_images_for_measure(settings)


def test_a_rebuild_tolerates_files_the_first_import_wrote_being_gone(tmp_path):
    settings = _real_drop(tmp_path, [(8, 8), (8, 8)])
    resolved, manifest = em._import_label_images_for_measure(settings)
    os.remove(os.path.join(resolved["src"], manifest["written"][0]))
    rebuilt, again = em._import_label_images_for_measure(
        dict(settings, label_import_channels=[1]))
    assert again["channels"] == ["1"]
    assert os.path.isfile(os.path.join(rebuilt["src"], "merged",
                                       "img_B02_1.npy"))


def test_requested_channels_within_the_import_are_left_alone(tmp_path):
    settings = _real_drop(tmp_path, [(8, 8), (8, 8)])
    resolved, _manifest = em._import_label_images_for_measure(
        dict(settings, channels=[0, 1]))
    assert resolved["channels"] == [0, 1]
