"""A settings file that says `img_size` keeps the size it chose.

Instruction 364 renamed `img_size` to `crop_size` on 2026-09-19. The
maintainer's decision that day was "crop_size", over the `image_size` the
audit first proposed, because `image_size` is already a live setting and
means something else:

* `image_size` -- the MODEL's input crop, default 224, read by training and
  inference;
* `crop_size` (was `img_size`) -- how many pixels each cell is DRAWN at,
  default 200, on the Annotate screen and the Cells tab.

Old files migrate `img_size` -> `crop_size` with a message. Both places an
old value lives are tested: a settings CSV, read through the factory the
picture defaults come from, and the picture settings a saved run carries.
"""
from __future__ import annotations

import csv
import logging

import pytest

from spacr.cli import load_settings_file
from spacr.picture_settings import ALL_KEYS, drop_retired, to_crop_settings
from spacr.settings import (RENAMED_SETTINGS, expected_types,
                            set_annotate_default_settings,
                            surviving_setting_name, tooltips)
from spacr.validate import RETIRED_SETTINGS, WARNING, _check_retired_keys


def _old_file(tmp_path, rows):
    """An annotate settings CSV as one saved before 2026-09-19."""
    path = tmp_path / "annotate_settings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows([("src", str(tmp_path))] + list(rows))
    return path


def test_an_old_file_keeps_its_crop_size(tmp_path):
    loaded = load_settings_file(str(_old_file(tmp_path, [("img_size", "96")])))
    settings = set_annotate_default_settings(loaded)
    assert settings["crop_size"] == 96, "the old value was replaced by 200"
    assert "img_size" not in settings


def test_the_migration_says_what_it_did(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="spacr.settings"):
        set_annotate_default_settings({"img_size": 96})
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "img_size=96" in said and "crop_size" in said, said
    assert "ignored" not in said, (
        "img_size always worked; a line saying it was ignored is false")


def test_a_file_carrying_both_keeps_the_new_name(tmp_path):
    settings = set_annotate_default_settings({"img_size": 96, "crop_size": 150})
    assert settings["crop_size"] == 150


def test_the_models_input_size_is_left_alone(tmp_path):
    """The two quantities stay distinct: neither value moves onto the other."""
    loaded = load_settings_file(str(_old_file(
        tmp_path, [("img_size", "96"), ("image_size", "224")])))
    settings = set_annotate_default_settings(loaded)
    assert settings["crop_size"] == 96
    assert settings["image_size"] == 224
    assert surviving_setting_name("image_size") == ()
    assert tooltips["crop_size"] != tooltips["image_size"]
    assert "Default 200" in tooltips["crop_size"]
    assert "Default 224" in tooltips["image_size"]


def test_the_rename_is_declared_in_both_tables():
    assert RENAMED_SETTINGS["img_size"] == "crop_size"
    assert RETIRED_SETTINGS["img_size"] == "crop_size"
    assert "crop_size" in expected_types and "img_size" not in expected_types
    assert "img_size" not in tooltips


def test_the_doctor_says_renamed_and_does_not_say_ignored():
    problems = _check_retired_keys({"img_size": 96})
    assert len(problems) == 1
    problem = problems[0]
    assert problem.severity == WARNING
    assert "'img_size' was renamed to 'crop_size'" in problem.message
    assert "ignored" not in problem.fix


def test_a_saved_runs_picture_settings_keep_their_size():
    """The Cells tab saves its picture settings inside a saved run."""
    assert "crop_size" in ALL_KEYS and "img_size" not in ALL_KEYS
    migrated, notes = drop_retired({"img_size": 96, "normalize_channels": "r"})
    assert migrated == {"crop_size": 96, "normalize_channels": "r"}
    assert notes == ["img_size: renamed to crop_size, and 96 moved across"]
    assert to_crop_settings(migrated)["png_size"] == [96, 96]


def test_a_saved_blob_carrying_both_keeps_the_new_name():
    migrated, notes = drop_retired({"img_size": 96, "crop_size": 120})
    assert migrated == {"crop_size": 120}
    assert "was not used" in notes[0]


@pytest.mark.parametrize("key", ["channels", "percentiles", "image_type"])
def test_a_current_blob_is_not_touched(key):
    blob = {key: "x", "crop_size": 64}
    assert drop_retired(blob) == (blob, [])
