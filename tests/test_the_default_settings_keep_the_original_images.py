"""Issue 154: the defaults must not make a run delete the user's raw images.

``keep_original_images`` defaulted to False, so a successful run moved the raw
TIFFs into orig/ and then deleted orig/ once merged/ was built, which only
holds projected pixel data. Both flags now default to True; a user who sets
False explicitly still gets the old behaviour.
"""
from __future__ import annotations

import pytest

from spacr import settings as spacr_settings
from spacr.utils import cleanup_pipeline_folders


@pytest.mark.parametrize("make", [
    spacr_settings.set_default_settings_preprocess_generate_masks,
    spacr_settings.set_default_settings_preprocess_img_data,
    spacr_settings.get_timelapse_settings,
])
def test_both_original_image_flags_default_to_true(make):
    defaults = make({})
    assert defaults["save_original_images"] is True
    assert defaults["keep_original_images"] is True


def test_an_explicit_false_is_still_respected():
    defaults = spacr_settings.set_default_settings_preprocess_generate_masks(
        {"keep_original_images": False, "save_original_images": False})
    assert defaults["keep_original_images"] is False
    assert defaults["save_original_images"] is False


def _finished_plate(tmp_path):
    for name in ("merged", "stack", "masks", "orig"):
        (tmp_path / name).mkdir()
    (tmp_path / "merged" / "f1.npy").write_bytes(b"m")
    (tmp_path / "stack" / "f1.npy").write_bytes(b"s")
    (tmp_path / "orig" / "f1_c1.tif").write_bytes(b"raw")
    return tmp_path


def test_the_default_cleanup_leaves_the_raw_images_in_orig(tmp_path):
    plate = _finished_plate(tmp_path)
    defaults = spacr_settings.set_default_settings_preprocess_generate_masks({})
    cleanup_pipeline_folders(
        str(plate), keep_intermediate=defaults["keep_intermediate"],
        keep_original=defaults["keep_original_images"], verbose=False)
    assert (plate / "orig" / "f1_c1.tif").read_bytes() == b"raw"
    assert (plate / "merged" / "f1.npy").exists()


def test_opting_out_deletes_orig_and_says_so(tmp_path, capsys):
    plate = _finished_plate(tmp_path)
    cleanup_pipeline_folders(str(plate), keep_original=False)
    assert not (plate / "orig").exists()
    assert "1 raw image(s)" in capsys.readouterr().out
