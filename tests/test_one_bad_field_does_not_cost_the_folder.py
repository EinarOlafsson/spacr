"""`overlay_masks_on_images` survives a field it cannot read.

The per-image loop had no try at all: a name shared by an image and a mask
is not a promise that both read, so a truncated TIFF ended the loop and
every overlay after it was silently never written -- with no record of
which ones had been done.
"""

import os

import numpy as np
import pytest

tiff = pytest.importorskip("tifffile")

from spacr.plot import overlay_masks_on_images  # noqa: E402


@pytest.fixture
def folder(tmp_path):
    """Three fields, each with a matching mask."""
    masks = tmp_path / "masks"
    masks.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a.tif", "b.tif", "c.tif"):
        tiff.imwrite(tmp_path / name,
                     (rng.random((64, 64)) * 4000).astype("uint16"))
        mask = np.zeros((64, 64), "uint16")
        mask[10:30, 10:30] = 1
        tiff.imwrite(masks / name, mask)
    return tmp_path


def test_every_good_field_is_written(folder):
    result = overlay_masks_on_images(str(folder), save=True)
    assert result["written"] == 3
    assert result["failed"] == []
    assert sorted(os.listdir(folder / "overlay")) == ["a.tif", "b.tif", "c.tif"]


def test_a_truncated_field_does_not_stop_the_others(folder):
    """The bug: 'b' failing used to lose 'c' as well."""
    (folder / "b.tif").write_bytes(b"not a tiff")
    result = overlay_masks_on_images(str(folder), save=True)
    assert result["written"] == 2
    assert sorted(os.listdir(folder / "overlay")) == ["a.tif", "c.tif"]


def test_the_failure_is_named_with_its_reason(folder):
    (folder / "b.tif").write_bytes(b"not a tiff")
    result = overlay_masks_on_images(str(folder), save=True)
    assert len(result["failed"]) == 1
    name, why = result["failed"][0]
    assert name == "b.tif"
    assert "TiffFileError" in why


def test_it_says_so_out_loud(folder, capsys):
    """A caller that ignores the return value still learns of the loss."""
    (folder / "b.tif").write_bytes(b"not a tiff")
    overlay_masks_on_images(str(folder), save=True)
    said = capsys.readouterr().out
    assert "2 of 3 overlaid; 1 failed" in said
    assert "b.tif" in said


def test_a_mask_of_the_wrong_rank_is_survivable(folder):
    """Not every failure is a bad read -- a mask cv2 cannot use is one too."""
    tiff.imwrite(folder / "masks" / "b.tif",
                 np.zeros((4, 8, 8, 3), "uint16"))
    result = overlay_masks_on_images(str(folder), save=True)
    assert result["written"] == 2
    assert [n for n, _ in result["failed"]] == ["b.tif"]


def test_every_field_failing_is_still_a_return_not_a_raise(folder):
    for name in ("a.tif", "b.tif", "c.tif"):
        (folder / name).write_bytes(b"not a tiff")
    result = overlay_masks_on_images(str(folder), save=True)
    assert result["written"] == 0
    assert len(result["failed"]) == 3


def test_a_long_failure_list_is_summarised(tmp_path, capsys):
    """Twelve broken fields must not print twelve lines and hide the count."""
    masks = tmp_path / "masks"
    masks.mkdir()
    for i in range(12):
        (tmp_path / f"f{i:02d}.tif").write_bytes(b"not a tiff")
        tiff.imwrite(masks / f"f{i:02d}.tif", np.zeros((8, 8), "uint16"))
    overlay_masks_on_images(str(tmp_path), save=True)
    said = capsys.readouterr().out
    assert "0 of 12 overlaid; 12 failed" in said
    assert "and 2 more" in said


def test_no_matching_names_still_returns_early(tmp_path, capsys):
    (tmp_path / "masks").mkdir()
    tiff.imwrite(tmp_path / "a.tif", np.zeros((8, 8), "uint16"))
    assert overlay_masks_on_images(str(tmp_path)) is None
    assert "No matching filenames" in capsys.readouterr().out
