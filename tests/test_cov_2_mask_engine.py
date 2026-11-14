"""What the make-masks backend refuses, and what it repairs, on the way in.

Everything here is a file or an array the editor should never have been given:
a four-channel screenshot dropped in with the plate, a mask saved as floats, a
label past what a uint32 mask can hold. The screen has no modal dialogs, so a
refusal has to be an exception carrying the path and the number -- a silent
truncation would put a mask on disk that no longer keys against the
measurements made from it.

Nothing here touches Qt: the backend is pure numpy on purpose.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr.curation import CurationLog
from spacr.qt import mask_engine as me



# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------

def test_a_folder_that_is_not_there_lists_no_images(tmp_path):
    """The file list is rebuilt on every folder change, including a bad one.

    An empty list leaves the screen showing "no images"; an exception here
    would take down the folder picker instead.
    """
    assert me.list_images(str(tmp_path / "missing")) == []
    assert me.list_images("") == []
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert me.list_images(str(tmp_path)) == ["a.tif"]


# ---------------------------------------------------------------------------
# loading an image
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# loading a mask
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# canonical labels and saving
# ---------------------------------------------------------------------------

def test_splitting_one_id_across_two_blobs_skips_the_ids_already_in_use():
    """A new piece is given the smallest id nobody else has.

    Painting a second blob with the brush writes the same value as the first,
    and one id must mean one object. The largest piece keeps the id; the rest
    need fresh ones, and reusing an id that another object already holds
    would fuse two objects in every table made from the mask.
    """
    mask = np.zeros((3, 9), dtype=np.uint16)
    mask[1, 0:3] = 1        # object 1, the larger piece
    mask[1, 5] = 1          # object 1 again, detached
    mask[1, 7] = 2          # object 2 is in the way of the obvious new id

    out = me.canonical_labels(mask)

    assert out.dtype == np.uint16
    assert set(np.unique(out[out > 0])) == {1, 2, 3}
    assert out[1, 0] == 1 and out[1, 2] == 1, "the larger piece keeps the id"
    assert out[1, 7] == 2, "the untouched object keeps its own id"
    assert out[1, 5] == 3, "the new piece took the first free id, not 1 or 2"


def test_saving_stamps_the_ledger_with_the_file_it_was_written_to(tmp_path):
    """A ledger with no artefact recorded is a ledger about nothing.

    ``save_mask`` is where the mask first gets a path, so it is where an
    in-memory log learns what it was a log of. A log that already names a
    file keeps that name -- the mask may have been renamed since.
    """
    (tmp_path / "masks").mkdir()
    tifffile.imwrite(tmp_path / "f.tif", np.zeros((4, 4), dtype=np.uint16))
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1, 1] = 1

    log = CurationLog(source=me.CURATION_SOURCE)
    log.append("paint", 1, n_changed=1)
    assert log.artifact == ""

    written = me.save_mask(str(tmp_path), "f.tif", mask, log=log)

    assert log.artifact == written
    from spacr.curation import is_curated
    assert is_curated(written)

    # An artefact already recorded is not overwritten by a second save.
    other = CurationLog(artifact="/somewhere/else.tif",
                        source=me.CURATION_SOURCE)
    other.append("paint", 1, n_changed=1)
    me.save_mask(str(tmp_path), "f.tif", mask, log=other)
    assert other.artifact == "/somewhere/else.tif"


# ---------------------------------------------------------------------------
# the edit helpers
# ---------------------------------------------------------------------------

def test_normalising_an_empty_image_returns_it_untouched():
    """Percentiles of nothing are undefined, so there is nothing to stretch.

    The display normaliser runs on whatever is loaded, and a zero-sized array
    reaches it from a truncated file.
    """
    empty = np.zeros((0, 0), dtype=np.uint16)
    assert me.normalize_uint16(empty) is empty


def test_a_brush_smaller_than_one_pixel_still_paints_one_pixel():
    """A zero-radius brush that painted nothing would read as a dead tool.

    The radius comes from a slider that can reach its own minimum, and a
    stroke that leaves no mark is indistinguishable from a broken canvas.
    """
    mask = np.zeros((5, 5), dtype=np.uint8)
    me.paint_disk(mask, 2, 2, 0, value=9)
    assert mask.any()
    assert mask[2, 2] == 9

    clamped = np.zeros((5, 5), dtype=np.uint8)
    me.paint_disk(clamped, 2, 2, 1, value=9)
    assert (clamped == mask).all(), "radius 0 is treated as radius 1"

    bigger = np.zeros((5, 5), dtype=np.uint8)
    me.paint_disk(bigger, 2, 2, 2, value=9)
    assert int(bigger.sum()) > int(mask.sum())


def test_removing_small_objects_from_an_empty_mask_changes_nothing():
    """A mask with no objects has none to drop, and must come back a copy.

    Returning the same array would make the undo snapshot alias the live
    mask, so the next brush stroke would edit the history as well.
    """
    mask = np.zeros((4, 4), dtype=np.uint16)

    out = me.remove_small_objects(mask, 10)

    assert out is not mask
    assert not out.any()
    out[0, 0] = 5
    assert mask[0, 0] == 0, "the copy is independent of the original"


def test_erasing_outside_the_field_or_on_background_removes_nothing():
    """A right-button sweep runs off the edge and over gaps constantly.

    Both must be no-ops that report 0, because the id returned is what the
    ledger records: reporting a deletion that did not happen would put a
    fictional object in the curation record.
    """
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1, 1] = 7

    assert me.erase_object_in_place(mask, -1, 1) == 0
    assert me.erase_object_in_place(mask, 1, 99) == 0
    assert me.erase_object_in_place(mask, 3, 3) == 0, "background is not an object"
    assert mask[1, 1] == 7

    assert me.erase_object_in_place(mask, 1, 1) == 7
    assert not mask.any()


def test_the_object_filter_measures_a_colour_image_on_its_mean_plane():
    """A colour image reaches the filter whenever the source was RGB.

    Intensity bounds are per-object means, and indexing a three-plane array
    with a 2-D region mask raises. Averaging the planes is what makes the
    bound mean the same thing it does for a grey field.
    """
    mask = np.zeros((6, 6), dtype=np.uint16)
    mask[1:3, 1:3] = 1       # a dim object
    mask[4:6, 4:6] = 2       # a bright one
    image = np.zeros((6, 6, 3), dtype=np.float32)
    image[1:3, 1:3, :] = 10.0
    image[4:6, 4:6, :] = 200.0

    out, dropped = me.filter_objects(mask, image, min_intensity=100.0)

    assert dropped == [1]
    assert not (out == 1).any()
    assert (out == 2).any()


def test_a_combined_mask_past_uint16_widens_without_aliasing_other_labels():
    """Replacement retains high IDs, low IDs and background pixel for pixel."""
    old = np.zeros((3, 3), dtype=np.uint16)
    old[0, 0] = 123
    new = np.zeros((3, 3), dtype=np.int64)
    new[1, 1] = 70000
    new[2, 2] = 4464
    old_before, new_before = old.copy(), new.copy()

    out = me.combine_masks(old, new, mode="replace")

    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, new_before)
    np.testing.assert_array_equal(old, old_before)
    np.testing.assert_array_equal(new, new_before)


def test_a_combined_mask_past_uint32_is_refused_with_the_label_that_broke_it():
    """The true capacity refusal keeps both input masks intact."""
    old = np.zeros((3, 3), dtype=np.uint16)
    old[0, 0] = 123
    huge = np.zeros((3, 3), dtype=np.int64)
    huge[1, 1] = 2**32
    old_before, huge_before = old.copy(), huge.copy()

    with pytest.raises(ValueError) as excinfo:
        me.combine_masks(old, huge, mode="replace")

    message = str(excinfo.value)
    assert "4294967296" in message
    assert "uint32" in message
    assert "minimum area" in message
    np.testing.assert_array_equal(old, old_before)
    np.testing.assert_array_equal(huge, huge_before)
