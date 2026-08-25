"""What the make-masks backend refuses, and what it repairs, on the way in.

Everything here is a file or an array the editor should never have been given:
a four-channel screenshot dropped in with the plate, a mask saved as floats, a
label past what a uint16 mask can hold. The screen has no modal dialogs, so a
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


@pytest.fixture()
def folder(tmp_path):
    """An image folder with its ``masks/`` subfolder, both empty."""
    (tmp_path / "masks").mkdir()
    return tmp_path


def _write(folder, name, array, *, mask=None):
    tifffile.imwrite(folder / name, array)
    if mask is not None:
        tifffile.imwrite(folder / "masks" / name, mask)
    return str(folder), name


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

def test_colour_images_are_collapsed_to_one_grey_field(folder):
    """RGB, RGBA and a degenerate single-channel stack all load as 2-D uint16.

    The editor draws one field. A colour image that arrived as three planes
    would index as three fields, and the mask painted on it would be the
    wrong shape for the image it was painted over. The BT.601 weights are
    asserted through the result, so a channel swap shows up as a value.
    """
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255         # pure red
    rgb[1, 1, 1] = 255         # pure green
    rgb[2, 2, 2] = 255         # pure blue
    grey, mask = me.load_image_and_mask(*_write(folder, "rgb.tif", rgb))
    assert grey.shape == (4, 4) and grey.dtype == np.uint16
    assert mask.shape == (4, 4)
    assert grey[1, 1] > grey[0, 0] > grey[2, 2] > 0, \
        "green weighs most and blue least, which is what BT.601 says"
    assert grey[1, 1] == 65535, "the brightest pixel fills the uint16 range"

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[0, 0, 0] = 255        # one red pixel
    rgba[..., 3] = 255         # opaque everywhere
    from_rgba, _ = me.load_image_and_mask(*_write(folder, "rgba.tif", rgba))
    assert from_rgba.shape == (4, 4)
    assert from_rgba[0, 0] == 65535
    assert not from_rgba[1:, 1:].any(), \
        "the alpha plane was dropped, not averaged into the grey"

    stack = np.zeros((4, 4, 1), dtype=np.uint16)
    stack[1, 1, 0] = 7
    squeezed, _ = me.load_image_and_mask(*_write(folder, "one.tif", stack))
    assert squeezed.shape == (4, 4)
    assert squeezed[1, 1] == 7, "already uint16, so the values are not rescaled"


def test_an_image_with_too_many_channels_is_refused_by_name(folder):
    """Five planes is not a colour space, so there is nothing to collapse.

    Guessing -- taking the first three, say -- would show the user a field
    that is not the one they think they are editing.
    """
    with pytest.raises(ValueError) as excinfo:
        me.load_image_and_mask(*_write(folder, "five.tif",
                                       np.zeros((4, 4, 5), dtype=np.uint8)))
    assert "Unsupported channel count 5" in str(excinfo.value)
    assert "five.tif" in str(excinfo.value)


def test_a_volume_is_refused_because_the_editor_shows_one_field(folder):
    """Make Masks edits a single plane; a z-stack has no single plane.

    Silently taking slice 0 would produce a mask that is correct for one
    slice of a stack and wrong for the file it is saved beside.
    """
    with pytest.raises(ValueError) as excinfo:
        me.load_image_and_mask(*_write(folder, "vol.tif",
                                       np.zeros((3, 4, 4, 2), dtype=np.uint8)))
    assert "expects one 2-D field" in str(excinfo.value)


def test_non_finite_and_negative_intensities_are_refused(folder):
    """Both would come out of the uint16 rescale as plausible pixel values.

    A NaN cast to uint16 is 0 and a negative one wraps to a large positive,
    so an image with either would display as ordinary data with fabricated
    bright or dark regions -- and the mask would be drawn against them.
    """
    nan_image = np.zeros((3, 3), dtype=np.float32)
    nan_image[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        me.load_image_and_mask(*_write(folder, "nan.tif", nan_image))

    negative = np.zeros((3, 3), dtype=np.float32)
    negative[0, 0] = -5.0
    with pytest.raises(ValueError, match="negative intensities"):
        me.load_image_and_mask(*_write(folder, "neg.tif", negative))


def test_an_all_zero_float_image_does_not_divide_by_its_own_maximum(folder):
    """A blank field loads as a blank field rather than as NaNs.

    The rescale divides by the image maximum, which is zero here. Empty
    fields are ordinary on a plate -- an out-of-focus corner, a well that was
    not seeded -- and the editor has to open them.
    """
    grey, mask = me.load_image_and_mask(
        *_write(folder, "blank.tif", np.zeros((3, 3), dtype=np.float32)))

    assert grey.dtype == np.uint16
    assert not grey.any()
    assert not mask.any()


# ---------------------------------------------------------------------------
# loading a mask
# ---------------------------------------------------------------------------

def test_a_mask_saved_with_a_trailing_axis_is_squeezed(folder):
    """A ``(H, W, 1)`` mask is a label image written by a careless writer.

    Squeezing it keeps the file readable; refusing it would lock the user out
    of masks another tool produced, for a difference that carries no
    information.
    """
    args = _write(folder, "img.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.ones((4, 4, 1), dtype=np.uint8))
    _grey, mask = me.load_image_and_mask(*args)

    assert mask.shape == (4, 4)
    assert mask.max() == 1


def test_a_mask_that_is_not_a_label_image_is_refused(folder):
    """Three planes cannot be one label per pixel, whatever they mean.

    An RGB overlay saved into ``masks/`` by mistake is the common case, and
    reading its red plane as labels would invent objects.
    """
    args = _write(folder, "img.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError) as excinfo:
        me.load_image_and_mask(*args)
    assert "expected a 2-D label image" in str(excinfo.value)


def test_a_float_mask_must_hold_whole_non_negative_labels(folder):
    """Labels are ids, and 1.5 is not an id.

    A float mask cast straight to uint16 would floor every value, fusing
    object 1.5 into object 1. Both failure modes -- non-finite, and
    fractional or negative -- name the mask file, because the user has to go
    and look at it.
    """
    args = _write(folder, "a.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.full((4, 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="Mask contains non-finite values"):
        me.load_image_and_mask(*args)

    args = _write(folder, "b.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.full((4, 4), 1.5, dtype=np.float32))
    with pytest.raises(ValueError,
                       match="non-negative integer labels"):
        me.load_image_and_mask(*args)

    # A float mask that IS whole and non-negative loads.
    args = _write(folder, "c.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.full((4, 4), 3.0, dtype=np.float32))
    _grey, mask = me.load_image_and_mask(*args)
    assert mask.dtype == np.uint8
    assert mask.max() == 3


def test_a_mask_label_past_uint16_is_refused_rather_than_wrapped(folder):
    """Object 70000 in a uint16 mask would come back as object 4464.

    That is a silent identity swap: two different objects sharing an id.
    The message carries the offending label so it can be found.
    """
    args = _write(folder, "img.tif", np.zeros((4, 4), dtype=np.uint16),
                  mask=np.full((4, 4), 70000, dtype=np.int32))
    with pytest.raises(ValueError) as excinfo:
        me.load_image_and_mask(*args)
    assert "70000" in str(excinfo.value)
    assert "uint16 capacity" in str(excinfo.value)


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


def test_a_combined_mask_past_uint16_is_refused_with_the_label_that_broke_it():
    """A detection with more objects than uint16 can hold is not written.

    Casting anyway would wrap the highest ids into low ones and fuse them
    with existing objects. The message names the label and says what to
    change, because the fix is a detection setting, not a file.
    """
    old = np.zeros((3, 3), dtype=np.uint16)
    huge = np.zeros((3, 3), dtype=np.int64)
    huge[1, 1] = 70000

    with pytest.raises(ValueError) as excinfo:
        me.combine_masks(old, huge, mode="replace")

    message = str(excinfo.value)
    assert "70000" in message
    assert "minimum area" in message
