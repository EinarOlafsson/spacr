"""The edges of the Make Masks backend: off-canvas strokes, odd frames, dirty files.

Everything here is a path the editor takes when the hand at the tablet, or the
folder on disk, is not the tidy case the happy path assumes: a stroke that
starts outside the image, a straight vertical drag, a wand click on ground the
mask already owns, a crop with nothing whole inside it, and an archive manifest
some other tool overwrote with the wrong shape of JSON.

None of these may raise or corrupt the label image. The screen has no modal
dialog to catch a traceback, so an exception here kills the annotation session
and loses whatever the user had drawn since the last save.

Pure numpy plus the filesystem -- the backend has no Qt dependency on purpose.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spacr.qt import mask_engine as me
from spacr.tiff_io import write_tiff


@pytest.fixture()
def queue_folder(tmp_path):
    """An image queue directory with its ``masks/`` subfolder."""
    (tmp_path / "masks").mkdir()
    return str(tmp_path)


def _lay_field(folder, name, size=8):
    """Write one blank field and its mask into an image queue."""
    write_tiff(os.path.join(folder, name), np.zeros((size, size), np.uint16))
    write_tiff(me.mask_save_path(folder, name), np.zeros((size, size), np.uint16))


# ---------------------------------------------------------------------------
# overlay_mask -- what the canvas actually shows
# ---------------------------------------------------------------------------

def test_an_overlay_of_an_already_colour_frame_keeps_its_three_planes():
    """The preview must accept a frame that arrived with a colour axis.

    ``overlay_mask`` grows a 2-D field into RGB before blending, but the
    canvas also hands it frames that are already three-plane -- an RGB source
    image, or a second overlay pass over a composite it made earlier. If the
    ndim==2 stack ran unconditionally the array would become (h, w, 3, 3) and
    the blend would raise, blanking the canvas mid-session.
    """
    colour = np.zeros((4, 4, 3), dtype=np.uint16)
    colour[..., 0] = 60000          # a red frame, already three planes
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1

    out = me.overlay_mask(colour, mask, alpha=0.5)

    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    # Unlabelled pixels are the source frame scaled to 8-bit and nothing else.
    assert out[0, 0].tolist() == [60000 // 256, 0, 0]
    # The labelled pixel is half source, half its object colour, so its red
    # drops off the source value and its blue rises off zero.
    assert out[1, 1][0] < out[0, 0][0]
    assert out[1, 1][2] > 0
    # A grayscale frame still comes back as RGB of the same footprint.
    grey = me.overlay_mask(np.zeros((4, 4), dtype=np.uint16), mask)
    assert grey.shape == (4, 4, 3)


def test_a_mask_carrying_negative_labels_stops_instead_of_drawing_nonsense():
    """A signed mask with negative entries must not be blended silently.

    Some upstream tools mark "unassigned" with -1 rather than 0. Those values
    are not object ids and there is no colour for them; the palette is built
    for 0..max_label. The call fails loudly, which is what the caller can see
    and report -- a silently wrapped colour lookup would paint object -1 in
    the colour of the highest-numbered real cell and the user would accept a
    mask whose objects do not mean what the picture showed.
    """
    image = np.zeros((3, 3), dtype=np.uint16)
    unassigned = np.full((3, 3), -1, dtype=np.int16)

    with pytest.raises(IndexError):
        me.overlay_mask(image, unassigned)

    # The same shape with ordinary nonnegative ids blends without complaint,
    # so it is the negative id that is refused and not the frame or the size.
    ok = np.zeros((3, 3), dtype=np.int16)
    ok[2, 2] = 1
    blended = me.overlay_mask(image, ok)
    assert blended.shape == (3, 3, 3)
    assert blended[2, 2].sum() > 0
    assert blended[0, 0].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# The brush -- strokes that leave the frame, and strokes that go straight down
# ---------------------------------------------------------------------------

def test_a_brush_stamp_outside_the_frame_leaves_the_mask_untouched():
    """Dragging off the edge of the image must not wrap paint to the far side.

    Canvas coordinates come from the mouse and routinely fall outside the
    image while the button is still held. The clipped box collapses to an
    empty slice; writing it anyway with a negative bound would paint a stripe
    at the opposite edge of the mask -- an object the user never drew, saved
    into the label image on the next stroke.
    """
    mask = np.zeros((6, 6), dtype=np.uint8)

    me.paint_disk(mask, -20, 3, radius=2, value=9)   # left of the frame
    me.paint_disk(mask, 30, 3, radius=2, value=9)    # right of the frame
    me.paint_disk(mask, 3, -20, radius=2, value=9)   # above the frame
    assert int(mask.sum()) == 0

    # The same call inside the frame does paint, so the no-op above is the
    # clipping and not a brush that never works.
    me.paint_disk(mask, 3, 3, radius=1, value=9)
    assert int((mask == 9).sum()) == 4
    assert mask[3, 3] == 9


def test_a_straight_vertical_drag_paints_every_row_it_crosses():
    """A vertical stroke must be continuous, not a dotted line.

    The Bresenham walk steps x and y under separate tests. A vertical drag
    never satisfies the x test, so if the y step were tied to it the stroke
    would advance one pixel and then loop on the same point until the guard
    at the endpoint fired -- users drawing a cell border straight down would
    get a single blob at the top of the drag.
    """
    mask = np.zeros((10, 10), dtype=np.uint8)

    me.paint_line(mask, 5, 1, 5, 8, radius=1, value=7)

    rows = sorted(set(np.nonzero(mask)[0].tolist()))
    cols = sorted(set(np.nonzero(mask)[1].tolist()))
    assert rows == list(range(0, 9))     # every row from the start to the end
    assert cols == [4, 5]                # a radius-1 stamp, two columns wide
    assert mask[8, 5] == 7               # the endpoint itself is stamped
    # A horizontal drag on the same canvas is equally continuous.
    across = np.zeros((10, 10), dtype=np.uint8)
    me.paint_line(across, 1, 5, 8, 5, radius=1, value=7)
    assert sorted(set(np.nonzero(across)[1].tolist())) == list(range(0, 9))


# ---------------------------------------------------------------------------
# The magic wand over ground the mask already owns
# ---------------------------------------------------------------------------

def test_a_wand_click_on_painted_ground_rewrites_the_whole_object():
    """Wanding inside an existing object relabels it to the brush value.

    The flood only counts a pixel against its budget when the pixel actually
    changes state, so a click on ground the mask already owns keeps walking.
    That is what lets a user re-run the wand with a wider tolerance on an
    object they already outlined and get the enlarged object rather than a
    partial one; if the walk stopped at the first owned pixel the second
    click would do nothing at all.
    """
    image = np.full((40, 40), 100, dtype=np.uint16)
    owned = np.full((40, 40), 3, dtype=np.uint8)

    out = me.magic_wand(image, owned, 20, 20, tolerance=5,
                        max_pixels=4, action="add")

    # Every pixel is rewritten to the brush value even though the budget was
    # four: already-owned pixels cost nothing against max_pixels.
    assert np.unique(out).tolist() == [255]
    assert int((out == 255).sum()) == 1600
    assert int((owned == 3).sum()) == 1600      # the caller's array is intact

    # An erase over ground that is already empty is the mirror case and is
    # likewise free, leaving the blank mask blank.
    blank = np.zeros((40, 40), dtype=np.uint8)
    erased = me.magic_wand(image, blank, 20, 20, tolerance=5,
                           max_pixels=4, action="erase")
    assert int(erased.sum()) == 0

    # And the budget really does bite when pixels change: adding into a blank
    # mask stops at four pixels.
    fresh = me.magic_wand(image, blank, 20, 20, tolerance=5,
                          max_pixels=4, action="add")
    assert int((fresh > 0).sum()) == 4


# ---------------------------------------------------------------------------
# Recrop -- a box with nothing whole in it
# ---------------------------------------------------------------------------

def test_a_recrop_of_empty_ground_yields_a_blank_mask_not_a_crash():
    """Cutting a region that holds no object must still produce a usable pair.

    Objects touching the crop border are dropped because their outlines are
    incomplete, so a box drawn over background -- or over nothing but a
    clipped cell -- legitimately ends up with an empty label image. The crop
    still has to come back as a (image, mask) pair of matching shape and
    uint16 dtype, because the child field is written to disk from it. A
    failure here loses the crop the user just drew.
    """
    image = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    empty = np.zeros((100, 100), dtype=np.uint16)

    sub_image, sub_mask = me.cut_recrop(image, empty, (10, 10, 60, 60))
    assert sub_image.shape == (50, 50)
    assert sub_mask.shape == (50, 50)
    assert sub_mask.dtype == np.uint16
    assert int(sub_mask.max()) == 0

    # A degenerate box gives an empty array rather than an error.
    tiny_image, tiny_mask = me.cut_recrop(image, empty, (10, 10, 10, 10))
    assert tiny_image.size == 0 and tiny_mask.shape == (0, 0)

    # The same call over a box that does contain a whole object keeps it and
    # renumbers it from one, so the blank results above are the emptiness of
    # the region and not a crop that always discards.
    with_object = np.zeros((100, 100), dtype=np.uint16)
    with_object[40:50, 40:50] = 7
    kept_image, kept_mask = me.cut_recrop(image, with_object, (30, 30, 70, 70))
    assert kept_image.shape == (40, 40)
    assert int(kept_mask.max()) == 1
    assert int((kept_mask > 0).sum()) == 100


# ---------------------------------------------------------------------------
# The recrop archive manifest
# ---------------------------------------------------------------------------

def test_a_manifest_of_the_wrong_shape_is_replaced_not_obeyed(queue_folder):
    """A manifest that is valid JSON but not a list must not block retirement.

    The archive manifest is plain JSON so other tools can read it, which means
    another tool can also write an object where a list belongs. The retirement
    is the part that matters -- the source field and its mask are already off
    the queue -- so the manifest is rebuilt around the new record. If the
    non-list value were appended to, or trusted as records, retiring a
    recropped field would raise and the field would stay in the training queue
    forever.
    """
    _lay_field(queue_folder, "well_A1.tif")
    _lay_field(queue_folder, "well_B2.tif")
    archive = me.recrop_archive_dir(queue_folder)
    os.makedirs(archive, exist_ok=True)
    manifest = os.path.join(archive, me.RECROP_MANIFEST)
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"note": "written by something that is not spacr"}, handle)

    record = me.retire_recropped_original(
        queue_folder, "well_A1.tif",
        children=["well_A1__r00.tif"], boxes=[(0, 0, 32, 32)])

    assert record["original"] == "well_A1.tif"
    assert record["children"] == ["well_A1__r00.tif"]
    on_disk = me.read_recrop_manifest(queue_folder)
    assert [r["original"] for r in on_disk] == ["well_A1.tif"]
    with open(manifest, "r", encoding="utf-8") as handle:
        assert isinstance(json.load(handle), list)      # the object is gone
    assert me.list_images(queue_folder) == ["well_B2.tif"]
    assert os.path.exists(os.path.join(archive, "well_A1.tif"))

    # A well-formed manifest is appended to rather than replaced, so the
    # rebuild above is the repair of a bad file and not a manifest that only
    # ever holds one record.
    me.retire_recropped_original(
        queue_folder, "well_B2.tif",
        children=["well_B2__r00.tif"], boxes=[(0, 0, 32, 32)])
    assert [r["original"] for r in me.read_recrop_manifest(queue_folder)] == [
        "well_A1.tif", "well_B2.tif"]
    assert me.list_images(queue_folder) == []
