"""Recrop: one field becoming the several fields it should have been.

Every other tool on this screen edits the mask on the field in view. This
one changes WHICH field is in view -- it writes new fields and retires the
one they came out of -- so nothing here is asserted by reading the canvas.
The claims are files on disk, an enumeration that no longer holds a name,
and a queue position, and each is read back from the thing itself.

FOUR THINGS THE STANDALONE CURATOR LEARNED THE HARD WAY, and the reason
each has a test of its own:

* a box under :data:`~spacr.qt.mask_engine.RECROP_MIN_SIDE` px on a side is
  refused -- it is a mis-click, and cutting it writes a field too small to
  hold what it was aimed at;
* a box that repeats one already cut is refused as a RE-DRAW rather than
  written as a second object. Without that refusal one object reached disk
  three times, because nothing on screen said the first box had worked;
* every object the box cuts through is dropped, so a shape whose boundary
  is where the mouse was released never becomes ground truth;
* what survives is renumbered from one, so the new field is a field rather
  than a view of another one.

WHAT "RETIRED" MEANS IN SPACR. This screen's queue is a folder walked by
:func:`~spacr.qt.mask_engine.list_images`, not a table. spaCR's crop
database -- ``png_list`` -- is keyed on each crop's absolute path and has
no lifecycle column, so there is no row to set to "recropped" and none a
folder-reading screen has any claim to rewrite. The nearest thing that is
recoverable is what is tested here: image, mask and ledger move into
``recropped_originals/``, out of the enumeration, onto a manifest that
says what moved and where from, and back again on request.

The canvas is pinned to 600x400 with a 256x256 image, so ``refresh()``
scales the composite to 400x400 centred with a 100 px margin either side
and the canvas->image mapping is exactly ``img = (canvas - (100, 0)) *
256/400``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr.curation import CurationLog, is_curated
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import (
    MODE_RECROP,
    RECROP_TOOLTIP,
    TOOL_MODES,
    MakeMasksScreen,
    tool_row_entries,
)

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 256
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2      # 100


# ===========================================================================
# Fields to cut up
# ===========================================================================

def three_object_field():
    """A 256x256 field holding three well-separated objects.

    Object 1 sits in the top-left quadrant, object 2 in the bottom-right,
    and object 3 is in the middle between them -- so a box drawn round
    object 1 a little too wide reaches object 3 and has something to clip.
    """
    image = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    for value, (top, left) in enumerate(((24, 24), (176, 176), (96, 96)), 1):
        image[top:top + 40, left:left + 40] = 20000 + 10000 * value
        mask[top:top + 40, left:left + 40] = value
    return image, mask


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder the screen can open: two fields, each with a draft mask."""
    root = tmp_path / "fields"
    (root / "masks").mkdir(parents=True)
    image, mask = three_object_field()
    for name in ("field_a", "field_b"):
        imageio.imwrite(root / f"{name}.tif", image)
        imageio.imwrite(root / "masks" / f"{name}.tif", mask)
    return root


# ===========================================================================
# The two refusals — asserted on the engine, not through the GUI alone
# ===========================================================================

def test_a_box_one_pixel_under_the_minimum_side_is_refused():
    """"a box under 32 px on a side is refused, with a message"."""
    shape = (IMG_N, IMG_N)
    short = engine.RECROP_MIN_SIDE - 1
    with pytest.raises(engine.RecropRefused) as caught:
        engine.recrop_box(shape, (10, 10), (10 + short, 10 + short))
    assert caught.value.reason == "too_small"
    assert str(engine.RECROP_MIN_SIDE) in str(caught.value)
    # ...and exactly one pixel more is cut, so the refusal is a threshold
    # and not a tool that refuses everything small-ish.
    exact = engine.RECROP_MIN_SIDE
    assert engine.recrop_box(shape, (10, 10), (10 + exact, 10 + exact)) == (
        10, 10, 10 + exact, 10 + exact)


def test_each_side_is_judged_on_its_own():
    """A wide, flat box is as unusable as a small square one."""
    shape = (IMG_N, IMG_N)
    with pytest.raises(engine.RecropRefused) as caught:
        engine.recrop_box(shape, (10, 10), (200, 30))
    assert caught.value.reason == "too_small"
    with pytest.raises(engine.RecropRefused):
        engine.recrop_box(shape, (10, 10), (30, 200))


def test_a_box_dragged_off_the_image_is_clipped_before_it_is_judged():
    """A drag that ran past the edge cuts what is there, or is refused.

    Clipping first is what stops a box whose corner is off the image
    indexing past the array; judging after is what stops a box that is
    almost entirely off the image from being written as a sliver.
    """
    shape = (IMG_N, IMG_N)
    assert engine.recrop_box(shape, (-40, -40), (100, 100)) == (0, 0, 100, 100)
    with pytest.raises(engine.RecropRefused) as caught:
        engine.recrop_box(shape, (250, 250), (400, 400))
    assert caught.value.reason == "too_small"


def test_a_box_that_repeats_one_already_cut_is_refused_as_a_redraw():
    """The refusal that exists because one object reached disk three times."""
    shape = (IMG_N, IMG_N)
    first = engine.recrop_box(shape, (20, 20), (100, 100))
    with pytest.raises(engine.RecropRefused) as caught:
        engine.recrop_box(shape, (22, 23), (98, 101),
                          existing=[(*first, "field_a__r00")])
    assert caught.value.reason == "redraw"
    # The message names the crop that already holds it, so the user is
    # told the well is saved rather than that the tool said no.
    assert "field_a__r00" in str(caught.value)


def test_a_box_beside_a_cut_one_is_a_second_object_not_a_redraw():
    """The refusal must not stop the job it exists to make possible."""
    shape = (IMG_N, IMG_N)
    first = engine.recrop_box(shape, (20, 20), (100, 100))
    second = engine.recrop_box(shape, (110, 110), (200, 200),
                               existing=[(*first, "field_a__r00")])
    assert second == (110, 110, 200, 200)


def test_overlap_is_measured_as_intersection_over_union():
    """A small box inside a large one is not the large one drawn again.

    Fraction-of-the-smaller would call a cell boxed inside a well already
    cut a re-draw; intersection over union calls it what it is.
    """
    big = (0, 0, 200, 200)
    inside = (10, 10, 60, 60)
    assert engine.box_overlap(big, inside) == pytest.approx(2500 / 40000)
    assert engine.box_overlap(big, big) == 1.0
    assert engine.box_overlap(big, (300, 300, 400, 400)) == 0.0
    # ...so the box inside is cut rather than refused.
    assert engine.recrop_box((IMG_N, IMG_N), (10, 10), (60, 60),
                             existing=[(*big, "r00")]) == inside


# ===========================================================================
# The two rules about what the box carries away
# ===========================================================================

def test_every_object_the_box_cuts_through_is_dropped():
    """"a half object never becomes ground truth"."""
    image, mask = three_object_field()
    # This box holds all of object 1 and slices the corner off object 3.
    box = engine.recrop_box(mask.shape, (10, 10), (110, 110))
    _sub_image, sub_mask = engine.cut_recrop(image, mask, box)
    # Object 3 really was inside the box before the drop, so the test is
    # about the drop and not about the box missing it.
    raw = mask[10:110, 10:110]
    assert 3 in set(np.unique(raw))
    kept = [int(v) for v in np.unique(sub_mask) if v]
    assert len(kept) == 1, f"a clipped object survived: {kept}"
    # ...and the whole one is untouched: 40x40 px, every pixel of it.
    assert int(np.count_nonzero(sub_mask)) == 40 * 40


def test_the_labels_that_survive_are_renumbered_from_one():
    """"the sub-mask is relabelled from one, so the crop is self-consistent"."""
    image, mask = three_object_field()
    box = engine.recrop_box(mask.shape, (170, 170), (230, 230))
    _sub_image, sub_mask = engine.cut_recrop(image, mask, box)
    assert 2 in set(np.unique(mask[170:230, 170:230]))
    assert set(int(v) for v in np.unique(sub_mask)) == {0, 1}
    assert sub_mask.dtype == np.uint16


def test_the_sub_image_is_that_region_of_the_image_pixel_for_pixel():
    """Both halves of the crop, or the mask no longer names the picture."""
    image, mask = three_object_field()
    box = (32, 48, 160, 176)
    sub_image, sub_mask = engine.cut_recrop(image, mask, box)
    assert np.array_equal(sub_image, image[48:176, 32:160])
    assert sub_image.shape == sub_mask.shape == (128, 128)


def test_a_box_that_cuts_every_object_writes_an_empty_field():
    """Refusing here would be worse: an empty crop is a legitimate answer.

    A box aimed between two touching cells carries neither of them. That is
    the border rule doing its job, so the crop is written -- and the count
    the screen reports is what tells the user it happened.
    """
    image, mask = three_object_field()
    # Object 1 spans 24..63; this box starts inside it, so the only object
    # it reaches is one it cuts.
    box = engine.recrop_box(mask.shape, (40, 40), (90, 90))
    assert int(mask[40:90, 40:90].max()) == 1
    _sub_image, sub_mask = engine.cut_recrop(image, mask, box)
    assert int(sub_mask.max()) == 0


# ===========================================================================
# The two writes: the image and the mask, where the folder walk finds them
# ===========================================================================

def test_the_child_is_written_as_a_field_the_folder_walk_enumerates(folder):
    """Both halves land where :func:`load_image_and_mask` looks for them."""
    image, mask = three_object_field()
    written = engine.write_recrop(str(folder), "field_a.tif", image, mask,
                                  (10, 10, 90, 90))
    assert written.name == "field_a__r00.tif"
    assert os.path.isfile(written.image_path)
    assert os.path.isfile(written.mask_path)
    assert written.mask_path == str(folder / "masks" / "field_a__r00.tif")
    assert written.name in engine.list_images(str(folder))

    back_image, back_mask = engine.load_image_and_mask(str(folder),
                                                       written.name)
    expect_image, expect_mask = engine.cut_recrop(image, mask,
                                                  (10, 10, 90, 90))
    # Bit for bit, both ways: a child re-read through the loader that will
    # actually open it must be the array the box carried away, or the mask
    # is drawn against pixels that moved on the way to disk.
    assert np.array_equal(back_image, expect_image)
    assert np.array_equal(back_mask, expect_mask)
    assert written.n_objects == int(expect_mask.max()) == 1


def test_the_child_says_in_its_own_ledger_that_it_was_cut(folder):
    """Which objects a mask does NOT hold was a human decision."""
    image, mask = three_object_field()
    written = engine.write_recrop(str(folder), "field_a.tif", image, mask,
                                  (10, 10, 90, 90))
    assert is_curated(written.mask_path)
    log = CurationLog.read_beside(written.mask_path)
    edit = log.edits[-1]
    assert edit.kind == engine.RECROP_KIND
    assert edit.detail["parent"] == "field_a.tif"
    assert edit.detail["box"] == [10, 10, 90, 90]


def test_two_boxes_out_of_one_field_get_two_names(folder):
    """A second crop must not land on the first one's filename."""
    image, mask = three_object_field()
    first = engine.write_recrop(str(folder), "field_a.tif", image, mask,
                                (10, 10, 90, 90))
    second = engine.write_recrop(str(folder), "field_a.tif", image, mask,
                                 (170, 170, 230, 230))
    assert (first.name, second.name) == ("field_a__r00.tif",
                                         "field_a__r01.tif")
    assert os.path.isfile(first.image_path)
    assert int(imageio.imread(second.mask_path).max()) == 1


def test_a_recrop_of_a_recrop_is_named_after_the_original_field(folder):
    """Names say which FIELD a crop came from, and stop growing there."""
    image, mask = three_object_field()
    engine.write_recrop(str(folder), "field_a.tif", image, mask,
                        (10, 10, 90, 90))
    again = engine.write_recrop(str(folder), "field_a__r00.tif", image, mask,
                                (180, 180, 240, 240))
    assert again.name == "field_a__r01.tif"


def test_a_child_name_is_not_reused_after_its_parent_is_archived(folder):
    """A second session must not overwrite the first session's crops."""
    image, mask = three_object_field()
    first = engine.write_recrop(str(folder), "field_a.tif", image, mask,
                                (10, 10, 90, 90))
    engine.retire_recropped_original(str(folder), "field_a.tif",
                                     children=[first.name],
                                     boxes=[(10, 10, 90, 90)])
    # The child is still in the folder; the retired PARENT is not. A name
    # is free only when nothing anywhere has it.
    reopened = engine.write_recrop(str(folder), "field_a__r00.tif",
                                   image, mask, (170, 170, 230, 230))
    assert reopened.name == "field_a__r01.tif"
    assert np.array_equal(imageio.imread(first.image_path),
                          engine.cut_recrop(image, mask,
                                            (10, 10, 90, 90))[0])


# ===========================================================================
# Retiring the original: out of the enumeration, not off the disk
# ===========================================================================

def test_the_original_leaves_the_queue_and_keeps_every_byte(folder):
    """"MOVED ASIDE rather than deleted, because a recrop drawn wrong has
    to be recoverable"."""
    image, mask = three_object_field()
    log = CurationLog(engine.mask_save_path(str(folder), "field_a.tif"))
    log.append("recrop", "field_a__r00.tif", n_changed=6400)
    log.write_beside(engine.mask_save_path(str(folder), "field_a.tif"))
    before = imageio.imread(folder / "field_a.tif")

    record = engine.retire_recropped_original(
        str(folder), "field_a.tif", children=["field_a__r00.tif"],
        boxes=[(10, 10, 90, 90)])

    assert "field_a.tif" not in engine.list_images(str(folder))
    assert not (folder / "field_a.tif").exists()
    archive = folder / engine.RECROP_ARCHIVE_DIRNAME
    assert np.array_equal(imageio.imread(archive / "field_a.tif"), before)
    assert (archive / "masks" / "field_a.tif").is_file()
    # The ledger travels with the file it describes.
    assert is_curated(archive / "masks" / "field_a.tif")
    assert len(record["moved"]) == 3
    # ...and the archive is not itself enumerated as a field.
    assert engine.list_images(str(folder)) == ["field_b.tif"]


def test_the_manifest_records_what_moved_and_which_boxes_were_cut(folder):
    image, mask = three_object_field()
    engine.retire_recropped_original(
        str(folder), "field_a.tif",
        children=["field_a__r00.tif", "field_a__r01.tif"],
        boxes=[(10, 10, 90, 90), (180, 180, 240, 240)])
    manifest = engine.read_recrop_manifest(str(folder))
    assert len(manifest) == 1
    assert manifest[0]["original"] == "field_a.tif"
    assert manifest[0]["children"] == ["field_a__r00.tif",
                                       "field_a__r01.tif"]
    assert manifest[0]["boxes"] == [[10, 10, 90, 90], [180, 180, 240, 240]]
    # Ordinary JSON, so recovery never needs this module.
    raw = json.loads(
        (folder / engine.RECROP_ARCHIVE_DIRNAME
         / engine.RECROP_MANIFEST).read_text())
    assert raw == manifest


def test_a_second_retirement_appends_rather_than_replaces(folder):
    engine.retire_recropped_original(str(folder), "field_a.tif",
                                     children=["field_a__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    engine.retire_recropped_original(str(folder), "field_b.tif",
                                     children=["field_b__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    manifest = engine.read_recrop_manifest(str(folder))
    assert [r["original"] for r in manifest] == ["field_a.tif", "field_b.tif"]


def test_an_unreadable_manifest_does_not_stop_a_retirement(folder):
    """The files are the recovery; the manifest is the map to them."""
    archive = folder / engine.RECROP_ARCHIVE_DIRNAME
    archive.mkdir()
    (archive / engine.RECROP_MANIFEST).write_text("{ not json")
    engine.retire_recropped_original(str(folder), "field_a.tif",
                                     children=["field_a__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    assert (archive / "field_a.tif").is_file()
    assert engine.read_recrop_manifest(str(folder))[0]["original"] == \
        "field_a.tif"


def test_a_retired_original_goes_back_where_it_came_from(folder):
    """The reason retiring moves instead of deleting, exercised."""
    before = imageio.imread(folder / "field_a.tif")
    engine.retire_recropped_original(str(folder), "field_a.tif",
                                     children=["field_a__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    assert "field_a.tif" not in engine.list_images(str(folder))

    restored = engine.restore_recropped_original(str(folder), "field_a.tif")

    assert len(restored) == 2                    # the image and its mask
    assert "field_a.tif" in engine.list_images(str(folder))
    assert np.array_equal(imageio.imread(folder / "field_a.tif"), before)
    assert (folder / "masks" / "field_a.tif").is_file()


def test_restoring_a_field_that_was_never_retired_does_nothing(folder):
    assert engine.restore_recropped_original(str(folder), "field_b.tif") == []


# ===========================================================================
# The tool in the row
# ===========================================================================

def test_recrop_is_one_of_the_tools_in_the_one_row(qtbot, qt_theme_applied):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert (MODE_RECROP, "Recrop", "recrop") in TOOL_MODES
    assert MODE_RECROP in dict((m, l) for m, l, _i in tool_row_entries())
    button = screen._mode_buttons[MODE_RECROP]
    assert screen._tool_row_layout.indexOf(button) >= 0
    # Last in the row: it is the only tool that does not edit the mask.
    modes = [m for m, _l, _i in tool_row_entries()]
    assert modes[-1] == MODE_RECROP
    # And it says what it does, because pressing it and looking does not.
    assert button.toolTip() == RECROP_TOOLTIP


def test_the_recrop_shortcut_does_not_collide(qtbot, qt_theme_applied):
    """R, and R is not taken."""
    from PySide6.QtGui import QKeySequence, QShortcut

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    keys = [s.key().toString() for s in screen.findChildren(QShortcut)]
    assert "R" in keys
    assert keys.count("R") == 1

    screen._set_mode("brush")
    for shortcut in screen.findChildren(QShortcut):
        if shortcut.key() == QKeySequence("R"):
            shortcut.activated.emit()
    assert screen._canvas.mode == MODE_RECROP
    assert screen._mode_buttons[MODE_RECROP].isChecked()


# ===========================================================================
# The gesture, driven on the real widget
# ===========================================================================

def _canvas_xy(img_x: float, img_y: float) -> tuple:
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


#: Image coordinates the drags below use. A mouse position is an INTEGER
#: widget pixel, and one widget pixel is 256/400 of an image pixel here, so
#: only image coordinates that land on a whole widget pixel survive the
#: round trip -- multiples of 16 on this canvas. Using them keeps every box
#: below exact instead of one pixel short at a rounding boundary.
BOX_A = ((16, 16), (112, 112))         # -> (16, 16, 113, 113), 97x97 px
BOX_B = ((128, 128), (240, 240))       # -> (128, 128, 241, 241), 113x113 px
BOX_A_REDRAWN = ((16, 16), (96, 96))   # 70% of BOX_A: a re-draw, not a well
BOX_TINY = ((16, 16), (32, 32))        # 17x17 px: under the minimum side


def box_drag(canvas, p0, p1) -> None:
    """Press, drag and release a rectangle, in image coordinates."""
    start = _canvas_xy(*p0)
    end = _canvas_xy(*p1)
    canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, *start))
    canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove, *end,
                               buttons=Qt.LeftButton, button=Qt.NoButton))
    canvas.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease, *end,
                                  buttons=Qt.NoButton, button=Qt.LeftButton))


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    """The real screen, laid out, with the folder open on field_a."""
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._canvas.setFixedSize(CANVAS_W, CANVAS_H)
    widget._open_folder(str(folder))
    widget._canvas.refresh()
    widget._set_mode(MODE_RECROP)
    assert widget._image_files[widget._current_index] == "field_a.tif"
    return widget


def test_a_dragged_box_writes_a_child_and_queues_it_next(screen, folder):
    """The whole gesture: drag, and a new field exists and is next."""
    box_drag(screen._canvas, *BOX_A)

    child = "field_a__r00.tif"
    assert (folder / child).is_file()
    assert (folder / "masks" / child).is_file()
    # Straight after the field it came from, so Next opens it.
    assert screen._image_files == ["field_a.tif", child, "field_b.tif"]
    assert screen._recrop_children == [child]
    assert "field_a__r00" in screen._status_label.text()


def test_the_children_come_out_in_the_order_they_were_drawn(screen):
    box_drag(screen._canvas, *BOX_A)
    box_drag(screen._canvas, *BOX_B)
    assert screen._image_files == ["field_a.tif", "field_a__r00.tif",
                                   "field_a__r01.tif", "field_b.tif"]


def test_a_box_too_small_writes_nothing_and_says_why(screen, folder):
    """The first refusal, through the tool the user actually holds."""
    before = set(engine.list_images(str(folder)))
    box_drag(screen._canvas, *BOX_TINY)
    assert set(engine.list_images(str(folder))) == before
    assert screen._recrop_children == []
    assert screen._canvas.recrop_boxes == []
    assert str(engine.RECROP_MIN_SIDE) in screen._status_label.text()


def test_the_same_object_boxed_twice_is_refused_not_written_twice(screen,
                                                                  folder):
    """The second refusal, and the failure it exists to prevent."""
    box_drag(screen._canvas, *BOX_A)
    assert screen._recrop_children == ["field_a__r00.tif"]

    box_drag(screen._canvas, *BOX_A_REDRAWN)

    assert screen._recrop_children == ["field_a__r00.tif"]
    assert not (folder / "field_a__r01.tif").exists()
    assert "field_a__r00" in screen._status_label.text()
    assert len(screen._canvas.recrop_boxes) == 1


def test_the_box_that_was_cut_stays_marked_on_the_canvas(screen):
    """A written box and a refused one must not look the same.

    The mark is kept in IMAGE pixels, so it stays on the object it was
    drawn round when the view is zoomed rather than sliding off it.
    """
    box_drag(screen._canvas, *BOX_A)
    assert len(screen._canvas.recrop_boxes) == 1
    box = screen._canvas.recrop_boxes[0]
    assert tuple(box[:4]) == (16, 16, 113, 113)
    assert box[4] == "field_a__r00"

    top_left = screen._canvas._image_to_canvas(16, 16)
    assert abs(top_left.x() - _canvas_xy(16, 16)[0]) <= 1
    assert abs(top_left.y() - _canvas_xy(16, 16)[1]) <= 1
    # Zoomed in, the same image pixel maps somewhere else on the widget --
    # which is the whole reason the box is not stored in widget coords.
    screen._canvas._zoom_x0, screen._canvas._zoom_y0 = 0, 0
    screen._canvas._zoom_x1, screen._canvas._zoom_y1 = 128, 128
    screen._canvas.refresh()
    zoomed = screen._canvas._image_to_canvas(16, 16)
    assert zoomed != top_left


def test_the_boxes_do_not_follow_the_user_to_the_next_field(screen):
    box_drag(screen._canvas, *BOX_A)
    screen._on_next()                      # retires field_a, opens the child
    assert screen._canvas.recrop_boxes == []
    assert screen._recrop_children == []


def test_the_parent_ledger_records_every_cut(screen, folder):
    box_drag(screen._canvas, *BOX_A)
    entries = [e for e in screen._log.edits if e.kind == engine.RECROP_KIND]
    assert len(entries) == 1
    assert entries[0].target == "field_a__r00.tif"
    assert entries[0].detail["box"] == [16, 16, 113, 113]
    assert entries[0].detail["n_objects"] == 1
    # An area, not a zero: the ledger's own summary must be able to tell a
    # cut that took a sixth of the field from one that took nothing.
    assert entries[0].n_changed == 97 * 97


def test_moving_on_retires_the_parent_and_opens_the_first_child(screen,
                                                                folder):
    box_drag(screen._canvas, *BOX_A)
    box_drag(screen._canvas, *BOX_B)

    screen._on_next()

    assert screen._image_files == ["field_a__r00.tif", "field_a__r01.tif",
                                   "field_b.tif"]
    assert screen._current_index == 0
    assert not (folder / "field_a.tif").exists()
    archive = folder / engine.RECROP_ARCHIVE_DIRNAME
    assert (archive / "field_a.tif").is_file()
    assert (archive / "masks" / "field_a.tif").is_file()
    # The parent's record of the boxes went into the archive with it.
    assert is_curated(archive / "masks" / "field_a.tif")
    kinds = [e.kind for e in
             CurationLog.read_beside(archive / "masks" / "field_a.tif").edits]
    assert kinds.count(engine.RECROP_KIND) == 2
    # The field on screen is the first child, loaded from disk.
    assert screen._canvas.mask.shape == (97, 97)
    assert int(screen._canvas.mask.max()) == 1


def test_next_on_a_field_that_was_not_recropped_is_unchanged(screen, folder):
    """Retiring must only ever happen to a field that was cut up."""
    screen._on_next()
    assert screen._image_files == ["field_a.tif", "field_b.tif"]
    assert screen._current_index == 1
    assert (folder / "field_a.tif").is_file()
    assert not (folder / engine.RECROP_ARCHIVE_DIRNAME).exists()


def test_leaving_backwards_retires_the_parent_too(screen, folder):
    """A parent must not be reachable again as though it were a field."""
    screen._current_index = 1               # field_b, so there is a "back"
    screen._load_current()
    box_drag(screen._canvas, *BOX_A)

    screen._on_prev()

    assert not (folder / "field_b.tif").exists()
    assert screen._image_files == ["field_a.tif", "field_b__r00.tif"]
    assert screen._current_index == 0


def test_recrop_with_no_field_open_refuses_instead_of_raising(qtbot,
                                                              qt_theme_applied):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget.recrop(0, 0, 100, 100) is None
    assert "no field open" in widget._status_label.text()
    assert widget.finish_recrop() is False


def test_restoring_skips_records_and_moves_it_cannot_honour(folder):
    """Recovery is best-effort on each move and never raises.

    The manifest is a map to files, and files move: a half-restored
    retirement has to put back what is still there rather than stopping at
    the first entry that has gone.
    """
    # field_a first, so the newest record in the manifest names field_b and
    # the search has to walk past a record that is not the one asked for.
    engine.retire_recropped_original(str(folder), "field_a.tif",
                                     children=["field_a__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    engine.retire_recropped_original(str(folder), "field_b.tif",
                                     children=["field_b__r00.tif"],
                                     boxes=[(10, 10, 90, 90)])
    archive = folder / engine.RECROP_ARCHIVE_DIRNAME
    manifest_path = archive / engine.RECROP_MANIFEST
    records = json.loads(manifest_path.read_text())
    # A malformed pair (a record hand-trimmed in the JSON) and a move whose
    # archived file somebody has already deleted.
    record = next(r for r in records if r["original"] == "field_a.tif")
    record["moved"].append(["only-one-half"])
    (archive / "masks" / "field_a.tif").unlink()
    manifest_path.write_text(json.dumps(records))

    restored = engine.restore_recropped_original(str(folder), "field_a.tif")

    assert restored == [str(folder / "field_a.tif")]
    assert (folder / "field_a.tif").is_file()
    # The record for the OTHER field was walked past, not applied.
    assert not (folder / "field_b.tif").exists()
    # ...and a second call is a no-op rather than a failure: the image is
    # back, so there is nothing left to put back.
    assert engine.restore_recropped_original(str(folder), "field_a.tif") == []


def test_the_cut_boxes_are_drawn_on_the_canvas(screen):
    """The mark is painted, not merely remembered.

    Rendered and counted rather than asserted on the list, because the list
    is not what tells the user the box worked.
    """
    from PySide6.QtGui import QColor, QImage

    from spacr.qt.theme import active_palette

    def accent_pixels() -> int:
        image = QImage(screen._canvas.size(), QImage.Format_RGB32)
        image.fill(QColor("black"))
        screen._canvas.render(image)
        arr = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
            image.height(), image.bytesPerLine() // 4)
        return int((arr == np.uint32(
            QColor(active_palette()["accent"]).rgb())).sum())

    before = accent_pixels()
    box_drag(screen._canvas, *BOX_A)
    after = accent_pixels()
    # A 97x97 image box is ~152 px a side on this canvas, so its 2 px
    # outline is several hundred pixels of accent that were not there.
    assert after - before > 400


def test_a_write_that_fails_leaves_the_queue_exactly_as_it_was(screen,
                                                               monkeypatch):
    """A crop that did not reach disk must not be queued as though it had."""
    def explode(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(engine, "write_recrop", explode)
    queue = list(screen._image_files)

    box_drag(screen._canvas, *BOX_A)

    assert screen._image_files == queue
    assert screen._recrop_children == []
    assert screen._canvas.recrop_boxes == []
    assert "no space left" in screen._status_label.text()


def test_retiring_the_only_field_leaves_an_empty_queue_and_says_so(
        qtbot, qt_theme_applied, tmp_path):
    """The end of the queue is reachable through a recrop like any other."""
    root = tmp_path / "one"
    (root / "masks").mkdir(parents=True)
    image, mask = three_object_field()
    imageio.imwrite(root / "only.tif", image)
    imageio.imwrite(root / "masks" / "only.tif", mask)

    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._canvas.setFixedSize(CANVAS_W, CANVAS_H)
    widget._open_folder(str(root))
    widget._canvas.refresh()
    widget._set_mode(MODE_RECROP)
    box_drag(widget._canvas, *BOX_A)
    # The child was queued after the parent, so the parent is not last.
    assert widget._image_files == ["only.tif", "only__r00.tif"]

    # Retiring from the child's own position: pop it too, so the queue runs
    # out and the screen has to say that rather than index past the end.
    widget._image_files = ["only.tif"]
    assert widget.finish_recrop() is True
    assert widget._image_files == []
    assert widget._canvas.mask is None
    assert "queue empty" in widget._status_label.text()
    assert (root / engine.RECROP_ARCHIVE_DIRNAME / "only.tif").is_file()


def test_a_box_is_not_drawn_before_the_canvas_has_composited_anything(screen):
    """A mark that cannot be placed is not placed at the widget origin.

    The canvas is asked to paint on resize and on show, both of which can
    arrive before ``refresh()`` has put a pixmap on it. A box drawn at (0, 0)
    then would be a blue square over an object it names nothing about.
    """
    box_drag(screen._canvas, *BOX_A)
    screen._canvas.clear()                    # no pixmap: nothing to map onto

    assert screen._canvas._image_to_canvas(16, 16) is None
    # The painter is never opened, so nothing is drawn and nothing warns.
    assert screen._canvas._paint_recrop_boxes() is None
    assert screen._canvas.recrop_boxes            # the box is still recorded


def test_a_parent_whose_mask_will_not_save_is_still_moved_to_safety(
        screen, folder, monkeypatch):
    """The archive is the recovery, so nothing may stop the original reaching it."""
    box_drag(screen._canvas, *BOX_A)

    def explode(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(engine, "save_mask", explode)
    assert screen.finish_recrop() is True
    assert (folder / engine.RECROP_ARCHIVE_DIRNAME / "field_a.tif").is_file()
    assert "field_a.tif" not in screen._image_files


def test_a_retirement_that_fails_leaves_the_parent_in_the_queue(
        screen, monkeypatch):
    """A parent that did not move must not be dropped from the queue.

    Popping it anyway would leave the multi-object field on disk with
    nothing enumerating it and nothing recording that it was meant to go.
    """
    box_drag(screen._canvas, *BOX_A)

    def explode(*_args, **_kwargs):
        raise OSError("device or resource busy")

    monkeypatch.setattr(engine, "retire_recropped_original", explode)
    assert screen.finish_recrop() is False
    assert screen._image_files[0] == "field_a.tif"
    assert screen._recrop_children == ["field_a__r00.tif"]
    assert "busy" in screen._status_label.text()
