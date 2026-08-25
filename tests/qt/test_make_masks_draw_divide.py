"""The two region tools on the make-masks canvas: draw and divide.

Both are gestures rather than array calls, so every test here drives the
real widget with real mouse events and then reads the result off the mask
the canvas is holding. A tool that "works" when its engine function is
called by hand is not a tool anyone can use.

What the two must be able to say for themselves:

* **draw** closes a traced outline and fills it as ONE object with one
  id, without touching any object already in the field;
* **divide** cuts one merged object into two, keeps the original id on
  the larger piece, and leaves every other object's pixels exactly where
  they were -- including its id, which is what the whole-field relabel
  the standalone curation tool does after its cut cannot promise.

The canvas is pinned to 600x400 with a 64x64 image, so refresh() scales
the composite to 400x400 centred with a 100 px margin either side and the
canvas->image mapping is exactly ``img = (canvas - (100, 0)) * 64/400``.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent

from scipy.ndimage import label as _label

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import (
    MODE_BRUSH,
    MODE_DIVIDE,
    MODE_DRAW,
    MakeMasksScreen,
    _MaskCanvas,
)
from spacr.qt.theme import DARK_PALETTE

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2      # 100

#: Eight-connectivity, the connectivity every mask consumer in spaCR uses.
EIGHT = np.ones((3, 3), dtype=np.uint8)


def canvas_xy(img_x: float, img_y: float) -> tuple:
    """Canvas-local point that maps onto full-image pixel (img_x, img_y)."""
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def press(x, y):
    return _evt(QEvent.Type.MouseButtonPress, x, y)


def move(x, y):
    return _evt(QEvent.Type.MouseMove, x, y, buttons=Qt.LeftButton,
                button=Qt.NoButton)


def release(x, y):
    return _evt(QEvent.Type.MouseButtonRelease, x, y,
                buttons=Qt.NoButton, button=Qt.LeftButton)


def drag(widget, points) -> None:
    """Press, move through every point, release — in image coordinates."""
    first = canvas_xy(*points[0])
    widget.mousePressEvent(press(*first))
    for point in points[1:]:
        widget.mouseMoveEvent(move(*canvas_xy(*point)))
    last = canvas_xy(*points[-1])
    widget.mouseReleaseEvent(release(*last))


def accent_pixels(widget) -> int:
    """Pixels the widget renders in the theme accent — the gesture preview."""
    image = QImage(widget.size(), QImage.Format_RGB32)
    image.fill(QColor("black"))
    widget.render(image)
    arr = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
        image.height(), image.bytesPerLine() // 4)
    return int((arr == np.uint32(QColor(DARK_PALETTE["accent"]).rgb())).sum())


def field_image() -> np.ndarray:
    """64x64 uint16 image with something on it for the overlay to sit on."""
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[8:56, 8:56] = 20000
    return img


def merged_mask() -> np.ndarray:
    """One dumbbell-shaped merged object (id 7) and two bystanders (3, 9).

    Object 7 is what a segmentation gives you when two cells touch: two
    lobes joined by a waist, one id. Objects 3 and 9 are the field around
    it, and their ids and pixel counts are what a divide must not move.
    """
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    left = (xx - 20) ** 2 + (yy - 32) ** 2 <= 8 ** 2
    right = (xx - 40) ** 2 + (yy - 32) ** 2 <= 8 ** 2
    waist = (np.abs(yy - 32) <= 3) & (xx >= 20) & (xx <= 40)
    mask[left | right | waist] = 7
    mask[4:10, 4:10] = 3
    mask[54:60, 44:54] = 9
    return mask


def counts_by_label(mask: np.ndarray) -> dict:
    """{label: pixel count} for every object in the mask."""
    values, counts = np.unique(mask[mask > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """A canvas at a known size holding the merged-object field."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(CANVAS_W, CANVAS_H)
    widget.set_image_and_mask(field_image(), merged_mask())
    assert widget.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return widget


@pytest.fixture
def strokes(canvas):
    """Counts of stroke_started / stroke_finished emitted by the canvas.

    A gesture that changes nothing must emit neither: the screen pushes an
    undo step and writes a ledger entry on stroke_finished, so an empty
    gesture that emits it costs the user a Ctrl+Z that undoes nothing.
    """
    seen = {"started": 0, "finished": 0}
    canvas.stroke_started.connect(lambda: seen.__setitem__(
        "started", seen["started"] + 1))
    canvas.stroke_finished.connect(lambda: seen.__setitem__(
        "finished", seen["finished"] + 1))
    return seen


# The outline traced by the draw tests: a square in empty background,
# clear of objects 3, 7 and 9.
OUTLINE = [(12, 44), (18, 44), (24, 44), (24, 50), (24, 56),
           (18, 56), (12, 56), (12, 50)]


# ===========================================================================
# Draw — a traced outline becomes one object
# ===========================================================================

def test_draw_fills_a_traced_outline_into_one_object(canvas, strokes):
    """One closed outline -> exactly one new id, and nothing else moves."""
    before = counts_by_label(canvas.mask)
    canvas.mode = MODE_DRAW
    drag(canvas, OUTLINE)

    after = counts_by_label(canvas.mask)
    new = sorted(set(after) - set(before))
    assert len(new) == 1, f"one gesture must make one object, made {new}"
    made = new[0]
    assert made not in before, "a new object must not reuse a live id"

    # It covers the traced region: the middle of the square carries the new
    # id, and the object is one blob rather than the traced rim.
    assert int(canvas.mask[50, 18]) == made
    assert after[made] >= 100, f"a 12x12 outline filled only {after[made]} px"
    _, blobs = _label(canvas.mask == made, structure=EIGHT)
    assert blobs == 1, "the fill must be one object, not a rim in pieces"

    # Every object that was already there is untouched, id and area alike.
    assert {k: after[k] for k in before} == before
    assert strokes == {"started": 1, "finished": 1}


def test_a_draw_that_encloses_nothing_leaves_no_trace(canvas, strokes):
    """A straight drag encloses no area, so it makes no object.

    skimage's ``polygon`` hands back the traced pixels themselves for a
    degenerate outline, which would turn a stray drag into a hairline
    object with a real id — and an undo step and a ledger entry to match.
    """
    before = canvas.mask.copy()
    canvas.mode = MODE_DRAW
    drag(canvas, [(12, 44), (16, 44), (20, 44), (24, 44)])

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}, (
        "an empty gesture must not cost an undo step")


def test_draw_reports_the_id_it_made_for_the_ledger(canvas):
    """``last_edit`` names the tool and the object it created."""
    canvas.mode = MODE_DRAW
    drag(canvas, OUTLINE)

    edit = canvas.last_edit
    assert edit["kind"] == "draw"
    assert int(canvas.mask[50, 18]) == edit["target"]
    assert edit["detail"]["n_points"] == len(OUTLINE)


# ===========================================================================
# Divide — one merged object becomes two
# ===========================================================================

def test_divide_splits_the_merged_object_and_leaves_the_others_alone(
        canvas, strokes):
    """The waist is cut: one object becomes two, the field stays put."""
    before = counts_by_label(canvas.mask)
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(30, 16), (30, 32), (30, 48)])

    after = counts_by_label(canvas.mask)
    assert len(after) == len(before) + 1, (
        f"divide must add exactly one object: {before} -> {after}")
    new = sorted(set(after) - set(before))
    assert len(new) == 1
    made = new[0]

    # Two objects where there was one, each a single blob, one either side.
    assert int(canvas.mask[32, 20]) in (7, made)
    assert int(canvas.mask[32, 40]) in (7, made)
    assert int(canvas.mask[32, 20]) != int(canvas.mask[32, 40])
    for value in (7, made):
        _, blobs = _label(canvas.mask == value, structure=EIGHT)
        assert blobs == 1, f"object {value} came out in {blobs} pieces"

    # Every other object is identical — id and pixel count both.
    assert {k: after[k] for k in (3, 9)} == {k: before[k] for k in (3, 9)}
    assert strokes == {"started": 1, "finished": 1}


def test_the_larger_piece_keeps_the_original_id(canvas):
    """An off-centre cut leaves the id on the piece that carries most of it.

    The rule :func:`spacr.qt.mask_engine.canonical_labels` already applies
    when one id names two blobs, so a divide followed by a save renumbers
    nothing, and the id stays on the piece that is most of what it named.
    """
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(25, 16), (25, 48)])          # closer to the left lobe

    counts = counts_by_label(canvas.mask)
    made = sorted(set(counts) - {3, 7, 9})
    assert len(made) == 1
    assert counts[7] > counts[made[0]], (
        f"the original id landed on the smaller piece: {counts}")
    # And it is the right-hand, larger side that kept it.
    assert int(canvas.mask[32, 40]) == 7


def test_a_line_that_separates_nothing_leaves_the_mask_alone(canvas, strokes):
    """A cut that stops inside the object is a miss, not a groove.

    Zeroing the pixels anyway would carve a slot into the object and call
    it a division; leaving the mask alone means the gesture can just be
    drawn again.
    """
    before = canvas.mask.copy()
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(30, 16), (30, 30)])          # stops inside the waist

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}


def test_a_divide_drawn_over_background_changes_nothing(canvas, strokes):
    """A line that crosses no object at all does nothing."""
    before = canvas.mask.copy()
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(2, 40), (2, 60)])

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}


def test_divide_reports_both_ends_of_the_split_for_the_ledger(canvas):
    """``last_edit`` says which object was cut and what the piece was called."""
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(30, 16), (30, 48)])

    edit = canvas.last_edit
    assert edit["kind"] == "divide"
    assert edit["target"] == [7]
    made = edit["detail"]["new_labels"]
    assert len(made) == 1 and int((canvas.mask == made[0]).sum()) > 0
    assert edit["detail"]["n_objects"] == 1


def test_a_one_pixel_cut_would_not_have_separated_the_halves():
    """Why :data:`DIVIDE_CUT_WIDTH` is wider than one pixel.

    Two pixels touching at a corner are one object to everything in spaCR
    (``_EIGHT``), so a one-pixel cut across a diagonal leaves the halves
    corner-to-corner and the split does not take. This drives the band
    straight so the failure and the fix are both measured rather than
    argued: 1.0 px does not separate the object, the shipped width does.
    """
    disk = np.zeros((64, 64), dtype=np.uint16)
    yy, xx = np.mgrid[0:64, 0:64]
    disk[(xx - 32) ** 2 + (yy - 32) ** 2 <= 20 ** 2] = 5

    thin = engine._segment_band(disk.shape, (10, 10), (54, 54), 1.0)
    _, thin_pieces = _label((disk > 0) & ~thin, structure=EIGHT)
    assert thin_pieces == 1, "a 1 px diagonal cut used to be enough?"

    shipped = engine._segment_band(disk.shape, (10, 10), (54, 54),
                                    engine.DIVIDE_CUT_WIDTH)
    _, pieces = _label((disk > 0) & ~shipped, structure=EIGHT)
    assert pieces == 2

    # And it stays a cut rather than a slot: a few percent of the object.
    cost = int(((disk > 0) & shipped).sum()) / int((disk > 0).sum())
    assert cost < 0.10, f"the cut ate {cost:.0%} of the object"


def test_divide_cuts_diagonally_through_the_real_canvas(canvas):
    """The same cut through the widget, at an angle, still yields two."""
    canvas.mode = MODE_DIVIDE
    drag(canvas, [(22, 44), (38, 20)])          # diagonal across the waist

    counts = counts_by_label(canvas.mask)
    assert len(counts) == 4, f"expected 3 + 1 objects, got {counts}"
    _, blobs = _label(canvas.mask > 0, structure=EIGHT)
    assert blobs == 4, "the two halves are still touching at a corner"


# ===========================================================================
# The gesture in flight
# ===========================================================================

def test_the_outline_being_traced_is_drawn_on_the_canvas(canvas):
    """A draw in progress is visible before it is committed."""
    canvas.mode = MODE_DRAW
    quiet = accent_pixels(canvas)
    canvas.mousePressEvent(press(*canvas_xy(12, 44)))
    for point in OUTLINE[1:]:
        canvas.mouseMoveEvent(move(*canvas_xy(*point)))
    assert accent_pixels(canvas) > quiet, (
        "the outline being traced is invisible until release")
    canvas.mouseReleaseEvent(release(*canvas_xy(*OUTLINE[-1])))
    assert accent_pixels(canvas) == quiet, "the preview outlived the gesture"


def test_the_dividing_line_is_drawn_while_it_is_aimed(canvas):
    """A divide in progress shows the line it will cut along."""
    canvas.mode = MODE_DIVIDE
    quiet = accent_pixels(canvas)
    canvas.mousePressEvent(press(*canvas_xy(30, 16)))
    canvas.mouseMoveEvent(move(*canvas_xy(30, 48)))
    assert accent_pixels(canvas) > quiet
    # The drag re-aims the far end rather than bending the line: two points,
    # however many move events it took.
    canvas.mouseMoveEvent(move(*canvas_xy(30, 52)))
    assert len(canvas._gesture_points) == 2
    canvas.mouseReleaseEvent(release(*canvas_xy(30, 52)))
    assert accent_pixels(canvas) == quiet


def test_the_other_tools_are_left_as_they_were(canvas, strokes):
    """A brush stroke still paints — the new branches did not swallow it."""
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 4
    drag(canvas, [(32, 8), (34, 8)])
    assert int(canvas.mask[8, 32]) == 255
    assert strokes == {"started": 1, "finished": 1}
    assert canvas.last_edit["kind"] == "paint"


# ===========================================================================
# Through the screen: history and ledger
# ===========================================================================

@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path: Path):
    """A screen on a one-image folder, holding the merged-object field."""
    folder = tmp_path / "field"
    folder.mkdir()
    imageio.imwrite(folder / "img_00.tif", field_image())
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    widget._canvas.set_image_and_mask(field_image(), merged_mask())
    widget._history.clear()
    widget._history.push(widget._canvas.mask)
    widget._canvas.resize(CANVAS_W, CANVAS_H)
    widget._canvas.refresh()
    assert widget._canvas.pixmap().width() == PIXMAP_N
    return widget


def test_the_screen_records_a_divide_and_can_undo_it(screen):
    """One divide: one ledger entry, one undo step back to the merged pair."""
    before = screen._canvas.mask.copy()
    screen._set_mode(MODE_DIVIDE)
    drag(screen._canvas, [(30, 16), (30, 48)])

    assert len(screen._canvas.mask[screen._canvas.mask > 0]) > 0
    assert len(counts_by_label(screen._canvas.mask)) == 4
    kinds = [edit.kind for edit in screen._log.edits]
    assert kinds == ["divide"], f"ledger says {kinds}"
    assert screen._log.edits[0].n_changed > 0

    screen._on_undo()
    assert np.array_equal(screen._canvas.mask, before)


def test_the_screen_records_a_draw(screen):
    """One traced outline: one ``draw`` entry naming the id it made."""
    screen._set_mode(MODE_DRAW)
    drag(screen._canvas, OUTLINE)

    edits = screen._log.edits
    assert [edit.kind for edit in edits] == ["draw"]
    assert edits[0].target == int(screen._canvas.mask[50, 18])
    assert edits[0].n_changed >= 100


def test_a_missed_gesture_writes_no_ledger_entry(screen):
    """A divide that separated nothing is not a correction."""
    screen._set_mode(MODE_DIVIDE)
    drag(screen._canvas, [(30, 16), (30, 30)])
    assert list(screen._log.edits) == []


def test_the_toolbar_button_puts_the_canvas_in_draw_mode(screen):
    """The whole path: press Draw on the row, trace, and read the object off.

    The mode constants are only worth anything if the row that offers them
    reaches the canvas, so this drives the button rather than assigning
    ``canvas.mode`` — the one thing the other tests here take for granted.
    """
    assert MODE_DRAW in screen._mode_buttons, "draw has no control on the row"
    screen._mode_buttons[MODE_DRAW].click()
    assert screen._canvas.mode == MODE_DRAW

    drag(screen._canvas, OUTLINE)
    assert int(screen._canvas.mask[50, 18]) not in (0, 3, 7, 9)


def test_the_toolbar_button_puts_the_canvas_in_divide_mode(screen):
    """The same, for the cut: press Divide, drag across the waist, count."""
    assert MODE_DIVIDE in screen._mode_buttons, "divide has no control"
    screen._mode_buttons[MODE_DIVIDE].click()
    assert screen._canvas.mode == MODE_DIVIDE

    drag(screen._canvas, [(30, 16), (30, 48)])
    assert len(counts_by_label(screen._canvas.mask)) == 4


# ===========================================================================
# The edges of the two operations
# ===========================================================================

def test_a_click_with_no_drag_draws_nothing(canvas, strokes):
    """One point encloses nothing, and must not raise on the way to saying so."""
    before = canvas.mask.copy()
    canvas.mode = MODE_DRAW
    canvas.mousePressEvent(press(*canvas_xy(30, 50)))
    canvas.mouseReleaseEvent(release(*canvas_xy(30, 50)))

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}


def test_a_divide_that_never_left_one_pixel_cuts_nothing(canvas, strokes):
    """A drag too small to leave the pixel it started in is not a cut.

    Both ends land on the same image pixel, so the segment has no length
    and the distance-to-segment maths would divide by it.
    """
    before = canvas.mask.copy()
    canvas.mode = MODE_DIVIDE
    canvas.mousePressEvent(press(*canvas_xy(30, 32)))
    canvas.mouseMoveEvent(move(*canvas_xy(30.4, 32.4)))
    canvas.mouseReleaseEvent(release(*canvas_xy(30.4, 32.4)))

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}


def test_drawing_on_a_brush_painted_mask_widens_it_rather_than_wrapping(
        qtbot, qt_theme_applied):
    """A uint8 mask that gains label 256 comes back as uint16, whole.

    Every brush stroke writes 255, so a mask that has only been painted is
    uint8 and full. Keeping uint8 for the object drawn after it would wrap
    256 round to 0 and the new object would BE the background — it would
    disappear on the way out of the fill.
    """
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(CANVAS_W, CANVAS_H)
    painted = np.zeros((IMG_N, IMG_N), dtype=np.uint8)
    painted[8:16, 8:16] = 255
    widget.set_image_and_mask(field_image(), painted)
    widget.mode = MODE_DRAW
    drag(widget, OUTLINE)

    assert widget.mask.dtype == np.uint16
    assert int(widget.mask[50, 18]) == 256
    assert int((widget.mask == 255).sum()) == 64, "the painted object moved"


def test_a_mask_at_the_top_of_uint16_refuses_a_new_object():
    """Past 65535 there is no id left to give, and the mask cannot be saved.

    Refused by name rather than wrapped: label 65536 stored in uint16 is 0,
    which is background, so the object would be silently lost instead.
    """
    full = np.zeros((16, 16), dtype=np.uint16)
    full[0, 0] = 65535
    with pytest.raises(ValueError, match="65536"):
        engine.fill_polygon(full, [(4, 4), (10, 4), (10, 10), (4, 10)])


def test_an_outline_entirely_off_the_image_makes_no_object():
    """A polygon with real area but no pixels on the field fills nothing."""
    mask = np.zeros((16, 16), dtype=np.uint16)
    out, made = engine.fill_polygon(mask, [(-30, -30), (-20, -30), (-20, -20)])
    assert made == 0
    assert not out.any()


def test_a_cut_aimed_off_the_image_touches_nothing():
    """A segment that misses the field entirely cuts nothing on it."""
    band = engine._segment_band((16, 16), (-40, -40), (-30, -30), 1.5)
    assert not band.any()

    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:12, 4:12] = 2
    out, splits = engine.divide_object(mask, (-40, -40), (-30, -30))
    assert splits == []
    assert np.array_equal(out, mask)


def test_a_drawn_object_can_be_given_an_id_of_its_own():
    """``label_value`` overrides the id, for a caller that is renumbering."""
    mask = np.zeros((32, 32), dtype=np.uint16)
    out, made = engine.fill_polygon(
        mask, [(4, 4), (20, 4), (20, 20), (4, 20)], label_value=42)
    assert made == 42
    assert int((out == 42).sum()) > 100


def test_a_click_with_no_drag_cuts_nothing(canvas, strokes):
    """One point is not a line, so a click in divide mode divides nothing."""
    before = canvas.mask.copy()
    canvas.mode = MODE_DIVIDE
    canvas.mousePressEvent(press(*canvas_xy(30, 32)))
    canvas.mouseReleaseEvent(release(*canvas_xy(30, 32)))

    assert np.array_equal(canvas.mask, before)
    assert strokes == {"started": 0, "finished": 0}


def test_a_field_that_goes_away_mid_gesture_takes_the_gesture_with_it(canvas):
    """Releasing onto no mask must not raise.

    Reachable without any misuse: the arrow keys move to the next field
    from anywhere, and a field that fails to load clears the canvas — see
    ``_handle_load_failure`` — which can happen between the press and the
    release of a traced outline.
    """
    canvas.mode = MODE_DRAW
    canvas.mousePressEvent(press(*canvas_xy(12, 44)))
    canvas.mouseMoveEvent(move(*canvas_xy(24, 44)))
    canvas.mask = None

    canvas.mouseReleaseEvent(release(*canvas_xy(24, 56)))
    assert canvas.mask is None


def test_a_gesture_does_not_follow_the_user_to_the_next_field(canvas, strokes):
    """An outline traced on one field must not land on the next one.

    The arrow keys move to the next field from anywhere, including the
    middle of a drag, and the field that arrives is a different image with
    a different mask. Carrying the collected points across would fill an
    outline nobody traced onto a mask nobody was looking at.
    """
    canvas.mode = MODE_DRAW
    canvas.mousePressEvent(press(*canvas_xy(24, 44)))
    for point in OUTLINE[1:]:
        canvas.mouseMoveEvent(move(*canvas_xy(*point)))

    next_field = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    next_field[20:28, 20:28] = 4
    canvas.set_image_and_mask(field_image(), next_field)
    canvas.mouseReleaseEvent(release(*canvas_xy(12, 56)))

    assert counts_by_label(canvas.mask) == {4: 64}
    assert strokes == {"started": 0, "finished": 0}
