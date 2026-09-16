"""A press-and-drag with the live magnifier merges what the cursor passes over.

Item 417, parts 5 and 6, on item 407's live magnifier. What has to hold:

* pressing and pulling adds objects along the path WHILE the button is down,
  and the objects the cursor passes over become ONE object -- one ledger
  entry and one undo step for the whole stroke;
* "Only objects touching the mouse" leaves out an object the cursor never
  passed over, and "All objects in the zoom area" keeps it;
* the pieces one object leaves in overlapping boxes are one object, on the
  pixels where the object lies;
* a press that does not move is still item 407's click, and the right button
  still sweeps;
* the model never runs on the GUI thread during a drag, and the event loop
  keeps turning while the stroke waits for its last box.

The first half tests :mod:`spacr.qt._magnifier_drag` on plain arrays, which
is where every rule lives. The screen tests use item 407's coded field and
stand-in model: a pixel's value is ``y * 64 + x``, and the stub labels
objects named in IMAGE pixels wherever the decoded coordinates fall, so a
box that was wrong by a pixel would move an object and fail the test.

The canvas is pinned to 600x400 with a 64x64 image, so image pixel (x, y)
has its centre at canvas point (100 + (x + .5) * 6.25, (y + .5) * 6.25), and
one image pixel of mouse travel is 6.25 widget pixels.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel

from spacr.qt import _magnifier_drag as drag
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

IMG_N = 64
SHAPE = (IMG_N, IMG_N)
CANVAS_W, CANVAS_H = 600, 400
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2
SCALE = PIXMAP_N / IMG_N
SIZE = 32


def paint(rects: dict, shape=SHAPE, dtype=np.int32) -> np.ndarray:
    """A label image with ``{id: (x0, y0, x1, y1)}`` painted in, in order."""
    out = np.zeros(shape, dtype=dtype)
    for value, (x0, y0, x1, y1) in rects.items():
        out[y0:y1, x0:x1] = value
    return out


# ---------------------------------------------------------------------------
# The stroke's arithmetic, on plain arrays
# ---------------------------------------------------------------------------

def feed(stroke, key, rects: dict, box) -> bool:
    """Ask for a frame under ``key`` and hand in what a model finds in ``box``."""
    x0, y0, x1, y1 = box
    stroke.expect(key)
    return stroke.deliver(key, paint(rects)[y0:y1, x0:x1], box)


def placed(found) -> np.ndarray:
    """A stroke's outcome laid on an empty image, to compare whole images."""
    out = np.zeros(SHAPE, dtype=np.int32)
    height, width = found.labels.shape
    x0, y0 = found.origin
    out[y0:y0 + height, x0:x0 + width] = found.labels
    return out


def test_the_line_between_two_mouse_events_skips_no_pixel():
    np.testing.assert_array_equal(
        drag._line_pixels((0, 0), (3, 1)), [[0, 0], [1, 0], [2, 1], [3, 1]])
    np.testing.assert_array_equal(
        drag._line_pixels((4, 0), (4, -3)),
        [[4, 0], [4, -1], [4, -2], [4, -3]])
    np.testing.assert_array_equal(drag._line_pixels((5, 7), (5, 7)), [[5, 7]])


def test_a_frame_is_wanted_every_quarter_box_and_all_along_a_jump():
    assert drag._frame_step(32) == 8
    assert drag._frame_step(2) == 1

    stroke = drag._DragStroke(SHAPE, (10, 10), step=4)
    assert stroke.extend((10, 10)) == []
    assert stroke.extend((13, 11)) == []
    assert stroke.extend((14, 11)) == [(14, 11)]
    assert stroke.extend((26, 11)) == [(18, 11), (22, 11), (26, 11)], (
        "a jump three steps long must want three frames, or its middle "
        "goes unseen")
    assert stroke.extend((99, -5)), "a jump off the image still wants frames"
    assert stroke._last == (63, 0), "points are clipped into the image"

    whole = drag._DragStroke(SHAPE, (10, 10), step=0)
    assert whole.extend((60, 60)) == []


def test_the_pieces_the_path_passes_over_become_one_object():
    stroke = drag._DragStroke(SHAPE, (22, 32), step=8)
    assert stroke.outcome() is None, "nothing to say before a frame"
    stroke.extend((36, 32))
    cells = {1: (20, 28, 30, 36), 2: (30, 28, 40, 36)}
    assert feed(stroke, "box", cells, (10, 20, 50, 44))
    assert stroke.dirty

    found = stroke.outcome()
    assert not stroke.dirty
    np.testing.assert_array_equal(placed(found), paint({1: (20, 28, 40, 36)}))
    assert (found.merged, found.objects, found.frames) == (2, 1, 1)
    assert found.origin == (10, 20)


@pytest.mark.parametrize("keep", [True, False])
def test_an_object_the_path_never_crossed_is_added_only_when_kept(keep):
    stroke = drag._DragStroke(SHAPE, (22, 32), step=8, keep_untouched=keep)
    stroke.extend((28, 32))
    feed(stroke, "box", {4: (20, 28, 30, 36), 9: (36, 20, 42, 26)},
         (14, 16, 46, 48))

    found = stroke.outcome()
    expected = {1: (20, 28, 30, 36)}
    if keep:
        expected[2] = (36, 20, 42, 26)
    np.testing.assert_array_equal(placed(found), paint(expected))
    assert found.objects == len(expected)


def test_the_pieces_of_one_object_from_overlapping_boxes_are_one_object():
    """Longer than the box, so every box cuts it; the path runs below it."""
    long_cell = {5: (10, 30, 58, 34)}
    stroke = drag._DragStroke(SHAPE, (14, 44), step=8)
    for x in range(14, 55, 8):
        stroke.extend((x, 44))
        feed(stroke, x, long_cell, engine._magnifier_box(SHAPE, x, 40, SIZE))

    found = stroke.outcome()
    np.testing.assert_array_equal(placed(found), paint({1: long_cell[5]}))
    assert (found.merged, found.objects, found.frames) == (0, 1, 6)


def test_a_piece_that_covers_two_objects_joins_them_into_one():
    stroke = drag._DragStroke(SHAPE, (2, 2), step=8)
    left, right = (10, 10, 20, 20), (30, 10, 40, 20)
    feed(stroke, "first", {1: left, 2: right}, (0, 0, 64, 32))
    feed(stroke, "again", {1: left, 2: right}, (0, 0, 64, 32))
    assert stroke.outcome().objects == 2
    feed(stroke, "bridge", {7: (10, 10, 40, 20)}, (0, 0, 64, 32))

    found = stroke.outcome()
    np.testing.assert_array_equal(placed(found), paint({1: (10, 10, 40, 20)}))
    assert found.objects == 1


def test_a_neighbour_that_shares_only_a_sliver_stays_its_own_object():
    """Rule 2 asks for half; rule 4 leaves the shared column where it was."""
    stroke = drag._DragStroke(SHAPE, (2, 2), step=8)
    feed(stroke, "one", {1: (20, 20, 30, 30)}, (0, 0, 64, 64))
    feed(stroke, "two", {1: (29, 20, 40, 30)}, (0, 0, 64, 64))

    np.testing.assert_array_equal(
        placed(stroke.outcome()),
        paint({1: (20, 20, 30, 30), 2: (30, 20, 40, 30)}))


def test_the_path_can_reach_an_object_after_its_frame_arrived():
    stroke = drag._DragStroke(SHAPE, (5, 50), step=8, keep_untouched=False)
    stroke.extend((6, 50))
    feed(stroke, "box", {3: (20, 28, 30, 36)}, (0, 16, 64, 64))
    assert stroke.outcome().objects == 0

    stroke.extend((10, 50))
    assert not stroke.dirty, "background under the path changes nothing"
    stroke.extend((25, 32))
    assert stroke.dirty

    found = stroke.outcome()
    np.testing.assert_array_equal(placed(found), paint({1: (20, 28, 30, 36)}))
    assert found.merged == 1


def test_frames_are_waited_for_by_key_and_the_stroke_is_ready_when_all_are_in():
    stroke = drag._DragStroke(SHAPE, (30, 30), step=8)
    assert not stroke.deliver("never asked", np.ones((4, 4)), (0, 0, 4, 4))
    stroke.expect("a")
    stroke.expect("b")
    assert stroke.waiting() == 2
    stroke.drop("b")
    stroke.release()
    assert stroke.released and not stroke.ready()

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[20:24, 20:24] = 3            # no object 2: a gap in the ids
    assert stroke.deliver("a", labels, (14, 14, 46, 46))
    assert stroke.ready() and stroke.waiting() == 0

    found = stroke.outcome()
    assert (found.objects, found.frames, found.merged) == (2, 1, 0)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def canvas_xy(img_x: int, img_y: int) -> tuple:
    """The canvas point at the centre of image pixel (img_x, img_y)."""
    return (MARGIN_X + (img_x + 0.5) * SCALE, (img_y + 0.5) * SCALE)


def _mouse(kind, img_x, img_y, button, buttons):
    pos = QPointF(*canvas_xy(img_x, img_y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def hover(screen, img_x, img_y):
    screen._canvas.mouseMoveEvent(_mouse(
        QEvent.Type.MouseMove, img_x, img_y, Qt.NoButton, Qt.NoButton))


def press(screen, img_x, img_y, button=Qt.LeftButton):
    screen._canvas.mousePressEvent(_mouse(
        QEvent.Type.MouseButtonPress, img_x, img_y, button, button))


def pull(screen, img_x, img_y, button=Qt.LeftButton):
    screen._canvas.mouseMoveEvent(_mouse(
        QEvent.Type.MouseMove, img_x, img_y, Qt.NoButton, button))


def release(screen, img_x, img_y, button=Qt.LeftButton):
    screen._canvas.mouseReleaseEvent(_mouse(
        QEvent.Type.MouseButtonRelease, img_x, img_y, button, Qt.NoButton))


def drag_along(screen, xs, y):
    """Press at the first x, pull through the rest along row y, release."""
    press(screen, xs[0], y)
    for x in xs[1:]:
        pull(screen, x, y)
    release(screen, xs[-1], y)


def coded_field() -> np.ndarray:
    """A 64x64 uint16 field whose every pixel says where it is."""
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    return (yy * IMG_N + xx).astype(np.uint16)


class CodedStub:
    """A stand-in model that returns known objects, wherever it is asked.

    :param objects: ``{label: (x0, y0, x1, y1)}`` in IMAGE pixels.
    :param delay: seconds each call takes.
    """

    def __init__(self, objects: dict, delay: float = 0.0):
        self.objects = dict(objects)
        self.delay = float(delay)
        self.gate = None
        self.fail = set()
        self.calls = []
        self.threads = []

    def __call__(self, request):
        self.threads.append(threading.get_ident())
        self.calls.append(request)
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.delay:
            time.sleep(self.delay)
        if tuple(request.box) in self.fail:
            raise RuntimeError("this box cannot be segmented")
        crop = request.crop.astype(np.int64)
        xs, ys = crop % IMG_N, crop // IMG_N
        labels = np.zeros(crop.shape, dtype=np.int32)
        for value, (x0, y0, x1, y1) in self.objects.items():
            labels[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = value
        return labels


@pytest.fixture
def fields(tmp_path: Path) -> Path:
    folder = tmp_path / "fields"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif", coded_field())
    imageio.imwrite(folder / "b.tif", coded_field())
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, fields: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(fields))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    assert made._canvas.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    made._min_area.setValue(0)
    made._mag_size.setValue(SIZE)
    assert QApplication.startDragDistance() > SCALE, (
        "one image pixel of travel must stay under the drag distance")
    yield made
    made._magnifier.close()
    made.close_folded()


def switch_on(screen, stub, save="zoom") -> None:
    screen._magnifier.segment = stub
    screen._mag_save.setCurrentIndex(screen._mag_save.findData(save))
    screen._btn_magnifier.setChecked(True)
    assert screen._magnifier.enabled
    assert screen._magnifier.save_mode == save


def settle_on(qtbot, screen, img_x, img_y) -> None:
    """Hover and wait until the box shows that region's objects."""
    hover(screen, img_x, img_y)
    magnifier = screen._magnifier
    qtbot.waitUntil(
        lambda: magnifier._shown is not None and not magnifier.updating(),
        timeout=10_000)


def wait_for_edits(qtbot, screen, n=1) -> None:
    qtbot.waitUntil(
        lambda: screen._log.counts().get("magnifier", 0) == n,
        timeout=10_000)


def wait_until_done(qtbot, screen) -> None:
    magnifier = screen._magnifier
    qtbot.waitUntil(
        lambda: magnifier._stroke is None and magnifier._worker.idle(),
        timeout=10_000)


def test_the_save_mode_is_a_choice_on_the_magnifier_card(screen):
    box = screen._mag_save
    assert [box.itemData(i) for i in range(box.count())] == [
        "zoom", "touching"]
    assert box.currentData() == "zoom"
    assert screen._magnifier.save_mode == "zoom", "407's behaviour is the default"
    # The panel moves a field's tooltip onto the label that names it.
    names = [label for label in screen.findChildren(QLabel)
             if label.text() == "Objects added"]
    assert names, "the save mode has no label on the card"
    assert box.toolTip() or any(label.toolTip() for label in names)
    box.setCurrentIndex(box.findData("touching"))
    assert screen._magnifier.save_mode == "touching"


def test_a_drag_across_two_adjacent_cells_adds_one_object_as_one_undo_step(
        qtbot, screen):
    stub = CodedStub({1: (20, 28, 30, 36), 2: (30, 28, 40, 36)})
    switch_on(screen, stub)
    settle_on(qtbot, screen, 22, 32)

    drag_along(screen, list(range(22, 37)), 32)
    wait_for_edits(qtbot, screen)
    wait_until_done(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (20, 28, 40, 36)}))
    edit = screen._log.edits[-1]
    assert edit.target == [1]
    assert edit.detail["drag"] is True
    assert edit.detail["merged"] >= 2
    assert screen._log.counts().get("magnifier") == 1

    screen._on_undo()
    assert not screen._canvas.mask.any()
    assert not screen._history.can_undo(), "the stroke was more than one step"


def test_the_drag_applies_its_objects_while_the_button_is_down(qtbot, screen):
    stub = CodedStub({1: (20, 28, 30, 36)})
    switch_on(screen, stub)
    settle_on(qtbot, screen, 22, 32)

    press(screen, 22, 32)
    for x in range(23, 29):
        pull(screen, x, 32)
    qtbot.waitUntil(lambda: screen._canvas.mask.any(), timeout=10_000)
    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (20, 28, 30, 36)}))
    assert screen._log.counts().get("magnifier") is None
    assert not screen._history.can_undo(), "a preview is not an undo step"

    release(screen, 28, 32)
    wait_for_edits(qtbot, screen)
    assert screen._history.can_undo()


@pytest.mark.parametrize("save, expected", [
    ("zoom", {1: (20, 28, 30, 36), 2: (36, 20, 42, 26)}),
    ("touching", {1: (20, 28, 30, 36)}),
])
def test_the_save_mode_decides_whether_an_object_off_the_path_is_added(
        qtbot, screen, save, expected):
    stub = CodedStub({1: (20, 28, 30, 36), 3: (36, 20, 42, 26)})
    switch_on(screen, stub, save)
    settle_on(qtbot, screen, 22, 32)

    drag_along(screen, list(range(22, 31)), 32)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(screen._canvas.mask, paint(expected))
    assert screen._log.edits[-1].detail["save"] == save


@pytest.mark.parametrize("save, row", [("zoom", 40), ("touching", 32)])
def test_the_pieces_one_object_leaves_in_successive_boxes_become_one_object(
        qtbot, screen, save, row):
    """The object is longer than the box, so every box along the drag cuts
    it; with the border option off each box offers its piece, and the pieces
    go in as one object on the pixels where the object lies -- whether the
    path runs along the object (touching) or beside it (zoom)."""
    stub = CodedStub({4: (10, 30, 58, 34)})
    switch_on(screen, stub, save)
    screen._mag_exclude_border.setChecked(False)
    settle_on(qtbot, screen, 14, row)

    drag_along(screen, list(range(14, 55)), row)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (10, 30, 58, 34)}))
    assert screen._log.edits[-1].detail["frames"] >= 6


def test_a_press_that_does_not_move_is_still_a_click(qtbot, screen):
    stub = CodedStub({7: (20, 20, 26, 25), 9: (32, 34, 40, 40)})
    switch_on(screen, stub)
    settle_on(qtbot, screen, 30, 30)

    press(screen, 30, 30)
    pull(screen, 31, 30)             # one image pixel: under the drag distance
    release(screen, 31, 30)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask,
        paint({1: (20, 20, 26, 25), 2: (32, 34, 40, 40)}))
    edit = screen._log.edits[-1]
    assert edit.target == [1, 2]
    assert "drag" not in edit.detail


def test_with_only_touching_a_click_adds_just_the_object_under_the_cursor(
        qtbot, screen):
    stub = CodedStub({7: (26, 26, 34, 34), 9: (36, 36, 42, 42)})
    switch_on(screen, stub, "touching")
    settle_on(qtbot, screen, 30, 30)

    press(screen, 30, 30)
    release(screen, 30, 30)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (26, 26, 34, 34)}))


def test_a_drag_never_waits_for_the_model_on_the_gui_thread(qtbot, screen):
    """Each box takes half a second; no move may wait for one, and the event
    loop keeps turning while the released stroke waits for its last box."""
    stub = CodedStub({1: (20, 28, 30, 36), 2: (30, 28, 40, 36)}, delay=0.5)
    gui = threading.get_ident()
    switch_on(screen, stub)
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start()
    try:
        slowest = 0.0
        press(screen, 18, 32)
        for x in range(19, 45):
            started = time.perf_counter()
            pull(screen, x, 32)
            screen._canvas.repaint()
            slowest = max(slowest, time.perf_counter() - started)
        release(screen, 44, 32)
        assert slowest < 0.4, f"a move during the drag took {slowest:.3f}s"

        before = len(ticks)
        wait_for_edits(qtbot, screen)
        assert len(ticks) - before >= 5, (
            "the event loop stood still while the drag waited for its boxes")
    finally:
        timer.stop()

    assert stub.threads and gui not in stub.threads
    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (20, 28, 40, 36)}))
    assert screen._log.counts().get("magnifier") == 1


def test_under_whole_image_a_drag_merges_the_objects_it_passes_over(
        qtbot, screen):
    stub = CodedStub({1: (20, 28, 30, 36), 2: (30, 28, 40, 36),
                      3: (50, 50, 56, 56)})
    screen._magnifier.segment = stub
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    screen._btn_magnifier.setChecked(True)
    magnifier = screen._magnifier
    qtbot.waitUntil(
        lambda: magnifier._image_result is not None
        and not magnifier.updating(), timeout=15_000)

    drag_along(screen, list(range(24, 37)), 32)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (20, 28, 40, 36)}))
    assert len(stub.calls) == 1, "a whole-image drag asked the model again"
    assert screen._log.edits[-1].detail["scope"] == "image"


def test_the_right_button_still_sweeps_objects_away(qtbot, screen):
    switch_on(screen, CodedStub({}))
    mask = paint({3: (20, 28, 30, 36), 5: (34, 28, 40, 36)}, dtype=np.uint8)
    screen._canvas.mask = mask
    screen._history.push(mask.copy())

    press(screen, 24, 32, Qt.RightButton)
    for x in range(25, 38):
        pull(screen, x, 32, Qt.RightButton)
    release(screen, 37, 32, Qt.RightButton)

    assert not screen._canvas.mask.any()
    assert screen._log.counts().get("sweep_delete") == 1
    assert screen._magnifier._stroke is None


def test_moving_to_another_field_mid_drag_adds_nothing(qtbot, screen):
    stub = CodedStub({1: (20, 28, 30, 36)})
    stub.gate = threading.Event()
    switch_on(screen, stub)
    try:
        press(screen, 22, 32)
        for x in range(23, 31):
            pull(screen, x, 32)
        screen._on_next()
    finally:
        stub.gate.set()
    release(screen, 30, 32)
    wait_until_done(qtbot, screen)
    qtbot.wait(100)

    assert not screen._canvas.mask.any()
    assert screen._log.counts().get("magnifier") is None


def test_a_box_that_cannot_be_segmented_does_not_hold_the_drag_up(
        qtbot, screen):
    stub = CodedStub({1: (20, 28, 30, 36)})
    stub.fail = {engine._magnifier_box(SHAPE, 30, 32, SIZE)}
    switch_on(screen, stub)
    settle_on(qtbot, screen, 22, 32)

    drag_along(screen, list(range(22, 31)), 32)
    wait_for_edits(qtbot, screen)

    np.testing.assert_array_equal(
        screen._canvas.mask, paint({1: (20, 28, 30, 36)}))


def test_a_drag_over_background_adds_nothing_and_says_so(qtbot, screen):
    stub = CodedStub({9: (50, 4, 58, 10)})
    switch_on(screen, stub, "touching")
    settle_on(qtbot, screen, 10, 40)

    drag_along(screen, list(range(10, 20)), 40)
    wait_until_done(qtbot, screen)

    assert not screen._canvas.mask.any()
    assert "nothing was added" in screen._status_label.text()
    assert not screen._history.can_undo()
