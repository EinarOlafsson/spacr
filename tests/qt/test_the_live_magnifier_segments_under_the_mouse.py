"""The live magnifier in Make Masks segments what is under the mouse.

A toggle in the tool row puts a box under the mouse that shows the region
around it magnified, with the objects a model finds there outlined live; a
click puts those objects into the mask. What has to hold for that to be a
tool rather than a demo:

* a click commits EXACTLY the objects the box outlines, at the right image
  pixels -- including where the box is clipped by the image border, which is
  where an off-by-one between crop and image coordinates would show;
* new ids never collide with ids already in the mask, and an object landing
  on an existing one follows the Overlap rule (clip by default);
* one click is one undo step, and what a click added survives the toggle
  going off, a save, and a trip to the next field and back;
* the model NEVER runs on the GUI thread, and the worker keeps only the
  newest request, so a moving mouse cannot queue up stale regions;
* with no model installed, the classical mode still finds objects.

Most tests use a STUB model that reads where it is from the pixels it is
given: the field is coded so a pixel's value is ``y * 64 + x``, and the stub
labels an object wherever the decoded coordinates fall inside a rectangle
named in IMAGE pixels. It never looks at the request's box, so a box that
was wrong by a pixel would move the committed object and fail the test
rather than cancel out.

The canvas is pinned to 600x400 with a 64x64 image, so the pixmap is 400x400
centred with a 100 px margin, and image pixel (x, y) has its centre at canvas
point (100 + (x + .5) * 6.25, (y + .5) * 6.25).
"""
from __future__ import annotations

import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent

from spacr.curation import CurationLog
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.theme import active_palette

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2
SCALE = PIXMAP_N / IMG_N
SIZE = 32


def canvas_xy(img_x: int, img_y: int) -> tuple:
    """The canvas point at the centre of image pixel (img_x, img_y)."""
    return (MARGIN_X + (img_x + 0.5) * SCALE, (img_y + 0.5) * SCALE)


def _mouse(kind, x, y, button=Qt.NoButton, buttons=Qt.NoButton):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def hover(screen, img_x, img_y):
    screen._canvas.mouseMoveEvent(
        _mouse(QEvent.Type.MouseMove, *canvas_xy(img_x, img_y)))


def click(screen, img_x, img_y):
    x, y = canvas_xy(img_x, img_y)
    screen._canvas.mousePressEvent(_mouse(
        QEvent.Type.MouseButtonPress, x, y, Qt.LeftButton, Qt.LeftButton))
    screen._canvas.mouseReleaseEvent(_mouse(
        QEvent.Type.MouseButtonRelease, x, y, Qt.LeftButton, Qt.NoButton))


def wheel(screen, img_x, img_y, delta=120):
    pos = QPointF(*canvas_xy(img_x, img_y))
    screen._canvas.wheelEvent(QWheelEvent(
        pos, pos, QPoint(0, 0), QPoint(0, delta), Qt.NoButton,
        Qt.NoModifier, Qt.NoScrollPhase, False))


def coded_field() -> np.ndarray:
    """A 64x64 uint16 field whose every pixel says where it is."""
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    return (yy * IMG_N + xx).astype(np.uint16)


class CodedStub:
    """A stand-in model that returns known objects, wherever it is asked.

    :param objects: ``{label: (x0, y0, x1, y1)}`` in IMAGE pixels, exclusive
        ends. Each is labelled where the crop's decoded pixels fall inside it.
    """

    def __init__(self, objects: dict, delay: float = 0.0):
        self.objects = dict(objects)
        self.delay = float(delay)
        self.gate = None
        self.calls = []
        self.threads = []

    def __call__(self, request):
        self.threads.append(threading.get_ident())
        self.calls.append(request)
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.delay:
            time.sleep(self.delay)
        crop = request.crop.astype(np.int64)
        xs, ys = crop % IMG_N, crop // IMG_N
        labels = np.zeros(crop.shape, dtype=np.int32)
        for value, (x0, y0, x1, y1) in self.objects.items():
            labels[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = value
        return labels


def rect_mask(shape, rects: dict, dtype=np.uint8) -> np.ndarray:
    """A label image with ``{id: (x0, y0, x1, y1)}`` painted in."""
    out = np.zeros(shape, dtype=dtype)
    for value, (x0, y0, x1, y1) in rects.items():
        out[y0:y1, x0:x1] = value
    return out


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
    yield made
    made._magnifier.close()
    made.close_folded()


def switch_on(screen, stub) -> None:
    screen._magnifier.segment = stub
    screen._btn_magnifier.setChecked(True)
    assert screen._magnifier.enabled


def wait_for_result(qtbot, screen) -> None:
    magnifier = screen._magnifier
    qtbot.waitUntil(
        lambda: magnifier._shown is not None and not magnifier.updating(),
        timeout=10_000)


# ---------------------------------------------------------------------------
# A click commits exactly what the box outlines
# ---------------------------------------------------------------------------

def test_a_click_commits_exactly_the_objects_the_model_found(qtbot, screen):
    stub = CodedStub({7: (20, 20, 26, 25), 9: (32, 34, 40, 40)})
    switch_on(screen, stub)

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    request = stub.calls[-1]
    assert request.box == (14, 14, 46, 46)
    np.testing.assert_array_equal(
        request.crop, screen._canvas.image[14:46, 14:46])

    click(screen, 30, 30)
    expected = rect_mask((IMG_N, IMG_N),
                         {1: (20, 20, 26, 25), 2: (32, 34, 40, 40)})
    np.testing.assert_array_equal(screen._canvas.mask, expected)
    assert screen._log.counts().get("magnifier") == 1
    assert screen._log.edits[-1].target == [1, 2]


def test_a_box_clipped_by_the_image_border_commits_at_the_right_pixels(
        qtbot, screen):
    """The crop starts at the image corner, not at the cursor minus half.

    The box round pixel (2, 3) would start at (-14, -13), and is clipped to
    (0, 0, 18, 19). An object touching the IMAGE border is whole and is
    offered; one touching the box's right edge, inside the image, is a piece
    of something the box cut and is not.
    """
    stub = CodedStub({1: (0, 0, 5, 4), 2: (10, 12, 18, 16)})
    switch_on(screen, stub)

    hover(screen, 2, 3)
    wait_for_result(qtbot, screen)
    assert stub.calls[-1].box == (0, 0, 18, 19)
    assert stub.calls[-1].crop.shape == (19, 18)

    click(screen, 2, 3)
    expected = rect_mask((IMG_N, IMG_N), {1: (0, 0, 5, 4)})
    np.testing.assert_array_equal(screen._canvas.mask, expected)


def test_the_far_corner_clips_the_same_way(qtbot, screen):
    stub = CodedStub({5: (58, 60, 64, 64)})
    switch_on(screen, stub)

    hover(screen, 63, 63)
    wait_for_result(qtbot, screen)
    assert stub.calls[-1].box == (47, 47, 64, 64)
    click(screen, 63, 63)
    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (58, 60, 64, 64)}))


def test_new_ids_do_not_collide_with_the_ids_already_in_the_mask(
        qtbot, screen):
    """Past the top id, never into a gap -- and wide enough to hold it."""
    existing = rect_mask((IMG_N, IMG_N), {3: (2, 2, 6, 6), 255: (50, 2, 60, 8)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    stub = CodedStub({1: (20, 20, 26, 25), 2: (32, 34, 40, 40)})
    switch_on(screen, stub)

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)

    mask = screen._canvas.mask
    assert mask.dtype == np.uint16, "id 256 cannot be held in uint8"
    assert set(np.unique(mask[20:25, 20:26])) == {256}
    assert set(np.unique(mask[34:40, 32:40])) == {257}
    assert np.array_equal(mask[2:6, 2:6], np.full((4, 4), 3))
    assert np.array_equal(mask[2:8, 50:60], np.full((6, 10), 255))


def test_an_object_on_an_existing_one_is_clipped_to_background(qtbot, screen):
    existing = rect_mask((IMG_N, IMG_N), {1: (24, 22, 30, 30)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    stub = CodedStub({4: (20, 20, 28, 26)})
    switch_on(screen, stub)
    assert screen._mag_overlap.currentData() == "clip"

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)

    mask = screen._canvas.mask
    np.testing.assert_array_equal(mask[22:30, 24:30], 1)
    new = (mask == 2)
    wanted = rect_mask((IMG_N, IMG_N), {1: (20, 20, 28, 26)}).astype(bool)
    wanted &= existing == 0
    np.testing.assert_array_equal(new, wanted)


def test_the_overlap_rule_is_a_choice_on_the_panel(qtbot, screen):
    """Skip leaves out the object that touches; the free one still goes in."""
    existing = rect_mask((IMG_N, IMG_N), {1: (24, 22, 30, 30)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("skip"))
    stub = CodedStub({4: (20, 20, 28, 26), 6: (32, 34, 40, 40)})
    switch_on(screen, stub)

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)

    mask = screen._canvas.mask
    np.testing.assert_array_equal(mask[20:22, 20:24], 0)
    assert set(np.unique(mask[34:40, 32:40])) == {2}


def test_the_paste_rules_at_the_engine():
    """Clip keeps the largest piece; replace takes pixels; bad rule refused."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 9:11] = 5                            # a bar through the middle
    labels = np.zeros((10, 16), dtype=np.int32)
    labels[2:8, 0:14] = 3                        # left 6 px, right 4 px of it

    out, added = engine._paste_region_objects(mask, labels, (3, 4))
    assert added == [6]
    assert np.array_equal(np.argwhere(out == 6)[:, 1].min(), 3)
    assert np.argwhere(out == 6)[:, 1].max() == 8, "only the larger piece"
    assert (out[:, 9:11] == 5).all()

    out, added = engine._paste_region_objects(mask, labels, (3, 4),
                                              overlap="skip")
    assert added == [] and np.array_equal(out, mask)

    out, added = engine._paste_region_objects(mask, labels, (3, 4),
                                              overlap="replace")
    assert added == [6] and int((out == 6).sum()) == 6 * 14

    out, added = engine._paste_region_objects(mask, labels, (-5, 15))
    assert out[15:20, 0:9].any() and not out[:15, :9].any(), (
        "a region hanging off the image is clipped, not wrapped")

    with pytest.raises(ValueError):
        engine._paste_region_objects(mask, labels, (0, 0), overlap="merge")


def test_the_box_and_the_cut_rule_at_the_engine():
    assert engine._magnifier_box((64, 64), 30, 30, 32) == (14, 14, 46, 46)
    assert engine._magnifier_box((64, 64), 0, 0, 32) == (0, 0, 16, 16)
    assert engine._magnifier_box((64, 64), 63, 63, 32) == (47, 47, 64, 64)
    assert engine._magnifier_box((64, 64), 30, 30, 31) == (15, 15, 46, 46)

    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:3, 0:3] = 1          # top-left corner
    labels[4:6, 4:6] = 2          # inside
    labels[7:10, 7:10] = 3        # bottom-right corner
    inside = engine._drop_cut_objects(labels, (20, 20, 30, 30), (64, 64))
    assert set(np.unique(inside)) == {0, 2}
    at_corner = engine._drop_cut_objects(labels, (0, 0, 10, 10), (64, 64))
    assert set(np.unique(at_corner)) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Undo and persistence
# ---------------------------------------------------------------------------

def test_one_undo_takes_back_one_clicks_objects(qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25), 2: (24, 50, 30, 56)})
    switch_on(screen, stub)

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)
    after_first = screen._canvas.mask.copy()
    assert set(np.unique(after_first)) == {0, 1}

    hover(screen, 27, 52)
    wait_for_result(qtbot, screen)
    click(screen, 27, 52)
    assert set(np.unique(screen._canvas.mask)) == {0, 1, 2}

    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, after_first)
    screen._on_undo()
    assert not screen._canvas.mask.any()
    assert screen._log.counts().get("magnifier") == 2


def test_committed_objects_survive_toggle_off_save_and_the_next_field(
        qtbot, screen, fields: Path):
    stub = CodedStub({1: (20, 20, 26, 25)})
    switch_on(screen, stub)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)
    committed = screen._canvas.mask.copy()

    screen._btn_magnifier.setChecked(False)
    np.testing.assert_array_equal(screen._canvas.mask, committed)

    screen._on_save()
    path = engine.mask_save_path(str(fields), "a.tif")
    np.testing.assert_array_equal(imageio.imread(path), committed)
    assert CurationLog.read_beside(path).counts().get("magnifier") == 1

    screen._on_next()
    assert not screen._canvas.mask.any(), "b.tif has a mask of its own"
    screen._on_prev()
    np.testing.assert_array_equal(screen._canvas.mask, committed)


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

Request = namedtuple("Request", "key")


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_the_worker_keeps_only_the_newest_request():
    """Three requests arrive while one runs; only the last of them is run."""
    gate, started = threading.Event(), threading.Event()
    ran, delivered = [], []

    def work(request):
        ran.append(request.key)
        started.set()
        assert gate.wait(10)
        return request.key

    worker = mm._NewestRequestWorker(
        work, lambda request, result, error: delivered.append(result))
    try:
        worker.submit(Request("first"))
        assert started.wait(10)
        for key in ("second", "third", "fourth"):
            worker.submit(Request(key))
        gate.set()
        _wait(worker.idle)
        assert ran == ["first", "fourth"]
        assert delivered == ["first", "fourth"]
        assert worker.superseded == 2
    finally:
        gate.set()
        worker.close()


def test_a_pinned_request_is_never_superseded():
    gate, started = threading.Event(), threading.Event()
    ran = []

    def work(request):
        ran.append(request.key)
        started.set()
        assert gate.wait(10)

    worker = mm._NewestRequestWorker(work, lambda *args: None)
    try:
        worker.submit(Request("moving"))
        assert started.wait(10)
        worker.submit(Request("clicked"))
        worker.submit(Request("clicked"), pin=True)
        worker.submit(Request("later"))
        worker.submit(Request("latest"))
        gate.set()
        _wait(worker.idle)
        assert ran == ["moving", "clicked", "latest"]
        assert worker.submit(Request("x")) is True
    finally:
        gate.set()
        worker.close()
    assert worker.submit(Request("after close")) is False


def test_a_click_before_the_box_is_up_to_date_commits_when_it_is(
        qtbot, screen):
    """The click's region is pinned: the mouse moving on cannot lose it."""
    stub = CodedStub({1: (4, 4, 9, 9), 2: (20, 20, 26, 25), 3: (50, 50, 56, 56)})
    stub.gate = threading.Event()
    switch_on(screen, stub)
    try:
        hover(screen, 10, 10)
        qtbot.waitUntil(lambda: len(stub.calls) == 1, timeout=10_000)
        hover(screen, 30, 30)
        click(screen, 30, 30)
        assert not screen._canvas.mask.any(), "nothing to commit yet"
        hover(screen, 45, 45)
        hover(screen, 50, 52)
    finally:
        stub.gate.set()
    qtbot.waitUntil(lambda: screen._canvas.mask.any(), timeout=10_000)
    qtbot.waitUntil(screen._magnifier._worker.idle, timeout=10_000)

    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25)}))
    boxes = [request.box for request in stub.calls]
    assert (14, 14, 46, 46) in boxes
    assert engine._magnifier_box((IMG_N, IMG_N), 45, 45, SIZE) not in boxes, (
        "a region the mouse only passed through was run anyway")


def test_moving_the_pointer_never_runs_the_model_on_the_gui_thread(
        qtbot, screen):
    """The model takes a full second; no mouse move may wait for it."""
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=1.0)
    gui = threading.get_ident()
    switch_on(screen, stub)

    slowest = 0.0
    for x in range(18, 44):
        started = time.perf_counter()
        hover(screen, x, 30)
        screen._canvas.repaint()
        slowest = max(slowest, time.perf_counter() - started)
    assert slowest < 0.5, f"a mouse move took {slowest:.3f}s"

    qtbot.waitUntil(screen._magnifier._worker.idle, timeout=15_000)
    assert stub.threads and gui not in stub.threads
    assert len(stub.calls) <= 3, (
        f"{len(stub.calls)} regions were run for 26 moves; superseded "
        f"requests should have been dropped")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def blob_field(n=IMG_N) -> np.ndarray:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 1000.0)
    for cy, cx in ((16, 16), (16, 44), (44, 30)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] += 3000
    img += rng.normal(0, 120, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_the_classical_mode_works_with_no_model_installed(monkeypatch):
    """Asked for Cellpose with no Cellpose, the answer is classical objects."""
    monkeypatch.setitem(sys.modules, "cellpose", None)
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    field = blob_field()
    request = mm._MagnifierRequest(
        key=("k",), crop=field, box=(0, 0, IMG_N, IMG_N), shape=field.shape,
        mode="cellpose", sensitivity=0.0, bright=True, min_area=20,
        model_name="cpsam", diameter=0, colour=(255, 0, 0))

    labels, used, note = mm._segment_region(
        request, load_model=mm.load_cellpose_model)
    assert used == "classical"
    assert "ImportError" in note or "ModuleNotFoundError" in note
    assert labels.max() == 3
    assert len({int(labels[16, 16]), int(labels[16, 44]),
                int(labels[44, 30])} - {0}) == 3

    empty = mm._MagnifierRequest(*request[:1], crop=np.full(
        (IMG_N, IMG_N), 1000, np.uint16) + np.random.default_rng(1).integers(
        0, 200, (IMG_N, IMG_N)).astype(np.uint16), box=request.box,
        shape=request.shape, mode="classical", sensitivity=0.0, bright=True,
        min_area=20, model_name="cpsam", diameter=0, colour=(255, 0, 0))
    assert mm._segment_region(empty)[0].max() == 0, (
        "background noise has an Otsu level too; it must not become objects")


def test_sensitivity_moves_the_classical_cut_the_way_its_name_says():
    """Raised, it finds dim objects the default misses; noise stays empty.

    Three disks one noise deviation above the background: below what the
    default cut takes, within reach of a sensitivity of +2. Measured before
    this was written: at 0 the five seeds found 1, 1, 0, 0, 0 objects and at
    +2 between 4 and 6 -- dim objects come back fragmented, which is the
    honest limit of a threshold, but they come back.
    """
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:128, 0:128]
    dim = np.full((128, 128), 1000.0)
    for cy, cx in ((30, 30), (80, 40), (60, 95)):
        dim[(yy - cy) ** 2 + (xx - cx) ** 2 <= 81] += 150
    dim = np.clip(dim + rng.normal(0, 150, dim.shape), 0, None).astype(np.uint16)
    noise = np.clip(rng.normal(1000, 150, (128, 128)), 0, None).astype(np.uint16)

    strict = engine._classical_region_labels(dim, sensitivity=-2, min_area=20)
    eager = engine._classical_region_labels(dim, sensitivity=2, min_area=20)
    assert strict.max() == 0 < eager.max()
    assert engine._classical_region_labels(
        noise, sensitivity=2, min_area=20).max() == 0


def test_the_screen_offers_and_runs_classical_with_no_model(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """No Cellpose: Mode offers only Classical, and a click still adds."""
    monkeypatch.setattr(mm, "find_spec", lambda name: None)
    folder = tmp_path / "blobs"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif", blob_field())
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._open_folder(str(folder))
        made._canvas.resize(CANVAS_W, CANVAS_H)
        made._canvas.refresh()
        made._min_area.setValue(20)
        made._mag_size.setValue(SIZE)
        assert [made._mag_mode.itemData(i)
                for i in range(made._mag_mode.count())] == ["classical"]
        made._btn_magnifier.setChecked(True)
        hover(made, 30, 44)
        wait_for_result(qtbot, made)
        click(made, 30, 44)
        assert made._canvas.mask[44, 30] > 0
        assert made._canvas.mask[16, 16] == 0, "that blob is outside the box"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_mode_that_cannot_load_falls_back_and_says_so(qtbot, screen):
    def broken(name):
        raise ImportError("no cellpose here")

    screen._magnifier.segment = mm.partial(mm._segment_region,
                                           load_model=broken)
    screen._magnifier.set_mode("cellpose")
    screen._btn_magnifier.setChecked(True)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    assert screen._magnifier._shown.mode == "classical"
    assert "cellpose could not run" in screen._status_label.text()
    assert screen._magnifier.build_request().mode == "classical", (
        "a model that failed is not retried on every mouse move")


# ---------------------------------------------------------------------------
# Zoom and drawing
# ---------------------------------------------------------------------------

def test_the_wheel_zooms_the_magnifier_and_not_the_canvas(qtbot, screen):
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    before = screen._magnifier.zoom
    speed = screen._canvas.zoom_speed

    wheel(screen, 30, 30, 120)
    assert screen._magnifier.zoom == pytest.approx(before * speed, abs=0.01)
    assert screen._mag_zoom.value() == pytest.approx(before * speed, abs=0.01)
    assert not screen._canvas.is_zoomed()
    wheel(screen, 30, 30, -120)
    assert screen._magnifier.zoom == pytest.approx(before, abs=0.02)

    screen._btn_magnifier.setChecked(False)
    wheel(screen, 30, 30, 120)
    assert screen._canvas.is_zoomed(), "off, the wheel is the canvas's again"


def test_the_lens_is_the_crop_magnified_with_the_cursor_pixel_underneath(
        qtbot, screen):
    switch_on(screen, CodedStub({}))
    for img_x, img_y in ((30, 30), (2, 3)):
        hover(screen, img_x, img_y)
        box, lens, scale = screen._magnifier.lens_geometry()
        assert scale == pytest.approx(SCALE * screen._magnifier.zoom)
        assert lens.width() == pytest.approx((box[2] - box[0]) * scale)
        cx, cy = canvas_xy(img_x, img_y)
        assert lens.left() + (img_x + 0.5 - box[0]) * scale == pytest.approx(cx)
        assert lens.top() + (img_y + 0.5 - box[1]) * scale == pytest.approx(cy)


def test_the_box_is_drawn_while_on_and_gone_when_off(qtbot, screen):
    accent = QColor(active_palette()["accent"]).rgb()

    def accent_pixels() -> int:
        image = screen._canvas.grab().toImage().convertToFormat(
            QImage.Format_RGB32)
        data = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
            image.height(), image.bytesPerLine() // 4)
        return int((data == np.uint32(accent)).sum())

    baseline = accent_pixels()
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    assert accent_pixels() > baseline + 100
    screen._btn_magnifier.setChecked(False)
    assert accent_pixels() == baseline


# ---------------------------------------------------------------------------
# The box-border option
# ---------------------------------------------------------------------------
#
# The box round (30, 30) is (14, 14, 46, 46). Object 3 runs from x 40 to 52,
# so the box's right edge -- inside the image -- cuts it at x 46.

CUT_BY_THE_BOX = {1: (20, 20, 26, 25), 3: (40, 30, 52, 36)}


def test_the_border_option_is_on_by_default_and_drops_what_the_box_cut(
        qtbot, screen):
    stub = CodedStub(CUT_BY_THE_BOX)
    switch_on(screen, stub)
    assert screen._mag_exclude_border.isChecked()
    assert screen._magnifier.exclude_border is True

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)
    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25)}))


def test_with_the_option_off_a_cut_object_goes_in_as_far_as_the_box_saw_it(
        qtbot, screen):
    """Unticked, the cut object is offered and added, clipped to the box.

    Changing the option re-reads the answer the model already gave for this
    region rather than asking the model again.
    """
    stub = CodedStub(CUT_BY_THE_BOX)
    switch_on(screen, stub)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    asked = len(stub.calls)

    screen._mag_exclude_border.setChecked(False)
    assert screen._magnifier.exclude_border is False
    wait_for_result(qtbot, screen)
    assert set(np.unique(screen._magnifier._shown.labels)) == {0, 1, 3}
    assert len(stub.calls) == asked, "the option asked the model again"

    click(screen, 30, 30)
    np.testing.assert_array_equal(
        screen._canvas.mask,
        rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25), 2: (40, 30, 46, 36)}))


def test_objects_at_the_image_border_are_offered_whatever_the_option(
        qtbot, screen):
    """At the corner the box is (0, 0, 18, 19); only x 18 is a box edge."""
    stub = CodedStub({1: (0, 0, 5, 4), 2: (10, 12, 24, 16)})
    switch_on(screen, stub)
    screen._mag_exclude_border.setChecked(False)

    hover(screen, 2, 3)
    wait_for_result(qtbot, screen)
    click(screen, 2, 3)
    np.testing.assert_array_equal(
        screen._canvas.mask,
        rect_mask((IMG_N, IMG_N), {1: (0, 0, 5, 4), 2: (10, 12, 18, 16)}))


# ---------------------------------------------------------------------------
# Whole-image mode
# ---------------------------------------------------------------------------

def right_click(screen, img_x, img_y, release_at=None):
    x, y = canvas_xy(img_x, img_y)
    canvas = screen._canvas
    canvas.mousePressEvent(_mouse(
        QEvent.Type.MouseButtonPress, x, y, Qt.RightButton, Qt.RightButton))
    if release_at is not None:
        x, y = canvas_xy(*release_at)
        canvas.mouseMoveEvent(_mouse(
            QEvent.Type.MouseMove, x, y, Qt.NoButton, Qt.RightButton))
    canvas.mouseReleaseEvent(_mouse(
        QEvent.Type.MouseButtonRelease, x, y, Qt.RightButton, Qt.NoButton))


def whole_image_on(screen, stub) -> None:
    screen._magnifier.segment = stub
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    screen._btn_magnifier.setChecked(True)
    assert screen._magnifier.scope == "image"


def wait_for_whole_image(qtbot, screen) -> None:
    magnifier = screen._magnifier
    qtbot.waitUntil(
        lambda: magnifier._image_result is not None and not magnifier.updating(),
        timeout=15_000)


def test_the_region_mode_stays_the_default_and_whole_image_is_offered(screen):
    scope = screen._mag_scope
    assert [scope.itemData(i) for i in range(scope.count())] == [
        "region", "image"]
    assert scope.currentData() == "region"
    assert screen._magnifier.scope == "region"

    scope.setCurrentIndex(scope.findData("image"))
    assert screen._magnifier.scope == "image"
    assert not screen._mag_exclude_border.isEnabled(), (
        "whole-image objects are never cut by the box")
    scope.setCurrentIndex(scope.findData("region"))
    assert screen._mag_exclude_border.isEnabled()


def test_whole_image_mode_segments_the_entire_image_once(qtbot, screen):
    """One model call for the field; moving the box never asks again."""
    stub = CodedStub({7: (20, 20, 26, 25), 9: (50, 50, 58, 56)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)

    assert len(stub.calls) == 1
    request = stub.calls[0]
    assert request.box == (0, 0, IMG_N, IMG_N)
    np.testing.assert_array_equal(request.crop, screen._canvas.image)
    assert "2 object(s)" in screen._status_label.text()

    for x in range(4, 60, 3):
        hover(screen, x, 30)
        screen._canvas.repaint()
    qtbot.wait(50)
    assert len(stub.calls) == 1, "the box asked the model again"


def test_a_left_click_adds_exactly_the_object_under_it(qtbot, screen):
    """The whole object, including what lies outside the box, and no other."""
    stub = CodedStub({7: (20, 20, 26, 25), 9: (28, 22, 34, 30),
                      4: (2, 40, 62, 43)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)

    hover(screen, 22, 22)
    click(screen, 22, 22)
    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25)}))
    assert screen._log.counts().get("magnifier") == 1
    edit = screen._log.edits[-1]
    assert edit.target == [1]
    assert edit.detail["scope"] == "image"

    hover(screen, 30, 41)
    click(screen, 30, 41)
    np.testing.assert_array_equal(
        screen._canvas.mask,
        rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25), 2: (2, 40, 62, 43)}))

    before = screen._canvas.mask.copy()
    hover(screen, 27, 34)
    click(screen, 27, 34)
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert "no object under the click" in screen._status_label.text()
    assert screen._log.counts().get("magnifier") == 2


def test_a_right_click_removes_exactly_one_mask_object_and_undo_restores_it(
        qtbot, screen):
    """Any object in the mask, not only one the magnifier put there."""
    existing = rect_mask((IMG_N, IMG_N), {1: (4, 4, 10, 10),
                                          2: (20, 20, 30, 30),
                                          3: (36, 36, 46, 46)})
    screen._canvas.mask = existing.copy()
    screen._history.push(existing)
    whole_image_on(screen, CodedStub({}))
    wait_for_whole_image(qtbot, screen)

    hover(screen, 25, 25)
    right_click(screen, 25, 25, release_at=(40, 40))
    expected = existing.copy()
    expected[expected == 2] = 0
    np.testing.assert_array_equal(screen._canvas.mask, expected)
    assert screen._log.counts().get("delete") == 1
    assert screen._log.edits[-1].target == 2
    assert screen._log.edits[-1].detail["tool"] == "magnifier"

    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, existing)

    hover(screen, 32, 5)
    right_click(screen, 32, 5)
    np.testing.assert_array_equal(screen._canvas.mask, existing)
    assert "nothing was removed" in screen._status_label.text()
    assert screen._log.counts().get("delete") == 1


def test_changing_a_model_setting_discards_the_whole_image_objects(
        qtbot, screen):
    stub = CodedStub({7: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)

    screen._mag_sensitivity.setValue(1.5)
    assert screen._magnifier._image_result is None
    assert "settings changed" in screen._status_label.text()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2
    assert stub.calls[-1].sensitivity == 1.5

    screen._min_area.setValue(3)
    assert screen._magnifier._image_result is None, (
        "a setting read from elsewhere on the panel counts too")
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 3
    assert stub.calls[-1].min_area == 3

    screen._mag_size.setValue(48)
    screen._mag_zoom.setValue(3.0)
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("skip"))
    assert screen._magnifier._image_result is not None
    qtbot.wait(50)
    assert len(stub.calls) == 3, "size, zoom and overlap do not reach the model"


def test_whole_image_segmentation_never_blocks_the_gui(qtbot, screen):
    """A one-second model: switching on and moving return at once."""
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=1.0)
    gui = threading.get_ident()

    started = time.perf_counter()
    whole_image_on(screen, stub)
    assert time.perf_counter() - started < 0.5
    assert not screen._mag_progress.isHidden()
    assert not screen._mag_cancel.isHidden()

    slowest = 0.0
    for x in range(18, 44):
        started = time.perf_counter()
        hover(screen, x, 30)
        screen._canvas.repaint()
        slowest = max(slowest, time.perf_counter() - started)
    assert slowest < 0.5, f"a mouse move took {slowest:.3f}s"

    wait_for_whole_image(qtbot, screen)
    assert stub.threads and gui not in stub.threads
    assert len(stub.calls) == 1
    assert screen._mag_progress.isHidden()
    assert screen._mag_cancel.isHidden()


def test_cancel_discards_the_run_and_a_click_starts_it_again(qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25)})
    stub.gate = threading.Event()
    try:
        whole_image_on(screen, stub)
        qtbot.waitUntil(lambda: len(stub.calls) == 1, timeout=10_000)
        screen._mag_cancel.click()
        assert screen._mag_progress.isHidden()
        assert "cancelled" in screen._status_label.text()
    finally:
        stub.gate.set()
    qtbot.waitUntil(screen._magnifier._image_worker.idle, timeout=10_000)
    qtbot.wait(100)
    assert screen._magnifier._image_result is None, "a cancelled run was kept"

    hover(screen, 22, 22)
    qtbot.wait(50)
    assert len(stub.calls) == 1, "moving the mouse restarted a cancelled run"

    click(screen, 22, 22)
    assert not screen._canvas.mask.any()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2
    click(screen, 22, 22)
    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25)}))


def test_whole_image_objects_survive_a_save_and_follow_the_field(
        qtbot, screen, fields: Path):
    stub = CodedStub({1: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    hover(screen, 22, 22)
    click(screen, 22, 22)
    committed = screen._canvas.mask.copy()

    screen._on_save()
    path = engine.mask_save_path(str(fields), "a.tif")
    np.testing.assert_array_equal(imageio.imread(path), committed)

    screen._on_next()
    assert screen._magnifier._image_result is None, (
        "the objects found on a.tif were offered on b.tif")
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2, "the next field is segmented of its own"
    screen._on_prev()
    np.testing.assert_array_equal(screen._canvas.mask, committed)


def test_the_worker_can_drop_what_is_waiting():
    gate, started = threading.Event(), threading.Event()
    ran = []

    def work(request):
        ran.append(request.key)
        started.set()
        assert gate.wait(10)

    worker = mm._NewestRequestWorker(work, lambda *args: None)
    try:
        worker.submit(Request("running"))
        assert started.wait(10)
        worker.submit(Request("waiting"))
        worker.drop_waiting()
        gate.set()
        _wait(worker.idle)
        assert ran == ["running"]
    finally:
        gate.set()
        worker.close()
