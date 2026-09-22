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
* with no model installed, the Otsu mode still finds objects.

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


def test_the_otsu_mode_works_with_no_model_installed(monkeypatch):
    """Asked for Cellpose with no Cellpose, the answer is Otsu objects.

    ``classical`` until item 419 point 5 renamed the mode to ``otsu``; the
    old name still reaches :func:`mm._segment_region`, which is the next
    assertion.
    """
    monkeypatch.setitem(sys.modules, "cellpose", None)
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    field = blob_field()
    request = mm._MagnifierRequest(
        key=("k",), crop=field, box=(0, 0, IMG_N, IMG_N), shape=field.shape,
        mode="cellpose", sensitivity=0.0, bright=True, min_area=20,
        model_name="cpsam", diameter=0, colour=(255, 0, 0))

    labels, used, note = mm._segment_region(
        request, load_model=mm.load_cellpose_model)
    assert used == "otsu"
    assert "ImportError" in note or "ModuleNotFoundError" in note
    assert labels.max() == 3
    assert len({int(labels[16, 16]), int(labels[16, 44]),
                int(labels[44, 30])} - {0}) == 3

    empty = mm._MagnifierRequest(*request[:1], crop=np.full(
        (IMG_N, IMG_N), 1000, np.uint16) + np.random.default_rng(1).integers(
        0, 200, (IMG_N, IMG_N)).astype(np.uint16), box=request.box,
        shape=request.shape, mode="classical", sensitivity=0.0, bright=True,
        min_area=20, model_name="cpsam", diameter=0, colour=(255, 0, 0))
    labels, used, _note = mm._segment_region(empty)
    assert labels.max() == 0, (
        "background noise has an Otsu level too; it must not become objects")
    assert used == "otsu", (
        "the old mode name still runs, under the name it has now")


def test_sensitivity_moves_the_otsu_cut_the_way_its_name_says():
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


def test_the_screen_offers_and_runs_otsu_with_no_model(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """No Cellpose: Otsu is the only mode that runs, and a click adds.

    Every optional backend stays listed, greyed, since c60d48e35 (item 419
    point 3): a model absent from the box teaches nobody it exists. This
    test still said Mode offers "only Classical" and had been red on nightly
    since that commit. Item 423 added the four Cellpose 3 models to that
    list, so the list is read from _MAGNIFIER_BACKENDS rather than typed
    out again, and item 419 point 5 renamed Classical to Otsu. Item 473
    added organelle detection's own methods and the CPU threshold
    algorithms, which are scikit-image and need nothing installed, so they
    are listed and RUNNABLE here.
    """
    from spacr import _segmentation_backends as backends

    monkeypatch.setattr(mm, "find_spec", lambda name: None)
    monkeypatch.setattr(backends, "_importable", lambda module: False)
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
                for i in range(made._mag_mode.count())] == [
            "otsu", *mm.cpu_modes.modes(), *mm.organelle_modes.modes(),
            *mm._MAGNIFIER_BACKENDS]
        assert made._mag_uninstalled == set(mm._MAGNIFIER_BACKENDS)
        assert made._mag_mode.currentData() == "otsu"
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
    assert screen._magnifier._shown.mode == "otsu"
    assert "Cellpose could not run" in screen._status_label.text()
    assert screen._magnifier.build_request().mode == "otsu", (
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


# ---------------------------------------------------------------------------
# Every sentence the magnifier writes reaches a translator
# ---------------------------------------------------------------------------
#
# Item 407's WHAT IS LEFT: "the region mode's other f-string status lines
# (could not segment this region, the Otsu fallback note) are still not
# catalogued". An f-string is built before anything can translate it, so a
# reader in one of the other eight languages met the magnifier's failures in
# English. These pin both halves: the sentence goes through `tr` at the point
# it is shown, and what it says about a mode is what the Mode box calls it.
#
# Translations themselves belong to the catalog pass; nothing here needs one.
# A stand-in `tr` that marks its input proves the call is made.


def _marked(text, *args, **values):
    """A stand-in translation that cannot be mistaken for the English."""
    rendered = str(text).format(**values) if values else str(text)
    return f"⟦{rendered}⟧"


@pytest.fixture
def marked(monkeypatch):
    """Make every translated string visibly translated."""
    monkeypatch.setattr("spacr.qt.i18n.tr", _marked)
    monkeypatch.setattr(mm, "tr", _marked, raising=False)


def test_a_click_that_waits_for_the_region_says_so_through_tr(
        qtbot, screen, marked):
    """The pinned-click notice was a bare literal until item 407's polish."""
    stub = CodedStub({1: (20, 20, 26, 25)})
    stub.gate = threading.Event()
    switch_on(screen, stub)
    try:
        hover(screen, 30, 30)
        qtbot.waitUntil(lambda: len(stub.calls) == 1, timeout=10_000)
        click(screen, 30, 30)
        assert screen._status_label.text() == _marked(
            "Magnifier: segmenting this region — its objects are added as "
            "soon as the box is up to date.")
    finally:
        stub.gate.set()
    qtbot.waitUntil(screen._magnifier._worker.idle, timeout=10_000)


def test_a_region_the_model_could_not_segment_says_so_through_tr(
        qtbot, screen, marked):
    """The error is a value in the template, not a piece of the sentence."""

    def explode(request):
        raise ValueError("the model fell over")

    switch_on(screen, explode)
    hover(screen, 30, 30)
    qtbot.waitUntil(
        lambda: "fell over" in screen._status_label.text(), timeout=10_000)
    assert screen._status_label.text() == _marked(
        "Magnifier could not segment this region: {error}",
        error="the model fell over")


def test_the_otsu_fallback_note_goes_through_tr_and_names_the_caption(
        qtbot, screen, marked):
    """"cellpose3:cyto3 could not run" sends a reader looking for a row that
    does not exist; the Mode box calls it "Cellpose 3 · cyto3"."""
    def broken(name):
        raise ImportError("no cellpose here")

    screen._magnifier.segment = mm.partial(mm._segment_region,
                                           load_model=broken)
    screen._magnifier.set_mode("cellpose")
    screen._btn_magnifier.setChecked(True)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)

    shown = screen._status_label.text()
    assert shown == _marked(
        "Magnifier: {mode} could not run ({reason}); the Otsu mode is "
        "segmenting instead.",
        mode=_marked("Cellpose"), reason="ImportError: no cellpose here")
    assert "cellpose could not run" not in shown, (
        "the mode was named by its key rather than by its caption")


def test_the_mode_caption_is_the_one_the_mode_box_shows(screen):
    """Every mode the box offers, named the way the box names it."""
    box = screen._mag_mode
    for index in range(box.count()):
        mode = box.itemData(index)
        assert mm._magnifier_mode_label(mode) == box.itemText(index), mode
    assert mm._magnifier_mode_label("otsu") == "Otsu"
    assert mm._magnifier_mode_label("classical") == "Otsu", (
        "item 419 renamed the mode; a saved session still says classical")
    assert mm._magnifier_mode_label("cellpose3:cyto3") == "Cellpose 3 · cyto3"
    assert mm._magnifier_mode_label("no such mode") == "no such mode", (
        "a mode with no caption is hidden rather than shown as itself")


def test_the_toggle_says_what_it_did_through_tr(qtbot, screen, marked):
    switch_on(screen, CodedStub({}))
    assert screen._status_label.text() == _marked(
        "Magnifier on: a click adds the objects outlined in the box; "
        "the mouse wheel changes its zoom.")
    screen._btn_magnifier.setChecked(False)
    assert screen._status_label.text() == _marked(
        "Magnifier off. The objects it added stay in the mask.")


def test_a_click_that_adds_nothing_says_so_through_tr(qtbot, screen, marked):
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    click(screen, 30, 30)
    assert screen._status_label.text() == _marked(
        "Magnifier: nothing to add — the box outlines no object, or "
        "every object it outlines overlaps one already in the mask.")


def test_a_paste_that_the_engine_refuses_says_so_through_tr(screen, marked):
    """The refusal carries the engine's reason as a value, not as prose."""
    stub_request = mm._MagnifierRequest(
        key=(0, (0, 0, 4, 4)), crop=np.zeros((4, 4), np.uint16),
        box=(0, 0, 4, 4), shape=(IMG_N, IMG_N), mode="otsu",
        sensitivity=0.0, bright=True, min_area=0, model_name="cpsam",
        diameter=0, colour=(1, 2, 3))
    result = mm._MagnifierResult(
        stub_request, np.zeros((4, 4), np.int32), "otsu", "", None, 0)
    screen._mag_overlap.addItem("Nonsense", "nonsense")
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("nonsense"))

    assert screen._commit_magnifier_result(result) == []
    shown = screen._status_label.text()
    assert shown.startswith("⟦Magnifier could not add objects: ")
    assert "nonsense" in shown


# ---------------------------------------------------------------------------
# The whole image is segmented once per field, not once per visit
# ---------------------------------------------------------------------------
#
# Item 407's WHAT IS LEFT: "The whole-image objects are not cached per field,
# so revisiting a field re-segments it." Measured on this repo's own GPU that
# is 3.7-10.3 s a field; on a CPU the same run is minutes, and a curator
# walking a plate goes back and forth constantly.


def test_a_field_segmented_whole_is_offered_again_on_returning(
        qtbot, screen, fields: Path):
    stub = CodedStub({1: (20, 20, 26, 25), 2: (50, 50, 56, 56)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 1

    screen._on_next()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2, "the next field is segmented of its own"

    screen._on_prev()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2, "the first field was segmented a second time"
    assert screen._magnifier._image_result.count == 2

    hover(screen, 22, 22)
    click(screen, 22, 22)
    np.testing.assert_array_equal(
        screen._canvas.mask, rect_mask((IMG_N, IMG_N), {1: (20, 20, 26, 25)}))


def test_what_is_offered_again_is_what_was_found_under_those_settings(
        qtbot, screen):
    """A setting the model reads is part of the name, so it never matches."""
    stub = CodedStub({1: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)

    screen._on_next()
    wait_for_whole_image(qtbot, screen)
    screen._on_prev()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 2

    screen._mag_sensitivity.setValue(2.0)
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 3, "the kept objects were offered for new settings"

    screen._mag_sensitivity.setValue(0.0)
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 3, "the first settings were segmented again"


def test_the_memory_budget_takes_the_field_nobody_came_back_to(
        qtbot, screen, monkeypatch):
    """The ceiling is the lower of this cache's own and the user's.

    A kept result is its int32 labels AND the worker's RGBA outlines of the
    whole field, which the box slices (32 KB each at 64x64), and the budget
    counts both -- `_MagnifierResult.nbytes`.
    """
    monkeypatch.setattr(mm, "_MAGNIFIER_IMAGE_CACHE_MB", 0.05)
    stub = CodedStub({1: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    assert len(screen._magnifier._image_cache) == 1

    screen._on_next()
    wait_for_whole_image(qtbot, screen)
    assert len(screen._magnifier._image_cache) == 1, (
        "two 64x64 results of 32 KB each are over a 0.05 MB ceiling")

    screen._on_prev()
    wait_for_whole_image(qtbot, screen)
    assert len(stub.calls) == 3, "the dropped field was not segmented again"


def test_an_idle_field_is_released_by_the_users_own_timeout(qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    magnifier = screen._magnifier
    assert len(magnifier._image_cache) == 1

    stale = time.time() - 10 * 60 * 60
    for name in magnifier._image_cache:
        magnifier._image_cache_used[name] = stale
    magnifier._trim_image_cache()
    assert not magnifier._image_cache, (
        "a field nobody has gone back to in ten hours is still held")
    assert magnifier._image_result is not None, (
        "a trim took the objects out from under the box")


def test_a_canvas_with_no_field_behind_it_is_never_kept(qtbot, screen):
    """Nothing names an array handed straight to the canvas, so nothing
    could tell two of them apart."""
    magnifier = screen._magnifier
    magnifier.set_field("")
    stub = CodedStub({1: (20, 20, 26, 25)})
    magnifier.segment = stub
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    screen._btn_magnifier.setChecked(True)
    wait_for_whole_image(qtbot, screen)

    assert magnifier._image_result is not None
    assert not magnifier._image_cache


def test_closing_the_screen_gives_the_label_images_back(qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25)})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    assert screen._magnifier._image_cache

    screen._magnifier.close()
    assert not screen._magnifier._image_cache
    assert not screen._magnifier._image_cache_used


# ---------------------------------------------------------------------------
# The Overlap rule is counted once over the pixels, and says the same thing
# ---------------------------------------------------------------------------
#
# The rule used to run a whole-region comparison per object, and under `clip`
# a whole connected-component pass per object as well. Item 417 let the box be
# as wide as the image, and a click then waited seconds on the GUI thread:
# measured on a 2048 px region holding 500 objects, 6.1 s for clip and 0.82 s
# for skip. These pin the answer against the rule as it was written, so the
# faster one is only faster.


def _rule_object_by_object(labels, occupied, *, overlap, min_area):
    """The Overlap rule as `_paste_region_objects` ran it before, per object."""
    from scipy import ndimage

    incoming = np.asarray(labels)
    kept = np.zeros(incoming.shape, dtype=np.int64)
    for value in (int(v) for v in np.unique(incoming) if int(v) > 0):
        body = incoming == value
        if overlap == "skip" and bool((body & occupied).any()):
            continue
        if overlap == "clip":
            body = body & ~occupied
            pieces, count = ndimage.label(body, structure=np.ones((3, 3)))
            if count > 1:
                areas = np.bincount(pieces.ravel())
                areas[0] = 0
                body = pieces == int(np.argmax(areas))
        if int(body.sum()) < max(1, int(min_area)):
            continue
        kept[body] = value
    return kept


@pytest.mark.parametrize("overlap", ["clip", "skip", "replace"])
@pytest.mark.parametrize("min_area", [0, 1, 7])
def test_the_overlap_rule_agrees_with_the_rule_it_replaced(overlap, min_area):
    """Random regions, including objects an existing one cuts in two."""
    rng = np.random.default_rng(11)
    for _ in range(12):
        labels = np.zeros((40, 40), np.int32)
        for value in range(1, 7):
            y, x = rng.integers(0, 33, 2)
            h, w = rng.integers(3, 8, 2)
            labels[y:y + h, x:x + w] = value
        occupied = np.zeros((40, 40), bool)
        for _ in range(4):
            y, x = rng.integers(0, 36, 2)
            occupied[y:y + rng.integers(1, 5), x:x + rng.integers(1, 5)] = True

        mine = engine._surviving_region_objects(
            labels, occupied, overlap=overlap, min_area=min_area)
        theirs = _rule_object_by_object(
            labels, occupied, overlap=overlap, min_area=min_area)
        np.testing.assert_array_equal(mine, theirs)


def test_an_object_an_existing_one_splits_keeps_its_largest_piece_only():
    labels = np.zeros((9, 9), np.int32)
    labels[2:7, 1:8] = 4
    occupied = np.zeros((9, 9), bool)
    occupied[:, 3] = True

    kept = engine._surviving_region_objects(labels, occupied, overlap="clip")
    assert set(np.unique(kept)) == {0, 4}
    np.testing.assert_array_equal(kept[2:7, 4:8], 4)
    assert not kept[:, :4].any(), "the two-pixel-wide piece was the smaller one"


def test_two_pieces_of_the_same_size_keep_the_one_labelled_first():
    """The tie-break is the rule's, not the new implementation's."""
    labels = np.zeros((5, 7), np.int32)
    labels[1:4, :] = 2
    occupied = np.zeros((5, 7), bool)
    occupied[:, 3] = True

    kept = engine._surviving_region_objects(labels, occupied, overlap="clip")
    np.testing.assert_array_equal(kept[1:4, :3], 2)
    assert not kept[:, 4:].any()
    np.testing.assert_array_equal(
        kept, _rule_object_by_object(labels, occupied, overlap="clip",
                                     min_area=0))


def test_the_rule_refuses_a_name_it_does_not_know():
    with pytest.raises(ValueError, match="overlap must be one of"):
        engine._surviving_region_objects(
            np.zeros((3, 3), np.int32), np.zeros((3, 3), bool),
            overlap="whatever")


# ---------------------------------------------------------------------------
# The box shows what a click would add, not what the model found
# ---------------------------------------------------------------------------
#
# Item 407's second pass, still open at the end: "Preview the overlap rule
# inside the box (outline what a click would actually add, not what the model
# found)." A click that added half an object, or nothing at all, said so only
# afterwards, in the status line.


def _alpha(picture: QImage, img_x: int, img_y: int, box) -> int:
    """The preview's alpha at image pixel (img_x, img_y)."""
    x0, y0 = int(box[0]), int(box[1])
    return QColor(picture.pixelColor(img_x - x0, img_y - y0)).alpha()


def preview_of(screen):
    """The picture the box draws, when the rule ghosted anything in it.

    The ghost is built on the WORKER, beside the outlines, and the result
    carries it; the box draws it and nothing else. None here means the rule
    took nothing away and the box is drawing the model's own outlines --
    which is what `_shown_image` then holds.
    """
    magnifier = screen._magnifier
    shown = magnifier._shown
    if shown is None or shown.ghost is None:
        return None
    return magnifier._shown_image


def test_what_the_overlap_rule_takes_away_is_ghosted_in_the_box(
        qtbot, screen):
    """Clip: the half of the object the mask already owns is not added."""
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)

    picture = preview_of(screen)
    assert picture is not None, "the rule takes half the object and said nothing"
    box = screen._magnifier._shown.request.box
    kept = _alpha(picture, 26, 22, box)
    lost = _alpha(picture, 22, 22, box)
    assert kept > 0 and lost > 0, "both halves are drawn"
    assert lost * 4 == kept, (
        f"the pixels a click would not add are not ghosted: {lost} vs {kept}")

    click(screen, 24, 22)
    added = screen._canvas.mask == 2
    assert added[22, 26] and not added[22, 22], (
        "the box promised exactly what the click did")


def test_replace_ghosts_nothing_because_it_takes_nothing_away(qtbot, screen):
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("replace"))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)

    assert screen._magnifier.overlap == "replace"
    assert preview_of(screen) is None, (
        "Replace adds every pixel, so nothing is ghosted")


def test_skip_ghosts_the_whole_object_it_would_leave_out(qtbot, screen):
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 22, 22)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26), 5: (30, 30, 36, 36)}))
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("skip"))
    hover(screen, 28, 28)
    wait_for_result(qtbot, screen)

    picture = preview_of(screen)
    assert picture is not None
    box = screen._magnifier._shown.request.box
    assert _alpha(picture, 32, 32, box) > _alpha(picture, 26, 24, box), (
        "the object Skip leaves out is drawn as solidly as the one it adds")


def test_an_empty_mask_draws_the_models_own_outlines(qtbot, screen):
    """Nothing to clip against, so nothing is recomputed or copied."""
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)
    assert preview_of(screen) is None


def test_changing_the_rule_redraws_the_box_and_asks_no_model(qtbot, screen):
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    stub = CodedStub({4: (20, 20, 28, 26)})
    switch_on(screen, stub)
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)
    calls = len(stub.calls)
    assert preview_of(screen) is not None

    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("replace"))
    wait_for_result(qtbot, screen)
    assert preview_of(screen) is None
    assert len(stub.calls) == calls, "the rule asked the model again"


def test_the_preview_is_built_once_until_something_moves(qtbot, screen):
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)

    first = preview_of(screen)
    assert first is preview_of(screen), "a repaint that moves nothing rebuilt it"
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData("skip"))
    wait_for_result(qtbot, screen)
    assert preview_of(screen) is not first


def _faded(result) -> int:
    """How many pixels the ghost fades out of what the model found."""
    return int(np.count_nonzero(result.ghost[..., 3] < result.overlay[..., 3]))


def test_the_box_is_drawn_from_a_picture_the_worker_built(qtbot, screen):
    """The rule is counted over the box's pixels, and not on this thread.

    Item 380's budget is a MOVE's budget, and the box can be as wide as the
    field: the first version of this ghosted inside paintEvent, where the
    same rule cost 125.5 ms per delivered result on the largest box item
    417 allows. What paint is allowed to do now is draw.
    """
    from PySide6.QtGui import QPainter, QPixmap

    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)

    magnifier = screen._magnifier
    assert magnifier._shown.ghost is not None, "the worker built no ghost"
    assert magnifier._shown.request.occupied is not None, (
        "the mask under the box never reached the worker")

    before = magnifier._shown_image
    surface = QPixmap(CANVAS_W, CANVAS_H)
    painter = QPainter(surface)
    magnifier.paint(painter)
    painter.end()
    assert magnifier._shown_image is before, (
        "painting the box built a picture instead of drawing the one it had")


def test_a_drag_frame_carries_no_mask_and_no_rule(qtbot, screen):
    """Its objects are never drawn as a box, so no ghost is built for them."""
    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)

    magnifier = screen._magnifier
    assert magnifier.build_request().occupied is not None
    frame = magnifier.build_request(ghost=False)
    assert frame.occupied is None and frame.overlap == "replace", (
        "a drag frame copied the mask for a picture nobody draws")
    assert frame.key == magnifier.build_request().key, (
        "a frame and the box under the mouse stopped being the same request")


def test_an_edit_under_the_box_asks_again_instead_of_promising_the_old_mask(
        qtbot, screen):
    """The mask the ghost was drawn against is gone, so the promise is.

    Every edit rebinds the canvas's mask, and no one place on the screen
    owns all of them, so the box notices when it is drawn: it goes dashed
    and asks the worker for the region again rather than counting the rule
    over the box on the GUI thread.
    """
    from PySide6.QtGui import QPainter, QPixmap

    existing = rect_mask((IMG_N, IMG_N), {1: (20, 20, 24, 26)})
    screen._canvas.mask = existing
    screen._history.push(existing)
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    hover(screen, 24, 22)
    wait_for_result(qtbot, screen)
    magnifier = screen._magnifier
    faded = _faded(magnifier._shown)
    assert not magnifier.updating()

    screen._canvas.mask = rect_mask((IMG_N, IMG_N), {1: (20, 20, 27, 26)})
    surface = QPixmap(CANVAS_W, CANVAS_H)
    painter = QPainter(surface)
    magnifier.paint(painter)
    painter.end()
    assert magnifier.updating(), (
        "the box went on promising against a mask that is gone")

    qtbot.waitUntil(lambda: not magnifier.updating(), timeout=10_000)
    assert _faded(magnifier._shown) > faded, (
        "the ghost was not built again against the mask that is there now")


def test_the_box_pays_for_the_part_of_it_that_is_on_the_canvas(qtbot, screen):
    """A box wider than the window does not stretch what nobody can see.

    Item 417 let the Size box go as high as the field is wide, and the lens
    draws the box `zoom` times larger again, so most of it can lie off the
    widget. Item 380's own harness measured a median move of 183.9 ms at
    2,048 px against 53.4 ms once only the visible part is built.
    """
    from PySide6.QtCore import QRectF

    magnifier = screen._magnifier
    switch_on(screen, CodedStub({4: (20, 20, 28, 26)}))
    screen._mag_size.setValue(IMG_N)
    screen._mag_zoom.setValue(8.0)
    hover(screen, 32, 32)
    wait_for_result(qtbot, screen)

    box, lens, scale = magnifier.lens_geometry()
    part, area = magnifier._visible_part(box, lens, scale)
    assert part is not None
    inside = (part[2] - part[0]) * (part[3] - part[1])
    whole = (box[2] - box[0]) * (box[3] - box[1])
    assert inside < whole, (
        "the whole box is on the canvas, so this proves nothing: "
        f"{part} of {box}")
    covered = lens.intersected(QRectF(screen._canvas.rect()))
    assert area.contains(covered), (
        "a part of the box that is on the canvas would not be drawn")
    assert (area.width() <= covered.width() + 2 * scale + 1
            and area.height() <= covered.height() + 2 * scale + 1), (
        "more than the edge pixel either side was built off the canvas")


def test_a_box_that_fits_is_stretched_exactly_as_it_always_was(screen):
    """The levels still come from the whole region the box magnifies.

    Splitting "read the levels" from "rescale these pixels" is only worth
    anything if the first half does not change: at a box the window holds,
    every pixel is sampled and every pixel is returned, and the answer must
    be the one `normalize_uint16` gives to the byte.
    """
    image = screen._canvas.image
    box = (10, 12, 42, 44)
    mine = mm._stretch_for_box(image, box, box, 1.0, 99.9)
    theirs = engine.normalize_uint16(
        np.ascontiguousarray(image[12:44, 10:42]), 1.0, 99.9)
    np.testing.assert_array_equal(mine, theirs)


def test_the_rule_the_box_draws_is_the_rule_the_click_applies(screen):
    for index in range(screen._mag_overlap.count()):
        screen._mag_overlap.setCurrentIndex(index)
        assert screen._magnifier.overlap == screen._mag_overlap.currentData()
    screen._magnifier.set_overlap("not a rule")
    assert screen._magnifier.overlap == screen._mag_overlap.currentData()


# ---------------------------------------------------------------------------
# The busy bar says how long, once something has been measured
# ---------------------------------------------------------------------------
#
# Item 407's WHAT IS LEFT: "The busy bar has no ETA." Neither Otsu nor
# Cellpose reports steps, so the bar was indeterminate and a run that takes
# minutes on a CPU looked exactly like one that takes three seconds. What CAN
# be known is what the last run under this mode and model cost per megapixel.


def test_the_first_run_of_a_session_promises_nothing(qtbot, screen):
    """Nothing has been measured, so the bar says only that it is working."""
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=0.4)
    whole_image_on(screen, stub)
    qtbot.waitUntil(lambda: not screen._mag_progress.isHidden(), timeout=5_000)

    assert screen._magnifier.remaining_seconds() is None
    assert (screen._mag_progress.minimum(),
            screen._mag_progress.maximum()) == (0, 0), "an indeterminate bar"
    assert not screen._mag_progress.isTextVisible()
    wait_for_whole_image(qtbot, screen)
    assert screen._mag_progress.isHidden()
    assert not screen._mag_eta_timer.isActive()


def test_the_second_run_counts_down_from_what_the_first_one_cost(
        qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=0.5)
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    magnifier = screen._magnifier
    assert magnifier._image_pace, "the first run was not measured"

    screen._mag_sensitivity.setValue(2.0)
    qtbot.waitUntil(lambda: magnifier._image_estimate is not None,
                    timeout=5_000)
    left = magnifier.remaining_seconds()
    assert left is not None and 0 < left <= magnifier._image_estimate
    screen._tick_magnifier_eta()
    assert screen._mag_progress.isTextVisible()
    assert "s left" in screen._mag_progress.format()
    assert screen._mag_progress.maximum() == 1000
    assert 0 <= screen._mag_progress.value() < 1000

    wait_for_whole_image(qtbot, screen)
    assert magnifier.remaining_seconds() is None, "the run is over"


def test_an_estimate_that_runs_out_goes_back_to_saying_nothing(
        qtbot, screen, monkeypatch):
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=0.3)
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)

    magnifier = screen._magnifier
    magnifier._busy = True
    magnifier._image_started = time.monotonic() - 30.0
    magnifier._image_estimate = 2.0
    assert magnifier.remaining_seconds() is None
    screen._tick_magnifier_eta()
    assert (screen._mag_progress.minimum(),
            screen._mag_progress.maximum()) == (0, 0)
    assert not screen._mag_progress.isTextVisible()
    magnifier._busy = False


def test_the_estimate_is_per_megapixel_and_not_per_field(qtbot, screen):
    """A field nothing has been measured on is still answered for."""
    magnifier = screen._magnifier
    key = magnifier._image_key_now()
    magnifier._note_pace(key, 1_000_000, 4.0)
    assert magnifier._image_pace[magnifier._pace_key(key)] == pytest.approx(4.0)

    magnifier._note_pace(key, 0, 4.0)
    magnifier._note_pace(key, 1_000_000, 0.0)
    assert magnifier._image_pace[magnifier._pace_key(key)] == pytest.approx(4.0), (
        "a run with no pixels or no time was allowed to set the pace")


def test_a_cancelled_run_leaves_no_estimate_behind(qtbot, screen):
    stub = CodedStub({1: (20, 20, 26, 25)}, delay=0.4)
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    screen._mag_sensitivity.setValue(2.0)
    magnifier = screen._magnifier
    qtbot.waitUntil(lambda: magnifier._image_estimate is not None,
                    timeout=5_000)

    magnifier.cancel_image()
    assert magnifier._image_started is None
    assert magnifier._image_estimate is None
    assert magnifier.remaining_seconds() is None
    assert not screen._mag_eta_timer.isActive()
