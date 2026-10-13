"""Item 407's last four: a key, a promise, a stoppable run, a fast picture.

The note of 2026-09-20 left four things open, and each has a section here:

1. THE TOGGLE'S KEYBOARD SHORTCUT. M, on the Make Masks screen, pressing
   the tool-row button itself; on the shortcut map under the card's own
   caption, which the generated catalog already translates.
2. THE OVERLAP PREVIEW UNDER WHOLE IMAGE. A click there adds one WHOLE
   object, most of which may lie outside the box, so the rule's answer is
   computed over the object's own extent -- and must be exactly what the
   click then adds, for every rule.
3. WHOLE-IMAGE CELLPOSE ON A CPU. Minutes per field, measured; so the run
   counts its tiles, says how far it has got from its first tile on, and
   stops between two tiles when it is cancelled rather than at the end.
4. THE BOX'S PICTURE AT THE LARGEST SIZES. One table look-up and an integer
   blend in place of two float passes, byte-identical to them.

The canvas geometry and the coded-field stub are item 407's, imported from
its test module, so a pixel here means what it means there.
"""
from __future__ import annotations

import threading
import time
from functools import partial

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QShortcut

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    IMG_N,
    CodedStub,
    click,
    fields,  # noqa: F401 - a fixture, used by name
    hover,
    rect_mask,
    screen,  # noqa: F401 - a fixture, used by name
    switch_on,
    wait_for_whole_image,
    whole_image_on,
)


# ---------------------------------------------------------------------------
# 1. M turns the magnifier on and off
# ---------------------------------------------------------------------------

def _key(screen, keys: str) -> QShortcut:
    """The screen's own shortcut for ``keys``."""
    found = [sc for sc in screen.findChildren(QShortcut)
             if sc.key() == QKeySequence(keys)]
    assert len(found) == 1, f"{keys} is bound {len(found)} times here"
    return found[0]


def test_m_is_on_the_map_for_the_make_masks_screen_only():
    from spacr.qt.shortcuts import EVERYWHERE, SCREEN_SHORTCUTS, SHORTCUTS

    spec = [s for s in SCREEN_SHORTCUTS if s.keys == "M"]
    assert len(spec) == 1
    spec = spec[0]
    assert spec.scope == "the Make Masks screen" != EVERYWHERE
    assert spec.category == "Make Masks"
    assert spec.label == "Live magnifier", (
        "the card's own caption, so the map and the card say one thing")
    assert not [s for s in SHORTCUTS if s.keys == "M"]


def test_m_clashes_with_nothing_and_asks_for_no_modifier_macos_keeps():
    """Bare, like B E W D V Z R: Qt turns Ctrl into Command on macOS, and
    Cmd+M minimises the window there, so M carries no modifier at all."""
    from spacr.qt.shortcuts import mapped

    sequence = QKeySequence("M")
    assert sequence.count() == 1
    combined = sequence[0]
    assert combined.keyboardModifiers().value == 0
    holders = [s for s in mapped()
               if QKeySequence(s.keys) == sequence
               and ("Make Masks" in s.scope or s.scope.startswith("anywhere"))]
    assert len(holders) == 1, holders


def test_m_presses_the_magnifier_button_on_and_off(qtbot, screen):
    shortcut = _key(screen, "M")
    button, magnifier = screen._btn_magnifier, screen._magnifier
    assert not button.isChecked() and not magnifier.enabled

    shortcut.activated.emit()
    assert button.isChecked() and magnifier.enabled
    assert "Magnifier on" in screen._status_label.text(), (
        "the key went around the button's own toggled handler")

    shortcut.activated.emit()
    assert not button.isChecked() and not magnifier.enabled
    assert "Magnifier off" in screen._status_label.text()


def test_m_does_nothing_while_the_button_cannot_be_pressed(qtbot, screen):
    screen._btn_magnifier.setEnabled(False)
    assert screen._toggle_magnifier_key() is False
    assert not screen._magnifier.enabled


def test_m_is_on_the_screens_own_shortcut_panel(screen):
    keys, does = screen._shortcut_rows["M"]
    assert keys.text() == "M" and does.text() == "Live magnifier"


# ---------------------------------------------------------------------------
# 2. Under Whole image the box promises what a click on the object adds
# ---------------------------------------------------------------------------

def _slice(screen):
    """``(picture, box)`` of the whole-image slice the box last drew."""
    view = screen._magnifier._image_view
    assert view is not None, "the box drew no whole-image slice"
    return view[2], view[1][0]


def _alpha_at(picture, box, img_x, img_y) -> int:
    return QColor(picture.pixelColor(img_x - box[0], img_y - box[1])).alpha()


def _paint_box(screen) -> None:
    """Run the box's own paint into a picture the canvas's size.

    Not ``grab()``: grabbing a widget that was never shown lays it out
    first, the canvas takes its layout's size, and the image-to-canvas
    mapping every hover here is written in moves under the next one.
    """
    canvas = screen._canvas
    target = QImage(canvas.size(), QImage.Format_ARGB32)
    target.fill(0)
    painter = QPainter(target)
    try:
        screen._magnifier.paint(painter)
    finally:
        painter.end()


def _drawn_over(qtbot, screen, img_x, img_y):
    hover(screen, img_x, img_y)
    _paint_box(screen)
    return _slice(screen)


BAR = (2, 40, 62, 43)


def _with_mask(screen, rects: dict) -> None:
    existing = rect_mask((IMG_N, IMG_N), rects)
    screen._canvas.mask = existing
    screen._history.push(existing)


def _rule(screen, name: str) -> None:
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData(name))
    assert screen._magnifier.overlap == name


def test_skip_ghosts_an_object_that_touches_the_mask_off_the_box(
        qtbot, screen):
    """The case the box's own slice cannot answer: the bar touches a mask
    object 50 px away, outside the box, and Skip leaves ALL of it out."""
    whole_image_on(screen, CodedStub({4: BAR}))
    wait_for_whole_image(qtbot, screen)
    _with_mask(screen, {1: (58, 40, 62, 43)})
    _rule(screen, "skip")

    picture, box = _drawn_over(qtbot, screen, 8, 41)
    assert box[2] <= 58, "the touching pixels must be off the box"
    inside = _alpha_at(picture, box, 8, 41)
    assert inside == 140 // 4, (
        f"Skip adds none of the bar, and the box drew it at {inside}")

    _rule(screen, "replace")
    picture, box = _drawn_over(qtbot, screen, 8, 41)
    assert _alpha_at(picture, box, 8, 41) == 140, (
        "Replace takes nothing, so the object under the mouse is solid")


def test_clip_ghosts_the_piece_that_is_not_the_largest_even_off_the_mask(
        qtbot, screen):
    """A mask object splits the bar; Clip keeps the LARGER piece, which is
    off the box, so the piece under the mouse -- on no mask pixel at all --
    is not what a click adds."""
    whole_image_on(screen, CodedStub({4: BAR}))
    wait_for_whole_image(qtbot, screen)
    _with_mask(screen, {1: (12, 40, 14, 43)})
    _rule(screen, "clip")

    picture, box = _drawn_over(qtbot, screen, 6, 41)
    assert _alpha_at(picture, box, 6, 41) == 140 // 4
    if box[2] > 20:
        assert _alpha_at(picture, box, 20, 41) == 140, (
            "the piece a click adds is drawn solid")

    click(screen, 6, 41)
    added = screen._canvas.mask == 2
    assert not added[41, 6] and added[41, 30], (
        "the click added the largest piece, as the box said it would")


@pytest.mark.parametrize("rule", ["clip", "skip", "replace"])
def test_the_promise_is_exactly_what_the_click_adds(qtbot, screen, rule):
    """Pixel for pixel, over the object's whole extent, for every rule."""
    whole_image_on(screen, CodedStub({4: BAR, 6: (20, 10, 30, 30)}))
    wait_for_whole_image(qtbot, screen)
    _with_mask(screen, {1: (12, 40, 14, 43), 2: (24, 20, 40, 24)})
    _rule(screen, rule)

    for target, (x, y) in ((4, (6, 41)), (6, (22, 12))):
        hover(screen, x, y)
        _paint_box(screen)
        magnifier = screen._magnifier
        result = magnifier._image_result
        window = mm._object_window(result, target)
        wx0, wy0, wx1, wy1 = window
        body = result.labels[wy0:wy1, wx0:wx1] == target
        lost = magnifier._image_promise(result, target)
        promised = body if lost is None else body & ~lost
        before = screen._canvas.mask.copy()
        click(screen, x, y)
        after = screen._canvas.mask
        added = (after != before)[wy0:wy1, wx0:wx1]
        np.testing.assert_array_equal(
            added, promised,
            err_msg=f"{rule}: the box promised other pixels than it added")
        assert not (after != before)[:wy0].any() and \
            not (after != before)[wy1:].any()
        screen._on_undo()


def test_a_mask_edit_redraws_the_promise_and_asks_no_model(qtbot, screen):
    stub = CodedStub({4: BAR})
    whole_image_on(screen, stub)
    wait_for_whole_image(qtbot, screen)
    _with_mask(screen, {1: (58, 40, 62, 43)})
    _rule(screen, "skip")
    picture, box = _drawn_over(qtbot, screen, 8, 41)
    assert _alpha_at(picture, box, 8, 41) == 140 // 4

    cleared = np.zeros((IMG_N, IMG_N), np.uint8)
    screen._canvas.mask = cleared
    screen._history.push(cleared)
    picture, box = _drawn_over(qtbot, screen, 8, 41)
    assert _alpha_at(picture, box, 8, 41) == 140, (
        "the object stopped touching the mask and the box still ghosted it")
    assert len(stub.calls) == 1


def test_the_whole_image_outlines_are_cut_from_the_workers_picture(
        qtbot, screen):
    """Built once on the worker, sliced per move: not outlined again."""
    whole_image_on(screen, CodedStub({4: BAR, 6: (20, 10, 30, 30)}))
    wait_for_whole_image(qtbot, screen)
    result = screen._magnifier._image_result
    assert result.overlay is not None
    assert result.overlay.shape == (IMG_N, IMG_N, 4)
    assert result.extents is not None and len(result.extents) == 6
    np.testing.assert_array_equal(
        result.overlay, mm._candidate_overlay(result.labels,
                                              result.request.colour))
    assert mm._object_window(result, 4) == BAR
    assert mm._object_window(result, 5) is None


# ---------------------------------------------------------------------------
# 3. A whole-image Cellpose run counts its tiles and stops between them
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class _CountingNet(torch.nn.Module):
    """A network that answers like Cellpose's, instantly, and is counted."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        n, _c, h, w = x.shape
        return torch.zeros(n, 3, h, w), torch.zeros(n, 256)


@pytest.mark.parametrize("shape, diameter", [
    ((64, 64), 0), ((256, 256), 0), ((300, 500), 0), ((1994, 1994), 0),
    ((1994, 1994), 60), ((200, 120), 20),
])
def test_the_tile_count_is_the_one_cellpose_runs(shape, diameter):
    """Against `cellpose.core.run_net` itself, with a network that counts."""
    core = pytest.importorskip("cellpose.core")
    net = _CountingNet()
    rescale = 30.0 / diameter if diameter else None
    image = np.zeros((1,) + tuple(shape) + (3,), np.float32)
    core.run_net(net, image, batch_size=1, bsize=256, tile_overlap=0.1,
                 rsz=rescale)
    assert mm._cellpose_tile_count(shape, diameter) == net.calls


class _SlowModel:
    """Cellpose's shape, minus Cellpose: a counted net, a tile at a time."""

    def __init__(self, tiles: int, pause: float = 0.02):
        self.net = _CountingNet()
        self.tiles = int(tiles)
        self.pause = float(pause)


def _tiled_detect(image, model, **_settings):
    """A stand-in for `cellpose_detect` that runs ``model.tiles`` tiles."""
    for _tile in range(model.tiles):
        time.sleep(model.pause)
        model.net(torch.zeros(1, 3, 8, 8))
    return np.zeros(np.asarray(image).shape[:2], np.int32), None, None


@pytest.fixture
def slow_cellpose(monkeypatch):
    """Cellpose mode, fifty tiles long, with no Cellpose installed."""
    model = _SlowModel(50)
    monkeypatch.setattr(mm, "cellpose_detect", _tiled_detect)
    monkeypatch.setattr(mm, "_cellpose_tile_count",
                        lambda shape, diameter=0, **_kw: model.tiles)
    return model


def _request(ticket):
    field = np.ones((32, 32), np.uint16)
    return mm._MagnifierRequest(
        key=("k",), crop=field, box=(0, 0, 32, 32), shape=(32, 32),
        mode="cellpose", sensitivity=0.0, bright=True, min_area=0,
        model_name="cpsam", diameter=0, colour=(0, 0, 0), scope="image",
        ticket=ticket)


def test_a_cancelled_run_stops_at_the_next_tile_and_is_not_an_error(
        slow_cellpose):
    ticket = mm._RunTicket()
    outcome = {}

    def run():
        try:
            outcome["value"] = mm._segment_region(
                _request(ticket), load_model=lambda _name: slow_cellpose)
        except BaseException as exc:                     # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 10
    while ticket.done < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert ticket.total == 50
    ticket.cancel()
    worker.join(5)
    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), mm._RunCancelled), (
        "a cancel became a failure, and a failure falls back to Otsu: "
        + repr(outcome))
    assert slow_cellpose.net.calls < 50
    assert slow_cellpose.net.calls <= ticket.done + 1


def test_the_hook_is_taken_off_the_network_afterwards(slow_cellpose):
    ticket = mm._RunTicket()
    mm._segment_region(_request(ticket),
                       load_model=lambda _name: slow_cellpose)
    assert ticket.done == 50
    assert not slow_cellpose.net._forward_pre_hooks, (
        "the next run would be counted twice, or stopped by this one")


def test_the_pace_is_this_runs_own_and_stops_at_the_last_tile(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(mm.time, "monotonic", lambda: now[0])
    ticket = mm._RunTicket(total=10)
    ticket.step()
    assert ticket.remaining_seconds() is None, "one tile is not a pace"
    now[0] += 2.0
    ticket.step()
    assert ticket.remaining_seconds() == pytest.approx(2.0 * 9)
    now[0] += 1.0
    assert ticket.remaining_seconds() == pytest.approx(2.0 * 9 - 1.0)
    for _ in range(8):
        now[0] += 2.0
        ticket.step()
    assert ticket.done == 10
    assert ticket.remaining_seconds() is None, (
        "what follows the last tile is not made of tiles")


def _cellpose_image_on(screen, model) -> None:
    magnifier = screen._magnifier
    magnifier.segment = partial(mm._segment_region,
                                load_model=lambda _name: model)
    magnifier.set_mode("cellpose")
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    screen._btn_magnifier.setChecked(True)


def test_the_bar_counts_tiles_from_the_first_run_and_cancel_stops_them(
        qtbot, screen, slow_cellpose):
    slow_cellpose.pause = 0.05
    magnifier = screen._magnifier
    _cellpose_image_on(screen, slow_cellpose)
    qtbot.waitUntil(lambda: (magnifier.image_progress() or (0,))[0] >= 3,
                    timeout=10_000)

    screen._tick_magnifier_eta()
    bar = screen._mag_progress
    assert bar.maximum() == 50, "the bar is not counting tiles"
    assert 0 < bar.value() < 50
    assert "s left" in bar.format(), (
        "the first run of a session says nothing about time: " + bar.format())

    screen._mag_cancel.click()
    qtbot.waitUntil(magnifier._image_worker.idle, timeout=3_000)
    stopped_at = slow_cellpose.net.calls
    assert stopped_at < 50, "Cancel let the model run to the end"
    qtbot.wait(200)
    assert slow_cellpose.net.calls == stopped_at
    assert not magnifier._busy and not bar.isVisible()
    assert "cancelled" in screen._status_label.text()
    assert "could not" not in screen._status_label.text()


def test_leaving_whole_image_stops_the_run_too(qtbot, screen, slow_cellpose):
    magnifier = screen._magnifier
    _cellpose_image_on(screen, slow_cellpose)
    qtbot.waitUntil(lambda: magnifier.image_progress() is not None,
                    timeout=10_000)
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("region"))
    qtbot.waitUntil(magnifier._image_worker.idle, timeout=3_000)
    assert slow_cellpose.net.calls < 50


def test_a_model_that_does_not_tile_keeps_the_old_estimate(qtbot, screen):
    """Otsu counts nothing, so the bar is the per-megapixel pace's."""
    whole_image_on(screen, CodedStub({4: BAR}, delay=0.3))
    assert screen._magnifier.image_progress() is None


# ---------------------------------------------------------------------------
# 4. The box's picture: one look-up and an integer blend, to the byte
# ---------------------------------------------------------------------------

def _slow_picture(image, box, part, mask_part, lo, hi):
    stretched = mm._stretch_for_box(image, box, part, lo, hi)
    return mm._rgb32(engine.overlay_mask(stretched, mask_part, alpha=0.5))


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fast_box_picture_is_the_slow_one_to_the_byte(dtype, seed):
    rng = np.random.default_rng(seed)
    top = np.iinfo(dtype).max
    image = rng.integers(0, top, (300, 260)).astype(dtype)
    image[::3] = (image[::3] // 7).astype(dtype)
    mask = np.zeros((300, 260), np.uint16)
    for label in range(1, 700):
        y, x = rng.integers(0, 290, 2)
        mask[y:y + 9, x % 250:x % 250 + 7] = label
    box = (10, 20, 250, 290)
    part = (30, 41, 199, 277)
    x0, y0, x1, y1 = part
    for lo, hi in ((1.0, 99.9), (0.0, 100.0), (20.0, 60.0)):
        fast = mm._box_picture(image, box, part, mask[y0:y1, x0:x1], lo, hi)
        slow = _slow_picture(image, box, part, mask[y0:y1, x0:x1], lo, hi)
        assert fast.dtype == np.uint32 and fast.flags.c_contiguous
        np.testing.assert_array_equal(fast, slow)


def test_an_empty_mask_and_a_flat_field_are_the_same_picture_too():
    image = np.full((50, 40), 1234, np.uint16)
    mask = np.zeros((50, 40), np.uint8)
    box = part = (0, 0, 40, 50)
    np.testing.assert_array_equal(
        mm._box_picture(image, box, part, mask, 1.0, 99.9),
        _slow_picture(image, box, part, mask, 1.0, 99.9))


def test_a_field_the_table_cannot_index_takes_the_two_passes():
    """A float field has no finite set of values to tabulate."""
    image = np.linspace(0, 1, 40 * 30, dtype=np.float32).reshape(30, 40)
    assert mm._box_grey_table(image.dtype, 0.1, 0.9) is None
    assert mm._box_grey_table(np.dtype(np.uint32), 0, 9) is None


def test_the_table_is_one_entry_and_follows_the_levels():
    first = mm._box_grey_table(np.dtype(np.uint16), 100.0, 5000.0)
    assert mm._box_grey_table(np.dtype(np.uint16), 100.0, 5000.0) is first
    mm._box_grey_table(np.dtype(np.uint16), 200.0, 5000.0)
    assert len(mm._BOX_GREY_TABLE) == 1


def test_a_thinned_picture_covers_the_same_rect(screen):
    """Below a device pixel per image pixel the picture is thinned, and a
    step that does not divide the part still covers it to the edge."""
    magnifier = screen._magnifier
    assert magnifier._picture_step(2.0) == 1
    assert magnifier._picture_step(1.0) == 1
    ratio = float(screen._canvas.devicePixelRatioF()) or 1.0
    assert magnifier._picture_step(0.3 / ratio) == 3

    image = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    mask = np.zeros((100, 100), np.uint8)
    thin = mm._box_picture(image, (0, 0, 100, 100), (0, 0, 100, 100),
                           mask, 1.0, 99.9, step=3)
    assert thin.shape == (34, 34)
    full = mm._box_picture(image, (0, 0, 100, 100), (0, 0, 100, 100),
                           mask, 1.0, 99.9)
    np.testing.assert_array_equal(thin, full[::3, ::3])


def test_the_box_still_draws_the_field_and_its_mask(qtbot, screen):
    """End to end through paint: the lens holds the stretched field, with
    the mask blended over it, exactly as the two passes drew it."""
    _with_mask(screen, {1: (28, 28, 36, 36)})
    switch_on(screen, CodedStub({}))
    hover(screen, 32, 32)
    canvas = screen._canvas
    target = QImage(canvas.size(), QImage.Format_ARGB32)
    target.fill(0)
    painter = QPainter(target)
    screen._magnifier.paint(painter)
    painter.end()
    box, lens, scale = screen._magnifier.lens_geometry()
    part, _area = screen._magnifier._visible_part(box, lens, scale)
    x0, y0, x1, y1 = part
    expected = _slow_picture(canvas.image, box, part,
                             canvas.mask[y0:y1, x0:x1],
                             canvas.norm_lo, canvas.norm_hi)
    for img_x, img_y in ((30, 30), (20, 22), (33, 35)):
        at = lens.topLeft() + QPointF((img_x - box[0] + 0.5) * scale,
                                      (img_y - box[1] + 0.5) * scale)
        drawn = target.pixel(int(at.x()), int(at.y())) & 0xFFFFFF
        assert drawn == int(expected[img_y - y0, img_x - x0]) & 0xFFFFFF, (
            img_x, img_y)
