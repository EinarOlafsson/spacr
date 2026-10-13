"""The live magnifier answers sensibly at the edges of what it is asked.

Pins what the box does when it is asked for something it cannot give, or
asked twice for the same thing:

* switched off, a click, a right-click, a press, a drag and a cancel do
  nothing and say so by returning False;
* over an empty canvas there is no whole-image run to start;
* an unknown scope is the region scope; a size outside the field's range is
  pulled back into it and the Size box is told;
* a region already on screen is not asked for again;
* the box magnifies the enhanced, or the normalized, field when the canvas
  shows that one, and a comparison covers the box under the mouse;
* a model that could not run is said once, not on every region;
* a whole-image run that was cancelled, or that belonged to an older ticket,
  leaves the box idle without a word; one whose model fell back says so
  instead of counting objects;
* a whole-image click while the run is on its way says it is still running;
* a Cellpose 3 model at diameter 0 says it will estimate the size, once; a
  secondary run with no primary mask says to load one;
* a model that answers with the wrong shape is reported, not committed;
* a box entirely off the canvas draws nothing, and one below one device
  pixel per image pixel is drawn thinned;
* the whole-image slice outlines objects even when the worker drew no
  picture of them, and the Overlap promise is None for an id that is not
  there.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter, QPixmap, QWheelEvent

from spacr.qt import cpu_modes
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

N = 100
SIDE = 400


def _field() -> np.ndarray:
    return np.arange(N * N, dtype=np.uint16).reshape(N, N)


def _objects() -> np.ndarray:
    labels = np.zeros((N, N), dtype=np.int32)
    labels[40:60, 40:60] = 1
    labels[5:15, 5:15] = 2
    return labels


def _at(x: int, y: int) -> QPointF:
    return QPointF((x + 0.5) * SIDE / N, (y + 0.5) * SIDE / N)


@pytest.fixture
def lens(qtbot, qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(SIDE, SIDE)
    canvas.set_image_and_mask(_field(), np.zeros((N, N), dtype=np.uint16))
    magnifier = mm._LiveMagnifier(canvas, canvas)
    canvas.magnifier = magnifier
    calls = []

    def stub(request):
        calls.append(request)
        if request.scope == "image":
            return _objects()
        return np.zeros(request.crop.shape, dtype=np.int32)

    magnifier.segment = stub
    magnifier.calls = calls
    magnifier.set_size(32)
    canvas.refresh()
    yield canvas, magnifier
    magnifier.close()
    canvas.close_enhancer()


def _statuses(magnifier):
    said = []
    magnifier.status.connect(said.append)
    return said


def _image_request(magnifier, key, **extra):
    return mm._MagnifierRequest(
        key=key, crop=None, box=(0, 0, N, N), shape=(N, N), mode="otsu",
        sensitivity=0.0, bright=True, min_area=0, model_name="cpsam",
        diameter=0, colour=(255, 0, 0), scope="image", **extra)


def _image_result(request, mode="otsu", note="", overlay=True):
    labels = _objects()
    return mm._MagnifierResult(
        request, labels, mode, note,
        mm._candidate_overlay(labels, (255, 0, 0)) if overlay else None,
        mm._object_count(labels), extents=mm._object_extents(labels))


# ---------------------------------------------------------------------------
# Switched off, or with nothing under it
# ---------------------------------------------------------------------------

def test_a_magnifier_that_is_off_does_nothing_it_is_asked(lens):
    canvas, magnifier = lens
    removed, strokes = [], []
    magnifier.remove_requested.connect(lambda x, y: removed.append((x, y)))
    magnifier.drag_ready.connect(strokes.append)

    assert magnifier.click() is False
    assert magnifier.remove() is False
    assert magnifier.press() is False
    assert magnifier.cancel_image() is False
    magnifier.drag()
    magnifier._stroke_show(final=True)
    magnifier.set_scope("image")
    assert magnifier.click() is False

    assert removed == [] and strokes == []
    assert magnifier.calls == []
    assert int(canvas.mask.max()) == 0


def test_an_empty_canvas_has_no_whole_image_run_to_start(qtbot,
                                                         qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    magnifier = mm._LiveMagnifier(canvas, canvas)
    said = _statuses(magnifier)
    try:
        magnifier.set_scope("image")
        magnifier.set_enabled(True)
        assert magnifier._image_key_now() is None
        assert magnifier._image_key is None
        assert magnifier._image_worker.idle()
        assert said == []
    finally:
        magnifier.set_enabled(False)
        magnifier.close()


def test_a_held_l_that_repeats_is_still_held(lens):
    canvas, magnifier = lens
    magnifier._lock_key_down = True
    repeat = QKeyEvent(QEvent.KeyRelease, Qt.Key_L, Qt.NoModifier, "l", True)
    magnifier.eventFilter(canvas, repeat)
    assert magnifier._lock_key_down is True
    release = QKeyEvent(QEvent.KeyRelease, Qt.Key_L, Qt.NoModifier, "l", False)
    magnifier.eventFilter(canvas, release)
    assert magnifier._lock_key_down is False


# ---------------------------------------------------------------------------
# Scope, size, and asking again
# ---------------------------------------------------------------------------

def test_an_unknown_scope_is_the_region_scope(lens):
    _canvas, magnifier = lens
    magnifier.set_scope("image")
    assert magnifier.scope == "image"
    magnifier.set_scope("the moon")
    assert magnifier.scope == "region"


def test_a_size_past_the_field_is_pulled_back_and_the_size_box_told(lens):
    _canvas, magnifier = lens
    told = []
    magnifier.size_changed.connect(told.append)
    low, high = magnifier.size_range()
    magnifier.size = high + 500
    magnifier._sync_size_range()
    assert magnifier.size == high
    assert told == [high]
    told.clear()
    magnifier._sync_size_range()
    assert told == []


def test_a_region_already_on_screen_is_not_asked_for_again(qtbot, lens):
    _canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    qtbot.waitUntil(lambda: magnifier._shown is not None
                    and not magnifier.updating(), timeout=10_000)
    asked = len(magnifier.calls)
    submitted = []
    real = magnifier._worker.submit
    magnifier._worker.submit = lambda *a, **k: submitted.append(a) or real(*a, **k)
    magnifier.refresh()
    assert submitted == []
    assert len(magnifier.calls) == asked
    magnifier.set_enabled(False)
    magnifier._ask_again()
    assert submitted == [] and magnifier._asking is False


def test_the_box_magnifies_the_field_the_canvas_shows(lens):
    canvas, magnifier = lens
    assert magnifier.detector_field() is canvas.image
    canvas.detect_on_normalized = True
    np.testing.assert_array_equal(
        magnifier.detector_field(),
        engine.normalize_for_detection(canvas.image, canvas.norm_lo,
                                       canvas.norm_hi))
    canvas.enhance_display = True
    assert magnifier.detector_field() is canvas.enhanced_picture()


def test_a_comparison_covers_the_box_under_the_mouse(lens):
    _canvas, magnifier = lens
    assert magnifier.compare_box() == (0, 0, N, N)
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    assert magnifier.compare_box() == engine._magnifier_box(
        (N, N), *magnifier._cursor, magnifier.size)
    assert magnifier.compare_box() != (0, 0, N, N)


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------

def test_a_model_that_could_not_run_is_said_once(lens):
    _canvas, magnifier = lens
    said = _statuses(magnifier)
    request = _image_request(magnifier, ("k",))._replace(mode="cellpose")
    result = _image_result(request, mode="otsu", note="no weights")
    assert magnifier._note_fallback(request, result) is True
    assert magnifier._note_fallback(request, result) is False
    assert len(said) == 1 and "no weights" in said[0]


def test_a_cancelled_or_superseded_whole_image_run_says_nothing(lens):
    _canvas, magnifier = lens
    said = _statuses(magnifier)
    busy = []
    magnifier.busy_changed.connect(busy.append)

    magnifier._image_key = ("run", "image")
    request = _image_request(magnifier, ("run", "image"),
                             ticket=mm._RunTicket())
    magnifier._on_image_delivered(request, None, mm._RunCancelled())
    assert magnifier._image_key is None
    assert magnifier._image_result is None
    assert said == []
    assert magnifier._image_halted is None
    assert not busy or busy[-1] is False


def test_a_whole_image_run_whose_model_fell_back_says_why_not_how_many(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.set_scope("image")
    magnifier._stop_image()
    said = _statuses(magnifier)
    key = ("stale", "image") + ("cellpose",)
    magnifier._image_key = key
    request = _image_request(magnifier, key)._replace(mode="cellpose")
    magnifier._on_image_delivered(
        request, _image_result(request, mode="otsu", note="no GPU"), None)
    assert magnifier._image_result is not None
    assert magnifier._image_result.request.key == key
    assert len(said) == 1
    assert "no GPU" in said[0] and "object(s) found" not in said[0]


def test_a_whole_image_click_while_the_run_is_on_its_way_says_so(qtbot, lens):
    canvas, magnifier = lens
    started, gate = threading.Event(), threading.Event()

    def slow(request):
        started.set()
        assert gate.wait(10)
        return _objects()

    magnifier.segment = slow
    magnifier.set_scope("image")
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    assert started.wait(10)
    said = _statuses(magnifier)
    assert magnifier.click() is True
    assert said and "still being segmented" in said[-1]
    gate.set()
    qtbot.waitUntil(lambda: magnifier._image_result is not None,
                    timeout=10_000)
    assert magnifier._image_result.count == 2


def test_cellpose3_at_diameter_zero_says_it_estimates_the_size_once(
        qtbot, lens):
    canvas, magnifier = lens
    magnifier._context = lambda: {"model_name": "cellpose3:cyto3",
                                  "diameter": 0}
    magnifier.mode = "cellpose"
    said = _statuses(magnifier)
    magnifier.set_scope("image")
    magnifier.set_enabled(True)
    qtbot.waitUntil(lambda: magnifier._image_result is not None,
                    timeout=10_000)
    notes = [s for s in said if s == mm._cellpose3_auto_diameter_note()]
    assert len(notes) == 1
    assert magnifier._said_diameter_note is True


def test_a_secondary_run_with_no_primary_mask_asks_for_one(lens):
    canvas, magnifier = lens
    magnifier._context = lambda: {"primary_token": ("p",)}
    magnifier.mode = cpu_modes.SECONDARY
    said = _statuses(magnifier)
    magnifier.set_scope("image")
    magnifier.set_enabled(True)
    assert "Load a primary mask before growing secondary objects." in said
    assert magnifier._image_key is None
    assert magnifier._image_halted is not None
    assert magnifier.calls == []


def test_a_model_answering_the_wrong_shape_is_reported(qtbot, lens):
    canvas, magnifier = lens
    magnifier.segment = lambda request: np.zeros((3, 3), dtype=np.int32)
    said = _statuses(magnifier)
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    qtbot.waitUntil(lambda: bool(said), timeout=10_000)
    assert "returned labels shaped (3, 3)" in said[-1]
    assert magnifier._shown is None
    assert int(canvas.mask.max()) == 0


def test_the_kept_whole_image_objects_survive_an_unreadable_ceiling(
        monkeypatch, lens):
    from spacr.qt import preferences

    _canvas, magnifier = lens

    def broken():
        raise OSError("settings unreadable")

    monkeypatch.setattr(preferences, "get_cache_ceiling_mb", broken)
    request = _image_request(magnifier, (0, "image", "otsu"))
    magnifier._field_name = "a.tif"
    magnifier._keep_image_result(_image_result(request))
    assert len(magnifier._image_cache) == 1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _paint(canvas, magnifier) -> QImage:
    picture = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32)
    picture.fill(0)
    painter = QPainter(picture)
    magnifier.paint(painter)
    painter.end()
    return picture


def _painted(picture: QImage) -> int:
    return sum(1 for y in range(0, picture.height(), 8)
               for x in range(0, picture.width(), 8)
               if picture.pixel(x, y) != 0)


def test_a_box_entirely_off_the_canvas_draws_nothing(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    assert _painted(_paint(canvas, magnifier)) > 0
    magnifier._anchor = QPointF(-5000.0, -5000.0)
    assert _painted(_paint(canvas, magnifier)) == 0


def test_a_canvas_with_no_picture_has_no_lens(monkeypatch, lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    assert magnifier.lens_geometry() is not None
    monkeypatch.setattr(canvas, "pixmap", lambda: QPixmap())
    assert magnifier.lens_geometry() is None
    assert _painted(_paint(canvas, magnifier)) == 0


def test_a_box_below_a_device_pixel_per_image_pixel_is_drawn_thinned(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    magnifier.zoom = 0.1
    box, lens_rect, scale = magnifier.lens_geometry()
    assert magnifier._picture_step(scale) > 1
    assert _painted(_paint(canvas, magnifier)) > 0


def test_the_whole_image_slice_is_outlined_without_the_workers_picture(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    magnifier._cursor = (50, 50)
    request = _image_request(magnifier, (0, "image"))
    magnifier._image_result = _image_result(request, overlay=False)
    view = magnifier._image_slice((30, 30, 70, 70))
    assert view is not None and (view.width(), view.height()) == (40, 40)
    centre = view.pixelColor(20, 20)
    corner = view.pixelColor(0, 0)
    assert centre.alpha() >= 140
    assert corner.alpha() == 0


def test_the_overlap_promise_is_none_for_an_id_that_is_not_there(lens):
    canvas, magnifier = lens
    mask = np.zeros((N, N), dtype=np.uint16)
    mask[45:50, 45:50] = 7
    canvas.mask = mask
    request = _image_request(magnifier, (0, "image"))
    result = _image_result(request)
    magnifier.overlap = "clip"
    assert magnifier._image_promise(result, 9) is None
    secondary = result._replace(mode=cpu_modes.SECONDARY)
    lost = magnifier._image_promise(secondary, 1)
    assert lost is not None
    assert lost.shape == (20, 20)
    assert lost[7, 7] and not lost[0, 0]


# ---------------------------------------------------------------------------
# The canvas's mouse, with the box on
# ---------------------------------------------------------------------------

def _press(pos, button, buttons, modifiers=Qt.NoModifier):
    return QMouseEvent(QEvent.MouseButtonPress, pos, pos, button, buttons,
                       modifiers)


def test_a_shift_wheel_with_no_turn_leaves_the_size_alone(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    size = magnifier.size
    still = QWheelEvent(_at(50, 50), _at(50, 50), QPoint(0, 0), QPoint(0, 0),
                        Qt.NoButton, Qt.ShiftModifier, Qt.NoScrollPhase,
                        False)
    canvas.wheelEvent(still)
    assert still.isAccepted()
    assert magnifier.size == size


def test_the_lock_chord_with_another_button_held_does_not_toggle(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier._lock_key_down = True
    canvas.mousePressEvent(_press(_at(50, 50), Qt.RightButton,
                                  Qt.RightButton | Qt.LeftButton,
                                  Qt.ControlModifier))
    assert magnifier.locked is False
    assert Qt.RightButton in canvas._swallowed


def test_a_second_button_pressed_without_ctrl_is_read_as_its_own_press(lens):
    canvas, magnifier = lens
    mask = np.zeros((N, N), dtype=np.uint16)
    mask[40:60, 40:60] = 3
    canvas.set_image_and_mask(_field(), mask)
    x, y = 50, 50
    canvas._ctrl_click = None
    point = QPointF(canvas._image_to_canvas(x + 0.5, y + 0.5))
    canvas.mousePressEvent(_press(point, Qt.RightButton,
                                  Qt.RightButton | Qt.LeftButton))
    assert canvas._sweeping is True
    assert Qt.RightButton not in canvas._swallowed


def test_a_move_after_a_ctrl_click_only_moves_the_readout(lens):
    canvas, magnifier = lens
    mask = np.zeros((N, N), dtype=np.uint16)
    mask[40:60, 40:60] = 3
    canvas.set_image_and_mask(_field(), mask)
    point = QPointF(canvas._image_to_canvas(50.5, 50.5))
    canvas.mousePressEvent(_press(point, Qt.RightButton, Qt.RightButton,
                                  Qt.ControlModifier))
    assert canvas._ctrl_click == Qt.RightButton
    edited = canvas.mask.copy()
    away = QPointF(canvas._image_to_canvas(20.5, 20.5))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, away, away,
                                      Qt.NoButton, Qt.RightButton,
                                      Qt.ControlModifier))
    np.testing.assert_array_equal(canvas.mask, edited)
    assert canvas._sweeping is False


def test_leaving_the_canvas_puts_the_box_away(lens):
    canvas, magnifier = lens
    magnifier.set_enabled(True)
    magnifier.hover(_at(50, 50))
    assert magnifier._cursor is not None
    canvas.leaveEvent(QEvent(QEvent.Leave))
    assert magnifier._cursor is None
    assert magnifier.lens_geometry() is None
