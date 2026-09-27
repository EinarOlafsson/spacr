"""Make Masks' console and canvas at their edges.

Pins what a curator sees when things happen out of the usual order:

* the console takes a line from a worker thread, ignores a blank one, keeps
  a finished progress bar ahead of the line that interrupts it, survives a
  worker that prints after the window is gone, and draws a bar that was
  kicked twice by a race exactly once;
* the worker-side helpers that build the enhanced and compare pictures give
  up on a cancelled request and hand an un-enhanced field back as it is;
* the canvas with no field answers None everywhere instead of raising; an
  enhancement that FAILED is said once and not retried on every repaint, and
  the detectors are told why rather than handed raw pixels;
* the readout falls back to the pixel when the object table cannot be built,
  and is not re-read after the mouse has left;
* a Ctrl+click off the mask changes nothing;
* a ruler line is drawn on the canvas.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import detect_chain
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.theme import active_palette

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64


def _mouse(kind, x, y, button=Qt.LeftButton, buttons=Qt.LeftButton,
           modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def block_image() -> np.ndarray:
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[20:40, 20:40] = 30000
    return img


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    made = mm._MaskCanvas()
    qtbot.addWidget(made)
    made.resize(CANVAS_W, CANVAS_H)
    mask = np.zeros((IMG_N, IMG_N), np.uint8)
    mask[20:40, 20:40] = 1
    made.set_image_and_mask(block_image(), mask)
    yield made
    made.close_enhancer()


@pytest.fixture
def empty_canvas(qtbot, qt_theme_applied):
    made = mm._MaskCanvas()
    qtbot.addWidget(made)
    yield made
    made.close_enhancer()


def _run(target):
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# The console
# ---------------------------------------------------------------------------

def test_a_line_posted_from_a_worker_thread_reaches_the_scrollback(qtbot):
    console = mm._MasksConsole()
    qtbot.addWidget(console)
    _run(lambda: console.post("segmented 12 objects", "info"))
    qtbot.waitUntil(lambda: "segmented 12 objects" in console.text(),
                    timeout=5000)


def test_a_blank_line_is_not_written(qtbot):
    console = mm._MasksConsole()
    qtbot.addWidget(console)
    console.post("first")
    before = console.text()
    console.post("   \n", "warning")
    console.post(None)
    assert console.text() == before


def test_a_finished_bar_is_written_before_the_line_that_interrupts_it(qtbot):
    console = mm._MasksConsole()
    qtbot.addWidget(console)
    console.stream("Downloading 50%\r", source="Cellpose 3")
    console.stream("Downloading 100%\n", source="Cellpose 3")
    console.post("Model ready")
    text = console.text()
    assert "Cellpose 3: Downloading 100%" in text
    assert text.index("Downloading 100%") < text.index("Model ready")
    assert "50%" not in text, "the unfinished state is dropped, not written"


def test_a_worker_printing_after_the_console_is_gone_does_not_raise(qtbot):
    console = mm._MasksConsole()
    console.stream("warming up\r")
    console._take_stream()
    shiboken6.delete(console)
    console.stream("still going 10%\r")
    assert console._stream_pending == "still going 10%"


def test_a_bar_kicked_twice_by_a_race_is_drawn_once(qtbot):
    """Two queued kicks from a worker around a GUI-thread line arm the
    throttle once: the second finds it already running."""
    console = mm._MasksConsole()
    qtbot.addWidget(console)
    _run(lambda: console.stream("step 1 10%\r"))
    console.post("interrupting line")
    _run(lambda: console.stream("step 2 20%\r"))
    QApplication.processEvents()
    assert console._stream_timer.isActive()
    qtbot.waitUntil(lambda: console.stream_updates == 1, timeout=5000)
    assert console.progress_text() == "step 2 20%"
    assert console.stream_updates == 1


# ---------------------------------------------------------------------------
# The worker-side picture helpers
# ---------------------------------------------------------------------------

def _compare_request(event, normalized=True):
    return mm._CompareRequest(
        key=("k",), image=block_image(), box=(10, 10, 30, 30),
        chain=detect_chain.NO_CHAIN, normalized=normalized,
        percentiles=(1.0, 99.0), cancelled=event)


def test_a_compare_request_cancelled_before_it_starts_gives_nothing():
    event = threading.Event()
    event.set()
    assert mm._compare_picture_for(_compare_request(event)) is None


def test_a_compare_request_cancelled_while_stretching_gives_nothing(
        monkeypatch):
    event = threading.Event()
    real = engine.normalize_for_detection

    def stretch_then_cancel(*args, **kwargs):
        out = real(*args, **kwargs)
        event.set()
        return out

    monkeypatch.setattr(engine, "normalize_for_detection", stretch_then_cancel)
    assert mm._compare_picture_for(_compare_request(event)) is None


def test_a_compare_request_gives_the_box_of_the_field():
    event = threading.Event()
    out = mm._compare_picture_for(_compare_request(event, normalized=False))
    assert out.shape == (20, 20)
    np.testing.assert_array_equal(out, block_image()[10:30, 10:30])


def test_an_enhance_request_with_no_field_gives_nothing():
    request = mm._EnhanceRequest(key=(), image=None,
                                 chain=detect_chain.NO_CHAIN)
    assert mm._enhanced_picture_for(request) is None


def test_an_unchanged_or_float_field_is_its_own_picture():
    image = block_image()
    same = mm._enhanced_picture_for(mm._EnhanceRequest(
        key=(), image=image, chain=detect_chain.NO_CHAIN))
    assert same.prepared is image and same.picture is image

    floats = image.astype(np.float32)
    sharpened = mm._enhanced_picture_for(mm._EnhanceRequest(
        key=(), image=floats, chain=detect_chain.Chain(sharpen=True)))
    assert sharpened.prepared is sharpened.picture
    assert sharpened.picture.dtype.kind == "f"


# ---------------------------------------------------------------------------
# The canvas with nothing loaded
# ---------------------------------------------------------------------------

def test_a_canvas_with_no_field_answers_none(empty_canvas):
    empty_canvas.enhance_display = True
    assert empty_canvas.displayed_source() is None
    assert empty_canvas.detection_source() is None
    assert empty_canvas.enhanced_picture() is None
    assert empty_canvas.wand_source() is None
    assert empty_canvas._object_lookup() is None


def test_the_wand_reads_the_field_when_no_enhancement_is_on(canvas):
    canvas.enhance_display = True
    canvas.enhance_chain = detect_chain.NO_CHAIN
    assert canvas.wand_source() is canvas.image


# ---------------------------------------------------------------------------
# A failed enhancement
# ---------------------------------------------------------------------------

def _break_prepare(monkeypatch):
    real = detect_chain.prepare

    def failing(image, chain, **kwargs):
        if detect_chain.pre_active(chain):
            raise RuntimeError("deconvolution diverged")
        return real(image, chain, **kwargs)

    monkeypatch.setattr(detect_chain, "prepare", failing)


def test_a_failed_enhancement_is_said_once_and_not_asked_again(
        qtbot, canvas, monkeypatch):
    _break_prepare(monkeypatch)
    said = []
    canvas.status.connect(said.append)
    chain = detect_chain.Chain(sharpen=True)
    canvas.enhance_chain = chain
    base = canvas.detection_base()
    assert canvas.enhanced_picture() is base, "the field while it builds"
    qtbot.waitUntil(lambda: canvas._enhance_failure is not None,
                    timeout=10_000)
    assert said == ["Image enhancement failed: deconvolution diverged"]
    asked_before = canvas._enhance_asked
    assert canvas.enhanced_picture() is base
    assert canvas._enhance_asked is asked_before, "not asked again"


def test_the_detectors_are_told_why_a_failed_enhancement_left_them_nothing(
        qtbot, canvas, monkeypatch):
    _break_prepare(monkeypatch)
    canvas.enhance_chain = detect_chain.Chain(psf_operation="deconvolve")
    with pytest.raises(ValueError, match="updating"):
        canvas.detection_source()
    qtbot.waitUntil(lambda: canvas._enhance_failure is not None,
                    timeout=10_000)
    with pytest.raises(ValueError, match="deconvolution diverged"):
        canvas.detection_source()


def test_a_plain_picture_is_kept_without_repainting_when_not_shown(canvas):
    chain = detect_chain.Chain(sharpen=True)
    canvas.enhance_chain = chain
    canvas.enhance_display = False
    base = canvas.detection_base()
    picture = np.full_like(base, 7)
    before = canvas.pixmap().toImage()
    canvas._take_enhanced((base, chain, picture))
    assert canvas.enhanced_picture() is picture
    assert canvas.pixmap().toImage() == before


def test_a_finished_run_that_produced_nothing_is_not_delivered(canvas):
    got = []
    canvas.enhanced_ready.connect(got.append)
    request = mm._EnhanceRequest(key=(), image=canvas.image,
                                 chain=detect_chain.NO_CHAIN,
                                 cancelled=threading.Event())
    canvas._enhanced_done(request, None, None)
    QApplication.processEvents()
    assert got == []


def test_a_picture_finished_after_the_canvas_is_gone_is_dropped(qtbot):
    made = mm._MaskCanvas()
    request = mm._EnhanceRequest(key=(), image=block_image(),
                                 chain=detect_chain.NO_CHAIN,
                                 cancelled=threading.Event())
    shiboken6.delete(made)
    made._enhanced_done(request, np.zeros((2, 2)), None)
    assert not shiboken6.isValid(made)


def test_a_settle_timer_with_nothing_waiting_sends_nothing(canvas):
    said = []
    canvas.status.connect(said.append)
    canvas.close_enhancer()
    canvas._submit_pending_enhance()
    assert said == []
    assert canvas._enhance_worker is None


# ---------------------------------------------------------------------------
# The readout
# ---------------------------------------------------------------------------

def _inside_block(canvas):
    """A canvas point over image pixel (30, 30)."""
    return QPointF(100 + 30.5 * 400 / IMG_N, 30.5 * 400 / IMG_N)


def test_the_readout_reports_the_pixel_when_objects_cannot_be_measured(
        canvas, monkeypatch):
    def broken(*args, **kwargs):
        raise MemoryError("no room")

    monkeypatch.setattr(engine, "ObjectLookup", broken)
    canvas._lookup = None
    canvas.update_readout(_inside_block(canvas))
    text = canvas.readout_text()
    assert "intensity 30000" in text
    assert "\n" not in text, "no object line without the object table"


def test_the_readout_is_not_read_again_after_the_mouse_left(qtbot, canvas):
    canvas.update_readout(_inside_block(canvas))
    assert canvas.readout_text()
    canvas.refresh()
    assert canvas._readout_queued
    canvas.update_readout(None)
    qtbot.waitUntil(lambda: not canvas._readout_queued, timeout=2000)
    assert canvas.readout_text() == ""


def test_a_missing_number_reads_as_blank():
    assert mm._readout_number(None) == ""
    assert mm._readout_number(3.0) == "3"


# ---------------------------------------------------------------------------
# Ctrl+click and the ruler
# ---------------------------------------------------------------------------

def test_a_ctrl_edit_off_the_mask_changes_nothing(canvas):
    before = canvas.mask.copy()
    assert canvas._ctrl_edit_at(None, split=True) is False
    assert canvas._ctrl_edit_at((IMG_N + 5, 3), split=False) is False
    assert canvas._ctrl_edit_at((3, -1), split=True) is False
    np.testing.assert_array_equal(canvas.mask, before)


def _accent_pixels(image: QImage) -> int:
    im = image.convertToFormat(QImage.Format_RGB32)
    arr = np.frombuffer(im.constBits(), dtype=np.uint32).reshape(
        im.height(), im.bytesPerLine() // 4)
    accent = np.uint32(QColor(active_palette()["accent"]).rgb())
    return int((arr == accent).sum())


def test_a_measured_line_is_drawn_on_the_canvas(canvas):
    before = _accent_pixels(canvas.grab().toImage())
    canvas.ruler.set_active(True)
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, 120, 50))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, 400, 300,
                                 button=Qt.NoButton))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, 400, 300,
                                    buttons=Qt.NoButton))
    assert canvas.ruler.start is not None and canvas.ruler.end is not None
    after = _accent_pixels(canvas.grab().toImage())
    assert after > before + 100
