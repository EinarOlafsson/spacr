"""The mask editor answers with a no-op instead of a traceback.

Every branch here is reached with no field open, with a stale field, or with
a worker that has already gone. The screen is a long-lived window a user
leaves open while folders come and go, so a guard that fails is a crash in
the middle of curating, and a guard that silently does the wrong thing paints
one field's correction onto another field's mask.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (MakeMasksScreen, _MaskCanvas,
                                         _MaskLoadWorker)

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64


def _block() -> np.ndarray:
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[20:40, 20:40] = 30000
    return img


@pytest.fixture
def folder_3(tmp_path: Path) -> Path:
    folder = tmp_path / "three"
    folder.mkdir()
    rng = np.random.default_rng(7)
    for i in range(3):
        imageio.imwrite(folder / f"img_{i:02d}.tif",
                        rng.integers(0, 65535, (IMG_N, IMG_N), dtype=np.uint16))
    return folder


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(CANVAS_W, CANVAS_H)
    widget.set_image_and_mask(_block(), np.zeros((IMG_N, IMG_N), np.uint8))
    return widget


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder_3: Path):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder_3))
    widget._canvas.resize(CANVAS_W, CANVAS_H)
    widget._canvas.refresh()
    return widget


def _wheel(canvas, x, y, notches):
    point = QPointF(float(x), float(y))
    return QWheelEvent(point, canvas.mapToGlobal(point.toPoint()),
                       QPoint(0, 0), QPoint(0, int(notches)),
                       Qt.NoButton, Qt.NoModifier,
                       Qt.NoScrollPhase, False)


# --------------------------------------------------------------------------- #
#  The loader thread
# --------------------------------------------------------------------------- #

def test_the_loader_keeps_the_pair_it_decoded(tmp_path):
    """A successful load leaves the arrays on the worker and no error.

    The worker runs on its own thread and cannot raise into the GUI, so the
    only way the screen learns what happened is what the worker kept.
    """
    imageio.imwrite(tmp_path / "img_00.tif", _block())
    worker = _MaskLoadWorker(str(tmp_path), "img_00.tif", token=3)

    worker.run()

    assert worker.error is None
    image, mask = worker.result
    assert image.shape == (IMG_N, IMG_N)
    assert mask.shape == (IMG_N, IMG_N)


def test_the_loader_keeps_the_original_exception(tmp_path):
    """An unreadable file leaves the exception itself, not a message.

    The screen re-raises or reports it, and a stringified error loses the
    type -- "file not found" and "not a TIFF" call for different advice.
    """
    (tmp_path / "img_00.tif").write_bytes(b"this is definitely not a TIFF")
    worker = _MaskLoadWorker(str(tmp_path), "img_00.tif", token=3)

    worker.run()

    assert worker.result is None
    assert isinstance(worker.error, Exception)


# --------------------------------------------------------------------------- #
#  Panning and zooming with nothing loaded, or nowhere to go
# --------------------------------------------------------------------------- #

def test_panning_an_empty_canvas_moves_nothing(qtbot, qt_theme_applied):
    """With no mask there is no viewport, so a drag converts to no movement.

    The screen exists before a folder is opened. A division by the pixmap
    width would raise on the very first mouse drag over an empty editor.
    """
    empty = _MaskCanvas()
    qtbot.addWidget(empty)
    empty.resize(CANVAS_W, CANVAS_H)

    assert empty._image_delta(40, 40) == (0, 0)
    assert empty.pan_by(5, 5) is False
    assert empty.zoom_at(10, 10, 2.0) is None


def test_zooming_by_a_nonsense_factor_changes_nothing(canvas):
    """A zero or negative factor leaves the viewport where it was.

    The factor comes from a wheel speed setting the user can type into.
    Dividing the viewport by zero would make it infinite, and by a negative
    number would turn it inside out.
    """
    canvas.zoom_at(32, 32, 4.0)
    before = canvas._viewport_bounds()

    canvas.zoom_at(32, 32, 0.0)
    canvas.zoom_at(32, 32, -2.0)

    assert canvas._viewport_bounds() == before


def test_panning_against_the_edge_reports_that_nothing_moved(canvas):
    """A pan that is already clamped returns False rather than repainting.

    The return value is what tells the drag handler whether to consume the
    gesture; always answering True repaints the canvas on every mouse move
    at the edge of the image.
    """
    canvas.zoom_at(32, 32, 4.0)
    assert canvas.pan_by(-1000, -1000) is True     # travels to the corner

    assert canvas.pan_by(-1000, -1000) is False    # already there
    assert canvas.pan_by(0, 0) is False


def test_the_wheel_is_ignored_until_there_is_something_to_zoom(
        qtbot, qt_theme_applied, canvas):
    """No mask, or no notches, hands the wheel back to Qt.

    Scrolling over an empty editor has to fall through to the enclosing
    scroll area; swallowing it would freeze the page under the pointer.
    """
    empty = _MaskCanvas()
    qtbot.addWidget(empty)
    empty.resize(CANVAS_W, CANVAS_H)

    event = _wheel(empty, 300, 200, 120)
    empty.wheelEvent(event)
    assert not event.isAccepted()

    flat = _wheel(canvas, 300, 200, 0)
    canvas.wheelEvent(flat)
    assert not flat.isAccepted()


def test_a_wheel_off_the_image_zooms_about_the_middle(canvas):
    """A cursor in the margin still zooms, centred on the view.

    The canvas is wider than the picture, so there is a strip either side
    where the pointer is over the widget and over no pixel. Doing nothing
    there makes the wheel feel broken; guessing a pixel would jump the view
    somewhere the user was not pointing.
    """
    assert canvas._canvas_to_image(5.0, 200.0) is None
    before = canvas._viewport_bounds()

    event = _wheel(canvas, 5, 200, 120)
    canvas.wheelEvent(event)

    after = canvas._viewport_bounds()
    assert event.isAccepted()
    assert after != before
    # Centred on the view, not on the margin.
    assert (after[0] + after[2]) // 2 == (before[0] + before[2]) // 2
    assert (after[1] + after[3]) // 2 == (before[1] + before[3]) // 2


def test_a_sweep_over_nothing_deletes_nothing(qtbot, qt_theme_applied, canvas):
    """A sweep point that is None, or an empty canvas, removes no object.

    The sweep opens its undo step on the first object actually hit, so a
    gesture that touches nothing has to leave no history entry to step back
    through.
    """
    empty = _MaskCanvas()
    qtbot.addWidget(empty)

    assert empty._sweep_delete_at((10, 10)) is False
    assert canvas._sweep_delete_at(None) is False


# --------------------------------------------------------------------------- #
#  The folded module buttons
# --------------------------------------------------------------------------- #

def test_a_fold_button_that_is_not_there_is_not_restated(screen):
    """Restating a missing button does nothing instead of raising.

    The strip is built from a registry that may not carry every key, so
    ``button_for`` can answer None -- and this runs in a loop over every key
    while the masthead is being built.
    """
    assert screen._restate_fold_button(None, "curate") is None


def test_seeding_a_module_that_has_no_screen_seeds_nothing(screen,
                                                           monkeypatch):
    """With no folded screen there is nothing to point at the folder.

    An empty dict is what tells the caller the module will open on its own
    file picker, exactly as its tile did.
    """
    monkeypatch.setattr(type(screen), "folded_screen",
                        lambda self, key: None)

    assert screen.seed_folded("curate") == {}


def test_opening_the_model_zoo_points_it_at_the_open_folder(screen,
                                                            monkeypatch):
    """The zoo is handed the fields this screen already has open.

    The whole reason these are buttons on the masthead is that the folder is
    chosen here; a zoo that opened empty would fold a file dialog in with it.
    The load itself runs off the GUI thread, so what is pinned is the folder
    the zoo was asked for.
    """
    zoo = screen.folded_screen("model_zoo")
    asked = []
    monkeypatch.setattr(type(zoo), "set_fields_source",
                        lambda self, folder: asked.append(folder) or True)

    seeded = screen.seed_folded("model_zoo")

    assert seeded == {"folder": screen._folder}
    assert asked == [screen._folder]


def test_the_training_half_of_cellpose_keeps_its_own_path(screen):
    """Training is opened on its own tab with nothing seeded.

    Training reads a labelled set that is usually not the folder being
    curated, and it is the one path that must not be guessed at.
    """
    assert screen.seed_folded("train_cellpose") == {}
    assert screen.seed_folded(mm.MASK_FOLDER_KEY) == {"src": screen._folder}


def test_a_compare_request_with_no_window_to_show_it_does_nothing(
        screen, monkeypatch):
    """When the folded dialog cannot open, the request is dropped quietly.

    The request arrives from the zoo's own signal, so there is nobody to
    raise at; configuring a None would take the zoo down with it.
    """
    monkeypatch.setattr(type(screen), "open_folded",
                        lambda self, key: None)

    assert screen._on_zoo_compare_requested({"model_a": "a"}) is None


def test_masking_the_whole_folder_needs_a_window_to_run_in(screen,
                                                           monkeypatch):
    """A folded dialog that will not open reports that no run started.

    The return value is what the masthead reads to decide whether to say
    "masking N images"; claiming a run that never started leaves the status
    line lying for the rest of the session.
    """
    monkeypatch.setattr(type(screen), "_confirm",
                        lambda self, title, body: True)
    monkeypatch.setattr(type(screen), "open_folded",
                        lambda self, key: None)

    assert screen.mask_whole_folder() is False


def test_starting_a_folded_run_presses_that_pages_run(screen):
    """The seam to a GPU job is one named call, and it is the page's Run.

    Naming it is what lets a test drive everything up to a segmentation
    without starting Cellpose.
    """
    pressed = []

    class _Page:
        def _on_run(self):
            pressed.append(True)

    MakeMasksScreen._start_folded_run(_Page())

    assert pressed == [True]


# --------------------------------------------------------------------------- #
#  Counting what an edit changed
# --------------------------------------------------------------------------- #

def test_a_mask_with_nothing_to_compare_against_counts_all_of_itself():
    """With no earlier mask, everything labelled counts as changed.

    The count decides whether an edit reaches the curation ledger. A silent
    zero for the first mask of a field would leave the whole of it
    unrecorded.
    """
    after = np.zeros((8, 8), np.uint8)
    after[2:5, 2:5] = 1

    assert MakeMasksScreen._diff(None, after) == 9
    assert MakeMasksScreen._diff(np.zeros((4, 4), np.uint8), after) == 9
    assert MakeMasksScreen._diff(after, after) == 0
    assert MakeMasksScreen._diff(after, None) == 0


# --------------------------------------------------------------------------- #
#  Operations with no field open
# --------------------------------------------------------------------------- #

def test_filtering_and_detecting_with_no_field_open_do_nothing(
        qtbot, qt_theme_applied):
    """The filter and the Otsu detector are no-ops before a folder is opened.

    Both are buttons on a screen that exists from launch. Reaching into a
    mask that is None would crash the editor on a stray click.
    """
    empty = MakeMasksScreen()
    qtbot.addWidget(empty)

    assert empty.apply_object_filter(on_load=False) == 0
    assert empty._on_detect_otsu() is None


def test_a_combine_that_fails_is_reported_and_the_mask_is_untouched(
        screen, monkeypatch):
    """An error folding the detection in leaves the mask exactly as it was.

    Half-applying a detection is the worst outcome: the mask on screen would
    no longer be the mask on disk and no longer be the detection either, and
    the undo history would not know.
    """
    warned = []
    monkeypatch.setattr(type(screen), "_warn",
                        lambda self, title, body: warned.append((title, body)))
    monkeypatch.setattr(engine, "otsu_instances",
                        lambda *a, **k: np.ones((IMG_N, IMG_N), np.uint16))

    def _explode(before, detected, mode):
        raise ValueError("the two masks disagree about shape")

    monkeypatch.setattr(engine, "combine_masks", _explode)
    before = screen._canvas.mask.copy()

    screen._on_detect_otsu()

    assert warned and "Otsu detect failed" in warned[0][0]
    assert "disagree about shape" in warned[0][1]
    assert np.array_equal(screen._canvas.mask, before)


# --------------------------------------------------------------------------- #
#  Loading in the background
# --------------------------------------------------------------------------- #

def test_a_big_field_is_loaded_off_the_gui_thread(tmp_path):
    """A file past the size threshold is decoded on a worker.

    Decoding a 4k 16-bit field on the GUI thread freezes the editor for
    seconds, which reads as a hang rather than as a load.
    """
    big = tmp_path / "big.tif"
    big.write_bytes(b"\0" * (9 * 1024 * 1024))

    assert MakeMasksScreen._should_background_load(str(big)) is True


def test_a_second_field_asked_for_while_one_is_loading_waits_its_turn(
        screen, monkeypatch):
    """A request arriving mid-load is remembered, and the user is told.

    Clicking through a folder faster than it decodes must not start a worker
    per click: they would land in an unpredictable order and the last one to
    finish would win, whatever the user last selected.
    """
    monkeypatch.setattr(MakeMasksScreen, "_should_background_load",
                        staticmethod(lambda path: True))
    started = []
    monkeypatch.setattr(type(screen), "_start_background_load",
                        lambda self, *request: started.append(request))

    screen._current_index = 0
    screen._load_current()
    assert len(started) == 1

    screen._load_worker = object()          # one already in flight
    screen._current_index = 1
    screen._load_current()

    assert len(started) == 1                # no second worker
    assert screen._pending_load is not None
    assert "Waiting to load" in screen._status_label.text()
    screen._load_worker = None
    screen._pending_load = None


def test_a_finished_load_with_no_worker_left_is_ignored(screen):
    """The finished handler answers nothing when the worker has gone.

    ``finished`` can arrive after the screen has already dropped the worker
    -- a folder change, or a close -- and reading its token then is a use of
    a destroyed QThread.
    """
    screen._load_worker = None

    assert screen._on_background_load_finished() is None


def test_a_background_load_that_failed_is_reported_and_the_next_one_starts(
        screen, monkeypatch):
    """A worker error goes through the failure path, then the queue drains.

    Dropping the pending request on a failure leaves the editor showing the
    previous field under the newly selected filename.
    """
    reported = []
    monkeypatch.setattr(type(screen), "_handle_load_failure",
                        lambda self, error: reported.append(error))
    started = []
    monkeypatch.setattr(type(screen), "_start_background_load",
                        lambda self, *request: started.append(request))

    class _Failed:
        token = screen._load_token
        filename = "img_01.tif"
        result = None
        error = OSError("the share went away mid-read")

        def deleteLater(self):
            pass

    screen._load_worker = _Failed()
    screen._pending_load = (screen._folder, "img_02.tif",
                            screen._load_token)

    screen._on_background_load_finished()

    assert reported and isinstance(reported[0], OSError)
    assert started == [(screen._folder, "img_02.tif", screen._load_token)]
    assert screen._pending_load is None


def test_a_pair_that_arrives_for_the_previous_field_is_dropped(screen):
    """A decoded pair whose token is stale never reaches the canvas.

    This is the whole point of the token: a slow load for field 1 finishing
    after the user has moved to field 2 would otherwise paint field 1's
    image under field 2's name, and every correction made after that would
    be saved to the wrong file.
    """
    on_screen = screen._canvas.mask.copy()
    other = np.ones((IMG_N, IMG_N), np.uint16)

    screen._apply_loaded_pair("img_01.tif", screen._load_token - 1,
                              other, other)

    assert np.array_equal(screen._canvas.mask, on_screen)


def test_closing_the_screen_stops_the_loader_before_qt_destroys_it(
        qtbot, qt_theme_applied, folder_3, monkeypatch):
    """A running loader is interrupted and drained on close.

    Qt aborts the process when a running QThread is destroyed, so a load
    still in flight when the window closes is a crash rather than a leak.
    """
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder_3))

    interrupted = []
    drained = []

    class _Running:
        def requestInterruption(self):
            interrupted.append(True)

    monkeypatch.setattr("spacr.qt.bridge.drain_thread",
                        lambda worker, timeout_ms=0: drained.append(worker))
    worker = _Running()
    widget._load_worker = worker
    widget._pending_load = (str(folder_3), "img_02.tif", 5)

    widget.close()

    assert interrupted == [True]
    assert drained == [worker]
    assert widget._load_worker is None
    assert widget._pending_load is None
    assert widget._loading is False


def test_a_loader_that_cannot_be_interrupted_is_still_drained(
        qtbot, qt_theme_applied, folder_3, monkeypatch):
    """A worker whose C++ half has gone is drained rather than raised over.

    ``requestInterruption`` on a destroyed QThread raises, and this runs from
    ``closeEvent`` where an exception loses the drain that follows it.
    """
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder_3))

    drained = []

    class _Gone:
        def requestInterruption(self):
            raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr("spacr.qt.bridge.drain_thread",
                        lambda worker, timeout_ms=0: drained.append(worker))
    widget._load_worker = _Gone()

    widget.close()

    assert len(drained) == 1
