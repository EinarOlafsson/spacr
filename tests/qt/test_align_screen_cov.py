"""Align & Stitch screen — the paths the first suite left untested.

``tests/qt/test_align_screen.py`` covers planning, writing and the
threaded round trip. This file goes after what is left:

* the **painted output**. The layout view's whole job is to make a tile
  that did not register countable at a glance, so the pixels it paints
  are read back and asserted — orange where a tile fell back to the
  stage position, accent where it registered, and the empty-state hint
  when there is nothing to draw;
* a plan whose **canvas has no area**, which must draw nothing rather
  than divide by zero;
* clicking the **void** between tiles, and asking for a tile index the
  plan does not have;
* the folder **pickers**, including the cancel that must leave the field
  alone;
* a plan with **unreadable tiles**, which the status line has to mention;
* the threaded ``_run_job`` plumbing — the worker payload, and a
  completion handler that raises, which must settle the job as failed
  instead of leaving the screen busy forever.

Offscreen, offline, no modal dialogs, no sleeps.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from spacr import align as align_mod
from spacr.qt.screens.align import AlignScreen, TileLayoutWidget
from spacr.qt.theme import DARK_PALETTE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _texture(height, width, seed=0, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(rng.random((height, width)).astype(np.float32),
                             sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


@pytest.fixture(scope="module")
def tile_folder(tmp_path_factory):
    """A real 2x2 grid of 128x128 tiles stepped 100 px."""
    folder = tmp_path_factory.mktemp("tiles")
    big = _texture(260, 260, seed=31)
    for k in range(4):
        row, col = divmod(k, 2)
        np.save(folder / f"plate1_B07_{k + 1:03d}.npy",
                big[row * 100:row * 100 + 128, col * 100:col * 100 + 128])
    return str(folder)


@pytest.fixture
def flat_folder(tmp_path):
    """Two tiles whose shared overlap is blank — neither can register."""
    folder = tmp_path / "flat"
    folder.mkdir()
    for k in range(2):
        np.save(folder / f"plate1_B07_{k + 1:03d}.npy",
                np.full((64, 100), 2000, dtype=np.uint16))
    return str(folder)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = AlignScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(900, 600)
    return widget


def _configure(screen, src, dst=None, overlap=1 - 100 / 128, grid=(2, 2)):
    screen._src_edit.setText(str(src))
    if grid:
        screen._rows_box.setValue(grid[0])
        screen._cols_box.setValue(grid[1])
    screen._overlap_box.setValue(overlap)
    if dst is not None:
        screen._dst_edit.setText(str(dst))
    return screen


def _painted(view):
    """Render the widget and return its QImage — forces ``paintEvent``."""
    pixmap = QPixmap(view.size())
    view.render(pixmap)
    return pixmap.toImage()


def _colour_at(image, point):
    return image.pixelColor(int(point.x()), int(point.y()))


@contextlib.contextmanager
def collect_job_finished(screen):
    """Collect every ``job_finished`` payload emitted inside the block."""
    record = {"ok": []}
    screen.job_finished.connect(lambda ok: record["ok"].append(bool(ok)))
    try:
        yield record
    finally:
        screen.job_finished.disconnect()


# ---------------------------------------------------------------------------
# What the layout view actually paints
# ---------------------------------------------------------------------------

def test_the_layout_paints_a_registered_tile_in_the_accent_colour(
        screen, tile_folder):
    """The drawn pixels, not just the rectangles, are asserted."""
    _configure(screen, tile_folder)
    assert screen.build_plan() is True
    view = screen._layout_view
    view.resize(420, 420)

    image = _painted(view)
    assert image.width() == 420 and image.height() == 420

    background = _colour_at(image, QPoint(1, 1))
    assert background == QColor(DARK_PALETTE["surface"]), \
        "the void around the layout is the surface colour"

    rects = dict(view.tile_rects())
    assert len(rects) == 4
    plan = screen.plan()
    by_index = {p.tile.index: p for p in plan.placements}
    for index, rect in rects.items():
        assert by_index[index].method == align_mod.METHOD_REGISTRATION
        # A point inside the rect but away from its border and its label.
        probe = QPoint(int(rect.left() + rect.width() * 0.2),
                       int(rect.top() + rect.height() * 0.2))
        painted = _colour_at(image, probe)
        assert painted != background, f"tile {index} was not drawn"
        assert painted == QColor(DARK_PALETTE["accent"]), \
            f"tile {index} registered at {by_index[index].confidence:.2f} " \
            f"but was not painted at full accent"


def test_a_nominal_tile_is_painted_orange_and_hatched(screen, flat_folder):
    """The failure this screen exists to expose, read back off the canvas."""
    _configure(screen, flat_folder, overlap=0.3, grid=(1, 2))
    assert screen.build_plan() is True
    assert screen.plan().n_nominal == 2

    view = screen._layout_view
    view.resize(420, 420)
    image = _painted(view)

    warning = QColor(DARK_PALETTE["warning"]).name()
    surface = QColor(DARK_PALETTE["surface"]).name()
    rects = dict(view.tile_rects())
    assert len(rects) == 2
    for index, rect in rects.items():
        # Sample the interior, well clear of the 1 px border.
        seen = {}
        for dy in range(6, int(rect.height()) - 6, 3):
            for dx in range(6, int(rect.width()) - 6, 3):
                name = _colour_at(image, QPoint(int(rect.left() + dx),
                                                int(rect.top() + dy))).name()
                seen[name] = seen.get(name, 0) + 1
        assert seen.get(warning, 0) > 0, f"tile {index} is not orange"
        assert seen.get(surface, 0) > 0, (
            f"tile {index} shows no hatch — a hatch painted in the same "
            f"colour as the fill is not a hatch. Saw only {sorted(seen)}")
        # The hatch is a texture, not a wash: the fill must still dominate.
        assert seen[warning] > seen[surface]


def test_the_empty_state_hint_is_painted_when_there_is_no_plan(
        qtbot, qt_theme_applied):
    view = TileLayoutWidget()
    qtbot.addWidget(view)
    view.resize(320, 240)
    assert view.tile_rects() == []

    image = _painted(view)
    surface = QColor(DARK_PALETTE["surface"])
    assert _colour_at(image, QPoint(2, 2)) == surface
    # The hint text is drawn centred, so *something* in the middle band is
    # not the background colour.
    middle = [image.pixelColor(x, 120) for x in range(0, 320, 2)]
    assert any(colour != surface for colour in middle), \
        "the 'press Plan' hint was not drawn"


def test_a_canvas_with_no_area_draws_nothing_instead_of_dividing_by_zero(
        qtbot, qt_theme_applied):
    """A plan can carry placements and still have an empty canvas."""
    tile = align_mod.Tile(path="ghost.npy", index=0, plate="p", well="B07",
                          field=1, shape=(0, 0, 1))
    plan = align_mod.AlignPlan(
        tiles=[tile],
        placements=[align_mod.Placement(tile=tile, y=0.0, x=0.0)],
        canvas_shape=(0, 0, 1))

    view = TileLayoutWidget()
    qtbot.addWidget(view)
    view.resize(200, 150)
    view.set_plan(plan)
    assert view.plan() is plan
    assert view.tile_rects() == []

    image = _painted(view)
    assert _colour_at(image, QPoint(1, 1)) == QColor(DARK_PALETTE["surface"])


def test_tiny_tiles_are_drawn_without_a_label(qtbot, qt_theme_applied):
    """Below 26x14 px the field number is skipped rather than clipped."""
    tiles = [align_mod.Tile(path=f"t{k}.npy", index=k, plate="p", well="B07",
                            field=k + 1, shape=(10, 10, 1))
             for k in range(400)]
    placements = [
        align_mod.Placement(tile=t, y=float(10 * (k // 20)),
                            x=float(10 * (k % 20)),
                            confidence=1.0,
                            method=align_mod.METHOD_REGISTRATION)
        for k, t in enumerate(tiles)]
    plan = align_mod.AlignPlan(tiles=tiles, placements=placements,
                               canvas_shape=(200, 200, 1))

    view = TileLayoutWidget()
    qtbot.addWidget(view)
    view.resize(120, 120)
    view.set_plan(plan)
    rects = view.tile_rects()
    assert len(rects) == 400
    assert all(r.width() <= 26 for _i, r in rects)
    # The grid really tiles the widget: 20 distinct lefts, 20 distinct tops.
    assert len({round(r.left(), 3) for _i, r in rects}) == 20
    assert len({round(r.top(), 3) for _i, r in rects}) == 20

    image = _painted(view)                      # the no-label branch
    assert image.width() == 120
    names = {image.pixelColor(x, y).name()
             for y in range(10, 110, 2) for x in range(10, 110, 2)}
    assert QColor(DARK_PALETTE["accent"]).name() in names, \
        "the tiles were not painted at all"
    # 400 outlined rectangles in a 120 px box: the antialiased borders
    # leave many blend colours, not a flat wash of accent.
    assert len(names) > 5, f"only {sorted(names)} was painted"


def test_clicking_between_the_tiles_reports_the_void(screen, tile_folder,
                                                     qtbot):
    _configure(screen, tile_folder)
    screen.build_plan()
    view = screen._layout_view
    view.resize(400, 400)

    rects = dict(view.tile_rects())
    corner = QPoint(1, 1)
    assert not any(r.contains(corner) for r in rects.values())

    with qtbot.waitSignal(view.tile_clicked, timeout=5000) as blocker:
        QTest.mouseClick(view, Qt.LeftButton, pos=corner)
    assert blocker.args[0] == -1
    assert screen.tile_info_text() == ""


def test_asking_for_a_tile_the_plan_does_not_have_clears_the_readout(
        screen, tile_folder):
    _configure(screen, tile_folder)
    screen.build_plan()
    screen._on_tile_clicked(1)
    assert "y=" in screen.tile_info_text()
    screen._on_tile_clicked(99)
    assert screen.tile_info_text() == "", \
        "an unknown index must clear the readout, not keep a stale one"


def test_the_tile_readout_names_the_fallback_reason(screen, flat_folder):
    _configure(screen, flat_folder, overlap=0.3, grid=(1, 2))
    screen.build_plan()
    screen._on_tile_clicked(0)
    text = screen.tile_info_text()
    assert "plate1_B07_001.npy" in text
    assert align_mod.METHOD_NOMINAL in text
    assert "0 pair(s)" in text
    assert "blank" in text, "the reason the pair was refused must be shown"


# ---------------------------------------------------------------------------
# Pickers
# ---------------------------------------------------------------------------

def test_the_source_picker_fills_the_field_and_cancel_leaves_it(
        screen, monkeypatch, tmp_path):
    chosen = str(tmp_path / "picked_tiles")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: chosen))
    screen._pick_source()
    assert screen._src_edit.text() == chosen

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._pick_source()
    assert screen._src_edit.text() == chosen, "cancel must not clear the field"


def test_the_destination_picker_starts_from_the_source_folder(
        screen, monkeypatch, tmp_path):
    seen = {}

    def _remember(_parent, _caption, start):
        seen["start"] = start
        return str(tmp_path / "picked_out")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(_remember))
    screen._src_edit.setText(str(tmp_path / "tiles"))
    screen._pick_destination()
    assert seen["start"] == str(tmp_path / "tiles"), \
        "with no destination yet, the picker opens next to the tiles"
    assert screen._dst_edit.text() == str(tmp_path / "picked_out")

    screen._pick_destination()
    assert seen["start"] == str(tmp_path / "picked_out")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._pick_destination()
    assert screen._dst_edit.text() == str(tmp_path / "picked_out")


def test_the_pickers_fall_back_to_home_when_nothing_is_typed(
        screen, monkeypatch):
    seen = []
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        staticmethod(lambda _p, _c, start: (seen.append(start), "")[1]))
    screen._src_edit.setText("")
    screen._dst_edit.setText("")
    screen._pick_source()
    screen._pick_destination()
    assert seen == [os.path.expanduser("~"), os.path.expanduser("~")]


# ---------------------------------------------------------------------------
# Unreadable tiles reach the status line
# ---------------------------------------------------------------------------

def test_tiles_that_could_not_be_read_are_named_in_the_status(screen,
                                                              tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    big = _texture(128, 228, seed=5)
    np.save(folder / "plate1_B07_001.npy", big[:, 0:128])
    np.save(folder / "plate1_B07_002.npy", big[:, 100:228])
    (folder / "plate1_B07_003.npy").write_bytes(b"this is not an npy")

    _configure(screen, str(folder), overlap=1 - 100 / 128, grid=(1, 3))
    assert screen.build_plan() is True

    plan = screen.plan()
    assert len(plan.unplaced) == 1
    assert os.path.basename(plan.unplaced[0][0].path) == "plate1_B07_003.npy"
    assert plan.n_registered == 2

    status = screen.status_text()
    assert "2 of 2 tile(s) registered" in status
    assert "1 tile(s) could not be read" in status
    assert screen.last_error, "an unreadable tile is an error-coloured status"
    assert "unplaced" in screen.report_text()


# ---------------------------------------------------------------------------
# The threaded job plumbing
# ---------------------------------------------------------------------------

class _InlineWorker(QObject):
    """Stands in for ``PipelineWorker``, running on the calling thread."""

    error = Signal(str)
    finished = Signal(bool)

    def __init__(self, fn, settings):
        super().__init__()
        self._fn = fn
        self._settings = settings

    def run(self):
        try:
            self._fn(self._settings)
        except Exception as exc:                # noqa: BLE001 - mirrors bridge
            self.error.emit(f"Traceback\n{type(exc).__name__}: {exc}")
            self.finished.emit(False)
            return
        self.finished.emit(True)


class _InlineThread(QObject):
    """Stands in for ``QThread`` — ``start()`` runs the worker inline."""

    finished = Signal()

    def __init__(self, worker):
        super().__init__()
        self._worker = worker

    def start(self):
        self._worker.run()
        self.finished.emit()


@pytest.fixture
def inline_threaded_screen(qtbot, qt_theme_applied, monkeypatch):
    """A ``threaded=True`` screen whose thread is real code run inline.

    The QThread itself is the only thing replaced: ``_run_job`` builds the
    same payload dict, connects the same signals and relays through the
    same ``_job_settled`` bound method, so everything asserted is the
    screen's own plumbing.
    """
    from spacr.qt.screens import align as screen_mod

    def _fake_make_thread(fn, settings):
        worker = _InlineWorker(fn, settings)
        return _InlineThread(worker), worker

    monkeypatch.setattr(screen_mod, "make_thread", _fake_make_thread)
    widget = AlignScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.resize(900, 600)
    return widget


def test_the_worker_payload_carries_the_result_back_to_the_handler(
        inline_threaded_screen, tile_folder):
    screen = inline_threaded_screen
    _configure(screen, tile_folder)

    with collect_job_finished(screen) as record:
        assert screen.build_plan() is True

    assert record["ok"] == [True]
    plan = screen.plan()
    assert plan is not None and plan.n_registered == 4
    assert "4 of 4 tile(s) registered" in screen.status_text()
    assert not screen.is_busy()
    assert screen._btn_plan.isEnabled()
    assert screen._pending == [], "the job must be retired from the queue"
    assert screen.active_jobs() == 0


def test_a_handler_that_raises_settles_the_job_as_failed(
        inline_threaded_screen):
    """A completion handler blowing up must not leave the screen busy.

    ``_on_plan_ready`` handed ``None`` raises; the screen has to report it
    inline, mark the job failed and re-enable its controls.
    """
    screen = inline_threaded_screen
    screen._src_edit.setText("anything")

    with collect_job_finished(screen) as record:
        assert screen._run_job(lambda: None, screen._on_plan_ready) is True

    assert record["ok"] == [False]
    assert screen.last_error, "the failure must be reported inline"
    assert screen.status_text() == screen.last_error
    assert not screen.is_busy()
    assert screen._btn_plan.isEnabled()
    assert screen.plan() is None


def test_a_worker_error_is_reduced_to_its_last_line(inline_threaded_screen,
                                                    tmp_path):
    screen = inline_threaded_screen
    _configure(screen, str(tmp_path / "not_here"), grid=None)

    with collect_job_finished(screen) as record:
        assert screen.build_plan() is True

    assert record["ok"] == [False]
    assert screen.status_text().startswith("Align failed: ")
    assert "\n" not in screen.status_text(), "only the last line is shown"
    assert "does not exist" in screen.status_text()
    assert screen.plan() is None
    assert not screen.is_busy()


def test_retiring_an_older_job_does_not_drop_the_current_one(screen):
    """Two jobs in flight: the first to finish must release only its own
    references. Clearing the *live* job's thread and worker is precisely
    the double-owner bug ``bridge.make_thread`` documents."""
    old_thread, old_worker = object(), object()
    live_thread, live_worker = object(), object()
    screen._jobs = [(old_thread, old_worker), (live_thread, live_worker)]
    screen._thread, screen._worker = live_thread, live_worker

    screen._retire_job(old_thread)
    assert screen._jobs == [(live_thread, live_worker)]
    assert screen._thread is live_thread, 'the running job lost its thread ref'
    assert screen._worker is live_worker

    screen._retire_job(live_thread)
    assert screen._jobs == []
    assert screen._thread is None and screen._worker is None
    assert screen.active_jobs() == 0


def test_an_empty_worker_error_still_says_something(inline_threaded_screen):
    screen = inline_threaded_screen
    screen._on_worker_error_text("")
    assert screen.status_text() == "Align failed: unknown error"
    assert screen.last_error


# ---------------------------------------------------------------------------
# Settings edge cases
# ---------------------------------------------------------------------------

def test_apply_settings_survives_a_sparse_dict(screen):
    """A settings dict with nothing in it must not crash the panel."""
    screen.apply_settings({})
    assert screen._src_edit.text() == ""
    assert screen.settings()["grid"] is None
    assert screen._budget_box.value() == \
        align_mod.DEFAULT_MAX_BUFFER_BYTES >> 20
    # An unknown order/blend name leaves the combos where they were.
    before = (screen._order_combo.currentText(),
              screen._blend_combo.currentText())
    screen.apply_settings({'order': 'spiral', 'blend': 'dissolve'})
    assert (screen._order_combo.currentText(),
            screen._blend_combo.currentText()) == before
    # A tiny budget is floored rather than set to zero.
    screen.apply_settings({'max_buffer_bytes': 1024})
    assert screen._budget_box.value() == 4
