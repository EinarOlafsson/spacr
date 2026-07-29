"""Align & Stitch — the Tools screen that plans a stitch before writing it.

Everything here runs offscreen against real temporary ``.npy`` tiles.
The suite pins the properties the panel lives or dies by:

* it **plans before it writes** — the write button stays disabled until a
  plan exists, and planning allocates no canvas;
* a tile that **did not register is visible**: orange in the layout,
  named in the report, and called out in the status line;
* the status line states the **canvas size and the RAM the write needs**
  before the write button is usable;
* the layout view draws **one rectangle per tile** in the right place;
* long work goes **off the GUI thread**, and the completion handler runs
  back on it (a bound method, never a closure);
* errors land **inline**, never in a modal dialog — a QMessageBox would
  hang a headless run forever;
* settings round-trip, so the Batch Runner can snapshot this screen.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from spacr import align as align_mod
from spacr.qt.screens.align import (
    AlignScreen,
    TileLayoutWidget,
    confidence_colour,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite
    on a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


def _texture(height, width, seed=0, sigma=2.0):
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
        r, c = divmod(k, 2)
        np.save(folder / f"plate1_B07_{k + 1:03d}.npy",
                big[r * 100:r * 100 + 128, c * 100:c * 100 + 128])
    return str(folder)


@pytest.fixture
def broken_tile_folder(tmp_path):
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
    screen._src_edit.setText(src)
    if grid:
        screen._rows_box.setValue(grid[0])
        screen._cols_box.setValue(grid[1])
    screen._overlap_box.setValue(overlap)
    if dst is not None:
        screen._dst_edit.setText(str(dst))
    return screen


# ---------------------------------------------------------------------------
# Registration in the app table
# ---------------------------------------------------------------------------

def test_registered_as_a_tools_app_with_a_title_and_intro():
    """The align key is wired into the app registry.

    ``qt/app.py`` is owned by another change in flight; until the
    registration diff lands this skips rather than failing, so the rest of
    the suite still tells the truth.
    """
    from spacr.qt import app as qt_app
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES

    entry = next((a for a in qt_app.APPS if a[0] == "align"), None)
    if entry is None:
        pytest.skip("align is not registered in qt/app.py yet — the "
                    "registration diff is applied separately")
    key, name, description, section = entry
    assert name == "Align & Stitch"
    assert description.strip()
    assert section in qt_app.SECTIONS
    assert APP_TITLES.get("align", "").strip() == "Align & Stitch"
    assert len(APP_INTROS.get("align", "")) > 40


def test_build_screen_returns_an_align_screen(qtbot, qt_theme_applied):
    from spacr.qt import app as qt_app

    if not any(a[0] == "align" for a in qt_app.APPS):
        pytest.skip("align is not registered in qt/app.py yet")
    window = qt_app.MainWindow()
    qtbot.addWidget(window)
    widget = window._build_screen("align")
    qtbot.addWidget(widget)
    assert isinstance(widget, AlignScreen)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_write_is_disabled_until_a_plan_exists(screen, tile_folder, tmp_path):
    """You cannot write 700 MB without having looked at the plan."""
    assert not screen._btn_write.isEnabled()
    _configure(screen, tile_folder, dst=tmp_path / "out")
    assert not screen._btn_write.isEnabled(), \
        "a destination alone must not enable the write"
    assert screen.build_plan() is True
    assert screen._btn_write.isEnabled()


def test_plan_draws_the_layout_and_reports_the_canvas(screen, tile_folder):
    _configure(screen, tile_folder)
    assert screen.build_plan() is True

    plan = screen.plan()
    assert plan is not None
    assert plan.n_registered == 4
    assert plan.canvas_shape[:2] == (228, 228)

    rects = screen._layout_view.tile_rects()
    assert len(rects) == 4
    assert {i for i, _r in rects} == {0, 1, 2, 3}
    # The grid really is a grid: tile 1 is right of tile 0, tile 2 below.
    by_index = dict(rects)
    assert by_index[1].left() > by_index[0].left()
    assert by_index[2].top() > by_index[0].top()
    assert by_index[3].left() > by_index[0].left()
    assert by_index[3].top() > by_index[0].top()

    report = screen.report_text()
    assert "canvas" in report and "registered      4" in report
    status = screen.status_text()
    assert "4 of 4 tile(s) registered" in status
    assert "228 x 228" in status
    # The RAM cost is stated before the write button is used.
    assert "of RAM" in status
    assert "band" in status
    assert screen.last_error == ""


def test_plan_emits_plan_ready(screen, tile_folder, qtbot):
    _configure(screen, tile_folder)
    with qtbot.waitSignal(screen.plan_ready, timeout=30000) as blocker:
        screen.build_plan()
    assert blocker.args[0] is screen.plan()


def test_a_tile_that_did_not_register_is_orange_and_named(screen,
                                                          broken_tile_folder):
    """The failure this screen exists to expose is visible three ways."""
    _configure(screen, broken_tile_folder, overlap=0.3, grid=(1, 2))
    assert screen.build_plan() is True

    plan = screen.plan()
    assert plan.n_nominal == 2

    # 1. the status line says so, and says so as an error
    assert "did NOT register" in screen.status_text()
    assert screen.last_error

    # 2. the report names them
    assert "placed by stage position only" in screen.report_text()

    # 3. the layout paints them in the warning colour
    from spacr.qt.theme import DARK_PALETTE
    for placement in plan.placements:
        colour = confidence_colour(placement.confidence, placement.method)
        assert colour == QColor(DARK_PALETTE["warning"])


def test_confidence_colour_ramps_and_flags():
    from spacr.qt.theme import DARK_PALETTE
    warning = QColor(DARK_PALETTE["warning"])
    assert confidence_colour(0.0, align_mod.METHOD_NOMINAL) == warning
    assert confidence_colour(0.9, align_mod.METHOD_NOMINAL) == warning, \
        "a nominal placement is orange whatever its score"
    assert confidence_colour(0.0, align_mod.METHOD_UNREADABLE) == \
        QColor(DARK_PALETTE["error"])
    weak = confidence_colour(0.35, align_mod.METHOD_REGISTRATION)
    strong = confidence_colour(1.0, align_mod.METHOD_REGISTRATION)
    assert weak != strong
    assert strong == QColor(DARK_PALETTE["accent"])


def test_plan_without_a_source_reports_inline(screen):
    assert screen.build_plan() is False
    assert "Choose a folder of tiles" in screen.status_text()
    assert screen.last_error


def test_a_bad_source_lands_in_the_status_label(screen, tmp_path):
    """No traceback, no dialog (the autouse fixture would fire), just text."""
    _configure(screen, str(tmp_path / "does_not_exist"), grid=None)
    assert screen.build_plan() is False
    assert screen.last_error
    assert "does not exist" in screen.status_text()
    assert screen.plan() is None


def test_an_empty_folder_names_the_problem(screen, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    _configure(screen, str(empty), grid=None)
    assert screen.build_plan() is False
    assert "no .npy/.tif tiles" in screen.status_text()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_write_produces_the_stack_and_reports_the_ratio(screen, tile_folder,
                                                        tmp_path):
    _configure(screen, tile_folder, dst=tmp_path / "out")
    screen.build_plan()
    assert screen.write_stack() is True

    result = screen.result()
    assert result is not None
    assert os.path.isfile(result.stack_path)
    written = np.load(result.stack_path, mmap_mode="r")
    assert written.shape == screen.plan().canvas_shape
    assert isinstance(written, np.memmap)

    assert "Stitched 4 tile(s)" in screen.status_text()
    assert "Peak RAM buffer" in screen.report_text()


def test_the_ram_budget_control_actually_bites(screen, tmp_path):
    """Turning the budget down splits the write into more, smaller bands."""
    # A canvas big enough that a 1 MB budget cannot hold it in one band:
    # 912 x 912 x uint16 costs 9120 bytes of buffer per canvas row.
    folder = tmp_path / "big"
    folder.mkdir()
    big = _texture(920, 920, seed=44)
    for k in range(4):
        r, c = divmod(k, 2)
        np.save(folder / f"plate1_B07_{k + 1:03d}.npy",
                big[r * 400:r * 400 + 512, c * 400:c * 400 + 512])
    tile_folder = str(folder)

    _configure(screen, tile_folder, dst=tmp_path / "roomy",
               overlap=1 - 400 / 512)
    screen._budget_box.setValue(64)
    screen.build_plan()
    assert "band" in screen.status_text()
    screen.write_stack()
    roomy = screen.result()

    _configure(screen, tile_folder, dst=tmp_path / "tight",
               overlap=1 - 400 / 512)
    screen._budget_box.setValue(1)
    screen.build_plan()
    screen.write_stack()
    tight = screen.result()

    assert tight.band_rows < roomy.band_rows
    assert tight.peak_buffer_bytes <= (1 << 20)
    assert np.array_equal(np.load(tight.stack_path),
                          np.load(roomy.stack_path)), \
        "the band height must not change a single output pixel"


def test_write_records_coordinates_when_a_database_is_given(screen,
                                                            tile_folder,
                                                            tmp_path):
    _configure(screen, tile_folder, dst=tmp_path / "out")
    db_path = tmp_path / "measurements.db"
    screen._db_edit.setText(str(db_path))
    screen.build_plan()
    assert screen.write_stack() is True

    assert screen.result().db_path == str(db_path)
    frame = align_mod.read_coordinates(db_path)
    assert len(frame) == 4
    assert set(frame["method"]) == {align_mod.METHOD_REGISTRATION}
    assert str(screen.result().stack_path) in set(frame["stack_path"])
    assert "Coordinates written to" in screen.status_text()


def test_write_without_a_plan_or_destination_reports_inline(screen,
                                                            tile_folder):
    assert screen.write_stack() is False
    assert "Press Plan first" in screen.status_text()
    _configure(screen, tile_folder)
    screen.build_plan()
    assert screen.write_stack() is False
    assert "Choose where to write" in screen.status_text()


def test_existing_output_is_refused_inline_not_clobbered(screen, tile_folder,
                                                         tmp_path):
    out = tmp_path / "out"
    _configure(screen, tile_folder, dst=out)
    screen.build_plan()
    assert screen.write_stack() is True
    first = np.load(screen.result().stack_path).copy()

    assert screen.write_stack() is False
    assert "already exists" in screen.status_text()
    assert screen.last_error
    assert np.array_equal(np.load(screen.result().stack_path), first)

    screen._overwrite_box.setChecked(True)
    assert screen.write_stack() is True
    assert screen.last_error == ""


def test_write_emits_stack_written(screen, tile_folder, tmp_path, qtbot):
    _configure(screen, tile_folder, dst=tmp_path / "out")
    screen.build_plan()
    with qtbot.waitSignal(screen.stack_written, timeout=30000) as blocker:
        screen.write_stack()
    assert blocker.args[0] == screen.result().stack_path


# ---------------------------------------------------------------------------
# Layout widget
# ---------------------------------------------------------------------------

def test_layout_widget_is_empty_until_a_plan_arrives(qtbot, qt_theme_applied):
    view = TileLayoutWidget()
    qtbot.addWidget(view)
    view.resize(300, 200)
    assert view.tile_rects() == []
    assert view.plan() is None
    view.repaint()                    # the empty-state path must not raise


def test_layout_widget_repaints_and_reports_clicks(screen, tile_folder, qtbot):
    _configure(screen, tile_folder)
    screen.build_plan()
    view = screen._layout_view
    view.resize(400, 400)
    view.repaint()

    rects = dict(view.tile_rects())
    assert len(rects) == 4
    centre = rects[3].center()
    with qtbot.waitSignal(view.tile_clicked, timeout=5000) as blocker:
        qtbot.mouseClick(view, Qt.LeftButton, pos=centre.toPoint())
    assert blocker.args[0] == 3
    assert "residual" in screen.tile_info_text()
    assert os.path.basename(screen.plan().placements[3].tile.path) in \
        screen.tile_info_text()


def test_clicking_the_void_clears_the_tile_readout(screen, tile_folder):
    _configure(screen, tile_folder)
    screen.build_plan()
    screen._on_tile_clicked(2)
    assert screen.tile_info_text()
    screen._on_tile_clicked(-1)
    assert screen.tile_info_text() == ""


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------

def test_settings_round_trip(screen, tmp_path):
    settings = align_mod.default_settings({
        'src': str(tmp_path / 'tiles'),
        'dst': str(tmp_path / 'out'),
        'db_path': str(tmp_path / 'm.db'),
        'grid': (4, 5),
        'overlap': 0.15,
        'order': 'snake-row',
        'reference_channel': 2,
        'min_confidence': 0.55,
        'neighbour_radius': 3,
        'blend': 'average',
        'max_buffer_bytes': 32 << 20,
        'overwrite': True,
    })
    screen.apply_settings(settings)
    back = screen.settings()
    for key in ('src', 'dst', 'db_path', 'grid', 'order', 'reference_channel',
                'neighbour_radius', 'blend', 'max_buffer_bytes', 'overwrite'):
        assert back[key] == settings[key], key
    assert back['overlap'] == pytest.approx(0.15)
    assert back['min_confidence'] == pytest.approx(0.55)


def test_auto_grid_is_the_default(screen):
    assert screen.settings()['grid'] is None
    assert screen._rows_box.specialValueText() == "auto"


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_threaded_plan_runs_off_the_gui_thread_and_settles(qtbot,
                                                           qt_theme_applied,
                                                           tile_folder):
    """The real threaded path: worker thread, bound-method completion."""
    widget = AlignScreen(threaded=True)
    qtbot.addWidget(widget)
    _configure(widget, tile_folder)

    with qtbot.waitSignal(widget.job_finished, timeout=60000) as blocker:
        assert widget.build_plan() is True
        assert widget.is_busy()
    assert blocker.args[0] is True
    assert widget.plan() is not None
    assert not widget.is_busy()
    assert widget._btn_plan.isEnabled()

    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)


def test_threaded_failure_reports_inline_and_re_enables(qtbot,
                                                        qt_theme_applied,
                                                        tmp_path):
    widget = AlignScreen(threaded=True)
    qtbot.addWidget(widget)
    _configure(widget, str(tmp_path / "nowhere"), grid=None)

    with qtbot.waitSignal(widget.job_finished, timeout=60000) as blocker:
        widget.build_plan()
    assert blocker.args[0] is False
    assert widget.last_error
    assert widget.plan() is None
    assert widget._btn_plan.isEnabled()

    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)


def test_worker_is_not_scheduled_for_deletion(qtbot, qt_theme_applied,
                                              tile_folder):
    """``deleteLater`` on the worker segfaulted the process; it must stay off.

    ``bridge.make_thread`` owns that decision, and this screen must not
    quietly re-add it — the crash was a double-owner race between the
    worker thread's deferred-delete flush and the GUI thread dropping the
    last Python reference.
    """
    import inspect

    from spacr.qt.screens import align as screen_mod

    source = inspect.getsource(screen_mod)
    assert "deleteLater" not in source, \
        "the align screen must not schedule the worker for deletion"

    widget = AlignScreen(threaded=True)
    qtbot.addWidget(widget)
    _configure(widget, tile_folder)
    with qtbot.waitSignal(widget.job_finished, timeout=60000):
        widget.build_plan()
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)
    # Still alive, still usable — the crash showed up here.
    assert widget.plan() is not None


def test_controls_are_disabled_while_busy(qtbot, qt_theme_applied,
                                          tile_folder):
    widget = AlignScreen(threaded=True)
    qtbot.addWidget(widget)
    _configure(widget, tile_folder)
    widget.build_plan()
    assert widget.is_busy()
    assert not widget._btn_plan.isEnabled()
    assert not widget._src_edit.isEnabled()
    qtbot.waitUntil(lambda: not widget.is_busy(), timeout=60000)
    assert widget._src_edit.isEnabled()
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)
