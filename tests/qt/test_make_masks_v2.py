"""Tests for the make-masks Qt screen's newer wand / zoom / undo wiring."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import Qt

from spacr.qt.screens.make_masks import (
    MODE_BRUSH,
    MODE_ERASE,
    MODE_WAND_ADD,
    MODE_ZOOM,
    MakeMasksScreen,
)


@pytest.fixture
def synth_mask_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "make_masks_v2"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        arr = rng.integers(0, 65535, size=(64, 64), dtype=np.uint16)
        imageio.imwrite(folder / f"img_{i:02d}.tif", arr)
    return folder


def test_screen_starts_with_mode_buttons_but_disabled(qtbot, qt_theme_applied):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    for m in (MODE_BRUSH, MODE_ERASE, MODE_WAND_ADD, MODE_ZOOM):
        btn = screen._mode_buttons[m]
        assert not btn.isEnabled()      # disabled until a folder is opened


def test_set_mode_toggles_only_selected(qtbot, qt_theme_applied,
                                          synth_mask_folder: Path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    screen._set_mode(MODE_BRUSH)
    assert screen._mode_buttons[MODE_BRUSH].isChecked()
    for m in (MODE_ERASE, MODE_WAND_ADD, MODE_ZOOM):
        assert not screen._mode_buttons[m].isChecked()
    screen._set_mode(MODE_WAND_ADD)
    assert screen._mode_buttons[MODE_WAND_ADD].isChecked()
    assert not screen._mode_buttons[MODE_BRUSH].isChecked()


def test_wand_tolerance_syncs_to_canvas(qtbot, qt_theme_applied):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._wand_tol.setValue(750.0)
    assert screen._canvas.wand_tolerance == 750.0


def test_single_wand_ctrl_edit_undo_redo_save_next_and_reopen(qtbot, qt_theme_applied, tmp_path):
    folder = tmp_path / 'wand-modifier'
    folder.mkdir()
    (folder / 'masks').mkdir()
    image = np.zeros((64, 64), dtype=np.uint16)
    image[20:40, 20:40] = 30000
    original_mask = np.full((64, 64), 77, dtype=np.uint16)
    for name in ('a.tif', 'b.tif'):
        imageio.imwrite(folder / name, image)
        imageio.imwrite(folder / 'masks' / name, original_mask)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1500, 900)
    screen.show()
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading)
    screen._set_mode(MODE_WAND_ADD)
    screen._canvas.wand_tolerance = 100.0
    screen._canvas.wand_max_pixels = 10000
    expected = original_mask.copy()
    expected[20:40, 20:40] = 0
    position = screen._canvas._image_to_canvas(30, 30)
    qtbot.mouseClick(screen._canvas, Qt.LeftButton, Qt.ControlModifier, pos=position)
    np.testing.assert_array_equal(screen._canvas.mask, expected)
    assert screen._log.edits[-1].kind == 'wand'
    assert screen._log.edits[-1].detail['action'] == 'erase'
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, original_mask)
    screen._on_redo()
    np.testing.assert_array_equal(screen._canvas.mask, expected)
    screen._on_save()
    screen._on_next()
    screen.close()
    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(str(folder))
    qtbot.waitUntil(lambda: not reopened._loading)
    np.testing.assert_array_equal(reopened._canvas.mask > 0, expected > 0)
    np.testing.assert_array_equal(imageio.imread(folder / 'a.tif'), image)
    reopened._set_mode(MODE_WAND_ADD)
    reopened._canvas.wand_tolerance = 100.0
    reopened._canvas.wand_max_pixels = 10000
    position = reopened._canvas._image_to_canvas(30, 30)
    qtbot.mouseClick(reopened._canvas, Qt.LeftButton, pos=position)
    assert np.all(reopened._canvas.mask > 0)
    reopened.close()


def test_history_undo_reverts_object_op(qtbot, qt_theme_applied,
                                          synth_mask_folder: Path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    # Paint a blob and then run clear-mask → should be undoable back to blob
    screen._canvas.mask[10:20, 10:20] = 255
    before = screen._canvas.mask.copy()
    screen._history.push(before)              # simulate stroke completion
    # Now clear
    screen._canvas.mask = np.zeros_like(screen._canvas.mask)
    screen._history.push(screen._canvas.mask)
    assert screen._history.can_undo()
    screen._on_undo()
    assert (screen._canvas.mask == before).all()
    assert screen._history.can_redo()
    screen._on_redo()
    assert not screen._canvas.mask.any()


def test_new_image_resets_history(qtbot, qt_theme_applied,
                                    synth_mask_folder: Path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    # Push a fake edit
    screen._canvas.mask[0:5, 0:5] = 100
    screen._history.push(screen._canvas.mask)
    assert screen._history.can_undo()
    # Navigating to next image should reset history to seed
    screen._on_next()
    assert not screen._history.can_undo()


def test_zoom_reset_button_enabled_when_zoomed(qtbot, qt_theme_applied,
                                                 synth_mask_folder: Path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    assert not screen._btn_reset_zoom.isEnabled()
    # Simulate a zoom via direct state (no widget-space click needed)
    screen._canvas._zoom_x0, screen._canvas._zoom_y0 = 5, 5
    screen._canvas._zoom_x1, screen._canvas._zoom_y1 = 40, 40
    screen._canvas.zoom_changed.emit(True)
    assert screen._btn_reset_zoom.isEnabled()
    screen._on_reset_zoom()
    assert not screen._canvas.is_zoomed()
    assert not screen._btn_reset_zoom.isEnabled()
