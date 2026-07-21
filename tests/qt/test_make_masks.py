"""Tests for the Qt make-masks screen + mask_engine."""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import Qt

from spacr.qt import mask_engine as engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_mask_folder(tmp_path: Path) -> Path:
    """Folder with 3 image files and no pre-existing masks."""
    folder = tmp_path / "make_masks_folder"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.integers(0, 65535, size=(80, 100), dtype=np.uint16)
        imageio.imwrite(folder / f"img_{i:02d}.tif", arr)
    return folder


@pytest.fixture
def synth_mask_folder_with_masks(tmp_path: Path) -> Path:
    """Folder with images AND matching masks in masks/ subdir."""
    folder = tmp_path / "make_masks_with_masks"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(2):
        img = rng.integers(0, 65535, size=(64, 64), dtype=np.uint16)
        imageio.imwrite(folder / f"a_{i:02d}.tif", img)
        m = np.zeros((64, 64), dtype=np.uint8)
        m[10:20, 10:20] = 255  # a single object
        imageio.imwrite(folder / "masks" / f"a_{i:02d}.tif", m)
    return folder


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

def test_list_images_orders_alphabetically(synth_mask_folder: Path):
    files = engine.list_images(str(synth_mask_folder))
    assert files == ["img_00.tif", "img_01.tif", "img_02.tif"]


def test_list_images_empty_returns_empty(tmp_path: Path):
    assert engine.list_images(str(tmp_path)) == []


def test_load_image_creates_blank_mask_when_missing(synth_mask_folder: Path):
    image, mask = engine.load_image_and_mask(str(synth_mask_folder), "img_00.tif")
    assert image.dtype == np.uint16
    assert mask.dtype == np.uint8
    assert mask.shape == image.shape[:2]
    assert not mask.any()


def test_load_image_reads_existing_mask(synth_mask_folder_with_masks: Path):
    image, mask = engine.load_image_and_mask(str(synth_mask_folder_with_masks), "a_00.tif")
    assert mask.any()
    assert mask.dtype == np.uint8


def test_save_mask_writes_labeled_tif(synth_mask_folder: Path):
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[40:50, 60:70] = 255
    save_path = engine.save_mask(str(synth_mask_folder), "img_00.tif", mask)
    assert os.path.isfile(save_path)
    written = imageio.imread(save_path)
    # Two connected components → labels 1 and 2
    assert set(np.unique(written)) == {0, 1, 2}


def test_paint_disk_and_line():
    mask = np.zeros((30, 30), dtype=np.uint8)
    engine.paint_disk(mask, 15, 15, 4)
    assert mask[15, 15] == 255
    assert mask[15, 20] == 0
    engine.paint_line(mask, 0, 0, 29, 29, radius=1)
    assert mask[0, 0] == 255
    assert mask[29, 29] == 255


def test_fill_holes_fills():
    mask = np.zeros((20, 20), dtype=np.uint8)
    # Draw a ring with a hole
    mask[3:17, 3:5] = 255
    mask[3:17, 15:17] = 255
    mask[3:5, 3:17] = 255
    mask[15:17, 3:17] = 255
    filled = engine.fill_holes(mask)
    # After fill, the inside should be labeled non-zero
    assert filled[10, 10] > 0


def test_relabel_and_clear():
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[1:3, 1:3] = 100
    mask[8:10, 8:10] = 200
    relabeled = engine.relabel_objects(mask)
    unique_nonzero = sorted(int(v) for v in np.unique(relabeled) if v > 0)
    assert unique_nonzero == [1, 2]
    cleared = engine.clear_mask(relabeled)
    assert not cleared.any()


def test_remove_small_objects_drops_below_threshold():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:2, 0:2] = 255       # 4 px object
    mask[10:20, 10:20] = 255   # 100 px object
    out = engine.remove_small_objects(mask, min_area=10)
    labeled_nonzero = int((out > 0).sum())
    assert labeled_nonzero == 100


def test_erase_object_at():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[6:9, 6:9] = 2
    out = engine.erase_object_at(mask, 2, 2)
    assert not (out == 1).any()
    assert (out == 2).any()
    # Out-of-bounds is a no-op
    out2 = engine.erase_object_at(mask, 999, 999)
    assert (out2 == mask).all()


def test_overlay_mask_returns_rgb_uint8():
    image = np.zeros((10, 10), dtype=np.uint16)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:6, 3:6] = 1
    overlaid = engine.overlay_mask(image, mask)
    assert overlaid.shape == (10, 10, 3)
    assert overlaid.dtype == np.uint8


# ---------------------------------------------------------------------------
# Widget tests
# ---------------------------------------------------------------------------

def test_make_masks_screen_constructs(qtbot, qt_theme_applied):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    # Nav buttons start disabled (no folder open)
    assert not screen._btn_prev.isEnabled()
    assert not screen._btn_next.isEnabled()
    assert not screen._btn_save.isEnabled()


def test_make_masks_open_folder_loads_first_image(qtbot, qt_theme_applied,
                                                    synth_mask_folder: Path):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    assert screen._current_index == 0
    assert len(screen._image_files) == 3
    assert screen._canvas.image is not None
    assert screen._canvas.mask is not None
    # Nav is now enabled
    assert screen._btn_next.isEnabled()
    assert screen._btn_save.isEnabled()


def test_make_masks_navigation(qtbot, qt_theme_applied, synth_mask_folder: Path):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    assert screen._current_index == 0
    screen._on_next()
    assert screen._current_index == 1
    screen._on_next()
    assert screen._current_index == 2
    # Next at last is a no-op
    screen._on_next()
    assert screen._current_index == 2
    screen._on_prev()
    assert screen._current_index == 1


def test_make_masks_save_creates_file(qtbot, qt_theme_applied,
                                        synth_mask_folder: Path):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    # Force something into the mask, then save
    screen._canvas.mask[10:20, 10:20] = 255
    screen._on_save()
    out = synth_mask_folder / "masks" / "img_00.tif"
    assert out.is_file()


def test_make_masks_object_ops(qtbot, qt_theme_applied, synth_mask_folder: Path):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    # Paint a rough blob
    screen._canvas.mask[10:20, 10:20] = 255
    screen._canvas.mask[5:7, 5:7] = 255
    # Remove small — 4px blob should vanish, 100px stays
    screen._min_area.setValue(10)
    screen._on_remove_small()
    assert not (screen._canvas.mask[5:7, 5:7] > 0).any()
    # Relabel + fill holes don't crash
    screen._on_relabel()
    screen._on_fill_holes()
