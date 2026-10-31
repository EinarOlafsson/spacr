"""A corrupt TIFF is reported where it is read, never raised out of a handler.

tifffile's own ``TiffFileError`` is not an ``OSError`` or a ``ValueError``
(in tifffile 2023.4 it derives from ``Exception`` only), and Pillow, which
imageio uses for TIFFs, raises ``SyntaxError`` for a broken directory. A
handler written for ``(OSError, ValueError)`` therefore let a corrupt file
escape as an unhandled exception in a Qt slot or a worker, or miscounted it.
Each handler below sees a deliberately corrupt file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.tiff_io import TIFF_READ_ERRORS

GARBAGE = b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40


def _corrupt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(GARBAGE)
    return path


def test_the_shared_tuple_names_tifffile_s_own_error(tmp_path):
    bad = _corrupt(tmp_path / "bad.tif")
    with pytest.raises(Exception) as raised:
        tifffile.imread(bad)
    assert isinstance(raised.value, TIFF_READ_ERRORS)
    assert OSError in TIFF_READ_ERRORS and ValueError in TIFF_READ_ERRORS


def test_a_layer_whose_mask_file_is_broken_has_no_objects(tmp_path):
    from spacr.qt import mask_engine as engine

    layer = tmp_path / "layer"
    _corrupt(layer / "f_0.tif")
    assert engine._layer_has_objects(str(layer), "f_0.tif") is False


def test_a_corrupt_pixel_size_file_has_none(tmp_path):
    import spacr.io as sio

    bad = _corrupt(tmp_path / "bad.tif")
    assert sio.tiff_pixel_size_nm(bad) is None


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    from spacr import cli_make_masks
    from spacr.qt.screens.make_masks import MakeMasksScreen

    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)


def test_opening_a_corrupt_volume_says_so_and_stays_open(screen, tmp_path):
    bad = _corrupt(tmp_path / "volume.tif")
    assert screen._show_volume(str(tmp_path), bad.name) is False


def test_a_corrupt_mask_is_skipped_and_counted_when_lineage_is_merged(
        screen, tmp_path):
    folder = tmp_path / "plates"
    folder.mkdir()
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(folder / name, np.ones((8, 8), np.uint16))
    (folder / "masks").mkdir()
    tifffile.imwrite(folder / "masks" / "a.tif", np.ones((8, 8), np.uint16))
    _corrupt(folder / "masks" / "b.tif")
    screen._folder = str(folder)
    screen._image_files = ["a.tif", "b.tif"]
    screen._masks_dir = str(folder / "masks")
    state = {"cells": {}}
    screen._merge_sequence_lineage(state, str(folder))
    assert screen._status_label.text()
