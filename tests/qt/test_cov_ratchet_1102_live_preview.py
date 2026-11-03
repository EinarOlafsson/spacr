"""The live preview rebuilds its scene for a new frame size and skips hidden files."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPixmap

from spacr.qt.widgets import live_preview as LP

pytestmark = pytest.mark.qt


def _pixmap(width, height, value):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(value)
    return QPixmap.fromImage(image)


def test_a_frame_of_another_size_or_the_first_frame_rebuilds_the_scene(qtbot):
    view = LP._ZoomView()
    qtbot.addWidget(view)
    assert view._pixmap_item is None
    view._show_frame(_pixmap(20, 10, 0xFF0000))          # no item yet: a new field
    first_item = view._pixmap_item
    assert first_item is not None and first_item.pixmap().size().width() == 20
    view._show_frame(_pixmap(40, 30, 0x00FF00))          # another size: a new field
    assert view._pixmap_item is not first_item
    assert view._pixmap_item.pixmap().size().height() == 30
    same_item = view._pixmap_item
    view._show_frame(_pixmap(40, 30, 0x0000FF))          # same size: the item is kept
    assert view._pixmap_item is same_item


def test_hidden_files_in_subfolders_are_not_the_first_image(tmp_path):
    nested = tmp_path / "plate"
    nested.mkdir()
    np_image = np.zeros((4, 4), dtype=np.uint8)
    import tifffile

    tifffile.imwrite(nested / ".hidden.tif", np_image)
    tifffile.imwrite(nested / "visible.tif", np_image)
    (tmp_path / "notes.txt").write_text("not an image")
    assert LP.first_supported_image(tmp_path) == nested / "visible.tif"
