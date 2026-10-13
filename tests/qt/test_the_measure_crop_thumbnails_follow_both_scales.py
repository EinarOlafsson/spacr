"""The Measure crop preview's thumbnails follow the preview and GUI scales.

Pinned behaviour of :class:`spacr.qt.widgets.measure_preview.MeasurePreviewPanel`:

* a panel opened while the GUI scale is not 100 % sizes its thumbnails for
  that scale from the start, not only after the next scale change;
* changing the preview scale while crops are on screen redraws them at the
  new size straight away.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt import gui_scale  # noqa: E402
from spacr.qt.widgets import measure_preview as MP  # noqa: E402

pytestmark = pytest.mark.qt


def _merged(directory):
    """One merged array: three image planes and a cell mask with two cells."""
    data = np.zeros((32, 32, 8), np.float32)
    data[..., :3] = 20
    cell = np.zeros((32, 32), np.int32)
    cell[2:14, 2:14] = 1
    cell[18:30, 18:30] = 2
    data[..., 4] = cell
    path = directory / "plate1_A01_f1.npy"
    np.save(path, data)
    return str(path)


def _panel(qtbot):
    panel = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


def _thumb_widths(panel):
    return [thumb.pixmap().width() for thumb in panel.findChildren(MP._CropThumb)]


def test_a_panel_opened_at_a_larger_gui_scale_starts_with_larger_thumbnails(
        qtbot, monkeypatch):
    plain = _panel(qtbot)
    assert plain._thumb_px == 132

    monkeypatch.setattr(gui_scale, "current_scale", lambda: 1.5)
    scaled = _panel(qtbot)

    assert scaled._thumb_px == 198


def test_a_preview_scale_change_redraws_the_crops_on_screen(qtbot, tmp_path):
    panel = _panel(qtbot)
    assert panel.load_array(_merged(tmp_path)) is True
    before = _thumb_widths(panel)
    assert len(before) == 2

    panel._scale_control.scaler.set_scale(2.0, persist=False)

    qtbot.waitUntil(lambda: len(_thumb_widths(panel)) == 2)
    after = _thumb_widths(panel)
    assert min(after) > max(before)
