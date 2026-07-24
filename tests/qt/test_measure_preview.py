"""Tests for the Measure crop live-preview panel."""
from __future__ import annotations

import numpy as np


def _merged_npy(tmp_path):
    H = W = 48
    data = np.zeros((H, W, 5), np.float32)   # 4 image channels + cell mask
    data[..., 0] = 12; data[..., 1] = 24; data[..., 2] = 36; data[..., 3] = 5
    mask = np.zeros((H, W), np.int32)
    mask[2:8, 2:8] = 1
    mask[20:40, 20:40] = 2
    data[..., 4] = mask
    p = tmp_path / "plate1_A01_f1.npy"
    np.save(str(p), data)
    return str(p)


def test_panel_loads_and_crops(qtbot, tmp_path):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    p = MeasurePreviewPanel()
    qtbot.addWidget(p)
    p._mask_dim.setValue(4)          # cell mask is at slice 4 here
    assert p.load_array(_merged_npy(tmp_path)) is True
    assert len(p._crops) == 2
    # crops are RGB uint8
    assert p._crops[0]["crop"].shape[2] == 3


def test_area_filter_and_settings(qtbot, tmp_path):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    p = MeasurePreviewPanel()
    qtbot.addWidget(p)
    p._mask_dim.setValue(4)
    p.load_array(_merged_npy(tmp_path))
    p._min_area.setValue(100)        # triggers a live re-crop
    assert len(p._crops) == 1        # only the big object survives


def test_propagation_maps_measure_keys(qtbot, tmp_path):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    p = MeasurePreviewPanel()
    qtbot.addWidget(p)
    p._channels.setText("0,2,4")
    p._crop_size.setValue(200)
    s = p.settings_for_propagation()
    assert s["png_dims"] == [0, 2, 4]
    assert s["png_size"] == [200, 200]
    assert s["crop_mode"] == ["cell"]
    captured = {}
    p.set_propagate_callback(lambda d: captured.update(d))
    p.propagate_settings()
    assert captured["png_dims"] == [0, 2, 4]


def test_click_selects_thumb(qtbot, tmp_path):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    p = MeasurePreviewPanel()
    qtbot.addWidget(p)
    p._mask_dim.setValue(4)
    p.load_array(_merged_npy(tmp_path))
    p._on_thumb_clicked(0)
    assert 0 in p._selected
    p._on_thumb_clicked(0)           # toggles off
    assert 0 not in p._selected


def test_measure_screen_has_preview(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    scr = AppScreen("measure")
    qtbot.addWidget(scr)
    assert getattr(scr, "_measure_preview", None) is not None
    # mask keeps its own live preview; other apps have neither
    other = AppScreen("umap")
    qtbot.addWidget(other)
    assert getattr(other, "_measure_preview", None) is None
