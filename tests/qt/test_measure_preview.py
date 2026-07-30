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


def _categorised_merged_npy(tmp_path):
    data = np.zeros((48, 48, 8), np.float32)
    data[..., :3] = 20
    cell = np.zeros((48, 48), np.int32)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    organelle = np.zeros_like(cell)
    cell[2:18, 2:18] = 1
    nucleus[5:10, 5:10] = 1
    organelle[11:14, 11:14] = 1
    cell[24:42, 24:42] = 2
    pathogen[28:33, 28:33] = 1
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    data[..., 7] = organelle
    path = tmp_path / "phenotypes.npy"
    np.save(path, data)
    return str(path)


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


def test_settings_dialog_has_pipeline_tabs_and_valid_normalize_contract(qtbot):
    from PySide6.QtWidgets import QTabWidget
    from spacr.qt.widgets.measure_preview import (
        CropSettingsDialog, MeasurePreviewPanel,
    )
    panel = MeasurePreviewPanel()
    qtbot.addWidget(panel)
    dialog = CropSettingsDialog(panel)
    qtbot.addWidget(dialog)
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "General", "Object crops", "Filter settings", "Preview",
    ]
    for widget in panel._managed_widgets():
        if widget is panel._propagate_btn:
            continue
        assert widget.property("apiTooltipHtml")
        label = getattr(widget, "_spacr_setting_label", None)
        if label is not None:
            assert widget.toolTip() == ""
            assert "https://" in label.toolTip()
            assert getattr(label, "_spacr_api_dot", None) is not None
        else:
            assert "https://" in widget.toolTip()
            assert getattr(widget, "_spacr_api_dot", None) is not None
    propagated = panel.settings_for_propagation()
    assert propagated["normalize"] == [1.0, 99.0]
    panel._normalise.setChecked(False)
    assert panel.settings_for_propagation()["normalize"] is False


def test_cells_are_grouped_by_nucleus_pathogen_and_organelle(qtbot, tmp_path):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    panel = MeasurePreviewPanel()
    qtbot.addWidget(panel)
    panel._mask_dims["organelle"].setValue(7)
    assert panel.load_array(_categorised_merged_npy(tmp_path))
    categories = {entry["label"]: entry["category"] for entry in panel._crops}
    assert "Nucleated" in categories[1]
    assert "Uninfected" in categories[1]
    assert "Organelle+" in categories[1]
    assert "Unnucleated" in categories[2]
    assert "Infected" in categories[2]
    assert "Organelle−" in categories[2]


def test_measure_filter_settings_are_a_separate_section(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    titles = [section.title() for section in screen._settings_sections]
    assert "FILTER SETTINGS" in titles
    filter_section = next(
        section for section in screen._settings_sections
        if section.title() == "FILTER SETTINGS"
    )
    labels = {
        label.text() for label, _widget in filter_section._row_widgets
        if label is not None
    }
    assert "Cell min size" in labels
    assert "Keep uninfected cells" in labels


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
