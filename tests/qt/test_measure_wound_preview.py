"""The Measure preview draws the wound edge, as an Alpha control.

The Wound button runs what a Measure run with ``wound_closure`` on runs for
the first frame of a field, and it is present only when Preferences -> Show
alpha features is on (``ALPHA_FEATURES[536]``). The wound-closure settings
are hidden from the Measure form on the same switch.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from spacr.qt import preferences
from spacr.qt.widgets import measure_preview as MP


def _field(directory, width=90):
    """A merged array: brightfield with a vertical scratch, and cell masks."""
    rng = np.random.default_rng(0)
    open_mask = np.zeros((256, 256), dtype=bool)
    open_mask[:, 128 - width // 2:128 + width // 2] = True
    texture = ndi.gaussian_filter(rng.standard_normal(open_mask.shape), 1.5)
    texture /= texture.std()
    bright = 1000 + rng.normal(0, 8, open_mask.shape) + (
        ~open_mask) * texture * 120
    cells, _count = ndi.label(~open_mask)
    data = np.zeros((256, 256, 7), np.uint16)
    data[..., 0] = np.clip(bright, 0, 65535)
    data[..., 4] = cells
    path = directory / "plate1_A01_1_0.npy"
    np.save(path, data)
    return str(path), open_mask


@pytest.fixture
def alpha_on(monkeypatch):
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)


@pytest.fixture
def panel(qtbot, alpha_on):
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_overlay_shows_the_open_wound_and_its_width(panel, tmp_path):
    path, open_mask = _field(tmp_path)
    assert panel.load_array(path)
    panel.apply_settings({"wound_source": "texture", "wound_channel": 0})
    panel._wound_btn.setChecked(True)
    assert not panel._wound_view.isHidden()
    assert not panel._wound_view.pixmap().isNull()
    text = panel._status.text()
    assert text.startswith("Wound ") and "px" in text
    percent = float(text.split("Wound ")[1].split(" %")[0])
    assert abs(percent / 100.0 - open_mask.mean()) <= 0.02
    mean = float(text.split("mean width ")[1].split(" ")[0])
    assert abs(mean - 90) <= 5

    panel._wound_btn.setChecked(False)
    assert panel._wound_view.isHidden()


def test_the_masks_source_reads_the_panels_cell_slice(panel, tmp_path):
    path, open_mask = _field(tmp_path)
    panel.load_array(path)
    panel.apply_settings({"wound_source": "masks"})
    settings = panel._wound_preview_settings()
    assert settings["cell_mask_dim"] == 4
    result = MP._compute_wound_preview(np.load(path), settings)
    assert result["status"] == "ok" and not result["error"]
    assert result["open_fraction"] == pytest.approx(open_mask.mean())


def test_a_failure_is_reported_not_raised(tmp_path):
    path, _open = _field(tmp_path)
    result = MP._compute_wound_preview(
        np.load(path), {"channels": [0], "cell_mask_dim": None,
                        "wound_source": "masks"})
    assert result["overlay"] is None and "cell_mask_dim" in result["error"]


def test_the_button_is_hidden_when_alpha_features_are(qtbot, monkeypatch):
    from spacr.settings import ALPHA_FEATURES

    assert "MeasureWoundToggle" in ALPHA_FEATURES[536]["widgets"]
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert widget._wound_btn.objectName() == "MeasureWoundToggle"
    assert not widget._wound_btn.isHidden()
    widget._wound_btn.setChecked(True)

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    widget._refresh_alpha_visibility()
    assert widget._wound_btn.isHidden()
    assert not widget._wound_btn.isChecked()


def test_the_wound_settings_follow_the_alpha_switch(monkeypatch):
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[536]["settings"]
    assert "wound_closure" in keys and "wound_conditions" in keys
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    assert not any(preferences._is_alpha_visible("settings", key)
                   for key in keys)
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)
    assert all(preferences._is_alpha_visible("settings", key)
               for key in keys)


def test_the_measure_form_hides_the_wound_settings_until_alpha_is_on(
        qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    from spacr.settings import ALPHA_FEATURES

    ini = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(ini), QSettings.IniFormat))
    keys = ALPHA_FEATURES[536]["settings"]
    screen = AppScreen("measure")
    try:
        for key in keys:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in keys)
        assert screen._settings_model.set_value_for_key("wound_closure", True)
        assert screen._settings_model.collect()["wound_closure"] is True
        assert screen._settings_model.set_value_for_key("wound_threshold",
                                                        0.05)
        assert screen._settings_model.collect()["wound_threshold"] == 0.05

        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in keys)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
