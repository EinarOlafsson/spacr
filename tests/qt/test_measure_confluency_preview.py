"""Item 541: the Measure preview draws the covered area, as an Alpha control.

The Confluency button runs the same function a Measure run with
``confluency`` on runs for one field, so the overlay and the database agree,
and it is present only when Preferences -> Show alpha features is on (item
569's registry, where item 541 registers it).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from spacr.qt import preferences
from spacr.qt.widgets import measure_preview as MP


def _field(directory, fraction=0.4):
    """A merged array: a fluorescent stain and the cell masks under it."""
    rng = np.random.default_rng(0)
    smooth = ndi.gaussian_filter(rng.standard_normal((256, 256)), 16)
    covered = smooth > np.quantile(smooth, 1.0 - fraction)
    stain = 100 + rng.normal(0, 5, covered.shape) + covered * 800.0
    cells, _count = ndi.label(covered)
    data = np.zeros((256, 256, 7), np.uint16)
    data[..., 0] = np.clip(stain, 0, 65535)
    data[..., 4] = cells
    path = directory / "plate1_A01_1.npy"
    np.save(path, data)
    return str(path), covered


@pytest.fixture
def alpha_on(monkeypatch):
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)


@pytest.fixture
def panel(qtbot, alpha_on):
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_overlay_shows_the_covered_fraction_of_the_loaded_field(
        panel, tmp_path):
    path, covered = _field(tmp_path)
    assert panel.load_array(path)
    panel.apply_settings({"confluency_source": "intensity",
                          "confluency_channel": 0,
                          "confluency_qc_threshold": 0.9})
    panel._confluency_btn.setChecked(True)
    assert not panel._confluency_view.isHidden()
    assert not panel._confluency_view.pixmap().isNull()
    text = panel._status.text()
    assert "(intensity)" in text and "below the monolayer QC" in text
    percent = float(text.split("Confluency ")[1].split(" %")[0])
    assert abs(percent / 100.0 - covered.mean()) <= 0.02

    panel._confluency_btn.setChecked(False)
    assert panel._confluency_view.isHidden()


def test_auto_reads_the_panels_cell_mask_slice(panel, tmp_path):
    path, covered = _field(tmp_path)
    panel.load_array(path)
    settings = panel.confluency_settings()
    assert settings["cell_mask_dim"] == 4
    result = MP.compute_confluency_preview(np.load(path), settings)
    assert result["source"] == "masks" and not result["error"]
    assert result["confluency"] == pytest.approx(covered.mean())


def test_a_failure_is_reported_not_raised(tmp_path):
    path, _covered = _field(tmp_path)
    result = MP.compute_confluency_preview(
        np.load(path), {"channels": [0], "cell_mask_dim": None,
                        "confluency_source": "masks"})
    assert result["overlay"] is None and "cell_mask_dim" in result["error"]


def test_the_button_is_hidden_when_alpha_features_are(qtbot, monkeypatch):
    from spacr.settings import ALPHA_FEATURES

    assert "MeasureConfluencyToggle" in ALPHA_FEATURES[541]["widgets"]
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert widget._confluency_btn.objectName() == "MeasureConfluencyToggle"
    assert not widget._confluency_btn.isHidden()
    widget._confluency_btn.setChecked(True)

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    widget.refresh_alpha_visibility()
    assert widget._confluency_btn.isHidden()
    assert not widget._confluency_btn.isChecked()
