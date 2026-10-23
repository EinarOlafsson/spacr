"""Organelle detector controls preserve the run's settings and real masks."""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QFormLayout

from spacr.qt.widgets import live_preview as lp

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = lp.LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget.apply_settings({"number_of_organelles": 2,
                           "organelle_channel": 0,
                           "organelleb_channel": 0,
                           "organelle_morphology": "network",
                           "organelle_method": "ridge",
                           "organelle_ridge_sigmas": [1, 3, 5],
                           "organelleb_morphology": "spots",
                           "organelleb_method": "adaptive",
                           "organelleb_adaptive_offset": 7})
    widget._object_box.setCurrentText("organelle")
    return widget


def test_column_follows_detector_and_restores_each_slot(panel, qtbot):
    dialog = lp.LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    dialog.show()
    controls = panel._organelle_widgets
    form = dialog._organelle_group.layout()
    assert dialog._organelle_group.isVisible()
    assert not form.isRowVisible(controls["ring_fill_method"])
    assert form.isRowVisible(controls["ridge_sigmas"])
    controls["ridge_sigmas"].setText("2, 4, 6")
    panel._object_box.setCurrentText("organelle 2")
    assert panel._organelle_morphology() == "spots"
    assert lp._combo_value(controls["method"]) == "adaptive"
    assert controls["adaptive_offset"].value() == 7
    assert "Organelle 2" in form.labelForField(controls["method"]).text()
    assert controls["method"].property("settingKey") == "organelleb_method"
    assert form.isRowVisible(controls["adaptive_offset"])
    assert not form.isRowVisible(controls["ridge_sigmas"])
    controls["adaptive_offset"].setValue(12)
    panel._object_box.setCurrentText("organelle")
    assert panel._widget_value(controls["ridge_sigmas"]) == [2, 4, 6]
    assert lp._combo_value(controls["method"]) == "ridge"
    assert "Organelle 1" in form.labelForField(controls["method"]).text()
    panel._object_box.setCurrentText("cell")
    assert not dialog._organelle_group.isVisible()
    panel._object_box.setCurrentText("organelle 2")
    assert controls["adaptive_offset"].value() == 12
    assert dialog.width() <= dialog.screen().availableGeometry().width()
    dialog.close()


def test_organelle_edits_propagate_and_build_real_worker_request(panel, qtbot):
    dialog = lp.LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    assert all(widget in dialog._propagate_sources
               for widget in panel._organelle_widgets.values())
    controls = panel._organelle_widgets
    restriction = controls["mask_within_cells"]
    label = dialog._organelle_group.layout().labelForField(restriction)
    assert "Preview cell masks are required" in label.toolTip()
    assert 'href="' in label.toolTip() and "API" in label.toolTip()
    assert "cell_mask_stack" not in label.toolTip()
    assert restriction.property("apiTooltipDescriptionSource") == (
        "Preview cell masks are required to restrict organelles to cells.")
    controls["rolling_ball"].setChecked(True)
    controls["rolling_ball_radius"].setValue(6)
    controls["clahe"].setChecked(True)
    controls["clahe_clip_limit"].setValue(.02)
    controls["ridge_sigmas"].setText("2, 5")
    panel._image = np.ones((32, 32, 1), np.float32)
    request = panel._build_request()
    propagated = panel.settings_for_propagation()
    for suffix, expected in (("rolling_ball", True), ("rolling_ball_radius", 6),
                             ("clahe", True), ("clahe_clip_limit", .02),
                             ("ridge_sigmas", [2, 5])):
        assert request.preprocess_settings[f"organelle_{suffix}"] == expected
        assert propagated[f"organelle_{suffix}"] == expected
    dialog.close()


@pytest.mark.parametrize("morphology,method", [("spots", "adaptive"),
                                               ("network", "ridge"),
                                               ("irregular", "otsu"),
                                               ("ring", "otsu")])
def test_preview_matches_real_classical_pipeline(morphology, method):
    from spacr.object import (_extract_classical_settings, _segment_single_image,
                              _preprocess_batch)
    from spacr.settings import _set_organelle_defaults

    y, x = np.mgrid[:48, :48]
    image = (100 * np.exp(-((x - 20) ** 2 + (y - 21) ** 2) / 30)
             + x * .2).astype(np.float32)
    settings = _set_organelle_defaults({"organelle_morphology": morphology,
                                       "organelle_method": method,
                                       "organelle_rolling_ball": True,
                                       "organelle_rolling_ball_radius": 5,
                                       "organelle_clahe": True,
                                       "organelle_clahe_clip_limit": .02,
                                       "organelle_adaptive_block_size": 11,
                                       "organelle_adaptive_offset": .01,
                                       "organelle_ridge_sigmas": [1, 2],
                                       "organelle_min_area": 2})
    expected = _segment_single_image(
        _preprocess_batch(image[None], settings)[0],
        _extract_classical_settings(settings))
    request = lp.PreviewRequest(image[..., None], object_types=("organelle",),
                               channels={"organelle": 0},
                               preprocess_settings=settings)
    actual = lp._segment_multi(request)[0]["organelle"]
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual)


def test_cell_restriction_and_slot_settings_reach_segmentation():
    from spacr.object import _extract_classical_settings, _segment_single_image
    from spacr.settings import _set_organelle_defaults

    image = np.zeros((40, 40), np.float32)
    image[8:16, 8:16] = 100
    image[25:33, 25:33] = 100
    cells = np.zeros_like(image, dtype=np.int32)
    cells[:20] = 1
    settings = _set_organelle_defaults({"organelle_morphology": "irregular",
                                       "organelle_method": "otsu",
                                       "organelle_morph_radius": 1,
                                       "organelle_min_area": 2})
    settings.update({"organelleb_mask_within_cells": True,
                     "organelleb_morphology": "irregular",
                     "organelleb_method": "otsu"})
    request = lp.PreviewRequest(image[..., None], object_types=("organelleb",),
                               channels={"organelleb": 0},
                               preprocess_settings=settings, cell_mask=cells)
    actual = lp._segment_multi(request)[0]["organelleb"]
    expected_image = image.copy()
    expected_image[cells == 0] = 0
    expected = _segment_single_image(expected_image, _extract_classical_settings(settings))
    np.testing.assert_array_equal(actual, expected)
    assert actual[10, 10] > 0
    assert actual[28, 28] == 0


def test_unet_controls_use_the_pipeline_loader_and_predictor(panel, qtbot, monkeypatch):
    from spacr import object as pipeline

    dialog = lp.LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    controls = panel._organelle_widgets
    controls["method"].setCurrentText("unet")
    controls["unet_model_path"].setText("/chosen/model.pt")
    controls["unet_threshold"].setValue(.7)
    assert dialog._organelle_group.layout().isRowVisible(controls["unet_model_path"])
    assert not dialog._organelle_group.layout().isRowVisible(controls["ridge_sigmas"])
    sentinel = object()
    seen = {}

    def load(settings):
        assert settings["organelle_unet_model_path"] == "/chosen/model.pt"
        return sentinel

    def predict(batch, model, settings):
        assert model is sentinel
        seen["threshold"] = settings["organelle_unet_threshold"]
        return [(batch[0] > 1).astype(np.int32)]

    monkeypatch.setattr(pipeline, "_load_unet_model", load)
    monkeypatch.setattr(pipeline, "_segment_unet", predict)
    panel._image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    result, _ = lp._segment_multi(panel._build_request())
    np.testing.assert_array_equal(result["organelle"], (panel._image[..., 0] > 1))
    assert seen["threshold"] == .7
    dialog.close()


def test_invalid_ridge_input_is_reported_by_worker_without_gui_failure(panel):
    panel._organelle_widgets["ridge_sigmas"].setText("not scales")
    settings = panel.settings_for_propagation()
    assert settings["organelle_ridge_sigmas"] == "not scales"
    with pytest.raises(ValueError, match="positive numbers"):
        lp._classical_organelle_mask(np.ones((8, 8)), "organelle", settings)


def test_detector_threshold_changes_real_preview_masks(panel):
    controls = panel._organelle_widgets
    controls["morphology"].setCurrentText("spots")
    controls["method"].setCurrentText("log")
    controls["log_min_sigma"].setValue(1.25)
    controls["log_max_sigma"].setValue(3.5)
    controls["watershed_spots"].setChecked(False)
    y, x = np.mgrid[:48, :48]
    panel._image = (np.exp(-((x - 24) ** 2 + (y - 24) ** 2) / 8)
                     * 200).astype(np.float32)[..., None]
    controls["log_threshold"].setValue(.01)
    low = panel._build_request()
    low_masks, _ = lp._segment_multi(low)
    assert low.preprocess_settings["organelle_log_min_sigma"] == 1.25
    assert np.any(low_masks["organelle"])
    controls["log_threshold"].setValue(2)
    high_masks, _ = lp._segment_multi(panel._build_request())
    assert not np.any(high_masks["organelle"])


def test_small_dialog_keeps_organelle_column_scrollable(panel, qtbot):
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual

    dialog = lp.LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    dialog.show()
    _resize_actual(dialog, QSize(480, 360))
    qtbot.waitUntil(lambda: dialog._columns_scroll.horizontalScrollBar().maximum() > 0)
    form = dialog._organelle_group.layout()
    widget = panel._organelle_widgets["ridge_sigmas"]
    assert form.isRowVisible(widget)
    dialog._columns_scroll.ensureWidgetVisible(widget)
    qtbot.waitUntil(lambda: dialog._columns_scroll.horizontalScrollBar().value() > 0)
    assert widget.isVisible()
    assert dialog.width() <= dialog.screen().availableGeometry().width()
    dialog.close()


@pytest.mark.parametrize("cell_mask", [None, np.ones((3, 3), np.int32)])
def test_cell_restriction_refuses_missing_or_different_field_masks(cell_mask):
    request = lp.PreviewRequest(
        np.ones((8, 8, 1), np.float32), object_types=("organelle",),
        preprocess_settings={"organelle_method": "otsu",
                             "organelle_mask_within_cells": True},
        cell_mask=cell_mask)
    with pytest.raises(ValueError, match="Preview cell masks are required"):
        lp._segment_multi(request)


@pytest.mark.parametrize("missing_name", ["spacr.qt.i18n_catalogs", "spacr.qt.i18n_catalogs.en"])
def test_column_keeps_readable_captions_and_slot_values_without_catalogs(
        panel, qtbot, monkeypatch, missing_name):
    import builtins
    from spacr.qt.i18n import tr

    original_import = builtins.__import__

    def without_catalogs(name, *args, **kwargs):
        if name == "i18n_catalogs" or name.startswith("spacr.qt.i18n_catalogs"):
            raise ModuleNotFoundError("catalog intentionally absent", name=missing_name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_catalogs)
    dialog = lp.LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    dialog.show()
    controls = panel._organelle_widgets
    form = dialog._organelle_group.layout()
    assert form.labelForField(controls["method"]).text() == tr("Method")
    controls["ridge_sigmas"].setText("2, 4, 6")
    panel._object_box.setCurrentText("organelle 2")
    assert form.labelForField(controls["method"]).text() == tr("Method")
    assert controls["method"].property("settingKey") == "organelleb_method"
    assert controls["adaptive_offset"].value() == 7
    controls["adaptive_offset"].setValue(12)
    panel._object_box.setCurrentText("organelle")
    assert panel._widget_value(controls["ridge_sigmas"]) == [2, 4, 6]
    panel._object_box.setCurrentText("organelle 2")
    assert controls["adaptive_offset"].value() == 12
    dialog.close()


def test_column_does_not_hide_an_unrelated_missing_dependency(panel, qtbot, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def broken_catalog(name, *args, **kwargs):
        if name == "i18n_catalogs":
            raise ModuleNotFoundError("unrelated dependency absent", name="unrelated_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_catalog)
    with pytest.raises(ModuleNotFoundError, match="unrelated dependency absent"):
        lp.LiveSettingsDialog(panel)
