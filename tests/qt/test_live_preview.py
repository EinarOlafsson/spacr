"""Tests for the live-preview segmentation widget (v2 — multi-object).

Cellpose is heavy + optional; the segmentation function is stubbed via
monkeypatch. Everything else is exercised for real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.qt.widgets import live_preview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tif(tmp_path: Path) -> Path:
    arr = (np.random.default_rng(0).integers(0, 4000, size=(64, 64))
           ).astype(np.uint16)
    p = tmp_path / "sample.tif"
    tifffile.imwrite(str(p), arr)
    return p


@pytest.fixture
def sample_tif_multichannel(tmp_path: Path) -> Path:
    arr = (np.random.default_rng(1).integers(
        0, 4000, size=(64, 64, 3))).astype(np.uint16)
    p = tmp_path / "sample_rgb.tif"
    tifffile.imwrite(str(p), arr)
    return p


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_load_preview_image_reads_tif(self, sample_tif):
        arr = live_preview.load_preview_image(sample_tif)
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint16

    def test_load_preview_image_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            live_preview.load_preview_image(tmp_path / "nope.tif")

    def test_overlay_masks_draws_green_boundary_for_cell(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[10:20, 10:20] = 1
        out = live_preview.overlay_masks(img, {"cell": mask})
        assert out.shape == (32, 32, 3)
        top = out[10, 15]
        assert top[1] > top[0] and top[1] > top[2]

    def test_overlay_masks_uses_distinct_colours_per_object(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        cell = np.zeros((32, 32), dtype=np.uint16); cell[5:10, 5:10] = 1
        nuc  = np.zeros((32, 32), dtype=np.uint16); nuc[20:25, 20:25] = 1
        out = live_preview.overlay_masks(
            img, {"cell": cell, "nucleus": nuc})
        # Cell boundary should be green-dominant, nucleus boundary
        # should be magenta-dominant (R+B > G).
        cell_edge = out[5, 7]
        nuc_edge = out[20, 22]
        assert cell_edge[1] > cell_edge[0]
        assert int(nuc_edge[0]) + int(nuc_edge[2]) > int(nuc_edge[1])

    def test_outline_thickness_dilates_boundary(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[10:20, 10:20] = 1
        thin = live_preview.overlay_masks(
            img, {"cell": mask}, outline_thickness=1)
        thick = live_preview.overlay_masks(
            img, {"cell": mask}, outline_thickness=4)
        # Thick outline colours more pixels than thin
        thin_green = int((thin[..., 1] > thin[..., 0]).sum())
        thick_green = int((thick[..., 1] > thick[..., 0]).sum())
        assert thick_green > thin_green

    def test_normalise_toggle_actually_stretches(self):
        arr = np.zeros((16, 16), dtype=np.uint16); arr[8:] = 100
        norm = live_preview._to_uint8(arr, normalise=True)
        raw = live_preview._to_uint8(arr, normalise=False)
        # Normalised image should span more than raw does (raw scales
        # 0→0 and 100→100).
        assert int(norm.max() - norm.min()) >= int(raw.max() - raw.min())


# ---------------------------------------------------------------------------
# Size filter
# ---------------------------------------------------------------------------

class TestSizeFilter:
    def test_min_size_drops_small(self):
        # Filtering relabels sequentially (matching the pipeline), so assert
        # by surviving area/count rather than by original label value.
        mask = np.zeros((16, 16), dtype=np.int32)
        mask[0, 0] = 1                   # 1 px
        mask[5:10, 5:10] = 2             # 25 px
        out = live_preview._apply_size_filter(
            mask, {"cell_min_size": 5}, "cell")
        assert (out > 0).sum() == 25     # only the 25 px object remains
        assert len(np.unique(out)) - 1 == 1

    def test_max_size_drops_big(self):
        mask = np.zeros((16, 16), dtype=np.int32)
        mask[5:10, 5:10] = 1             # 25 px
        mask[0, 15] = 2                  # 1 px
        out = live_preview._apply_size_filter(
            mask, {"cell_max_size": 10}, "cell")
        assert (out > 0).sum() == 1      # 25 > 10 dropped, 1 px survives
        assert len(np.unique(out)) - 1 == 1

    def test_no_settings_is_identity(self):
        mask = np.zeros((16, 16), dtype=np.int32); mask[5:10, 5:10] = 1
        out = live_preview._apply_size_filter(mask, {}, "cell")
        assert np.array_equal(out, mask)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class TestPanel:
    def test_panel_constructs(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        params = panel.current_params()
        expected = {"model", "diameter", "flow_threshold", "cellprob",
                     "object_types", "cell_channel", "nucleus_channel",
                     "normalise", "lo_pct", "hi_pct",
                     "outline_thickness", "outline_colour"}
        assert expected.issubset(params.keys())

    def test_settings_widgets_hidden_in_compact_layout(self, qtbot):
        """Every option lives behind the Live Settings dialog now —
        the compact layout only shows the file picker, Run button,
        Live Settings button, hover label, and canvases."""
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.show()
        # Model / object / channels / diameter / flow / prob /
        # normalise / percentiles / outline / pre / post are all hidden
        for name in ("_model_box", "_object_box", "_cell_channel",
                       "_nucleus_channel", "_diameter", "_flow",
                       "_prob", "_normalise_check", "_lo_pct",
                       "_hi_pct", "_outline_colour",
                       "_outline_thickness"):
            w = getattr(panel, name)
            assert not w.isVisible(), (
                f"{name} should be hidden in the compact layout")

    def test_open_live_settings_shows_dialog(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.open_live_settings()
        dlg = panel._live_settings_dialog
        assert dlg is not None
        assert dlg.isVisible()
        # The dialog reveals the widgets while open
        assert panel._model_box.isVisible()
        dlg.close()
        # Widgets go hidden again once the dialog closes
        assert not panel._model_box.isVisible()

    def test_default_model_is_cpsam(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        assert panel.current_params()["model"] == "cpsam"

    def test_load_image_updates_panel(self, qtbot, sample_tif):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        assert panel.load_image(sample_tif) is True
        assert panel._image is not None
        assert panel._path_label.text() == str(sample_tif)

    def test_load_image_bad_path_is_silent(self, qtbot, tmp_path):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        assert panel.load_image(tmp_path / "missing.tif") is False
        assert "failed" in panel._status.text().lower()

    def test_apply_settings_copies_values(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.apply_settings({
            "diameter": 42.0, "flow_threshold": 0.6, "CP_prob": -1.5,
            "cell_channel": 2, "nucleus_channel": 3, "model_name": "cyto2",
        })
        params = panel.current_params()
        assert params["diameter"] == pytest.approx(42.0)
        assert params["flow_threshold"] == pytest.approx(0.6)
        assert params["cellprob"] == pytest.approx(-1.5)
        assert params["cell_channel"] == 2
        assert params["nucleus_channel"] == 3
        assert params["model"] == "cyto2"

    def test_run_preview_without_image_is_no_op(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.run_preview()
        assert "load" in panel._status.text().lower()

    def test_run_preview_stubbed_segment_single_object(
            self, qtbot, monkeypatch, sample_tif):
        def _stub(req):
            m = np.zeros(req.image.shape[:2], dtype=np.int32)
            m[8:24, 8:24] = 1
            return {"cell": m}
        monkeypatch.setattr(live_preview, "_segment_multi", _stub)
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.load_image(sample_tif)
        received = []
        panel.preview_ready.connect(lambda m: received.append(m))
        panel.run_preview()
        qtbot.waitUntil(lambda: len(received) == 1, timeout=3000)
        assert received[0] is not None
        assert "cell" in received[0]

    def test_run_preview_stubbed_multi_object(
            self, qtbot, monkeypatch, sample_tif):
        def _stub(req):
            assert req.object_types == ("cell", "nucleus")
            out = {}
            for obj in req.object_types:
                m = np.zeros(req.image.shape[:2], dtype=np.int32)
                m[0:5, 0:5] = 1
                out[obj] = m
            return out
        monkeypatch.setattr(live_preview, "_segment_multi", _stub)
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel._object_box.setCurrentIndex(
            panel._object_box.findText("cell + nucleus"))
        panel.load_image(sample_tif)
        received = []
        panel.preview_ready.connect(lambda m: received.append(m))
        panel.run_preview()
        qtbot.waitUntil(lambda: len(received) == 1, timeout=3000)
        assert set(received[0].keys()) == {"cell", "nucleus"}


# ---------------------------------------------------------------------------
# Model-aware option hiding
# ---------------------------------------------------------------------------

class TestModelAwareOptions:
    def test_cpsam_disables_only_diameter_in_dialog(self, qtbot):
        # Cellpose-SAM auto-estimates object size, so only the diameter is
        # ignored; flow threshold + cell probability remain in use.
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel._model_box.setCurrentIndex(panel._model_box.findText("cpsam"))
        panel.open_live_settings()
        try:
            assert not panel._diameter.isEnabled(), (
                "diameter should be disabled for cpsam")
            assert panel._flow.isEnabled(), (
                "flow threshold should stay enabled for cpsam")
            assert panel._prob.isEnabled(), (
                "cell probability should stay enabled for cpsam")
        finally:
            panel._live_settings_dialog.close()

    def test_legacy_model_enables_all_options(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel._model_box.setCurrentIndex(panel._model_box.findText("cyto2"))
        panel.open_live_settings()
        try:
            for w in (panel._diameter, panel._flow, panel._prob):
                assert w.isEnabled()
        finally:
            panel._live_settings_dialog.close()


# ---------------------------------------------------------------------------
# Pre / Post toggles pipe settings through the request
# ---------------------------------------------------------------------------

class TestSettingsFlowIntoRequest:
    """The common + per-compartment controls always flow into the request —
    there are no Pre/Post checkboxes anymore."""

    def test_remove_background_flows_from_common_control(
            self, qtbot, monkeypatch, sample_tif):
        captured = {}
        def _stub(req):
            captured["pre"] = dict(req.preprocess_settings)
            return {"cell": np.zeros(req.image.shape[:2], dtype=np.int32)}
        monkeypatch.setattr(live_preview, "_segment_multi", _stub)
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel._object_box.setCurrentText("cell")
        panel._common_widgets["remove_background"].setChecked(True)
        panel._common_widgets["background"].setValue(50)
        panel.load_image(sample_tif)
        panel.run_preview()
        qtbot.waitUntil(lambda: "pre" in captured, timeout=3000)
        assert captured["pre"].get("remove_background_cell") is True
        assert captured["pre"].get("cell_background") == 50

    def test_filter_flows_from_compartment_control(
            self, qtbot, monkeypatch, sample_tif):
        captured = {}
        def _stub(req):
            captured["post"] = dict(req.postprocess_settings)
            return {"cell": np.zeros(req.image.shape[:2], dtype=np.int32)}
        monkeypatch.setattr(live_preview, "_segment_multi", _stub)
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel._object_box.setCurrentText("cell")
        panel._compartment_widgets["cell"]["min_area"].setValue(50)
        panel.load_image(sample_tif)
        panel.run_preview()
        qtbot.waitUntil(lambda: "post" in captured, timeout=3000)
        assert captured["post"].get("cell_min_area") == 50


# ---------------------------------------------------------------------------
# Hover pixel info
# ---------------------------------------------------------------------------

class TestHover:
    def test_hover_out_of_range_shows_default_hint(self, qtbot, sample_tif):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.load_image(sample_tif)
        panel._on_hover(-1, -1)
        assert "hover" in panel._hover_label.text().lower()

    def test_hover_in_range_shows_intensity(self, qtbot, sample_tif):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.load_image(sample_tif)
        panel._on_hover(10, 10)
        assert "intensity=" in panel._hover_label.text()
        # Should also include coordinates
        assert "x=" in panel._hover_label.text()

    def test_hover_with_mask_reports_object_label(self, qtbot, sample_tif):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.load_image(sample_tif)
        mask = np.zeros((64, 64), dtype=np.int32); mask[5:15, 5:15] = 7
        panel._masks = {"cell": mask}
        panel._on_hover(10, 10)
        assert "cell=#7" in panel._hover_label.text()


# ---------------------------------------------------------------------------
# Integration into AppScreen
# ---------------------------------------------------------------------------

class TestAppScreenIntegration:
    def test_mask_screen_has_live_preview(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert getattr(scr, "_live_preview", None) is not None

    def test_mask_screen_uses_splitter_between_lp_and_console(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert getattr(scr, "_runtime_splitter", None) is not None

    # measure now has its own crop preview (a splitter) — covered separately in
    # test_measure_preview.py — so it's excluded here.
    @pytest.mark.parametrize("app_key", ["classify", "umap"])
    def test_other_screens_omit_live_preview(self, qtbot, app_key):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen(app_key)
        qtbot.addWidget(scr)
        assert getattr(scr, "_live_preview", None) is None
        assert getattr(scr, "_runtime_splitter", None) is None

    def test_autoload_from_folder(self, qtbot, tmp_path, sample_tif):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._autoload_live_preview(str(tmp_path))
        assert scr._live_preview._image is not None


class TestCompartmentSettings:
    """Per-compartment tuning panels in the Live settings dialog."""

    def _panel(self, qtbot):
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        p = LivePreviewPanel()
        qtbot.addWidget(p)
        return p

    def test_all_compartments_have_widgets(self, qtbot):
        from spacr.qt.widgets.live_preview import COMPARTMENTS, COMPARTMENT_FIELDS
        p = self._panel(qtbot)
        for comp in COMPARTMENTS:
            for suffix, *_ in COMPARTMENT_FIELDS:
                assert suffix in p._compartment_widgets[comp]

    def test_dialog_shows_only_primary_object_panel(self, qtbot):
        from spacr.qt.widgets.live_preview import LiveSettingsDialog
        p = self._panel(qtbot)
        dlg = LiveSettingsDialog(p)
        qtbot.addWidget(dlg); dlg.show()
        # object "cell": only the Cell panel is shown.
        p._object_box.setCurrentText("cell")
        dlg.refresh_visibility()
        assert dlg._compartment_groupboxes["cell"].isVisibleTo(dlg)
        assert not dlg._compartment_groupboxes["pathogen"].isVisibleTo(dlg)
        assert not dlg._compartment_groupboxes["organelle"].isVisibleTo(dlg)
        assert not dlg._compartment_groupboxes["nucleus"].isVisibleTo(dlg)
        # Switch to organelle: only Organelle shown.
        p._object_box.setCurrentText("organelle")
        dlg.refresh_visibility()
        assert dlg._compartment_groupboxes["organelle"].isVisibleTo(dlg)
        assert not dlg._compartment_groupboxes["cell"].isVisibleTo(dlg)

    def test_cell_plus_nucleus_shows_primary_and_secondary(self, qtbot):
        from spacr.qt.widgets.live_preview import LiveSettingsDialog
        p = self._panel(qtbot)
        dlg = LiveSettingsDialog(p)
        qtbot.addWidget(dlg); dlg.show()
        p._object_box.setCurrentText("cell + nucleus")
        dlg.refresh_visibility()
        assert dlg._compartment_groupboxes["cell"].isVisibleTo(dlg)
        assert dlg._compartment_groupboxes["nucleus"].isVisibleTo(dlg)
        assert "secondary" in dlg._compartment_groupboxes["nucleus"].title().lower()
        assert not dlg._compartment_groupboxes["pathogen"].isVisibleTo(dlg)

    def test_common_controls_target_chosen_object(self, qtbot):
        p = self._panel(qtbot)
        p._common_widgets["signal_to_noise"].setValue(42)
        p._common_widgets["background"].setValue(77)
        p._common_widgets["remove_background"].setChecked(True)
        # Object = pathogen -> common keys are pathogen_*.
        p._object_box.setCurrentIndex(p._object_box.findText("pathogen"))
        s = p.settings_for_propagation()
        assert s["pathogen_Signal_to_noise"] == 42
        assert s["pathogen_background"] == 77
        assert s["remove_background_pathogen"] is True

    def test_compartment_values_propagate(self, qtbot):
        p = self._panel(qtbot)
        p._compartment_widgets["organelle"]["min_area"].setValue(555)
        p._compartment_widgets["organelle"]["intensity_threshold_method"].setCurrentText("percentile")
        s = p.settings_for_propagation()
        assert s["organelle_min_area"] == 555
        assert s["organelle_intensity_threshold_method"] == "percentile"


class TestViewModes:
    """Right-canvas view modes: Overlay / Masks / Flows."""

    def _panel_with_image(self, qtbot):
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        p = LivePreviewPanel()
        qtbot.addWidget(p)
        p._image = np.random.RandomState(0).randint(
            0, 255, (32, 32), dtype=np.uint16)
        return p

    def test_view_mode_options(self, qtbot):
        p = self._panel_with_image(qtbot)
        opts = [p._view_mode.itemText(i) for i in range(p._view_mode.count())]
        assert opts == ["Overlay", "Masks", "Flows"]

    def test_masks_mode_renders_label_rgb(self, qtbot):
        p = self._panel_with_image(qtbot)
        mask = np.zeros((32, 32), np.int32)
        mask[4:10, 4:10] = 1
        mask[20:26, 20:26] = 2
        p._masks = {"cell": mask}
        rgb = p._label_rgb()
        assert rgb.shape == (32, 32, 3)
        assert rgb.max() > 0                       # objects coloured
        assert (rgb[0, 0] == 0).all()              # background stays black

    def test_flows_ready_stores_and_flows_rgb(self, qtbot):
        p = self._panel_with_image(qtbot)
        flow = np.random.RandomState(1).randint(
            0, 255, (32, 32, 3), dtype=np.uint8)
        p._on_flows_ready({"cell": flow})
        assert "cell" in p._flows
        assert np.array_equal(p._flows_rgb(), flow)

    def test_switch_modes_no_crash(self, qtbot):
        p = self._panel_with_image(qtbot)
        mask = np.zeros((32, 32), np.int32); mask[4:10, 4:10] = 1
        p._masks = {"cell": mask}
        p._flows = {"cell": np.zeros((32, 32, 3), np.uint8)}
        for m in ("Overlay", "Masks", "Flows"):
            p._view_mode.setCurrentText(m)
            p._refresh_canvases()   # must not raise
