"""Tests for the live-preview segmentation widget.

Cellpose is heavy + optional so we can't rely on it being importable
in CI. What we CAN test without touching Cellpose:

* The pure image helpers (``load_preview_image``, ``overlay_mask``,
  ``numpy_to_qpixmap``).
* The Qt panel wires up and displays a loaded image.
* The Mask :class:`AppScreen` embeds the panel; other apps don't.
* When ``src`` gets a folder, the panel auto-loads a tile.
* The worker path is exercised via a monkey-patched
  ``_segment_once`` — the actual Cellpose call is stubbed to a
  small dummy mask.
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
    """Write a tiny 2-D uint16 TIFF and return the path."""
    arr = (np.random.default_rng(0).integers(0, 4000,
                                              size=(64, 64))
           ).astype(np.uint16)
    p = tmp_path / "sample.tif"
    tifffile.imwrite(str(p), arr)
    return p


@pytest.fixture
def sample_folder(tmp_path: Path, sample_tif: Path) -> Path:
    """Return the tmp_path so autoload's folder-scan test can use it."""
    return tmp_path


# ---------------------------------------------------------------------------
# Pure helpers — no Qt / no Cellpose
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_load_preview_image_reads_tif(self, sample_tif):
        arr = live_preview.load_preview_image(sample_tif)
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint16

    def test_load_preview_image_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            live_preview.load_preview_image(tmp_path / "nope.tif")

    def test_overlay_mask_draws_boundary(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[10:20, 10:20] = 1
        out = live_preview.overlay_mask(img, mask)
        assert out.shape == (32, 32, 3)
        # The green channel should be brighter than red on the boundary
        # (the boundary is drawn (32,220,32)).
        top = out[10, 15]     # top border of the box
        assert top[1] > top[0] and top[1] > top[2]

    def test_overlay_empty_mask_returns_rgb(self):
        img = (np.random.default_rng(0).integers(0, 255, (16, 16))
               ).astype(np.uint8)
        out = live_preview.overlay_mask(
            img, np.zeros_like(img, dtype=np.uint16))
        assert out.shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# Qt widget — construction + load + settings apply
# ---------------------------------------------------------------------------

class TestPanel:
    def test_panel_constructs(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        assert panel is not None
        # Default parameter snapshot has all the expected keys
        params = panel.current_params()
        assert set(params) == {"model", "diameter",
                                 "flow_threshold", "cellprob", "channel"}

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
            "diameter": 42.0,
            "flow_threshold": 0.6,
            "CP_prob": -1.5,
            "cell_channel": 2,
            "model_name": "cyto2",
        })
        params = panel.current_params()
        assert params["diameter"] == pytest.approx(42.0)
        assert params["flow_threshold"] == pytest.approx(0.6)
        assert params["cellprob"] == pytest.approx(-1.5)
        assert params["channel"] == 2
        assert params["model"] == "cyto2"

    def test_run_preview_without_image_is_no_op(self, qtbot):
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.run_preview()
        assert "load" in panel._status.text().lower()

    def test_run_preview_stubbed_segment(self, qtbot, monkeypatch,
                                            sample_tif):
        """Replace ``_segment_once`` with a stub so we can exercise
        the worker path without Cellpose."""
        def _stub(image, params):
            m = np.zeros(image.shape[:2], dtype=np.uint16)
            m[8:24, 8:24] = 1
            return m
        monkeypatch.setattr(live_preview, "_segment_once", _stub)
        panel = live_preview.LivePreviewPanel()
        qtbot.addWidget(panel)
        panel.load_image(sample_tif)
        received = []
        panel.preview_ready.connect(lambda m: received.append(m))
        panel.run_preview()
        # Wait for the worker thread to complete and Qt to marshal
        # the signal back to the main thread.
        qtbot.waitUntil(lambda: len(received) == 1, timeout=3000)
        assert received[0] is not None
        assert (received[0] > 0).any()
        assert "1 object" in panel._status.text() or \
            "found" in panel._status.text().lower()


# ---------------------------------------------------------------------------
# Integration into AppScreen
# ---------------------------------------------------------------------------

class TestAppScreenIntegration:
    def test_mask_screen_has_live_preview(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert getattr(scr, "_live_preview", None) is not None

    @pytest.mark.parametrize("app_key", ["measure", "classify", "umap"])
    def test_other_screens_omit_live_preview(self, qtbot, app_key):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen(app_key)
        qtbot.addWidget(scr)
        assert getattr(scr, "_live_preview", None) is None

    def test_autoload_from_folder(self, qtbot, tmp_path, sample_tif):
        """A folder containing a TIFF should feed the first tile into
        the live-preview panel automatically."""
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        # Bypass the debounce timer: call the loader directly
        scr._autoload_live_preview(str(tmp_path))
        assert scr._live_preview._image is not None

    def test_autoload_empty_folder_is_noop(self, qtbot, tmp_path):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._autoload_live_preview(str(tmp_path))
        assert scr._live_preview._image is None

    def test_autoload_placeholder_is_noop(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        for placeholder in ("", "path", "/path/to/src", "/path"):
            scr._autoload_live_preview(placeholder)
        assert scr._live_preview._image is None
