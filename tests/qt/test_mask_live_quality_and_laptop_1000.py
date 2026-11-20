"""Item 1000: Mask Generation's Live view gets OPS Live's image quality and laptop Refresh view.

The same 10-100 % image-quality slider draws the canvases at fewer pixels
while zoom and the hovered pixel stay in the image's own pixels, and in
laptop mode a settings change waits for Refresh view.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QPixmap  # noqa: E402

from spacr.qt.widgets import live_preview as LP  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_view_draws_at_the_quality_and_keeps_its_scene(qtbot):
    view = LP._ZoomView()
    qtbot.addWidget(view)
    view.set_pixmap(QPixmap(400, 200))
    rect = view.sceneRect()
    view.set_render_percent(25)
    shown = view.shown_pixmap()
    assert (shown.width(), shown.height()) == (100, 50)
    assert view.sceneRect() == rect, "zoom and pan stay in the image's pixels"
    assert view._pixmap_item.sceneBoundingRect().width() == pytest.approx(400)
    assert view.picture().width() == 400, "saving keeps the whole picture"
    view.set_pixmap(QPixmap(80, 60))
    assert (view.shown_pixmap().width(), view.shown_pixmap().height()) == (20, 15)
    view.set_render_percent(100)
    assert view.shown_pixmap().width() == 80
    assert view._pixmap_item.transform().isIdentity()


def _panel(qtbot):
    panel = LP.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    rng = np.random.default_rng(1)
    panel._image = rng.integers(0, 4000, (120, 160, 2)).astype(np.uint16)
    mask = np.zeros((120, 160), np.int32)
    mask[10:30, 10:30] = 1
    panel._raw_masks = {"cell": mask}
    panel._recompute_masks(snapshot=True)
    return panel


def test_the_mask_preview_has_the_same_quality_slider(qtbot, monkeypatch):
    monkeypatch.setattr(LP, "_laptop_mode", lambda: False)
    panel = _panel(qtbot)
    try:
        control = panel._quality_control
        assert (control.slider.minimum(), control.slider.maximum()) == (10, 100)
        assert control.slider.accessibleName() == "Image quality"
        assert panel._src_view.shown_pixmap().width() == 160
        control.slider.setValue(50)
        control._announce()
        assert panel._src_view.shown_pixmap().width() == 80
        assert panel._mask_view.shown_pixmap().width() == 80
        assert panel._laptop_refresh_btn.isHidden()
    finally:
        panel.shutdown()


def test_laptop_mode_holds_mask_changes_for_refresh_view(qtbot, monkeypatch):
    monkeypatch.setattr(LP, "_laptop_mode", lambda: True)
    panel = _panel(qtbot)
    started = []
    monkeypatch.setattr(panel, "_start_filter_job", lambda: started.append(1))
    try:
        panel._schedule_recompute(debounce=False)
        assert started == [] and panel._laptop_pending
        assert "Refresh view" in panel._status.text()
        assert not panel._laptop_refresh_btn.isHidden()
        panel._quality_control.slider.setValue(50)
        panel._quality_control._announce()
        assert panel._src_view.shown_pixmap().width() == 160, "waits for Refresh view"
        panel._laptop_refresh_btn.click()
        assert started == [1] and not panel._laptop_pending
        assert panel._src_view.shown_pixmap().width() == 80
    finally:
        panel.shutdown()
