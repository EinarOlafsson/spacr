"""418: cached-mask filtering must see each object's own raw channel."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import live_preview as LP


@pytest.mark.parametrize("role,channel", [
    ("cell", 3), ("nucleus", 2), ("pathogen", 4),
    ("organelle", 1), ("organelleb", 5), ("organellec", 0),
])
def test_refilter_receives_the_same_own_channel_as_the_segmentation_request(
        qtbot, monkeypatch, tmp_path, role, channel):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.apply_settings({
        "number_of_organelles": 3,
        "cell_channel": 3, "nucleus_channel": 2, "pathogen_channel": 4,
        "organelle_channel": 1, "organelleb_channel": 5,
        "organellec_channel": 0,
    })
    image = np.broadcast_to(np.arange(1, 7, dtype=np.uint16) * 100,
                            (16, 16, 6)).copy()
    path = tmp_path / "field.tif"
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "YXC"})
    assert panel.load_image(path), panel._status.text()
    raw_mask = np.zeros((16, 16), np.uint16)
    raw_mask[3:8, 3:8] = 1
    panel._raw_masks = {role: raw_mask.copy()}
    calls = []

    def capture_plane(mask, settings, obj, intensity_img=None):
        """Observe the input boundary; the actual filter is tested separately."""
        calls.append((obj, intensity_img.copy()))
        return mask

    monkeypatch.setattr(LP, "_apply_size_filter", capture_plane)
    try:
        request = panel._build_request()
        assert request.channels[role] == channel
        panel._recompute_masks()
        assert len(calls) == 1
        assert calls[0][0] == role
        np.testing.assert_array_equal(calls[0][1], image[..., channel])
        np.testing.assert_array_equal(panel._raw_masks[role], raw_mask)
        assert panel._worker is None, "refiltering must not start segmentation"
    finally:
        panel.shutdown()


def test_switching_organelle_slots_keeps_each_filter_channel(qtbot):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.apply_settings({"number_of_organelles": 2,
                          "organelle_channel": 1, "organelleb_channel": 5})
    try:
        for active, caption in (("organelle", "organelle"),
                                ("organelleb", "organelle 2"),
                                ("organelle", "organelle")):
            index = panel._object_box.findData(caption)
            assert index >= 0, f"{active} must actually be offered"
            panel._object_box.setCurrentIndex(index)
            assert panel._active_organelle_role == active
            assert panel._obj_channel("organelle") == 1
            assert panel._obj_channel("organelleb") == 5
        panel._organelle_channel.setValue(4)
        assert panel._obj_channel("organelle") == 4
        assert panel._obj_channel("organelleb") == 5
    finally:
        panel.shutdown()


def test_second_organelle_is_segmented_from_its_own_plane(qtbot, monkeypatch):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.apply_settings({"number_of_organelles": 2,
                          "organelle_channel": 1, "organelleb_channel": 3,
                          "organelleb_method": "otsu"})
    index = panel._object_box.findData("organelle 2")
    assert index >= 0
    panel._object_box.setCurrentIndex(index)
    panel._image = np.broadcast_to(np.arange(1, 5, dtype=np.uint16) * 100,
                                   (16, 16, 4)).copy()
    seen = []

    def classical(image, role, settings):
        """Observe the plane delivered to the real segmentation boundary."""
        seen.append((role, image.copy()))
        return np.ones(image.shape, np.int32)

    monkeypatch.setattr(LP, "preview_cellpose_model", lambda model: object())
    monkeypatch.setattr(LP, "_classical_organelle_mask", classical)
    try:
        request = panel._build_request()
        assert request.object_types == ("organelleb",)
        assert request.preprocess_settings["organelleb_method"] == "otsu"
        masks, _ = LP._segment_multi(request)
        assert len(seen) == 1
        assert seen[0][0] == "organelleb"
        np.testing.assert_array_equal(seen[0][1], panel._image[..., 3])
        np.testing.assert_array_equal(masks["organelleb"], np.ones((16, 16)))
    finally:
        panel.shutdown()
