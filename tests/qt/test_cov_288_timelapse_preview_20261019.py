"""Timelapse preview: failed and stale loads, a whole-frame object, a tight picture budget."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import timelapse_preview as TP  # noqa: E402

pytestmark = pytest.mark.qt

FRAMES = 6
SIZE = 48


def _labels():
    stack = np.zeros((FRAMES, SIZE, SIZE), np.int32)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for t in range(FRAMES):
        stack[t][(yy - 12) ** 2 + (xx - 10 - 2 * t) ** 2 <= 16] = 1
    return stack


@pytest.fixture()
def panel(qtbot, tmp_path):
    import tifffile
    folder = tmp_path / "frames"
    folder.mkdir()
    labels = _labels()
    rng = np.random.default_rng(3)
    for t in range(FRAMES):
        image = (labels[t] > 0) * 700 + rng.integers(0, 90, (SIZE, SIZE))
        tifffile.imwrite(str(folder / f"field_t{t}.tif"), image.astype(np.uint16))
    widget = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(900, 700)
    widget.show()
    assert widget.load_sequence(str(folder))
    widget._masks = labels
    widget._tracked = labels.copy()
    widget._frame_slider.setMaximum(FRAMES - 1)
    widget._refresh_canvases()
    return widget


def test_a_frame_that_is_one_object_edge_to_edge_has_no_outline_to_colour():
    image = np.arange(SIZE * SIZE, dtype=np.uint16).reshape(SIZE, SIZE)
    whole = np.ones((SIZE, SIZE), np.int32)
    with_object = TP.render_frame(image, labels=whole, tracks=None, frame=0)
    without = TP.render_frame(image, labels=None, tracks=None, frame=0)
    assert np.array_equal(with_object, without)


def test_a_failure_with_no_placeholder_changes_nothing(panel):
    panel._transient_status = None
    panel._set_status("kept")
    panel._on_job_failed("boom")
    assert panel._status.text() == "kept"


def test_a_failure_replaces_only_the_placeholder_still_on_screen(panel):
    panel._set_transient_status("Opening field3…")
    panel._on_job_failed("disk gone")
    assert panel._status.text() == "Load failed: disk gone"
    assert panel._transient_status is None

    panel._set_transient_status("Opening field4…")
    panel._status.setText("a newer load reported")
    panel._on_job_failed("late failure")
    assert panel._status.text() == "a newer load reported"
    assert panel._transient_status is None


def test_a_failure_after_the_widget_died_is_survived(panel):
    class Dead:
        def text(self):
            raise RuntimeError("Internal C++ object already deleted")

    panel._transient_status = "Opening…"
    panel._status = Dead()
    panel._on_job_failed("x")
    assert panel._transient_status is None


def test_with_no_sequence_the_name_decides_whether_a_source_is_a_folder(
        panel, monkeypatch):
    asked = []
    monkeypatch.setattr(TP.path_probe, "isdir",
                        lambda text, default: asked.append((text, default)) or default)
    panel._sequence = None
    assert panel._loaded_source_is_a_folder("/plates/field3") is True
    assert panel._loaded_source_is_a_folder("/plates/movie.tif") is False
    assert asked == [("/plates/field3", True), ("/plates/movie.tif", False)]


def test_an_empty_mask_path_submits_nothing(panel):
    token = panel._mask_load_token
    assert panel.load_masks_async("  ") is False
    assert panel.load_masks_async(None) is False
    assert panel._mask_load_token == token


def test_a_mask_open_that_is_stale_failed_or_empty_installs_nothing(panel, monkeypatch):
    installed = []
    monkeypatch.setattr(panel, "_install_masks", installed.append)
    token = panel._mask_load_token
    panel._on_masks_loaded(token - 1, {"sequence": object()})
    panel._on_masks_loaded(token, None)
    panel._on_masks_loaded(token, {"sequence": None})
    assert installed == []
    panel._on_masks_loaded(token, {"error": "unreadable"})
    assert "Mask load failed: unreadable" in panel._status.text()
    assert installed == []
    seq = object()
    panel._on_masks_loaded(token, {"sequence": seq})
    assert installed == [seq]


def test_pictures_drawn_with_other_settings_are_dropped_first_when_over_budget(
        panel, monkeypatch):
    each = 2 * TP._pixmap_bytes(panel._src_view.picture())
    monkeypatch.setattr(TP, "_FRAME_PICTURE_BUDGET_BYTES", 2 * each)
    for index in range(2):
        panel._frame_slider.setValue(index)
    panel._tail.setValue(panel._tail.value() + 3)
    old = set(panel._frame_pictures)
    panel._frame_slider.setValue(2)
    keys = set(panel._frame_pictures)
    assert len(keys) <= 2
    assert any(k[0] == 2 for k in keys)
    assert len(old - keys) >= 1, "an entry drawn with the old settings was dropped"
