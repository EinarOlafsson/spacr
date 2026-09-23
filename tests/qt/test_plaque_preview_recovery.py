"""Plaque preview recovery keeps results tied to their selected image/mode."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr.qt.widgets import plaque_preview as ppv


def _image(path, value=80):
    Image.fromarray(np.full((40, 50, 3), value, np.uint8)).save(path)
    return path


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("change", ["image", "mode", "empty_folder"])
def test_inflight_segmentation_cannot_replace_a_new_selection(qtbot, monkeypatch, tmp_path, change):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda: [])
    widget = ppv.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(widget)
    _image(tmp_path / "a.png")
    _image(tmp_path / "b.png", 160)
    assert widget.load_source_async(tmp_path)
    qtbot.waitUntil(lambda: widget.current_path() == tmp_path / "a.png"
                    and not widget._load_jobs.is_busy(), timeout=5000)
    entered, release = threading.Event(), threading.Event()
    ready = []
    widget.preview_ready.connect(ready.append)

    def segment(path):
        entered.set()
        assert release.wait(5), "test releases the worker even after failure"
        labels = np.zeros((40, 50), np.int32)
        labels[5:15, 5:15] = 1
        return labels

    try:
        assert widget.run_preview(segment=segment)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        if change == "image":
            widget._step(1)
        elif change == "mode":
            widget.set_mode("figure")
        else:
            empty = tmp_path / "empty"
            empty.mkdir()
            assert widget.load_source_async(empty)
        qtbot.waitUntil(lambda: not widget._load_jobs.is_busy(), timeout=3000)
        release.set()
        qtbot.waitUntil(lambda: widget._jobs.active_jobs() == 0, timeout=3000)
        assert ready == [], "the old selection's result must be discarded"
        assert widget._plaque_result is None and widget._figure is None
        assert widget._run_btn.isEnabled() and not widget._cancel_btn.isEnabled()
        if change == "image":
            assert widget.current_path() == tmp_path / "b.png"
            np.testing.assert_array_equal(widget._view.array(), np.full((40, 50, 3), 160, np.uint8))
        elif change == "empty_folder":
            assert widget.current_path() is None and not widget._view.has_image()
        else:
            assert widget.mode() == "figure"
    finally:
        release.set()
        widget.shutdown()


def test_loading_an_empty_folder_clears_previous_results_and_tables(panel, tmp_path):
    source = _image(tmp_path / "a.png")
    assert panel.load_source_async(source)
    assert panel.run_preview(segment=lambda path: np.ones((40, 50), np.int32))
    assert panel._plaque_result is not None and panel._objects_view.has_image()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert panel.load_source_async(empty)
    assert panel._plaque_result is None
    assert not panel._objects_view.has_image()
    assert panel._table.rowCount() == panel._plaque_table.rowCount() == 0
    assert not panel._view.has_image() and panel.current_path() is None
    assert not panel.run_preview()


def test_worker_failure_returns_to_idle_and_next_preview_can_succeed(panel, tmp_path):
    assert panel.load_source_async(_image(tmp_path / "a.png"))

    def fail(path):
        raise OSError("unreadable checkpoint")

    assert panel.run_preview(segment=fail)
    assert "unreadable checkpoint" in panel.preview_status()
    assert panel._run_btn.isEnabled() and not panel._cancel_btn.isEnabled()
    assert panel.run_preview(segment=lambda path: np.zeros((40, 50), np.int32))
    assert panel._plaque_result["count"] == 0
    assert "No plaques" in panel._objects_view.text()


def test_stale_callback_cannot_enable_run_during_a_new_preview(panel):
    old = panel.preview_token()
    panel.cancel_preview()
    panel.set_preview_busy(True)
    panel._on_result(old, {"error": "old failure"})
    assert not panel._run_btn.isEnabled() and panel._cancel_btn.isEnabled()
    assert "old failure" not in panel.preview_status()
