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
    sources = [_image(tmp_path / "a.png"), _image(tmp_path / "b.png", 160)]
    original_bytes = {path: path.read_bytes() for path in sources}
    assert widget.load_source_async(tmp_path)
    qtbot.waitUntil(lambda: widget.current_path() == sources[0]
                    and not widget._load_jobs.is_busy()
                    and widget._load_jobs.active_jobs() == 0
                    and widget._view.has_image(), timeout=5000)
    np.testing.assert_array_equal(widget._view.array(), np.full((40, 50, 3), 80, np.uint8))
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
        qtbot.waitUntil(widget._run_btn.isEnabled, timeout=3000)
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
        assert {path: path.read_bytes() for path in sources} == original_bytes
    finally:
        release.set()
        widget.shutdown()


def test_loading_an_empty_folder_clears_previous_results_and_tables(panel, tmp_path):
    source = _image(tmp_path / "a.png")
    assert panel.load_source_async(source)
    assert panel.run_preview(segment=lambda path: np.ones((40, 50), np.int32))
    panel.views().set_view("Masks")
    assert panel._plaque_result is not None and panel._view.has_image()
    assert not panel.views().is_showing_message()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert panel.load_source_async(empty)
    assert panel._plaque_result is None
    assert panel.views().is_showing_message()
    panel.views().set_view("Overlay")
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
    panel.views().set_view("Masks")
    assert "No plaques" in panel.views().message_text()


def test_stale_callback_cannot_enable_run_during_a_new_preview(panel):
    old = panel.preview_token()
    panel.cancel_preview()
    panel.set_preview_busy(True)
    panel._on_result(old, {"error": "old failure"})
    assert not panel._run_btn.isEnabled() and panel._cancel_btn.isEnabled()
    assert "old failure" not in panel.preview_status()


def test_model_download_failure_can_be_retried_without_automatic_inference(panel, tmp_path, monkeypatch):
    import types

    from PySide6.QtCore import QObject, Signal
    from spacr.qt import model_install

    monkeypatch.setenv("HOME", str(tmp_path))
    created = []

    class Download(QObject):
        progressed = Signal(int, int)
        finished = Signal(bool, str)

        def __init__(self, entry, folder):
            super().__init__()
            self.entry, self.folder, self.started = entry, folder, False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(model_install, "CheckpointDownload", Download)
    monkeypatch.setattr(model_install, "_confirm_checkpoint_download", lambda *_args: False)
    ready = []
    panel.preview_ready.connect(ready.append)
    panel._offer_download(None)
    panel._start_download()
    assert not created and panel._download_btn.isHidden()
    entry = types.SimpleNamespace(key="missing-plaque", name="model.pt")
    panel._offer_download(entry)
    assert "missing-plaque" in panel._download_btn.text()
    panel._download_btn.click()
    first = created[0]
    assert first.entry is entry and first.started
    assert first.folder == str(tmp_path / ".spacr/models")
    assert not panel._download_btn.isEnabled()
    first.progressed.emit(1024, 0)
    assert "?" in panel.preview_status()
    first.progressed.emit(1024, 2048)
    assert "?" not in panel.preview_status()
    first.finished.emit(False, "transfer interrupted")
    assert "transfer interrupted" in panel.preview_status()
    assert panel._download_btn.isEnabled() and not panel._download_btn.isHidden()
    panel._download_btn.click()
    assert len(created) == 2 and created[1].started
    created[1].finished.emit(True, str(tmp_path / ".spacr/models/model.pt"))
    assert panel._download_btn.isEnabled() and panel._download_btn.isHidden()
    assert "Press Run preview" in panel.preview_status() and not ready


def test_unchanged_source_refresh_keeps_existing_mask_and_selection(panel, tmp_path):
    source = _image(tmp_path / "a.png")
    assert not panel.load_source_async("")
    assert panel.load_source_async(source)
    assert panel.run_preview(segment=lambda path: np.ones((40, 50), np.int32))
    previous = panel._plaque_result
    assert panel.load_source_async(source)
    assert panel._plaque_result is previous and panel.current_path() == source
    assert "unchanged" in panel.preview_status()


def test_fetched_paper_still_loads_when_form_propagation_fails(panel, tmp_path):
    source = _image(tmp_path / "a.png")
    propagated = []

    def cannot_write(settings):
        propagated.append(settings)
        raise RuntimeError("closed form")

    panel.set_propagate_callback(cannot_write)
    assert not panel.fetch_paper("", tmp_path)
    assert not panel.fetch_paper("paper", "")
    panel._on_paper_fetched({"paper": "fixture", "figures": 0})
    assert panel.current_path() is None
    assert panel.fetch_paper("fixture", tmp_path, fetch=lambda *args:
                             {"folder": str(tmp_path), "paper": "fixture", "figures": 1})
    assert propagated == [{"src": str(tmp_path)}]
    assert panel.current_path() == source and panel._paper_btn.isEnabled()


def test_run_starts_while_a_delivered_load_thread_waits_for_retirement(qtbot, monkeypatch, tmp_path):
    """A load that has delivered its image no longer blocks Run, even before its thread retires."""
    from PySide6.QtCore import Qt

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda: [])
    widget = ppv.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(widget)
    loads = widget._load_jobs
    held = []
    monkeypatch.setattr(loads, "_retire_finished_jobs", lambda: held.append(True))
    path = _image(tmp_path / "a.png")
    assert widget.load_source_async(tmp_path)
    qtbot.waitUntil(lambda: widget.current_path() == path and not loads.is_busy()
                    and widget._view.has_image() and len(held) >= 2, timeout=5000)
    assert loads.active_jobs() > 0, "the delivered load threads are still unretired"
    assert loads._jobs_still_working() == 0
    entered, release = threading.Event(), threading.Event()

    def segment(_path):
        entered.set()
        assert release.wait(5)
        return np.zeros((40, 50), np.int32)

    real_pass = ppv.plaque_pass
    monkeypatch.setattr(ppv, "plaque_pass",
                        lambda image, settings, segment=None: real_pass(image, settings, segment=segment_fn))
    segment_fn = segment
    try:
        assert widget._preview_blocked_reason() == ""
        qtbot.mouseClick(widget._run_btn, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=3000)
    finally:
        release.set()
        monkeypatch.undo()
        loads._retire_finished_jobs()
    qtbot.waitUntil(lambda: widget._jobs.active_jobs() == 0, timeout=5000)
    widget.shutdown()


def test_run_still_refuses_while_a_load_is_in_flight(qtbot, monkeypatch, tmp_path):
    """A load whose worker is still running keeps Run refused, also after it was cancelled."""
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda: [])
    widget = ppv.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(widget)
    path = _image(tmp_path / "a.png")
    assert widget.load_source_async(tmp_path)
    qtbot.waitUntil(lambda: widget.current_path() == path and not widget._load_jobs.is_busy()
                    and widget._load_jobs.active_jobs() == 0, timeout=5000)
    entered, release = threading.Event(), threading.Event()

    def slow(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return np.full((40, 50, 3), 80, np.uint8)

    monkeypatch.setattr(ppv, "load_display_image", slow)
    try:
        widget._show_selected_image()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert widget._preview_blocked_reason()
        widget._load_jobs.cancel()
        assert not widget._load_jobs.is_busy()
        assert widget._load_jobs._jobs_still_working() == 1
        assert widget._preview_blocked_reason()
    finally:
        release.set()
    qtbot.waitUntil(lambda: widget._load_jobs.active_jobs() == 0, timeout=5000)
    assert widget._preview_blocked_reason() == ""
    widget.shutdown()
