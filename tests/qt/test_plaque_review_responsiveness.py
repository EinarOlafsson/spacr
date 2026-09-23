"""Figure sidecars and scale inference must not block the Qt event thread."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QThread, QTimer

from spacr import plaque_papers as papers
from spacr.qt.widgets import plaque_preview as preview


def detect(image, weights, **kwargs):
    return [SimpleNamespace(x0=5, y0=5, x1=45, y1=45, confidence=.9)]


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(preview, 'missing_papers_packages', lambda: [])
    widget = preview.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(widget)
    widget.set_mode('figure')
    for name in ('a.png', 'b.png'):
        Image.fromarray(np.full((60, 70, 3), 120, np.uint8)).save(tmp_path / name)
    widget.load_source_async(tmp_path)
    qtbot.waitUntil(lambda: not widget._load_jobs.is_busy(), timeout=5000)
    yield widget
    widget.shutdown()


def test_figure_sidecars_and_scale_inference_run_off_the_gui_thread(panel, qtbot, qapp, monkeypatch):
    threads = {}
    for owner, name in [(papers, 'read_legends'), (papers, 'read_annotation_overrides'),
                        (preview, '_figure_scales')]:
        original = getattr(owner, name)

        def record(*args, _name=name, _fn=original, **kwargs):
            threads[_name] = QThread.currentThread() == qapp.thread()
            return _fn(*args, **kwargs)

        monkeypatch.setattr(owner, name, record)
    assert panel.run_preview(detect=detect, read_text=lambda path: [])
    qtbot.waitUntil(lambda: panel._figure is not None, timeout=5000)
    assert threads == {'read_legends': False, 'read_annotation_overrides': False,
                       '_figure_scales': False}
    assert panel._table.rowCount() == 1 and panel._well_view.has_image()


def test_slow_old_legend_read_keeps_window_responsive_and_cannot_replace_new_image(
        panel, qtbot, qapp, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = papers.read_legends

    def blocked(path):
        assert QThread.currentThread() != qapp.thread(), 'disk read on GUI thread'
        entered.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(papers, 'read_legends', blocked)
    ticks = []
    timer = QTimer(panel)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    try:
        assert panel.run_preview(detect=detect, read_text=lambda path: [])
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: len(ticks) >= 3, timeout=1000)
        panel._step(1)
        qtbot.waitUntil(lambda: not panel._load_jobs.is_busy(), timeout=3000)
        assert not panel._run_btn.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: not panel.preview_running(), timeout=3000)
        qtbot.waitUntil(panel._run_btn.isEnabled, timeout=3000)
        assert panel.current_path().name == 'b.png' and panel._figure is None
        assert panel._run_btn.isEnabled()
    finally:
        release.set()
        timer.stop()


def test_late_reader_error_cannot_overwrite_new_image(panel, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def delayed_failure(path):
        entered.set()
        assert release.wait(5)
        raise OSError('old figure sidecar failed')

    monkeypatch.setattr(papers, 'read_legends', delayed_failure)
    try:
        assert panel.run_preview(detect=detect, read_text=lambda path: [])
        qtbot.waitUntil(entered.is_set, timeout=3000)
        panel._step(1)
        qtbot.waitUntil(lambda: not panel._load_jobs.is_busy(), timeout=3000)
        release.set()
        qtbot.waitUntil(lambda: not panel._jobs.is_busy(), timeout=3000)
        assert 'old figure sidecar failed' not in panel.preview_status()
        assert panel._figure is None and panel.current_path().name == 'b.png'
    finally:
        release.set()


def test_reannotation_preserves_newer_manual_edit_and_runs_off_gui(panel, qtbot, qapp, monkeypatch):
    assert panel.run_preview(detect=detect, read_text=lambda path: [])
    qtbot.waitUntil(lambda: panel._figure is not None, timeout=5000)
    entered, release = threading.Event(), threading.Event()
    original = preview._figure_scales

    def blocked(*args, **kwargs):
        assert QThread.currentThread() != qapp.thread()
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(preview, '_figure_scales', blocked)
    try:
        panel._reannotate()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        panel._table.item(0, preview.PIXELS_PER_UM_COLUMN).setText('0.2')
        panel._table.item(0, preview.CONDITION_COLUMN).setText('edited during review')
        release.set()
        qtbot.waitUntil(lambda: not panel._review_jobs.is_busy(), timeout=3000)
        assert panel._annotations[0].pixels_per_um == .2
        assert panel._annotations[0].condition == 'edited during review'
    finally:
        release.set()


def test_annotation_save_is_off_gui_and_keeps_snapshot_after_navigation(panel, qtbot, qapp, monkeypatch):
    assert panel.run_preview(detect=detect, read_text=lambda path: [])
    qtbot.waitUntil(lambda: panel._figure is not None, timeout=5000)
    entered, release = threading.Event(), threading.Event()
    original = papers.write_annotation_overrides

    def blocked(*args, **kwargs):
        assert QThread.currentThread() != qapp.thread()
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(papers, 'write_annotation_overrides', blocked)
    try:
        panel._table.item(0, preview.CONDITION_COLUMN).setText('saved old figure')
        destination = panel.save_annotations()
        qtbot.waitUntil(entered.is_set, timeout=3000)
        assert panel.save_annotations() is None
        panel._step(1)
        release.set()
        qtbot.waitUntil(lambda: not panel._save_jobs.is_busy(), timeout=3000)
        saved = papers.read_annotation_overrides(destination)
        assert saved[('a', 1)]['condition'] == 'saved old figure'
        assert panel.current_path().name == 'b.png'
        assert 'Saved 1 annotations' not in panel.preview_status()
    finally:
        release.set()
