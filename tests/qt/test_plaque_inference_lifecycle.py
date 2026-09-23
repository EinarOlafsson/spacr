"""Cancelled model calls finish before any replacement inference starts."""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QThread, QTimer

from spacr.qt.widgets import plaque_preview as pv


@pytest.fixture
def panel(qtbot, tmp_path):
    path = tmp_path / 'plaque.png'
    Image.fromarray(np.full((40, 50, 3), 80, np.uint8)).save(path)
    made = pv.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(made)
    made.load_source_async(path)
    qtbot.waitUntil(lambda: not made._load_jobs.is_busy())
    yield made
    made.shutdown()


def test_cancel_and_changed_settings_cannot_overlap_inference(panel, qtbot, qapp):
    entered, release = threading.Event(), threading.Event()
    calls, ready = [], []
    panel.preview_ready.connect(ready.append)

    def segment(path):
        assert QThread.currentThread() != qapp.thread()
        calls.append(path)
        entered.set()
        assert release.wait(5)
        return np.zeros((40, 50), np.int32)

    try:
        assert panel.run_preview(segment=segment)
        qtbot.waitUntil(entered.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        panel.cancel_preview()
        panel._flow.setValue(0.2)
        assert not panel.run_preview(segment=segment), 'cancelled inference still owns the model'
        assert not panel._run_btn.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: panel._jobs.active_jobs() == 0)
        qtbot.waitUntil(panel._run_btn.isEnabled)
        assert ready == []
        assert panel.run_preview(segment=segment)
        qtbot.waitUntil(lambda: len(ready) == 1)
        assert len(calls) == 2
    finally:
        release.set()
        panel.shutdown()


def test_different_panels_never_evaluate_the_shared_model_concurrently(
        tmp_path, monkeypatch):
    path = tmp_path / 'plaque.png'
    Image.fromarray(np.full((40, 50, 3), 80, np.uint8)).save(path)
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    counter_lock = threading.Lock()
    active, maximum = 0, 0

    def segment(path):
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        entered.set()
        try:
            assert release.wait(5)
            return np.zeros((40, 50), np.int32)
        finally:
            with counter_lock:
                active -= 1

    def second():
        attempted.set()
        return pv.plaque_pass(path, {}, segment=segment)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(pv.plaque_pass, path, {}, segment=segment)
        try:
            assert entered.wait(3)
            next_run = pool.submit(second)
            assert attempted.wait(3)
            threading.Event().wait(.1)
        finally:
            release.set()
        assert first.result(timeout=3)['count'] == 0
        assert next_run.result(timeout=3)['count'] == 0
    assert maximum == 1


def test_all_wells_keep_the_settings_captured_at_start(panel, qtbot, monkeypatch):
    from types import SimpleNamespace

    first, release = threading.Event(), threading.Event()
    settings_seen = []
    panel._figure = {
        'image': np.ones((40, 50, 3), np.uint8),
        'regions': [SimpleNamespace(x0=0, y0=0, x1=20, y1=20),
                    SimpleNamespace(x0=20, y0=0, x1=40, y1=20)],
    }

    def segment(image, region, settings, **kwargs):
        settings_seen.append(settings['flow_threshold'])
        if len(settings_seen) == 1:
            first.set()
            assert release.wait(5)
        return {'labels': np.zeros((20, 20), np.int32), 'rows': [],
                'count': 0, 'mean_area': 0.0, 'note': ''}

    monkeypatch.setattr(pv, 'segment_well', segment)
    for name in ('_repaint_overlay', '_fill_table', '_fill_plaque_table', '_redraw_boxes'):
        monkeypatch.setattr(panel, name, lambda: None)
    monkeypatch.setattr(panel, 'select_well', lambda *args, **kwargs: None)
    panel._flow.setValue(.4)
    try:
        assert panel.find_plaques_in_all_wells()
        qtbot.waitUntil(first.is_set)
        panel._flow.setValue(.2)
        release.set()
        qtbot.waitUntil(lambda: not panel.preview_running())
        assert settings_seen == [.4, .4]
        assert panel.find_plaques_in_all_wells()
        qtbot.waitUntil(lambda: not panel.preview_running())
        assert settings_seen == [.4, .4, .2, .2]
    finally:
        release.set()
        panel.shutdown()


def test_cancelled_well_failure_cannot_overwrite_a_new_selection(panel, qtbot, monkeypatch):
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    panel._figure = {
        'image': np.ones((40, 50, 3), np.uint8),
        'regions': [SimpleNamespace(x0=0, y0=0, x1=20, y1=20)],
    }

    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise ValueError('old well failure')

    monkeypatch.setattr(pv, 'segment_well', fail)
    try:
        assert panel.find_plaques_in_all_wells()
        qtbot.waitUntil(entered.is_set)
        panel.cancel_preview()
        panel.set_preview_status('new selection')
        release.set()
        qtbot.waitUntil(lambda: panel._jobs.active_jobs() == 0)
        assert panel.preview_status() == 'new selection'
    finally:
        release.set()
        panel.shutdown()
