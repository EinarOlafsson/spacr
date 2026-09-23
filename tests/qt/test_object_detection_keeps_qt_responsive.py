"""A slow field detector never owns Qt or a mask edited after its snapshot."""
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_make_masks_cellpose_detect import field_folder, screen


pytestmark = pytest.mark.qt


@pytest.fixture
def held_detector(screen, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    seen = []
    screen._cp_loaded['cpsam'] = object()

    def detect(image, model, **kwargs):
        seen.append((threading.get_ident(), image.copy(), kwargs))
        entered.set()
        assert release.wait(10), 'test did not release detector'
        labels = np.zeros(image.shape, np.int32)
        labels[10:30, 10:30] = 1
        return labels, np.ones(image.shape), np.zeros(image.shape + (3,), np.uint8)

    monkeypatch.setattr(mm, 'cellpose_detect', detect)
    yield entered, release, seen
    release.set()
    worker = screen._detection_worker
    if worker is not None:
        assert worker.close(timeout=5)


def test_button_returns_and_qt_ticks_while_model_runs(screen, held_detector, qtbot):
    entered, release, seen = held_detector
    ticks = []
    timer = QTimer(screen)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    screen._btn_cellpose.click()
    qtbot.waitUntil(entered.is_set)
    qtbot.waitUntil(lambda: len(ticks) >= 3)
    assert seen[0][0] != threading.get_ident()
    assert not screen._btn_cellpose.isEnabled()
    assert not screen._canvas.mask.any()
    assert screen.run_cellpose() == 0
    screen._on_detect_cellpose()
    assert len(seen) == 1
    release.set()
    qtbot.waitUntil(lambda: screen._detection_request is None)
    assert screen._canvas.mask.max() == 1
    assert screen._prob_pane.has_image() and screen._flow_pane.has_image()
    assert screen._btn_cellpose.isEnabled()
    assert screen._log.edits[-1].kind == 'detect'
    timer.stop()


@pytest.mark.parametrize('change', ['navigate', 'edit', 'replace_image', 'edit_image'])
def test_changed_field_or_mask_rejects_late_results(screen, held_detector, qtbot, change):
    entered, release, _ = held_detector
    screen._on_detect_cellpose()
    qtbot.waitUntil(entered.is_set)
    if change == 'navigate':
        screen._on_next()
        qtbot.waitUntil(lambda: not screen._loading)
    elif change == 'edit':
        screen._canvas.mask[2:8, 2:8] = 9
    elif change == 'edit_image':
        screen._canvas.image[0, 0] += 1
    else:
        screen._canvas.image = screen._canvas.image.copy()
    before = screen._canvas.mask.copy()
    release.set()
    qtbot.waitUntil(lambda: screen._detection_request is None)
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert not screen._prob_pane.has_image()
    assert 'discarded' in screen._status_label.text()
    assert screen._btn_cellpose.isEnabled()


def test_settings_and_input_are_snapshots_and_provenance_matches(screen, held_detector, qtbot):
    entered, release, seen = held_detector
    screen._canvas.image = np.arange(6400, dtype=np.uint16).reshape(80, 80)
    screen._canvas.detect_on_normalized = True
    screen._canvas.norm_lo, screen._canvas.norm_hi = 10, 90
    screen._enh_gamma.setValue(0.6)
    screen._btn_apply.click()
    screen._cp_invert.setChecked(True)
    screen._cp_diameter.setValue(40)
    expected = screen._detector_image().copy()
    screen._on_detect_cellpose()
    qtbot.waitUntil(entered.is_set)
    screen._cp_invert.setChecked(False)
    screen._cp_diameter.setValue(70)
    screen._enh_gamma.setValue(0.9)
    release.set()
    qtbot.waitUntil(lambda: screen._detection_request is None)
    np.testing.assert_array_equal(seen[0][1], expected)
    assert seen[0][2]['diameter'] == 40
    details = screen._log.edits[-1].detail
    assert details['diameter'] == 40 and details['invert'] is True
    assert details['enhancement']['gamma'] == 0.6


def test_close_detaches_without_waiting_or_delivering_to_closed_screen(screen, held_detector, qtbot):
    entered, release, _ = held_detector
    screen._on_detect_cellpose()
    qtbot.waitUntil(entered.is_set)
    worker = screen._detection_worker
    before = screen._canvas.mask.copy()
    started = time.monotonic()
    screen.close()
    assert time.monotonic() - started < 1
    assert screen._detection_request is None
    release.set()
    assert worker.close(timeout=5)
    qtbot.wait(20)
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_worker_failures_leave_mask_unchanged_and_allow_retry(screen, monkeypatch, qtbot):
    screen._cp_loaded['cpsam'] = object()
    before = screen._canvas.mask.copy()
    warnings = []
    monkeypatch.setattr(screen, '_warn', lambda *args: warnings.append(args))

    def fail(*args, **kwargs):
        raise RuntimeError('failed model')

    monkeypatch.setattr(mm, 'cellpose_detect', fail)
    for _ in range(2):
        screen._on_detect_cellpose()
        qtbot.waitUntil(lambda: screen._detection_request is None)
        assert screen._btn_cellpose.isEnabled()
    assert len(warnings) == 2 and warnings[-1][1] == 'failed model'
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_model_loading_and_enhancement_also_run_off_qt(screen, monkeypatch, qtbot):
    identities = []
    screen._cp_loaded.clear()
    original_prepare = mm.detect_chain.prepare

    def prepare(*args, **kwargs):
        identities.append(threading.get_ident())
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(mm.detect_chain, 'prepare', prepare)
    monkeypatch.setattr(mm, 'load_cellpose_model', lambda name: identities.append(threading.get_ident()) or object())
    monkeypatch.setattr(mm, 'cellpose_detect', lambda image, *args, **kwargs: (np.zeros(image.shape, np.int32), np.ones(image.shape), None))
    screen._on_detect_cellpose()
    qtbot.waitUntil(lambda: screen._detection_request is None)
    assert identities and all(identity != threading.get_ident() for identity in identities)
    assert not screen._canvas.mask.any()
    assert screen._prob_pane.has_image()
