"""A slow comparison cannot freeze Qt or replace a newer field's result."""
import threading

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    made._canvas.image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    made._canvas.mask = np.zeros((64, 64), dtype=np.uint32)
    made._canvas.refresh()
    yield made
    made.close()


def test_slow_compare_is_responsive_and_cancellation_suppresses_completion(
        screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    threads = []

    def work(request):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return np.ones((64, 64), np.float32)

    monkeypatch.setattr(mm, '_compare_picture_for', work)
    try:
        screen._on_compare_enhanced()
        qtbot.waitUntil(entered.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        dialog = screen._compare_dialog
        assert dialog.views[1]._item is None
        assert threads == [screen._comparison_worker._thread.ident]
        assert threads[0] != threading.get_ident()
        dialog.close()
        assert screen._comparison_request is None
        release.set()
        qtbot.waitUntil(screen._comparison_worker.idle)
        qtbot.wait(20)
        assert not dialog.isVisible()
        assert dialog.views[1]._item is None
    finally:
        release.set()
        screen._comparison_worker.close()


@pytest.mark.parametrize('change', ['field', 'settings', 'normalization'])
@pytest.mark.parametrize('failure', [False, True])
def test_changed_context_discards_old_success_and_old_error(
        screen, qtbot, monkeypatch, change, failure):
    entered, release = threading.Event(), threading.Event()

    def work(request):
        entered.set()
        assert release.wait(5)
        if failure:
            raise ValueError('obsolete failure')
        return request.image

    monkeypatch.setattr(mm, '_compare_picture_for', work)
    try:
        screen._on_compare_enhanced()
        qtbot.waitUntil(entered.is_set)
        if change == 'field':
            screen._canvas.image = screen._canvas.image.copy()
        elif change == 'settings':
            screen._enh_gamma.setValue(0.5)
        else:
            screen._canvas.norm_hi = 98.0
        release.set()
        qtbot.waitUntil(lambda: screen._comparison_request is None)
        dialog = screen._compare_dialog
        assert dialog.views[1]._item is None
        assert 'changed' in dialog.caption.text()
        assert 'obsolete failure' not in dialog.caption.text()
    finally:
        release.set()
        screen._comparison_worker.close()


def test_comparison_uses_whole_field_normalization_before_cropping():
    image = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    request = mm._CompareRequest((1,), image, (30, 20, 50, 40),
                                mm.detect_chain.NO_CHAIN, True, (10, 90),
                                threading.Event())
    expected = mm.engine.normalize_for_detection(image, 10, 90)[20:40, 30:50]
    np.testing.assert_array_equal(mm._compare_picture_for(request), expected)
    np.testing.assert_array_equal(image, np.arange(10000).reshape(100, 100))


def test_comparison_failure_is_visible_without_enabling_apply(screen, qtbot, monkeypatch):
    def fail(request):
        raise ValueError('bad kernel')

    monkeypatch.setattr(mm, '_compare_picture_for', fail)
    screen._on_compare_enhanced()
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    assert 'bad kernel' in screen._compare_dialog.caption.text()
    assert not screen._btn_apply.isChecked()


def test_new_comparison_replaces_waiting_work_and_ignores_old_failure(
        screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def work(request):
        calls.append(request.key[0])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            raise ValueError('old failure')
        return request.image

    monkeypatch.setattr(mm, '_compare_picture_for', work)
    try:
        screen._on_compare_enhanced()
        qtbot.waitUntil(entered.is_set)
        screen._on_compare_enhanced()
        second = screen._compare_dialog
        screen._on_compare_enhanced()
        latest = screen._compare_dialog
        release.set()
        qtbot.waitUntil(lambda: screen._comparison_request is None)
        assert calls == [1, 3]
        assert not second.isVisible()
        assert latest.isVisible()
        assert not latest.views[1]._item.pixmap().isNull()
        assert 'old failure' not in latest.caption.text()
    finally:
        release.set()
        screen._comparison_worker.close()
