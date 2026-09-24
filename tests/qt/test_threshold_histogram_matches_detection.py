"""The threshold preview describes the selected method's processed field."""
from __future__ import annotations

import threading

import imageio.v3 as imageio
import numpy as np
import pytest
from skimage import filters

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    rng = np.random.default_rng(43)
    field = rng.normal(100, 5, (96, 96)).astype(np.float32)
    field[20:76, 20:76] += 100
    imageio.imwrite(tmp_path / 'field.tif', field)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(tmp_path))
    yield made
    made._magnifier.close()
    made.close_folded()


def choose(screen, mode):
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))


def preview(screen, qtbot):
    screen._on_show_otsu_histogram()
    dialog = screen._otsu_histogram_dialog
    assert dialog is not None
    qtbot.waitUntil(lambda: getattr(dialog, 'ready', True), timeout=10000)
    return dialog


@pytest.mark.parametrize('mode', ['otsu', 'li', 'yen', 'triangle', 'isodata', 'mean', 'minimum', 'multiotsu'])
@pytest.mark.parametrize('bright', [True, False])
@pytest.mark.parametrize('enhanced', [False, True])
def test_histogram_and_markers_match_processed_detector_input(screen, qtbot, mode, bright, enhanced):
    choose(screen, mode)
    screen._otsu_bright.setChecked(bright)
    screen._otsu_smoothing.setValue(1.0)
    screen._otsu_correction.setValue(1.1)
    if enhanced:
        screen._cp_invert.setChecked(True)
        screen._canvas.detect_on_normalized = True
        screen._enh_gamma.setValue(0.7)
    dialog = preview(screen, qtbot)
    values = engine._otsu_values(screen._detector_image(), 1.0)
    counts, edges = np.histogram(values, bins=mm.OTSU_HISTOGRAM_BINS)
    np.testing.assert_array_equal(dialog.plot.counts, counts)
    np.testing.assert_allclose(dialog.plot.edges, edges)
    if mode == 'multiotsu':
        expected = filters.threshold_multiotsu(values, classes=screen._otsu_classes.value()) * 1.1
    else:
        level = getattr(filters, 'threshold_' + mode)(values)
        expected = [level * 1.1 if bright else float(values.max()) - (float(values.max()) - level) * 1.1]
    assert dialog.plot.levels == pytest.approx(expected, rel=1e-6)
    assert mm._magnifier_mode_label(mode) in dialog.windowTitle()


@pytest.mark.parametrize('mode', ['otsu', 'sauvola', 'niblack'])
def test_local_methods_do_not_show_a_false_global_marker(screen, qtbot, mode):
    choose(screen, mode)
    if mode == 'otsu':
        screen._otsu_local.setChecked(True)
    dialog = preview(screen, qtbot)
    assert dialog.plot.levels == []
    assert 'varies' in dialog.caption.text()
    assert 'local' in dialog.caption.text().lower()
    assert 'single' in dialog.caption.text().lower()


def test_minimum_error_is_shown_without_an_otsu_marker(screen, qtbot):
    choose(screen, 'minimum')
    screen._canvas.image = np.ones((32, 32), np.float32)
    dialog = preview(screen, qtbot)
    assert getattr(dialog, 'error', None) is not None
    assert dialog.plot.levels == []
    assert 'failed' in dialog.caption.text().lower()


def test_non_threshold_methods_do_not_open_an_otsu_preview(screen):
    choose(screen, 'adaptive')
    screen._on_show_otsu_histogram()
    assert screen._otsu_histogram_dialog is None


def test_histogram_work_is_off_thread_and_closed_requests_do_not_replace_newer_results(screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = mm.detect_chain.prepare
    calls = []
    def slow(image, chain, *, cancel=None):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(image, chain, cancel=cancel)
    monkeypatch.setattr(mm.detect_chain, 'prepare', slow)
    gui_thread = threading.get_ident()
    try:
        choose(screen, 'li')
        screen._on_show_otsu_histogram()
        first = screen._otsu_histogram_dialog
        qtbot.waitUntil(entered.is_set, timeout=2000)
        assert not first.ready
        first.close()
        choose(screen, 'yen')
        screen._on_show_otsu_histogram()
        second = screen._otsu_histogram_dialog
        release.set()
        qtbot.waitUntil(lambda: second.ready, timeout=5000)
        assert calls and all(call != gui_thread for call in calls)
        assert first.closed
        assert 'Yen' in second.windowTitle()
        assert second.error is None
        assert second.plot.levels
    finally:
        release.set()


def test_corrected_levels_outside_the_image_range_have_space_on_the_axis(screen, qtbot):
    screen._otsu_correction.setValue(5.0)
    dialog = preview(screen, qtbot)
    assert dialog.plot.levels[0] > dialog.plot.edges[-1]
    assert dialog.plot.level_x(dialog.plot.edges[-1]) < dialog.plot.level_x(dialog.plot.levels[0])


@pytest.mark.parametrize('mode', ['li', 'niblack'])
def test_wrapped_captions_do_not_overlap_the_histogram_at_minimum_width(screen, qtbot, mode):
    choose(screen, mode)
    dialog = preview(screen, qtbot)
    dialog.resize(442, 200)
    qtbot.waitUntil(lambda: dialog.height() >= dialog.layout().totalHeightForWidth(dialog.width()))
    visible = [dialog.layout().itemAt(i).widget() for i in range(dialog.layout().count())
               if dialog.layout().itemAt(i).widget().isVisible()]
    for above, below in zip(visible, visible[1:]):
        assert above.geometry().bottom() < below.geometry().top()


def test_pending_preview_keeps_its_image_and_settings_snapshot(screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = mm.detect_chain.prepare
    def slow(image, chain, *, cancel=None):
        entered.set()
        assert release.wait(5)
        return original(image, chain, cancel=cancel)
    monkeypatch.setattr(mm.detect_chain, 'prepare', slow)
    choose(screen, 'li')
    screen._otsu_smoothing.setValue(0)
    original_image = screen._canvas.image.copy()
    try:
        screen._on_show_otsu_histogram()
        dialog = screen._otsu_histogram_dialog
        qtbot.waitUntil(entered.is_set, timeout=2000)
        screen._canvas.image[:] = 0
        screen._otsu_correction.setValue(2)
        choose(screen, 'mean')
        release.set()
        qtbot.waitUntil(lambda: dialog.ready, timeout=5000)
        assert dialog.plot.levels == pytest.approx([filters.threshold_li(original_image)])
        assert 'Li' in dialog.windowTitle()
        counts, edges = np.histogram(original_image.astype(np.float32), bins=mm.OTSU_HISTOGRAM_BINS)
        np.testing.assert_array_equal(dialog.plot.counts, counts)
        np.testing.assert_array_equal(dialog.plot.edges, edges)
    finally:
        release.set()


def test_closing_screen_retires_a_running_histogram_without_waiting_for_library_call(screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = mm.detect_chain.prepare
    def slow(image, chain, *, cancel=None):
        entered.set()
        assert release.wait(5)
        return original(image, chain, cancel=cancel)
    monkeypatch.setattr(mm.detect_chain, 'prepare', slow)
    try:
        screen._on_show_otsu_histogram()
        dialog = screen._otsu_histogram_dialog
        worker = screen._histogram_worker
        qtbot.waitUntil(entered.is_set, timeout=2000)
        screen.close_folded()
        assert dialog.closed
        assert dialog.request.ticket.cancelled()
        assert screen._histogram_worker is None
        assert not worker.idle()
    finally:
        release.set()
    qtbot.waitUntil(worker.idle, timeout=5000)
    assert not dialog.ready
