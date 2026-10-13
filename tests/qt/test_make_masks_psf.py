"""PSF controls capture calibrated kernels and feed the actual detection chain."""
import json
import threading

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from spacr.point_spread import ProcessingCancelled, apply_psf, gaussian_psf
from spacr.qt import detect_chain as dc
from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import psf_controls as controls


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    image = np.zeros((48, 48), np.uint16)
    image[20:28, 20:28] = 4000
    made._canvas.set_image_and_mask(image, np.zeros_like(image, dtype=np.uint32))
    yield made
    made.close()


def _gaussian(screen, qtbot, operation='convolve'):
    widget = screen._psf_controls
    widget.image_y.setValue(.2)
    widget.image_x.setValue(.3)
    widget.fwhm_y.setValue(.6)
    widget.fwhm_x.setValue(.9)
    widget.operation.setCurrentIndex(widget.operation.findData(operation))
    qtbot.waitUntil(lambda: widget._kernel is not None)
    return widget


def test_default_psf_is_off_and_its_common_calibration_says_so(screen):
    """509: off by default, but already filled with common optics that name their source."""
    widget = screen._psf_controls
    assert widget.operation.currentData() == 'none'
    assert widget.image_x.value() == widget.image_y.value() == 0.325
    assert widget.source_of('image_x')[0] == 'calculated'
    assert widget.source_of('magnification') == ('default', '20x/0.75 air')
    assert screen._enhancement_chain() == dc.NO_CHAIN
    assert not widget._jobs.is_busy()
    assert widget.operation.property('apiTooltipHtml')
    assert 'point_spread/index.html#spacr.point_spread.apply_psf' in widget.operation.property('apiTooltipHtml')


@pytest.mark.parametrize('case,error', [
    ('gaussian', ''), ('measured', ''), ('uncalibrated', 'positive image pixel'),
    ('missing_file', 'Choose a measured PSF file'),
    ('different_sampling', 'spacing must match'),
    ('volume', '3 positive finite values'),
])
def test_captured_kernel_request_validates_calibration_before_publication(
        qtbot, monkeypatch, tmp_path, case, error):
    """Exercise captured request validation; background scheduling is checked separately."""
    widget = controls._PSFControls()
    qtbot.addWidget(widget)
    submitted = []
    monkeypatch.setattr(widget._jobs, 'submit',
                        lambda work, complete: submitted.append((work, complete)))
    path = tmp_path / 'kernel.npy'
    np.save(path, np.ones((3, 3, 3) if case == 'volume' else (3, 3)))
    original = path.read_bytes()
    try:
        for spin in (widget.image_y, widget.image_x, widget.kernel_y,
                     widget.kernel_x, widget.fwhm_y, widget.fwhm_x):
            spin.setValue(.2)
        if case != 'gaussian':
            widget.source.setCurrentIndex(widget.source.findData('measured'))
        if case != 'missing_file':
            widget.path.setText(str(path))
        if case == 'uncalibrated':
            widget.image_y.setValue(0)
        if case == 'different_sampling':
            widget.kernel_x.setValue(.4)
        widget.operation.setCurrentIndex(widget.operation.findData('convolve'))
        widget._timer.stop()
        widget._load()
        work, complete = submitted.pop()
        kernel, reason = work()
        complete((kernel, reason))
        if error:
            assert kernel is None and error in reason
            assert error in widget.status.text()
            assert widget._chain_fields()['psf'] is None
        else:
            assert reason == '' and kernel is widget._chain_fields()['psf']
            assert kernel.sampling_um == (.2, .2)
            assert 'PSF ready' in widget.status.text()
        assert path.read_bytes() == original
        status = widget.status.text()
        widget._shutdown()
        widget._loaded((None, 'late result'))
        assert widget.status.text() == status
    finally:
        widget._shutdown()


@pytest.mark.parametrize('accepted', [False, True])
def test_measured_kernel_picker_preserves_path_when_cancelled(qtbot, monkeypatch, accepted):
    widget = controls._PSFControls()
    qtbot.addWidget(widget)
    widget.path.setText('previous.npy')
    monkeypatch.setattr(controls.QFileDialog, 'getOpenFileName',
                        lambda *args: ('chosen.tif' if accepted else '', ''))
    try:
        widget._browse()
        assert widget.path.text() == ('chosen.tif' if accepted else 'previous.npy')
        assert not widget._timer.isActive()
    finally:
        widget._shutdown()


@pytest.mark.parametrize('operation', ['convolve', 'deconvolve'])
def test_compare_and_apply_use_actual_psf_without_modifying_source(screen, qtbot, operation):
    widget = _gaussian(screen, qtbot, operation)
    original = screen._canvas.image.copy()
    chain = screen._enhancement_chain()
    expected = apply_psf(original, widget._kernel, operation=operation,
                         image_sampling_um=(.2, .3)).image
    np.testing.assert_array_equal(dc.prepare(original, chain), expected)
    assert screen._detect_chain() == dc.NO_CHAIN
    screen._on_compare_enhanced()
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    assert screen._compare_dialog.views[1]._item is not None
    assert not screen._btn_apply.isChecked()
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: screen._canvas._enhanced_picture is not None)
    np.testing.assert_array_equal(screen._canvas.detection_source(), expected)
    np.testing.assert_array_equal(screen._canvas.wand_source(), expected)
    record = screen._chain_provenance()['enhancement']['psf']
    assert record['kernel']['kernel_sha256'] == widget._kernel.provenance()['kernel_sha256']
    assert record['intensity_source_for_measurement'] == 'original image'
    json.dumps(record, allow_nan=False)
    screen._btn_apply.click()
    np.testing.assert_array_equal(screen._canvas.detection_source(), original)
    np.testing.assert_array_equal(screen._canvas.image, original)


def test_uncalibrated_psf_fails_visibly_and_never_floods_raw_pixels(screen, qtbot):
    """509 fills common values, so the uncalibrated case is a pixel size cleared by hand."""
    widget = screen._psf_controls
    widget.operation.setCurrentIndex(widget.operation.findData('convolve'))
    widget.image_y.setValue(0)
    assert widget.source_of('image_y')[0] == 'entered'
    qtbot.waitUntil(lambda: 'positive' in widget._error)
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: screen._canvas._enhance_failure is not None)
    assert screen._canvas.wand_source() is None
    assert not screen._canvas.mask.any()
    with pytest.raises(ValueError, match='positive'):
        dc.prepare(screen._canvas.image, screen._detect_chain())
    screen._on_compare_enhanced()
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    assert 'positive' in screen._compare_dialog.caption.text()
    assert screen._compare_dialog.views[1]._item is None


def test_measured_kernel_bytes_stay_captured_until_explicit_reload(screen, qtbot, tmp_path):
    widget = screen._psf_controls
    path = tmp_path / 'psf.npy'
    np.save(path, np.ones((3, 3)))
    for spin in (widget.image_y, widget.image_x, widget.kernel_y, widget.kernel_x):
        spin.setValue(.2)
    widget.source.setCurrentIndex(widget.source.findData('measured'))
    widget.path.setText(str(path))
    widget.operation.setCurrentIndex(widget.operation.findData('convolve'))
    qtbot.waitUntil(lambda: widget._kernel is not None)
    captured = screen._enhancement_chain()
    result = dc.prepare(screen._canvas.image, captured)
    np.save(path, np.eye(3))
    assert screen._enhancement_chain() == captured
    np.testing.assert_array_equal(dc.prepare(screen._canvas.image, captured), result)
    widget.reload.click()
    qtbot.waitUntil(lambda: widget._kernel is not None)
    assert screen._enhancement_chain() != captured
    assert not np.array_equal(dc.prepare(screen._canvas.image, screen._enhancement_chain()), result)
    widget.kernel_y.setValue(.4)
    qtbot.waitUntil(lambda: 'spacing must match' in widget._error)
    assert widget._kernel is None


def test_kernel_preparation_is_background_and_stale_results_are_discarded(screen, qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = controls.gaussian_psf
    calls = []

    def slow(**kwargs):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return real(**kwargs)

    monkeypatch.setattr(controls, 'gaussian_psf', slow)
    widget = screen._psf_controls
    for spin in (widget.image_y, widget.image_x, widget.fwhm_y, widget.fwhm_x):
        spin.setValue(.2)
    widget.operation.setCurrentIndex(widget.operation.findData('convolve'))
    try:
        qtbot.waitUntil(entered.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        assert calls[0] != threading.get_ident()
        widget.operation.setCurrentIndex(0)
        release.set()
        qtbot.waitUntil(lambda: widget._jobs.active_jobs() == 0)
        assert widget._kernel is None
        assert widget._chain_fields() == {}
        assert widget.status.text() == ''
    finally:
        release.set()


def test_psf_errors_and_cancellation_are_not_skipped_as_optional_filters():
    image = np.ones((20, 20), np.float32)
    kernel = gaussian_psf(fwhm_um=2, sampling_um=1)
    chain = dc.Chain(psf_operation='deconvolve', psf=kernel)
    with pytest.raises(ValueError, match='sampling differ'):
        dc.prepare(image, chain._replace(psf_sampling_um=(2, 2)))
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ProcessingCancelled):
        dc.prepare(image, chain, cancel=cancelled)
    with pytest.raises(ValueError, match='calibrated PSF'):
        dc.prepare(image, chain._replace(psf=None))


def test_psf_stage_runs_between_background_and_denoising():
    image = np.arange(900, dtype=np.float32).reshape(30, 30)
    kernel = gaussian_psf(fwhm_um=2, sampling_um=1)
    chain = dc.Chain(background='tophat', background_radius=3, background_scale=1,
                     psf_operation='convolve', psf=kernel,
                     denoise='gaussian', denoise_strength=1.5, gamma=.7)
    background = dc._background(image, chain)
    convolved = apply_psf(background, kernel, operation='convolve', image_sampling_um=1).image
    expected = dc._contrast(dc._denoise(convolved, chain), chain)
    np.testing.assert_array_equal(dc.prepare(image, chain), expected)


@pytest.mark.parametrize('kind', ['compare', 'apply'])
def test_running_psf_stops_on_cancel_or_apply_off(screen, qtbot, monkeypatch, kind):
    _gaussian(screen, qtbot, 'deconvolve')
    entered, finished = threading.Event(), threading.Event()

    def wait_for_cancel(image, kernel, *, cancel, **kwargs):
        entered.set()
        try:
            assert cancel.wait(5), 'processing did not receive cancellation'
            raise ProcessingCancelled('cancelled')
        finally:
            finished.set()

    monkeypatch.setattr(dc, 'apply_psf', wait_for_cancel)
    if kind == 'compare':
        screen._on_compare_enhanced()
        qtbot.waitUntil(entered.is_set)
        screen._compare_dialog.close()
    else:
        screen._btn_apply.click()
        qtbot.waitUntil(entered.is_set)
        screen._btn_apply.click()
    qtbot.waitUntil(finished.is_set)
    assert screen._canvas._enhanced_picture is None


def test_psf_detection_never_recomputes_on_gui_and_records_kernel(screen, qtbot, tmp_path, monkeypatch):
    import tifffile

    folder = tmp_path / 'images'
    folder.mkdir()
    tifffile.imwrite(folder / 'field.tif', screen._canvas.image)
    assert screen._open_folder(str(folder))
    screen._min_area.setValue(5)
    widget = _gaussian(screen, qtbot)
    real = dc.apply_psf
    calls = []

    def recorded(*args, **kwargs):
        calls.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(dc, 'apply_psf', recorded)
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: screen._canvas._enhanced_cache is not None)
    screen._on_detect_otsu()
    assert screen._canvas.mask.any(), screen._status_label.text()
    assert calls and threading.get_ident() not in calls
    receipt = screen._log.edits[-1].detail['enhancement']['psf']
    assert receipt['kernel']['kernel_sha256'] == widget._kernel.provenance()['kernel_sha256']
    screen._on_save()
    saved = list(folder.glob('masks/*.curation.json'))
    assert len(saved) == 1
    ledger = json.loads(saved[0].read_text())
    assert ledger['edits'][-1]['detail']['enhancement']['psf'] == receipt


def test_pending_psf_detection_never_runs_the_filter_on_gui(screen, qtbot, monkeypatch):
    _gaussian(screen, qtbot)
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        started.set()
        assert release.wait(5)
        raise ValueError('controlled stop')

    monkeypatch.setattr(dc, 'apply_psf', blocked)
    screen._btn_apply.click()
    try:
        qtbot.waitUntil(started.is_set)
        with pytest.raises(ValueError, match='updating'):
            screen._canvas.detection_source()
        assert screen._canvas.wand_source() is None
    finally:
        release.set()
        screen._canvas.close_enhancer()
