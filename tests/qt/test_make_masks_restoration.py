"""Restoration uses background Compare/Apply and preserves raw measurements."""
import json
import threading

import numpy as np
import pytest

from spacr._segmentation_backends import _BackendCancelled, _RestorationPlan
from spacr.qt import detect_chain as dc
from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import restoration_controls as controls


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    calls = []

    def load(model, diameter, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        return _RestorationPlan('/isolated', model, diameter, 'cpu', 'a'*64, '3.1.1.3')

    def restore(image, plan, **kwargs):
        calls.append(threading.get_ident())
        return image.astype(np.float32) / 4000 - .1, plan._identity()

    monkeypatch.setattr(controls, '_restoration_plan', load)
    monkeypatch.setattr(dc, '_restore_plane', restore)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    image = np.zeros((48, 48), np.uint16)
    image[20:28, 20:28] = 4000
    made._canvas.set_image_and_mask(image, np.zeros_like(image, dtype=np.uint32))
    made._test_restore_calls = calls
    yield made
    made.close()


def _ready(screen, qtbot):
    widget = screen._restoration_controls
    widget.structure.setCurrentIndex(widget.structure.findData('nuclei'))
    widget.operation.setCurrentIndex(widget.operation.findData('denoise'))
    qtbot.waitUntil(lambda: widget._plan is not None)
    return widget


def test_restoration_default_off_with_api_tooltip(screen):
    widget = screen._restoration_controls
    assert widget.operation.currentData() == 'none'
    assert screen._enhancement_chain() == dc.NO_CHAIN
    assert not widget._jobs.is_busy()
    assert not widget.diameter.isEnabled()
    assert 'qt/detect_chain/index.html' in widget.operation.property('apiTooltipHtml')


@pytest.mark.parametrize('missing', [False, True])
def test_captured_restoration_request_publishes_identity_or_loading_error(
        qtbot, monkeypatch, missing):
    """Validate the load request without installing weights; thread behavior has separate checks."""
    widget = controls._RestorationControls()
    qtbot.addWidget(widget)
    requests, loaded = [], []
    monkeypatch.setattr(widget._jobs, 'submit',
                        lambda work, complete: requests.append((work, complete)))

    def load(model, diameter, *, device, should_cancel):
        loaded.append((model, diameter, device, should_cancel()))
        if missing:
            raise ImportError('isolated restoration weights are unavailable')
        return _RestorationPlan('/isolated', model, diameter, device, 'a'*64, '3.1.1.3')

    monkeypatch.setattr(controls, '_restoration_plan', load)
    try:
        widget.diameter.setValue(40)
        assert not widget._timer.isActive()
        widget.operation.setCurrentIndex(widget.operation.findData('denoise'))
        widget.diameter.setValue(45)
        widget._timer.stop()
        widget._load()
        assert not loaded
        work, complete = requests.pop()
        result = work()
        complete(result)
        assert loaded == [('denoise_cyto3', 45, 'cpu', False)]
        if missing:
            assert widget._plan is None
            assert 'weights are unavailable' in widget._chain_fields()['restoration_error']
            assert 'weights are unavailable' in widget.status.text()
        else:
            assert widget._chain_fields()['restoration_plan'] is result[1]
            assert widget._plan.diameter == 45
            assert widget._plan.weights_sha256 == 'a'*64
        status = widget.status.text()
        widget._shutdown()
        widget._loaded((widget._generation, None, 'late error'))
        assert widget.status.text() == status
    finally:
        widget._shutdown()


@pytest.mark.parametrize('accepted', [False, True])
def test_restoration_install_only_retries_after_acceptance(qtbot, monkeypatch, accepted):
    from spacr.qt.widgets import model_zoo_picker

    widget = controls._RestorationControls()
    qtbot.addWidget(widget)
    choices = []
    monkeypatch.setattr(model_zoo_picker, 'install_backend',
                        lambda parent, name: choices.append((parent, name)) or accepted)
    try:
        widget.operation.setCurrentIndex(widget.operation.findData('denoise'))
        widget._timer.stop()
        generation = widget._generation
        widget.install.click()
        assert choices == [(widget, 'cellpose3')]
        assert widget._generation == generation + int(accepted)
        assert widget._timer.isActive() is accepted
    finally:
        widget._shutdown()


def test_compare_and_apply_restore_without_altering_source(screen, qtbot):
    widget = _ready(screen, qtbot)
    original = screen._canvas.image.copy()
    expected = original.astype(np.float32) / 4000 - .1
    screen._on_compare_enhanced()
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    assert screen._compare_dialog.views[1]._item is not None
    assert not screen._btn_apply.isChecked()
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: screen._canvas._enhanced_cache is not None)
    np.testing.assert_array_equal(screen._canvas.detection_source(), expected)
    np.testing.assert_array_equal(screen._canvas.wand_source(), expected)
    record = screen._chain_provenance()['enhancement']['restoration']
    assert record['weights_sha256'] == widget._plan.weights_sha256
    assert record['diameter_px'] == widget.diameter.value()
    assert record['intensity_source_for_measurement'] == 'original image'
    json.dumps(record, allow_nan=False)
    assert screen._test_restore_calls
    assert threading.get_ident() not in screen._test_restore_calls
    screen._btn_apply.click()
    np.testing.assert_array_equal(screen._canvas.detection_source(), original)
    np.testing.assert_array_equal(screen._canvas.image, original)


def test_missing_model_never_falls_back_to_unenhanced_detection(screen, qtbot, monkeypatch):
    def missing(*a, **kw):
        raise ImportError('Cellpose 3 is not installed')

    monkeypatch.setattr(controls, '_restoration_plan', missing)
    widget = screen._restoration_controls
    widget.operation.setCurrentIndex(widget.operation.findData('denoise'))
    qtbot.waitUntil(lambda: 'not installed' in widget._error)
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: screen._canvas._enhance_failure is not None)
    assert screen._canvas.wand_source() is None
    assert not screen._canvas.mask.any()
    assert 'not installed' in widget.status.text()


def test_diameter_changes_request_identity_without_reloading_weights(screen, qtbot):
    widget = _ready(screen, qtbot)
    first = screen._enhancement_chain()
    widget.diameter.setValue(75)
    second = screen._enhancement_chain()
    assert first != second
    assert first.restoration_plan.weights_sha256 == second.restoration_plan.weights_sha256
    assert first.restoration_plan.diameter == 30
    assert second.restoration_plan.diameter == 75
    assert not widget._jobs.is_busy()


@pytest.mark.parametrize('action', ['compare', 'apply'])
def test_cancellation_stops_restoration_and_discards_output(screen, qtbot, monkeypatch, action):
    _ready(screen, qtbot)
    entered, finished = threading.Event(), threading.Event()

    def blocked(image, plan, *, should_cancel):
        assert threading.current_thread() is not threading.main_thread()
        entered.set()
        try:
            for _ in range(250):
                if should_cancel():
                    raise _BackendCancelled('cancelled')
                threading.Event().wait(.02)
            raise AssertionError('cancellation was not delivered')
        finally:
            finished.set()

    monkeypatch.setattr(dc, '_restore_plane', blocked)
    if action == 'compare':
        screen._on_compare_enhanced()
        qtbot.waitUntil(entered.is_set)
        screen._compare_dialog.close()
    else:
        screen._btn_apply.click()
        qtbot.waitUntil(entered.is_set)
        with pytest.raises(ValueError, match='updating'):
            screen._canvas.detection_source()
        screen._btn_apply.click()
    qtbot.waitUntil(finished.is_set)
    assert screen._canvas._enhanced_picture is None


def test_stale_loaded_model_is_never_applied(screen, qtbot):
    widget = _ready(screen, qtbot)
    stale = (widget._generation, widget._plan, '')
    widget.operation.setCurrentIndex(widget.operation.findData('deblur'))
    widget._loaded(stale)
    assert widget._plan is None
    qtbot.waitUntil(lambda: widget._plan is not None)
    assert widget._plan.model == 'deblur_nuclei'


def test_restoration_order_and_errors_are_explicit(monkeypatch):
    image = np.arange(900, dtype=np.float32).reshape(30, 30)
    plan = _RestorationPlan('/isolated', 'denoise_nuclei', 25, 'cpu', 'a'*64, '3.1.1.3')
    chain = dc.Chain(background='tophat', background_radius=3,
                     background_scale=1, restoration=True, restoration_plan=plan,
                     denoise='gaussian', denoise_strength=1.5)
    seen = []

    def restore(source, captured, **kw):
        seen.append(source.copy())
        return source * .5, {}

    monkeypatch.setattr(dc, '_restore_plane', restore)
    expected_background = dc._background(image, chain)
    expected = dc._denoise(expected_background * .5, chain)
    np.testing.assert_array_equal(dc.prepare(image, chain), expected)
    np.testing.assert_array_equal(seen[0], expected_background)
    with pytest.raises(ValueError, match='not ready'):
        dc.prepare(image, chain._replace(restoration_plan=None, restoration_error='not ready'))
